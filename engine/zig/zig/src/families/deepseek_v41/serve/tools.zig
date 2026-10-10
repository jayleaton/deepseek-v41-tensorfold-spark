//! DeepSeek-V4.1 replies: reasoning, content and DSML tool calls, whole (``parse``) or while they stream (``Stream``),
//! equal to the Python engine's ``dsml.py``:
//!
//!     reasoning</think>content
//!
//!     <｜DSML｜ calls>
//!     <｜DSML｜ invoke name="get_weather">
//!     <｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>
//!     </｜DSML｜ invoke>
//!     </｜DSML｜ calls>
//!
//! Lenient where models slip: a calls block inside an unclosed think block ends the reasoning there; ``calls`` /
//! ``tool_calls`` / ``function_calls`` block names, missing blank lines, an unclosed last invoke or block are read;
//! ``string="true"`` keeps the text unless the tool's schema does not allow a string; ``string="false"`` is JSON,
//! closed when it stops a bracket short, else read by the schema, else kept as text; an invoke of a tool the request
//! did not offer is still a call, under the name the model gave (``dsml.py`` keeps it in the content: the client then
//! sees a finished reply and an agent stops, where a call it cannot run is answered and the model tries another way);
//! an invoke without a name stays in the content. Arguments are compact JSON; streamed, each completed parameter is one fragment
//! (``{`` + key + value, ``,`` + key + value ..., ``}``), so the fragments concatenate to the final arguments.
//!
//! ``Stream`` reads the reply incrementally: every search resumes where the last one stopped and a finished call is
//! never read again, so a token costs time in its own text only.
const std = @import("std");
const json = @import("json");
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const THINK_END = "</think>";
const MARK = "<｜DSML｜"; // every DSML tag starts here
const MARK_CLOSE = "</｜DSML｜";
const BLOCK_NAMES = [_][]const u8{ "calls", "tool_calls", "function_calls" };
/// Text that may still become a calls block's opening tag (with its blank line) is held back.
const OPEN_TAGS = blk: {
    var tags: [8][]const u8 = undefined;
    var i: usize = 0;
    for ([_][]const u8{ "", "\n\n" }) |sep| for ([_][]const u8{ " calls", "calls", "tool_calls", "function_calls" }) |name| {
        tags[i] = sep ++ MARK ++ name ++ ">";
        i += 1;
    };
    break :blk tags;
};
/// The longest a tag can be (a Python whitespace character is up to 3 bytes): searches resume this far back.
const tag_window = 64;

// -- text helpers --------------------------------------------------------------------------------------------------

/// The byte length of the Python whitespace character (``str.isspace``) at ``i``, else 0.
fn pySpace(s: []const u8, i: usize) usize {
    if (i >= s.len) return 0;
    const b = s[i];
    if (b < 0x80) return if ((b >= 9 and b <= 13) or (b >= 0x1c and b <= 0x20)) 1 else 0;
    const n = std.unicode.utf8ByteSequenceLength(b) catch return 0;
    if (i + n > s.len) return 0;
    const cp = std.unicode.utf8Decode(s[i .. i + n]) catch return 0;
    return switch (cp) {
        0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => n,
        else => 0,
    };
}

/// Python's ``str.strip()``.
pub fn pyStrip(s: []const u8) []const u8 {
    var a: usize = 0;
    while (a < s.len) {
        const n = pySpace(s, a);
        if (n == 0) break;
        a += n;
    }
    var b = s.len;
    while (b > a) {
        var c = b - 1;
        while (c > a and s[c] & 0xC0 == 0x80) c -= 1;
        if (pySpace(s, c) != b - c) break;
        b = c;
    }
    return s[a..b];
}

/// ``_partial``: the longest proper prefix of ``tag`` that ``text`` ends with.
fn partial(text: []const u8, tag: []const u8) usize {
    var k = @min(tag.len - 1, text.len);
    while (k > 0) : (k -= 1) if (std.mem.endsWith(u8, text, tag[0..k])) return k;
    return 0;
}

/// ``_hold``: trailing text that may still become a calls block's opening tag.
fn hold(text: []const u8) usize {
    var most: usize = 0;
    for (OPEN_TAGS) |t| most = @max(most, partial(text, t));
    return most;
}

pub const Match = struct { start: usize, end: usize };

/// ``<｜DSML｜`` (or ``</｜DSML｜``) at ``i``, then ``\s?``: where the tag's name starts.
fn markAt(s: []const u8, i: usize, close: bool) ?usize {
    const m = if (close) MARK_CLOSE else MARK;
    if (!std.mem.startsWith(u8, s[i..], m)) return null;
    return i + m.len;
}

fn word(s: []const u8, at: usize, w: []const u8) ?usize {
    return if (std.mem.startsWith(u8, s[at..], w)) at + w.len else null;
}

/// ``\s?`` + one of ``names``: the end of the name, else null (the optional space given back when the name fails).
fn spacedWord(s: []const u8, at: usize, names: []const []const u8) ?usize {
    const sp = pySpace(s, at);
    for ([_]usize{ at + sp, at }) |from| {
        for (names) |n| if (word(s, from, n)) |e| return e;
        if (sp == 0) break;
    }
    return null;
}

fn spaces(s: []const u8, at: usize) usize {
    var k = at;
    while (true) {
        const n = pySpace(s, k);
        if (n == 0) return k;
        k += n;
    }
}

/// The first ``<｜DSML｜\s?(calls|tool_calls|function_calls)>`` (or its ``</`` form) in ``s[from..to]``.
fn blockTag(s: []const u8, from: usize, to: usize, close: bool) ?Match {
    const t = s[0..to];
    var i = from;
    while (std.mem.indexOfPos(u8, t, i, if (close) MARK_CLOSE else MARK)) |p| : (i = p + 1) {
        const n = markAt(t, p, close).?;
        const e = spacedWord(t, n, &BLOCK_NAMES) orelse continue;
        if (e < t.len and t[e] == '>') return .{ .start = p, .end = e + 1 };
    }
    return null;
}

/// A header read at a position: matched, refused, or cut off by the end of the text (it may match later).
const Got = enum { ok, no, more };
const Step = struct { got: Got, at: usize = 0 };

fn wordStep(t: []const u8, at: usize, w: []const u8) Step {
    const have = t[at..];
    if (std.mem.startsWith(u8, have, w)) return .{ .got = .ok, .at = at + w.len };
    return .{ .got = if (have.len < w.len and std.mem.startsWith(u8, w, have)) .more else .no };
}

/// ``\s?`` + one of ``names``, three-way.
fn spacedStep(t: []const u8, at: usize, names: []const []const u8) Step {
    if (at >= t.len) return .{ .got = .more };
    const sp = pySpace(t, at);
    var more = false;
    for (names) |n| {
        const st = wordStep(t, at + sp, n);
        if (st.got == .ok) return st;
        more = more or st.got == .more;
    }
    if (sp > 0) for (names) |n| if (wordStep(t, at, n).got == .ok) return wordStep(t, at, n);
    return .{ .got = if (more) .more else .no };
}

/// ``\s+`` (``plus``) or ``\s*`` then ``c``.
fn spacedChar(t: []const u8, at: usize, plus: bool, c: u8) Step {
    const sp = spaces(t, at);
    if (sp >= t.len) return .{ .got = .more };
    if (plus and sp == at) return .{ .got = .no };
    return if (t[sp] == c) .{ .got = .ok, .at = sp + 1 } else .{ .got = .no };
}

/// ``\s+`` then ``w``.
fn spacedThen(t: []const u8, at: usize, w: []const u8) Step {
    const sp = spaces(t, at);
    if (sp >= t.len) return .{ .got = .more };
    if (sp == at) return .{ .got = .no };
    return wordStep(t, sp, w);
}

/// ``<｜DSML｜\s?invoke\s+name="([^"]*)"\s*>``: the match and the name.
const Invoke = struct { m: Match, name: []const u8 };

fn Probe(comptime T: type) type {
    return union(Got) { ok: T, no, more };
}

fn invokeAt(t: []const u8, p: usize) Probe(Invoke) {
    const n = markAt(t, p, false) orelse return .no;
    const e = spacedStep(t, n, &.{"invoke"});
    if (e.got != .ok) return if (e.got == .more) .more else .no;
    const q = spacedThen(t, e.at, "name=\"");
    if (q.got != .ok) return if (q.got == .more) .more else .no;
    const close = std.mem.indexOfScalarPos(u8, t, q.at, '"') orelse return .more;
    const gt = spacedChar(t, close + 1, false, '>');
    if (gt.got != .ok) return if (gt.got == .more) .more else .no;
    return .{ .ok = .{ .m = .{ .start = p, .end = gt.at }, .name = t[q.at..close] } };
}

/// The first whole header at or after ``from``, and where a later search must start again: the first header cut off
/// by the end of ``t``, else just before a ``<｜DSML｜`` the end may still be completing.
fn Search(comptime T: type) type {
    return struct { hit: ?T, again: usize };
}

fn searchHeads(comptime T: type, t: []const u8, from: usize, comptime at: fn ([]const u8, usize) Probe(T)) Search(T) {
    var i = from;
    var first_more: ?usize = null;
    while (std.mem.indexOfPos(u8, t, i, MARK)) |p| : (i = p + 1) switch (at(t, p)) {
        .ok => |h| return .{ .hit = h, .again = p },
        .more => if (first_more == null) {
            first_more = p;
        },
        .no => {},
    };
    return .{ .hit = null, .again = first_more orelse @max(from, t.len -| (MARK.len - 1)) };
}

fn invokeOpen(s: []const u8, from: usize, to: usize) ?Invoke {
    if (from > to) return null;
    return searchHeads(Invoke, s[0..to], from, invokeAt).hit;
}

fn invokeEnd(s: []const u8, from: usize, to: usize) ?Match {
    if (from > to) return null;
    const t = s[0..to];
    var i = from;
    while (std.mem.indexOfPos(u8, t, i, MARK_CLOSE)) |p| : (i = p + 1) {
        const e = spacedWord(t, markAt(t, p, true).?, &.{"invoke>"}) orelse continue;
        return .{ .start = p, .end = e };
    }
    return null;
}

/// A parameter's opening tag: ``<｜DSML｜\s?parameter\s+name="([^"]*)"\s+string="(true|false)"\s*>``.
const ParamHead = struct { start: usize, end: usize, name: []const u8, string: bool };

fn paramHeadAt(t: []const u8, p: usize) Probe(ParamHead) {
    const n = markAt(t, p, false) orelse return .no;
    const e = spacedStep(t, n, &.{"parameter"});
    if (e.got != .ok) return if (e.got == .more) .more else .no;
    const q = spacedThen(t, e.at, "name=\"");
    if (q.got != .ok) return if (q.got == .more) .more else .no;
    const close = std.mem.indexOfScalarPos(u8, t, q.at, '"') orelse return .more;
    const v = spacedThen(t, close + 1, "string=\"");
    if (v.got != .ok) return if (v.got == .more) .more else .no;
    const yes = wordStep(t, v.at, "true\"");
    const no = wordStep(t, v.at, "false\"");
    const flag = if (yes.got == .ok) yes else if (no.got == .ok) no else return if (yes.got == .more or no.got == .more) .more else .no;
    const gt = spacedChar(t, flag.at, false, '>');
    if (gt.got != .ok) return if (gt.got == .more) .more else .no;
    return .{ .ok = .{ .start = p, .end = gt.at, .name = t[q.at..close], .string = yes.got == .ok } };
}

/// The first ``</｜DSML｜\s?parameter>`` in ``t[from..]``.
fn paramClose(t: []const u8, from: usize) ?Match {
    if (from > t.len) return null;
    var i = from;
    while (std.mem.indexOfPos(u8, t, i, MARK_CLOSE)) |p| : (i = p + 1) {
        const e = spacedWord(t, markAt(t, p, true).?, &.{"parameter>"}) orelse continue;
        return .{ .start = p, .end = e };
    }
    return null;
}

const Param = struct { name: []const u8, string: bool, raw: []const u8, end: usize };

/// ``_PARAM.search(block, from, to)``: the leftmost whole parameter (the lazy value up to the first closing tag).
fn paramIn(s: []const u8, from: usize, to: usize) ?Param {
    if (from > to) return null;
    const t = s[0..to];
    var i = from;
    while (std.mem.indexOfPos(u8, t, i, MARK)) |p| : (i = p + 1) {
        const h = switch (paramHeadAt(t, p)) {
            .ok => |h| h,
            else => continue,
        };
        const c = paramClose(t, h.end) orelse continue;
        return .{ .name = h.name, .string = h.string, .raw = t[h.end..c.start], .end = c.end };
    }
    return null;
}

// -- values ----------------------------------------------------------------------------------------------------------

const TypeSet = packed struct(u8) { string: bool = false, integer: bool = false, number: bool = false, boolean: bool = false, array: bool = false, object: bool = false, null: bool = false, _: u1 = 0 };

fn typeOfName(name: []const u8) TypeSet {
    var t: TypeSet = .{};
    const eq = std.mem.eql;
    if (eq(u8, name, "string")) t.string = true;
    if (eq(u8, name, "integer")) t.integer = true;
    if (eq(u8, name, "number")) t.number = true;
    if (eq(u8, name, "boolean")) t.boolean = true;
    if (eq(u8, name, "array")) t.array = true;
    if (eq(u8, name, "object")) t.object = true;
    if (eq(u8, name, "null")) t.null = true;
    return t;
}

fn typeOfValue(v: Value) TypeSet {
    return switch (v) {
        .bool => .{ .boolean = true },
        .int => .{ .integer = true },
        .float => .{ .number = true },
        .string => .{ .string = true },
        .array => .{ .array = true },
        .object => .{ .object = true },
        .null => .{ .null = true },
    };
}

fn merge(x: TypeSet, y: TypeSet) TypeSet {
    return @bitCast(@as(u8, @bitCast(x)) | @as(u8, @bitCast(y)));
}

fn any(x: TypeSet) bool {
    return @as(u8, @bitCast(x)) != 0;
}

/// ``schema_types``: the JSON types a parameter schema allows, or null when it does not say. A type name the set
/// does not know (a typo, ``"any"``) still counts as saying (an empty set, as Python's set of unknown names).
pub fn schemaTypes(prop: ?Value) ?TypeSet {
    const p = prop orelse return null;
    if (p != .object) return null;
    if (p.get("type")) |kind| {
        if (kind == .string) return typeOfName(kind.string);
        if (kind == .array and kind.array.len > 0) {
            var all = true;
            for (kind.array) |k| all = all and k == .string;
            if (all) {
                var t: TypeSet = .{};
                for (kind.array) |k| t = merge(t, typeOfName(k.string));
                return t;
            }
        }
    }
    for ([_][]const u8{ "anyOf", "oneOf" }) |key| if (p.get(key)) |members| if (members == .array and members.array.len > 0) {
        var t: TypeSet = .{};
        for (members.array) |m| t = merge(t, schemaTypes(m) orelse return null);
        return t;
    };
    if (p.get("enum")) |e| if (e == .array and e.array.len > 0) {
        var t: TypeSet = .{};
        for (e.array) |v| t = merge(t, typeOfValue(v));
        return t;
    };
    if (p.get("const")) |c| return typeOfValue(c);
    return null;
}

fn fits(v: Value, t: TypeSet) bool {
    const k = typeOfValue(v);
    return any(.{ .string = k.string and t.string, .integer = k.integer and t.integer, .number = k.number and t.number, .boolean = k.boolean and t.boolean, .array = k.array and t.array, .object = k.object and t.object, .null = k.null and t.null }) or (k.integer and t.number);
}

/// ``closed_json``: ``text`` with the arrays and objects it left open closed, or null if it does not only stop short.
pub fn closedJson(a: Allocator, text: []const u8) Allocator.Error!?[]u8 {
    var closers: std.ArrayList(u8) = .empty;
    var in_string = false;
    var escaped = false;
    for (text) |c| {
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else {
                escaped = c == '\\';
                in_string = c != '"';
            }
        } else if (c == '"') {
            in_string = true;
        } else if (c == '[' or c == '{') {
            try closers.append(a, if (c == '[') ']' else '}');
        } else if (c == ']' or c == '}') {
            if (closers.items.len == 0 or closers.pop().? != c) return null;
        }
    }
    if (closers.items.len == 0 or in_string) return null;
    std.mem.reverse(u8, closers.items);
    return try std.mem.concat(a, u8, &.{ pyRstrip(text), closers.items });
}

fn pyRstrip(s: []const u8) []const u8 {
    const stripped = pyStrip(s);
    return s[0 .. @intFromPtr(stripped.ptr) - @intFromPtr(s.ptr) + stripped.len];
}

fn loads(a: Allocator, text: []const u8) Allocator.Error!?Value {
    return switch (try json.parseText(a, text)) {
        .ok => |v| v,
        .err => null,
    };
}

/// An integral float as Python's ``int(value)``; null past i128 (kept a float).
fn integral(a: Allocator, f: f64) Allocator.Error!?Value {
    if (@abs(f) >= 1.7e38) return null;
    return .{ .int = try std.fmt.allocPrint(a, "{d}", .{@as(i128, @intFromFloat(f))}) };
}

/// ``typed_value``: a value read by its parameter schema (see the module comment).
pub fn typedValue(a: Allocator, raw: []const u8, prop: ?Value) Allocator.Error!Value {
    const types = schemaTypes(prop) orelse {
        if (try loads(a, raw)) |v| return v;
        return .{ .string = raw };
    };
    if (types.string) return if (types.null and std.mem.eql(u8, pyStrip(raw), "null")) .null else .{ .string = raw };
    const text = pyStrip(raw);
    var candidates: std.ArrayList(Value) = .empty;
    if (try loads(a, text)) |v| try candidates.append(a, v);
    if (types.array or types.object) if (try closedJson(a, text)) |fixed| if (try loads(a, fixed)) |v| try candidates.append(a, v);
    if (candidates.items.len > 0 and candidates.items[0] == .string) if (try loads(a, pyStrip(candidates.items[0].string))) |v| try candidates.append(a, v);
    if (std.mem.eql(u8, text, "True")) try candidates.append(a, .{ .bool = true });
    if (std.mem.eql(u8, text, "False")) try candidates.append(a, .{ .bool = false });
    if (std.mem.eql(u8, text, "None")) try candidates.append(a, .null);
    for (candidates.items) |v| {
        var x = v;
        if (x == .float and types.integer and !types.number and std.math.isFinite(x.float) and @floor(x.float) == x.float) {
            if (try integral(a, x.float)) |n| x = n;
        }
        if (fits(x, types)) return x;
    }
    return .{ .string = raw };
}

/// ``value``: a parameter's value from its text, ``string`` flag and schema.
pub fn paramValue(a: Allocator, raw: []const u8, string: bool, prop: ?Value) Allocator.Error!Value {
    const types = schemaTypes(prop);
    if (string) {
        if (types == null or types.?.string) return .{ .string = raw };
        return typedValue(a, raw, prop);
    }
    if (try loads(a, raw)) |v| return v;
    if (try closedJson(a, pyStrip(raw))) |fixed| if (try loads(a, fixed)) |v| return v;
    return if (types != null) typedValue(a, raw, prop) else .{ .string = raw };
}

/// ``fragment``: ``{`` (the first) or ``,``, the key, ``:``, the compact value.
fn fragment(a: Allocator, out: *std.ArrayList(u8), first: bool, key: []const u8, v: Value) Allocator.Error!void {
    try out.append(a, if (first) '{' else ',');
    var w: std.Io.Writer.Allocating = .fromArrayList(a, out);
    json.writeString(&w.writer, key, .{ .ascii = false }) catch return error.OutOfMemory;
    w.writer.writeByte(':') catch return error.OutOfMemory;
    json.write(&w.writer, v, .{ .ascii = false, .compact = true }) catch return error.OutOfMemory;
    out.* = w.toArrayList();
}

// -- tools and calls -------------------------------------------------------------------------------------------------

pub const Arg = struct { key: []const u8, value: Value };

pub const Call = struct {
    id: []const u8,
    name: []const u8,
    args: std.ArrayList(Arg) = .empty,
    closed: bool = false,

    /// The arguments as compact JSON (``{}`` without any).
    pub fn arguments(c: *const Call, a: Allocator) Allocator.Error![]u8 {
        if (c.args.items.len == 0) return a.dupe(u8, "{}");
        var out: std.ArrayList(u8) = .empty;
        for (c.args.items, 0..) |arg, i| try fragment(a, &out, i == 0, arg.key, arg.value);
        try out.append(a, '}');
        return out.items;
    }
};

/// A new call's id (``call_`` + 24 hex in the server); ``index`` counts the reply's calls.
pub const IdSource = struct {
    ctx: ?*anyopaque = null,
    make: *const fn (ctx: ?*anyopaque, a: Allocator, index: usize) Allocator.Error![]const u8 = counted,

    fn counted(_: ?*anyopaque, a: Allocator, index: usize) Allocator.Error![]const u8 {
        return std.fmt.allocPrint(a, "call_{d}", .{index});
    }
};

/// The offered tools by name: their parameter properties (``_schemas``), for typing values and matching names.
pub const Tools = struct {
    names: []const []const u8 = &.{},
    props: []const ?Value = &.{},
    /// Calls to tools not offered go out as calls; false keeps them in the content as ``dsml.py`` does.
    unknown_calls: bool = true,

    pub fn init(a: Allocator, tools: []const Value) Allocator.Error!Tools {
        var names: std.ArrayList([]const u8) = .empty;
        var props: std.ArrayList(?Value) = .empty;
        for (tools) |t| {
            if (t != .object) continue;
            const inner = t.get("function");
            const f = if (inner != null and inner.? == .object) inner.? else t;
            const n = f.get("name") orelse continue;
            if (!n.truthy()) continue;
            const name = switch (n) {
                .string => |s| s,
                .int => |s| s,
                else => try json.stringify(a, n, .{ .ascii = false }),
            };
            const params = f.get("parameters") orelse Value.null;
            const properties: ?Value = if (params == .object) (if (params.get("properties")) |p| (if (p.truthy()) p else null) else null) else null;
            // later tools of the same name replace earlier ones, as a dict does
            for (names.items, 0..) |seen, i| {
                if (!std.mem.eql(u8, seen, name)) continue;
                props.items[i] = properties;
                break;
            } else {
                try names.append(a, name);
                try props.append(a, properties);
            }
        }
        return .{ .names = names.items, .props = props.items };
    }

    pub fn offered(t: Tools) bool {
        return t.names.len > 0;
    }

    /// ``_known``: the offered name a call means: exact, else case-insensitive, else its last ``::`` part.
    pub fn known(t: Tools, name: []const u8) ?usize {
        for (t.names, 0..) |n, i| if (std.mem.eql(u8, n, name)) return i;
        var found: ?usize = null;
        for (t.names, 0..) |n, i| if (std.ascii.eqlIgnoreCase(n, name)) {
            found = i; // the last of equal lower-case names, as the dict built from them keeps
        };
        if (found) |f| return f;
        const tail = if (std.mem.lastIndexOf(u8, name, "::")) |p| name[p + 2 ..] else name;
        for (t.names, 0..) |n, i| if (std.ascii.eqlIgnoreCase(n, tail)) {
            found = i;
        };
        return found;
    }

    /// The name a call goes out under: the offered name it means, else its own (null for an empty one).
    pub fn resolve(t: Tools, name: []const u8) ?[]const u8 {
        if (t.known(name)) |i| return t.names[i];
        return if (t.unknown_calls and name.len > 0) name else null;
    }

    fn prop(t: Tools, call_name: []const u8, key: []const u8) ?Value {
        const i = t.known(call_name) orelse return null;
        const p = t.props[i] orelse return null;
        return if (p == .object) p.get(key) else null;
    }
};

/// One invoke's parameters read up to ``stop`` (``_calls``' inner loop), the first of each name kept.
fn readParams(a: Allocator, s: []const u8, from: usize, stop: usize, tools: Tools, call: *Call, seen: *std.StringHashMapUnmanaged(void), pos: *usize) Allocator.Error!void {
    while (paramIn(s, pos.*, stop)) |p| {
        pos.* = p.end;
        if (seen.contains(p.name)) continue;
        try seen.put(a, p.name, {});
        try call.args.append(a, .{ .key = p.name, .value = try paramValue(a, p.raw, p.string, tools.prop(call.name, p.name)) });
    }
    _ = from;
}

/// ``_calls``: the invokes of a calls block's body ``s[from..to]``.
fn readCalls(a: Allocator, s: []const u8, from: usize, to: usize, tools: Tools, ids: IdSource, first_index: usize, out: *std.ArrayList(Call)) Allocator.Error!void {
    var pos = from;
    while (invokeOpen(s, pos, to)) |m| {
        const end = invokeEnd(s, m.m.end, to);
        const nxt = invokeOpen(s, m.m.end, to);
        const stop = if (end != null and (nxt == null or end.?.start < nxt.?.m.start)) end.?.start else if (nxt) |n| n.m.start else to;
        var call: Call = .{ .id = try ids.make(ids.ctx, a, first_index + out.items.len), .name = m.name };
        var seen: std.StringHashMapUnmanaged(void) = .empty;
        var p = m.m.end;
        try readParams(a, s, m.m.end, stop, tools, &call, &seen, &p);
        call.closed = end != null and stop == end.?.start;
        try out.append(a, call);
        pos = if (call.closed) end.?.end else stop;
        if (!call.closed and nxt == null) break;
    }
}

pub const Reply = struct {
    reasoning: []const u8,
    content: []const u8,
    calls: []Call,
};

const Split = struct { reasoning: []const u8, rest_at: usize };

/// ``_split``: (reasoning, where the rest starts).
fn split(text: []const u8, thinking: bool, end: ?usize, blk: ?Match) Split {
    if (!thinking) return .{ .reasoning = "", .rest_at = 0 };
    if (end) |e| if (blk == null or e < blk.?.start) return .{ .reasoning = text[0..e], .rest_at = e + THINK_END.len };
    if (blk) |b| {
        const head = text[0..b.start];
        const r = if (std.mem.endsWith(u8, head, "\n\n")) head[0 .. head.len - 2] else std.mem.trimEnd(u8, head, "\n");
        return .{ .reasoning = r, .rest_at = r.len };
    }
    return .{ .reasoning = text, .rest_at = text.len };
}

/// ``parse``: a finished reply's parts (end-of-sentence tokens already removed).
pub fn parse(a: Allocator, text: []const u8, thinking: bool, tools: Tools, ids: IdSource) Allocator.Error!Reply {
    const sp = split(text, thinking, std.mem.indexOf(u8, text, THINK_END), blockTag(text, 0, text.len, false));
    const rest = text[sp.rest_at..];
    const blk = if (tools.offered()) blockTag(rest, 0, rest.len, false) else null;
    const b = blk orelse return .{ .reasoning = sp.reasoning, .content = rest, .calls = &.{} };
    var content = rest[0..b.start];
    if (std.mem.endsWith(u8, content, "\n\n")) content = content[0 .. content.len - 2];
    const close = blockTag(rest, b.end, rest.len, true);
    var calls: std.ArrayList(Call) = .empty;
    try readCalls(a, rest, b.end, if (close) |c| c.start else rest.len, tools, ids, 0, &calls);
    var usable: std.ArrayList(Call) = .empty;
    for (calls.items) |c| if (tools.resolve(c.name)) |n| {
        var k = c;
        k.name = n;
        try usable.append(a, k);
    };
    if (usable.items.len == 0) return .{ .reasoning = sp.reasoning, .content = rest, .calls = &.{} };
    // ids in order of the usable calls, as Python's ``Call()`` made them while reading
    return .{ .reasoning = sp.reasoning, .content = content, .calls = usable.items };
}

// -- streaming -------------------------------------------------------------------------------------------------------

pub const Delta = union(enum) {
    reasoning: []const u8,
    content: []const u8,
    /// A call opens: its index among the reply's calls, id and name (arguments "").
    call: struct { index: usize, id: []const u8, name: []const u8 },
    /// More of a call's arguments.
    arguments: struct { index: usize, text: []const u8 },
};

/// A search for a fixed-length tag at or after ``from``, resumed where it stopped as the text grows.
const Finder = struct {
    from: usize = 0,
    scanned: usize = 0,
    found: ?Match = null,

    fn reset(f: *Finder, from: usize) void {
        f.* = .{ .from = from, .scanned = from };
    }

    fn block(f: *Finder, text: []const u8, close: bool) ?Match {
        if (f.found) |m| return m;
        const start = @max(f.from, f.scanned -| tag_window);
        f.found = blockTag(text, start, text.len, close);
        f.scanned = text.len;
        return f.found;
    }
};

/// The invoke being read: its header, the searches for its end and the next invoke, and its parameters so far.
const Open = struct {
    m: Invoke,
    end: Finder,
    nxt: Finder,
    end_m: ?Match = null,
    nxt_m: ?Invoke = null,
    param_pos: usize,
    call: Call,
    seen: std.StringHashMapUnmanaged(void) = .empty,
    pending: ?ParamHead = null, // a parameter whose closing tag has not come yet
    close_scan: usize = 0,
};

pub const Stream = struct {
    a: Allocator,
    thinking: bool,
    tools: Tools,
    ids: IdSource,
    sent_reasoning: usize = 0,
    sent_content: usize = 0,
    in_calls: bool = false,
    /// The reply's usable calls, as sent.
    calls: std.ArrayList(Call) = .empty,
    sent_args: std.ArrayList(usize) = .empty,
    sent_close: std.ArrayList(bool) = .empty,
    // incremental state over the growing text
    think_end: ?usize = null,
    think_scan: usize = 0,
    text_blk: Finder = .{},
    rest_at: ?usize = null,
    rest_blk: Finder = .{},
    close_blk: Finder = .{},
    body_at: usize = 0,
    done: std.ArrayList(Call) = .empty, // every invoke read to its fixed end, usable or not
    settled: usize = 0, // done calls before this one are sent whole (or unusable)
    settled_known: usize = 0, // the usable calls among them
    resume_at: usize = 0,
    open: ?Open = null,
    scratch: std.ArrayList(u8) = .empty,

    pub fn init(a: Allocator, thinking: bool, tools: Tools, ids: IdSource) Stream {
        return .{ .a = a, .thinking = thinking, .tools = tools, .ids = ids };
    }

    fn findThinkEnd(s: *Stream, text: []const u8) ?usize {
        if (s.think_end) |e| return e;
        const from = s.think_scan -| (THINK_END.len - 1);
        s.think_end = std.mem.indexOfPos(u8, text, from, THINK_END);
        s.think_scan = text.len;
        return s.think_end;
    }

    /// The deltas ``text`` (the whole reply decoded so far, a growth of the last one) adds; ``finished`` releases
    /// what was held back and closes an open call.
    pub fn feed(s: *Stream, text: []const u8, finished: bool, out: *std.ArrayList(Delta)) Allocator.Error!void {
        const a = s.a;
        const offered = s.tools.offered();
        var end: ?usize = null;
        var blk_all: ?Match = null;
        if (s.thinking) {
            end = s.findThinkEnd(text);
            blk_all = s.text_blk.block(text, false);
            if (end == null and (blk_all == null or !offered)) {
                var h = partial(text, THINK_END);
                if (offered) h = @max(h, hold(text));
                const upto = if (finished) text.len else text.len - h;
                if (upto > s.sent_reasoning) {
                    try out.append(a, .{ .reasoning = text[s.sent_reasoning..upto] });
                    s.sent_reasoning = upto;
                }
                return;
            }
        }
        const sp = split(text, s.thinking, end, blk_all);
        if (sp.reasoning.len > s.sent_reasoning) {
            try out.append(a, .{ .reasoning = sp.reasoning[s.sent_reasoning..] });
            s.sent_reasoning = sp.reasoning.len;
        }
        if (s.rest_at == null or s.rest_at.? != sp.rest_at) {
            s.rest_at = sp.rest_at;
            s.rest_blk.reset(sp.rest_at);
        }
        const rest = text[sp.rest_at..];
        const blk = if (offered) s.rest_blk.block(text, false) else null;
        var limit: usize = undefined;
        if (blk) |b| {
            limit = b.start - sp.rest_at;
            if (std.mem.endsWith(u8, rest[0..limit], "\n\n")) limit -= 2;
        } else if (finished) {
            limit = rest.len;
        } else limit = rest.len - (if (offered) hold(rest) else 0);
        if (limit > s.sent_content and !s.in_calls) {
            try out.append(a, .{ .content = rest[s.sent_content..limit] });
            s.sent_content = limit;
        }
        const b = blk orelse return;
        if (!s.in_calls) {
            s.in_calls = true;
            s.body_at = b.end;
            s.resume_at = b.end;
            s.close_blk.reset(b.end);
        }
        const close = s.close_blk.block(text, true);
        const to = if (close) |c| c.start else text.len;
        try s.readCalls(text, to);
        // the calls in order: the finished ones, then the one still open; finished calls already sent whole are
        // skipped, so a token costs the calls still open, not every call of the reply
        while (s.settled < s.done.items.len) : (s.settled += 1) {
            const c = &s.done.items[s.settled];
            const name = s.tools.resolve(c.name) orelse continue;
            if (s.settled_known == s.calls.items.len or !s.sent_close.items[s.settled_known]) break;
            c.name = name;
            s.settled_known += 1;
        }
        var k: usize = s.settled_known;
        const total = s.done.items.len + @intFromBool(s.open != null);
        for (s.settled..total) |j| {
            const c: *Call = if (j < s.done.items.len) &s.done.items[j] else &s.open.?.call;
            c.name = s.tools.resolve(c.name) orelse continue;
            if (k == s.calls.items.len) {
                try s.calls.append(a, .{ .id = c.id, .name = c.name });
                try s.sent_args.append(a, 0);
                try s.sent_close.append(a, false);
                try out.append(a, .{ .call = .{ .index = k, .id = c.id, .name = c.name } });
            }
            const mine = &s.calls.items[k];
            mine.args = c.args;
            s.scratch = .empty;
            for (c.args.items[s.sent_args.items[k]..], s.sent_args.items[k]..) |arg, n| try fragment(a, &s.scratch, n == 0, arg.key, arg.value);
            s.sent_args.items[k] = c.args.items.len;
            if ((c.closed or finished) and !s.sent_close.items[k]) {
                try s.scratch.appendSlice(a, if (c.args.items.len > 0) "}" else "{}");
                s.sent_close.items[k] = true;
                mine.closed = true;
            }
            if (s.scratch.items.len > 0) try out.append(a, .{ .arguments = .{ .index = k, .text = s.scratch.items } });
            k += 1;
        }
    }

    /// ``_calls`` on the body ``text[body_at..to]``, resumed: invokes whose end is fixed are read once.
    fn readCalls(s: *Stream, text: []const u8, to: usize) Allocator.Error!void {
        const a = s.a;
        while (true) {
            if (s.open == null) {
                const found = searchHeads(Invoke, text[0..to], @min(s.resume_at, to), invokeAt);
                const m = found.hit orelse {
                    s.resume_at = @max(s.resume_at, found.again); // a header still arriving is read again
                    return;
                };
                // names are copied: ``text`` is the caller's growing buffer, gone by a later feed
                var o: Open = .{ .m = m, .end = .{}, .nxt = .{}, .param_pos = m.m.end, .call = .{ .id = try s.ids.make(s.ids.ctx, a, s.done.items.len), .name = try a.dupe(u8, m.name) } };
                o.end.reset(m.m.end);
                o.nxt.reset(m.m.end);
                s.open = o;
            }
            const o = &s.open.?;
            if (o.end_m == null) {
                const start = @max(o.end.from, o.end.scanned -| tag_window);
                o.end_m = invokeEnd(text, start, to);
                o.end.scanned = to;
            }
            if (o.nxt_m == null) {
                const found = searchHeads(Invoke, text[0..to], @min(@max(o.nxt.from, o.nxt.scanned), to), invokeAt);
                o.nxt_m = found.hit;
                o.nxt.scanned = found.again;
            }
            const end = o.end_m;
            const nxt = o.nxt_m;
            const stop = if (end != null and (nxt == null or end.?.start < nxt.?.m.start)) end.?.start else if (nxt) |n| n.m.start else to;
            try s.readParams(text, stop);
            o.call.closed = end != null and stop == end.?.start;
            if (o.call.closed or nxt != null) { // its end is fixed: read for good
                s.resume_at = if (o.call.closed) end.?.end else stop;
                try s.done.append(a, o.call);
                s.open = null;
                continue;
            }
            return;
        }
    }

    /// The open invoke's parameters up to ``stop``; a parameter whose closing tag is still missing is resumed.
    fn readParams(s: *Stream, text: []const u8, stop: usize) Allocator.Error!void {
        const a = s.a;
        const o = &s.open.?;
        const t = text[0..stop];
        while (true) {
            if (o.pending == null) {
                const found = searchHeads(ParamHead, t, @min(o.param_pos, t.len), paramHeadAt);
                o.pending = found.hit orelse {
                    o.param_pos = @max(o.param_pos, found.again);
                    return;
                };
                // its closing tag may come in a later feed, from a reallocated ``text``: the key is copied now
                o.pending.?.name = try a.dupe(u8, o.pending.?.name);
                o.close_scan = o.pending.?.end;
            }
            const h = o.pending.?;
            const c = paramClose(t, @max(h.end, o.close_scan -| tag_window)) orelse {
                o.close_scan = stop;
                return;
            };
            o.pending = null;
            o.param_pos = c.end;
            if (o.seen.contains(h.name)) continue;
            try o.seen.put(a, h.name, {});
            try o.call.args.append(a, .{ .key = h.name, .value = try paramValue(a, t[h.end..c.start], h.string, s.tools.prop(o.call.name, h.name)) });
        }
    }

    pub fn finish(s: *Stream, text: []const u8, out: *std.ArrayList(Delta)) Allocator.Error!void {
        return s.feed(text, true, out);
    }
};

test "a thinking reply with two calls, streamed in pieces and parsed whole, gives the same arguments" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools_json = (try json.parseText(a,
        \\[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}},
        \\ {"type":"function","function":{"name":"search","parameters":{"properties":{"q":{"type":"string"},"tags":{"type":"array"}}}}}]
    )).ok;
    const tools = try Tools.init(a, tools_json.array);
    const text = "plan it</think>Sure.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n" ++
        "<｜DSML｜ parameter name=\"days\" string=\"true\">3</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"Search\">\n<｜DSML｜ parameter name=\"tags\" string=\"false\">[\"a\", [1</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    const whole = try parse(a, text, true, tools, .{});
    try std.testing.expectEqualStrings("plan it", whole.reasoning);
    try std.testing.expectEqualStrings("Sure.", whole.content);
    try std.testing.expectEqual(@as(usize, 2), whole.calls.len);
    try std.testing.expectEqualStrings("{\"city\":\"Paris\",\"days\":3}", try whole.calls[0].arguments(a));
    try std.testing.expectEqualStrings("search", whole.calls[1].name);
    try std.testing.expectEqualStrings("{\"tags\":[\"a\",[1]]}", try whole.calls[1].arguments(a));
    var s = Stream.init(a, true, tools, .{});
    var out: std.ArrayList(Delta) = .empty;
    var i: usize = 0;
    while (i < text.len) {
        i = @min(text.len, i + 3);
        while (i < text.len and text[i] & 0xC0 == 0x80) i += 1;
        try s.feed(text[0..i], false, &out);
    }
    try s.finish(text, &out);
    var reasoning: std.ArrayList(u8) = .empty;
    var content: std.ArrayList(u8) = .empty;
    var args: [2]std.ArrayList(u8) = .{ .empty, .empty };
    for (out.items) |d| switch (d) {
        .reasoning => |r| try reasoning.appendSlice(a, r),
        .content => |c| try content.appendSlice(a, c),
        .call => {},
        .arguments => |g| try args[g.index].appendSlice(a, g.text),
    };
    try std.testing.expectEqualStrings("plan it", reasoning.items);
    try std.testing.expectEqualStrings("Sure.", content.items);
    try std.testing.expectEqualStrings("{\"city\":\"Paris\",\"days\":3}", args[0].items);
    try std.testing.expectEqualStrings("{\"tags\":[\"a\",[1]]}", args[1].items);
}

test "streamed keys and call names outlive the text they were read from (each feed a new buffer, the old one freed)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools_json = (try json.parseText(a,
        \\[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"integer"}}}}}]
    )).ok;
    const tools = try Tools.init(a, tools_json.array);
    // a long value: its parameter's closing tag arrives many feeds after the head that named it
    const text = "ok</think><｜DSML｜ calls>\n<｜DSML｜ invoke name=\"bash\">\n<｜DSML｜ parameter name=\"command\" string=\"true\">grep -n 'pub fn stdout' /opt/zig/lib/std/Io/File.zig | head -40</｜DSML｜ parameter>\n" ++
        "<｜DSML｜ parameter name=\"timeout\" string=\"false\">30</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    var s = Stream.init(a, true, tools, .{});
    var out: std.ArrayList(Delta) = .empty;
    var prev: ?[]u8 = null;
    var i: usize = 0;
    while (i < text.len) {
        i = @min(text.len, i + 3);
        while (i < text.len and text[i] & 0xC0 == 0x80) i += 1;
        // the server's text grows by reallocation: the bytes a feed saw are freed (and poisoned) before the next
        const buf = try std.testing.allocator.dupe(u8, text[0..i]);
        if (prev) |p| {
            @memset(p, 0xAA);
            std.testing.allocator.free(p);
        }
        prev = buf;
        try s.feed(buf, false, &out);
    }
    try s.finish(prev.?, &out);
    std.testing.allocator.free(prev.?);
    var args: std.ArrayList(u8) = .empty;
    var name: []const u8 = "";
    for (out.items) |d| switch (d) {
        .call => |c| name = c.name,
        .arguments => |g| try args.appendSlice(a, g.text),
        else => {},
    };
    try std.testing.expectEqualStrings("bash", name);
    try std.testing.expectEqualStrings("{\"command\":\"grep -n 'pub fn stdout' /opt/zig/lib/std/Io/File.zig | head -40\",\"timeout\":30}", args.items);
}

test "a call to a tool the request did not offer goes out as a call, whole and streamed; a nameless one stays text" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = try Tools.init(a, (try json.parseText(a,
        \\[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}}}}}]
    )).ok.array);
    const text = "look</think>Here it is.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"take_screenshot\">\n<｜DSML｜ parameter name=\"full\" string=\"false\">true</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    const whole = try parse(a, text, true, tools, .{});
    try std.testing.expectEqualStrings("Here it is.", whole.content);
    try std.testing.expectEqual(@as(usize, 1), whole.calls.len);
    try std.testing.expectEqualStrings("take_screenshot", whole.calls[0].name);
    try std.testing.expectEqualStrings("{\"full\":true}", try whole.calls[0].arguments(a));
    var s = Stream.init(a, true, tools, .{});
    var out: std.ArrayList(Delta) = .empty;
    var i: usize = 0;
    while (i < text.len) {
        i = @min(text.len, i + 5);
        while (i < text.len and text[i] & 0xC0 == 0x80) i += 1;
        try s.feed(text[0..i], false, &out);
    }
    try s.finish(text, &out);
    var name: []const u8 = "";
    var args: std.ArrayList(u8) = .empty;
    var content: std.ArrayList(u8) = .empty;
    for (out.items) |d| switch (d) {
        .call => |c| name = c.name,
        .arguments => |g| try args.appendSlice(a, g.text),
        .content => |c| try content.appendSlice(a, c),
        .reasoning => {},
    };
    try std.testing.expectEqualStrings("take_screenshot", name);
    try std.testing.expectEqualStrings("{\"full\":true}", args.items);
    try std.testing.expectEqualStrings("Here it is.", content.items);
    // a known tool still goes out under its offered spelling
    const cased = try parse(a, "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"BASH\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>", false, tools, .{});
    try std.testing.expectEqualStrings("bash", cased.calls[0].name);
    // an invoke without a name is not a call: the reply stays text
    const nameless = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    const n = try parse(a, nameless, false, tools, .{});
    try std.testing.expectEqual(@as(usize, 0), n.calls.len);
    try std.testing.expectEqualStrings(nameless, n.content);
}
