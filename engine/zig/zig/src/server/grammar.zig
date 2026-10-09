//! Structured-output fields checked as ``engine.grammar.request_spec`` checks them; the engine enforces them.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const pyrepr = @import("pyrepr.zig");
const tool_specs = @import("tool_specs.zig");
const Value = json.Value;
const Cx = errors.Cx;

pub const fields = [_][]const u8{ "response_format", "guided_json", "guided_regex", "guided_choice", "guided_grammar", "structured_outputs" };

pub const Kind = @FieldType(api.Structure, "kind");
pub const Spec = struct { kind: Kind, text: []const u8 = "", field: []const u8 = "response_format" };

/// The JSON decoder's own message, without its position.
fn bareMessage(full: []const u8) []const u8 {
    const at = std.mem.lastIndexOf(u8, full, ": line ") orelse return full;
    return full[0..at];
}

fn schemaText(cx: *Cx, raw: Value, where: []const u8) errors.Refused![]const u8 {
    var value = raw;
    if (raw == .string) value = switch (try json.parseText(cx.a, raw.string)) {
        .ok => |v| v,
        .err => |e| return cx.fail(.request, "{s} is not valid JSON: {s}", .{ where, bareMessage(e) }),
    };
    if (value == .bool) value = if (value.bool) .{ .object = try json.newObject(cx.a) } else .null;
    if (value != .object) return cx.fail(.request, "{s} must be a JSON schema object", .{where});
    return json.stringify(cx.a, value, .{ .ascii = false });
}

fn text(cx: *Cx, v: Value, where: []const u8) errors.Refused![]const u8 {
    if (v != .string or v.string.len == 0) return cx.fail(.request, "{s} must be a non-empty string", .{where});
    return v.string;
}

fn choices(cx: *Cx, v: Value, where: []const u8) errors.Refused![]const u8 {
    const ok = v == .array and v.array.len > 0 and for (v.array) |c| {
        if (c != .string or c.string.len == 0) break false;
    } else true;
    if (!ok) return cx.fail(.request, "{s} must be a non-empty list of non-empty strings", .{where});
    return json.stringify(cx.a, v, .{ .ascii = false });
}

fn read(cx: *Cx, kind: Kind, v: Value, where: []const u8) errors.Refused![]const u8 {
    return switch (kind) {
        .json_schema => schemaText(cx, v, where),
        .choice => choices(cx, v, where),
        else => text(cx, v, where),
    };
}

/// The body's structured-output request, null for plain text; a refusal when it is malformed.
pub fn spec(cx: *Cx, body: Value) errors.Refused!?Spec {
    if (body.field("response_format")) |rf| {
        if (rf != .object) return cx.refuse("response_format must be an object such as {\"type\": \"json_object\"}");
        const kind = rf.get("type");
        const name = if (kind != null and kind.? == .string) kind.?.string else "";
        if (kind != null and kind.? == .string and std.mem.eql(u8, name, "json_object")) return .{ .kind = .json };
        if (kind != null and kind.? == .string and std.mem.eql(u8, name, "json_schema")) {
            const js = rf.get("json_schema");
            if (js == null or js.? != .object or js.?.field("schema") == null) return cx.refuse("response_format json_schema needs json_schema.schema (a JSON schema object)");
            return .{ .kind = .json_schema, .text = try schemaText(cx, js.?.get("schema").?, "response_format json_schema.schema") };
        }
        if (!(kind != null and kind.? == .string and std.mem.eql(u8, name, "text")))
            return cx.fail(.request, "response_format type must be text, json_object or json_schema, not {s}", .{try pyrepr.repr(cx.a, kind)});
    }
    const guided = [_]struct { []const u8, Kind }{ .{ "guided_json", .json_schema }, .{ "guided_regex", .regex }, .{ "guided_choice", .choice }, .{ "guided_grammar", .grammar } };
    for (guided) |g| if (body.field(g[0])) |v| return .{ .kind = g[1], .text = try read(cx, g[1], v, g[0]), .field = g[0] };
    const so = body.field("structured_outputs") orelse return null;
    if (so != .object) return cx.refuse("structured_outputs must be an object such as {\"json\": {...}}");
    var given: std.ArrayList([]const u8) = .empty;
    for (so.object.keys(), so.object.values()) |k, v| if (v != .null and !(v == .bool and !v.bool)) try given.append(cx.a, k);
    std.mem.sort([]const u8, given.items, {}, struct {
        fn less(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.order(u8, x, y) == .lt;
        }
    }.less);
    const known = [_]struct { []const u8, Kind }{ .{ "json", .json_schema }, .{ "regex", .regex }, .{ "choice", .choice }, .{ "grammar", .grammar } };
    var other: std.ArrayList([]const u8) = .empty;
    for (given.items) |k| {
        const is_known = for (known) |kn| {
            if (std.mem.eql(u8, kn[0], k)) break true;
        } else std.mem.eql(u8, k, "json_object");
        if (!is_known) try other.append(cx.a, k);
    }
    if (other.items.len > 0) return cx.fail(.request, "structured_outputs {s} is not supported: use json, json_object, regex, choice or grammar", .{try std.mem.join(cx.a, ", ", other.items)});
    for (given.items) |k| for (known) |kn| if (std.mem.eql(u8, kn[0], k))
        return .{ .kind = kn[1], .text = try read(cx, kn[1], so.get(k).?, try std.fmt.allocPrint(cx.a, "structured_outputs.{s}", .{k})), .field = "structured_outputs" };
    if (json.truthyField(so, "json_object")) return .{ .kind = .json, .field = "structured_outputs" };
    return null;
}

/// ``grammar.refusal``: a malformed grammar, or one beside a required call, refused before any header.
pub fn refusal(cx: *Cx, body: Value) errors.Refused!void {
    const s = try spec(cx, body) orelse return;
    const tools = body.get("tools");
    if (tools != null and tools.?.truthy() and try tool_specs.choiceRequiresCall(cx.a, body.get("tool_choice")))
        return cx.fail(.request, "{s} cannot be combined with tool_choice \"required\" or a named function: send one", .{s.field});
}

/// TF_DSV41_TOOL_GRAMMAR=1 (``structured.tool_grammar``): every auto tools request is held to its schemas, as strict
/// tools are.
pub fn toolGrammarEnv() bool {
    const v = std.c.getenv("TF_DSV41_TOOL_GRAMMAR") orelse return false;
    return std.mem.eql(u8, std.mem.trim(u8, std.mem.span(v), " "), "1");
}

fn functionOf(t: Value) ?Value {
    if (t != .object) return null;
    if (t.get("type")) |ty| if (!(ty == .string and std.mem.eql(u8, ty.string, "function"))) return null;
    const f = t.get("function") orelse return null;
    return if (f == .object) f else null;
}

/// GLM 0610's ``_tool_spec`` (DeepSeek-V4.1's structured output, with ``structured.Host.spec``'s TF_DSV41_TOOL_GRAMMAR
/// rule as `auto_all`): the tools spec text when the calls are held to their schemas (``tool_choice`` "required" or a
/// named function, any tool marked ``"strict": true``), null when they are not; a refusal when the tools are
/// malformed. The text is ``{"tools": [...], "tool_choice": ..., "parallel_tool_calls": bool}``.
pub fn toolSpec(cx: *Cx, body: Value, auto_all: bool) errors.Refused!?[]const u8 {
    const tools = body.get("tools") orelse return null;
    if (!tools.truthy()) return null;
    if (tools != .array) return cx.refuse("tools must be a list");
    const a = cx.a;
    var choice: Value = .{ .string = "auto" };
    const asked = body.get("tool_choice") orelse Value.null;
    const auto = asked == .null or (asked == .string and std.mem.eql(u8, asked.string, "auto"));
    if (auto) {
        const strict = for (tools.array) |t| {
            const f = functionOf(t) orelse continue;
            if (f.get("strict")) |s| if (s == .bool and s.bool) break true;
        } else false;
        if (!strict and !auto_all) return null;
    } else if (asked == .string and std.mem.eql(u8, asked.string, "none")) {
        return null;
    } else if (!(asked == .string and std.mem.eql(u8, asked.string, "required"))) {
        const ok = asked == .object and asked.typeIs("function") and asked.get("function") != null and
            asked.get("function").? == .object and asked.get("function").?.get("name") != null and asked.get("function").?.get("name").? == .string;
        if (!ok) return cx.refuse("tool_choice must be \"none\", \"auto\", \"required\" or {\"type\": \"function\", \"function\": {\"name\": ...}}");
        const inner = try json.newObject(a);
        try inner.put(a, "name", asked.get("function").?.get("name").?);
        const outer = try json.newObject(a);
        try outer.put(a, "type", .{ .string = "function" });
        try outer.put(a, "function", .{ .object = inner });
        choice = .{ .object = outer };
    } else choice = asked;
    var kept: std.ArrayList(Value) = .empty;
    for (tools.array) |t| {
        const f = functionOf(t) orelse continue;
        const name = f.get("name");
        if (name == null or name.? != .string or name.?.string.len == 0) return cx.refuse("every function tool needs a name");
        const entry = try json.newObject(a);
        try entry.put(a, "name", name.?);
        if (f.field("parameters")) |p| {
            if (p != .object) return cx.fail(.request, "tool {s}: parameters must be a JSON schema object", .{name.?.string});
            try entry.put(a, "parameters", p);
        }
        if (auto_all and auto) {
            try entry.put(a, "strict", .{ .bool = true });
        } else if (f.get("strict")) |s| try entry.put(a, "strict", .{ .bool = s.truthy() });
        const wrap = try json.newObject(a);
        try wrap.put(a, "type", .{ .string = "function" });
        try wrap.put(a, "function", .{ .object = entry });
        try kept.append(a, .{ .object = wrap });
    }
    if (kept.items.len == 0) return null;
    const parallel = if (body.get("parallel_tool_calls")) |p| !(p == .bool and !p.bool) else true;
    const spec_obj = try json.newObject(a);
    try spec_obj.put(a, "tools", .{ .array = kept.items });
    try spec_obj.put(a, "tool_choice", choice);
    try spec_obj.put(a, "parallel_tool_calls", .{ .bool = parallel });
    return try json.stringify(a, .{ .object = spec_obj }, .{ .ascii = false });
}
