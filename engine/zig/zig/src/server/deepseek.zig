//! DeepSeek-V4.1 behind the server, as the Python engine's app (``deepseek_v41/cuda/app.py``) serves it: the release
//! tokenizer and encoding (``model_text.Text``), and the family rules (``family.Family``):
//!
//! - **Thinking** on by default (``--no-thinking`` or TF_DSV41_THINKING=0: off); ``chat_template_kwargs.enable_thinking``
//!   / ``thinking`` per request; ``reasoning_effort`` (top level, then ``chat_template_kwargs``): ``none`` / ``minimal``
//!   thinking off, ``low`` 50, ``medium`` / ``high`` 75, ``xhigh`` / ``max`` 100 (the Python app's map; the release
//!   template's medium is 62, but prompts must equal prod's), or an int 1-100; default TF_DSV41_DEFAULT_EFFORT (high).
//! - **Tools**: all offered tools (``tool_choice`` none: none) join the system block; replies' DSML calls are parsed
//!   and streamed as they complete (``serve/tools.zig``); ``parallel_tool_calls: false`` keeps the first call.
//! - **Reasoning memory**: the reasoning of replies this server gave, by tool-call id and content digest, restored
//!   into history messages that come back without it (TF_DSV41_REASONING_MEMORY entries, default 1024).
//! - **Images** (TF_DSV41_IMAGES): ``placeholder`` (default) renders each image part as TF_DSV41_IMAGE_TEXT, a notice
//!   that the model cannot see it; ``reject`` refuses such requests; ``native`` sends them to the vision tower on rank 0
//!   (``serve.vision``: each part a sentinel, then ``<｜deepseek_image｜>``, then its span of virtual ids).
//! - ``response_format``'s schema joins the system block (``## Response Format``; TF_DSV41_SCHEMA_PROMPT=0: not).
//! - Replies' reasoning as ``reasoning_content`` and ``reasoning`` (TF_DSV41_REASONING_FIELDS).
const std = @import("std");
const json = @import("json");
const serve = @import("dsv41_serve");
const errors = @import("errors.zig");
const family = @import("family.zig");
const model_text = @import("model_text.zig");
const ids_mod = @import("ids.zig");
const log = @import("log.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;
const tokenizer = serve.tokenizer;
const template = serve.template;
const tools_mod = serve.tools;
const vision = serve.vision;

pub const model_type = "deepseek_v41";

pub const image_text = "[image omitted: this server does not pass images to the model, so the model cannot see this image. Do not guess what it shows; say that it could not be seen.]";

pub const Images = enum { placeholder, reject, native };

pub const Options = struct {
    images: Images = .placeholder,
    image_text: []const u8 = image_text,
    default_effort: []const u8 = "high",
    reasoning_fields: family.ReasoningFields = .both,
    schema_prompt: bool = true,
    memory_entries: usize = 1024,
    eos: []const u32 = &.{1},
    /// native: the environment the vision limits (TF_DSV41_VISION_*) and TF_DSV41_BIAS_VL are read from at load
    env: ?*const std.process.Environ.Map = null,

    /// The TF_DSV41_* knobs from ``env``; ``problem`` says which one is wrong.
    pub fn fromEnv(env: ?*const std.process.Environ.Map, problem: *[]const u8) error{Invalid}!Options {
        var o: Options = .{};
        const m = env orelse return o;
        if (m.get("TF_DSV41_IMAGES")) |v| {
            const t = std.mem.trim(u8, v, " ");
            o.images = std.meta.stringToEnum(Images, if (t.len == 0) "placeholder" else t) orelse {
                problem.* = "TF_DSV41_IMAGES: expected placeholder, reject or native";
                return error.Invalid;
            };
            o.env = m;
        }
        if (m.get("TF_DSV41_IMAGE_TEXT")) |v| if (std.mem.trim(u8, v, " ").len > 0) {
            o.image_text = std.mem.trim(u8, v, " ");
        };
        if (m.get("TF_DSV41_DEFAULT_EFFORT")) |v| {
            const t = std.mem.trim(u8, v, " ");
            if (t.len > 0) {
                if (level(t) == null or level(t).? == null) {
                    problem.* = "TF_DSV41_DEFAULT_EFFORT: expected low (50), medium (75), high (75), max (100) or an integer 1-100";
                    return error.Invalid;
                }
                o.default_effort = t;
            }
        }
        if (m.get("TF_DSV41_REASONING_FIELDS")) |v| {
            o.reasoning_fields = std.meta.stringToEnum(family.ReasoningFields, std.mem.trim(u8, v, " ")) orelse {
                problem.* = "TF_DSV41_REASONING_FIELDS: expected both, reasoning_content or reasoning";
                return error.Invalid;
            };
        }
        if (m.get("TF_DSV41_SCHEMA_PROMPT")) |v| o.schema_prompt = !std.mem.eql(u8, std.mem.trim(u8, v, " "), "0");
        if (m.get("TF_DSV41_REASONING_MEMORY")) |v| o.memory_entries = std.fmt.parseInt(usize, std.mem.trim(u8, v, " "), 10) catch 1024;
        return o;
    }
};

/// A ``reasoning_effort`` value as a budget (``template.effortOf``: the Python app's names).
const level = template.effortOf;

/// The budget an effort label renders (defaults to high).
pub fn budget(label: []const u8) u8 {
    return ((level(label) orelse return template.default_budget) orelse template.default_budget);
}

/// Python's reasoning memory (GLM 0620): reasoning by tool-call id or content digest, oldest dropped first.
const Memory = struct {
    gpa: Allocator,
    entries: usize,
    lock: std.atomic.Value(bool) = .init(false),
    map: std.StringArrayHashMapUnmanaged([]const u8) = .empty,

    fn acquire(m: *Memory) void {
        while (m.lock.swap(true, .acquire)) std.atomic.spinLoopHint();
    }
    fn release(m: *Memory) void {
        m.lock.store(false, .release);
    }

    fn contentKey(buf: *[66]u8, content: []const u8) []const u8 {
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(content, &d, .{});
        buf[0] = 'c';
        buf[1] = ':';
        _ = std.fmt.bufPrint(buf[2..], "{x}", .{d}) catch unreachable;
        return buf[0..66];
    }

    fn put(m: *Memory, reasoning: []const u8, content: []const u8, calls: []const Value) void {
        if (reasoning.len == 0 or m.entries == 0) return;
        m.acquire();
        defer m.release();
        var buf: [66]u8 = undefined;
        var keys_buf: [64][]const u8 = undefined;
        var keys: []const []const u8 = keys_buf[0..0];
        var n: usize = 0;
        for (calls) |c| if (c.strField("id")) |id| if (n < keys_buf.len) {
            keys_buf[n] = id;
            n += 1;
        };
        keys = keys_buf[0..n];
        if (n == 0) {
            keys_buf[0] = contentKey(&buf, content);
            keys = keys_buf[0..1];
        }
        const value = m.gpa.dupe(u8, reasoning) catch return;
        var used = false;
        for (keys) |k| {
            if (m.map.fetchOrderedRemove(k)) |old| {
                m.gpa.free(old.key);
                m.gpa.free(old.value);
            }
            const key = m.gpa.dupe(u8, k) catch continue;
            const v = if (used) m.gpa.dupe(u8, reasoning) catch {
                m.gpa.free(key);
                continue;
            } else value;
            used = true;
            m.map.put(m.gpa, key, v) catch {
                m.gpa.free(key);
                m.gpa.free(v);
            };
        }
        if (!used) m.gpa.free(value);
        while (m.map.count() > m.entries) {
            const k = m.map.keys()[0];
            const v = m.map.values()[0];
            m.map.orderedRemoveAt(0);
            m.gpa.free(k);
            m.gpa.free(v);
        }
    }

    /// A copy of ``messages`` with remembered reasoning put back where an assistant message lacks it.
    fn restore(m: *Memory, a: Allocator, messages: Value) Allocator.Error!Value {
        if (messages != .array or m.map.count() == 0) return messages;
        var out: ?[]Value = null;
        m.acquire();
        defer m.release();
        for (messages.array, 0..) |msg, i| {
            if (msg != .object) continue;
            const role = msg.get("role") orelse continue;
            if (role != .string or !std.mem.eql(u8, role.string, "assistant")) continue;
            const rc: Value = msg.get("reasoning_content") orelse .null;
            const rs: Value = msg.get("reasoning") orelse .null;
            if (rc.truthy() or rs.truthy()) continue;
            var key: ?[]const u8 = null;
            var buf: [66]u8 = undefined;
            if (msg.get("tool_calls")) |tc| if (tc == .array and tc.array.len > 0 and tc.array[0] == .object) if (tc.array[0].get("id")) |id| if (id == .string and id.string.len > 0) {
                key = id.string;
            };
            if (key == null) if (msg.get("content")) |c| if (c == .string) {
                key = contentKey(&buf, c.string);
            };
            const got = m.map.get(key orelse continue) orelse continue;
            if (out == null) out = try a.dupe(Value, messages.array);
            const copy = try json.copyObject(a, msg.object);
            try copy.put(a, "reasoning_content", .{ .string = try a.dupe(u8, got) });
            out.?[i] = .{ .object = copy };
        }
        return if (out) |o| .{ .array = o } else messages;
    }

    fn deinit(m: *Memory) void {
        for (m.map.keys(), m.map.values()) |k, v| {
            m.gpa.free(k);
            m.gpa.free(v);
        }
        m.map.deinit(m.gpa);
    }
};

/// The DeepSeek-V4.1 serving side of one checkpoint: its tokenizer, the encoding and the family rules.
pub const DeepSeek = struct {
    gpa: Allocator,
    tok: *tokenizer.Tokenizer,
    options: Options,
    memory: Memory,
    images_omitted: std.atomic.Value(u64) = .init(0),
    /// TF_DSV41_IMAGES=native: the image front end (rank 0's prepared-image cache, virtual ids)
    vision: ?*vision.host.Host = null,
    io: ?std.Io = null,

    pub fn load(gpa: Allocator, io: std.Io, dir: []const u8, options: Options, problem: *[]const u8) !*DeepSeek {
        const d = try gpa.create(DeepSeek);
        errdefer gpa.destroy(d);
        const tok = tokenizer.Tokenizer.load(gpa, io, dir) catch |e| {
            problem.* = try std.fmt.allocPrint(gpa, "cannot read {s}/tokenizer.json ({s})", .{ dir, @errorName(e) });
            return error.Load;
        };
        d.* = .{ .gpa = gpa, .tok = tok, .options = options, .memory = .{ .gpa = gpa, .entries = options.memory_entries }, .io = io };
        if (options.images == .native) d.vision = try openVision(gpa, io, dir, options.env, problem);
        return d;
    }

    /// A DeepSeek side over an already loaded tokenizer (tests).
    pub fn init(gpa: Allocator, tok: *tokenizer.Tokenizer, options: Options) Allocator.Error!*DeepSeek {
        const d = try gpa.create(DeepSeek);
        d.* = .{ .gpa = gpa, .tok = tok, .options = options, .memory = .{ .gpa = gpa, .entries = options.memory_entries } };
        return d;
    }

    pub fn deinit(d: *DeepSeek, owns_tokenizer: bool) void {
        if (d.vision) |v| {
            v.deinit();
            d.gpa.destroy(v);
        }
        d.memory.deinit();
        if (owns_tokenizer) d.tok.deinit();
        d.gpa.destroy(d);
    }

    fn self(ctx: *anyopaque) *DeepSeek {
        return @ptrCast(@alignCast(ctx));
    }

    // -- model_text.Text --------------------------------------------------------------------------------------------

    pub fn text(d: *DeepSeek) model_text.Text {
        return .{ .ctx = d, .vtable = &.{
            .encode = encodeFn,
            .decode = decodeFn,
            .token_id = tokenIdFn,
            .token_string = tokenStringFn,
            .vocab_size = vocabFn,
            .eos_ids = eosFn,
            .render = renderTextFn,
            .template_source = sourceFn,
            .special = specialFn,
        } };
    }

    fn encodeFn(ctx: *anyopaque, a: Allocator, s: []const u8, _: bool) model_text.Error![]u32 {
        return self(ctx).tok.encodeAlloc(a, s); // the release tokenizer adds nothing for add_special_tokens
    }

    fn decodeFn(ctx: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        return self(ctx).tok.decodeAlloc(a, ids);
    }

    fn tokenIdFn(ctx: *anyopaque, piece: []const u8) ?u32 {
        return self(ctx).tok.tokenId(piece);
    }

    fn tokenStringFn(ctx: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return a.dupe(u8, self(ctx).tok.tokenString(id));
    }

    fn vocabFn(ctx: *anyopaque) u32 {
        return self(ctx).tok.vocabSize();
    }

    fn eosFn(ctx: *anyopaque) []const u32 {
        return self(ctx).options.eos;
    }

    fn sourceFn(_: *anyopaque) []const u8 {
        return ""; // the encoding is code, not a template: no effort names to read from it
    }

    fn specialFn(ctx: *anyopaque, id: u32) bool {
        return self(ctx).tok.isSpecial(id);
    }

    fn imageHook(ctx: ?*anyopaque, _: Value) []const u8 {
        const d: *DeepSeek = @ptrCast(@alignCast(ctx.?));
        _ = d.images_omitted.fetchAdd(1, .monotonic);
        return d.options.image_text;
    }

    fn templateOptions(d: *DeepSeek, offered: []const Value, thinking: bool, effort: []const u8, generation: bool) template.Options {
        return .{
            .tools = offered,
            .thinking = thinking,
            .budget = budget(effort),
            .add_generation_prompt = generation,
            .image = if (d.options.images == .placeholder) .{ .ctx = d, .text = imageHook } else null,
        };
    }

    /// The generic path's render (template probes, /tokenize without the family): the encoding with its options.
    fn renderTextFn(ctx: *anyopaque, a: Allocator, messages: Value, o: model_text.RenderOptions, problem: *[]const u8) model_text.Error![]u8 {
        const d = self(ctx);
        var p: template.Problem = .{};
        const offered: []const Value = if (o.tools) |t| (if (t == .array) t.array else &.{}) else &.{};
        return template.render(a, messages, d.templateOptions(offered, o.enable_thinking, o.reasoning_effort orelse d.options.default_effort, o.add_generation_prompt), &p) catch |e| switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            error.Refused => {
                problem.* = p.message;
                return error.Template;
            },
        };
    }

    // -- family.Family -------------------------------------------------------------------------------------------------

    pub fn family_(d: *DeepSeek) family.Family {
        return .{ .ctx = d, .vt = switch (d.options.reasoning_fields) {
            inline else => |f| &struct {
                const vt: family.Family.VTable = .{
                    .name = model_type,
                    .thinking = thinkingFn,
                    .default_effort = defaultEffortFn,
                    .tools = toolsFn,
                    .render = renderFn,
                    .reader = readerFn,
                    .remember = rememberFn,
                    .reasoning_fields = f,
                    .reads_min_p = false,
                };
            }.vt,
        } };
    }

    /// ``options``: ``reasoning_effort`` (top level, then ``chat_template_kwargs``), then ``enable_thinking`` and
    /// ``thinking`` in ``chat_template_kwargs``.
    fn thinkingFn(_: *anyopaque, cx: *Cx, body: Value) errors.Refused!family.Thinking {
        var out: family.Thinking = .{};
        const kwargs = body.get("chat_template_kwargs") orelse Value.null;
        if (kwargs != .null and kwargs != .object) return cx.refuse("chat_template_kwargs must be a JSON object or null");
        for ([_]?Value{ body.get("reasoning_effort"), if (kwargs == .object) kwargs.get("reasoning_effort") else null }) |raw| {
            const v = raw orelse continue;
            if (v == .null) continue;
            const t: []const u8 = switch (v) {
                .string => |s| s,
                .int => |s| s,
                else => return cx.refuse("reasoning_effort must be 1-100 or low / medium / high / max"),
            };
            const got = level(t) orelse return cx.refuse("reasoning_effort must be 1-100 or low / medium / high / max");
            if (got) |n| {
                out.enable = true;
                out.effort = try std.fmt.allocPrint(cx.a, "{d}", .{n});
            } else out.enable = false;
        }
        if (kwargs == .object) for ([_][]const u8{ "enable_thinking", "thinking" }) |name| if (kwargs.get(name)) |v| {
            if (v != .bool) return cx.fail(.request, "chat_template_kwargs.{s} must be true or false", .{name});
            out.enable = v.bool;
        };
        return out;
    }

    fn defaultEffortFn(ctx: *anyopaque) []const u8 {
        return self(ctx).options.default_effort;
    }

    fn toolsFn(_: *anyopaque, cx: *Cx, body: Value) errors.Refused![]const Value {
        const t = body.get("tools") orelse return &.{};
        if (t == .null) return &.{};
        if (t != .array) return cx.refuse("tools must be a list of objects");
        for (t.array) |x| if (x != .object) return cx.refuse("tools must be a list of objects");
        if (body.get("tool_choice")) |c| if (c == .string and std.mem.eql(u8, c.string, "none")) return &.{};
        return t.array;
    }

    /// ``schema_for_prompt``: what ``## Response Format`` shows for ``response_format``, or null.
    fn schemaForPrompt(a: Allocator, body: Value) Allocator.Error!?Value {
        const rf = body.get("response_format") orelse return null;
        if (rf != .object) return null;
        const kind = rf.get("type") orelse return null;
        if (kind != .string) return null;
        if (std.mem.eql(u8, kind.string, "json_object")) return (try json.parseText(a, "{\"type\": \"object\"}")).ok;
        if (!std.mem.eql(u8, kind.string, "json_schema")) return null;
        const js = rf.get("json_schema") orelse return null;
        if (!js.truthy() or js != .object) return null;
        var schema = js.get("schema") orelse js.get("json_schema") orelse return null;
        if (schema == .string) schema = switch (try json.parseText(a, schema.string)) {
            .ok => |v| v,
            .err => return null,
        };
        return schema;
    }

    fn renderFn(ctx: *anyopaque, cx: *Cx, p: family.Prompt) errors.Refused!family.Rendered {
        const d = self(ctx);
        const a = cx.a;
        if (p.messages != .array or p.messages.array.len == 0) return cx.refuse("messages must be a non-empty list");
        const messages = try d.memory.restore(a, p.messages);
        var o = d.templateOptions(p.tools, p.thinking, p.effort, p.generation);
        if (d.options.schema_prompt) o.response_format = try schemaForPrompt(a, p.body);
        var counter: Counter = .{ .d = d };
        if (o.image != null) o.image = .{ .ctx = &counter, .text = Counter.hook };
        if (d.vision) |vh| return d.renderNative(cx, vh, messages, &o);
        var problem: template.Problem = .{};
        const prompt = template.render(a, messages, o, &problem) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Refused => return cx.refuse(problem.message),
        };
        const ids = try d.tok.encodeAlloc(a, prompt);
        if (counter.n > 0 and p.generation) log.line("images: {d} image part(s) replaced by the placeholder notice (TF_DSV41_IMAGES=placeholder)", .{counter.n});
        return .{ .ids = ids, .images_omitted = counter.n };
    }

    /// TF_DSV41_IMAGES=native: each image part a sentinel, the sentinels as the placeholder in prompt order, the
    /// images fetched / decoded / preprocessed, each placeholder id as its span of virtual ids (``app._render``,
    /// ``prompt_ids``); the prepared images held for the request.
    fn renderNative(d: *DeepSeek, cx: *Cx, vh: *vision.host.Host, messages: Value, o: *template.Options) errors.Refused!family.Rendered {
        const a = cx.a;
        var marker = vision.parts.Marker.init(a, d.io.?);
        o.image = .{ .ctx = &marker, .text = vision.parts.Marker.hook };
        var problem: template.Problem = .{};
        const prompt = template.render(a, messages, o.*, &problem) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Refused => return cx.refuse(problem.message),
        };
        var vp: vision.parts.Problem = .{};
        const sp = marker.splice(prompt, &vp) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Refused => return cx.refuse(vp.message),
        };
        const ids = try d.tok.encodeAlloc(a, sp.text);
        if (sp.urls.len == 0) return .{ .ids = ids };
        const shared = vh.images(a, sp.urls, &vp) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Refused => return if (vp.busy) cx.fail(.capacity, "{s}", .{vp.message}) else cx.refuse(vp.message),
        };
        const hold = try a.create(Hold);
        hold.* = .{ .vh = vh, .shared = shared };
        errdefer hold.release();
        const preps = try a.alloc(vision.prep.Prepared, shared.len);
        const imgs = try a.alloc(vision.held.HeldImage, shared.len);
        for (shared, preps, imgs) |s, *p, *h| {
            p.* = s.p;
            h.* = .{ .patches = s.p.patches.ptr, .vit_h = s.p.vit_h, .vit_w = s.p.vit_w, .llm_h = s.p.llm_h, .llm_w = s.p.llm_w, .digest = s.p.digest, .vids = s.p.vids.ptr, .n_vids = s.p.vids.len };
        }
        const expanded = vision.parts.expand(a, ids, preps, vh.s.image_token, true, &vp) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Refused => return cx.refuse(vp.message),
        };
        hold.h = .{ .images = imgs.ptr, .n = imgs.len };
        return .{ .ids = expanded, .images = .{ .held = &hold.h, .token = vh.s.image_token, .ctx = hold, .release_fn = Hold.releaseFn } };
    }

    /// A request's prepared images, given back to the cache's counts at the request's end.
    const Hold = struct {
        vh: *vision.host.Host,
        shared: []const *vision.host.Shared,
        h: vision.held.Held = .{ .images = undefined, .n = 0 },
        done: bool = false,

        fn release(x: *Hold) void {
            if (x.done) return;
            x.done = true;
            x.vh.releaseAll(x.shared);
        }
        fn releaseFn(ctx: *anyopaque) void {
            release(@ptrCast(@alignCast(ctx)));
        }
    };

    fn openVision(gpa: Allocator, io: std.Io, dir: []const u8, env: ?*const std.process.Environ.Map, problem: *[]const u8) !*vision.host.Host {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const cfg = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24)) catch null;
        defer if (cfg) |c| gpa.free(c);
        const arena = try gpa.create(std.heap.ArenaAllocator); // the settings' vision_config lives as long as the server
        arena.* = std.heap.ArenaAllocator.init(gpa);
        var msg: []const u8 = "";
        const s = vision.prep.Settings.read(arena.allocator(), if (cfg) |c| try arena.allocator().dupe(u8, c) else null, env, &msg) catch |e| {
            problem.* = try std.fmt.allocPrint(gpa, "TF_DSV41_IMAGES=native: {s} ({t})", .{ msg, e });
            return error.Load;
        };
        const h = try gpa.create(vision.host.Host);
        h.* = vision.host.Host.init(gpa, io, s);
        try h.enableFetch(); // TF_DSV41_VISION_FETCH (default 1): https image URLs as prod fetches them
        log.line("images: native (TF_DSV41_IMAGES), at most {d} a request, {d} positions an image; image URLs: {s}", .{ s.max_images, s.budget(), if (s.fetch) "data: and https (public addresses, pinned)" else "data: only (TF_DSV41_VISION_FETCH=0)" });
        return h;
    }

    const Counter = struct {
        d: *DeepSeek,
        n: u32 = 0,
        fn hook(ctx: ?*anyopaque, _: Value) []const u8 {
            const c: *Counter = @ptrCast(@alignCast(ctx.?));
            c.n += 1;
            return c.d.options.image_text;
        }
    };

    fn rememberFn(ctx: *anyopaque, reasoning: []const u8, content: []const u8, calls: []const Value) void {
        self(ctx).memory.put(reasoning, content, calls);
    }

    fn readerFn(ctx: *anyopaque, a: Allocator, thinking: bool, offered: []const Value, single: bool) Allocator.Error!family.Reader {
        const d = self(ctx);
        const r = try a.create(Reply);
        r.* = .{ .a = a, .d = d, .detok = .{ .tok = d.tok }, .stream = tools_mod.Stream.init(a, thinking, try tools_mod.Tools.init(a, offered), .{ .make = callId }), .thinking = thinking, .offered = try tools_mod.Tools.init(a, offered), .single = single };
        return .{ .ctx = r, .vt = &.{ .push = Reply.push, .pending = Reply.pending, .feed = Reply.feed, .parse = Reply.parse, .calls = Reply.calls } };
    }

    fn callId(_: ?*anyopaque, a: Allocator, _: usize) Allocator.Error![]const u8 {
        return ids_mod.make(a, "call_", 24);
    }
};

/// One reply's text and DSML stream (``family.Reader``).
const Reply = struct {
    a: Allocator,
    d: *DeepSeek,
    detok: tokenizer.Detokenizer,
    text: std.ArrayList(u8) = .empty,
    stream: tools_mod.Stream,
    deltas: std.ArrayList(tools_mod.Delta) = .empty,
    thinking: bool,
    offered: tools_mod.Tools,
    single: bool,
    flushed: bool = false,

    fn self(ctx: *anyopaque) *Reply {
        return @ptrCast(@alignCast(ctx));
    }

    fn push(ctx: *anyopaque, ids: []const u32) Allocator.Error!void {
        const r = self(ctx);
        for (ids) |id| try r.detok.push(r.a, id, &r.text);
    }

    fn pending(ctx: *anyopaque) bool {
        return self(ctx).detok.pending();
    }

    fn feed(ctx: *anyopaque, finished: bool, out: *std.ArrayList(Value)) Allocator.Error!void {
        const r = self(ctx);
        const a = r.a;
        if (finished and !r.flushed) {
            try r.detok.flush(a, &r.text);
            r.flushed = true;
        }
        r.deltas.clearRetainingCapacity();
        try r.stream.feed(r.text.items, finished, &r.deltas);
        const fam = r.d.family_();
        for (r.deltas.items) |delta| switch (delta) {
            .reasoning => |t| try out.append(a, try fam.reasoningDelta(a, t)),
            .content => |t| try out.append(a, .{ .string = t }),
            .call => |c| {
                if (r.single and c.index > 0) continue;
                const f = try json.newObject(a);
                try f.put(a, "name", .{ .string = c.name });
                try f.put(a, "arguments", .{ .string = "" });
                const call = try json.newObject(a);
                try call.put(a, "index", try json.intValue(a, c.index));
                try call.put(a, "id", .{ .string = c.id });
                try call.put(a, "type", .{ .string = "function" });
                try call.put(a, "function", .{ .object = f });
                try out.append(a, try toolCallsDelta(a, .{ .object = call }));
            },
            .arguments => |g| {
                if (r.single and g.index > 0) continue;
                const f = try json.newObject(a);
                try f.put(a, "arguments", .{ .string = try a.dupe(u8, g.text) });
                const call = try json.newObject(a);
                try call.put(a, "index", try json.intValue(a, g.index));
                try call.put(a, "function", .{ .object = f });
                try out.append(a, try toolCallsDelta(a, .{ .object = call }));
            },
        };
    }

    fn toolCallsDelta(a: Allocator, call: Value) Allocator.Error!Value {
        const list = try a.alloc(Value, 1);
        list[0] = call;
        const o = try json.newObject(a);
        try o.put(a, "tool_calls", .{ .array = list });
        return .{ .object = o };
    }

    fn openai(a: Allocator, c: *const tools_mod.Call) Allocator.Error!Value {
        const f = try json.newObject(a);
        try f.put(a, "name", .{ .string = c.name });
        try f.put(a, "arguments", .{ .string = try c.arguments(a) });
        const o = try json.newObject(a);
        try o.put(a, "id", .{ .string = c.id });
        try o.put(a, "type", .{ .string = "function" });
        try o.put(a, "function", .{ .object = f });
        return .{ .object = o };
    }

    fn parse(ctx: *anyopaque) Allocator.Error!family.Parsed {
        const r = self(ctx);
        if (!r.flushed) {
            try r.detok.flush(r.a, &r.text);
            r.flushed = true;
        }
        const p = try tools_mod.parse(r.a, r.text.items, r.thinking, r.offered, .{ .make = DeepSeek.callId });
        const n = if (r.single) @min(1, p.calls.len) else p.calls.len;
        const list = try r.a.alloc(Value, n);
        for (p.calls[0..n], list) |*c, *slot| slot.* = try openai(r.a, c);
        return .{ .reasoning = p.reasoning, .content = p.content, .calls = list };
    }

    fn calls(ctx: *anyopaque) Allocator.Error![]Value {
        const r = self(ctx);
        const all = r.stream.calls.items;
        const n = if (r.single) @min(1, all.len) else all.len;
        const out = try r.a.alloc(Value, n);
        for (all[0..n], out) |*c, *slot| slot.* = try openai(r.a, c);
        return out;
    }
};

test "reasoning_effort and the thinking switches read as the Python app reads them" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const body = (try json.parseText(cx.a, "{\"reasoning_effort\": \"medium\", \"chat_template_kwargs\": {\"reasoning_effort\": 90}}")).ok;
    const got = try DeepSeek.thinkingFn(undefined, &cx, body);
    try std.testing.expectEqualStrings("90", got.effort.?);
    try std.testing.expect(got.enable.?);
    const off = try DeepSeek.thinkingFn(undefined, &cx, (try json.parseText(cx.a, "{\"reasoning_effort\": \"none\"}")).ok);
    try std.testing.expect(!off.enable.?);
    try std.testing.expectEqual(@as(u8, 75), budget("medium"));
    try std.testing.expectEqual(@as(u8, 100), budget("xhigh"));
    try std.testing.expectError(error.Refused, DeepSeek.thinkingFn(undefined, &cx, (try json.parseText(cx.a, "{\"reasoning_effort\": 101}")).ok));
    try std.testing.expectError(error.Refused, DeepSeek.thinkingFn(undefined, &cx, (try json.parseText(cx.a, "{\"chat_template_kwargs\": {\"thinking\": 1}}")).ok));
}

test "TF_DSV41_IMAGES=native: an image part becomes its span of virtual ids, held for the request (mini tokenizer)" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const tok = try serve.tokenizer.Tokenizer.parse(gpa, serve.fixtures.mini_tokenizer);
    defer tok.deinit();
    const d = try DeepSeek.init(gpa, tok, .{ .images = .native });
    defer d.deinit(false);
    const vh = try gpa.create(vision.host.Host);
    vh.* = vision.host.Host.init(gpa, std.testing.io, .{ .image_token = 1003 }); // the mini tokenizer's image id
    d.vision = vh;
    d.io = std.testing.io;
    const png = "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAGCAIAAABxZ0isAAAAoUlEQVR4nAGWAGn/AAAAAB4eHjw8PFpaWnh4eJaWlrS0tNLS0gAHBwclJSVDQ0NhYWF/f3+dnZ27u7vZ2dkADg4OLCwsSkpKaGhohoaGpKSkwsLC4ODgABUVFTMzM1FRUW9vb42Njaurq8nJyefn5wAcHBw6OjpYWFh2dnaUlJSysrLQ0NDu7u4AIyMjQUFBX19ffX19m5ububm519fX9fX1CLhE6bQqUiMAAAAASUVORK5CYII=";
    const body = "[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"a typed <\u{ff5c}deepseek_image\u{ff5c}> stays text\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64," ++ png ++ "\"}},{\"type\":\"text\",\"text\":\"what is it\"}]}]";
    const messages = (try json.parseText(cx.a, body)).ok;
    const got = try d.family_().render(&cx, .{ .messages = messages, .tools = &.{}, .thinking = false, .effort = "high", .body = .null });
    const im = got.images orelse return error.NoImages;
    defer im.release();
    const h: *const vision.held.Held = @ptrCast(@alignCast(im.held));
    try std.testing.expectEqual(@as(u64, 1), h.n);
    const img = h.slice()[0];
    var span: usize = 0;
    var at: ?usize = null;
    for (got.ids, 0..) |t, i| if (vision.vids.isVid(t)) {
        if (at == null) at = i;
        span += 1;
    };
    try std.testing.expectEqual(@as(usize, img.n_vids), span); // one image: its whole span, in one run
    try std.testing.expectEqualSlices(u32, img.vids[0..img.n_vids], got.ids[at.? .. at.? + span]);
    try std.testing.expect(std.mem.indexOfScalar(u32, got.ids, 1003) == null); // the typed placeholder stayed text
    try std.testing.expectEqual(@as(u32, 12), img.llm_h); // 8 x 6 scaled up to 544^2: 630 x 476 -> 45 x 34 patches
}
