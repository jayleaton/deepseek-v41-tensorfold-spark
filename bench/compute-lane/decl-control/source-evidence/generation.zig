//! The token loop of one reply segment and the text it streams (chat.zig submits; compute_turns.zig resumes).
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const reply_text = @import("reply_text.zig");
const tool_stream = @import("tool_stream.zig");
const tool_parse = @import("tool_parse.zig");
const tool_lfm2 = @import("tool_lfm2.zig");
const log = @import("log.zig");
const clock = @import("clock.zig");
const chat = @import("chat.zig");
const projection = @import("runner_projection.zig");
const Server = @import("server.zig").Server;
const Mailbox = @import("mailbox.zig").Mailbox;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;
const Sink = chat.Sink;
const Reply = chat.Reply;
const Failure = chat.Failure;
const deltaOf = chat.deltaOf;
const nowNs = clock.nowNs;
const seconds = clock.seconds;

/// The token loop and the text it streams.
pub const Generation = struct {
    srv: *Server,
    a: Allocator,
    box: *Mailbox,
    id: api.Id,
    reply_id: []const u8, // the id the client gets, which the done line prints
    seed: ?u64 = null,
    sink: ?Sink,
    thinking: bool,
    stops: reply_text.Stops,
    ignore_eos: bool,
    max_tokens: u32,
    token_ids: bool, // "return_token_ids": the response's tensorfold block carries the reply's token ids
    guide: ?[]const u8 = null, // the runner guide text the chat got ("" none); null: the runner is off
    tools: []const Value,
    calls: ?tool_stream.Streamer = null,
    collected: std.ArrayList(u32) = .empty,
    visible: reply_text.Incremental = .{},
    hidden: std.ArrayList(u8) = .empty, // reused for the answer without its call blocks
    streamed: reply_text.Shown = .{}, // what content streamed, kept apart from the decode buffer that grows under slices
    streamed_reasoning: reply_text.Shown = .{},
    streaming_done: bool = false,
    wrote_ns: ?i96 = null,
    first_ns: ?i96 = null,
    last_ns: ?i96 = null,
    reason: ?[]const u8 = null, // the server ended the reply (a stop string, the length) before the engine said so
    engine_done: bool = false,
    consumed: usize = 0,
    chunk_index: usize = 0,
    lp: ?@import("logprobs.zig").Stream = null, // the request asked for logprobs: its rows, copied with each round
    loop_stop: bool = false, // --loop-guard-stop: the engine ends the reply where the loop guard fires
    hold: bool = false, // the engine holds the reply at an EOS (Request.hold_eos): this segment ends there
    held: bool = false, // it did: the engine has not finished, and compute_turns.zig resumes or ends it
    skip: []const u32 = &.{}, // a resumed segment's first tokens: the inserted tool result, checked and not shown
    cells_sent: usize = 0, // the runner records the stream sent last: a prefix the client has every landed byte of
    cells_sig: u64 = 0, // their Mailbox.covered sig
    deadline_ns: ?i96 = null, // "deadline_s": when the server ends the reply itself
    deadline_hit: bool = false, // it did: the engine was cancelled there; what it gave before its end still lands
    base: usize = 0, // the reply's tokens before this segment's (compute_segments.zig): the runner records count them

    fn eos(g: *const Generation, t: u32) bool {
        return !g.ignore_eos and std.mem.indexOfScalar(u32, g.srv.eos, t) != null;
    }

    /// Close a tool call left open at the model end token, except when another stop condition ended the reply.
    fn closeCall(g: *const Generation, text: []const u8) Allocator.Error![]const u8 {
        const t = g.collected.items;
        if (g.tools.len == 0 or t.len == 0 or !g.eos(t[t.len - 1])) return "";
        return tool_parse.closeCall(g.a, text, g.tools);
    }

    /// The engine's stop check, here: the newest tokens' text holds a stop string.
    fn stopHit(g: *Generation) Allocator.Error!bool {
        if (g.stops.strings.len == 0) return false;
        const t = g.collected.items;
        const tail = t[t.len -| g.stops.tail()..];
        const text = try g.srv.text.decode(g.a, tail);
        for (g.stops.strings) |s| if (std.mem.indexOf(u8, text, s) != null) return true;
        return false;
    }

    /// Commits a round's tokens as ``LaneStream.commit`` would, then streams what they add.
    fn commit(g: *Generation, chunk: []const u32) Failure!void {
        var landed: usize = 0;
        for (chunk) |t| {
            if (g.reason != null) break;
            try g.collected.append(g.a, t);
            landed += 1;
            if (g.eos(t)) {
                g.reason = "stop";
                g.held = g.hold;
            } else if (try g.stopHit()) {
                g.reason = "stop";
                g.srv.engine.cancel(g.id); // the engine ends EOS and length itself; a stop string is the server's
            } else if (g.collected.items.len >= g.max_tokens) g.reason = "length";
        }
        if (landed == 0) return;
        const arrived = nowNs(g.srv.io);
        if (g.first_ns == null) g.first_ns = arrived;
        g.last_ns = arrived;
        const sink = g.sink orelse return;
        if (g.streaming_done) return;
        var fresh: std.ArrayList(u32) = .empty;
        for (chunk[0..landed]) |t| {
            if (g.eos(t)) {
                g.streaming_done = true;
                break;
            }
            try fresh.append(g.a, t);
        }
        const all = try g.visible.extend(g.a, g.srv.text, reply_text.decodeText, fresh.items);
        const text = g.stops.visible(all, true);
        if (g.visible.pending()) return; // a character still split across tokens
        var answer = text;
        if (g.thinking) {
            const close = try g.visible.closeOf(g.a, reply_text.decodeText, g.srv.text, g.srv.close_id);
            const split = try reply_text.splitThinkingAt(g.a, text, false, g.srv.markers, close);
            const piece = g.streamed_reasoning.after(split.reasoning, true);
            if (piece.len > 0) {
                try g.streamed_reasoning.set(g.a, split.reasoning, piece);
                try g.emit(sink, try deltaOf(g.a, "reasoning_content", piece));
            }
            answer = split.answer;
        }
        answer = try tool_lfm2.dsml(g.a, answer, g.tools, false); // the answer only, so a call the reasoning mentions is not made
        const shown = if (g.calls != null) try reply_text.hideInto(g.a, &g.hidden, answer, false, true) else answer;
        const vis = reply_text.heldBack(reply_text.streamingVisible(shown), g.calls != null);
        const copied = @intFromPtr(vis.ptr) >= @intFromPtr(g.hidden.items.ptr) and @intFromPtr(vis.ptr) <= @intFromPtr(g.hidden.items.ptr) + g.hidden.items.len;
        const delta = g.streamed.after(vis, !copied);
        if (delta.len > 0) {
            try g.streamed.set(g.a, vis, delta);
            try g.emit(sink, if (g.lp) |*l| (try l.delta(g.srv, g.a, g.thinking, g.ignore_eos, g.collected.items, delta, g.streamed.text.items.len, false)).? else .{ .string = delta });
        }
        if (g.calls) |*c| {
            var out: std.ArrayList(Value) = .empty;
            try c.feed(answer, &out); // never the reasoning: a call it mentions is not made
            for (out.items) |d| try g.emit(sink, d);
        }
    }

    fn emit(g: *Generation, sink: Sink, delta: Value) Failure!void {
        sink.call(sink.ctx, delta) catch {
            g.cancel();
            return error.Cancelled;
        };
        g.wrote_ns = nowNs(g.srv.io);
    }

    /// End the request and wait for the engine; unfinished requests count as disconnects.
    pub fn cancel(g: *Generation) void {
        if (!g.engine_done) g.srv.engine.cancel(g.id);
        g.drain();
        if (g.box.reason == .cancelled and g.reason == null) g.srv.metrics.disconnected(g.srv.io);
    }

    /// Waits for the engine's own end, so the request it holds may be freed.
    pub fn drain(g: *Generation) void {
        const m = g.box;
        m.mutex.lockUncancelable(m.io);
        defer m.mutex.unlock(m.io);
        while (!m.finished) m.cond.waitUncancelable(m.io, &m.mutex);
        g.engine_done = true;
    }

    /// Takes rounds until the reply ends; a client that leaves cancels it.
    pub fn loop(g: *Generation, gone: anytype) Failure!void {
        const Gone = switch (@typeInfo(@TypeOf(gone))) {
            .pointer => |p| p.child,
            else => @TypeOf(gone),
        };
        const m = g.box;
        while (true) {
            m.mutex.lockUncancelable(m.io);
            var chunk: ?[]u32 = null;
            var ended = false;
            if (g.chunk_index < m.chunks.items.len) {
                const end = m.chunks.items[g.chunk_index];
                chunk = g.a.dupe(u32, m.tokens.items[g.consumed..end]) catch |e| {
                    m.mutex.unlock(m.io);
                    return e;
                };
                if (g.lp) |*l| l.rows.appendSlice(g.a, m.rows.items[g.consumed..end]) catch |e| {
                    g.a.free(chunk.?);
                    m.mutex.unlock(m.io);
                    return e;
                };
                g.chunk_index += 1;
                g.consumed = end;
            } else if (m.finished) {
                ended = true;
            } else {
                m.cond.waitTimeout(m.io, &m.mutex, .{ .duration = .{ .raw = .fromMilliseconds(50), .clock = .awake } }) catch {};
            }
            m.mutex.unlock(m.io);
            if (ended) {
                g.engine_done = true;
                return;
            }
            if (gone.check()) {
                g.cancel();
                return error.Cancelled;
            }
            if (g.deadline_ns) |d| if (!g.deadline_hit and g.reason == null and nowNs(g.srv.io) >= d) {
                g.deadline_hit = true;
                g.srv.engine.cancel(g.id); // the engine ends the stream: its tokens, usage and receipt stay whole
            };
            if (chunk == null) if (g.sink) |sink| try g.liveCells(sink);
            if (chunk == null and @hasDecl(Gone, "beat")) {
                // queueing and a long prefill send nothing: a client that times out a silent stream would leave
                const now = nowNs(g.srv.io);
                const since = g.wrote_ns orelse now;
                if (g.wrote_ns == null) g.wrote_ns = now;
                if (gone.beat(now - since) catch {
                    g.cancel();
                    return error.Cancelled;
                }) g.wrote_ns = now;
            }
            const tokens = try g.skipInserted(chunk orelse continue);
            if (g.reason == null) try g.commit(tokens);
            if (g.sink) |sink| try g.liveCells(sink);
            if (g.held) return;
        }
    }

    /// Sends the runner records whose bytes the client has (`tensorfold.runner.cells`), so a cut stream keeps them.
    pub fn liveCells(g: *Generation, sink: Sink) Failure!void {
        const m = g.box;
        var text: std.ArrayList(u8) = .empty;
        {
            m.mutex.lockUncancelable(m.io);
            defer m.mutex.unlock(m.io);
            const got = m.covered(.{ g.streamed_reasoning.text.items.len, g.streamed.text.items.len });
            // never fewer than sent: the text may have been replaced shorter (a stop string's held prefix)
            if (got.n < g.cells_sent or got.sig == g.cells_sig) return; // records off: n 0, sig 0, so never
            try text.appendSlice(g.a, "{\"runner\":{\"live\":true,\"cells\":[");
            for (m.cells.items[0..got.n], 0..) |versions, i| {
                if (i > 0) try text.append(g.a, ',');
                try text.appendSlice(g.a, versions.items[0].json);
            }
            try text.appendSlice(g.a, "]}}");
            g.cells_sent = got.n;
            g.cells_sig = got.sig;
        }
        const v = switch (try json.parse(g.a, text.items)) {
            .ok => |v| v,
            .err => return,
        };
        const delta = try json.newObject(g.a);
        try delta.put(g.a, "tensorfold", v); // openai.zig's chunk puts it beside the choices, as the final chunk's receipt
        try g.emit(sink, .{ .object = delta });
    }

    /// Drops the inserted result that opens a resumed segment; a different token there fails the reply.
    fn skipInserted(g: *Generation, tokens: []u32) Failure![]u32 {
        const n = @min(tokens.len, g.skip.len);
        if (!std.mem.eql(u32, tokens[0..n], g.skip[0..n])) {
            log.line("error: {s}: the engine's tokens after a tool result are not the result it was given", .{g.reply_id});
            g.cancel();
            return error.Failed;
        }
        g.skip = g.skip[n..];
        return tokens[n..];
    }

    /// The engine-owned bytes of `content` (runner_owned.zig), from the runner records the stream has had so far.
    fn ownedOf(g: *Generation, tokens: []const u32, raw: []const u8, content: []const u8) Allocator.Error!?reply_text.Owned {
        if (g.srv.text.vtable.token_bytes == null) return null;
        var records: std.ArrayList([]const u8) = .empty;
        {
            const m = g.box;
            m.mutex.lockUncancelable(m.io);
            defer m.mutex.unlock(m.io);
            for (m.cells.items) |versions| if (versions.items.len > 0)
                try records.append(g.a, try g.a.dupe(u8, versions.items[versions.items.len - 1].json));
        }
        if (records.items.len == 0) return null;
        const pieces = try g.a.alloc([]const u8, tokens.len);
        for (tokens, pieces) |t, *p| p.* = (try g.srv.text.tokenBytes(g.a, t)) orelse return null;
        return @import("runner_owned.zig").of(g.a, records.items, g.base, pieces, raw, content);
    }

    pub fn finish(g: *Generation, cx: *Cx, prompt_len: usize, received: i96, thinking: bool, effort: ?[]const u8, exact: bool, drafts: bool) Failure!Reply {
        const a = g.a;
        const m = g.box;
        if (!g.engine_done and !g.held) g.drain();
        const natural: []const u8 = g.reason orelse switch (m.reason) {
            .stop => "stop",
            .length => "length",
            .silence => return cx.other("the text engine returned a speech-only finish reason"),
            .cancelled => "cancelled",
            .failed => return cx.other(if (m.message.len > 0) try a.dupe(u8, m.message) else "the reply failed"),
        };
        // the server ended the reply at its deadline: a late token's EOS, stop string or cap never hides that
        const ended = if (g.deadline_hit) "deadline" else natural;
        const stopped: ?u32 = if (g.loop_stop) m.stats.loop_period else null;
        const reason = if (stopped != null) @import("loop_stop.zig").finish_reason else ended;
        const finished_ns = nowNs(g.srv.io);
        const content_tokens = if (g.ignore_eos) g.collected.items else reply_text.stripTrailing(g.collected.items, g.srv.eos);
        const raw = try g.srv.text.decode(a, content_tokens);
        const visible = g.stops.visible(raw, false);
        // closed before the think split, so a call ending an unclosed think block is the answer
        const close = try g.closeCall(visible);
        const text = if (close.len > 0) try std.mem.concat(a, u8, &.{ visible, close }) else visible;
        var content: []const u8 = text;
        var reasoning: ?[]const u8 = null;
        if (g.thinking) {
            const split = try reply_text.splitThinkingTokens(a, reply_text.decodeText, g.srv.text, text, true, g.srv.markers, content_tokens, g.srv.close_id);
            const r = reply_text.pyStrip(split.reasoning);
            reasoning = if (r.len > 0) r else null;
            content = split.answer;
        } else {
            const h = reply_text.parseHarmony(text);
            content = h.content;
            reasoning = h.reasoning;
        }
        content = try tool_lfm2.dsml(a, content, g.tools, true);
        const owned = try g.ownedOf(content_tokens, raw, content);
        if (g.sink) |sink| {
            const thought = g.streamed_reasoning.text.items;
            if (reasoning) |r| if (std.mem.startsWith(u8, r, thought) and r.len > thought.len)
                try g.emit(sink, try deltaOf(a, "reasoning_content", r[thought.len..]));
            const shown = if (g.calls != null) try tool_stream.endProse(a, content, if (g.thinking) content else text, g.tools, reason, owned) else content;
            const sent = g.streamed.text.items;
            const rest = if (std.mem.startsWith(u8, shown, sent) and shown.len > sent.len) shown[sent.len..] else "";
            if (g.lp) |*l| (if (try l.delta(g.srv, a, g.thinking, g.ignore_eos, g.collected.items, rest, shown.len, true)) |d| try g.emit(sink, d)) else if (rest.len > 0) try g.emit(sink, .{ .string = rest });
            if (g.calls) |*c| if (close.len > 0 or std.mem.eql(u8, reason, "length")) {
                // the streamer reads the closers as markup the model wrote, so the call ends with the same deltas
                var out: std.ArrayList(Value) = .empty;
                try c.finishCut(if (g.thinking) content else text, &out); // a cut value sends its held bytes
                for (out.items) |d| try g.emit(sink, d);
            };
        }
        const prefilled = m.prefilled_ns;
        const decode_s: f64 = if (prefilled) |p| @max(0, seconds(finished_ns - p)) else 0;
        const decode_tokens = g.collected.items.len -| 1;
        const runtime = try json.newObject(a);
        try runtime.put(a, "enable_thinking", .{ .bool = thinking });
        try runtime.put(a, "reasoning_effort", if (!thinking) .{ .string = "none" } else if (effort) |e| .{ .string = e } else .null);
        try runtime.put(a, "engine", .{ .string = g.srv.info.name });
        try runtime.put(a, "tokens_per_second", .{ .float = if (decode_s > 0) @as(f64, @floatFromInt(decode_tokens)) / decode_s else 0 });
        try runtime.put(a, "seconds", .{ .float = @max(0, decode_s) }); // the reply's decode seconds, as tokens_per_second divides them
        if (m.stats.gpu_seconds) |gpu| try runtime.put(a, "gpu_seconds", .{ .float = gpu }); // TF_Q27_PROFILE only
        try runtime.put(a, "prefill_seconds", if (m.stats.prefill_seconds) |p| .{ .float = p } else .null);
        const widths = try a.alloc(Value, m.widths.len);
        for (m.widths, widths) |w, *slot| slot.* = try json.intValue(a, w);
        try runtime.put(a, "prefill_widths", .{ .array = widths });
        const raised = try a.alloc(Value, m.raised.len);
        for (m.raised, raised) |r, *slot| slot.* = .{ .bool = r };
        try runtime.put(a, "prefill_raised", .{ .array = raised });
        try runtime.put(a, "time_to_first_token", if (g.first_ns) |t| .{ .float = seconds(t - received) } else .null);
        try runtime.put(a, "sampling", .{ .string = if (exact) "exact" else "greedy" });
        try runtime.put(a, "drafts", .{ .bool = drafts });
        const sha = @import("tokens.zig").tokenSha(g.collected.items);
        try runtime.put(a, "token_sha", .{ .string = try a.dupe(u8, &sha) });
        const sha256 = @import("tokens.zig").tokenSha256(g.collected.items);
        try runtime.put(a, "token_sha256", .{ .string = try a.dupe(u8, &sha256) });
        // "return_token_ids": the reply's tokens, the same slice the shas above hash, so ids and sha cannot disagree
        if (g.token_ids) {
            const reply_ids = try a.alloc(Value, g.collected.items.len);
            for (g.collected.items, reply_ids) |t, *slot| slot.* = try json.intValue(a, t);
            try runtime.put(a, "token_ids", .{ .array = reply_ids });
        }
        try runtime.put(a, "min_rows", try json.intValue(a, m.stats.min_rows));
        if (m.stats.loop_period) |period| {
            const loop_field = try json.newObject(a);
            try loop_field.put(a, "period", try json.intValue(a, period));
            if (stopped != null) try loop_field.put(a, "stopped", .{ .bool = true });
            try runtime.put(a, "loop", .{ .object = loop_field });
        }
        const cells = try a.dupe(u8, m.runner);
        var runner_raw: ?[2][]const u8 = null; // abstain anchors index these; openai.zig maps them to what it sends
        if (cells.len > 0) switch (try json.parse(a, cells)) {
            .ok => |v| {
                if (v.get("abstains") != null) runner_raw = try projection.rawFields(a, g.srv.text, content_tokens, if (g.thinking) g.srv.think_close_end else null);
                try runtime.put(a, "runner", v);
            },
            .err => {},
        };
        if (g.guide) |given| {
            const guide = try json.newObject(a);
            var digest: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(given, &digest, .{});
            try guide.put(a, "sha256", .{ .string = try a.dupe(u8, &std.fmt.bytesToHex(digest, .lower)) });
            try guide.put(a, "bytes", try json.intValue(a, given.len));
            try runtime.put(a, "runner_guide", .{ .object = guide });
        }
        const s = m.stats;
        const spec = try json.newObject(a);
        try spec.put(a, "rounds", try json.intValue(a, s.rounds));
        try spec.put(a, "drafted", try json.intValue(a, s.drafted));
        try spec.put(a, "accepted", try json.intValue(a, s.accepted));
        try spec.put(a, "acceptance_rate", .{ .float = if (s.drafted > 0) @as(f64, @floatFromInt(s.accepted)) / @as(f64, @floatFromInt(s.drafted)) else 0 });
        try spec.put(a, "tokens_per_round", .{ .float = if (s.rounds > 0) @as(f64, @floatFromInt(g.collected.items.len)) / @as(f64, @floatFromInt(s.rounds)) else 0 });
        const counters = try a.dupe(u8, m.telemetry); // parsed strings point into it: it outlives the box
        if (counters.len > 0) switch (try json.parse(a, counters)) {
            .ok => |v| try spec.put(a, "proposer", v),
            .err => {},
        };
        const think_end: ?u32 = if (thinking) g.srv.text.tokenId(g.srv.markers.close) else null;
        const reply: Reply = .{
            .content = content,
            .stop_sequence = g.stops.matched(raw),
            .reasoning = reasoning,
            .tool_calls_streamed = g.calls != null and g.calls.?.streamed,
            .finish_reason = reason,
            .prompt_tokens = prompt_len,
            .cached_tokens = m.cached orelse 0,
            .completion_tokens = g.collected.items.len,
            .reasoning_tokens = reply_text.reasoningCount(g.collected.items, think_end),
            .runtime = .{ .object = runtime },
            .speculative = .{ .object = spec },
            .thinking = thinking,
            .effort = effort,
            .logprobs = if (g.lp) |l| try @import("logprobs.zig").value(g.srv, cx, a, g.thinking, content_tokens, l.rows.items) else null,
            .prompt_logprobs = if (m.prompt_rows) |r| try @import("logprobs.zig").promptValue(g.srv, a, r) else null,
            .loop_stop = stopped,
            .runner_raw = runner_raw,
            .owned = owned,
        };
        if (std.mem.eql(u8, reason, "length") and thinking and reply_text.pyStrip(content).len == 0)
            log.line("warning: a reply reached max_tokens while still thinking, so its content is empty and its text is all in reasoning_content; raise max_tokens, or send chat_template_kwargs {{\"enable_thinking\": false}} (server: --no-thinking)", .{});
        var cycle_text: [32]u8 = undefined;
        const cycle = if (s.loop_period) |period| std.fmt.bufPrint(&cycle_text, " loop=period:{d}", .{period}) catch "" else "";
        const done_ns = nowNs(g.srv.io);
        const ttft_s = if (g.first_ns) |at| @max(0, seconds(at - received)) else 0;
        const decode_log_s = if (g.first_ns) |at| @max(0, seconds(done_ns - at)) else 0;
        var seed_text: [32]u8 = undefined;
        const seed = if (g.seed) |v| std.fmt.bufPrint(&seed_text, " seed={d}", .{v}) catch unreachable else "";
        const rate = if (decode_log_s > 0) @as(f64, @floatFromInt(decode_tokens)) / decode_log_s else 0;
        log.line("done {s} prompt={d} cached={d} thinking={s} effort={s} tokens={d} sha={s} finish={s}{s} rounds={d} accepted={d}/{d} ttft={d:.3} decode={d:.3} elapsed={d:.3} queued={d:.3} tok/s={d:.1}{s}", .{ g.reply_id, prompt_len, reply.cached_tokens, if (thinking) "True" else "False", if (thinking) effort orelse "none" else "none", g.collected.items.len, sha, reason, cycle, s.rounds, s.accepted, s.drafted, ttft_s, decode_log_s, @max(0, seconds(done_ns - received)), s.queued_seconds, rate, seed });
        return reply;
    }
};

test "Server recovery: token staging allocation failure ends the loop without consuming tokens" {
    for (0..2) |fail_index| {
        var m: Mailbox = .{ .io = std.testing.io, .gpa = std.testing.allocator, .finished = true };
        defer m.deinit();
        try m.tokens.append(std.testing.allocator, 42);
        try m.chunks.append(std.testing.allocator, 1);
        try m.rows.append(std.testing.allocator, .{ .token = 42, .logprob = -1 });
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        var g: Generation = undefined;
        g.a = failing.allocator();
        g.box = &m;
        g.chunk_index = 0;
        g.consumed = 0;
        g.lp = .{};
        const Gone = struct {
            pub fn check(_: @This()) bool {
                return false;
            }
        };
        try std.testing.expectError(error.OutOfMemory, g.loop(Gone{}));
        try std.testing.expectEqual(@as(usize, 0), g.consumed);
        try std.testing.expectEqual(@as(usize, 0), g.chunk_index);
    }
}
