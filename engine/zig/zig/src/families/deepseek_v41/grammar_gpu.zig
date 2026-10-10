//! Grammar-constrained decoding on the GPU forward (TF_DSV41_GRAMMAR=1; Python prod's ``structured.py`` over GLM
//! 0610's ``grammar.py``). Every rank holds the same `grammar.Grammars` and, per slot, the same `Constraint`: rank 0
//! sends what to compile and which tokens to follow on the plan link (`Forward.op_grammar`), so the followers' matchers
//! walk the same tokens and fill the same masks for their vocabulary half.
//!
//! Where the masks apply (as in Python): a window's drafts the grammar rejects are cut before the forward (only R
//! changes; rows are row-invariant); each constrained row's disallowed columns are set to -inf in "w.logits" before any
//! pick reads them (`Forward.greedy`, the keyed sampler), on every rank, by a PTX kernel (core/grammar/mask_kernel.zig);
//! the prefill's last row (the reply's first token) likewise. The fills run on `io.concurrent` tasks started before
//! the forward and are awaited at the pick; nothing is filled for a request without a grammar.
//!
//! - `Masks` (every rank): the constraints, the staged rows, the device copy and the kernel; the follower side of
//!   op_grammar; `Forward.grammar`'s hook.
//! - `GrammarTarget` (rank 0): the `iface.Target` decorator that binds a slot's grammar at prefill, follows the
//!   committed tokens, cuts drafts and stages the masks.
//! - `GrammarEngine` (rank 0): the `engine_api.Engine` decorator that compiles a request's structure on the HTTP thread
//!   (a grammar that cannot be enforced fails at once with xgrammar's message) and hands it to the target by prompt.

const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const grammar = @import("grammar");
const fwd = @import("forward.zig");
const iface = @import("draft/iface.zig");
const lanes = @import("lanes");
const Allocator = std.mem.Allocator;

/// TF_DSV41_GRAMMAR: 1 serves structured output; unset or 0 (the default here) refuses it as before. Every rank must
/// agree (the followers compile what rank 0 sends).
pub fn enabled() !bool {
    const v = std.c.getenv("TF_DSV41_GRAMMAR") orelse return false;
    const s = std.mem.span(v);
    if (s.len == 0 or std.mem.eql(u8, s, "0")) return false;
    if (std.mem.eql(u8, s, "1")) return true;
    return error.BadGrammarKnob;
}

/// DeepSeek-V4.1's tool-call markup token, kept in the tools view (``structured.DSML_TOKENS``).
pub const tool_tokens = [_][]const u8{"｜DSML｜"};

const sub_bind: i64 = 0;
const sub_step: i64 = 1;
const sub_release: i64 = 2;

/// One window's (or the prefill's last row's) masks waiting for the next pick.
const Stage = struct {
    slot: u32,
    /// the first "w.logits" row of the window, or null: the last row the next pick reads (a prefill's)
    row0: ?u32,
    tokens: []u32,
    cut: grammar.Cut,
    /// its rows' bits in `Masks.host`, from this row
    at: u32,
    fill: ?std.Io.Future(anyerror!void) = null,
};

pub const Masks = struct {
    gpa: Allocator,
    io: std.Io,
    f: *fwd.Forward,
    g: *grammar.Grammars,
    cons: []?grammar.Constraint,
    kinds: []grammar.Kind,
    staged: std.ArrayList(Stage) = .empty,
    /// rows staged so far (the host bits' rows in use)
    rows: u32 = 0,
    max_rows: u32,
    /// pinned: bits [max_rows][words] then the rows' "w.logits" indices [max_rows]
    host: cuda.HostBuffer,
    dev: cuda.DeviceBuffer,
    module: cuda.Module,
    kernel: cuda.Function,
    /// this rank's columns: [lo, lo + width) of the vocabulary
    lo: u32,
    width: u32,
    /// the last apply's failure (a row that allows nothing, a rejected token): rank 0's target fails the requests
    failed: ?anyerror = null,
    scratch: std.ArrayList(u8) = .empty,
    words_buf: std.ArrayList(i64) = .empty,

    /// Every rank, at load: the checkpoint's compilers (tokenizer.json of `dir`), the kernel, `max_rows` rows a pick.
    pub fn init(gpa: Allocator, io: std.Io, f: *fwd.Forward, dir: []const u8, slots: u32, max_rows: u32) !*Masks {
        const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const path = try std.fs.path.join(a, &.{ dir, "tokenizer.json" });
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30));
        const g = try grammar.Grammars.init(gpa, text, .{ .vocab_size = f.cfg.vocab, .stops = &.{f.cfg.eos}, .tool_tokens = &tool_tokens, .tag_model = .deepseek_v4_1 });
        errdefer g.deinit();
        const W = f.comm.world();
        const width: u32 = @intCast(f.cfg.vocab / W);
        const lo: u32 = @intCast(f.comm.rank() * width);
        if (lo % 32 != 0 or width % 32 != 0) return error.GrammarHalves; // Python: "vocabulary halves on 32-token words"
        const d = f.runner.d;
        const bytes = 4 * @as(usize, max_rows) * (g.words + 1);
        var host = try cuda.HostBuffer.alloc(d, bytes);
        errdefer host.free();
        var dev = try cuda.DeviceBuffer.alloc(d, bytes);
        errdefer dev.free();
        var log: [4096]u8 = undefined;
        var module = cuda.Module.loadPtx(d, @embedFile("grammar_mask_ptx"), &log) catch |e| {
            std.log.scoped(.dsv41).err("grammar: the mask kernel's PTX did not load: {s}", .{std.mem.sliceTo(&log, 0)});
            return e;
        };
        errdefer module.unload();
        const x = try gpa.create(Masks);
        x.* = .{
            .gpa = gpa,
            .io = io,
            .f = f,
            .g = g,
            .cons = try gpa.alloc(?grammar.Constraint, slots),
            .kinds = try gpa.alloc(grammar.Kind, slots),
            .max_rows = max_rows,
            .host = host,
            .dev = dev,
            .module = module,
            .kernel = try module.function("tf_grammar_mask"),
            .lo = lo,
            .width = width,
        };
        @memset(x.cons, null);
        f.grammar = .{ .ptr = x, .follow = follow, .apply = applyFn };
        if (f.comm.rank() == 0) {
            const ms = @divTrunc(std.Io.Clock.awake.now(io).toNanoseconds() - t0, std.time.ns_per_ms);
            std.log.scoped(.dsv41).info("structured output: on (xgrammar 0.2.8 core, {d} columns, DSML tool tag deepseek_v4_1; built in {d} ms)", .{ f.cfg.vocab, ms });
        }
        return x;
    }

    pub fn deinit(x: *Masks) void {
        x.f.grammar = null;
        x.settle();
        for (x.cons) |*c| if (c.*) |*k| k.deinit();
        x.gpa.free(x.cons);
        x.gpa.free(x.kinds);
        x.staged.deinit(x.gpa);
        x.scratch.deinit(x.gpa);
        x.words_buf.deinit(x.gpa);
        x.module.unload();
        x.dev.free();
        x.host.free();
        x.g.deinit();
        x.gpa.destroy(x);
    }

    fn bits(x: *Masks) []u32 {
        return @alignCast(std.mem.bytesAsSlice(u32, x.host.bytes[0 .. 4 * @as(usize, x.max_rows) * x.g.words]));
    }

    fn indices(x: *Masks) []u32 {
        const off = 4 * @as(usize, x.max_rows) * x.g.words;
        return @alignCast(std.mem.bytesAsSlice(u32, x.host.bytes[off..][0 .. 4 * @as(usize, x.max_rows)]));
    }

    // -- the slots' grammars (every rank alike) --------------------------------------------------------------------

    fn bindLocal(x: *Masks, slot: u32, kind: grammar.Kind, compiled: grammar.engine.Compiled, active: bool) !void {
        x.releaseLocal(slot);
        x.cons[slot] = try x.g.bound(kind, compiled, active);
        x.kinds[slot] = kind;
    }

    fn releaseLocal(x: *Masks, slot: u32) void {
        if (x.cons[slot]) |*c| c.deinit();
        x.cons[slot] = null;
    }

    pub fn bound(x: *const Masks, slot: u32) bool {
        return slot < x.cons.len and x.cons[slot] != null;
    }

    /// Rank 0: `slot`'s reply follows `spec` (compiled here as `compiled`); the followers compile it themselves.
    pub fn bind(x: *Masks, slot: u32, spec: grammar.Spec, compiled: grammar.engine.Compiled, active: bool) !void {
        if (leads(x.f)) {
            try x.words_buf.resize(x.gpa, 3 + grammar.pack.len(spec.text.len));
            const w = x.words_buf.items;
            w[0] = fwd.Forward.op_grammar;
            w[1] = sub_bind;
            w[2] = slot;
            grammar.pack.encode(spec, active, w[3..]);
            try x.f.link.?.send(w);
        }
        try x.bindLocal(slot, spec.kind, compiled, active);
    }

    /// Rank 0: `slot` has no grammar any more.
    pub fn release(x: *Masks, slot: u32) !void {
        if (!x.bound(slot)) return;
        if (leads(x.f)) try x.f.link.?.send(&.{ fwd.Forward.op_grammar, sub_release, slot });
        x.releaseLocal(slot);
    }

    /// What rank 0 asks of one slot before a forward: follow `advance`, then constrain a window of `tokens` at "w.logits"
    /// rows from `row0` (null: the prefill's last row, `tokens` its one token).
    pub const Ask = struct { slot: u32, row0: ?u32, advance: []const u32, tokens: []const u32 };

    /// Rank 0, before a forward: each ask's slot follows its `advance`, then its window is cut (rows [0, keep) run;
    /// Python's drafts the grammar rejects are dropped). A grammar failure (a rejected chosen token) is that slot's,
    /// in `bad`. Nothing is sent or staged yet: `commit` does that once the rows' places are known.
    pub fn prepare(x: *Masks, asks: []const Ask, cuts: []grammar.Cut, bad: []?anyerror) void {
        x.settle(); // a window that failed before its pick: its fills end before the matchers move
        for (asks, cuts, bad) |q, *k, *b| {
            b.* = null;
            k.* = .{ .keep = @intCast(q.tokens.len), .first = @intCast(q.tokens.len) };
            const c = &(x.cons[q.slot] orelse continue);
            c.advance(q.advance) catch |e| {
                b.* = e;
                continue;
            };
            k.* = c.cut(q.tokens) catch |e| {
                b.* = e;
                k.* = .{ .keep = @intCast(q.tokens.len), .first = @intCast(q.tokens.len) };
                continue;
            };
        }
    }

    /// Rank 0: the prepared asks (now with their rows) to the followers, and every ask's masks staged and filling.
    pub fn commit(x: *Masks, asks: []const Ask, cuts: []const grammar.Cut, bad: []const ?anyerror) !void {
        if (leads(x.f)) {
            // [op, step, n, per ask: slot, row0 + 1 (0: the last row), n_adv, adv..., keep, first, n, tokens[0..keep]]
            var n: usize = 3;
            for (asks, cuts) |q, k| n += 6 + q.advance.len + k.keep;
            try x.words_buf.resize(x.gpa, n);
            const w = x.words_buf.items;
            w[0] = fwd.Forward.op_grammar;
            w[1] = sub_step;
            w[2] = @intCast(asks.len);
            var at: usize = 3;
            for (asks, cuts, bad) |q, k, b| {
                w[at] = q.slot;
                w[at + 1] = if (q.row0) |r| r + 1 else 0;
                w[at + 2] = @intCast(q.advance.len);
                at += 3;
                for (q.advance) |t| {
                    w[at] = t;
                    at += 1;
                }
                // a failed ask stages nothing on any rank (first = keep: no rows)
                w[at] = k.keep;
                w[at + 1] = if (b != null) k.keep else k.first;
                w[at + 2] = @intCast(k.keep);
                at += 3;
                for (q.tokens[0..k.keep]) |t| {
                    w[at] = t;
                    at += 1;
                }
            }
            try x.f.link.?.send(w[0..at]);
        }
        for (asks, cuts, bad) |q, k, b| {
            if (b != null or x.cons[q.slot] == null) continue;
            try x.push(q.slot, q.row0, q.tokens[0..k.keep], k);
        }
    }

    /// Stages one window's masks (every rank): its rows' bits filled on a concurrent task.
    fn push(x: *Masks, slot: u32, row0: ?u32, tokens: []const u32, k: grammar.Cut) !void {
        if (k.rows() == 0) return;
        if (x.rows + k.rows() > x.max_rows) return error.GrammarRows;
        const own = try x.gpa.dupe(u32, tokens);
        try x.staged.append(x.gpa, .{ .slot = slot, .row0 = row0, .tokens = own, .cut = k, .at = x.rows });
        x.rows += k.rows();
        const s = &x.staged.items[x.staged.items.len - 1];
        s.fill = x.io.concurrent(fillOne, .{ x, slot, own, k, s.at }) catch null; // no concurrency: filled at the pick
    }

    fn fillOne(x: *Masks, slot: u32, tokens: []const u32, k: grammar.Cut, at: u32) anyerror!void {
        const c = &(x.cons[slot] orelse return error.NoGrammar);
        const words = x.g.words;
        return c.fill(tokens, k, x.bits()[@as(usize, at) * words ..][0 .. @as(usize, k.rows()) * words]);
    }

    /// Awaits the staged fills and drops them (a window that never reached a pick).
    fn settle(x: *Masks) void {
        for (x.staged.items) |*s| {
            if (s.fill) |*fu| _ = fu.await(x.io) catch {};
            x.gpa.free(s.tokens);
        }
        x.staged.clearRetainingCapacity();
        x.rows = 0;
    }

    /// Before a pick reads "w.logits" rows [row0, row0 + n) (every rank, in the same order): the staged rows'
    /// disallowed columns -inf. A failed fill masks nothing and is kept in `failed` for rank 0's target.
    pub fn apply(x: *Masks, row0: u32, n: u32) !void {
        if (x.staged.items.len == 0) return;
        defer x.settle();
        const idx = x.indices();
        var count: u32 = 0;
        for (x.staged.items) |*s| {
            const res = if (s.fill) |*fu| fu.await(x.io) else fillOne(x, s.slot, s.tokens, s.cut, s.at);
            s.fill = null;
            res catch |e| {
                x.failed = e;
                return;
            };
            const base = s.row0 orelse (row0 + n - s.cut.keep); // the prefill's row: the pick's last
            for (0..s.cut.rows()) |j| idx[count + j] = base + s.cut.first + @as(u32, @intCast(j));
            count += s.cut.rows();
        }
        if (count == 0) return;
        const words = x.g.words;
        const r = x.f.runner;
        const bits_bytes = 4 * @as(usize, count) * words;
        const idx_off = 4 * @as(usize, x.max_rows) * words;
        try x.dev.uploadAsync(0, x.host.bytes[0..bits_bytes], r.stream.handle);
        try x.dev.uploadAsync(idx_off, x.host.bytes[idx_off..][0 .. 4 * @as(usize, count)], r.stream.handle);
        const logits = r.addressOf("w.logits") orelse return error.Unbound;
        const sp = grammar.rows.span(x.lo, x.width);
        var args: cuda.launch.Args = .{};
        args.add(@as(u64, logits));
        args.add(@as(u64, x.width));
        args.add(@as(u64, x.dev.ptr + idx_off));
        args.add(@as(u64, x.dev.ptr));
        args.add(@as(u32, words));
        args.add(@as(u32, sp.w0));
        args.add(@as(u32, sp.n));
        args.add(@as(u32, x.width));
        try cuda.launch.launch(x.kernel, .{ .grid = .{ .x = (sp.n + 127) / 128, .y = count, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, r.stream, &args);
    }

    fn applyFn(ptr: *anyopaque, row0: u32, n: u32) anyerror!void {
        const x: *Masks = @ptrCast(@alignCast(ptr));
        return x.apply(row0, n);
    }

    /// A follower: rank 0's op_grammar message.
    fn follow(ptr: *anyopaque, msg: []const i64) anyerror!void {
        const x: *Masks = @ptrCast(@alignCast(ptr));
        if (msg.len < 3) return error.BadPlan;
        switch (msg[1]) {
            sub_bind => {
                const slot: u32 = @intCast(msg[2]);
                if (slot >= x.cons.len) return error.BadPlan;
                const d = try grammar.pack.decode(x.gpa, msg[3..], &x.scratch);
                var why: grammar.Grammars.Why = .{};
                const compiled = x.g.compile(d.spec, &why) catch |e| {
                    // rank 0 compiled it; a follower that cannot is a broken install: its masks would differ
                    std.log.scoped(.dsv41).err("grammar: rank {d} cannot compile rank 0's grammar ({s}): {s}", .{ x.f.comm.rank(), @errorName(e), why.text() });
                    return e;
                };
                defer compiled.deinit();
                try x.bindLocal(slot, d.spec.kind, compiled, d.active);
            },
            sub_release => {
                const slot: u32 = @intCast(msg[2]);
                if (slot >= x.cons.len) return error.BadPlan;
                x.releaseLocal(slot);
            },
            sub_step => {
                x.settle();
                const n: usize = @intCast(msg[2]);
                var at: usize = 3;
                for (0..n) |_| {
                    if (at + 3 > msg.len) return error.BadPlan;
                    const slot: u32 = @intCast(msg[at]);
                    if (slot >= x.cons.len) return error.BadPlan;
                    const row0: ?u32 = if (msg[at + 1] == 0) null else @intCast(msg[at + 1] - 1);
                    const n_adv: usize = @intCast(msg[at + 2]);
                    at += 3;
                    if (at + n_adv + 3 > msg.len) return error.BadPlan;
                    const adv = try x.gpa.alloc(u32, n_adv);
                    defer x.gpa.free(adv);
                    for (adv, msg[at .. at + n_adv]) |*t, v| t.* = @intCast(v);
                    at += n_adv;
                    const k: grammar.Cut = .{ .keep = @intCast(msg[at]), .first = @intCast(msg[at + 1]) };
                    const len: usize = @intCast(msg[at + 2]);
                    at += 3;
                    if (at + len > msg.len) return error.BadPlan;
                    const toks = try x.gpa.alloc(u32, len);
                    defer x.gpa.free(toks);
                    for (toks, msg[at .. at + len]) |*t, v| t.* = @intCast(v);
                    at += len;
                    const c = &(x.cons[slot] orelse continue);
                    // rank 0 followed the same tokens; a failure here is rank 0's too (its slot fails there)
                    c.advance(adv) catch continue;
                    x.push(slot, row0, toks, k) catch |e| if (e != error.GrammarRows) return e;
                }
            },
            else => return error.BadPlan,
        }
    }
};

// -- rank 0: the lanes target ---------------------------------------------------------------------------------------

/// A slot's reply as the decorator follows it.
const SlotState = struct {
    /// the last window's tokens as run (after the cut)
    last: std.ArrayList(u32) = .empty,
    /// rows of `last` kept by `keep` (null: the window was never kept, so every row was)
    kept: ?u32 = null,
    windows: u32 = 0,
};

/// The value a cut row's choice gets: no draft equals it, so the lanes never accept past the cut.
pub const cut_choice: u32 = std.math.maxInt(u32);

pub const GrammarTarget = struct {
    gpa: Allocator,
    inner: iface.Target,
    masks: *Masks,
    requests: *GrammarEngine,
    slots: []SlotState,
    failed: []?anyerror,
    arena: std.heap.ArenaAllocator,

    pub fn init(gpa: Allocator, inner: iface.Target, masks: *Masks, requests: *GrammarEngine) !GrammarTarget {
        const slots = try gpa.alloc(SlotState, masks.cons.len);
        @memset(slots, .{});
        const failed = try gpa.alloc(?anyerror, masks.cons.len);
        @memset(failed, null);
        return .{ .gpa = gpa, .inner = inner, .masks = masks, .requests = requests, .slots = slots, .failed = failed, .arena = .init(gpa) };
    }

    pub fn deinit(t: *GrammarTarget) void {
        for (t.slots) |*s| s.last.deinit(t.gpa);
        t.gpa.free(t.slots);
        t.gpa.free(t.failed);
        t.arena.deinit();
    }

    pub fn target(t: *GrammarTarget) iface.Target {
        return .{ .ptr = t, .vtable = &vtable };
    }

    const vtable: iface.Target.VTable = .{ .prefill = prefill, .window = window, .keep = keep, .taps = taps, .release = release, .admit = admit, .begin = begin, .piece = piece, .finish = finish, .tail = tail, .pieces = pieces, .replays = replays, .warm = warm };

    fn admit(p: *anyopaque, slot: u32, max_new: u64) void {
        self(p).inner.admit(slot, max_new);
    }

    /// A prompt in pieces: the slot's grammar state starts over at `begin`; the grammar binds at `finish`, before the
    /// prompt's last token's window (its mask: the grammar's start), as `prefill` binds it before the whole prompt.
    fn begin(p: *anyopaque, slot: u32, ids: []const u32) anyerror!iface.Target.Begun {
        const t = self(p);
        if (slot < t.slots.len) {
            t.slots[slot].last.clearRetainingCapacity();
            t.slots[slot].kept = null;
            t.slots[slot].windows = 0;
            t.failed[slot] = null;
        }
        return t.inner.begin(slot, ids);
    }

    fn piece(p: *anyopaque, slot: u32, ids: []const u32, start: u64, end: u64, save: bool) anyerror!void {
        return self(p).inner.piece(slot, ids, start, end, save);
    }

    fn replays(p: *anyopaque, list: []const iface.Target.Final) anyerror!void {
        return self(p).inner.runReplays(list);
    }

    fn pieces(p: *anyopaque, list: []const iface.Target.Piece) anyerror!void {
        return self(p).inner.runPieces(list);
    }

    fn finish(p: *anyopaque, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) anyerror!u32 {
        const t = self(p);
        if (slot >= t.slots.len) return t.inner.finish(slot, ids, sampling, draw);
        try t.bindPrompt(slot, ids);
        const first = try t.inner.finish(slot, ids, sampling, draw);
        if (t.masks.failed) |e| {
            t.masks.failed = null;
            return e;
        }
        return first;
    }

    /// TF_DSV41_TAIL_JOIN: a request without a grammar leaves its last token to the round; one with a grammar runs it
    /// as `finish` (the grammar's first mask on the prompt's last row).
    fn tail(p: *anyopaque, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) anyerror!?u32 {
        const t = self(p);
        if (slot >= t.slots.len) return t.inner.tail(slot, ids, sampling, draw);
        const reg = t.requests.claim(ids);
        if (reg == null or reg.?.compiled == null) {
            try t.masks.release(slot);
            return t.inner.tail(slot, ids, sampling, draw);
        }
        try t.bindClaimed(slot, ids, reg.?);
        const first = try t.inner.finish(slot, ids, sampling, draw);
        if (t.masks.failed) |e| {
            t.masks.failed = null;
            return e;
        }
        return first;
    }

    /// The request's grammar bound to the slot (none: the slot's masks released), the prompt's last row's mask staged.
    fn bindPrompt(t: *GrammarTarget, slot: u32, ids: []const u32) !void {
        const reg = t.requests.claim(ids);
        if (reg == null or reg.?.compiled == null) return t.masks.release(slot);
        return t.bindClaimed(slot, ids, reg.?);
    }

    fn bindClaimed(t: *GrammarTarget, slot: u32, ids: []const u32, r: *Registration) !void {
        const active = grammar.constraint.thinkActive(ids, t.masks.g.think_open, t.masks.g.think_end);
        try t.masks.bind(slot, r.spec, r.compiled.?, active);
        var cuts: [1]grammar.Cut = undefined;
        var bad: [1]?anyerror = undefined;
        const ask = [_]Masks.Ask{.{ .slot = slot, .row0 = null, .advance = &.{}, .tokens = ids[ids.len - 1 ..] }};
        t.masks.prepare(&ask, &cuts, &bad);
        if (bad[0]) |e| return e;
        try t.masks.commit(&ask, &cuts, &bad);
    }

    fn warm(p: *anyopaque, asks: []const iface.Warm) void {
        self(p).inner.warm(asks);
    }

    fn self(p: *anyopaque) *GrammarTarget {
        return @ptrCast(@alignCast(p));
    }

    fn prefill(p: *anyopaque, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) anyerror!u32 {
        const t = self(p);
        if (slot >= t.slots.len) return t.inner.prefill(slot, ids, sampling, draw);
        t.slots[slot].last.clearRetainingCapacity();
        t.slots[slot].kept = null;
        t.slots[slot].windows = 0;
        t.failed[slot] = null;
        const reg = t.requests.claim(ids);
        if (reg == null or reg.?.compiled == null) {
            try t.masks.release(slot);
            return t.inner.prefill(slot, ids, sampling, draw);
        }
        const r = reg.?;
        const active = grammar.constraint.thinkActive(ids, t.masks.g.think_open, t.masks.g.think_end);
        try t.masks.bind(slot, r.spec, r.compiled.?, active);
        // the reply's first token: the prompt's last row under the grammar's start (a one-row window's mask)
        var cuts: [1]grammar.Cut = undefined;
        var bad: [1]?anyerror = undefined;
        const ask = [_]Masks.Ask{.{ .slot = slot, .row0 = null, .advance = &.{}, .tokens = ids[ids.len - 1 ..] }};
        t.masks.prepare(&ask, &cuts, &bad);
        if (bad[0]) |e| return e;
        try t.masks.commit(&ask, &cuts, &bad);
        const first = try t.inner.prefill(slot, ids, sampling, draw);
        if (t.masks.failed) |e| {
            t.masks.failed = null;
            return e;
        }
        return first;
    }

    fn window(p: *anyopaque, segments: []const iface.Segment, choices: [][]u32) anyerror!void {
        const t = self(p);
        _ = t.arena.reset(.retain_capacity);
        const a = t.arena.allocator();
        var any = false;
        for (segments) |s| any = any or t.masks.bound(s.slot);
        if (!any) return t.inner.window(segments, choices);
        const segs = try a.dupe(iface.Segment, segments);
        const outs = try a.alloc([]u32, segments.len);
        const asks = try a.alloc(Masks.Ask, segments.len);
        const which = try a.alloc(usize, segments.len);
        var n_asks: usize = 0;
        for (segs, outs, segments, choices, 0..) |*g, *o, s, c, i| {
            o.* = c;
            var chain = s.tokens.len;
            if (t.masks.bound(s.slot) and s.parents != null) {
                // with a grammar the window is its main chain (Python: no tree rows beside grammar masks)
                const par = s.parents.?;
                chain = 1;
                while (chain < par.len and par[chain] == @as(i32, @intCast(chain)) - 1) chain += 1;
                g.parents = null;
                g.tokens = s.tokens[0..chain];
                g.draws = s.draws[0..chain];
            }
            if (t.masks.bound(s.slot)) {
                const st = &t.slots[s.slot];
                var adv: std.ArrayList(u32) = .empty;
                if (st.windows > 0) {
                    const kept = st.kept orelse @as(u32, @intCast(st.last.items.len));
                    try adv.appendSlice(a, st.last.items[1..kept]);
                }
                try adv.append(a, s.tokens[0]); // the pending token: the last round's choice
                asks[n_asks] = .{ .slot = s.slot, .row0 = 0, .advance = adv.items, .tokens = g.tokens };
                which[n_asks] = i;
                n_asks += 1;
            }
        }
        const cuts = try a.alloc(grammar.Cut, n_asks);
        const bad = try a.alloc(?anyerror, n_asks);
        t.masks.prepare(asks[0..n_asks], cuts, bad);
        // rows dropped by a cut (not by a stop: those rows run unconstrained, as Python's later windows would run them)
        for (asks[0..n_asks], cuts, bad, which[0..n_asks]) |*q, k, b, i| {
            if (b) |e| t.failed[q.slot] = e;
            if (b == null and !k.stop and k.keep < segs[i].tokens.len) {
                segs[i].tokens = segs[i].tokens[0..k.keep];
                segs[i].draws = segs[i].draws[0..k.keep];
            }
            q.tokens = segs[i].tokens;
        }
        // each segment's first "w.logits" row: segments back to back, in order (target.zig, batch.zig)
        var at: u32 = 0;
        var qi: usize = 0;
        for (segs, 0..) |g, i| {
            if (qi < n_asks and which[qi] == i) {
                asks[qi].row0 = at;
                qi += 1;
            }
            at += @intCast(g.tokens.len);
        }
        try t.masks.commit(asks[0..n_asks], cuts, bad);
        for (segs, outs, choices) |g, *o, c| {
            @memset(c[g.tokens.len..], cut_choice);
            o.* = c[0..g.tokens.len];
        }
        try t.inner.window(segs, outs);
        for (segs) |g| {
            if (!t.masks.bound(g.slot)) continue;
            const st = &t.slots[g.slot];
            st.last.clearRetainingCapacity();
            try st.last.appendSlice(t.gpa, g.tokens);
            st.kept = null;
            st.windows += 1;
        }
        if (t.masks.failed) |e| {
            t.masks.failed = null;
            return e;
        }
        for (segs) |g| if (g.slot < t.failed.len) if (t.failed[g.slot]) |e| {
            t.failed[g.slot] = null;
            return e;
        };
    }

    fn keep(p: *anyopaque, slot: u32, path: []const u32) anyerror!void {
        const t = self(p);
        try t.inner.keep(slot, path);
        if (slot < t.slots.len) t.slots[slot].kept = @intCast(path.len);
    }

    fn taps(p: *anyopaque, slot: u32) iface.Taps {
        return self(p).inner.taps(slot);
    }

    fn release(p: *anyopaque, slot: u32) void {
        const t = self(p);
        t.inner.release(slot);
        t.masks.release(slot) catch |e| std.log.scoped(.dsv41).err("grammar: releasing slot {d} on the other ranks failed ({t})", .{ slot, e });
    }
};

// -- rank 0: the engine -----------------------------------------------------------------------------------------------

/// A submitted request's grammar, until its reply finishes.
const Registration = struct {
    id: api.Id,
    prompt: []const u32,
    spec: grammar.Spec,
    compiled: ?grammar.engine.Compiled,
    claimed: bool = false,
    sink: api.Sink,
    owner: *GrammarEngine,
};

pub const GrammarEngine = struct {
    gpa: Allocator,
    io: std.Io,
    inner: api.Engine,
    g: *grammar.Grammars,
    mutex: std.Io.Mutex = .init,
    live: std.ArrayList(*Registration) = .empty,

    pub fn init(gpa: Allocator, io: std.Io, inner: api.Engine, g: *grammar.Grammars) GrammarEngine {
        return .{ .gpa = gpa, .io = io, .inner = inner, .g = g };
    }

    pub fn deinit(e: *GrammarEngine) void {
        for (e.live.items) |r| e.drop(r);
        e.live.deinit(e.gpa);
    }

    fn drop(e: *GrammarEngine, r: *Registration) void {
        if (r.compiled) |c| c.deinit();
        e.gpa.destroy(r);
    }

    pub fn engine(e: *GrammarEngine) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn self(ctx: *anyopaque) *GrammarEngine {
        return @ptrCast(@alignCast(ctx));
    }
    fn info(ctx: *anyopaque) api.Info {
        var i = self(ctx).inner.info();
        i.structures = true;
        return i;
    }
    fn cancel(ctx: *anyopaque, id: api.Id) void {
        self(ctx).inner.cancel(id);
    }
    fn status(ctx: *anyopaque, out: *api.Status, stream_tokens: []u32) void {
        self(ctx).inner.status(out, stream_tokens);
    }
    fn memory(ctx: *anyopaque, reset_peak: bool) ?api.Memory {
        return self(ctx).inner.memory(reset_peak);
    }

    /// The request's spec in grammar's terms (``structured.Host.spec``), or null for plain text.
    pub fn specOf(s: api.Structure) grammar.Spec {
        const kind: grammar.Kind = switch (s.kind) {
            .json => .json,
            .json_schema => .json_schema,
            .regex => .regex,
            .choice => .choice,
            .grammar => .grammar,
            .tools => .tools,
        };
        return .{ .kind = kind, .text = s.text };
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e = self(ctx);
        const r = e.gpa.create(Registration) catch return error.Busy;
        r.* = .{ .id = id, .prompt = request.prompt, .spec = .{ .kind = .json }, .compiled = null, .sink = sink, .owner = e };
        if (request.structure) |s| {
            r.spec = specOf(s);
            var why: grammar.Grammars.Why = .{};
            r.compiled = e.g.compile(r.spec, &why) catch {
                e.gpa.destroy(r);
                // Python answers HTTP 400 "<field>: the grammar cannot be enforced: <why>" before the request runs
                var buf: [1200]u8 = undefined;
                const msg = std.fmt.bufPrint(&buf, "the grammar cannot be enforced: {s}", .{why.text()}) catch "the grammar cannot be enforced";
                sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .failed, .message = msg } });
                return;
            };
        }
        {
            e.mutex.lockUncancelable(e.io);
            defer e.mutex.unlock(e.io);
            e.live.append(e.gpa, r) catch {
                e.drop(r);
                return error.Busy;
            };
        }
        e.inner.submit(id, request, .{ .ctx = r, .event = event }) catch |err| {
            e.forget(r);
            return err;
        };
    }

    fn forget(e: *GrammarEngine, r: *Registration) void {
        e.mutex.lockUncancelable(e.io);
        defer e.mutex.unlock(e.io);
        if (std.mem.indexOfScalar(*Registration, e.live.items, r)) |i| _ = e.live.orderedRemove(i);
        e.drop(r);
    }

    fn event(ctx: *anyopaque, id: api.Id, ev: *const api.Event) void {
        const r: *Registration = @ptrCast(@alignCast(ctx));
        const sink = r.sink;
        if (ev.* == .finished) r.owner.forget(r);
        sink.event(sink.ctx, id, ev);
    }

    /// The target, at a prefill: the oldest unclaimed request whose prompt is `ids` (admission is in submit order, and
    /// every request registers, grammar or not, so an equal prompt without a grammar claims its own entry).
    pub fn claim(e: *GrammarEngine, ids: []const u32) ?*Registration {
        e.mutex.lockUncancelable(e.io);
        defer e.mutex.unlock(e.io);
        for (e.live.items) |r| if (!r.claimed and std.mem.eql(u32, r.prompt, ids)) {
            r.claimed = true;
            return r;
        };
        return null;
    }
};

/// Rank 0 with followers: it sends op_grammar (Forward.leads).
fn leads(f: *const fwd.Forward) bool {
    return f.link != null and !f.quiet and f.comm.rank() == 0;
}
