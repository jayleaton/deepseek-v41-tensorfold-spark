//! TF_DSV41_CALIB=measure (and `zig` without a stored Zig table): the served engine's own draft-cost table, measured
//! at boot as Python prod's calib.measure measures its engine (tensorfold-decode1 8474f31 deepseek_v41/cuda/calib.py;
//! draft/calib_measure.zig holds the plan and the arithmetic, host-tested against Python).
//!
//! - **The text.** glm5_next.spark.calib.prompt_ids: calib_measure.text in the checkpoint's tokens (tokenizer.json of
//!   TF_DSV41_GRAMMAR_TOKENIZER, else the pack dir), the ids below the vocabulary, the first 256; under 16 (or no
//!   tokenizer): fallback_ids. Rank 0 only: the followers get every id on the plan link with the operations.
//! - **The windows.** `n` slots (calib_measure.Plan: 1 on a one-slot engine; 4 on prod's 4-slot shape, the widest
//!   bucket's 64 rows over 16-row slots), slot i's text the prompt less its last 7 x i tokens. Each slot prefills its
//!   text but the last token through the served target (GpuTarget.prefill, the session store detached on rank 0, so no
//!   entry is saved or resumed; the served verify tail's pending window is kept), decodes its greedy continuation with
//!   1-row windows (keep 1 each), and is released and prefilled again. Every timed window is then a verify window at
//!   the text's last token (row 0) over the model's own continuation, through `Target.window` (the greedy pick
//!   included, as Python's fw.window(count=1)), and is **dropped** after its end mark (GpuTarget.dropWindow: op_rows_drop
//!   in row mode, op 20 on one slot): nothing is committed, so every timed run sees the same slot state (position,
//!   Engram tail, ratio-2 carries, session ids); the next window rewrites the dropped rows' positions before any read,
//!   as a rejected draft's. The DSpark pass is timed as one ask on slot 0 at its text's last token. Then every slot is
//!   released and its drafter context reset: the pool's pages are back and nothing is pending.
//! - **Both ranks in step.** The followers run every operation as rank 0 sends it (Forward.follow), so the collectives
//!   pair up as in served rounds. Around each timed run rank 0 sends op_calib_mark (64): every rank synchronizes its
//!   device (cuCtxSynchronize, Python's torch.cuda.synchronize) and reads its monotonic clock; at the end op 65
//!   all-gathers every rank's samples (end - start, ns), and rank 0 takes each entry's statistic on each rank, the
//!   slower rank's microsecond int (Python's gather_max), and builds the table (calib_measure.Plan.build).
//! - **Graphs.** The warm-ups capture each window's graph (row graphs, one-slot graphs: first use); the timed runs
//!   replay them, and the captures stay in the cache for serving.
//! - **Stored.** calib_env.store: calib-zig-<shape hash>.json in TF_DSV41_CALIB_DIR (or ~/.cache/tensorfold/
//!   dsv41-calib), Python's keys plus `"engine": "zig"`; TF_DSV41_CALIB=zig reads it at the next boot.
const std = @import("std");
const model = @import("model.zig");
const iface = @import("draft/iface.zig");
const cm = @import("draft/calib_measure.zig");
const calib_env = @import("draft/calib_env.zig");
const costs_mod = @import("draft/costs.zig");
const dspark = @import("draft/dspark.zig");
const rowtab = @import("rowtab.zig");
const m3 = @import("m3.zig");
const ph = @import("phases.zig");
const tokenizer = @import("tokenizer");
const Allocator = std.mem.Allocator;

const log = std.log.scoped(.dsv41);

pub const Measured = struct {
    costs: costs_mod.Costs,
    /// rank 0's rows 1..rows as kept (the lower of sweep and recheck), ms
    kept: [cm.max_rows]f64 = @splat(0.0),
    rows: u32 = 0,
    /// slots the windows used, timed runs
    n: u32 = 0,
    runs: usize = 0,
};

/// The calibration prompt (rank 0): glm5_next.spark.calib.prompt_ids over the checkpoint's tokenizer. gpa-owned.
pub fn promptOf(gpa: Allocator, io: std.Io, dir: []const u8, vocab: u32) ![]u32 {
    var fb: [64]u32 = undefined;
    const tok_dir = if (std.c.getenv("TF_DSV41_GRAMMAR_TOKENIZER")) |d| std.mem.span(d) else dir;
    var t = tokenizer.loadTokenizer(io, gpa, tok_dir) catch |e| {
        log.warn("calibration: no tokenizer in {s} ({t}); the fixed fallback ids", .{ tok_dir, e });
        return gpa.dupe(u32, cm.fallbackIds(vocab, &fb));
    };
    defer t.deinit();
    const ids = t.encode(gpa, cm.text) catch |e| {
        log.warn("calibration: the text did not encode ({t}); the fixed fallback ids", .{e});
        return gpa.dupe(u32, cm.fallbackIds(vocab, &fb));
    };
    defer gpa.free(ids);
    return gpa.dupe(u32, cm.promptIds(ids, vocab, &fb));
}

/// Rank 0: the measurement (calib.measure) on the loaded engine; the followers run it through the plan link.
pub fn measure(gpa: Allocator, m: *model.Model, ids: []const u32, how: cm.Method, rows_cap: u64) !Measured {
    const gt = &m.gt;
    const f = &m.f;
    const tgt = gt.target();
    const pass = m.draftPass();
    const rows: u32 = @min(cm.max_rows, if (gt.rows != null) rowtab.seg_max else gt.max_rows);
    var plan = try cm.Plan.init(gpa, m.slots, rows_cap, rows, how, pass != null);
    defer plan.deinit(gpa);
    if (ids.len < 2) return error.CalibPrompt;
    var lens: [4]usize = undefined;
    for (lens[0..plan.n], 0..) |*l, i| {
        l.* = cm.cutOf(ids.len, @intCast(i));
        if (l.* + rows + 2 > f.opts.limit) return error.CalibContext;
    }
    // rank 0's session store detached: a calibration text is neither resumed nor saved (Python prefills fw directly)
    const store = gt.sessions;
    gt.sessions = null;
    defer gt.sessions = store;
    try prefillAll(tgt, gt, pass, ids, lens[0..plan.n], false);
    // each slot's greedy continuation, 1-row windows kept one by one (calib.measure's `conts`)
    var conts: [4][cm.max_rows]u32 = undefined;
    for (lens[0..plan.n], 0..) |l, i| {
        const slot: u32 = @intCast(i);
        var pend = ids[l - 1];
        for (0..rows - 1) |j| {
            const at: u64 = l - 1 + j;
            var one: [1]u32 = undefined;
            var out = [_][]u32{&one};
            try tgt.window(&.{.{ .slot = slot, .start = at, .tokens = &.{pend}, .parents = null, .draws = &.{at + 1}, .sampling = null }}, &out);
            try tgt.keep(slot, &.{0});
            pend = one[0];
            conts[i][j] = pend;
        }
    }
    try prefillAll(tgt, gt, pass, ids, lens[0..plan.n], true);
    // the timed runs, in the plan's order; each between two marks on every rank
    const samples = try gpa.alloc(i64, plan.runs.items.len);
    defer gpa.free(samples);
    var toks: [4][cm.max_rows]u32 = undefined;
    for (lens[0..plan.n], 0..) |l, i| {
        toks[i][0] = ids[l - 1];
        @memcpy(toks[i][1..rows], conts[i][0 .. rows - 1]);
    }
    const shape = m3.shapeOf(f);
    if (shape.block > cm.max_rows) return error.CalibBlock;
    var drafts: [cm.max_rows]u32 = undefined;
    var conf: [cm.max_rows]f32 = undefined;
    var chosen: [4][cm.max_rows]u32 = undefined;
    for (plan.runs.items, samples) |run, *x| {
        if (run.what == .draft) {
            const p0: u64 = lens[0] - 1;
            const asks = [_]iface.Ask{.{ .slot = 0, .anchor = ids[lens[0] - 1], .start = p0, .params = dspark.Params.of(null, p0) }};
            var props = [_]iface.Proposal{.{ .drafts = drafts[0..shape.block], .conf = conf[0..shape.block] }};
            const t0 = try f.calibMark();
            try pass.?.propose(&asks, &props);
            x.* = @intCast((try f.calibMark()) - t0);
            continue;
        }
        var segs: [4]iface.Segment = undefined;
        var outs: [4][]u32 = undefined;
        const draws: [cm.max_rows]u64 = @splat(0);
        for (run.segments(), segs[0..run.nsegs], outs[0..run.nsegs]) |s, *sg, *o| {
            if (s.slot >= plan.n) return error.CalibPlan;
            sg.* = .{ .slot = s.slot, .start = lens[s.slot] - 1, .tokens = toks[s.slot][0..s.rows], .parents = null, .draws = draws[0..s.rows], .sampling = null };
            o.* = chosen[s.slot][0..s.rows];
        }
        const t0 = try f.calibMark();
        try tgt.window(segs[0..run.nsegs], outs[0..run.nsegs]);
        x.* = @intCast((try f.calibMark()) - t0);
        for (run.segments()) |s| try gt.dropWindow(s.slot);
    }
    for (0..plan.n) |i| {
        const slot: u32 = @intCast(i);
        tgt.release(slot);
        if (pass) |p| p.reset(slot);
    }
    // every rank's samples; each entry the slower rank's (Python's gather_max)
    const w: usize = f.comm.world();
    const all = try gpa.alloc(i64, samples.len * w);
    defer gpa.free(all);
    try f.calibGather(samples, all);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const ranks = try a.alloc([]const f64, w);
    for (ranks, 0..) |*r, k| {
        const ms = try a.alloc(f64, samples.len);
        for (ms, all[k * samples.len ..][0..samples.len]) |*y, ns| {
            if (ns < 0) return error.CalibFollower; // a follower's marks did not pair up with ours
            y.* = @as(f64, @floatFromInt(ns)) / 1e6;
        }
        const v = try a.alloc(f64, plan.entries());
        try plan.values(ms, v);
        r.* = v;
    }
    var out: Measured = .{ .costs = try plan.build(gpa, ranks), .rows = rows, .n = plan.n, .runs = plan.runs.items.len };
    @memcpy(out.kept[0..rows], ranks[0][1 .. rows + 1]);
    return out;
}

/// Every text slot released (the second time) and prefilled again: its drafter context reset, the text but its last
/// token through the served prefill, a pending verify tail kept (the slot at the last token, nothing pending).
fn prefillAll(tgt: iface.Target, gt: anytype, pass: ?iface.Pass, ids: []const u32, lens: []const usize, again: bool) !void {
    for (lens, 0..) |l, i| {
        const slot: u32 = @intCast(i);
        if (again) tgt.release(slot);
        if (pass) |p| p.reset(slot);
        _ = try tgt.prefill(slot, ids[0 .. l - 1], null, l - 1);
        if (gt.hasPending(slot)) try tgt.keep(slot, &.{0});
    }
}

/// Served.open's TF_DSV41_CALIB=measure / zig: the method (TF_DSV41_CALIB_*), the prompt, the measurement, the
/// stored entry and the boot lines. Returns the table and the stored file (gpa-owned; null: not stored).
pub fn boot(gpa: Allocator, io: std.Io, m: *model.Model, dir: []const u8, k: calib_env.Knobs, want: calib_env.Shape) !struct { costs: costs_mod.Costs, path: ?[]u8 } {
    const how = try cm.method(&calib_env.env);
    const rows_cap = (try calib_env.depthSettings(&calib_env.env)).max_rows;
    const ids = try promptOf(gpa, io, dir, @intCast(m.f.cfg.vocab));
    defer gpa.free(ids);
    const t0 = ph.nowNs();
    var got = try measure(gpa, m, ids, how, rows_cap);
    errdefer got.costs.deinit(gpa);
    const secs = @as(f64, @floatFromInt(ph.nowNs() - t0)) / 1e9;
    log.info("dsv41 calib method (zig): table {t}, {t} of {d} after {d} warm-up(s), rows 1-{d} cycled, rows 1-{d} rechecked, rows 9+ {d} runs; {d} slots, {d} timed runs, a {d}-token text; rows 1-4 kept {d:.3} / {d:.3} / {d:.3} / {d:.3} ms (rank 0)", .{ how.shape, how.stat, how.reps, how.warm, how.cycle, how.recheck, how.deep, got.n, got.runs, ids.len, got.kept[0], got.kept[1], got.kept[2], got.kept[3] });
    var path: ?[]u8 = null;
    var db: [std.fs.max_path_bytes]u8 = undefined;
    if (calib_env.dirOf(k, &db)) |d| {
        path = calib_env.store(gpa, io, d, want, got.costs, .{ .method = how, .seconds = secs, .kept = got.kept[0..got.rows], .tokens = ids.len }, std.Io.Clock.real.now(io).toSeconds()) catch |e| blk: {
            log.warn("calibration: the table was not stored in {s} ({t})", .{ d, e });
            break :blk null;
        };
    }
    var line: [512]u8 = undefined;
    var wr = std.Io.Writer.fixed(&line);
    var first = true;
    for ([_]u32{ 1, 2, 4, 6, 8, 16, 24, 32, 48, 64 }) |r| if (r <= got.costs.verify.len) {
        wr.print("{s}{d}", .{ if (first) "" else "/", r }) catch {};
        first = false;
    };
    wr.writeAll(" rows ") catch {};
    first = true;
    for ([_]u32{ 1, 2, 4, 6, 8, 16, 24, 32, 48, 64 }) |r| if (r <= got.costs.verify.len) {
        wr.print("{s}{d:.1}", .{ if (first) "" else ", ", got.costs.windowMs(r) }) catch {};
        first = false;
    };
    log.info("dsv41 calibration (zig): draft {d:.2} ms, slot {d:.2} ms, verify {s} ms, measured in {d:.1} s{s}{s}{s}", .{ got.costs.draft, got.costs.slot, wr.buffered(), secs, if (path != null) " (stored " else " (not stored", path orelse "", ")" });
    return .{ .costs = got.costs, .path = path };
}
