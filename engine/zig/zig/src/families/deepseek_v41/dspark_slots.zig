//! DSpark drafting over several live slots on the GPU (TF_DSV41_SLOT_DRAFTS=1 with TF_DSV41_SLOTS > 1): draft/iface.zig's
//! `Pass` as Python's Drafter runs it in row mode (G8), every rank alike (the leader sends ops 40-42 first):
//!
//! - **state a slot**: every block's ring is S rings back to back (Python's `stacked`), a slot's rows at slot x ring;
//!   `valid` a slot (the first position whose ring rows are its own). The Markov / confidence heads and the weights are
//!   the one-slot pass's (dspark_gpu.zig).
//! - **ingest**: a slot's committed taps rows into its rings: the row window's rows in "w.taps", or batch.zig's stash
//!   when a later forward of the round overwrote them; the prompt's rows from the one-slot prefill.
//! - **propose**: the asks sorted by slot (Python's order), split into passes of at most `group` slots
//!   (dspark_rows.groups); each pass one launch set over its slots' rows, then the host's candidates, the ranks'
//!   gather and each slot's chain, as the one-slot pass does.
//!
//! Two default-off knobs move drafter work off the round's critical path, the same launches and bits:
//! - TF_DSV41_DRAFT_CHAIN=1 (dspark_dev.Chain): the chain on the device, Python's own `_chain` after the round's passes.
//! - TF_DSV41_DRAFT_OVERLAP=1: a device-started round's ingests (its window's taps rows into the rings; no
//!   collective, and independent of the verify: spec.py ingests the whole window) run on their own stream from the
//!   row window's taps mark (run.zig: recorded after the window's last taps write, inside its graph), beside the
//!   window's last layer, head and pick instead of after them; the main stream joins them where it ran them before.
//!   Their scratch roles are their own (dspark_emit.sideRoles), the rings and taps the ones the serial ingest uses.

const std = @import("std");
const cuda = @import("cuda");
const calls = @import("calls.zig");
const run = @import("run.zig");
const buffers = @import("buffers.zig");
const fwd = @import("forward.zig");
const emit = @import("dspark_emit.zig");
const rows = @import("dspark_rows.zig");
const gpu = @import("dspark_gpu.zig");
const batch = @import("batch.zig");
const iface = @import("draft/iface.zig");
const dspark = @import("draft/dspark.zig");
const dsd = @import("dspark_dev.zig");
const ph = @import("phases.zig");
const rowtab = @import("rowtab.zig");

pub const Error = error{ Slots, NotAPrefix, CandidateWidth, Unbound };

pub const SlotPass = struct {
    gpa: std.mem.Allocator,
    f: *fwd.Forward,
    one: *gpu.GpuPass,
    /// live slots (the rings' count)
    nslots: u32,
    /// slots a pass at most
    group: u32,
    valid: []u64,
    arena: std.heap.ArenaAllocator,
    /// a pass's launches by its slot count - 1
    programs: [][]const calls.Call,
    h64: []i64,
    h32: []i32,
    /// the last pass's host results, [rows][W k] / [rows][D]; the host step's scratch
    cand: []i32,
    cval: []f32,
    hid: []f32,
    logits: []f32,
    hid16: []u16,
    packed_: []f32,
    all: []f32,
    ids: []i32,
    send: ?cuda.DeviceBuffer = null,
    gathered: ?cuda.DeviceBuffer = null,
    prev_ext: ?fwd.Forward.Ext = null,
    /// batch.zig's taps stash (base, bytes; 0, 0: none)
    stash: [2]u64,
    stats: struct { passes: u64 = 0, slots: u64 = 0 } = .{},
    /// every proposal's (slot, drafts) folded in the order the lanes got them: two runs drafted alike iff equal
    /// (mdraft's stats line; the GPU gate of TF_DSV41_DRAFT_OVERLAP / _CHAIN against the serial path)
    digest: u64 = 0,
    /// TF_DSV41_DRAFT_GRAPHS (dspark_dev.zig): candidates on the device, each pass size graphed; null: the host step
    dev: ?*dsd.Dev = null,
    /// the group being issued: its first row in the round (the device step's region)
    cur_at: usize = 0,
    /// the group being issued: its slot count (its program)
    cur_k: usize = 1,
    last_logits: u64 = 0,
    last_v: usize = 0,
    /// the speculative pass's asks (sorted by slot) and its groups, between `begin` and `collect`
    begun: ?Begun = null,
    begun_host: bool = false,
    order: [16]usize = undefined,
    begun_asks: [16]iface.Ask = undefined,

    /// the speculative round from the row window's device pick (spec.py's order; Pass.arm, forward.after_pick):
    /// `bat` the row windows' state (each slot's rows in the window), the armed windows by slot until the pick, then
    /// the round launched from it until `begin` takes it (rank 0); `dscr` / `hscr` the accept's inputs and outputs
    bat: ?*const batch.Batch = null,
    armed: [16]?Arm = @splat(null),
    dl: ?Dl = null,
    begun_dev: bool = false,
    dscr: ?cuda.DeviceBuffer = null,
    hscr: ?cuda.HostBuffer = null,
    staged: ?cuda.Event = null,
    prev_pick: ?fwd.Forward.AfterPick = null,
    prev_sample: ?fwd.Forward.AfterSample = null,
    dstats: struct { device_starts: u64 = 0, device_dropped: u64 = 0 } = .{},
    /// TF_DSV41_DRAFT_OVERLAP=1: the ingests' stream and its done event (the runner's taps mark is the start)
    side: ?Side = null,
    /// TF_DSV41_DRAFT_BATCH_PROJ: one main_proj over a device-started round's slots (`projRound`)
    bproj: BatchProj = .off,
    bstats: struct { rounds: u64 = 0, slots: u64 = 0, checked: u64 = 0, check_rows: u64 = 0, differ: u64 = 0 } = .{},
    /// TF_DSV41_DRAFT_CAPTURE=1: the pass keys (size << 16 | row) a regular round has issued since the graphs mode
    /// (each one's graph captured there, under the cache's agreement); a device-started round over a key not in it
    /// runs the regular path once (its speculation replays only, never captures). Every rank sees the same rounds.
    capture_first: bool = false,
    seen: [32]u32 = undefined,
    nseen: usize = 0,
    declined: u64 = 0,

    const Side = struct { stream: cuda.Stream, done: cuda.Event, mark: cuda.Event, rounds: u64 = 0, slots: u64 = 0, serial: u64 = 0 };

    const Begun = struct { n: usize, groups: [16][2]usize, ng: usize };

    const Arm = struct { start: u64, n: u32, sampled: bool, chosen: bool = false };
    const Dl = struct { k: usize, slots: [16]u32 };
    /// dscr / hscr layout (int64 elements): ids [64], picks [64] (the window's device choices: the GPU pick's, each
    /// sampled slot's over its rows), segs [16][3], acc [16][3], mem [16][16][3] (a group each)
    const scr_ids = 0;
    const scr_picks = 64;
    const scr_segs = 128;
    const scr_acc = scr_segs + 48;
    const scr_mem = scr_acc + 48;
    const scr_len = scr_mem + 16 * 48;

    /// The roles several-slot passes add to the forward's plan (before Runner.bind): the stacked rings, each pass
    /// size's statics and rows, the ingests' slot views.
    pub fn plan(a: std.mem.Allocator, f: *const fwd.Forward, p: *buffers.Plan, nslots: u32, group: u32) !void {
        const n: i64 = f.cfg.dspark_block;
        for (1..group + 1) |k| try p.add(try emit.emitPassSlots(a, f.cfg, f.widths, f.opts, n, @intCast(k), nslots));
        // the overlapped ingests' own scratch, at a row window's most rows a slot
        const ovl = try overlapFromEnv();
        if (ovl) try p.add(try emit.sideRoles(a, try emit.emitIngestAt(a, f.cfg, f.widths, f.opts, rowtab.seg_max, 0, .{ .ring_slots = nslots })));
        // the batched projection at a row window's most rows (on its own scratch too when overlapped), the check's
        // one-slot projections on theirs
        const bp = try batchProjFromEnv();
        if (bp != .off) {
            const pj = try emit.emitIngestProj(a, f.cfg, f.widths, f.opts, proj_rows_max, 0, "w.taps");
            try p.add(pj);
            if (ovl) try p.add(try emit.sideRoles(a, pj));
        }
        if (bp == .check) try p.add(try emit.renameRoles(a, try emit.emitIngestProj(a, f.cfg, f.widths, f.opts, rowtab.seg_max, 0, "w.taps"), "w.chk."));
    }

    /// On every rank after the one-slot pass and the row windows (`prev`: the forward's extension so far, slots').
    pub fn init(gpa: std.mem.Allocator, f: *fwd.Forward, one: *gpu.GpuPass, b: *const batch.Batch, nslots: u32, group: u32) !*SlotPass {
        const x = try gpa.create(SlotPass);
        errdefer gpa.destroy(x);
        const n: usize = f.cfg.dspark_block;
        const R = n * group;
        // the device round unpacks every group of the round at its slots' offset (devFinishRound: g[0] x n), so the
        // candidates / values / head hidden cover every live slot's rows (Dev's rows_max), not one group's
        const Rr = n * @max(group, nslots);
        const c: usize = f.comm.world() * one.k;
        const lay = emit.Layout.of(group, @intCast(n), f.cfg.window);
        const V: usize = f.cfg.vocab / f.comm.world();
        x.* = .{
            .gpa = gpa,
            .f = f,
            .one = one,
            .nslots = nslots,
            .group = group,
            .valid = try gpa.alloc(u64, nslots),
            .arena = std.heap.ArenaAllocator.init(gpa),
            .programs = undefined,
            .h64 = try gpa.alloc(i64, @intCast(lay.len64)),
            .h32 = try gpa.alloc(i32, @intCast(lay.len32)),
            .cand = try gpa.alloc(i32, Rr * c),
            .cval = try gpa.alloc(f32, Rr * c),
            .hid = try gpa.alloc(f32, Rr * f.cfg.hidden),
            .logits = try gpa.alloc(f32, R * V),
            .hid16 = try gpa.alloc(u16, R * f.cfg.hidden),
            .packed_ = try gpa.alloc(f32, R * 2 * one.k),
            .all = try gpa.alloc(f32, R * 2 * c),
            .ids = try gpa.alloc(i32, one.k),
            .prev_ext = f.ext,
            .stash = if (b.taps_stash) |t| .{ t.ptr, t.len } else .{ 0, 0 },
        };
        @memset(x.valid, 0);
        const a = x.arena.allocator();
        x.programs = try a.alloc([]const calls.Call, group);
        for (x.programs, 1..) |*pg, k| pg.* = try emit.emitPassSlots(a, f.cfg, f.widths, f.opts, @intCast(n), @intCast(k), nslots);
        f.ext = .{ .ptr = x, .run = followFn };
        const mode = try dsd.modeFromEnv();
        if (mode != .host) {
            const r = f.runner;
            if (r.kernels.others(r.stream).f.ds_cands == null) {
                std.log.scoped(.dsv41).warn("draft graphs: TF_DSV41_DRAFT_GRAPHS set, but the glue fatbin has no ds_cands kernel: the host step", .{});
            } else {
                _ = try x.device(&x.send, 2 * R * f.cfg.hidden); // the embedding's and exchanges' send, at its largest
                x.dev = try dsd.Dev.init(gpa, r.d, f.comm, r.stream, mode, one.k, f.cfg.hidden, n * nslots, one.conf != null, @intCast(8 * lay.len64 + 4 * lay.len32), nslots);
                // the speculative round from the row window's device pick (TF_DSV41_GREEDY_GPU, ds_accept_rows /
                // ds_stage_rows): this pass takes the forward's hook over from the one-slot pass
                const ko = r.kernels.others(r.stream).f;
                if (ko.ds_accept_rows != null and ko.ds_stage_rows != null and nslots <= 16) {
                    x.bat = b;
                    x.dscr = try cuda.DeviceBuffer.alloc(r.d, 8 * scr_len);
                    x.hscr = try cuda.HostBuffer.alloc(r.d, 8 * scr_len);
                    x.staged = try cuda.Event.init(r.d, false);
                    x.prev_pick = f.after_pick;
                    f.after_pick = .{ .ctx = x, .run = afterPick };
                    x.prev_sample = f.after_sample;
                    if (ko.vs_choose != null) f.after_sample = .{ .ctx = x, .run = afterSample };
                }
                if (try dsd.chainFromEnv()) {
                    // refused, not a silent host chain: the variant comes from the kit's fill (aot-needs lists it)
                    const ch = try dsd.Chain.init(gpa, r, n, nslots, c, f.cfg.hidden, one.conf != null);
                    errdefer ch.deinit();
                    if (try ch.probe(r, f.cfg, f.widths, f.opts)) |S| {
                        std.log.err("draft chain: TF_DSV41_DRAFT_CHAIN=1, but the AOT set has no _chain variant for {d} slot(s) (N {d}, KP {d}, candidates {d}, confidence head {}): merge a fill that holds it (tf-dsv41-m1 aot-needs lists it, tools/zig/triton_fill.py compile), or set TF_DSV41_DRAFT_CHAIN=0", .{ S, n, emit.chain_kp, c, one.conf != null });
                        return error.MissingChainVariant;
                    }
                    x.dev.?.chain = ch;
                }
                if (try overlapFromEnv()) {
                    if (x.bat == null) {
                        std.log.scoped(.dsv41).warn("draft overlap: TF_DSV41_DRAFT_OVERLAP=1 needs the device-started round (TF_DSV41_GREEDY_GPU and the glue's ds_accept_rows): the serial ingests", .{});
                    } else {
                        var st = try cuda.Stream.initPriority(r.d, .highest);
                        errdefer st.deinit();
                        x.side = .{ .stream = st, .done = try cuda.Event.init(r.d, false), .mark = try cuda.Event.init(r.d, false) };
                        r.taps_mark = .{ .ev = x.side.?.mark };
                    }
                }
                if (try captureFromEnv()) {
                    if (x.bat == null) {
                        std.log.scoped(.dsv41).warn("draft capture: TF_DSV41_DRAFT_CAPTURE=1 needs the device-started round: off", .{});
                    } else {
                        x.capture_first = true;
                        std.log.scoped(.dsv41).info("draft capture: a device-started round over a pass size not yet graphed runs the regular path once (its graph captured there)", .{});
                    }
                }
                const bp = try batchProjFromEnv();
                if (bp != .off) {
                    if (x.bat == null) {
                        std.log.scoped(.dsv41).warn("draft batched main_proj: TF_DSV41_DRAFT_BATCH_PROJ needs the device-started round: one main_proj a slot", .{});
                    } else x.bproj = bp;
                }
            }
        } else if (try overlapFromEnv() or try dsd.chainFromEnv() or try batchProjFromEnv() != .off) {
            std.log.scoped(.dsv41).warn("draft overlap / chain / batched main_proj: they need TF_DSV41_DRAFT_GRAPHS' device path: off", .{});
        }
        std.log.scoped(.dsv41).info("slot drafts: {d} slots, DSpark passes of up to {d} slots ({d} rows); chain on the {s}, ingests {s}, main_proj {t}", .{ nslots, group, R, if (x.dev != null and x.dev.?.chain != null) "device" else "host", if (x.side != null) "overlapped" else "serial", x.bproj });
        return x;
    }

    pub fn deinit(x: *SlotPass) void {
        const gpa = x.gpa;
        x.f.ext = x.prev_ext;
        if (x.bproj != .off or x.bstats.rounds > 0) std.log.scoped(.dsv41).info("draft batched main_proj: {d} rounds ({d} slots) one main_proj, {d} check rounds, {d} of {d} check rows differ", .{ x.bstats.rounds, x.bstats.slots, x.bstats.checked, x.bstats.differ, x.bstats.check_rows });
        if (x.side) |*sd| {
            std.log.scoped(.dsv41).info("draft overlap: {d} rounds' ingests ({d} slots) on their own stream, {d} serial", .{ sd.rounds, sd.slots, sd.serial });
            x.f.runner.taps_mark = null;
            sd.stream.deinit();
            sd.done.deinit();
            sd.mark.deinit();
        }
        if (x.bat != null) {
            x.f.after_pick = x.prev_pick;
            x.f.after_sample = x.prev_sample;
        }
        if (x.capture_first) std.log.scoped(.dsv41).info("draft capture: {d} device-started rounds ran the regular path to capture {d} pass key(s)", .{ x.declined, x.nseen });
        if (x.dscr) |*b| b.free();
        if (x.hscr) |*b| b.free();
        if (x.staged) |*e| e.deinit();
        if (x.dev) |d| {
            if (x.dstats.device_starts + x.dstats.device_dropped > 0) std.log.scoped(.dsv41).info("slot drafts: {d} speculative rounds started from the device pick, {d} unused", .{ x.dstats.device_starts, x.dstats.device_dropped });
            d.deinit();
        }
        if (x.send) |*b| b.free();
        if (x.gathered) |*b| b.free();
        inline for (.{ "valid", "h64", "h32", "cand", "cval", "hid", "logits", "hid16", "packed_", "all", "ids" }) |name| gpa.free(@field(x, name));
        x.arena.deinit();
        gpa.destroy(x);
    }

    pub fn pass(x: *SlotPass) iface.Pass {
        return .{ .ptr = x, .vtable = if (x.dev != null) &vtable_begin else &vtable };
    }

    const vtable: iface.Pass.VTable = .{ .ingest = ingest, .propose = proposeFold, .reset = reset, .markov = heads };
    /// with the device path: the speculative pass too (draft/spec.zig), the whole round launched by `begin`
    const vtable_begin: iface.Pass.VTable = .{ .ingest = ingest, .propose = proposeFold, .reset = reset, .markov = heads, .begin = begin, .collect = collectFold, .arm = arm };

    fn proposeFold(p: *anyopaque, asks: []const iface.Ask, out: []iface.Proposal) anyerror!void {
        try propose(p, asks, out);
        self(p).fold(asks, out);
    }

    fn collectFold(p: *anyopaque, out: []iface.Proposal) anyerror!void {
        const x = self(p);
        const n = if (x.begun) |b| b.n else 0;
        try collect(p, out);
        x.fold(x.begun_asks[0..n], out);
    }

    fn fold(x: *SlotPass, asks: []const iface.Ask, out: []const iface.Proposal) void {
        var h = std.hash.Wyhash.init(x.digest);
        for (asks, out) |a, o| {
            h.update(std.mem.asBytes(&a.slot));
            h.update(std.mem.sliceAsBytes(o.drafts));
        }
        x.digest = h.final();
    }

    /// The device path is on and past its checks.
    fn devReady(x: *const SlotPass) bool {
        const d = x.dev orelse return false;
        return d.on() and d.stats.passes >= dsd.check_passes;
    }

    /// Pass.arm (rank 0): the slot's coming row window, for the speculative round from its device choices; the
    /// followers get op 43 [op, slot, start, rows, sampled] (the one-slot pass's arm has no slot) before the window.
    fn arm(p: *anyopaque, slot: u32, start: u64, n_rows: u32, sampled: bool) void {
        const x = self(p);
        if (x.dl != null) x.dropDevice();
        if (x.bat == null or slot >= x.nslots or !x.devReady()) return;
        if (sampled and x.f.after_sample == null) return;
        x.sendOp(&.{ gpu.GpuPass.op_arm, slot, @intCast(start), n_rows, @intFromBool(sampled) }) catch |e| {
            std.log.err("slot drafts: the arm did not reach the followers ({t})", .{e});
            return;
        };
        x.armed[slot] = .{ .start = start, .n = n_rows, .sampled = sampled };
    }

    fn dropDevice(x: *SlotPass) void {
        if (x.dl != null) x.dstats.device_dropped += 1;
        x.dl = null;
        x.armed = @splat(null);
    }

    /// forward.after_pick, every rank, after a row window's GPU pick: its picks into the window's device choices, the
    /// greedy armed slots chosen; then the round starts if every armed slot of this forward has its choice.
    fn afterPick(ctx: *anyopaque, pick: fwd.Forward.PickDevice) anyerror!void {
        const x: *SlotPass = @ptrCast(@alignCast(ctx));
        const b = x.bat orelse return;
        if (pick.chain or pick.n != b.n or pick.n > 64) return;
        const r = x.f.runner;
        try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(x.dscr.?.ptr + 8 * scr_picks, pick.out, 8 * @as(usize, pick.n), r.stream.handle), "cuMemcpyDtoDAsync");
        for (&x.armed) |*am| if (am.*) |*a| if (!a.sampled) {
            a.chosen = true;
        };
        return x.tryLaunch();
    }

    /// forward.after_sample, every rank, after a sampled call's device choice (vsample.choose): a sampled armed
    /// slot whose rows these are gets them in the window's device choices; then the round starts if it can.
    fn afterSample(ctx: *anyopaque, row0: u32, n: u32, chosen: u64) anyerror!void {
        const x: *SlotPass = @ptrCast(@alignCast(ctx));
        const b = x.bat orelse return;
        for (&x.armed, 0..) |*am, sl| if (am.*) |*a| if (a.sampled) {
            const pd = b.pend[sl] orelse continue;
            if (pd.stashed or pd.a != row0 or pd.n != n or pd.a + pd.n > 64) continue;
            const r = x.f.runner;
            try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(x.dscr.?.ptr + 8 * (scr_picks + pd.a), chosen, 8 * @as(usize, n), r.stream.handle), "cuMemcpyDtoDAsync");
            a.chosen = true;
        };
        return x.tryLaunch();
    }

    /// The round from the window's device choices, every rank alike, once every armed slot whose rows are all in
    /// this forward (batch.pend) has its choice: each slot's (accepted, bonus, next position) (ds_accept_rows, copied
    /// back for `begin`), its window taps into its rings (no stream drain), then each group's statics
    /// (dspark_rows.stage with placeholders, ds_stage_rows) and pass (its graph if captured, never a capture here).
    fn tryLaunch(x: *SlotPass) !void {
        const b = x.bat orelse return;
        if (x.begun != null or !x.devReady()) return;
        var slots: [16]u32 = undefined;
        var k: usize = 0;
        for (x.armed[0..x.nslots], 0..) |am, sl| if (am) |a| {
            const pd = b.pend[sl] orelse continue;
            if (pd.stashed or pd.start != a.start or pd.n != a.n or pd.a + pd.n > b.n or a.n > x.one.ingest_rows) continue;
            if (!a.chosen) return; // a slot of this forward still waits for its choice
            slots[k] = @intCast(sl);
            k += 1;
        };
        if (k == 0) return;
        const d = x.dev.?;
        const f = x.f;
        const r = f.runner;
        const n: u32 = f.cfg.dspark_block;
        if (x.capture_first and d.active == .graphs) {
            // TF_DSV41_DRAFT_CAPTURE: a group whose pass no regular round has issued yet: the regular path this round
            var gs0: [16][2]usize = undefined;
            for (gs0[0..rows.groups(k, x.group, &gs0)]) |g| if (!x.wasSeen(passKey(g[1] - g[0], g[0] * n))) {
                x.armed = @splat(null);
                x.declined += 1;
                return;
            };
        }
        x.armed = @splat(null);
        // the accept's inputs: the armed slots' window ids at their rows, their (first row, rows, start)
        const h: []i64 = @alignCast(std.mem.bytesAsSlice(i64, x.hscr.?.bytes[0 .. 8 * scr_len]));
        for (slots[0..k], 0..) |sl, j| {
            const pd = b.pend[sl].?;
            for (pd.ids[0..pd.n], 0..) |t, i| h[scr_ids + pd.a + i] = t;
            h[scr_segs + 3 * j ..][0..3].* = .{ pd.a, pd.n, @intCast(pd.start) };
        }
        const dev = x.dscr.?.ptr;
        // ids, then segs (the picks between them are the device's, written above)
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = dev, .len = 8 * scr_picks }, 0, std.mem.sliceAsBytes(h[0..scr_picks]), r.stream.handle);
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = dev + 8 * scr_segs, .len = 8 * (scr_acc - scr_segs) }, 0, std.mem.sliceAsBytes(h[scr_segs..scr_acc]), r.stream.handle);
        const ops = r.kernels.others(r.stream);
        try ops.dsAcceptRows(dev + 8 * scr_picks, dev + 8 * scr_ids, dev + 8 * scr_segs, k, dev + 8 * scr_acc);
        try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(x.hscr.?.bytes.ptr + 8 * scr_acc, dev + 8 * scr_acc, 8 * 3 * k, r.stream.handle), "cuMemcpyDtoHAsync");
        try x.staged.?.record(r.stream);
        // each slot's whole window into its rings (spec.py: rows past the commit lie past every later pass's context):
        // overlapped from the window's taps mark when this forward recorded it, else here in the stream
        if (x.side != null and b.marked) {
            try x.ingestRound(slots[0..k], true);
        } else if (x.bproj != .off) {
            try x.ingestRound(slots[0..k], false);
            if (x.side) |*sd| sd.serial += 1;
        } else {
            for (slots[0..k]) |sl| {
                const pd = b.pend[sl].?;
                try x.ingestAsync(sl, pd.start, .{ .src = 0, .skip = pd.a }, pd.n);
            }
            if (x.side) |*sd| sd.serial += 1;
        }
        // the groups as devLaunchRound forms them (slots sorted, rows.groups), each staged on the device and launched
        var gs: [16][2]usize = undefined;
        const ng = rows.groups(k, x.group, &gs);
        const W: u32 = f.cfg.window;
        const ring: u32 = @intCast(emit.ringRows(f.cfg));
        const s64 = r.addressOf("w.ds.i64") orelse return error.Unbound;
        const s32 = r.addressOf("w.ds.i32") orelse return error.Unbound;
        for (gs[0..ng]) |g| {
            var ms: [16]rows.Member = undefined;
            const gk = g[1] - g[0];
            const mem = h[scr_mem + 48 * (g[0] % 16) ..][0 .. 3 * gk];
            for (slots[g[0]..g[1]], ms[0..gk], 0..) |sl, *m, i| {
                m.* = .{ .slot = sl, .anchor = 0, .start = b.pend[sl].?.start, .valid = x.valid[sl] };
                mem[3 * i ..][0..3].* = .{ @intCast(g[0] + i), sl, @intCast(x.valid[sl]) };
            }
            const lay = emit.Layout.of(@intCast(gk), n, W);
            const at = g[0] * n;
            const sv = try d.statics(at / n, @intCast(lay.len64), @intCast(lay.len32));
            rows.stage(lay, n, W, ring, x.one.shape.noise, ms[0..gk], sv[0], sv[1]);
            try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = s64, .len = 8 * @as(usize, @intCast(lay.len64)) }, 0, std.mem.sliceAsBytes(sv[0]), r.stream.handle);
            try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = s32, .len = 4 * @as(usize, @intCast(lay.end32)) }, 0, std.mem.sliceAsBytes(sv[1][0..@intCast(lay.end32)]), r.stream.handle);
            const moff = 8 * (scr_mem + 48 * (g[0] % 16));
            try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = dev + moff, .len = 8 * 3 * gk }, 0, std.mem.sliceAsBytes(mem), r.stream.handle);
            try ops.dsStageRows(dev + 8 * scr_acc, dev + moff, gk, s64, s32, n, W, ring, x.one.shape.noise, .{ lay.ids, lay.positions, lay.tokens, lay.counts, lay.lo, lay.hi });
            x.stats.passes += 1;
            x.stats.slots += gk;
            x.cur_at = at;
            x.cur_k = gk;
            r.glue = .{ .ctx = x, .run = glueFn };
            const extra = [_]u64{ if (x.send) |sb| sb.ptr else 0, if (x.send) |sb| sb.len else 0 };
            try d.issueReplay(passKey(gk, at), dsd.fingerprint(r, d, &extra), .{ .ctx = x, .run = groupBody });
            try d.settle(); // counted here on every rank (rank 0's collect does not count it again)
        }
        try d.mark();
        if (x.f.comm.rank() == 0 or x.f.link == null) {
            x.dl = .{ .k = k, .slots = undefined };
            @memcpy(x.dl.?.slots[0..k], slots[0..k]);
        }
    }

    /// A device-started round's ingests: the slots' window rows into their rings, in the slots' order.
    /// - `side` (TF_DSV41_DRAFT_OVERLAP): on the side stream from the runner's taps mark (the last forward's taps final
    ///   there), on scratch roles of their own ("w.ovl."); the main stream joins them here, where the serial ingests
    ///   ran. Else on the main stream.
    /// - TF_DSV41_DRAFT_BATCH_PROJ: main_proj and main_norm once over every slot's rows (`projRound`), then each slot's
    ///   blocks from its rows of "w.ds.mx"; else each slot's whole ingest (emitIngestAt), the serial ingest's programs.
    fn ingestRound(x: *SlotPass, slots: []const u32, side: bool) !void {
        const r = x.f.runner;
        const b = x.bat.?;
        const st = if (side) x.side.?.stream else r.stream;
        if (side) try st.wait(x.side.?.mark);
        const prefix: ?[]const u8 = if (side) "w.ovl." else null;
        var la = std.heap.ArenaAllocator.init(x.gpa);
        defer la.deinit();
        const a = la.allocator();
        const proj = if (x.bproj != .off and slots.len > 1) try x.projRound(a, slots, st, prefix) else null;
        const pos_at = r.addressOf(if (side) "w.ovl.w.ds.pos" else "w.ds.pos") orelse return error.Unbound;
        for (slots) |sl| {
            const pd = b.pend[sl].?;
            const pos: i32 = @intCast(pd.start);
            try gpu.setPos(r.d, pos_at, pos, st);
            const sl_: emit.Slots = .{ .ring_slots = x.nslots, .slot = sl, .taps_role = "w.taps" };
            var cs = if (proj) |a0|
                try emit.emitIngestBlocks(a, x.f.cfg, x.f.widths, x.f.opts, pd.n, pd.a - a0, sl_)
            else
                try emit.emitIngestAt(a, x.f.cfg, x.f.widths, x.f.opts, pd.n, pd.a, sl_);
            if (prefix) |px| cs = try emit.renameRoles(a, cs, px);
            try r.windowOn(st, cs);
        }
        if (side) {
            const sd = &x.side.?;
            try sd.done.record(st);
            try r.stream.wait(sd.done);
            sd.rounds += 1;
            sd.slots += slots.len;
        }
    }

    /// TF_DSV41_DRAFT_BATCH_PROJ: main_proj and main_norm over window rows [a0, a1) (every slot's rows of the round, and
    /// any rows between them) into "w.ds.mx" rows [0, a1 - a0) on `st`; the first row a0, or null (the per-slot
    /// ingests run: too many rows, or a check found a difference). Mode `check`: the first rounds also project each
    /// slot alone ("w.chk." roles) and compare its rows byte for byte before any block reads them.
    fn projRound(x: *SlotPass, a: std.mem.Allocator, slots: []const u32, st: cuda.Stream, prefix: ?[]const u8) !?u32 {
        const r = x.f.runner;
        const b = x.bat.?;
        var a0: u32 = std.math.maxInt(u32);
        var a1: u32 = 0;
        for (slots) |sl| {
            const pd = b.pend[sl].?;
            a0 = @min(a0, pd.a);
            a1 = @max(a1, pd.a + pd.n);
        }
        const n = a1 - a0;
        if (n > proj_rows_max) return null;
        var cs = try emit.emitIngestProj(a, x.f.cfg, x.f.widths, x.f.opts, n, a0, "w.taps");
        if (prefix) |px| cs = try emit.renameRoles(a, cs, px);
        try r.windowOn(st, cs);
        x.bstats.rounds += 1;
        x.bstats.slots += slots.len;
        if (x.bproj == .check and x.bstats.checked < proj_check_rounds) {
            if (!try x.projCheck(a, slots, st, a0, prefix)) return null;
        }
        return a0;
    }

    /// One check round: each slot's rows projected alone ("w.chk." roles) against its rows of the batched "w.ds.mx".
    /// Any byte differing turns the batching off for the run (logged): this round's blocks then run the per-slot
    /// ingests, so no ring holds a batched row that differs.
    fn projCheck(x: *SlotPass, a: std.mem.Allocator, slots: []const u32, st: cuda.Stream, a0: u32, prefix: ?[]const u8) !bool {
        const r = x.f.runner;
        const b = x.bat.?;
        const D: usize = x.f.cfg.hidden;
        const mx = r.addressOf(if (prefix != null) "w.ovl.w.ds.mx" else "w.ds.mx") orelse return error.Unbound;
        const chk = r.addressOf("w.chk.w.ds.mx") orelse return error.Unbound;
        const got = try a.alloc(u16, rowtab.seg_max * D);
        const want = try a.alloc(u16, rowtab.seg_max * D);
        var ok = true;
        for (slots) |sl| {
            const pd = b.pend[sl].?;
            const one = try emit.renameRoles(a, try emit.emitIngestProj(a, x.f.cfg, x.f.widths, x.f.opts, pd.n, pd.a, "w.taps"), "w.chk.");
            try r.windowOn(st, one);
            try st.synchronize();
            const bytes = 2 * @as(usize, pd.n) * D;
            try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = mx + 2 * @as(u64, pd.a - a0) * D, .len = bytes }, 0, std.mem.sliceAsBytes(got[0 .. pd.n * D]));
            try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = chk, .len = bytes }, 0, std.mem.sliceAsBytes(want[0 .. pd.n * D]));
            x.bstats.check_rows += pd.n;
            for (0..pd.n) |i| if (!std.mem.eql(u16, got[i * D ..][0..D], want[i * D ..][0..D])) {
                x.bstats.differ += 1;
                ok = false;
            };
        }
        x.bstats.checked += 1;
        if (!ok) {
            x.bproj = .off;
            std.log.err("draft batched main_proj: {d} of {d} check rows differ from their slot's projection alone: one main_proj a slot from here on", .{ x.bstats.differ, x.bstats.check_rows });
        } else if (x.bstats.checked == proj_check_rounds) {
            std.log.scoped(.dsv41).info("draft batched main_proj: {d} check rounds, {d} rows equal to their slot's projection alone, bit for bit", .{ x.bstats.checked, x.bstats.check_rows });
        }
        return ok;
    }

    /// An ingest of a slot's rows on the device without draining the stream first (dspark_gpu.ingestAsync's form).
    fn ingestAsync(x: *SlotPass, slot: u32, start: u64, at: rows.TapsAt, n: u32) !void {
        const r = x.f.runner;
        const pos: i32 = @intCast(start);
        try gpu.setPos(r.d, r.addressOf("w.ds.pos") orelse return error.Unbound, pos, r.stream);
        var la = std.heap.ArenaAllocator.init(x.gpa);
        defer la.deinit();
        const sl: emit.Slots = .{ .ring_slots = x.nslots, .slot = slot, .taps_role = if (at.src == 1) batch.taps_stash_role else "w.taps" };
        const cs = try emit.emitIngestAt(la.allocator(), x.f.cfg, x.f.widths, x.f.opts, n, at.skip, sl);
        r.glue = .{ .ctx = x, .run = glueFn };
        try r.window(cs);
    }

    fn self(p: *anyopaque) *SlotPass {
        return @ptrCast(@alignCast(p));
    }

    fn leads(x: *const SlotPass) bool {
        return x.f.link != null and !x.f.quiet and x.f.comm.rank() == 0;
    }

    fn sendOp(x: *SlotPass, msg: []const i64) !void {
        if (x.leads()) try x.f.link.?.send(msg);
    }

    fn heads(p: *anyopaque) ?dspark.Markov {
        return self(p).one.markov;
    }

    fn reset(p: *anyopaque, slot: u32) void {
        const x = self(p);
        x.dropDevice();
        x.sendOp(&.{ rows.op_reset, slot }) catch |e| std.log.err("slot drafts: the draft reset did not reach the followers ({t})", .{e});
        if (slot < x.nslots) x.valid[slot] = 0;
    }

    // -------------------------------------------------------------------------------------------------------------
    // ingest

    fn ingest(p: *anyopaque, slot: u32, start: u64, taps: iface.Taps, rws: []const u32) anyerror!void {
        const x = self(p);
        if (slot >= x.nslots) return error.Slots;
        if (rws.len == 0) return;
        for (rws, 0..) |r, i| if (r != rws[0] + i) return error.NotAPrefix;
        if (taps.map != null) return error.NotAPrefix; // trees are refused with several slots (batch.zig)
        // a device-launched round already ingested this slot's window rows (afterPick): the same rows, positions
        if (x.dl) |dl| if (rws[0] == 0 and std.mem.indexOfScalar(u32, dl.slots[0..dl.k], slot) != null) if (x.bat.?.pend[slot]) |pd| if (pd.start == start and rws.len <= pd.n) return;
        const n: u32 = @intCast(rws.len);
        const r = x.f.runner;
        const at = try rows.locate(taps.address, taps.rows, 2 * @as(u64, taps.stride), rws[0], n, x.one.ingest_rows, r.addressOf("w.taps") orelse return error.Unbound, x.stash);
        try x.sendOp(&.{ rows.op_ingest, slot, @intCast(start), at.src, at.skip, n });
        try x.doIngest(slot, start, at, n);
    }

    fn doIngest(x: *SlotPass, slot: u32, start: u64, at: rows.TapsAt, n: u32) !void {
        if (at.src < 0) {
            x.valid[slot] = start + n; // the rows are not on the device: the context restarts after them
            return;
        }
        const r = x.f.runner;
        try r.stream.synchronize();
        const pos: i32 = @intCast(start);
        try gpu.setPos(r.d, r.addressOf("w.ds.pos") orelse return error.Unbound, pos, r.stream);
        var la = std.heap.ArenaAllocator.init(x.gpa);
        defer la.deinit();
        const sl: emit.Slots = .{ .ring_slots = x.nslots, .slot = slot, .taps_role = if (at.src == 1) batch.taps_stash_role else "w.taps" };
        const cs = try emit.emitIngestAt(la.allocator(), x.f.cfg, x.f.widths, x.f.opts, n, at.skip, sl);
        r.glue = .{ .ctx = x, .run = glueFn };
        try r.window(cs);
    }

    // -------------------------------------------------------------------------------------------------------------
    // propose

    fn bySlot(asks: []const iface.Ask, a: usize, b: usize) bool {
        return asks[a].slot < asks[b].slot;
    }

    /// Every ask's drafts: the asks sorted by slot (Python's order), one pass a group (dspark_rows.groups).
    fn propose(p: *anyopaque, asks: []const iface.Ask, out: []iface.Proposal) anyerror!void {
        const x = self(p);
        if (asks.len == 0) return;
        if (asks.len > x.nslots) return error.Slots;
        if (x.begun != null) return error.PassBegun;
        x.dropDevice();
        if (x.devOn()) {
            try x.devLaunchRound(asks);
            return x.devFinishRound(asks, out);
        }
        var order: [16]usize = undefined;
        for (order[0..asks.len], 0..) |*o, i| o.* = i;
        std.sort.insertion(usize, order[0..asks.len], asks, bySlot);
        var gs: [16][2]usize = undefined;
        const ng = rows.groups(asks.len, x.group, &gs);
        const n: usize = x.f.cfg.dspark_block;
        const c: usize = x.f.comm.world() * x.one.k;
        const D: usize = x.f.cfg.hidden;
        var scratch = std.heap.ArenaAllocator.init(x.gpa);
        defer scratch.deinit();
        for (gs[0..ng]) |g| {
            var ms: [16]rows.Member = undefined;
            const idx = order[g[0]..g[1]];
            for (idx, ms[0..idx.len]) |i, *m| {
                const a = asks[i];
                if (a.slot >= x.nslots) return error.Slots;
                m.* = .{ .slot = a.slot, .anchor = a.anchor, .start = a.start, .valid = x.valid[a.slot] };
            }
            var buf: [2 + 3 * 16]i64 = undefined;
            try x.sendOp(try rows.proposeMsg(ms[0..idx.len], &buf));
            try x.runGroup(ms[0..idx.len]);
            // each slot's chain (Python's chain_host) over its own rows of the pass
            for (idx, 0..) |i, j| {
                const a = asks[i];
                const o = out[i];
                const cd = x.cand[j * n * c ..][0 .. n * c];
                const cv = x.cval[j * n * c ..][0 .. n * c];
                try dspark.chain(scratch.allocator(), x.one.markov, cd, cv, c, a.anchor, a.params, if (x.one.conf != null) x.hid[j * n * D ..][0 .. n * D] else null, o.drafts, o.conf);
                if (o.cand) |t| {
                    if (t.len != cd.len) return error.CandidateWidth;
                    @memcpy(t, cd);
                }
                if (o.base) |t| {
                    if (t.len != cv.len) return error.CandidateWidth;
                    @memcpy(t, cv);
                }
            }
        }
    }

    fn devOn(x: *const SlotPass) bool {
        const d = x.dev orelse return false;
        return d.on();
    }

    /// The speculative pass over several slots: every group launched (device path), collected by `collect`.
    fn begin(p: *anyopaque, asks: []const iface.Ask) anyerror!void {
        const x = self(p);
        if (asks.len == 0 or asks.len > x.nslots) return error.Slots;
        if (x.begun != null) return error.PassBegun;
        @memcpy(x.begun_asks[0..asks.len], asks);
        if (x.dl != null and try x.takeDevice(x.begun_asks[0..asks.len])) return;
        if (!x.devOn()) {
            // the checks turned the device path off: the pass runs at `collect` (exact; only the overlap is lost:
            // between the two only work without collectives runs, and the rings are as `begin` saw them)
            x.begun = .{ .n = asks.len, .groups = undefined, .ng = 0 };
            x.begun_host = true;
            return;
        }
        try x.devLaunchRound(x.begun_asks[0..asks.len]);
    }

    /// `begin` over the device-launched round when its slots are the asks' and every slot's (bonus, next position)
    /// on the device is the ask's (anchor, start) - the same picks the host merged; else it is dropped (false).
    fn takeDevice(x: *SlotPass, asks: []const iface.Ask) !bool {
        const dl = x.dl.?;
        x.dl = null;
        for (x.order[0..asks.len], 0..) |*o, i| o.* = i;
        std.sort.insertion(usize, x.order[0..asks.len], asks, bySlot);
        var ok = asks.len == dl.k;
        if (ok) {
            try x.staged.?.synchronize(); // the accept's copy back (not the passes)
            const h: []const i64 = @alignCast(std.mem.bytesAsSlice(i64, x.hscr.?.bytes[0 .. 8 * scr_len]));
            for (x.order[0..asks.len], 0..) |i, j| {
                const a = asks[i];
                if (a.slot != dl.slots[j] or h[scr_acc + 3 * j + 1] != a.anchor or h[scr_acc + 3 * j + 2] != @as(i64, @intCast(a.start))) ok = false;
            }
        }
        if (!ok) {
            x.dstats.device_dropped += 1;
            return false;
        }
        var bg: Begun = .{ .n = asks.len, .groups = undefined, .ng = 0 };
        bg.ng = rows.groups(asks.len, x.group, &bg.groups);
        x.begun = bg;
        x.begun_dev = true;
        x.dstats.device_starts += 1;
        // the chain after the round's passes (enqueued at the pick), the round's event re-recorded after it
        if (try x.launchChain(asks)) try x.dev.?.mark();
        return true;
    }

    fn collect(p: *anyopaque, out: []iface.Proposal) anyerror!void {
        const x = self(p);
        const b = x.begun orelse return error.NoPassBegun;
        if (out.len != b.n) return error.Slots;
        x.begun_dev = false; // a device-launched round collects as a begun one (devFinishRound: wait, unpack, chains)
        if (x.begun_host) {
            x.begun_host = false;
            x.begun = null;
            return propose(p, x.begun_asks[0..b.n], out);
        }
        return x.devFinishRound(x.begun_asks[0..b.n], out);
    }

    /// Every group of the round on the device (the leader's op first, each), in Python's slot order; during the check
    /// passes each group is waited for and checked at once, as a follower does (the ranks' verdict pairs up).
    fn devLaunchRound(x: *SlotPass, asks: []const iface.Ask) !void {
        const d = x.dev.?;
        for (x.order[0..asks.len], 0..) |*o, i| o.* = i;
        std.sort.insertion(usize, x.order[0..asks.len], asks, bySlot);
        var b: Begun = .{ .n = asks.len, .groups = undefined, .ng = 0 };
        b.ng = rows.groups(asks.len, x.group, &b.groups);
        const n: usize = x.f.cfg.dspark_block;
        for (b.groups[0..b.ng], 0..) |g, gi| {
            var ms: [16]rows.Member = undefined;
            const idx = x.order[g[0]..g[1]];
            for (idx, ms[0..idx.len]) |i, *m| {
                const a = asks[i];
                if (a.slot >= x.nslots) return error.Slots;
                m.* = .{ .slot = a.slot, .anchor = a.anchor, .start = a.start, .valid = x.valid[a.slot] };
            }
            // the device path's op carries the group's first row of the round (its region and graph key), last
            var buf: [3 + 3 * 16]i64 = undefined;
            const msg = try rows.proposeMsg(ms[0..idx.len], buf[0 .. 2 + 3 * 16]);
            buf[msg.len] = @intCast(g[0] * n);
            try x.sendOp(buf[0 .. msg.len + 1]);
            _ = gi;
            try x.devGroup(ms[0..idx.len], g[0] * n);
            if (d.stats.passes < dsd.check_passes) {
                try d.mark();
                try d.wait();
                try d.check(x.last_logits, x.last_v, idx.len * n, g[0] * n);
            }
            try d.settle();
        }
        _ = try x.launchChain(asks);
        try d.mark();
        x.begun = b;
    }

    /// TF_DSV41_DRAFT_CHAIN: the round's chain on the device after its passes, the asks in the round's order
    /// (`x.order`, sorted by slot); false: the host's chain at collect.
    fn launchChain(x: *SlotPass, asks: []const iface.Ask) !bool {
        const ch = (x.dev orelse return false).chain orelse return false;
        var an: [16]u32 = undefined;
        var ps: [16]dspark.Params = undefined;
        for (x.order[0..asks.len], 0..) |i, j| {
            an[j] = asks[i].anchor;
            ps[j] = asks[i].params;
        }
        return ch.launch(x.f.runner, x.f.cfg, x.f.widths, x.f.opts, an[0..asks.len], ps[0..asks.len]);
    }

    /// The round's results: the event waited for, each group's candidates unpacked, each slot's chain.
    fn devFinishRound(x: *SlotPass, asks: []const iface.Ask, out: []iface.Proposal) !void {
        const d = x.dev.?;
        const b = x.begun.?;
        x.begun = null;
        {
            const tw = ph.start(x.f.phases, .@"draft.wait");
            defer tw.stop();
            try d.wait();
        }
        const n: usize = x.f.cfg.dspark_block;
        const c: usize = x.f.comm.world() * x.one.k;
        const D: usize = x.f.cfg.hidden;
        if (d.chain) |ch| if (ch.launched == b.n) return x.chainResults(ch, asks, out, b);
        for (b.groups[0..b.ng]) |g| {
            const at = g[0] * n;
            const R = (g[1] - g[0]) * n;
            try d.unpack(R, at, x.f.cfg.vocab, x.cand[at * c ..], x.cval[at * c ..], if (x.one.conf != null) x.hid[at * D ..] else null);
        }
        var scratch = std.heap.ArenaAllocator.init(x.gpa);
        defer scratch.deinit();
        const tc = ph.start(x.f.phases, .@"draft.chain");
        defer tc.stop();
        // rows of the round are in sorted-slot order: the j-th sorted ask's rows at j x n
        for (x.order[0..b.n], 0..) |i, j| {
            const a = asks[i];
            const o = out[i];
            const cd = x.cand[j * n * c ..][0 .. n * c];
            const cv = x.cval[j * n * c ..][0 .. n * c];
            try dspark.chain(scratch.allocator(), x.one.markov, cd, cv, c, a.anchor, a.params, if (x.one.conf != null) x.hid[j * n * D ..][0 .. n * D] else null, o.drafts, o.conf);
            if (o.cand) |t| {
                if (t.len != cd.len) return error.CandidateWidth;
                @memcpy(t, cd);
            }
            if (o.base) |t| {
                if (t.len != cv.len) return error.CandidateWidth;
                @memcpy(t, cv);
            }
        }
    }

    /// The device chain's drafts and confidences (round order) into each ask's proposal; a tree's candidates and base
    /// logits (before the bias) from the gathered copy, as the host path gives them.
    fn chainResults(x: *SlotPass, ch: *dsd.Chain, asks: []const iface.Ask, out: []iface.Proposal, b: Begun) !void {
        const tc = ph.start(x.f.phases, .@"draft.chain");
        defer tc.stop();
        ch.launched = 0;
        const n: usize = x.f.cfg.dspark_block;
        const c: usize = x.f.comm.world() * x.one.k;
        const res = ch.results();
        var trees = false;
        for (out) |o| trees = trees or o.cand != null or o.base != null;
        // the first rounds: the host's chain over the same candidates too (the kernel agrees with it up to fp32
        // summation order, so a near tie may differ; wiring, anchors or parameters gone wrong differ everywhere)
        const checking = ch.stats.rounds <= dsd.Chain.check_rounds;
        const D: usize = x.f.cfg.hidden;
        if (trees or checking) for (b.groups[0..b.ng]) |g| {
            const at = g[0] * n;
            try x.dev.?.unpack((g[1] - g[0]) * n, at, x.f.cfg.vocab, x.cand[at * c ..], x.cval[at * c ..], if (checking and x.one.conf != null) x.hid[at * D ..] else null);
        };
        if (checking) {
            var scratch = std.heap.ArenaAllocator.init(x.gpa);
            defer scratch.deinit();
            var drafts: [64]u32 = undefined;
            var conf: [64]f32 = undefined;
            for (x.order[0..b.n], 0..) |i, j| {
                const a = asks[i];
                try dspark.chain(scratch.allocator(), x.one.markov, x.cand[j * n * c ..][0 .. n * c], x.cval[j * n * c ..][0 .. n * c], c, a.anchor, a.params, if (x.one.conf != null) x.hid[j * n * D ..][0 .. n * D] else null, drafts[0..n], conf[0..n]);
                for (drafts[0..n], res.draft[j * n ..][0..n]) |h, d| {
                    ch.checked += 1;
                    if (h != @as(u32, @intCast(d))) ch.differ += 1;
                }
            }
            if (ch.stats.rounds == dsd.Chain.check_rounds) ch.verdict();
        }
        for (x.order[0..b.n], 0..) |i, j| {
            const o = out[i];
            if (o.drafts.len != n or o.conf.len != n) return error.CandidateWidth;
            for (o.drafts, res.draft[j * n ..][0..n]) |*t, v| t.* = @intCast(v);
            if (ch.conf) @memcpy(o.conf, res.conf[j * n ..][0..n]) else @memset(o.conf, 0.5);
            if (o.cand) |t| {
                if (t.len != n * c) return error.CandidateWidth;
                @memcpy(t, x.cand[j * n * c ..][0 .. n * c]);
            }
            if (o.base) |t| {
                if (t.len != n * c) return error.CandidateWidth;
                @memcpy(t, x.cval[j * n * c ..][0 .. n * c]);
            }
        }
    }

    /// One group on the device: statics from its pinned staging region (no sync), the pass (its (size, row) graph once
    /// the checks passed); its candidates land at round row `at`.
    fn devGroup(x: *SlotPass, ms: []const rows.Member, at: usize) !void {
        const d = x.dev.?;
        const f = x.f;
        const n: u32 = f.cfg.dspark_block;
        const lay = emit.Layout.of(@intCast(ms.len), n, f.cfg.window);
        const sv = try d.statics(at / @as(usize, n), @intCast(lay.len64), @intCast(lay.len32));
        rows.stage(lay, n, f.cfg.window, @intCast(emit.ringRows(f.cfg)), x.one.shape.noise, ms, sv[0], sv[1]);
        const r = f.runner;
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.i64") orelse return error.Unbound, .len = 8 * @as(usize, @intCast(lay.len64)) }, 0, std.mem.sliceAsBytes(sv[0]), r.stream.handle);
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.i32") orelse return error.Unbound, .len = 4 * @as(usize, @intCast(lay.end32)) }, 0, std.mem.sliceAsBytes(sv[1][0..@intCast(lay.end32)]), r.stream.handle);
        x.stats.passes += 1;
        x.stats.slots += ms.len;
        x.cur_at = at;
        r.glue = .{ .ctx = x, .run = glueFn };
        const extra = [_]u64{ if (x.send) |b| b.ptr else 0, if (x.send) |b| b.len else 0 };
        x.cur_k = ms.len;
        const key = passKey(ms.len, at);
        if (x.capture_first and d.active == .graphs) x.markSeen(key);
        try d.issue(key, dsd.fingerprint(r, d, &extra), .{ .ctx = x, .run = groupBody });
    }

    /// A pass's graph key: its slot count and its first row in the round.
    fn passKey(k: usize, at: usize) u32 {
        return @intCast(k << 16 | at);
    }

    fn wasSeen(x: *const SlotPass, key: u32) bool {
        return std.mem.indexOfScalar(u32, x.seen[0..x.nseen], key) != null;
    }

    fn markSeen(x: *SlotPass, key: u32) void {
        if (x.wasSeen(key) or x.nseen == x.seen.len) return;
        x.seen[x.nseen] = key;
        x.nseen += 1;
    }

    fn groupBody(ctx: *anyopaque, stream: cuda.Stream) anyerror!void {
        const x: *SlotPass = @ptrCast(@alignCast(ctx));
        _ = stream;
        try x.f.runner.window(x.programs[x.cur_k - 1]);
    }

    /// One pass over `ms` on this rank: the statics staged, the launches, the host's candidates (every rank).
    fn runGroup(x: *SlotPass, ms: []const rows.Member) !void {
        const f = x.f;
        const n: u32 = f.cfg.dspark_block;
        const lay = emit.Layout.of(@intCast(ms.len), n, f.cfg.window);
        rows.stage(lay, n, f.cfg.window, @intCast(emit.ringRows(f.cfg)), x.one.shape.noise, ms, x.h64, x.h32);
        const r = f.runner;
        try r.stream.synchronize();
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.i64") orelse return error.Unbound, .len = 8 * @as(usize, @intCast(lay.len64)) }, 0, std.mem.sliceAsBytes(x.h64[0..@intCast(lay.len64)]), r.stream.handle);
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.i32") orelse return error.Unbound, .len = 4 * @as(usize, @intCast(lay.end32)) }, 0, std.mem.sliceAsBytes(x.h32[0..@intCast(lay.end32)]), r.stream.handle);
        x.stats.passes += 1;
        x.stats.slots += ms.len;
        r.glue = .{ .ctx = x, .run = glueFn };
        try r.window(x.programs[ms.len - 1]);
    }

    // -------------------------------------------------------------------------------------------------------------
    // followers

    fn followFn(ptr: *anyopaque, msg: []const i64) anyerror!void {
        const x: *SlotPass = @ptrCast(@alignCast(ptr));
        switch (msg[0]) {
            rows.op_reset => {
                if (msg.len != 2) return error.BadPlan;
                const s: usize = @intCast(msg[1]);
                if (s < x.nslots) x.valid[s] = 0;
            },
            rows.op_ingest => {
                if (msg.len != 6 or msg[1] < 0 or msg[1] >= x.nslots) return error.BadPlan;
                try x.doIngest(@intCast(msg[1]), @intCast(msg[2]), .{ .src = msg[3], .skip = msg[4] }, @intCast(msg[5]));
            },
            gpu.GpuPass.op_arm => {
                // [op, slot, start, rows, sampled]: this pass's arm; [op, start, rows, sampled]: the one-slot pass's
                if (msg.len == 5) {
                    const sl: usize = @intCast(msg[1]);
                    if (sl >= x.nslots) return error.BadPlan;
                    x.armed[sl] = .{ .start = @intCast(msg[2]), .n = @intCast(msg[3]), .sampled = msg[4] != 0 };
                    return;
                }
                const e = x.prev_ext orelse return error.BadPlan;
                try e.run(e.ptr, msg);
            },
            rows.op_propose => {
                var ms: [16]rows.Member = undefined;
                if (x.devOn()) {
                    // the leader's group, run here alike at its row of the round (the op's last word): the same
                    // graph keys in the same order on every rank; a follower reads no results
                    if (msg.len < 3) return error.BadPlan;
                    const at: usize = @intCast(msg[msg.len - 1]);
                    const got = try rows.proposeOf(msg[0 .. msg.len - 1], x.valid, ms[0..x.group]);
                    const d = x.dev.?;
                    try x.devGroup(got, at);
                    if (d.stats.passes < dsd.check_passes) {
                        try d.mark();
                        try d.wait();
                        try d.check(x.last_logits, x.last_v, got.len * x.f.cfg.dspark_block, at);
                    }
                    try d.settle();
                    return;
                }
                try x.runGroup(try rows.proposeOf(msg, x.valid, ms[0..x.group]));
            },
            else => {
                const e = x.prev_ext orelse return error.BadPlan;
                try e.run(e.ptr, msg);
            },
        }
    }

    // -------------------------------------------------------------------------------------------------------------
    // glue

    fn device(x: *SlotPass, slot: *?cuda.DeviceBuffer, bytes: usize) !u64 {
        if (slot.* == null or slot.*.?.len < bytes) {
            if (slot.*) |*b| b.free();
            slot.* = try cuda.DeviceBuffer.alloc(x.f.runner.d, bytes);
        }
        return slot.*.?.ptr;
    }

    fn numel(t: calls.Tensor) usize {
        var n: usize = 1;
        for (t.shape) |d| n *= @intCast(d);
        return n;
    }

    /// dspark_gpu.zig's glue steps at the pass's rows (`R`: every slot's).
    fn glueFn(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
        const x: *SlotPass = @ptrCast(@alignCast(ctx));
        const step = c.name["glue.".len..];
        const s = r.stream.handle;
        const k = r.kernels.others(r.stream);
        const comm = x.f.comm;
        const D: usize = x.f.cfg.hidden;
        const Wd: usize = comm.world();
        if (std.mem.eql(u8, step, "ds_embed")) {
            const n: usize = numel(c.args[2].arg.t);
            const V: usize = x.f.cfg.vocab / Wd;
            const embed = r.weights.get("embed") orelse return error.MissingWeight;
            const rws = try x.device(&x.send, 2 * n * D);
            try k.embedSend(try r.tensorAddr(c.args[2].arg.t), n, embed.ptr, @intCast(comm.rank() * V), V, D, rws);
            const recv = try r.tensorAddr(c.args[1].arg.t);
            try comm.allGather(rws, recv, n * D, .bf16, s);
            return k.embedSum(recv, Wd, n, D, try r.tensorAddr(c.args[0].arg.t));
        }
        if (std.mem.eql(u8, step, "exchange_f32")) {
            const part = c.args[0].arg.t;
            const n = numel(part);
            const half = try x.device(&x.send, 2 * n);
            try k.castBf16(try r.tensorAddr(part), half, n);
            return comm.allGather(half, try r.tensorAddr(c.args[1].arg.t), n, .bf16, s);
        }
        if (std.mem.eql(u8, step, "widen"))
            return k.widenF32(try r.tensorAddr(c.args[0].arg.t), try r.tensorAddr(c.args[1].arg.t), numel(c.args[0].arg.t));
        if (std.mem.eql(u8, step, "ds_out")) {
            if (x.dev) |dv| if (dv.on()) {
                x.last_logits = try r.tensorAddr(c.args[0].arg.t);
                x.last_v = @intCast(c.args[0].arg.t.shape[1]);
                return dv.launch(r, c.args[0].arg.t, c.args[1].arg.t, @intCast(c.args[0].arg.t.shape[0]), x.cur_at);
            };
            return x.hostStep(r, c.args[0].arg.t, c.args[1].arg.t);
        }
        return error.GlueNotBuilt;
    }

    /// dspark_gpu.zig's host step over the pass's rows: each rank's best k a row, the ranks' lists gathered, the head
    /// hidden for the confidences.
    fn hostStep(x: *SlotPass, r: *run.Runner, logits_t: calls.Tensor, hid_t: calls.Tensor) !void {
        const th = ph.start(x.f.phases, .@"draft.host");
        defer th.stop();
        const comm = x.f.comm;
        const R: usize = @intCast(logits_t.shape[0]);
        const V: usize = @intCast(logits_t.shape[1]);
        const D: usize = x.f.cfg.hidden;
        const k: usize = x.one.k;
        const Wd: usize = comm.world();
        const logits = x.logits[0 .. R * V];
        const hid16 = x.hid16[0 .. R * D];
        try r.stream.synchronize();
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = try r.tensorAddr(logits_t), .len = 4 * logits.len }, 0, std.mem.sliceAsBytes(logits));
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = try r.tensorAddr(hid_t), .len = 2 * hid16.len }, 0, std.mem.sliceAsBytes(hid16));
        for (hid16, x.hid[0 .. R * D]) |b, *h| h.* = dspark.bf16(b);
        const packed_ = x.packed_[0 .. R * 2 * k];
        const id0: u32 = @intCast(comm.rank() * V);
        for (0..R) |i| {
            const row = packed_[i * 2 * k ..][0 .. 2 * k];
            try dspark.candidates(x.gpa, logits[i * V ..][0..V], id0, x.ids, row[0..k]);
            for (x.ids, row[k..]) |id, *o| o.* = @bitCast(id);
        }
        const bytes = 4 * packed_.len;
        const src = try x.device(&x.send, bytes);
        const dst = try x.device(&x.gathered, bytes * Wd);
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = src, .len = bytes }, 0, std.mem.sliceAsBytes(packed_), r.stream.handle);
        try comm.allGather(src, dst, packed_.len, .f32, r.stream.handle);
        try r.stream.synchronize();
        const all = x.all[0 .. packed_.len * Wd];
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = dst, .len = 4 * all.len }, 0, std.mem.sliceAsBytes(all));
        rows.ungather(all, Wd, R, k, x.cand, x.cval);
        try gpu.checkCandidates(x.cand[0 .. R * Wd * k], x.f.cfg.vocab); // the pass's rows (x.cand holds the most)
    }
};

/// TF_DSV41_DRAFT_OVERLAP: 1 = a device-started round's ingests overlapped with its window's tail, unset / 0 = serial.
pub fn overlapFromEnv() !bool {
    const v = std.mem.span(std.c.getenv("TF_DSV41_DRAFT_OVERLAP") orelse return false);
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.BadDraftOverlap;
}

/// TF_DSV41_DRAFT_CAPTURE: 1 = a device-started round over a pass key no regular round issued yet runs the regular
/// path once (its graph captured), unset / 0 = the speculation replays or runs eagerly, never captures (today).
pub fn captureFromEnv() !bool {
    const v = std.mem.span(std.c.getenv("TF_DSV41_DRAFT_CAPTURE") orelse return false);
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.BadDraftCapture;
}

/// The engine's AOT set has some variant of Triton function `name`.
/// TF_DSV41_DRAFT_BATCH_PROJ: 1 = one main_proj over a device-started round's slots, `check` = the same with the first
/// rounds' rows compared with each slot's projection alone, unset / 0 = one a slot.
pub const BatchProj = enum { off, on, check };

pub fn batchProjFromEnv() !BatchProj {
    const v = std.mem.span(std.c.getenv("TF_DSV41_DRAFT_BATCH_PROJ") orelse return .off);
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.BadDraftBatchProj;
}

/// The batched projection's most rows: a row window's (rowtab's largest bucket; linear.cu takes up to 128).
pub const proj_rows_max = rowtab.buckets[rowtab.buckets.len - 1];
/// TF_DSV41_DRAFT_BATCH_PROJ=check: rounds whose rows are compared
pub const proj_check_rounds = 8;
