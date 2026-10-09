//! DSpark's drafting pass on the GPU: draft/iface.zig's `Pass` over dspark_emit.zig's launches, run by the forward's
//! runner on every rank (the leader sends each ingest / pass to the followers: forward.Drafter). One slot.
//!
//! - **ingest**: the committed rows' taps ("w.taps", the target's last window) into the three blocks' rings. A run
//!   whose rows are not on the device (a slot resumed from another engine's state) restarts the context after them
//!   (Python's restore without DSpark rows: `valid`).
//! - **propose**: the statics staged as Python's DraftInputs (ids, positions, the context list, lo / hi, rows), the
//!   pass, then the host: each rank's best K head columns (dspark.candidates), the ranks' lists gathered (rank order,
//!   Python's gather_cols), the Markov chain and confidences (dspark.chain), as drafter.chain_host. Drafts only
//!   propose (verification is exact): this rule equals the `_chain` kernel's at T = 0 up to fp32 summation order.
//!
//! Glue the pass needs: the embedding of its ids, q's fp32 copy before `_rope` (glue.cu widen_f32), the fp32
//! partials' exchanges, and "ds_out" (the host's step).

const std = @import("std");
const cuda = @import("cuda");
const calls = @import("calls.zig");
const run = @import("run.zig");
const buffers = @import("buffers.zig");
const fwd = @import("forward.zig");
const emit = @import("dspark_emit.zig");
const iface = @import("draft/iface.zig");
const dspark = @import("draft/dspark.zig");
const dsd = @import("dspark_dev.zig");
const rows_mod = @import("dspark_rows.zig");
const ph = @import("phases.zig");

pub const Error = error{ Slots, NotAPrefix, NoWeight, CandidateWidth, BadCandidates };

pub const GpuPass = struct {
    gpa: std.mem.Allocator,
    f: *fwd.Forward,
    shape: dspark.Shape,
    /// candidates a rank (drafter.k: min(64, 128 / world))
    k: u32,
    /// the most rows an ingest takes at once (its buffers are planned for it)
    ingest_rows: u32,
    markov: dspark.Markov,
    w1: []u16,
    w2: []u16,
    conf: ?[]f32,
    arena: std.heap.ArenaAllocator,
    pass_calls: []const calls.Call = &.{},
    /// where the pass's host step ("glue.ds_out") starts: `begin` launches the calls before it, `collect` the rest
    head: usize = 0,
    /// the ask a `begin` launched (TF_DSV41_SPEC_DRAFT, draft/spec.zig), until its `collect`
    begun: ?iface.Ask = null,
    /// the first position whose ring rows are this slot's
    valid: u64 = 0,
    st: emit.Statics,
    h64: []i64,
    h32: []i32,
    send: ?cuda.DeviceBuffer = null,
    gathered: ?cuda.DeviceBuffer = null,
    /// the last pass's results on the host: gathered candidates / base logits [n][W k], head hidden [n][D]
    cand: []i32,
    cval: []f32,
    hid: []f32,
    /// TF_DSV41_DRAFT_GRAPHS (dspark_dev.zig): candidates on the device, the pass graphed; null: the host step
    dev: ?*dsd.Dev = null,
    /// the last eager pass's logits (a check pass compares the device candidates with the host's on them)
    last_logits: u64 = 0,
    last_v: usize = 0,
    /// the speculative pass from the device pick (spec.py's order; Pass.arm, forward.after_pick): the armed window
    /// (start, rows) until its pick, then the pass launched from it until `begin` takes it (rank 0)
    armed: ?struct { start: u64, n: u32, sampled: bool } = null,
    /// `wait`: the host values come from the accept's copy (a sampled window): `begin` waits for `staged` first
    dl: ?struct { start: u64, n: u32, host: [*]const i64, rows: u32, wait: bool = false } = null,
    /// a sampled window's accept: device ids / segment / result, its pinned copy and event
    sscr: ?cuda.DeviceBuffer = null,
    shost: ?cuda.HostBuffer = null,
    staged: ?cuda.Event = null,
    /// `begin` took a device-launched pass: `collect` waits and unpacks without counting a pass again
    begun_dev: bool = false,
    prev_ext: ?fwd.Forward.Ext = null,
    stats: struct { device_starts: u64 = 0, device_dropped: u64 = 0 } = .{},

    /// The buffer roles DSpark adds to the forward's plan (call before Runner.bind).
    pub fn plan(a: std.mem.Allocator, f: *const fwd.Forward, p: *buffers.Plan, ingest_rows: u32) !void {
        try p.add(try emit.emitIngest(a, f.cfg, f.widths, f.opts, ingest_rows, 0));
        try p.add(try emit.emitPass(a, f.cfg, f.widths, f.opts, f.cfg.dspark_block));
    }

    pub fn init(gpa: std.mem.Allocator, f: *fwd.Forward, ingest_rows: u32) !*GpuPass {
        const cfg = f.cfg;
        const r = f.runner;
        const world = f.comm.world();
        const x = try gpa.create(GpuPass);
        errdefer gpa.destroy(x);
        const n: i64 = cfg.dspark_block;
        const k: u32 = @min(64, 128 / world);
        const rank = cfg.dspark_markov_rank;
        const V: usize = cfg.vocab;
        const w1 = try gpa.alloc(u16, V * rank);
        const w2 = try gpa.alloc(u16, V * rank);
        try fetch(r, "dspark.markov.w1", std.mem.sliceAsBytes(w1));
        try fetch(r, "dspark.markov.w2", std.mem.sliceAsBytes(w2));
        var conf: ?[]f32 = null;
        if (r.weights.get("dspark.conf")) |t| {
            conf = try gpa.alloc(f32, t.len / 4);
            try fetch(r, "dspark.conf", std.mem.sliceAsBytes(conf.?));
        }
        const st: emit.Statics = .{ .n = n, .window = cfg.window };
        const W: usize = world * k;
        x.* = .{
            .gpa = gpa,
            .f = f,
            .shape = .{ .block = cfg.dspark_block, .noise = cfg.dspark_noise_token, .window = cfg.window, .rank = rank, .hidden = cfg.hidden, .candidates = world * k },
            .k = k,
            .ingest_rows = ingest_rows,
            .markov = .{ .w1 = w1, .w2 = w2, .rank = rank, .conf = conf },
            .w1 = w1,
            .w2 = w2,
            .conf = conf,
            .arena = std.heap.ArenaAllocator.init(gpa),
            .st = st,
            .h64 = try gpa.alloc(i64, @intCast(st.len64())),
            .h32 = try gpa.alloc(i32, @intCast(st.len32())),
            .cand = try gpa.alloc(i32, @as(usize, @intCast(n)) * W),
            .cval = try gpa.alloc(f32, @as(usize, @intCast(n)) * W),
            .hid = try gpa.alloc(f32, @as(usize, @intCast(n)) * cfg.hidden),
        };
        x.pass_calls = try emit.emitPass(x.arena.allocator(), cfg, f.widths, f.opts, n);
        x.head = x.pass_calls.len;
        for (x.pass_calls, 0..) |c, i| if (std.mem.eql(u8, c.name, "glue.ds_out")) {
            x.head = i;
        };
        f.drafter = .{ .ptr = x, .ingest = followIngest, .propose = followPropose, .reset = followReset };
        const mode = try dsd.modeFromEnv();
        if (mode != .host) {
            if (r.kernels.others(r.stream).f.ds_cands == null) {
                std.log.scoped(.dsv41).warn("draft graphs: TF_DSV41_DRAFT_GRAPHS set, but the glue fatbin has no ds_cands kernel: the host step", .{});
            } else {
                // every buffer a graphed pass bakes in, at its largest: the embedding's and exchanges' send
                _ = try x.device(&x.send, 2 * @as(usize, @intCast(n)) * cfg.hidden);
                x.dev = try dsd.Dev.init(gpa, r.d, f.comm, r.stream, mode, k, cfg.hidden, @intCast(n), conf != null, @intCast(8 * st.len64() + 4 * st.len32()), 1);
                // the speculative pass from the device pick (needs TF_DSV41_GREEDY_GPU's pick and ds_stage)
                if (r.kernels.others(r.stream).f.ds_stage != null) {
                    f.after_pick = .{ .ctx = x, .run = afterPick };
                    if (r.kernels.others(r.stream).f.ds_accept_rows != null) {
                        x.sscr = try cuda.DeviceBuffer.alloc(r.d, 8 * 128);
                        x.shost = try cuda.HostBuffer.alloc(r.d, 8 * 128);
                        x.staged = try cuda.Event.init(r.d, false);
                        f.after_sample = .{ .ctx = x, .run = afterSample };
                    }
                    x.prev_ext = f.ext;
                    f.ext = .{ .ptr = x, .run = followExt };
                }
            }
        }
        return x;
    }

    pub fn deinit(x: *GpuPass) void {
        const gpa = x.gpa;
        x.f.drafter = null;
        if (x.f.after_pick != null and x.f.after_pick.?.ctx == @as(*anyopaque, x)) {
            x.f.after_pick = null;
            x.f.ext = x.prev_ext;
        }
        if (x.f.after_sample != null and x.f.after_sample.?.ctx == @as(*anyopaque, x)) x.f.after_sample = null;
        if (x.sscr) |*b| b.free();
        if (x.shost) |*b| b.free();
        if (x.staged) |*e| e.deinit();
        if (x.dev) |d| {
            if (x.stats.device_starts + x.stats.device_dropped > 0) std.log.scoped(.dsv41).info("draft graphs: {d} speculative passes started from the device pick, {d} unused", .{ x.stats.device_starts, x.stats.device_dropped });
            d.deinit();
        }
        if (x.send) |*b| b.free();
        if (x.gathered) |*b| b.free();
        gpa.free(x.w1);
        gpa.free(x.w2);
        if (x.conf) |c| gpa.free(c);
        gpa.free(x.h64);
        gpa.free(x.h32);
        gpa.free(x.cand);
        gpa.free(x.cval);
        gpa.free(x.hid);
        x.arena.deinit();
        gpa.destroy(x);
    }

    fn fetch(r: *run.Runner, name: []const u8, out: []u8) !void {
        const t = r.weights.get(name) orelse return error.NoWeight;
        if (t.len < out.len) return error.NoWeight;
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = t.ptr, .len = out.len }, 0, out);
    }

    pub fn pass(x: *GpuPass) iface.Pass {
        return .{ .ptr = x, .vtable = &vtable };
    }

    const vtable: iface.Pass.VTable = .{ .ingest = ingest, .propose = propose, .reset = reset, .markov = heads, .begin = begin, .collect = collect, .arm = arm };

    /// 43 (dspark_rows' range 40-43): [op, start, rows, sampled] the armed window (every rank: the pass starts from
    /// its pick); SlotPass's has the slot first (5 words)
    pub const op_arm: i64 = 43;

    /// Pass.arm (rank 0): the slot's coming window of `rows` rows at `start`, for the speculative pass from its device
    /// pick; the followers get it before the window. Only once the checks passed and with the device path on.
    fn arm(p: *anyopaque, slot: u32, start: u64, n_rows: u32, sampled: bool) void {
        const x = self(p);
        x.dropDevice();
        if (slot != 0 or x.f.after_pick == null or !x.devReady()) return;
        if (sampled and x.f.after_sample == null) return;
        x.f.sendDrafter(&.{ op_arm, @intCast(start), n_rows, @intFromBool(sampled) }) catch |e| {
            std.log.err("dspark: the arm did not reach the followers ({t})", .{e});
            return;
        };
        x.armed = .{ .start = start, .n = n_rows, .sampled = sampled };
    }

    fn followExt(ptr: *anyopaque, msg: []const i64) anyerror!void {
        const x: *GpuPass = @ptrCast(@alignCast(ptr));
        if (msg[0] == op_arm) {
            if (msg.len != 4) return error.BadPlan;
            x.armed = .{ .start = @intCast(msg[1]), .n = @intCast(msg[2]), .sampled = msg[3] != 0 };
            return;
        }
        const e = x.prev_ext orelse return error.BadPlan;
        return e.run(e.ptr, msg);
    }

    /// The device path is on and past its checks (a pass launched outside `propose` / `begin` then counts no check).
    fn devReady(x: *const GpuPass) bool {
        const d = x.dev orelse return false;
        return d.on() and d.stats.passes >= dsd.check_passes;
    }

    /// forward.after_pick, every rank, after the armed window's device pick (its copy back recorded): the window's
    /// taps into the rings, the pass's statics from (bonus, next position) on the device (ds_stage), the pass (its
    /// graph if captured, never a capture here), the event `collect` waits on. Python's spec.py order: the GPU drafts
    /// while the hosts sample, plan and commit.
    fn afterPick(ctx: *anyopaque, pick: fwd.Forward.PickDevice) anyerror!void {
        const x: *GpuPass = @ptrCast(@alignCast(ctx));
        const a = x.armed orelse return;
        if (a.sampled) return; // the sampler's choice starts it (afterSample)
        x.armed = null;
        if (!pick.chain or pick.n != a.n or pick.start != a.start or pick.host == null or x.begun != null) return;
        if (!x.devReady() or a.n > x.ingest_rows) return;
        try x.launchFrom(a.start, a.n, pick.out, pick.n);
        if (x.f.comm.rank() == 0 or x.f.link == null) x.dl = .{ .start = a.start, .n = a.n, .host = pick.host.?, .rows = a.n };
    }

    /// forward.after_sample, every rank, after a sampled window's device choice (sampling_gpu.zig, vsample.choose):
    /// the window's accept on the device (ds_accept_rows over its one segment, copied back for `begin`), then the
    /// pass as afterPick starts it.
    fn afterSample(ctx: *anyopaque, row0: u32, n: u32, chosen: u64) anyerror!void {
        const x: *GpuPass = @ptrCast(@alignCast(ctx));
        const a = x.armed orelse return;
        if (!a.sampled) return;
        x.armed = null;
        const p = x.f.slot.pending orelse return;
        if (row0 != 0 or n != a.n or p.start != a.start or p.n != a.n or x.begun != null) return;
        if (!x.devReady() or a.n > x.ingest_rows or x.f.ids.len < n) return;
        const r = x.f.runner;
        // [ids n | segment 3 | acc 3] int64
        const h: []i64 = @alignCast(std.mem.bytesAsSlice(i64, x.shost.?.bytes[0 .. 8 * 128]));
        for (x.f.ids[0..n], 0..) |t, i| h[i] = t;
        h[64..67].* = .{ 0, n, @intCast(a.start) };
        const dev = x.sscr.?.ptr;
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = dev, .len = 8 * 67 }, 0, std.mem.sliceAsBytes(h[0..67]), r.stream.handle);
        try r.kernels.others(r.stream).dsAcceptRows(chosen, dev, dev + 8 * 64, 1, dev + 8 * 68);
        try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(x.shost.?.bytes.ptr + 8 * 68, dev + 8 * 68, 8 * 3, r.stream.handle), "cuMemcpyDtoHAsync");
        try x.staged.?.record(r.stream);
        // acc [accepted, bonus, next position] reads as a pick of 0 rows: ds_stage takes pick[1], pick[2]
        try x.launchFrom(a.start, a.n, dev + 8 * 68, 0);
        if (x.f.comm.rank() == 0 or x.f.link == null) x.dl = .{ .start = a.start, .n = 0, .host = @ptrCast(h[68..].ptr), .rows = a.n, .wait = true };
    }

    /// The device start's launches, every rank: the window's taps, the statics (placeholders, then ds_stage from
    /// `pick` [nw + 3]'s bonus and next position), the pass (its graph if captured), the event; counted here.
    fn launchFrom(x: *GpuPass, start: u64, n: u32, pick: u64, nw: u32) !void {
        const a = .{ .start = start, .n = n };
        const d = x.dev.?;
        const r = x.f.runner;
        // every row of the window (Python ingests the whole window: rows past the commit lie past every later pass's
        // context and the pass's own block rows overwrite them; the commit's ingest of the kept rows is the same bytes)
        try x.ingestAsync(a.start, 0, a.n);
        // the statics as fillStatics stages them (placeholders), then every position-dependent one from the pick
        const st = x.st;
        const sv = try d.statics(0, @intCast(st.len64()), @intCast(st.len32()));
        x.fillStatics(0, a.start, sv[0], sv[1]);
        const s64 = r.addressOf("w.ds.i64") orelse return error.Unbound;
        const s32 = r.addressOf("w.ds.i32") orelse return error.Unbound;
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = s64, .len = 8 * sv[0].len }, 0, std.mem.sliceAsBytes(sv[0]), r.stream.handle);
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = s32, .len = 4 * sv[1].len }, 0, std.mem.sliceAsBytes(sv[1]), r.stream.handle);
        try r.kernels.others(r.stream).dsStage(pick, nw, s64, s32, @intCast(st.n), @intCast(st.window), @intCast(emit.ringRows(x.f.cfg)), x.valid, x.shape.noise, stageOffsets(st));
        r.glue = .{ .ctx = x, .run = glueFn };
        const extra = [_]u64{ if (x.send) |b| b.ptr else 0, if (x.send) |b| b.len else 0 };
        try d.issueReplay(0, dsd.fingerprint(r, d, &extra), .{ .ctx = x, .run = passBody });
        try d.mark();
        try d.settle(); // counted here on every rank (rank 0's collect does not count it again)
    }

    /// ds_stage's offsets: positions, tokens, counts, lo, hi, anchor (ids and start at 0).
    fn stageOffsets(st: emit.Statics) [6]i64 {
        return .{ st.positions(), st.tokens(), st.counts(), st.lo(), st.hi(), st.anchor() };
    }

    /// A device-launched pass nobody took (the round did not speculate after all): its results are dropped; the next
    /// pass follows it on the stream.
    fn dropDevice(x: *GpuPass) void {
        if (x.dl != null) x.stats.device_dropped += 1;
        x.dl = null;
        x.armed = null;
    }

    fn self(p: *anyopaque) *GpuPass {
        return @ptrCast(@alignCast(p));
    }

    /// A new request's draft context, on every rank: the followers' `valid` decides their half of the candidates, so a
    /// follower that kept the last request's (the Spark window: the same request drafted 207 vs 199) drafts differently.
    fn reset(p: *anyopaque, slot: u32) void {
        _ = slot;
        const x = self(p);
        x.dropDevice();
        if (x.begun != null) std.log.err("dspark: a reset while a speculative pass is begun", .{});
        const op = fwd.Forward.dsResetOp();
        x.f.sendDrafter(&op) catch |e| std.log.err("dspark: the draft reset did not reach the followers ({t})", .{e});
        x.valid = 0;
    }

    fn followReset(p: *anyopaque) void {
        self(p).valid = 0;
        self(p).armed = null;
    }

    fn heads(p: *anyopaque) ?dspark.Markov {
        return self(p).markov;
    }

    /// `rows`: the taps' rows (the window's or prompt's row indices, lanes already kept the last `window`), landing at
    /// positions start .. ; a chain's run is contiguous.
    fn ingest(p: *anyopaque, slot: u32, start: u64, taps: iface.Taps, rows: []const u32) anyerror!void {
        const x = self(p);
        if (slot != 0) return error.Slots;
        if (x.begun != null) return error.PassBegun;
        if (rows.len == 0) return;
        for (rows, 0..) |r, i| if (r != rows[0] + i) return error.NotAPrefix;
        // the device-launched pass already ingested its window's rows (afterPick): the same rows at the same positions
        if (x.dl) |dl| if (start == dl.start and rows[0] == 0 and rows.len <= dl.rows) return;
        const n: u32 = @intCast(rows.len);
        const ops = fwd.Forward.dsIngestOp(start, rows[0], n, taps.rows);
        try x.f.sendDrafter(&ops);
        try x.doIngest(start, rows[0], n, taps.rows);
    }

    fn followIngest(p: *anyopaque, start: u64, row0: u32, n: u32, avail: u32) anyerror!void {
        return self(p).doIngest(start, row0, n, avail);
    }

    /// An ingest of rows on the device (row0 + n <= the taps' rows) without draining the stream first: the position from
    /// pageable memory (copied out before the call returns), then the launches in stream order.
    fn ingestAsync(x: *GpuPass, start: u64, row0: u32, n: u32) !void {
        const r = x.f.runner;
        const pos: i32 = @intCast(start);
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.pos") orelse return error.Unbound, .len = 4 }, 0, std.mem.asBytes(&pos), r.stream.handle);
        var la = std.heap.ArenaAllocator.init(x.gpa);
        defer la.deinit();
        const cs = try emit.emitIngest(la.allocator(), x.f.cfg, x.f.widths, x.f.opts, n, row0);
        r.glue = .{ .ctx = x, .run = glueFn };
        try r.window(cs);
    }

    fn doIngest(x: *GpuPass, start: u64, row0: u32, n: u32, avail: u32) !void {
        if (n == 0) return;
        if (row0 + n > avail or n > x.ingest_rows) {
            // rows not on the device (a resumed slot's prompt): the context restarts after them
            x.valid = start + n;
            return;
        }
        const r = x.f.runner;
        try r.stream.synchronize();
        const pos: i32 = @intCast(start);
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.pos") orelse return error.Unbound, .len = 4 }, 0, std.mem.asBytes(&pos), r.stream.handle);
        var la = std.heap.ArenaAllocator.init(x.gpa);
        defer la.deinit();
        const cs = try emit.emitIngest(la.allocator(), x.f.cfg, x.f.widths, x.f.opts, n, row0);
        r.glue = .{ .ctx = x, .run = glueFn };
        try r.window(cs);
    }

    fn propose(p: *anyopaque, asks: []const iface.Ask, out: []iface.Proposal) anyerror!void {
        const x = self(p);
        if (asks.len != 1) return error.Slots;
        if (x.begun != null) return error.PassBegun;
        x.dropDevice();
        const a = asks[0];
        const ops = fwd.Forward.dsProposeOp(a.anchor, a.start);
        try x.f.sendDrafter(&ops);
        try x.runPass(a.anchor, a.start);
        return x.finish(a, out);
    }

    /// The speculative pass (draft/spec.zig): the followers get the same op_ds_propose and run the whole pass; this
    /// rank launches the pass up to its host step and returns, so the GPU drafts while the lanes commit. Until
    /// `collect`, only work without collectives may follow on the stream (the target's keep): the host step's gather
    /// pairs with the followers' next one.
    fn begin(p: *anyopaque, asks: []const iface.Ask) anyerror!void {
        const x = self(p);
        if (asks.len != 1) return error.Slots;
        if (x.begun != null) return error.PassBegun;
        const a = asks[0];
        if (x.dl) |dl| {
            // the pass already ran from the window's device pick: it is this ask's when the device's (bonus, next
            // position) are the host's (the same merged picks), else it is dropped and the pass runs as below
            const h = dl.host;
            x.dl = null;
            if (dl.wait) try x.staged.?.synchronize(); // the accept's copy (not the pass)
            if (h[dl.n + 1] == a.anchor and h[dl.n + 2] == @as(i64, @intCast(a.start))) {
                x.stats.device_starts += 1;
                x.begun = a;
                x.begun_dev = true;
                return;
            }
            x.stats.device_dropped += 1;
            std.log.warn("dspark: the device pick's pass ({d} at {d}) is not the ask ({d} at {d}); run again", .{ h[dl.n + 1], h[dl.n + 2], a.anchor, a.start });
        }
        const ops = fwd.Forward.dsProposeOp(a.anchor, a.start);
        try x.f.sendDrafter(&ops);
        if (x.devOn()) {
            // the whole pass, its candidates and their copy back: nothing in it waits for the host
            try x.devLaunch(a.anchor, a.start);
            x.begun = a;
            return;
        }
        try x.stagePass(a.anchor, a.start);
        try x.f.runner.window(x.pass_calls[0..x.head]);
        x.begun = a;
    }

    fn collect(p: *anyopaque, out: []iface.Proposal) anyerror!void {
        const x = self(p);
        const a = x.begun orelse return error.NoPassBegun;
        x.begun = null;
        if (out.len != 1) return error.Slots;
        if (x.begun_dev) {
            x.begun_dev = false;
            try x.devResults();
            return x.finish(a, out);
        }
        if (x.devOn()) {
            try x.devCollect(true);
            return x.finish(a, out);
        }
        const r = x.f.runner;
        r.glue = .{ .ctx = x, .run = glueFn };
        try r.window(x.pass_calls[x.head..]);
        return x.finish(a, out);
    }

    /// The host's chain over the pass's gathered candidates (the last `runPass` or `collect`).
    fn finish(x: *GpuPass, a: iface.Ask, out: []iface.Proposal) !void {
        // the chain (Python's chain_host): greedy at T = 0, keyed sampling at the ask's parameters
        const c: usize = x.f.comm.world() * x.k;
        var scratch = std.heap.ArenaAllocator.init(x.gpa);
        defer scratch.deinit();
        const tc = ph.start(x.f.phases, .@"draft.chain");
        defer tc.stop();
        try dspark.chain(scratch.allocator(), x.markov, x.cand, x.cval, c, a.anchor, a.params, if (x.conf != null) x.hid else null, out[0].drafts, out[0].conf);
        // trees (lanes' siblings): the gathered candidates and base logits [block][world x k], Python's last_tree
        if (out[0].cand) |cd| {
            if (cd.len != x.cand.len) return error.CandidateWidth;
            @memcpy(cd, x.cand);
        }
        if (out[0].base) |b| {
            if (b.len != x.cval.len) return error.CandidateWidth;
            @memcpy(b, x.cval);
        }
    }

    fn followPropose(p: *anyopaque, anchor: u32, start: u64) anyerror!void {
        return self(p).runPass(anchor, start);
    }

    /// The pass on this rank: statics staged, the launches, the host's candidates (every rank: the gather pairs up).
    fn runPass(x: *GpuPass, anchor: u32, start: u64) !void {
        if (x.devOn()) {
            try x.devLaunch(anchor, start);
            return x.devCollect(x.f.comm.rank() == 0 or x.f.link == null);
        }
        try x.stagePass(anchor, start);
        try x.f.runner.window(x.pass_calls);
    }

    fn devOn(x: *const GpuPass) bool {
        const d = x.dev orelse return false;
        return d.on();
    }

    /// The device path: the statics from pinned staging (no sync), the pass (its key's graph once the checks passed),
    /// the event the host waits on.
    fn devLaunch(x: *GpuPass, anchor: u32, start: u64) !void {
        const d = x.dev.?;
        const st = x.st;
        const sv = try d.statics(0, @intCast(st.len64()), @intCast(st.len32()));
        x.fillStatics(anchor, start, sv[0], sv[1]);
        const r = x.f.runner;
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.i64") orelse return error.Unbound, .len = 8 * sv[0].len }, 0, std.mem.sliceAsBytes(sv[0]), r.stream.handle);
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.i32") orelse return error.Unbound, .len = 4 * sv[1].len }, 0, std.mem.sliceAsBytes(sv[1]), r.stream.handle);
        r.glue = .{ .ctx = x, .run = glueFn };
        const extra = [_]u64{ if (x.send) |b| b.ptr else 0, if (x.send) |b| b.len else 0 };
        try d.issue(0, dsd.fingerprint(r, d, &extra), .{ .ctx = x, .run = passBody });
        try d.mark();
    }

    fn passBody(ctx: *anyopaque, stream: cuda.Stream) anyerror!void {
        const x: *GpuPass = @ptrCast(@alignCast(ctx));
        _ = stream; // the runner's stream
        try x.f.runner.window(x.pass_calls);
    }

    /// The device pass's results on the host (`unpack`: rank 0's chain reads them), a check pass's comparison, and
    /// the ranks' verdict after the checks.
    fn devCollect(x: *GpuPass, unpack: bool) !void {
        const d = x.dev.?;
        {
            const tw = ph.start(x.f.phases, .@"draft.wait");
            defer tw.stop();
            try d.wait();
        }
        if (d.checked < dsd.check_passes) try d.check(x.last_logits, x.last_v, @intCast(x.st.n), 0);
        if (unpack) try d.unpack(@intCast(x.st.n), 0, x.f.cfg.vocab, x.cand, x.cval, if (x.conf != null) x.hid else null);
        try d.settle();
    }

    /// A device-launched pass's results (rank 0): waited for and unpacked; it was counted at its launch.
    fn devResults(x: *GpuPass) !void {
        const d = x.dev.?;
        {
            const tw = ph.start(x.f.phases, .@"draft.wait");
            defer tw.stop();
            try d.wait();
        }
        try d.unpack(@intCast(x.st.n), 0, x.f.cfg.vocab, x.cand, x.cval, if (x.conf != null) x.hid else null);
    }

    /// The pass's statics (Python's DraftInputs) on the device, and its glue in place.
    fn stagePass(x: *GpuPass, anchor: u32, start: u64) !void {
        x.fillStatics(anchor, start, x.h64, x.h32);
        const r = x.f.runner;
        try r.stream.synchronize();
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.i64") orelse return error.Unbound, .len = 8 * x.h64.len }, 0, std.mem.sliceAsBytes(x.h64), r.stream.handle);
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.i32") orelse return error.Unbound, .len = 4 * x.h32.len }, 0, std.mem.sliceAsBytes(x.h32), r.stream.handle);
        r.glue = .{ .ctx = x, .run = glueFn };
    }

    /// The statics (Python's DraftInputs, one slot) into `h64` / `h32`.
    fn fillStatics(x: *GpuPass, anchor: u32, start: u64, h64: []i64, h32: []i32) void {
        return rows_mod.stageOne(x.st, @intCast(emit.ringRows(x.f.cfg)), x.shape.noise, x.valid, anchor, start, h64, h32);
    }

    fn device(x: *GpuPass, slot: *?cuda.DeviceBuffer, bytes: usize) !u64 {
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

    fn glueFn(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
        const x: *GpuPass = @ptrCast(@alignCast(ctx));
        const step = c.name["glue.".len..];
        const s = r.stream.handle;
        const k = r.kernels.others(r.stream);
        const comm = x.f.comm;
        const D: usize = x.f.cfg.hidden;
        const Wd: usize = comm.world();
        if (std.mem.eql(u8, step, "ds_embed")) {
            // the block's ids (staged) through this rank's vocabulary rows, their all-sum, x hc_mult into the streams
            const n: usize = @intCast(x.st.n);
            const V: usize = x.f.cfg.vocab / Wd;
            const embed = r.weights.get("embed") orelse return error.MissingWeight;
            const rows = try x.device(&x.send, 2 * n * D);
            try k.embedSend(try r.tensorAddr(c.args[2].arg.t), n, embed.ptr, @intCast(comm.rank() * V), V, D, rows);
            const recv = try r.tensorAddr(c.args[1].arg.t);
            try comm.allGather(rows, recv, n * D, .bf16, s);
            return k.embedSum(recv, Wd, n, D, try r.tensorAddr(c.args[0].arg.t));
        }
        if (std.mem.eql(u8, step, "exchange_f32")) {
            const part = c.args[0].arg.t;
            const n = numel(part);
            const half = try x.device(&x.send, 2 * n);
            try k.castBf16(try r.tensorAddr(part), half, n);
            return comm.allGather(half, try r.tensorAddr(c.args[1].arg.t), n, .bf16, s);
        }
        if (std.mem.eql(u8, step, "widen")) // q.to(fp32): exact
            return k.widenF32(try r.tensorAddr(c.args[0].arg.t), try r.tensorAddr(c.args[1].arg.t), numel(c.args[0].arg.t));
        if (std.mem.eql(u8, step, "ds_out")) {
            if (x.dev) |dv| if (dv.on()) {
                x.last_logits = try r.tensorAddr(c.args[0].arg.t);
                x.last_v = @intCast(c.args[0].arg.t.shape[1]);
                return dv.launch(r, c.args[0].arg.t, c.args[1].arg.t, @intCast(x.st.n), 0);
            };
            return x.hostStep(r, c.args[0].arg.t, c.args[1].arg.t);
        }
        return error.GlueNotBuilt;
    }

    /// The host's step after the head: this rank's best k columns a row (as token ids), every rank's lists gathered in
    /// rank order (value bits then ids, gather_cols' layout), and the head hidden for the confidences.
    fn hostStep(x: *GpuPass, r: *run.Runner, logits_t: calls.Tensor, hid_t: calls.Tensor) !void {
        const th = ph.start(x.f.phases, .@"draft.host");
        defer th.stop();
        const comm = x.f.comm;
        const n: usize = @intCast(x.st.n);
        const V: usize = @intCast(logits_t.shape[1]);
        const D: usize = x.f.cfg.hidden;
        const k: usize = x.k;
        const Wd: usize = comm.world();
        const gpa = x.gpa;
        const logits = try gpa.alloc(f32, n * V);
        defer gpa.free(logits);
        const hid16 = try gpa.alloc(u16, n * D);
        defer gpa.free(hid16);
        try r.stream.synchronize();
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = try r.tensorAddr(logits_t), .len = 4 * logits.len }, 0, std.mem.sliceAsBytes(logits));
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = try r.tensorAddr(hid_t), .len = 2 * hid16.len }, 0, std.mem.sliceAsBytes(hid16));
        for (hid16, x.hid) |b, *h| h.* = dspark.bf16(b);
        // packed [n][2k] fp32: values, then ids as bits
        const packed_ = try gpa.alloc(f32, n * 2 * k);
        defer gpa.free(packed_);
        const ids = try gpa.alloc(i32, k);
        defer gpa.free(ids);
        const id0: u32 = @intCast(comm.rank() * V);
        for (0..n) |i| {
            const row = packed_[i * 2 * k ..][0 .. 2 * k];
            try dspark.candidates(gpa, logits[i * V ..][0..V], id0, ids, row[0..k]);
            for (ids, row[k..]) |id, *o| o.* = @bitCast(id);
        }
        const bytes = 4 * packed_.len;
        const src = try x.device(&x.send, bytes);
        const dst = try x.device(&x.gathered, bytes * Wd);
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = src, .len = bytes }, 0, std.mem.sliceAsBytes(packed_), r.stream.handle);
        try comm.allGather(src, dst, packed_.len, .f32, r.stream.handle); // bits only (ids as fp32 bit patterns)
        try r.stream.synchronize();
        const all = try gpa.alloc(f32, packed_.len * Wd);
        defer gpa.free(all);
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = dst, .len = 4 * all.len }, 0, std.mem.sliceAsBytes(all));
        // [W][n][2k] -> per row: rank 0's k, rank 1's k, ...
        const c = Wd * k;
        for (0..Wd) |w| for (0..n) |i| {
            const src_row = all[(w * n + i) * 2 * k ..][0 .. 2 * k];
            @memcpy(x.cval[i * c + w * k ..][0..k], src_row[0..k]);
            for (src_row[k..], x.cand[i * c + w * k ..][0..k]) |v, *o| o.* = @bitCast(v);
        };
        try checkCandidates(x.cand, x.f.cfg.vocab);
    }
};

/// The gathered candidates are token ids (or -1): anything else is a gather that read its source before the upload
/// landed, or a mis-paired collective. Named here on every rank, before the chain indexes the Markov tables with it.
pub fn checkCandidates(cand: []const i32, vocab: usize) error{BadCandidates}!void {
    for (cand, 0..) |id, i| if (id < -1 or id >= @as(i64, @intCast(vocab))) {
        std.log.err("dspark: gathered candidate {d} at {d} is not a token id", .{ id, i });
        return error.BadCandidates;
    };
}
