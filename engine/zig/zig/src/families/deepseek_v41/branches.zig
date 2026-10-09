//! TF_DSV41_BRANCHES (branches.py, prod: 1 with TF_DSV41_BRANCHES_PRIO=side): a decode window's attention sublayer as
//! two branches. The emitter (`block.Emitter.attention`) issues the sublayer in `branches.plan`'s order already (the
//! captures were taken with the knob on: e4's folded copies, the split q group); with the knob it also tags each call
//! with its stream and the plan's fork / join points (`calls.Call.side / fork / join`), and the runner (`run.zig`)
//! issues the tagged calls on a side stream, a glue step's launches included (the runner's stream is the side's while
//! it runs; the ext launchers take no scratch, so nothing else is shared).
//!
//! TF_DSV41_MHC_DEFER (mhc_cuda.defer, prod: 1 with R1) lives here too: R1's mHC coefficient launch goes to a stream
//! of its own, forked after its boundary, and the next mHC call (the coefficients' reader, mhc.await_coefs) joins it. The launches, their arguments and their order are unchanged, so the bits are
//! (every value is made by the same kernel from the same inputs; `hazards` checks the plan's stream ordering on the
//! real programs).
//!
//! Index layers: fork, side ix_wp (`_plain`), main x group (+ `_rms2`), fork, side [compressor] ix_wq_b rope_qi select,
//! main wq_b, L2PF o, RoPE q, SWA store, join, attention core. Other layers: x group, `_rms2`, fork, side SWA store
//! (R1: its kv_norm with it), main wq_b, L2PF o, RoPE q, join, attention core.
//!
//! Knobs: TF_DSV41_BRANCHES 0 | 1 (on, streams) | serial (e4 alone: the Zig emitter's order without a side stream, i.e.
//! off here); TF_DSV41_BRANCHES_ROWS (64: wider windows stay on one stream); TF_DSV41_BRANCHES_PRIO main | side |
//! equal (Python's: the side stream at the device's greatest priority for `side`; `main` raises the main stream, which
//! is the model's own here, so it runs as `equal` with a warning). As in Python, the graphs are instantiated without
//! node priorities, so the priority acts on eager windows only.

const std = @import("std");
const cuda = @import("cuda");
const calls = @import("calls.zig");
const run = @import("run.zig");
const fwd = @import("forward.zig");

const log = std.log.scoped(.dsv41);

pub const Prio = enum { main, side, equal };

pub const Settings = struct {
    on: bool = false,
    rows: i64 = 64,
    prio: Prio = .main,
};

fn get(name: [:0]const u8) ?[]const u8 {
    const v = std.mem.trim(u8, std.mem.span(std.c.getenv(name) orelse return null), " \t");
    return if (v.len == 0) null else v;
}

pub fn settings() !Settings {
    var s: Settings = .{};
    if (get("TF_DSV41_BRANCHES")) |raw| {
        var low: [16]u8 = undefined;
        if (raw.len > low.len) return error.BadBranches;
        const v = std.ascii.lowerString(&low, raw);
        const one = std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "on") or std.mem.eql(u8, v, "streams");
        const serial = std.mem.eql(u8, v, "serial") or std.mem.eql(u8, v, "e4");
        if (!one and !serial and !std.mem.eql(u8, v, "0") and !std.mem.eql(u8, v, "off")) {
            log.err("TF_DSV41_BRANCHES={s}: expected 0, 1 or serial", .{raw});
            return error.BadBranches;
        }
        s.on = one;
    }
    if (get("TF_DSV41_BRANCHES_ROWS")) |v| s.rows = std.fmt.parseInt(i64, v, 10) catch return error.BadBranches;
    if (s.rows < 1 or s.rows > 4096) return error.BadBranches;
    if (get("TF_DSV41_BRANCHES_PRIO")) |v| s.prio = std.meta.stringToEnum(Prio, v) orelse {
        log.err("TF_DSV41_BRANCHES_PRIO={s}: expected main, side or equal", .{v});
        return error.BadBranches;
    };
    return s;
}

/// The side streams and their events (the runner's `br` / `dfr` point here).
pub const Branches = struct {
    side: ?run.Runner.Side = null,
    deferred: ?run.Runner.Side = null,

    pub fn deinit(b: *Branches) void {
        for ([_]*?run.Runner.Side{ &b.side, &b.deferred }) |x| if (x.*) |*s| {
            s.stream.deinit();
            s.fork.deinit();
            s.join.deinit();
        };
    }
};

fn sideOf(d: *const cuda.Driver, prio: cuda.Stream.Priority) !run.Runner.Side {
    return .{ .stream = try cuda.Stream.initPriority(d, prio), .fork = try cuda.Event.init(d, false), .join = try cuda.Event.init(d, false) };
}

/// TF_DSV41_MHC_DEFER (with R1): on unless unset / 0.
pub fn mhcDefer() bool {
    const v = get("TF_DSV41_MHC_DEFER") orelse return false;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or std.ascii.eqlIgnoreCase(v, "false"));
}

/// With TF_DSV41_BRANCHES and / or TF_DSV41_MHC_DEFER (R1) on: the emitter's options, the side streams (before any
/// capture), the runner pointed at them. Null when both are off.
pub fn install(gpa: std.mem.Allocator, f: *fwd.Forward, rank: u32) !?*Branches {
    const s = try settings();
    const dfr = mhcDefer() and f.opts.r1;
    if (!s.on and !dfr) return null;
    const r = f.runner;
    const b = try gpa.create(Branches);
    errdefer gpa.destroy(b);
    b.* = .{};
    errdefer b.deinit();
    if (s.on) {
        b.side = try sideOf(r.d, if (s.prio == .side) .highest else .default);
        f.opts.branches = true;
        f.opts.branch_rows = s.rows;
        r.br = &b.side.?;
    }
    if (dfr) {
        b.deferred = try sideOf(r.d, .default); // mhc_cuda._side: torch's default priority
        f.opts.mhc_defer = true;
        r.dfr = &b.deferred.?;
    }
    if (rank == 0) {
        if (s.on and s.prio == .main) log.warn("branches: TF_DSV41_BRANCHES_PRIO=main raises the main stream, the model's own here: run as equal", .{});
        if (s.on) log.info("branches: the indexer / compressor on a side stream (priority {s}), windows <= {d} rows", .{ if (s.prio == .side) "highest" else "default", s.rows });
        if (dfr) log.info("mhc defer: R1's coefficient launches on their own stream, joined by the next mHC call", .{});
    }
    return b;
}

// -- the plan's ordering on a window program ----------------------------------------------------------------------

/// Roles both streams touch inside one fork region of `cs` (calls after a fork on the side, calls on main from the
/// fork until the join), with branches.check's ordering: a fork orders the side after everything main issued, a
/// join orders main after everything the side issued. Weights are left out (read only). Also checks the marks'
/// structure: a side call only inside an open fork, a join only after a fork, no fork open at the end.
pub fn shared(a: std.mem.Allocator, cs: []const calls.Call) ![]const []const u8 {
    var main_t: std.StringHashMapUnmanaged(void) = .empty;
    var side_t: std.StringHashMapUnmanaged(void) = .empty;
    var out: std.StringArrayHashMapUnmanaged(void) = .empty;
    var open = false;
    for (cs) |c| {
        if (c.join) {
            if (!open) return error.JoinWithoutFork;
            open = false;
            side_t.clearRetainingCapacity();
        }
        if (c.fork) {
            open = true;
            main_t.clearRetainingCapacity();
        }
        if (c.side and !open) return error.SideOutsideFork;
        const mine = if (c.side) &side_t else &main_t;
        const other = if (c.side) &main_t else &side_t;
        for (c.args) |x| try roles(a, x.arg, mine, other, &out);
    }
    if (open) return error.OpenFork;
    return out.keys();
}

/// Roles the deferred coefficient launches share with the main stream's calls before their join (MHC_DEFER), and the
/// marks' structure: a join only after a deferred launch, none open at the end.
pub fn sharedDeferred(a: std.mem.Allocator, cs: []const calls.Call) ![]const []const u8 {
    var main_t: std.StringHashMapUnmanaged(void) = .empty;
    var side_t: std.StringHashMapUnmanaged(void) = .empty;
    var out: std.StringArrayHashMapUnmanaged(void) = .empty;
    var open = false;
    for (cs) |c| {
        if (c.defer_join) {
            if (!open) return error.JoinWithoutFork;
            open = false;
            side_t.clearRetainingCapacity();
        }
        if (c.defer_side) {
            if (open) return error.DeferredTwice; // mhc: one pending coefficient set at a time
            open = true;
            main_t.clearRetainingCapacity();
        }
        const mine = if (c.defer_side) &side_t else &main_t;
        const other = if (c.defer_side) &main_t else &side_t;
        for (c.args) |x| try roles(a, x.arg, mine, other, &out);
    }
    if (open) return error.OpenFork;
    return out.keys();
}

fn roles(a: std.mem.Allocator, x: calls.Arg, mine: *std.StringHashMapUnmanaged(void), other: *std.StringHashMapUnmanaged(void), out: *std.StringArrayHashMapUnmanaged(void)) !void {
    switch (x) {
        .t, .opaque_table => |t| switch (t.role) {
            .buf => |b| {
                try mine.put(a, b, {});
                if (other.contains(b)) try out.put(a, b, {});
            },
            else => {},
        },
        .list => |items| for (items) |y| try roles(a, y, mine, other, out),
        else => {},
    }
}
