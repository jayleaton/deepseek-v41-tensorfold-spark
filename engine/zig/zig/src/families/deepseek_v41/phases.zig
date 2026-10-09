//! TF_DSV41_PHASES: where every millisecond of a serving round goes, by Python phases.py's names, so a Zig report and
//! Python's (m2bench's `phases`, TF_DSV41_PHASES=1 / sync) line up name by name (tools/zig/dsv41_perf/gap.py).
//!
//! - `0` (default): off; a timer is one null check.
//! - `1`: host wall time of each phase (what the round thread spends, its waits for the GPU included).
//! - `sync`: the stream synchronized at both ends of every phase, so each phase owns its GPU work and the top-level
//!   phases add up to the round; it removes the overlap it measures (attribution, not throughput).
//!
//! Phases nest: `window` > `forward` > `graph.stage` (> `engram.wait`) / `graph.replay` / `graph.capture`, and
//! `window` > `candidates` (the choices' reads). Top level: `prefill`, `draft` (ingest), `draft.pass`, `window`,
//! `commit`. `rounds` counts verify windows and `tokens` the committed tokens, for per-round and per-token figures.
//! `host`: the time between one top-level phase's end and the next one's start (the lanes' own work, the token events,
//! the plan link, everything no phase covers), so the top-level phases plus `host` add up to the decode rounds; a gap
//! of `idle_ns` or more is the engine waiting for a request and is not counted.
//! TF_DSV41_PHASES_OUT=<file>: the report as JSON (Python's shape), rewritten every 64 rounds and at close.

const std = @import("std");

pub const Name = enum {
    prefill,
    draft,
    @"draft.pass",
    window,
    forward,
    @"graph.stage",
    @"graph.replay",
    @"graph.capture",
    @"engram.wait",
    prefetch,
    candidates,
    commit,
    /// the speculative pass's launch after a window (TF_DSV41_SPEC_DRAFT; its collect is `draft.pass`)
    spec,
    /// host time between top-level phases (not a phase of Python's: its report has no name for it)
    host,
    /// inside `draft.pass` (dspark_dev.zig, TF_DSV41_DRAFT_GRAPHS): the host waiting for the device pass and its copy
    @"draft.wait",
    /// inside `draft.pass`: the host's chain (Markov bias, choice, confidences) over the gathered candidates
    @"draft.chain",
    /// inside `draft.pass`, the host step (DRAFT_GRAPHS off): the logits' download, the host candidates, their gather
    @"draft.host",
};

/// A gap between top-level phases this long or longer is idle (no live stream), not a round's host time.
pub const idle_ns: u64 = 50 * std.time.ns_per_ms;

const count = std.enums.values(Name).len;

pub const Mode = enum { off, wall, sync };

/// The stream the `sync` mode waits for.
pub const Sync = struct { ctx: *anyopaque, run: *const fn (ctx: *anyopaque) anyerror!void };

/// Counters beside the phases (an object without "ms", so gap.py's phase filter skips it): written into the report.
pub const Extra = struct { ctx: *anyopaque, write: *const fn (ctx: *anyopaque, w: *std.Io.Writer) anyerror!void };

pub const Phases = struct {
    mode: Mode,
    sync: ?Sync = null,
    extra: ?Extra = null,
    ns: [count]u64 = @splat(0),
    n: [count]u64 = @splat(0),
    rounds: u64 = 0,
    tokens: u64 = 0,
    t0: u64,
    path: ?[]const u8 = null,
    reported: u64 = 0,
    /// phases open now (a top-level phase starts at depth 0), and when the last top-level phase ended (0: none yet)
    depth: u32 = 0,
    last_end: u64 = 0,

    pub fn parseMode(v: []const u8) !Mode {
        if (std.mem.eql(u8, v, "0") or v.len == 0) return .off;
        if (std.mem.eql(u8, v, "1")) return .wall;
        if (std.mem.eql(u8, v, "sync")) return .sync;
        return error.BadPhasesMode;
    }

    /// From TF_DSV41_PHASES / TF_DSV41_PHASES_OUT; null when off.
    pub fn fromEnv(gpa: std.mem.Allocator) !?*Phases {
        const mode = try parseMode(if (std.c.getenv("TF_DSV41_PHASES")) |v| std.mem.span(v) else "0");
        if (mode == .off) return null;
        const p = try gpa.create(Phases);
        p.* = .{ .mode = mode, .t0 = nowNs(), .path = if (std.c.getenv("TF_DSV41_PHASES_OUT")) |v| std.mem.span(v) else null };
        return p;
    }

    pub fn reset(p: *Phases) void {
        p.ns = @splat(0);
        p.n = @splat(0);
        p.rounds = 0;
        p.tokens = 0;
        p.t0 = nowNs();
        p.depth = 0;
        p.last_end = 0;
    }

    fn settle(p: *Phases) void {
        if (p.mode != .sync) return;
        const s = p.sync orelse return;
        s.run(s.ctx) catch {};
    }

    pub fn add(p: *Phases, name: Name, ns: u64) void {
        p.ns[@intFromEnum(name)] += ns;
        p.n[@intFromEnum(name)] += 1;
    }

    /// The report: Python's {name: {ms, n, ms_avg}} plus wall_ms, mode, rounds, tokens.
    pub fn write(p: *const Phases, w: *std.Io.Writer) !void {
        const wall = @as(f64, @floatFromInt(nowNs() - p.t0)) / 1e6;
        try w.print("{{\"mode\": \"{s}\", \"wall_ms\": {d:.2}, \"rounds\": {d}, \"tokens\": {d}", .{ if (p.mode == .sync) "sync" else "1", wall, p.rounds, p.tokens });
        for (0..count) |i| {
            if (p.n[i] == 0) continue;
            const ms = @as(f64, @floatFromInt(p.ns[i])) / 1e6;
            try w.print(", \"{s}\": {{\"ms\": {d:.2}, \"n\": {d}, \"ms_avg\": {d:.3}}}", .{ @tagName(@as(Name, @enumFromInt(i))), ms, p.n[i], ms / @as(f64, @floatFromInt(p.n[i])) });
        }
        if (p.extra) |x| try x.write(x.ctx, w);
        try w.writeAll("}\n");
    }

    /// The report into TF_DSV41_PHASES_OUT (when set).
    pub fn flush(p: *Phases, gpa: std.mem.Allocator, io: std.Io) void {
        const path = p.path orelse return;
        var w: std.Io.Writer.Allocating = .init(gpa);
        defer w.deinit();
        p.write(&w.writer) catch return;
        std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = w.written() }) catch |e| std.log.err("phases: {s} not written ({t})", .{ path, e });
        p.reported = p.rounds;
    }

    /// A verify window's round finished with `kept` tokens committed; the report is rewritten every 64 rounds.
    pub fn round(p: *Phases, gpa: std.mem.Allocator, io: std.Io, kept: usize) void {
        p.rounds += 1;
        p.tokens += kept;
        if (p.rounds - p.reported >= 64) p.flush(gpa, io);
    }
};

/// One phase's timer: `const t = phases.start(p, .window); defer t.stop();` (p null: nothing).
pub const Timer = struct {
    p: ?*Phases,
    name: Name,
    t0: u64,

    pub fn stop(t: Timer) void {
        const p = t.p orelse return;
        p.settle();
        const now = nowNs();
        p.add(t.name, now -| t.t0);
        p.depth -|= 1;
        if (p.depth == 0) p.last_end = now;
    }
};

pub fn start(p: ?*Phases, name: Name) Timer {
    const x = p orelse return .{ .p = null, .name = name, .t0 = 0 };
    if (x.depth == 0 and x.last_end != 0) {
        const gap = nowNs() -| x.last_end;
        if (gap < idle_ns) x.add(.host, gap);
    }
    x.depth += 1;
    x.settle();
    return .{ .p = x, .name = name, .t0 = nowNs() };
}

pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

test "phases: Python's names and report shape; sync mode syncs at both ends" {
    const Count = struct {
        n: u32 = 0,
        fn run(ctx: *anyopaque) anyerror!void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            c.n += 1;
        }
    };
    var c: Count = .{};
    var p: Phases = .{ .mode = .sync, .t0 = nowNs(), .sync = .{ .ctx = &c, .run = Count.run } };
    {
        const t = start(&p, .@"draft.pass");
        defer t.stop();
    }
    const off = start(null, .window);
    off.stop();
    try std.testing.expectEqual(@as(u32, 2), c.n);
    {
        // a second top-level phase: one `host` gap before it; the nested phase opens none
        const t = start(&p, .window);
        defer t.stop();
        const n = start(&p, .forward);
        n.stop();
    }
    try std.testing.expectEqual(@as(u64, 1), p.n[@intFromEnum(Name.host)]);
    p.last_end = nowNs() - idle_ns; // an idle gap: not counted
    start(&p, .commit).stop();
    try std.testing.expectEqual(@as(u64, 1), p.n[@intFromEnum(Name.host)]);
    p.add(.@"engram.wait", 1_500_000);
    p.add(.@"engram.wait", 500_000);
    p.rounds = 2;
    p.tokens = 7;
    var w: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer w.deinit();
    try p.write(&w.writer);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, w.written(), .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try std.testing.expectEqualStrings("sync", o.get("mode").?.string);
    try std.testing.expectEqual(@as(i64, 7), o.get("tokens").?.integer);
    const ew = o.get("engram.wait").?.object;
    try std.testing.expectEqual(@as(i64, 2), ew.get("n").?.integer);
    try std.testing.expectEqual(@as(f64, 2.0), ew.get("ms").?.float);
    try std.testing.expect(o.get("draft.pass") != null and o.get("window") != null);
    try std.testing.expect(o.get("host") != null); // the gap between the two top-level phases above
    try std.testing.expectEqual(Mode.off, try Phases.parseMode("0"));
    try std.testing.expectError(error.BadPhasesMode, Phases.parseMode("nvtx"));
}
