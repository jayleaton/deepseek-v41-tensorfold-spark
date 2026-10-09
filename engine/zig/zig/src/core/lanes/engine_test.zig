//! The round loop on the fake target: drafted == one-token rounds, shared rounds == solo, greedy and sampled.
const std = @import("std");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const Sampling = @import("sampling.zig").Sampling;

const gpa = std.testing.allocator;

const Case = struct {
    prompt: []const u32,
    max_new: u32 = 40,
    sampling: ?Sampling = null,
    drafts: bool = true,
    think_budget: u32 = 0,
};

fn model() !Config {
    var costs: [16]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    return Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true }, 16, 15);
}

/// Every case's emitted tokens, the cases admitted together and stepped until done.
fn run(cases: []const Case) ![][]u32 {
    var cfg = try model();
    defer cfg.deinit(gpa);
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, cases.len);
    defer gpa.free(proposers);
    for (cases, streams, proposers) |c, *s, *p| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{96}, .sampling = c.sampling, .drafts = c.drafts, .proposer = p.proposer(), .think_budget = c.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91 });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return out;
}

fn free(runs: [][]u32) void {
    for (runs) |r| gpa.free(r);
    gpa.free(runs);
}

const GuardCase = struct {
    prompt: []const u32,
    max_new: u32 = 400,
    drafts: bool = true,
    think_budget: u32 = 0,
    loop_guard: bool = true,
    think_open: ?bool = null,
    cycle_after: usize = 70,
    pattern: []const u32 = &.{ 11, 12, 13 },
    answer_cycles: bool = false,
};

const GuardOut = struct {
    emitted: []u32,
    reason: sm.Reason,
    period: ?u32,
    finished: bool,
};

fn runGuard(cases: []const GuardCase, step_limit: usize) ![]GuardOut {
    var cfg = try model();
    defer cfg.deinit(gpa);
    const c = cases[0];
    var target: fake.Fake = .{ .gpa = gpa, .cycle_after = c.cycle_after, .probe_prompt = c.prompt.len, .pattern = c.pattern, .answer_cycles = c.answer_cycles };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    const proposers = try gpa.alloc(SuffixLookup, cases.len);
    defer gpa.free(proposers);
    for (cases, streams, proposers) |item, *s, *p| {
        p.* = try SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try sm.Stream.init(gpa, .{ .id = "guard", .prompt = item.prompt, .max_new = item.max_new, .eos = &.{96}, .drafts = item.drafts, .proposer = p.proposer(), .think_budget = item.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .think_open = item.think_open, .loop_guard = item.loop_guard });
    }
    defer for (streams, proposers) |*s, *p| {
        s.deinit(gpa);
        p.deinit();
    };
    for (streams) |*s| try engine.addStream(s);
    var steps: usize = 0;
    while (engine.activeCount() > 0 and steps < step_limit) : (steps += 1) try engine.step();
    const out = try gpa.alloc(GuardOut, cases.len);
    for (out, streams) |*result, *s| result.* = .{ .emitted = try gpa.dupe(u32, s.emitted()), .reason = s.reason, .period = s.loop_period, .finished = s.finished };
    return out;
}

fn freeGuard(runs: []GuardOut) void {
    for (runs) |run_result| gpa.free(run_result.emitted);
    gpa.free(runs);
}

const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 };
const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9 };

test "drafted rounds commit the one-token decode, greedy and sampled" {
    for ([_]?Sampling{ null, .{ .seed = 5, .temperature = 0.7, .top_k = 0, .top_p = 0.95 } }) |s| {
        const drafted = try run(&.{.{ .prompt = &p1, .sampling = s }});
        defer free(drafted);
        const plain = try run(&.{.{ .prompt = &p1, .sampling = s, .drafts = false }});
        defer free(plain);
        try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
        // and the fake target's own decode
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, &p1);
        for (drafted[0]) |t| {
            try std.testing.expectEqual(fake.next(history.items, s, history.items.len), t);
            try history.append(gpa, t);
        }
    }
}

test "shared rounds commit what each stream commits alone" {
    const sampled: Sampling = .{ .seed = 9, .temperature = 1.0, .top_k = 0, .top_p = 0.9 };
    const together = try run(&.{ .{ .prompt = &p1 }, .{ .prompt = &p2, .sampling = sampled, .max_new = 30 } });
    defer free(together);
    const one = try run(&.{.{ .prompt = &p1 }});
    defer free(one);
    const two = try run(&.{.{ .prompt = &p2, .sampling = sampled, .max_new = 30 }});
    defer free(two);
    try std.testing.expectEqualSlices(u32, one[0], together[0]);
    try std.testing.expectEqualSlices(u32, two[0], together[1]);
}

test "the thinking budget's forced close is the same drafted, plain and shared" {
    const drafted = try run(&.{.{ .prompt = &p2, .think_budget = 9 }});
    defer free(drafted);
    const plain = try run(&.{.{ .prompt = &p2, .think_budget = 9, .drafts = false }});
    defer free(plain);
    const shared = try run(&.{ .{ .prompt = &p2, .think_budget = 9 }, .{ .prompt = &p1, .drafts = false } });
    defer free(shared);
    try std.testing.expectEqualSlices(u32, plain[0], drafted[0]);
    try std.testing.expectEqualSlices(u32, plain[0], shared[0]);
    try std.testing.expectEqual(@as(u32, 90), drafted[0][8]);
}

test "the loop guard matches drafted and plain, reports its period, and answers after close" {
    const plain = try runGuard(&.{.{ .prompt = &p2, .drafts = false }}, 5000);
    defer freeGuard(plain);
    const drafted = try runGuard(&.{.{ .prompt = &p2 }}, 5000);
    defer freeGuard(drafted);
    try std.testing.expectEqualSlices(u32, plain[0].emitted, drafted[0].emitted);
    try std.testing.expectEqual(@as(?u32, 3), drafted[0].period);
    try std.testing.expect(plain[0].finished and plain[0].reason == .stop);
    try std.testing.expectEqualSlices(u32, &.{ 90, 91, 92, 40, 41, 42, 96 }, plain[0].emitted[329..]);

    const off = try runGuard(&.{.{ .prompt = &p2, .loop_guard = false, .think_open = true }}, 5000);
    defer freeGuard(off);
    try std.testing.expect(off[0].finished and off[0].reason == .length and off[0].emitted.len == 400);
}

test "a cycle in the answer does not refire or reclose" {
    const plain = try runGuard(&.{.{ .prompt = &p2, .drafts = false, .max_new = 800, .answer_cycles = true }}, 5000);
    defer freeGuard(plain);
    const drafted = try runGuard(&.{.{ .prompt = &p2, .max_new = 800, .answer_cycles = true }}, 5000);
    defer freeGuard(drafted);
    try std.testing.expectEqualSlices(u32, plain[0].emitted, drafted[0].emitted);
    try std.testing.expectEqual(@as(?u32, 3), plain[0].period);
    try std.testing.expect(plain[0].finished and plain[0].reason == .length);
    var close_count: usize = 0;
    for (plain[0].emitted) |token| close_count += @intFromBool(token == 90);
    try std.testing.expectEqual(@as(usize, 1), close_count);
}

/// The cases' prompts in pieces (Engine.beginStream / piece / finishStream), case i opened before round 2 i, the
/// others decoding meanwhile; `join`: Config.join_tail (the fake counts its joins).
fn runPieces(cases: []const Case, join: bool, joins: *usize) ![][]u32 {
    var cfg = try model();
    defer cfg.deinit(gpa);
    cfg.join_tail = join;
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.joinBackend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    for (cases, streams) |c, *s| s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{96}, .sampling = c.sampling, .drafts = c.drafts, .think_budget = c.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91 });
    defer for (streams) |*s| s.deinit(gpa);
    var opened: usize = 0;
    var round: usize = 0;
    while (opened < streams.len or engine.activeCount() > 0) : (round += 1) {
        if (opened < streams.len and round == 2 * opened) {
            const s = &streams[opened];
            _ = try engine.beginStream(s);
            if (s.prompt_len > 1) try engine.piece(s, 0, s.prompt_len - 1, false);
            try engine.finishStream(s);
            opened += 1;
        }
        if (engine.activeCount() > 0) try engine.step();
    }
    joins.* = target.joins;
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return out;
}

test "a prompt's last token joins its first round's shared window: the same tokens, greedy and sampled" {
    const cases = [_]Case{
        .{ .prompt = &.{ 3, 1, 4, 1, 5, 9, 2, 6 }, .max_new = 37 },
        .{ .prompt = &.{ 2, 7, 1, 8, 2, 8 }, .max_new = 29, .sampling = .{ .seed = 5, .temperature = 0.8, .top_k = 7 } },
        .{ .prompt = &.{4}, .max_new = 21 },
        .{ .prompt = &.{ 1, 1, 2, 3, 5 }, .max_new = 25, .drafts = false },
        .{ .prompt = &.{ 6, 6, 6, 1 }, .max_new = 30, .think_budget = 1 },
    };
    var j0: usize = 0;
    var j1: usize = 0;
    const off = try runPieces(&cases, false, &j0);
    defer free(off);
    const on = try runPieces(&cases, true, &j1);
    defer free(on);
    try std.testing.expectEqual(@as(usize, 0), j0);
    try std.testing.expectEqual(@as(usize, 3), j1); // not the undrafted stream nor the budget cutting the first token
    for (off, on) |a, b| try std.testing.expectEqualSlices(u32, a, b);
    // and the whole prompts' tokens
    const whole = try run(&cases);
    defer free(whole);
    for (whole, on) |a, b| try std.testing.expectEqualSlices(u32, a, b);
}

/// Every case's prompt in one piece, all finished in the same round (`together`: Engine.finishStreams, one first-draft
/// call for the joined ones; else finishStream a prompt), then rounds until done; the fake's draft counters.
fn runFinals(cases: []const Case, together: bool, calls: *usize, most: *usize) ![][]u32 {
    var cfg = try model();
    defer cfg.deinit(gpa);
    cfg.join_tail = true;
    cfg.join_drafts = together;
    var target: fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(gpa, &cfg, target.joinBackend(), clock.clock());
    defer engine.deinit();
    const streams = try gpa.alloc(sm.Stream, cases.len);
    defer gpa.free(streams);
    for (cases, streams) |c, *s| s.* = try sm.Stream.init(gpa, .{ .id = "s", .prompt = c.prompt, .max_new = c.max_new, .eos = &.{96}, .sampling = c.sampling, .drafts = c.drafts, .think_budget = c.think_budget, .think_close = &.{ 90, 91, 92 }, .think_end = 91 });
    defer for (streams) |*s| s.deinit(gpa);
    const ptrs = try gpa.alloc(*sm.Stream, cases.len);
    defer gpa.free(ptrs);
    for (streams, ptrs) |*s, *p| {
        _ = try engine.beginStream(s);
        if (s.prompt_len > 1) try engine.piece(s, 0, s.prompt_len - 1, false);
        p.* = s;
    }
    // the finish phase alone: its draft calls and their largest
    target.drafts_calls = 0;
    target.drafts_most = 0;
    if (together) {
        const errs = try gpa.alloc(?anyerror, cases.len);
        defer gpa.free(errs);
        _ = try engine.finishStreams(ptrs, errs);
        for (errs) |x| if (x) |e| return e;
    } else for (ptrs) |s| try engine.finishStream(s);
    calls.* = target.drafts_calls;
    most.* = target.drafts_most;
    while (engine.activeCount() > 0) try engine.step();
    const out = try gpa.alloc([]u32, cases.len);
    for (out, streams) |*o, *s| o.* = try gpa.dupe(u32, s.emitted());
    return out;
}

test "a round's finished prompts get their first drafts in one call (Python's _windows), the same tokens" {
    const cases = [_]Case{
        .{ .prompt = &.{ 3, 1, 4, 1, 5, 9, 2, 6 }, .max_new = 37 },
        .{ .prompt = &.{ 2, 7, 1, 8, 2, 8 }, .max_new = 29, .sampling = .{ .seed = 5, .temperature = 0.8, .top_k = 7 } },
        .{ .prompt = &.{4}, .max_new = 21 },
        .{ .prompt = &.{ 1, 1, 2, 3, 5 }, .max_new = 25, .drafts = false },
        .{ .prompt = &.{ 6, 6, 6, 1 }, .max_new = 30, .think_budget = 1 },
    };
    var c1: usize = 0;
    var m1: usize = 0;
    var c0: usize = 0;
    var m0: usize = 0;
    const together = try runFinals(&cases, true, &c1, &m1);
    defer free(together);
    const apart = try runFinals(&cases, false, &c0, &m0);
    defer free(apart);
    for (apart, together) |a, b| try std.testing.expectEqualSlices(u32, a, b);
    const whole = try run(&cases);
    defer free(whole);
    for (whole, together) |a, b| try std.testing.expectEqualSlices(u32, a, b);
    // the three joined prompts in one first-draft call; the budget-cut first token keeps its own window and draft
    // (`opened`), the undrafted prompt none: 2 calls, against 4 a prompt at a time
    try std.testing.expectEqual(@as(usize, 2), c1);
    try std.testing.expectEqual(@as(usize, 3), m1);
    try std.testing.expectEqual(@as(usize, 4), c0);
    try std.testing.expectEqual(@as(usize, 1), m0);
}
