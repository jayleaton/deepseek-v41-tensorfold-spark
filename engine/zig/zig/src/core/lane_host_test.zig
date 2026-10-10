//! Host memory reporting and isolated request refusal through LaneHost.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Memory = api.Memory;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;

test "a lane host reports its backend's memory counts, and none without them" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{}, 1, 0);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    try std.testing.expect(host.engine().memory(false) == null);
    const Counts = struct {
        resets: u32 = 0,
        fn read(ctx: ?*anyopaque, reset_peak: bool) ?Memory {
            const c: *@This() = @ptrCast(@alignCast(ctx.?));
            if (reset_peak) c.resets += 1;
            return .{ .active = 5, .peak = if (reset_peak) 5 else 9 };
        }
    };
    var counts: Counts = .{};
    host.memory = .{ .ctx = &counts, .read = Counts.read };
    try std.testing.expectEqual(@as(u64, 9), host.engine().memory(false).?.peak);
    try std.testing.expectEqual(@as(u64, 5), host.engine().memory(true).?.peak);
    try std.testing.expectEqual(@as(u32, 1), counts.resets);
}

test "a request the backend refuses fails alone, in the backend's words" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .refuse_sampled = true };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const Words = struct {
        fn text(_: ?*anyopaque, err: anyerror) ?[]const u8 {
            return if (err == error.SamplingRefused) "send temperature 0" else null;
        }
    };
    host.explain = .{ .text = Words.text };
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        done: ?Reason = null,
        message: []const u8 = "",
        tokens: usize = 0,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens += t.len,
                .finished => |f| {
                    b.done = f.reason;
                    b.message = f.message;
                },
                else => {},
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var plain: Box = .{};
    var sampled: Box = .{};
    const greedy: Request = .{ .prompt = &prompt, .max_tokens = 64 };
    const keyed: Request = .{ .prompt = &prompt, .max_tokens = 64, .sampling = .{ .seed = 3, .temperature = 0.7, .top_k = 5 } };
    const e = host.engine();
    try e.submit(1, &greedy, .{ .ctx = &plain, .event = Box.event });
    try e.submit(2, &keyed, .{ .ctx = &sampled, .event = Box.event });
    try std.testing.expectEqual(Reason.failed, sampled.wait());
    try std.testing.expectEqualStrings("send temperature 0", sampled.message);
    try std.testing.expectEqual(Reason.length, plain.wait());
    try std.testing.expectEqual(@as(usize, 64), plain.tokens);
}

test "a failed piece fails its request alone: its final of the same round does not run, the host keeps serving" {
    // Spark 2026-10-09: a job dropped for a failed piece was still in the round's finals (the planner listed its prompt
    // as finished in the same round); `drop` freed it and runFinals read the freed job (segfault)
    const gpa = std.testing.allocator;
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true, .join_tail = true }, 16, 15);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .fail_pieces = 1 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.joinBackend(), clock.clock());
    defer core.deinit();
    // the host's jobs on the page allocator: a freed job's pages are unmapped, so reading one faults as on the Spark
    // (the testing allocator's freed bytes read as a finished stream and hide it)
    const pa = std.heap.page_allocator;
    var host = LaneHost.init(pa, std.testing.io, &core, .{ .lanes = 4 });
    // every admitted prompt's rows in one piece and its final in the same round (as prod's batcher does short prompts)
    const Plan = struct {
        const Seq = struct { key: usize, n: u64, done: bool = false };
        seqs: std.ArrayList(Seq) = .empty,
        fn self(ctx: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ctx));
        }
        fn beginAdmit(_: *anyopaque) void {}
        fn admit(ctx: *anyopaque, ask: api.Rounds.Ask, _: *[]const u8) api.Rounds.Verdict {
            self(ctx).seqs.append(gpa, .{ .key = ask.key, .n = ask.prompt.len }) catch return .refuse;
            return .admit;
        }
        fn admitted(_: *anyopaque) anyerror!void {}
        fn arm(_: *anyopaque, _: usize) void {}
        fn began(_: *anyopaque, _: usize, _: bool) void {}
        fn plan(ctx: *anyopaque, pieces: *std.ArrayList(api.Rounds.Piece), finals: *std.ArrayList(usize)) anyerror!void {
            pieces.clearRetainingCapacity();
            finals.clearRetainingCapacity();
            for (self(ctx).seqs.items) |*x| if (!x.done) {
                // the host's lists: its allocator (it frees them)
                try pieces.append(pa, .{ .key = x.key, .start = 0, .end = x.n - 1, .save = false });
                try finals.append(pa, x.key);
                x.done = true;
            };
        }
        fn left(ctx: *anyopaque, key: usize) void {
            const p = self(ctx);
            for (p.seqs.items, 0..) |x, i| if (x.key == key) {
                _ = p.seqs.orderedRemove(i);
                return;
            };
        }
        fn after(_: *anyopaque, _: f64, _: f64) void {}
    };
    var pl: Plan = .{};
    defer pl.seqs.deinit(gpa);
    host.rounds = .{ .ctx = &pl, .vtable = &.{ .begin_admit = Plan.beginAdmit, .admit = Plan.admit, .admitted = Plan.admitted, .arm = Plan.arm, .began = Plan.began, .plan = Plan.plan, .left = Plan.left, .after = Plan.after } };
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: usize = 0,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens += t.len,
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const p2 = [_]u32{ 2, 7, 1, 8, 2, 8 };
    var b1: Box = .{};
    var b2: Box = .{};
    var b3: Box = .{};
    const r1: Request = .{ .prompt = &p1, .max_tokens = 12 };
    const r2: Request = .{ .prompt = &p2, .max_tokens = 12 };
    const e = host.engine();
    // both admitted before the host's first round: the first piece fails, the second's request is served
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try host.start();
    defer host.stop();
    try std.testing.expectEqual(Reason.failed, b1.wait());
    try std.testing.expectEqual(Reason.length, b2.wait());
    try std.testing.expectEqual(@as(usize, 12), b2.tokens);
    // the host serves the next request
    try e.submit(3, &r1, .{ .ctx = &b3, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b3.wait());
    try std.testing.expectEqual(@as(usize, 12), b3.tokens);
}

test "a round planner: prompts in pieces between the other streams' decode rounds, queued admissions, the same tokens" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.pieceBackend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    // one prompt prefilling at a time (the next waits queued), 3 rows a round, a final when its rows are in
    const Plan = struct {
        const Seq = struct { key: usize, n: u64, done: u64 = 0, decoding: bool = false };
        seqs: std.ArrayList(Seq) = .empty,
        waits: u32 = 0,
        rounds: u32 = 0,
        mixed: u32 = 0, // rounds that ran a piece while another stream decoded
        fn self(ctx: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ctx));
        }
        fn beginAdmit(_: *anyopaque) void {}
        fn admit(ctx: *anyopaque, ask: api.Rounds.Ask, _: *[]const u8) api.Rounds.Verdict {
            const p = self(ctx);
            for (p.seqs.items) |x| if (!x.decoding) {
                p.waits += 1;
                return .wait;
            };
            p.seqs.append(gpa, .{ .key = ask.key, .n = ask.prompt.len }) catch return .refuse;
            return .admit;
        }
        fn admitted(_: *anyopaque) anyerror!void {}
        fn arm(_: *anyopaque, _: usize) void {}
        fn began(_: *anyopaque, _: usize, _: bool) void {}
        fn plan(ctx: *anyopaque, pieces: *std.ArrayList(api.Rounds.Piece), finals: *std.ArrayList(usize)) anyerror!void {
            const p = self(ctx);
            pieces.clearRetainingCapacity();
            finals.clearRetainingCapacity();
            var decoding = false;
            for (p.seqs.items) |x| decoding = decoding or x.decoding;
            for (p.seqs.items) |*x| if (!x.decoding) {
                if (x.done < x.n - 1) {
                    const end = @min(x.done + 3, x.n - 1);
                    try pieces.append(gpa, .{ .key = x.key, .start = x.done, .end = end, .save = false });
                    x.done = end;
                    if (decoding) p.mixed += 1;
                }
                if (x.done == x.n - 1) {
                    try finals.append(gpa, x.key);
                    x.decoding = true;
                }
            };
            p.rounds += 1;
        }
        fn left(ctx: *anyopaque, key: usize) void {
            const p = self(ctx);
            for (p.seqs.items, 0..) |x, i| if (x.key == key) {
                _ = p.seqs.orderedRemove(i);
                return;
            };
        }
        fn after(_: *anyopaque, _: f64, _: f64) void {}
    };
    var pl: Plan = .{};
    defer pl.seqs.deinit(gpa);
    host.rounds = .{ .ctx = &pl, .vtable = &.{ .begin_admit = Plan.beginAdmit, .admit = Plan.admit, .admitted = Plan.admitted, .arm = Plan.arm, .began = Plan.began, .plan = Plan.plan, .left = Plan.left, .after = Plan.after } };
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    const p2 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9 };
    var b1: Box = .{};
    var b2: Box = .{};
    defer b1.tokens.deinit(gpa);
    defer b2.tokens.deinit(gpa);
    const r1: Request = .{ .prompt = &p1, .max_tokens = 40 };
    const r2: Request = .{ .prompt = &p2, .max_tokens = 12 };
    const e = host.engine();
    try host.start();
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b1.wait());
    try std.testing.expectEqual(Reason.length, b2.wait());
    host.stop();
    // the replies are the whole-prompt path's tokens
    for ([_]struct { p: []const u32, b: *Box }{ .{ .p = &p1, .b = &b1 }, .{ .p = &p2, .b = &b2 } }) |c| {
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, c.p);
        for (c.b.tokens.items) |t| {
            try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
            try history.append(gpa, t);
        }
    }
    try std.testing.expectEqual(@as(usize, 40), b1.tokens.items.len);
    try std.testing.expectEqual(@as(usize, 12), b2.tokens.items.len);
    // the second prompt waited queued, then ran its pieces while the first decoded
    try std.testing.expect(pl.waits > 0 and pl.mixed >= 3);
    try std.testing.expectEqual(@as(usize, 3 + 4), target.pieces_run);
    try std.testing.expectEqual(@as(usize, 0), pl.seqs.items.len);
}

test "a failed piece run fails its requests alone: their finals of the same round do not run, the host keeps serving" {
    // Spark 2026-10-09 (TF_DSV41_PIECE_RUNS + TAIL_JOIN): a run's error dropped its jobs, whose finals the planner had
    // listed in the same round; runFinals then read the freed jobs (segfault)
    const gpa = std.testing.allocator;
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .window_costs = &costs, .mtp_step_ms = 0.5, .hidden_rows = true, .batch_rows = 32, .max_streams = 8, .draft_streams = true, .join_tail = true, .piece_runs = true }, 16, 15);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .fail_runs = 1 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.runBackend(), clock.clock());
    defer core.deinit();
    // the host's jobs on the page allocator: a freed job's pages are unmapped, so reading one faults as on the Spark
    // (the testing allocator's freed bytes read as a finished stream and hide it)
    var host = LaneHost.init(std.heap.page_allocator, std.testing.io, &core, .{ .lanes = 4 });
    // every admitted prompt's rows in one piece and its final in the same round (as prod's batcher does short prompts)
    const Plan = struct {
        const Seq = struct { key: usize, n: u64, done: bool = false };
        seqs: std.ArrayList(Seq) = .empty,
        fn self(ctx: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ctx));
        }
        fn beginAdmit(_: *anyopaque) void {}
        fn admit(ctx: *anyopaque, ask: api.Rounds.Ask, _: *[]const u8) api.Rounds.Verdict {
            self(ctx).seqs.append(gpa, .{ .key = ask.key, .n = ask.prompt.len }) catch return .refuse;
            return .admit;
        }
        fn admitted(_: *anyopaque) anyerror!void {}
        fn arm(_: *anyopaque, _: usize) void {}
        fn began(_: *anyopaque, _: usize, _: bool) void {}
        fn plan(ctx: *anyopaque, pieces: *std.ArrayList(api.Rounds.Piece), finals: *std.ArrayList(usize)) anyerror!void {
            pieces.clearRetainingCapacity();
            finals.clearRetainingCapacity();
            for (self(ctx).seqs.items) |*x| if (!x.done) {
                // the host's lists: its allocator (it frees them)
                try pieces.append(std.heap.page_allocator, .{ .key = x.key, .start = 0, .end = x.n - 1, .save = false });
                try finals.append(std.heap.page_allocator, x.key);
                x.done = true;
            };
        }
        fn left(ctx: *anyopaque, key: usize) void {
            const p = self(ctx);
            for (p.seqs.items, 0..) |x, i| if (x.key == key) {
                _ = p.seqs.orderedRemove(i);
                return;
            };
        }
        fn after(_: *anyopaque, _: f64, _: f64) void {}
    };
    var pl: Plan = .{};
    defer pl.seqs.deinit(gpa);
    host.rounds = .{ .ctx = &pl, .vtable = &.{ .begin_admit = Plan.beginAdmit, .admit = Plan.admit, .admitted = Plan.admitted, .arm = Plan.arm, .began = Plan.began, .plan = Plan.plan, .left = Plan.left, .after = Plan.after } };
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: usize = 0,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens += t.len,
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const p2 = [_]u32{ 2, 7, 1, 8, 2, 8 };
    var b1: Box = .{};
    var b2: Box = .{};
    var b3: Box = .{};
    const r1: Request = .{ .prompt = &p1, .max_tokens = 12 };
    const r2: Request = .{ .prompt = &p2, .max_tokens = 12 };
    const e = host.engine();
    // both admitted before the host's first round: one run of their pieces, which fails
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try host.start();
    defer host.stop();
    try std.testing.expectEqual(Reason.failed, b1.wait());
    try std.testing.expectEqual(Reason.failed, b2.wait());
    try std.testing.expectEqual(@as(usize, 0), target.joins); // no final of theirs ran
    // the host serves the next request
    try e.submit(3, &r1, .{ .ctx = &b3, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b3.wait());
    try std.testing.expectEqual(@as(usize, 12), b3.tokens);
    try std.testing.expectEqual(@as(usize, 1), target.runs);
}

test "halt (a stop's drain deadline): the rounds end at a round boundary, every request queued or running fails with the reason" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.pieceBackend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    // one prompt prefilling at a time (the next waits queued), 3 rows a round
    const Plan = struct {
        const Seq = struct { key: usize, n: u64, done: u64 = 0, decoding: bool = false };
        seqs: std.ArrayList(Seq) = .empty,
        rounds: u32 = 0,
        fn self(ctx: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ctx));
        }
        fn beginAdmit(_: *anyopaque) void {}
        fn admit(ctx: *anyopaque, ask: api.Rounds.Ask, _: *[]const u8) api.Rounds.Verdict {
            const p = self(ctx);
            for (p.seqs.items) |x| if (!x.decoding) return .wait;
            p.seqs.append(gpa, .{ .key = ask.key, .n = ask.prompt.len }) catch return .refuse;
            return .admit;
        }
        fn admitted(_: *anyopaque) anyerror!void {}
        fn arm(_: *anyopaque, _: usize) void {}
        fn began(_: *anyopaque, _: usize, _: bool) void {}
        fn plan(ctx: *anyopaque, pieces: *std.ArrayList(api.Rounds.Piece), finals: *std.ArrayList(usize)) anyerror!void {
            const p = self(ctx);
            pieces.clearRetainingCapacity();
            finals.clearRetainingCapacity();
            for (p.seqs.items) |*x| if (!x.decoding) {
                if (x.done < x.n - 1) {
                    const end = @min(x.done + 3, x.n - 1);
                    try pieces.append(gpa, .{ .key = x.key, .start = x.done, .end = end, .save = false });
                    x.done = end;
                }
                if (x.done == x.n - 1) {
                    try finals.append(gpa, x.key);
                    x.decoding = true;
                }
            };
            p.rounds += 1;
        }
        fn left(ctx: *anyopaque, key: usize) void {
            const p = self(ctx);
            for (p.seqs.items, 0..) |x, i| if (x.key == key) {
                _ = p.seqs.orderedRemove(i);
                return;
            };
        }
        fn after(_: *anyopaque, _: f64, _: f64) void {}
    };
    var pl: Plan = .{};
    defer pl.seqs.deinit(gpa);
    host.rounds = .{ .ctx = &pl, .vtable = &.{ .begin_admit = Plan.beginAdmit, .admit = Plan.admit, .admitted = Plan.admitted, .arm = Plan.arm, .began = Plan.began, .plan = Plan.plan, .left = Plan.left, .after = Plan.after } };
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        done: ?Reason = null,
        message: []const u8 = "",
        /// the planner's rounds when the finish arrived (on the engine thread, between rounds)
        rounds_at_end: u32 = 0,
        plan: *Plan,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| {
                    b.done = f.reason;
                    b.message = f.message;
                    b.rounds_at_end = b.plan.rounds;
                },
                else => {},
            }
        }
        fn count(b: *@This()) usize {
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            return b.tokens.items.len;
        }
    };
    var long_prompt: [600]u32 = undefined;
    for (&long_prompt, 0..) |*t, i| t.* = @intCast(1 + i % 13);
    const p1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    var boxes: [3]Box = .{ .{ .plan = &pl }, .{ .plan = &pl }, .{ .plan = &pl } };
    defer for (&boxes) |*b| b.tokens.deinit(gpa);
    // 1: decoding (no end of its own), 2: prefilling a long prompt in pieces, 3: queued behind 2
    const r1: Request = .{ .prompt = &p1, .max_tokens = 1 << 30 };
    const r2: Request = .{ .prompt = &long_prompt, .max_tokens = 1 << 30 };
    const r3: Request = .{ .prompt = &p1, .max_tokens = 8 };
    const e = host.engine();
    try host.start();
    try e.submit(1, &r1, .{ .ctx = &boxes[0], .event = Box.event });
    while (boxes[0].count() < 4) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    try e.submit(2, &r2, .{ .ctx = &boxes[1], .event = Box.event });
    try e.submit(3, &r3, .{ .ctx = &boxes[2], .event = Box.event });
    while (boxes[0].count() < 20) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    host.halt("server restarting: retry shortly");
    const rounds = pl.rounds;
    for (&boxes) |*b| {
        try std.testing.expectEqual(@as(?Reason, .failed), b.done);
        try std.testing.expectEqualStrings("server restarting: retry shortly", b.message);
        // no round was planned after the finishes: the loop stopped at the boundary they were sent at
        try std.testing.expectEqual(rounds, b.rounds_at_end);
    }
    // the decoding stream's tokens are whole rounds' (the fake's own sequence), the prefilling one had none yet
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, &p1);
    for (boxes[0].tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    try std.testing.expectEqual(@as(usize, 0), boxes[1].tokens.items.len);
    try std.testing.expectEqual(@as(usize, 0), boxes[2].tokens.items.len);
    try std.testing.expectEqual(@as(usize, 0), pl.seqs.items.len);
    // nothing runs after it; the stop that follows (a close) finds the thread gone
    std.Io.sleep(std.testing.io, .fromMilliseconds(20), .awake) catch {};
    try std.testing.expectEqual(rounds, pl.rounds);
    host.stop();
}
