//! The round plan from rank 0 to the other ranks over its own TCP connection (Python's planlink.TcpLink): one frame a plan, a 4-byte length then little-endian int64s.
const std = @import("std");
const sock = @import("sock.zig");
const builtin = @import("builtin");
const config = @import("config.zig");
const linux = std.os.linux;

pub const max_ranks = 64;
const more: u32 = 0x8000; // MSG_MORE: the length and the ints leave as one segment

pub const Error = error{ PeerClosed, TooLarge, NotRankZero, NotFollower };

/// Largest plan either side accepts: 64M words (512 MiB). An admission carries its prompt ids (~1M words for a 1M-token
/// prompt); the cap keeps a garbled length from becoming a 4 GiB allocation.
pub const max_frame_words: usize = 64 << 20;

comptime {
    std.debug.assert(builtin.cpu.arch.endian() == .little); // the ints go out as they are in memory
}

pub const PlanLink = struct {
    rank: u32,
    world: u32,
    /// Rank 0: rank i's socket at index i; the others: rank 0's at index 0 (the bootstrap's `releasePlan`).
    fds: [max_ranks]sock.Fd,
    sent: u64 = 0,
    received: u64 = 0,
    /// TF_DSV41_PLAN_LINK=rdma (config.plan_spin_ns): the follower polls the socket this long before it blocks, so a
    /// plan is read without waking a sleeping thread (Python's rdma link spins on its mailbox flag the same way)
    spin_ns: u64 = 0,
    /// TF_DSV41_PLAN_PIN: the sending (rank 0) / receiving thread pins itself at its first plan; `avoid`: the RoCE
    /// proxy's CPU, which auto never takes
    pin: ?config.PlanPin = null,
    avoid: ?u32 = null,
    pinned: bool = false,

    pub fn init(rank: u32, world: u32, fds: [max_ranks]sock.Fd) PlanLink {
        return .{ .rank = rank, .world = world, .fds = fds };
    }

    /// Rank 0: one plan to every other rank. Host only: nothing waits on a GPU.
    pub fn send(p: *PlanLink, ints: []const i64) Error!void {
        if (p.rank != 0) return error.NotRankZero;
        if (ints.len > max_frame_words) return error.TooLarge;
        p.pinOnce();
        const bytes = std.mem.sliceAsBytes(ints);
        const len: u32 = @intCast(bytes.len);
        for (p.fds[1..p.world]) |fd| {
            if (std.c.send(fd, &len, 4, std.posix.MSG.NOSIGNAL | more) != 4) return error.PeerClosed;
            sock.sendAll(fd, bytes) catch return error.PeerClosed;
        }
        p.sent += 1;
    }

    /// The other ranks: the next plan into `out` (returned: the ints it holds). Blocks as long as rank 0 is idle; a
    /// rank 0 that is gone ends it (`error.PeerClosed`; the fate channel ends the process right after). A frame larger
    /// than `out` is read and dropped (`error.TooLarge`), so the link stays on frame boundaries; `recvGrow` instead
    /// takes any plan up to `max_frame_words`.
    pub fn recv(p: *PlanLink, out: []i64) Error![]i64 {
        const n = try p.header();
        if (n > out.len) {
            try p.drop(n);
            return error.TooLarge;
        }
        return p.body(out[0..n]);
    }

    /// The next plan into `buf`, grown to the frame (admissions with long prompts); the buffer is kept for the next
    /// call, so a steady stream of window plans allocates nothing.
    pub fn recvGrow(p: *PlanLink, gpa: std.mem.Allocator, buf: *std.ArrayList(i64)) (Error || std.mem.Allocator.Error)![]i64 {
        const n = try p.header();
        if (n > max_frame_words) {
            try p.drop(n);
            return error.TooLarge;
        }
        try buf.resize(gpa, n);
        return p.body(buf.items);
    }

    /// The next frame's length in words.
    fn header(p: *PlanLink) Error!usize {
        if (p.rank == 0) return error.NotFollower;
        p.pinOnce();
        if (p.spin_ns > 0) p.spin();
        var len: u32 = undefined;
        sock.recvExact(p.fds[0], std.mem.asBytes(&len), null) catch return error.PeerClosed;
        if (len % 8 != 0) return error.PeerClosed; // not a frame boundary: the stream is lost
        return len / 8;
    }

    fn body(p: *PlanLink, out: []i64) Error![]i64 {
        sock.recvExact(p.fds[0], std.mem.sliceAsBytes(out), null) catch return error.PeerClosed;
        p.received += 1;
        return out;
    }

    /// Reads and drops `n` words.
    fn drop(p: *PlanLink, n: usize) Error!void {
        var left: usize = n * 8;
        var scratch: [4096]u8 = undefined;
        while (left > 0) : (left -= @min(left, scratch.len)) sock.recvExact(p.fds[0], scratch[0..@min(left, scratch.len)], null) catch return error.PeerClosed;
    }

    /// Polls rank 0's socket until it is readable (or hung up) or `spin_ns` passed; the read that follows blocks.
    fn spin(p: *PlanLink) void {
        if (comptime builtin.os.tag != .linux) return;
        var fd = [1]linux.pollfd{.{ .fd = p.fds[0], .events = linux.POLL.IN, .revents = 0 }};
        const end = sock.nowNs() + p.spin_ns;
        while (true) {
            const rc = linux.poll(&fd, 1, 0);
            if (linux.errno(rc) == .SUCCESS and rc > 0) return;
            if (sock.nowNs() >= end) return;
            std.atomic.spinLoopHint();
        }
    }

    /// Pins the calling thread once (planlink.Pinned): an explicit CPU, or auto's highest cpu_capacity.
    fn pinOnce(p: *PlanLink) void {
        if (p.pinned) return;
        p.pinned = true;
        if (comptime builtin.os.tag != .linux) return;
        const want = p.pin orelse return;
        const cpu = switch (want) {
            .cpu => |c| c,
            .auto => bestCpu(p.avoid) orelse {
                std.log.info("[tensorfold] tp: TF_DSV41_PLAN_PIN=auto: no cpu_capacity differs among the allowed CPUs; the plan thread stays unpinned", .{});
                return;
            },
        };
        pinTo(cpu) catch |e| {
            std.log.warn("[tensorfold] tp: TF_DSV41_PLAN_PIN: CPU {d} refused ({t}); the plan thread stays unpinned", .{ cpu, e });
            return;
        };
        std.log.info("[tensorfold] tp: plan link rank {d} pinned to CPU {d}", .{ p.rank, cpu });
    }

    pub fn close(p: *PlanLink) void {
        for (&p.fds) |*fd| if (fd.* >= 0) {
            sock.close(fd.*);
            fd.* = -1;
        };
    }
};

/// The calling thread onto `cpu` alone.
pub fn pinTo(cpu: u32) !void {
    if (cpu >= linux.CPU_SETSIZE) return error.BadCpu;
    var set: linux.cpu_set_t = @splat(0);
    set[cpu / @bitSizeOf(usize)] |= @as(usize, 1) << @intCast(cpu % @bitSizeOf(usize));
    try linux.sched_setaffinity(0, &set);
}

/// planlink.pin_cpu("auto"): the allowed CPU (other than `avoid`) with the highest cpu_capacity, the lowest on ties;
/// null when the kernel reports no capacities or they are all equal (nothing to prefer: GB10's X925 vs A725 differ).
pub fn bestCpu(avoid: ?u32) ?u32 {
    var set: linux.cpu_set_t = undefined;
    if (linux.errno(linux.sched_getaffinity(0, @sizeOf(linux.cpu_set_t), &set)) != .SUCCESS) return null;
    var best: ?u32 = null;
    var best_cap: u32 = 0;
    var low_cap: u32 = std.math.maxInt(u32);
    for (set, 0..) |word, wi| {
        var bits = word;
        while (bits != 0) : (bits &= bits - 1) {
            const cpu: u32 = @intCast(wi * @bitSizeOf(usize) + @ctz(bits));
            if (avoid == cpu) continue;
            const c = capacity(cpu) orelse continue;
            low_cap = @min(low_cap, c);
            if (best == null or c > best_cap) {
                best = cpu;
                best_cap = c;
            }
        }
    }
    return if (best != null and best_cap > low_cap) best else null;
}

fn capacity(cpu: u32) ?u32 {
    var pb: [64]u8 = undefined;
    const path = std.fmt.bufPrintSentinel(&pb, "/sys/devices/system/cpu/cpu{d}/cpu_capacity", .{cpu}, 0) catch return null;
    const rc = linux.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
    if (linux.errno(rc) != .SUCCESS) return null;
    const fd: i32 = @intCast(rc);
    defer _ = linux.close(fd);
    var b: [16]u8 = undefined;
    const n = linux.read(fd, &b, b.len);
    if (linux.errno(n) != .SUCCESS) return null;
    return std.fmt.parseInt(u32, std.mem.trim(u8, b[0..n], " \n"), 10) catch null;
}

test "plans arrive whole and in order; a closed rank 0 ends the follower" {
    const bootstrap = @import("bootstrap.zig");
    const l = try bootstrap.testListener();
    defer sock.close(l.fd);
    const deadline = sock.nowNs() + 5 * std.time.ns_per_s;
    const a = try sock.connect("localhost", l.port, deadline);
    const b = try sock.accept(l.fd, deadline);
    var f0: [max_ranks]sock.Fd = @splat(-1);
    var f1: [max_ranks]sock.Fd = @splat(-1);
    f0[1] = b;
    f1[0] = a;
    var lead = PlanLink.init(0, 2, f0);
    var follow = PlanLink.init(1, 2, f1);
    defer follow.close();
    var big: [5000]i64 = undefined;
    for (&big, 0..) |*v, i| v.* = @as(i64, @intCast(i)) * -3 + 0x7FFF_0000_0000;
    try lead.send(&.{ 0x504C414E, 2, 1 });
    try lead.send(&big);
    try lead.send(&.{});
    var buf: [8192]i64 = undefined;
    try std.testing.expectEqualSlices(i64, &.{ 0x504C414E, 2, 1 }, try follow.recv(&buf));
    try std.testing.expectEqualSlices(i64, &big, try follow.recv(&buf));
    try std.testing.expectEqual(@as(usize, 0), (try follow.recv(&buf)).len);
    try std.testing.expectError(error.NotRankZero, follow.send(&.{1}));
    var small: [2]i64 = undefined;
    try lead.send(&.{ 1, 2, 3 });
    try std.testing.expectError(error.TooLarge, follow.recv(&small));
    lead.close();
    try std.testing.expectError(error.PeerClosed, follow.recv(&buf));
}

test "an admission-sized plan (1.2M words) arrives whole through the growable path" {
    const bootstrap = @import("bootstrap.zig");
    const l = try bootstrap.testListener();
    defer sock.close(l.fd);
    const deadline = sock.nowNs() + 5 * std.time.ns_per_s;
    const a = try sock.connect("localhost", l.port, deadline);
    const b = try sock.accept(l.fd, deadline);
    var f0: [max_ranks]sock.Fd = @splat(-1);
    var f1: [max_ranks]sock.Fd = @splat(-1);
    f0[1] = b;
    f1[0] = a;
    var lead = PlanLink.init(0, 2, f0);
    defer lead.close();
    var follow = PlanLink.init(1, 2, f1);
    defer follow.close();
    const gpa = std.testing.allocator;
    const prompt = try gpa.alloc(i64, 1_200_000);
    defer gpa.free(prompt);
    for (prompt, 0..) |*v, i| v.* = @intCast((i *% 2654435761) % 163840);
    // the socket buffer holds far less than 9.6 MB: rank 0 sends from its own thread, as it would
    const Send = struct {
        fn go(link: *PlanLink, ints: []const i64) void {
            link.send(&.{ 0x504C414E, 3, 20 }) catch {};
            link.send(ints) catch {};
            link.send(&.{ 0x504C414E, 3, 21 }) catch {};
        }
    };
    const t = try std.Thread.spawn(.{}, Send.go, .{ &lead, prompt });
    var buf: std.ArrayList(i64) = .empty;
    defer buf.deinit(gpa);
    try std.testing.expectEqualSlices(i64, &.{ 0x504C414E, 3, 20 }, try follow.recvGrow(gpa, &buf));
    try std.testing.expectEqualSlices(i64, prompt, try follow.recvGrow(gpa, &buf));
    try std.testing.expectEqualSlices(i64, &.{ 0x504C414E, 3, 21 }, try follow.recvGrow(gpa, &buf));
    t.join();
    try std.testing.expect(buf.capacity >= prompt.len); // kept for the next admission
}

test "TF_DSV41_PLAN_LINK=rdma: a spinning follower gets every plan, late or early; TF_DSV41_PLAN_PIN pins the thread" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const bootstrap = @import("bootstrap.zig");
    const l = try bootstrap.testListener();
    defer sock.close(l.fd);
    const deadline = sock.nowNs() + 5 * std.time.ns_per_s;
    const a = try sock.connect("localhost", l.port, deadline);
    const b = try sock.accept(l.fd, deadline);
    var f0: [max_ranks]sock.Fd = @splat(-1);
    var f1: [max_ranks]sock.Fd = @splat(-1);
    f0[1] = b;
    f1[0] = a;
    var lead = PlanLink.init(0, 2, f0);
    defer lead.close();
    var follow = PlanLink.init(1, 2, f1);
    defer follow.close();
    follow.spin_ns = 2 * std.time.ns_per_ms;
    // the CPU this thread may run on first: the pin keeps it allowed, so the test runner is not starved
    var set: linux.cpu_set_t = undefined;
    _ = linux.sched_getaffinity(0, @sizeOf(linux.cpu_set_t), &set);
    var first: u32 = 0;
    for (set, 0..) |w, i| if (w != 0) {
        first = @intCast(i * @bitSizeOf(usize) + @ctz(w));
        break;
    };
    const Follower = struct {
        fn run(f: *PlanLink, cpu: u32, out: *[3]i64, ok: *bool) void {
            f.pin = .{ .cpu = cpu };
            var buf: [8]i64 = undefined;
            const x = (f.recv(&buf) catch return)[0]; // arrives within the spin
            const y = (f.recv(&buf) catch return)[0]; // arrives after it: the blocking read
            var now: linux.cpu_set_t = undefined;
            _ = linux.sched_getaffinity(0, @sizeOf(linux.cpu_set_t), &now);
            var count: usize = 0;
            for (now) |w| count += @popCount(w);
            out.* = .{ x, y, @intCast(count) };
            ok.* = f.pinned and (now[cpu / @bitSizeOf(usize)] >> @intCast(cpu % @bitSizeOf(usize))) & 1 == 1;
        }
    };
    var out: [3]i64 = .{ 0, 0, 0 };
    var ok = false;
    const t = try std.Thread.spawn(.{}, Follower.run, .{ &follow, first, &out, &ok });
    try lead.send(&.{11});
    var ts: linux.timespec = .{ .sec = 0, .nsec = 20 * std.time.ns_per_ms };
    _ = linux.nanosleep(&ts, null);
    try lead.send(&.{22});
    t.join();
    try std.testing.expectEqual([3]i64{ 11, 22, 1 }, out);
    try std.testing.expect(ok);
    if (bestCpu(null)) |c| try std.testing.expect((set[c / @bitSizeOf(usize)] >> @intCast(c % @bitSizeOf(usize))) & 1 == 1);
}
