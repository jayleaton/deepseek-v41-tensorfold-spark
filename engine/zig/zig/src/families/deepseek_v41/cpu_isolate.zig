//! TF_DSV41_PIN_ISOLATE=1: the engine's worker threads (the Engram readers, the Engram gate's worker) keep off the CPU
//! the plan thread pinned itself to (TF_DSV41_PLAN_PIN: on a follower the plan thread is the one that runs every
//! window). Rank 1's nsys (code T0 x4, 2026-10-09): ~250 us a window pass with no CUDA or intercepted call between the
//! gate's arm (a cond broadcast that wakes the worker, which wakes the readers) and the window's graph launch; a pinned
//! thread cannot move off a core a woken thread took. Affinity only: the same work in the same order.
//!
//! `reserved` is published by the forward (Forward.publishPin, from tp.planlink.pinned_cpu) and read by each worker
//! thread once a job (`Isolate.apply`): the first time it sees a CPU it narrows its own mask to its start mask
//! without that CPU. No imports beyond std, so the host test roots that build the Engram modules alone still compile.

const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;

/// The pinned plan thread's CPU, -1 until there is one.
pub var reserved: std.atomic.Value(i32) = .init(-1);

pub fn fromEnv() bool {
    const v = std.c.getenv("TF_DSV41_PIN_ISOLATE") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// One worker thread's state (a local of its loop).
pub const Isolate = struct {
    on: bool,
    seen: i32 = -1,
    narrowed: u32 = 0,

    pub fn init() Isolate {
        return .{ .on = fromEnv() };
    }

    /// Before a job: once a CPU is reserved, this thread's mask without it (its own current mask, so a thread
    /// confined elsewhere stays there); nothing when the knob is off or nothing changed.
    pub fn apply(x: *Isolate) void {
        if (!x.on) return;
        const c = reserved.load(.acquire);
        if (c == x.seen) return;
        x.seen = c;
        if (c < 0) return;
        if (comptime builtin.os.tag != .linux) return;
        var set: linux.cpu_set_t = undefined;
        if (linux.errno(linux.sched_getaffinity(0, @sizeOf(linux.cpu_set_t), &set)) != .SUCCESS) return;
        if (!without(&set, @intCast(c))) return; // the reserved CPU is all it has, or not in its mask
        linux.sched_setaffinity(0, &set) catch return;
        x.narrowed += 1;
    }
};

/// `set` without `cpu`; false (unchanged) when `cpu` is not in it or is its only CPU.
pub fn without(set: *linux.cpu_set_t, cpu: u32) bool {
    const w = cpu / @bitSizeOf(usize);
    if (w >= set.len) return false;
    const bit = @as(usize, 1) << @intCast(cpu % @bitSizeOf(usize));
    if (set[w] & bit == 0) return false;
    var count: usize = 0;
    for (set) |word| count += @popCount(word);
    if (count <= 1) return false;
    set[w] &= ~bit;
    return true;
}

test "TF_DSV41_PIN_ISOLATE: the reserved CPU leaves a mask, never its last CPU; off reads nothing" {
    var set: linux.cpu_set_t = @splat(0);
    set[0] = 0b1011;
    try std.testing.expect(without(&set, 1));
    try std.testing.expectEqual(@as(usize, 0b1001), set[0]);
    try std.testing.expect(!without(&set, 2)); // not in it
    var one: linux.cpu_set_t = @splat(0);
    one[0] = 0b100;
    try std.testing.expect(!without(&one, 2)); // its only CPU
    try std.testing.expectEqual(@as(usize, 0b100), one[0]);
    var off: Isolate = .{ .on = false };
    reserved.store(3, .release);
    off.apply();
    try std.testing.expectEqual(@as(i32, -1), off.seen);
    reserved.store(-1, .release);
}
