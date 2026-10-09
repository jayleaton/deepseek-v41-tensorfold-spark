//! TF_DSV41_ARENA (default off): the engine's device buffers carved from a few large chunks (cuda/arena.zig) instead
//! of one cuMemAlloc each: the weights and their run-time forms (mHC `fn16`, norms, the dense lanes), the persistent
//! roles, the window arena, scratch, KV staging, the drafter's buffers. PyTorch's caching allocator does the same
//! (2 / 20 MiB+ segments); Zig's per-buffer allocations leave each small buffer on a page of its own.
//!
//! Values: unset / 0 off; 1 or `vmm`: CUDA virtual memory (one reserved range, physical chunks of the device's
//! recommended granularity, at least 2 MiB), falling back to `plain` when the device or driver has no VMM; `plain`:
//! one cuMemAlloc a chunk. Installed after the TP session opens (its fabric buffers keep their own allocations) and
//! before the weights load, on every rank. Only addresses change: the same kernels, arguments and launches, the same
//! bits.

const std = @import("std");
const cuda = @import("cuda");

const log = std.log.scoped(.dsv41);

pub const Mode = enum { off, vmm, plain };

/// TF_CUDA_SCHED (unset: the driver's `auto`): how the host waits for the GPU (cuda.Context.Schedule). On GB10 the
/// CPU and GPU share one power budget; a host spinning in cuEventSynchronize between rounds may cost the GPU clock
/// that latency-bound kernels need. Host scheduling only, the same launches and bits.
pub fn schedule(get: *const fn ([]const u8) ?[]const u8) !?cuda.Context.Schedule {
    const raw = std.mem.trim(u8, get("TF_CUDA_SCHED") orelse return null, " \t");
    if (raw.len == 0) return null;
    return std.meta.stringToEnum(cuda.Context.Schedule, raw) orelse {
        log.warn("TF_CUDA_SCHED={s}: expected auto, spin, yield or blocking", .{raw});
        return error.BadKnob;
    };
}

/// Sets TF_CUDA_SCHED on `ordinal`'s primary context; call before cuda.Context.init.
pub fn applySchedule(d: *const cuda.Driver, ordinal: c_int) !void {
    const s = try schedule(&env) orelse return;
    try cuda.Context.setSchedule(d, ordinal, s);
    log.info("cuda schedule: {t}", .{s});
}

pub fn mode(get: *const fn ([]const u8) ?[]const u8) !Mode {
    const raw = std.mem.trim(u8, get("TF_DSV41_ARENA") orelse return .off, " \t");
    if (raw.len == 0 or std.mem.eql(u8, raw, "0")) return .off;
    if (std.mem.eql(u8, raw, "1") or std.mem.eql(u8, raw, "vmm")) return .vmm;
    if (std.mem.eql(u8, raw, "plain")) return .plain;
    log.warn("TF_DSV41_ARENA={s}: expected 0, 1 / vmm or plain", .{raw});
    return error.BadKnob;
}

/// The arena and its backend, for the process's lifetime (never freed: buffers it holds may outlive any owner).
pub const Installed = struct {
    vmm: cuda.arena.Vmm = undefined,
    plain: cuda.arena.Plain,
    arena: cuda.arena.Arena = undefined,
    backend: Mode,
};

fn env(name: []const u8) ?[]const u8 {
    return @import("prod_knobs.zig").env(name);
}

/// Reads the knob and installs the arena under DeviceBuffer.alloc; null when off.
pub fn install(gpa: std.mem.Allocator, ctx: *const cuda.Context, rank: u32) !?*Installed {
    const m = try mode(&env);
    if (m == .off) return null;
    const d = ctx.d;
    const x = try gpa.create(Installed);
    x.* = .{ .plain = .{ .d = d }, .backend = .plain };
    var be = x.plain.backend();
    if (m == .vmm) {
        var total: usize = 0;
        try d.check(d.api.cuDeviceTotalMem_v2(&total, ctx.device), "cuDeviceTotalMem");
        if (cuda.arena.Vmm.open(d, ctx.device, 2 * total)) |v| {
            x.vmm = v;
            x.backend = .vmm;
            be = x.vmm.backend();
        } else |e| log.warn("TF_DSV41_ARENA: no CUDA virtual memory ({t}): one cuMemAlloc a chunk", .{e});
    }
    x.arena = cuda.arena.Arena.init(gpa, be, .{});
    x.arena.install();
    if (rank == 0) log.info("device arena: {s}, {d} KiB granularity, shared chunks of {d} MiB for buffers up to {d} MiB", .{ be.name, be.granularity >> 10, x.arena.o.small_chunk >> 20, x.arena.o.small_max >> 20 });
    return x;
}

test "TF_CUDA_SCHED reader" {
    const t = std.testing;
    const F = struct {
        var v: ?[]const u8 = null;
        fn get(_: []const u8) ?[]const u8 {
            return v;
        }
    };
    F.v = null;
    try t.expectEqual(@as(?cuda.Context.Schedule, null), try schedule(&F.get));
    F.v = "blocking";
    try t.expectEqual(@as(?cuda.Context.Schedule, .blocking), try schedule(&F.get));
    F.v = " yield ";
    try t.expectEqual(@as(?cuda.Context.Schedule, .yield), try schedule(&F.get));
    F.v = "sleep";
    try t.expectError(error.BadKnob, schedule(&F.get));
}

test "TF_DSV41_ARENA reader" {
    const t = std.testing;
    const F = struct {
        var v: ?[]const u8 = null;
        fn get(_: []const u8) ?[]const u8 {
            return v;
        }
    };
    for ([_]?[]const u8{ null, "0", "" }) |v| {
        F.v = v;
        try t.expectEqual(Mode.off, try mode(&F.get));
    }
    F.v = "1";
    try t.expectEqual(Mode.vmm, try mode(&F.get));
    F.v = " vmm ";
    try t.expectEqual(Mode.vmm, try mode(&F.get));
    F.v = "plain";
    try t.expectEqual(Mode.plain, try mode(&F.get));
    F.v = "yes";
    try t.expectError(error.BadKnob, mode(&F.get));
}
