//! A rank's named weights on the GPU: each tensor its own device buffer (the address the kernels and the replay bind by name), digested on the way up.

const std = @import("std");
const cuda = @import("cuda");
const named = @import("named.zig");

pub const Tensor = struct {
    ptr: u64,
    len: usize,
    kind: named.Kind,
    sha256: [32]u8,
    shape: [4]usize = @splat(1),
    rank: u8 = 0,

    pub fn dims(t: *const Tensor) []const usize {
        return t.shape[0..t.rank];
    }
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    map: std.StringHashMapUnmanaged(Tensor) = .empty,
    bytes: usize = 0,
    /// each tensor's SHA-256 taken on the way up (M1's weights check reads it); off, `sha256` is zeros: the served
    /// model's boot, where nothing reads it (95 GiB a rank through one thread's SHA-256, in software on the generic
    /// aarch64 build: the likely bulk of the Spark night's 261 s "upload + digest")
    digest: bool = true,

    pub fn deinit(w: *Weights) void {
        for (w.bufs.items) |*b| b.free();
        w.bufs.deinit(w.gpa);
        var it = w.map.keyIterator();
        while (it.next()) |k| w.gpa.free(k.*);
        w.map.deinit(w.gpa);
        w.* = undefined;
    }

    /// Uploads every tensor `b` holds (its host images can be freed afterwards); a name already on the GPU is refused.
    pub fn upload(w: *Weights, d: *const cuda.Driver, b: *const named.Builder) !void {
        for (b.out.items) |n| {
            if (w.map.contains(n.name)) return error.DuplicateWeight;
            var sha: [32]u8 = @splat(0);
            if (w.digest) std.crypto.hash.sha2.Sha256.hash(n.bytes, &sha, .{});
            var buf = try cuda.DeviceBuffer.fromHost(d, n.bytes);
            errdefer buf.free();
            try w.bufs.append(w.gpa, buf);
            try w.map.put(w.gpa, try w.gpa.dupe(u8, n.name), .{ .ptr = buf.ptr, .len = n.bytes.len, .kind = n.kind, .sha256 = sha, .shape = n.shape, .rank = n.rank });
            w.bytes += n.bytes.len;
        }
        // pageable uploads can still be in flight when cuMemcpyHtoD returns; the weights are whole before any launch
        try d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize");
    }

    pub fn get(w: *const Weights, name: []const u8) ?Tensor {
        return w.map.get(name);
    }
};
