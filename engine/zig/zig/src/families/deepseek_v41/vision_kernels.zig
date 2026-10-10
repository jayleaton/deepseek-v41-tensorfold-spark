//! vision.cu's launches (zig/kernels/cuda/deepseek_v41_vision): the vision tower's torch-order glue and its FA2-shaped
//! attention, in their own fatbin (`dsv41_vision`). The fatbin is optional in the build: an empty image, and `load`
//! refuses (TF_DSV41_IMAGES=native then refuses at boot).
const std = @import("std");
const cuda = @import("cuda");

const Blob = struct {
    pub const bytes align(16) = @embedFile("dsv41_fatbin_dsv41_vision").*;
};
const image: []const u8 = &Blob.bytes;

/// The build embedded the kernels (-Dnvcc, or -Dfatbins with dsv41_vision.fatbin).
pub const available = image.len > 0;

const names = .{
    .rms = "dsv41_vision_rms",
    .rope = "dsv41_vision_rope",
    .attn = "dsv41_vision_attn",
    .silu = "dsv41_vision_silu_mul",
    .gelu = "dsv41_vision_gelu",
    .add = "dsv41_vision_add",
    .unfold = "dsv41_vision_unfold",
    .span = "dsv41_vision_span",
};

pub const Functions = struct {
    module: cuda.Module,
    rms: cuda.Function,
    rope: cuda.Function,
    attn: cuda.Function,
    silu: cuda.Function,
    gelu: cuda.Function,
    add: cuda.Function,
    unfold: cuda.Function,
    span: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Functions {
        if (!available) return error.VisionKernelNotBuilt;
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        var f: Functions = undefined;
        f.module = m;
        inline for (@typeInfo(@TypeOf(names)).@"struct".field_names) |name| @field(f, name) = try m.function(@field(names, name));
        return f;
    }

    pub fn unload(f: *Functions) void {
        f.module.unload();
    }

    pub fn ops(f: *const Functions, s: cuda.Stream) Ops {
        return .{ .f = f, .s = s };
    }
};

fn dims(x: usize, y: usize, z: usize) cuda.launch.Dim3 {
    return .{ .x = @intCast(x), .y = @intCast(y), .z = @intCast(z) };
}

fn blocks(n: usize, per: usize) usize {
    return (n + per - 1) / per;
}

/// The tower's launches on one stream.
pub const Ops = struct {
    f: *const Functions,
    s: cuda.Stream,

    fn go(o: Ops, func: cuda.Function, grid: cuda.launch.Dim3, block: cuda.launch.Dim3, args: *cuda.Args) !void {
        try cuda.launch.launch(func, .{ .grid = grid, .block = block }, o.s, args);
    }

    /// _rms over [n, D] (n >= 16: torch's reduce config is then one warp a row; D a multiple of 128)
    pub fn rms(o: Ops, x: u64, w: u64, out: u64, n: usize, D: usize, eps: f32) !void {
        if (n < 16 or D % 128 != 0) return error.Shape;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(out);
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(D)));
        a.add(eps);
        try o.go(o.f.rms, dims(blocks(n, 16), 1, 1), dims(32, 16, 1), &a);
    }

    pub fn rope(o: Ops, qkv: u64, q: u64, k: u64, v: u64, n: usize, nw: usize, heads: usize, theta: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ qkv, q, k, v }) |p| a.add(p);
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(nw)));
        a.add(@as(c_int, @intCast(heads)));
        a.add(theta);
        try o.go(o.f.rope, dims(n, 1, 1), dims(heads * 32, 1, 1), &a);
    }

    pub fn attn(o: Ops, q: u64, k: u64, v: u64, out: u64, n: usize, heads: usize, scale_log2: f32) !void {
        var a: cuda.Args = .{};
        inline for (.{ q, k, v, out }) |p| a.add(p);
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(heads)));
        a.add(scale_log2);
        try o.go(o.f.attn, dims(blocks(n, 128), heads, 1), dims(128, 1, 1), &a);
    }

    pub fn siluMul(o: Ops, h: u64, out: u64, n: usize, I: usize) !void {
        var a: cuda.Args = .{};
        a.add(h);
        a.add(out);
        a.add(@as(i64, @intCast(n)));
        a.add(@as(c_int, @intCast(I)));
        try o.go(o.f.silu, dims(blocks(n * I, 256), 1, 1), dims(256, 1, 1), &a);
    }

    pub fn gelu(o: Ops, x: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(n)));
        try o.go(o.f.gelu, dims(blocks(n, 256), 1, 1), dims(256, 1, 1), &a);
    }

    pub fn add(o: Ops, x: u64, y: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(@as(i64, @intCast(n)));
        try o.go(o.f.add, dims(blocks(n, 256), 1, 1), dims(256, 1, 1), &a);
    }

    pub fn unfold(o: Ops, x: u64, out: u64, h: usize, w: usize, C: usize, r: usize, L: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(out);
        inline for (.{ h, w, C, r }) |v| a.add(@as(c_int, @intCast(v)));
        try o.go(o.f.unfold, dims(L, 1, 1), dims(256, 1, 1), &a);
    }

    pub fn span(o: Ops, feats: u64, start: u64, newline: u64, end: u64, out: u64, lh: usize, lw: usize, D: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ feats, start, newline, end, out }) |p| a.add(p);
        inline for (.{ lh, lw, D }) |v| a.add(@as(c_int, @intCast(v)));
        try o.go(o.f.span, dims(lh * (lw + 1) + 2, 1, 1), dims(256, 1, 1), &a);
    }
};
