//! DeepSeek-V4.1's dense EXL3 linears (mul1): upstream's rot_in / linear_kernel / unpack_kernel (linear.cu), x3seg's
//! segment tables (fused_proj: one rot_in launch + one linear launch a width for up to 8 projections) and dense3's
//! "lanes" layout of the same programs (TF_DSV41_DENSE_V3, prod). Plans and tables are the Python engine's
//! (linear.plan, fused_proj.layout), so a Zig launch runs the same programs on the same k ranges.

const std = @import("std");
const cuda = @import("cuda");

pub const DType = enum(c_int) { f16 = 0, bf16 = 1, f32 = 2 };

pub fn dtypeSize(t: DType) usize {
    return switch (t) {
        .f16, .bf16 => 2,
        .f32 => 4,
    };
}

// ---------------------------------------------------------------------------------------------------------------
// Plans and tables (pure: host tests cover them)

/// linear.plan: (K splits, warps a program) for a K x N layer, the shape's alone (every row count one reduction).
pub fn plan(k: usize, n: usize) [2]usize {
    const blocks = 192;
    const min_tiles = 8;
    const kt = k / 16;
    const nb = n / 128;
    var sk: usize = 1;
    var wk: usize = if (nb >= 64) 8 else 4;
    if (nb < 64) {
        while (nb * sk < blocks and sk < 64) sk *= 2;
    }
    while ((wk > 4 or sk > 1) and (kt % (sk * wk) != 0 or kt / (sk * wk) < min_tiles)) {
        if (wk > 4) {
            wk = 4;
        } else if (sk > 1) {
            sk /= 2;
        } else break;
    }
    return .{ sk, wk };
}

/// Words between k tiles and between 128-column blocks of a matrix in the "strips" layout (both dense layouts).
pub fn strides(k: usize, k2: u32) [2]i64 {
    const tw: i64 = 4 * @as(i64, k2);
    return .{ 8 * tw, @as(i64, @intCast(k / 16)) * 8 * tw };
}

pub const max_seg = 8; // x3seg.cu MAXSEG
pub const rows_max = 128; // fused_proj.ROWS / dense3.ROWS: rows a launch
pub const x3seg_k2s = [_]u32{ 4, 6, 8, 10, 12, 16 };
pub const dense3_k2s = [_]u32{ 4, 6, 8, 10, 12 };

/// fused_proj.Seg: one projection as the kernel sees it, and where its output goes (buffer `out`, column `col`).
pub const Seg = struct {
    k: usize,
    n: usize,
    k2: u32,
    sk: usize,
    wk: usize,
    out: usize = 0,
    col: usize = 0,

    pub fn programs(s: Seg) usize {
        return (s.n / 128) * s.sk;
    }
};

pub const Launch = struct {
    k2: u32,
    wk: usize,
    segs: [max_seg]usize = undefined, // indices into the call's segments
    count: usize = 0,
    programs: usize = 0,
    p0: [max_seg]usize = undefined, // first program of each segment in this launch
};

/// fused_proj.Layout: scratch offsets (fp16 rotated inputs, Z floats, counter ints) and the launches at `m` rows.
pub const Layout = struct {
    m: usize,
    xh_off: [64]usize = undefined,
    z_off: [64]usize = undefined,
    c_off: [64]usize = undefined,
    xh_total: usize = 0,
    z_total: usize = 0,
    c_total: usize = 0,
    launches: [16]Launch = undefined,
    n_launches: usize = 0,
};

pub const Error = error{ Shape, Unsupported, TooMany };

/// fused_proj.layout: one launch a width in first-appearance order, the segments in their order, up to 8 a launch;
/// WK of a launch = its segments' largest.
pub fn layout(segs: []const Seg, m: usize) Error!Layout {
    if (m < 1 or m > rows_max) return error.Shape;
    if (segs.len > 64) return error.TooMany;
    var lay: Layout = .{ .m = m };
    for (segs, 0..) |s, i| {
        if (s.k % 128 != 0 or s.n % 128 != 0 or s.sk < 1 or (s.wk != 4 and s.wk != 8) or (s.k / 16) % (s.sk * s.wk) != 0) return error.Shape;
        lay.xh_off[i] = lay.xh_total;
        lay.z_off[i] = lay.z_total;
        lay.c_off[i] = lay.c_total;
        lay.xh_total += m * s.k;
        if (s.sk > 1) lay.z_total += s.sk * m * s.n;
        lay.c_total += 8 * (s.n / 128);
    }
    var seen: [64]u32 = undefined;
    var nseen: usize = 0;
    for (segs) |s| {
        if (std.mem.indexOfScalar(u32, seen[0..nseen], s.k2) == null) {
            seen[nseen] = s.k2;
            nseen += 1;
        }
    }
    for (seen[0..nseen]) |k2| {
        var l: Launch = .{ .k2 = k2, .wk = 0 };
        for (segs, 0..) |s, i| {
            if (s.k2 != k2) continue;
            if (l.count == max_seg) {
                if (lay.n_launches == lay.launches.len) return error.TooMany;
                lay.launches[lay.n_launches] = l;
                lay.n_launches += 1;
                l = .{ .k2 = k2, .wk = 0 };
            }
            l.segs[l.count] = i;
            l.p0[l.count] = l.programs;
            l.count += 1;
            l.programs += s.programs();
            l.wk = @max(l.wk, s.wk);
        }
        if (lay.n_launches == lay.launches.len) return error.TooMany;
        lay.launches[lay.n_launches] = l;
        lay.n_launches += 1;
    }
    return lay;
}

/// dense3.group_steps: k steps a load group (whole 16-byte chunks a lane).
pub fn groupSteps(k2: u32) usize {
    return if (k2 == 16) 1 else 2;
}

// x3seg.cu's tables, passed by value
/// R1c: with `nw` set, the segment's input is rmsnorm's row bf16(fp32((x * rn) * w)) of bf16 x (_rms_row's rn,
/// computed in the kernel); inv_k = fp32(1 / K) and eps as Triton's float arguments round them.
pub const RotSeg = extern struct { x: u64, suh: u64, xh: u64, ld: i64, K: c_int, x_dtype: c_int, nw: u64 = 0, nw_dtype: c_int = 0, inv_k: f32 = 0, eps: f32 = 0 };
pub const RotTable = extern struct { s: [max_seg]RotSeg, n: c_int };
pub const LinSeg = extern struct {
    xh: u64,
    T: u64,
    stride_k: i64,
    stride_nb: i64,
    svh: u64,
    y: u64,
    Z: u64,
    counters: u64,
    K: c_int,
    N: c_int,
    SK: c_int,
    wk: c_int,
    p0: c_int,
    ld: c_int,
};
pub const LinTable = extern struct { s: [max_seg]LinSeg, n: c_int };

comptime {
    std.debug.assert(@sizeOf(RotSeg) == 64 and @offsetOf(RotSeg, "nw") == 40 and @sizeOf(RotTable) == 520);
    std.debug.assert(@sizeOf(LinSeg) == 88 and @sizeOf(LinTable) == 712);
}

/// One segment's buffers for a linear launch: the matrix words and svh, the output (row stride `ld` elements, the
/// segment's columns from `y`), the call's xh / Z / counters bases.
pub const SegBufs = struct { T: u64, svh: u64, y: u64, ld: usize };

/// One segment's input for the rotation: rows at x + row * ld (elements), its dtype, and the matrix's suh [K].
/// `nw` (R1c's folded q_norm): the [K] norm weight and its dtype, 0 for none; the input is then bf16 un-normed rows.
pub const RotIn = struct { x: u64, ld: usize, dtype: DType, suh: u64, nw: u64 = 0, nw_dtype: DType = .bf16 };

/// fp32(1 / K): the binding's `static_cast<float>(1.0 / K)`.
pub fn invK(k: usize) f32 {
    return @floatCast(1.0 / @as(f64, @floatFromInt(k)));
}

/// fused_proj._block's rot_in tables: one launch per 8 segments (`first` = 0, 8, ...), each segment's rotated rows
/// at its xh offset of `lay` (dsv41_x3seg_rot_in_cuda's RotTable).
pub fn rotTable(lay: Layout, segs: []const Seg, ins: []const RotIn, first: usize, xh: u64, eps: f32) RotTable {
    var t = std.mem.zeroes(RotTable);
    const end = @min(segs.len, first + max_seg);
    t.n = @intCast(end - first);
    for (first..end, 0..) |i, j| {
        t.s[j] = .{ .x = ins[i].x, .suh = ins[i].suh, .xh = xh + 2 * lay.xh_off[i], .ld = @intCast(ins[i].ld), .K = @intCast(segs[i].k), .x_dtype = @intFromEnum(ins[i].dtype) };
        if (ins[i].nw != 0) {
            t.s[j].nw = ins[i].nw;
            t.s[j].nw_dtype = @intFromEnum(ins[i].nw_dtype);
            t.s[j].inv_k = invK(segs[i].k);
            t.s[j].eps = eps;
        }
    }
    return t;
}

/// dsv41_x3seg_linear_cuda / dsv41_dense3_linear_cuda's table of one launch.
pub fn linTable(lay: Layout, l: Launch, segs: []const Seg, bufs: []const SegBufs, y_dtype: DType, xh: u64, z: u64, counters: u64) LinTable {
    var t = std.mem.zeroes(LinTable);
    t.n = @intCast(l.count);
    const esz = dtypeSize(y_dtype);
    for (l.segs[0..l.count], 0..) |i, j| {
        const s = segs[i];
        const st = strides(s.k, s.k2);
        t.s[j] = .{
            .xh = xh + 2 * lay.xh_off[i],
            .T = bufs[i].T,
            .stride_k = st[0],
            .stride_nb = st[1],
            .svh = bufs[i].svh,
            .y = bufs[i].y + s.col * esz,
            .Z = if (s.sk > 1) z + 4 * lay.z_off[i] else 0,
            .counters = counters + 4 * lay.c_off[i],
            .K = @intCast(s.k),
            .N = @intCast(s.n),
            .SK = @intCast(s.sk),
            .wk = @intCast(s.wk),
            .p0 = @intCast(l.p0[j]),
            .ld = @intCast(bufs[i].ld),
        };
    }
    return t;
}

// ---------------------------------------------------------------------------------------------------------------
// Symbols

pub const linear_k2s = [_]u32{ 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16 };

fn linearSymbol(comptime k2: u32, comptime wk: u32) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN14tf_exl3_linear13linear_kernelILi{d}ELi2ELi{d}EEEvPK6__halfPKjxxS3_S3_PviPfPiiiiiPKi", .{ k2, wk });
}
fn unpackSymbol(comptime k2: u32) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN14tf_exl3_linear13unpack_kernelILi{d}ELi2EEEvPKjP6__halfill", .{k2});
}
fn segSymbol(comptime k2: u32, comptime wk: u32, comptime mb: u32) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN11dsv41_x3seg17seg_linear_kernelILi{d}ELi2ELi{d}ELi{d}EEEvNS_8LinTableEii", .{ k2, wk, mb });
}
fn lanesSymbol(comptime k2: u32, comptime wk: u32, comptime mb: u32) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN12dsv41_dense319lanes_linear_kernelILi{d}ELi2ELi{d}ELi{d}EEEvN11dsv41_x3seg8LinTableEii", .{ k2, wk, mb });
}
fn lanesUnpackSymbol(comptime k2: u32) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN12dsv41_dense319lanes_unpack_kernelILi{d}ELi2EEEvPKjP6__halfill", .{k2});
}

/// x3seg's (WK, MINB) instances (TF_SEG_WIDTH: WK 8 takes any minb) and dense3's (TF_D3_WIDTH).
pub const seg_cfgs = [_][2]u32{ .{ 4, 3 }, .{ 4, 2 }, .{ 8, 1 } };
pub const lanes_cfgs = [_][2]u32{ .{ 4, 3 }, .{ 4, 4 }, .{ 8, 1 }, .{ 8, 2 } };

pub const Functions = struct {
    rot_in: cuda.Function,
    linear: [linear_k2s.len][3]cuda.Function, // WK 2, 4, 8
    unpack: [linear_k2s.len]cuda.Function,
    seg_rot_in: cuda.Function,
    seg: [x3seg_k2s.len][seg_cfgs.len]cuda.Function,
    lanes: [dense3_k2s.len][lanes_cfgs.len]cuda.Function,
    lanes_unpack: [dense3_k2s.len]cuda.Function,
    to_lanes: cuda.Function,
    to_strips: cuda.Function,

    pub fn resolve(linear: cuda.Module, x3seg: cuda.Module, dense3: cuda.Module) !Functions {
        var f: Functions = undefined;
        f.rot_in = try linear.function("_ZN14tf_exl3_linear13rot_in_kernelEPKviPK6__halfPS2_i");
        inline for (linear_k2s, 0..) |k2, i| {
            inline for (.{ 2, 4, 8 }, 0..) |wk, j| f.linear[i][j] = try linear.function(linearSymbol(k2, wk));
            f.unpack[i] = try linear.function(unpackSymbol(k2));
        }
        f.seg_rot_in = try x3seg.function("_ZN11dsv41_x3seg17seg_rot_in_kernelENS_8RotTableE");
        inline for (x3seg_k2s, 0..) |k2, i| inline for (seg_cfgs, 0..) |c, j| {
            f.seg[i][j] = try x3seg.function(segSymbol(k2, c[0], c[1]));
        };
        inline for (dense3_k2s, 0..) |k2, i| {
            inline for (lanes_cfgs, 0..) |c, j| f.lanes[i][j] = try dense3.function(lanesSymbol(k2, c[0], c[1]));
            f.lanes_unpack[i] = try dense3.function(lanesUnpackSymbol(k2));
        }
        f.to_lanes = try dense3.function("_ZN12dsv41_dense315to_lanes_kernelEPKjPjxii");
        f.to_strips = try dense3.function("_ZN12dsv41_dense316to_strips_kernelEPKjPjxii");
        return f;
    }
};

fn dim(x: usize) u32 {
    return @intCast(x);
}

fn int(x: usize) c_int {
    return @intCast(x);
}

fn go(f: cuda.Function, s: cuda.Stream, grid: [3]usize, block: usize, shared: usize, pdl: bool, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = dim(grid[0]), .y = dim(grid[1]), .z = dim(grid[2]) }, .block = .{ .x = dim(block) }, .shared = dim(shared), .pdl = pdl }, s, a);
}

/// WK * min(M, 8) * 128 floats: the linear kernels' reduction buffer (at most 32 KiB: no opt-in needed).
fn redSmem(wk: usize, m: usize) usize {
    return wk * @min(m, 8) * 128 * 4;
}

pub const Ops = struct {
    f: *const Functions,
    s: cuda.Stream,

    /// exl3_rot_in_cuda: xh [M, K] fp16 = (x * suh) H / sqrt 128 by 128-blocks.
    pub fn rotIn(o: Ops, x: u64, x_dtype: DType, suh: u64, xh: u64, M: usize, K: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@intFromEnum(x_dtype));
        a.add(suh);
        a.add(xh);
        a.add(int(K));
        try go(o.f.rot_in, o.s, .{ (K / 128 + 3) / 4, M, 1 }, 128, 0, false, &a);
    }

    /// exl3_linear_cuda (mul1): y [M, N] from xh [M, K]; `z` fp32 [SK, M, N] when SK > 1; `counters` int32
    /// [8 N / 128] left zero; `skip` an int32 device flag or 0.
    pub fn linear(o: Ops, xh: u64, T: u64, stride_k: i64, stride_nb: i64, svh: u64, bias: u64, y: u64, y_dtype: DType, z: u64, counters: u64, M: usize, K: usize, N: usize, k2: u32, SK: usize, WK: usize, skip: u64) !void {
        const ki = std.mem.indexOfScalar(u32, &linear_k2s, k2) orelse return error.Unsupported;
        const wi: usize = switch (WK) {
            2 => 0,
            4 => 1,
            8 => 2,
            else => return error.Unsupported,
        };
        if (SK == 0 or (K / 16) % (SK * WK) != 0) return error.Shape;
        if (SK > 1 and z == 0) return error.Shape;
        if (M < 1 or M > rows_max) return error.Shape;
        var a: cuda.Args = .{};
        for ([_]u64{ xh, T }) |v| a.add(v);
        a.add(stride_k);
        a.add(stride_nb);
        for ([_]u64{ svh, bias, y }) |v| a.add(v);
        a.add(@intFromEnum(y_dtype));
        for ([_]u64{ z, counters }) |v| a.add(v);
        for ([_]usize{ M, K, N, SK }) |v| a.add(int(v));
        a.add(skip);
        try go(o.f.linear[ki][wi], o.s, .{ N / 128, SK, 1 }, WK * 32, redSmem(WK, M), false, &a);
    }

    /// exl3_unpack_cuda (mul1): W_q [K, N] fp16.
    pub fn unpack(o: Ops, T: u64, W: u64, K: usize, N: usize, stride_k: i64, stride_nb: i64, k2: u32) !void {
        const ki = std.mem.indexOfScalar(u32, &linear_k2s, k2) orelse return error.Unsupported;
        var a: cuda.Args = .{};
        a.add(T);
        a.add(W);
        a.add(int(N));
        a.add(stride_k);
        a.add(stride_nb);
        try go(o.f.unpack[ki], o.s, .{ N / 16, K / 16, 1 }, 32, 0, false, &a);
    }

    /// dsv41_x3seg_rot_in_cuda: up to 8 (input, suh) pairs into one xh (each at its xh offset), one launch.
    pub fn segRotIn(o: Ops, tab: RotTable, M: usize, pdl: bool) !void {
        if (tab.n < 1 or tab.n > max_seg) return error.Shape;
        for (tab.s[0..@intCast(tab.n)]) |s| {
            if (s.nw != 0 and s.x_dtype != @intFromEnum(DType.bf16)) return error.Shape; // the fold takes bf16 rows
        }
        var kmax: usize = 0;
        for (tab.s[0..@intCast(tab.n)]) |s| kmax = @max(kmax, @as(usize, @intCast(s.K)));
        var a: cuda.Args = .{};
        a.add(tab);
        try go(o.f.seg_rot_in, o.s, .{ (kmax / 128 + 3) / 4, M, @intCast(tab.n) }, 128, 0, pdl, &a);
    }

    /// dsv41_x3seg_linear_cuda: one launch of a layout (words in the strips layout).
    pub fn segLinear(o: Ops, tab: LinTable, y_dtype: DType, M: usize, l: Launch, minb: u32, pdl: bool) !void {
        try o.segLinearRaw(tab, y_dtype, M, l.programs, l.k2, l.wk, minb, pdl);
    }

    /// segLinear with the binding's own (programs, K2, WK, minb): WK 8 takes any minb (TF_SEG_WIDTH).
    pub fn segLinearRaw(o: Ops, tab: LinTable, y_dtype: DType, M: usize, programs: usize, k2: u32, wk: usize, minb: u32, pdl: bool) !void {
        const ki = std.mem.indexOfScalar(u32, &x3seg_k2s, k2) orelse return error.Unsupported;
        const ci: usize = if (wk == 8) 2 else if (wk == 4 and minb == 3) 0 else if (wk == 4 and minb == 2) 1 else return error.Unsupported;
        try o.tableLaunch(o.f.seg[ki][ci], tab, y_dtype, M, programs, wk, pdl);
    }

    /// dsv41_dense3_linear_cuda: the same programs on words in the lanes layout.
    pub fn lanesLinear(o: Ops, tab: LinTable, y_dtype: DType, M: usize, l: Launch, minb: u32, pdl: bool) !void {
        try o.lanesLinearRaw(tab, y_dtype, M, l.programs, l.k2, l.wk, minb, pdl);
    }

    /// lanesLinear with the binding's own arguments; every segment's k steps a warp must split into whole load
    /// groups (dense3.why_not).
    pub fn lanesLinearRaw(o: Ops, tab: LinTable, y_dtype: DType, M: usize, programs: usize, k2: u32, wk: usize, minb: u32, pdl: bool) !void {
        const ki = std.mem.indexOfScalar(u32, &dense3_k2s, k2) orelse return error.Unsupported;
        const want = [2]u32{ @intCast(wk), minb };
        var ci: ?usize = null;
        for (lanes_cfgs, 0..) |c, j| if (std.mem.eql(u32, &c, &want)) {
            ci = j;
        };
        for (tab.s[0..@intCast(tab.n)]) |s| {
            const per_warp: usize = @intCast(@divTrunc(@divTrunc(@divTrunc(s.K, 16), s.SK), s.wk));
            if (per_warp % groupSteps(k2) != 0) return error.Shape;
        }
        try o.tableLaunch(o.f.lanes[ki][ci orelse return error.Unsupported], tab, y_dtype, M, programs, wk, pdl);
    }

    fn tableLaunch(o: Ops, f: cuda.Function, tab: LinTable, y_dtype: DType, M: usize, programs: usize, wk: usize, pdl: bool) !void {
        if (M < 1 or M > rows_max or tab.n < 1 or tab.n > max_seg) return error.Shape;
        var a: cuda.Args = .{};
        a.add(tab);
        a.add(@intFromEnum(y_dtype));
        a.add(int(M));
        try go(f, o.s, .{ programs, 1, 1 }, wk * 32, redSmem(wk, M), pdl, &a);
    }

    /// dsv41_dense3_relayout_cuda: int32 words of one matrix ([N/128, K/16, 8, 4 K2]) to the lanes layout or back.
    pub fn relayout(o: Ops, src: u64, dst: u64, words: usize, k2: u32, K: usize, lanes: bool) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(@as(i64, @intCast(words)));
        a.add(@as(c_int, @intCast(k2)));
        a.add(int(K / 16));
        try go(if (lanes) o.f.to_lanes else o.f.to_strips, o.s, .{ (words + 255) / 256, 1, 1 }, 256, 0, false, &a);
    }

    /// dsv41_dense3_unpack_cuda: W_q [K, N] fp16 from lanes-layout words.
    pub fn lanesUnpack(o: Ops, T: u64, W: u64, K: usize, N: usize, stride_k: i64, stride_nb: i64, k2: u32) !void {
        const ki = std.mem.indexOfScalar(u32, &dense3_k2s, k2) orelse return error.Unsupported;
        var a: cuda.Args = .{};
        a.add(T);
        a.add(W);
        a.add(int(N));
        a.add(stride_k);
        a.add(stride_nb);
        try go(o.f.lanes_unpack[ki], o.s, .{ N / 16, K / 16, 1 }, 32, 0, false, &a);
    }
};

test "linear.plan as upstream" {
    // wide layers: 8 warps, no split; narrow ones split K until 8 k tiles a warp would not hold
    try std.testing.expectEqual([2]usize{ 1, 8 }, plan(1024, 128 * 64));
    try std.testing.expectEqual([2]usize{ 8, 4 }, plan(4096, 1024));
    try std.testing.expectEqual([2]usize{ 1, 4 }, plan(512, 4096));
}

test "rot_in tables: one launch a group of 8, xh offsets from the layout" {
    var segs: [10]Seg = undefined;
    var ins: [10]RotIn = undefined;
    for (&segs, &ins, 0..) |*s, *r, i| {
        s.* = .{ .k = 1024, .n = 256, .k2 = 10, .sk = 1, .wk = 4 };
        r.* = .{ .x = 0x1000 * (i + 1), .ld = 1024, .dtype = .bf16, .suh = 0x100 * (i + 1) };
    }
    const lay = try layout(&segs, 2);
    const t0 = rotTable(lay, &segs, &ins, 0, 0x10000, 0);
    const t1 = rotTable(lay, &segs, &ins, 8, 0x10000, 0);
    try std.testing.expectEqual(@as(c_int, 8), t0.n);
    try std.testing.expectEqual(@as(c_int, 2), t1.n);
    try std.testing.expectEqual(@as(u64, 0x10000 + 2 * 8 * 2 * 1024), t1.s[0].xh);
    try std.testing.expectEqual(@as(c_int, @intFromEnum(DType.bf16)), t1.s[1].x_dtype);
}

test "fused_proj.layout: offsets, one launch a width, 8 segments a launch" {
    const segs = [_]Seg{
        .{ .k = 4096, .n = 1024, .k2 = 10, .sk = 4, .wk = 4 },
        .{ .k = 4096, .n = 512, .k2 = 12, .sk = 8, .wk = 4, .out = 1 },
        .{ .k = 4096, .n = 256, .k2 = 10, .sk = 8, .wk = 8, .out = 0, .col = 1024 },
    };
    const lay = try layout(&segs, 3);
    try std.testing.expectEqual(@as(usize, 3 * 4096 * 3), lay.xh_total);
    try std.testing.expectEqual(@as(usize, 4 * 3 * 1024 + 8 * 3 * 512 + 8 * 3 * 256), lay.z_total);
    try std.testing.expectEqual(@as(usize, 8 * (8 + 4 + 2)), lay.c_total);
    try std.testing.expectEqual(@as(usize, 2), lay.n_launches);
    const l0 = lay.launches[0];
    try std.testing.expectEqual(@as(u32, 10), l0.k2);
    try std.testing.expectEqual(@as(usize, 8), l0.wk);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2 }, l0.segs[0..l0.count]);
    try std.testing.expectEqualSlices(usize, &.{ 0, 32 }, l0.p0[0..l0.count]);
    try std.testing.expectEqual(@as(usize, 32 + 16), l0.programs);
    try std.testing.expectError(error.Shape, layout(&segs, 129));
}
