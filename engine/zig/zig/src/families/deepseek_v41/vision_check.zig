//! `tf-dsv41-m1 vision PACK DIR`: the vision tower's gate. DIR is tools/zig/dsv41_vision/vision_ref.py's output: each
//! image's patches (Python's ``vision_prep.preprocess``), its stage dumps from the Python tower on the same GPU
//! (``<name>.<stage>.bin``: embed, block<i>, vit, gelu) and its span rows (``<name>.span.bin``). Each stage and the span
//! are compared bit for bit; a differing stage reports its first differing element, the count and the largest
//! difference, so the first op that leaves torch's bits is named. PASS when every image's span rows are equal.
//! Block 0 also dumps each op (``b0.norm1`` .. ``b0.w2``). TF_DSV41_VISION_RESYNC=1 tests the ops apart: after each
//! compared stage the buffer takes Python's bits, so every stage's count is that op's alone (the span then is not the
//! tower's: the PASS line stays the chained run's job).
const std = @import("std");
const cuda = @import("cuda");
const pack_mod = @import("pack.zig");
const tower_mod = @import("vision_tower.zig");
const rows_mod = @import("vision_rows.zig");

const Cmp = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    name: []const u8,
    resync: bool = false,
    bad_stages: usize = 0,
    first_bad: ?[]const u8 = null,

    fn at(ctx: *anyopaque, s: cuda.Stream, stage: []const u8, ptr: u64, bytes: usize) anyerror!void {
        const c: *Cmp = @ptrCast(@alignCast(ctx));
        var pb: [512]u8 = undefined;
        const path = try std.fmt.bufPrint(&pb, "{s}/{s}.{s}.bin", .{ c.dir, c.name, stage });
        const want = std.Io.Dir.cwd().readFileAlloc(c.io, path, c.gpa, .limited(1 << 31)) catch return; // not dumped
        defer c.gpa.free(want);
        const got = try c.gpa.alloc(u8, bytes);
        defer c.gpa.free(got);
        try s.synchronize();
        try cuda.DeviceBuffer.download(.{ .d = s.d, .ptr = ptr, .len = bytes }, 0, got);
        if (want.len != bytes) {
            std.debug.print("  {s} {s}: {d} bytes, Python {d}\n", .{ c.name, stage, bytes, want.len });
            c.bad_stages += 1;
            return;
        }
        const r = diff(got, want);
        if (r.count != 0 and c.resync) try cuda.DeviceBuffer.upload(.{ .d = s.d, .ptr = ptr, .len = bytes }, 0, want);
        if (r.count == 0) return;
        c.bad_stages += 1;
        if (c.first_bad == null) c.first_bad = try c.gpa.dupe(u8, stage);
        std.debug.print("  {s} {s}: {d} / {d} bf16 differ, first at {d}, max |diff| {e}\n", .{ c.name, stage, r.count, bytes / 2, r.first, r.max });
    }
};

fn bf(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

const Diff = struct { count: usize, first: usize, max: f32 };

fn diff(got: []const u8, want: []const u8) Diff {
    const a: []align(1) const u16 = std.mem.bytesAsSlice(u16, got);
    const b: []align(1) const u16 = std.mem.bytesAsSlice(u16, want);
    var r: Diff = .{ .count = 0, .first = 0, .max = 0 };
    for (a, b, 0..) |x, y, i| if (x != y) {
        if (r.count == 0) r.first = i;
        r.count += 1;
        r.max = @max(r.max, @abs(bf(x) - bf(y)));
    };
    return r;
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, pack_dir: []const u8, dir: []const u8) !u8 {
    var driver = try cuda.Driver.open();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var pack = try pack_mod.Pack.open(gpa, io, pack_dir);
    defer pack.deinit();
    const t = try tower_mod.Tower.load(gpa, io, &driver, &pack, .{});
    defer t.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const got = try rows_mod.readManifest(arena.allocator(), io, dir);
    const resync = if (std.c.getenv("TF_DSV41_VISION_RESYNC")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
    if (resync) std.debug.print("vision: TF_DSV41_VISION_RESYNC=1, each stage on Python's input (ops apart)\n", .{});
    var equal: usize = 0;
    for (got.held.slice(), got.m.images) |*img, e| {
        const np = @as(usize, img.vit_h) * img.vit_w * 3 * 14 * 14;
        var patches = try cuda.DeviceBuffer.fromHost(&driver, std.mem.sliceAsBytes(img.patches[0..np]));
        defer patches.free();
        const bytes = img.n_vids * t.shape.out * 2;
        var out = try cuda.DeviceBuffer.alloc(&driver, bytes);
        defer out.free();
        var cmp: Cmp = .{ .gpa = gpa, .io = io, .dir = dir, .name = e.name, .resync = resync };
        const t0 = std.Io.Clock.awake.now(io).nanoseconds;
        try t.spanTapped(stream, img, patches.ptr, out.ptr, .{ .ctx = &cmp, .at = Cmp.at });
        try Cmp.at(&cmp, stream, "span", out.ptr, bytes);
        const ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).nanoseconds - t0)) / 1e6;
        const ok = cmp.bad_stages == 0;
        equal += @intFromBool(ok);
        std.debug.print("vision {s}: {d} x {d} patches, span {d} rows, {s}{s}{s} ({d:.0} ms with the stage reads)\n", .{ e.name, img.vit_h, img.vit_w, img.n_vids, if (ok) "every stage equal" else "first differing stage ", if (ok) "" else cmp.first_bad orelse "span", "", ms });
    }
    const n = got.held.n;
    std.debug.print("{s} vision ViT + aligner == Python: {d}/{d} images\n", .{ if (equal == n and n > 0) "PASS" else "FAIL", equal, n });
    return if (equal == n and n > 0) 0 else 1;
}
