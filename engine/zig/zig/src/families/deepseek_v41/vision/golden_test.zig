//! The image front end against Pillow 12.3 + the prod engine's ``vision_prep`` (8474f31): every golden image decoded
//! to the same RGB bytes, planned to the same grids, preprocessed to the same bf16 patches and digest, given the same
//! virtual ids. The committed small set always runs; TF_DSV41_VISION_GOLDEN names a fuller set
//! (tools/zig/dsv41_vision/gen_prep_golden.py).
const std = @import("std");
const testing = std.testing;
const decode = @import("decode.zig");
const prep = @import("prep.zig");
const vids = @import("vids.zig");

const fixtures = "zig/src/families/deepseek_v41/vision/fixtures";

const Rec = struct {
    name: []const u8,
    size: [2]u32,
    rgb_sha256: []const u8,
    grid: [4]u32,
    patches_sha256: []const u8,
    digest: []const u8,
    vids_head: []const u32,
    vids_sha256: []const u8,
};

fn hex(buf: *[64]u8, bytes: []const u8) []const u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bufPrint(buf, "{x}", .{d}) catch unreachable;
}

/// Every record of ``dir``/golden.jsonl; returns (equal, total).
fn run(dir: []const u8) ![2]usize {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    const cwd = std.Io.Dir.cwd();
    const cfg = try cwd.readFileAlloc(testing.io, fixtures ++ "/config.json", ar, .limited(1 << 20));
    var problem: []const u8 = "";
    var s = try prep.Settings.read(ar, cfg, null, &problem);
    s.bias_vl = std.mem.indexOf(u8, dir, "bias") != null; // the generator's --bias-vl sets (gold-bias)
    const lines = try cwd.readFileAlloc(testing.io, try std.fmt.allocPrint(ar, "{s}/golden.jsonl", .{dir}), ar, .limited(1 << 26));
    var reg: vids.Registry = .{ .gpa = a };
    defer reg.deinit();
    var ok: usize = 0;
    var total: usize = 0;
    var it = std.mem.tokenizeScalar(u8, lines, '\n');
    while (it.next()) |line| {
        total += 1;
        const r = try std.json.parseFromSliceLeaky(Rec, ar, line, .{});
        // a set without its images (golden-bias) reads the committed set's
        const raw = cwd.readFileAlloc(testing.io, try std.fmt.allocPrint(ar, "{s}/{s}", .{ dir, r.name }), ar, .limited(1 << 26)) catch
            try cwd.readFileAlloc(testing.io, try std.fmt.allocPrint(ar, fixtures ++ "/golden/{s}", .{r.name}), ar, .limited(1 << 26));
        var which: []const u8 = "";
        var rgb = decode.rgb(a, raw, s.max_pixels, &which) catch |e| {
            std.debug.print("vision golden {s}: decode failed ({t}, {s})\n", .{ r.name, e, which });
            continue;
        };
        defer rgb.deinit(a);
        var hb: [64]u8 = undefined;
        if (rgb.w != r.size[0] or rgb.h != r.size[1] or !std.mem.eql(u8, hex(&hb, rgb.data), r.rgb_sha256)) {
            std.debug.print("vision golden {s} ({s}): RGB differs from Pillow's\n", .{ r.name, which });
            continue;
        }
        var p = try prep.preprocess(a, rgb, &s);
        defer p.deinit(a);
        p.vids = try reg.vids(testing.io, &p.digest, p.tokens());
        const grid = [4]u32{ p.llm_h, p.llm_w, p.vit_h, p.vit_w };
        var db: [64]u8 = undefined;
        const digest = std.fmt.bufPrint(&db, "{x}", .{p.digest}) catch unreachable;
        var vb: std.ArrayList(u8) = .empty;
        for (p.vids) |v| try vb.appendSlice(ar, std.mem.asBytes(&std.mem.nativeToLittle(u32, v)));
        var vh: [64]u8 = undefined;
        if (!std.mem.eql(u32, &grid, &r.grid)) {
            std.debug.print("vision golden {s}: grid {any} != {any}\n", .{ r.name, grid, r.grid });
        } else if (!std.mem.eql(u8, hex(&hb, std.mem.sliceAsBytes(p.patches)), r.patches_sha256)) {
            std.debug.print("vision golden {s}: patches differ\n", .{r.name});
        } else if (!std.mem.eql(u8, digest, r.digest)) {
            std.debug.print("vision golden {s}: digest differs\n", .{r.name});
        } else if (!std.mem.eql(u8, hex(&vh, vb.items), r.vids_sha256)) {
            std.debug.print("vision golden {s}: virtual ids differ (first {any} vs {any})\n", .{ r.name, p.vids[0..@min(4, p.vids.len)], r.vids_head });
        } else ok += 1;
    }
    return .{ ok, total };
}

test "images decode and preprocess as Pillow + vision_prep (committed set)" {
    for ([_][]const u8{ fixtures ++ "/golden", fixtures ++ "/golden-bias" }) |dir| {
        const got = try run(dir);
        std.debug.print("vision golden {s}: {d}/{d} images equal\n", .{ dir, got[0], got[1] });
        try testing.expect(got[1] > 0);
        try testing.expectEqual(got[1], got[0]);
    }
}

test "images decode and preprocess as Pillow + vision_prep (TF_DSV41_VISION_GOLDEN)" {
    const dir = testing.environ.getPosix("TF_DSV41_VISION_GOLDEN") orelse return error.SkipZigTest;
    const got = try run(dir);
    std.debug.print("vision golden {s}: {d}/{d} images equal\n", .{ dir, got[0], got[1] });
    try testing.expectEqual(got[1], got[0]);
}
