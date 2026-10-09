//! GPU test runner for DeepSeek-V4.1's kernels: `tf-dsv41-test <command> [fixture dir]`, PASS/FAIL lines, exit 1 on
//! failure. Fixtures come from zig/tests/cuda/deepseek_v41/oracle.py (the Python engine's own kernels on the same
//! inputs); every check is bit for bit.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("check.zig");
const experts = @import("deepseek_v41/experts.zig");
const kvglue = @import("deepseek_v41/kv_glue.zig");
const kvsplit = @import("deepseek_v41/kvsplit.zig");
const mhcdec = @import("deepseek_v41/mhcdec.zig");
const mhc_alloc = @import("deepseek_v41/mhc_alloc_bench.zig");
const mhc_probe = @import("deepseek_v41/mhc_probe.zig");
const replay_case = @import("deepseek_v41/replay.zig");
const x3gm3_bench = @import("deepseek_v41/x3gm3_bench.zig");

const usage =
    \\usage: tf-dsv41-test <command> [fixture dir]
    \\  symbols            load every DeepSeek-V4.1 fatbin and resolve every instance
    \\  kvglue <dir>      fused KV RMS/SWA store against Python Triton bytes
    \\  kvsplit            split KV's exchange copies against kv/split.zig's host reference (no fixture)
    \\  experts <dir>      decode chain: group, rot_in, x3ld (every setting) / upstream grouped / x3pf, epilogues, combine
    \\  x3gm <dir>         fast prefill: rot, gate/up and down at every configuration (uniform and ragged), combine
    \\  x3gm3-bench [rows,..] [reps]  x3gm v3 against v2 at prod's prefill shapes: bits and time (synthetic data)
    \\  mhc-alloc [windows]  the 16-row mHC boundary's time with buffers per cuMemAlloc / arena / VMM (TF_DSV41_ARENA)
    \\  mhc-probe [windows]  the 16-row mHC boundary under cache states, buffer layouts and a cross-stream dependency
    \\  replay <dir>       one recorded Python extension call (dsv41_capture.py) re-issued and compared, storage by storage
    \\  all <fixtures>     every case directory under <fixtures> by its manifest's "kind"
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const gpu: check.Gpu = .{ .d = &driver, .ctx = &ctx, .gpa = init.gpa, .io = init.io };
    run(gpu, args[1], args[2..]) catch |e| {
        std.debug.print("FAIL {s}: {t}\n", .{ args[1], e });
        return 1;
    };
    return 0;
}

fn arg(rest: []const [:0]const u8, i: usize) ![]const u8 {
    if (i >= rest.len) {
        std.debug.print("{s}", .{usage});
        return error.MissingArgument;
    }
    return rest[i];
}

fn run(gpu: check.Gpu, cmd: []const u8, rest: []const [:0]const u8) !void {
    var k = try dsv41.Kernels.load(gpu.ctx);
    defer k.deinit();
    if (std.mem.eql(u8, cmd, "symbols")) {
        check.pass("dsv41 symbols: every instance resolved ({d} SMs)", .{k.sms});
        return;
    }
    if (std.mem.eql(u8, cmd, "kvsplit")) return kvsplit.run(gpu, &k);
    if (std.mem.eql(u8, cmd, "kvglue")) return kvglue.run(gpu, &k, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "experts")) return experts.decodeChain(gpu, &k, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "x3gm")) return experts.x3gm(gpu, &k, try arg(rest, 0));
    if (std.mem.eql(u8, cmd, "mhc-probe")) return mhc_probe.run(gpu, &k, if (rest.len > 0) rest[0] else null);
    if (std.mem.eql(u8, cmd, "mhc-alloc")) return mhc_alloc.run(gpu, &k, if (rest.len > 0) rest[0] else null);
    if (std.mem.eql(u8, cmd, "x3gm3-bench")) return x3gm3_bench.run(gpu, &k, if (rest.len > 0) rest[0] else null, if (rest.len > 1) rest[1] else null);
    if (std.mem.eql(u8, cmd, "replay")) {
        if (try replay_case.run(gpu, &k, try arg(rest, 0)) == .skipped) std.debug.print("SKIP no binding\n", .{});
        return;
    }
    if (std.mem.eql(u8, cmd, "all")) return all(gpu, &k, try arg(rest, 0));
    std.debug.print("{s}", .{usage});
    return error.UnknownCommand;
}

/// Every case under `root` (one directory a case, its manifest's "kind" naming the check); fails if any fails.
fn all(gpu: check.Gpu, k: *const dsv41.Kernels, root: []const u8) !void {
    var dir = try std.Io.Dir.cwd().openDir(gpu.io, root, .{ .iterate = true });
    defer dir.close(gpu.io);
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |n| gpu.gpa.free(n);
        names.deinit(gpu.gpa);
    }
    var it = dir.iterate();
    while (try it.next(gpu.io)) |e| {
        if (e.kind == .directory) try names.append(gpu.gpa, try gpu.gpa.dupe(u8, e.name));
    }
    std.mem.sort([]u8, names.items, {}, struct {
        fn lt(_: void, a: []u8, b: []u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);
    var failed: usize = 0;
    var skipped: usize = 0;
    for (names.items) |n| {
        const path = try std.fs.path.join(gpu.gpa, &.{ root, n });
        defer gpu.gpa.free(path);
        const kind = experts.caseKind(gpu, path) catch |e| {
            std.debug.print("FAIL {s}: no manifest ({t})\n", .{ n, e });
            failed += 1;
            continue;
        };
        const r = if (std.mem.eql(u8, kind.slice(), "experts"))
            experts.decodeChain(gpu, k, path)
        else if (std.mem.eql(u8, kind.slice(), "kvglue"))
            kvglue.run(gpu, k, path)
        else if (std.mem.eql(u8, kind.slice(), "x3gm"))
            experts.x3gm(gpu, k, path)
        else if (std.mem.eql(u8, kind.slice(), "plan"))
            experts.planCases(gpu, k, path)
        else if (std.mem.eql(u8, kind.slice(), "topk"))
            experts.topkCases(gpu, k, path)
        else if (std.mem.eql(u8, kind.slice(), "pointwise"))
            experts.pointwiseCases(gpu, k, path)
        else if (std.mem.eql(u8, kind.slice(), "glue"))
            experts.glueCases(gpu, k, path)
        else if (std.mem.eql(u8, kind.slice(), "pfglue"))
            experts.pfglueCases(gpu, k, path)
        else if (std.mem.eql(u8, kind.slice(), "mhcdec"))
            mhcdec.cases(gpu, k, path)
        else if (std.mem.eql(u8, kind.slice(), "replay")) blk: {
            const o = replay_case.run(gpu, k, path) catch |e| break :blk e;
            if (o == .skipped) {
                std.debug.print("SKIP {s}: no Zig binding\n", .{n});
                skipped += 1;
            }
            break :blk {};
        } else error.UnknownKind;
        r catch |e| {
            std.debug.print("FAIL {s}: {t}\n", .{ n, e });
            failed += 1;
        };
    }
    std.debug.print("RESULT {d} cases, {d} failed, {d} skipped (no binding)\n", .{ names.items.len, failed, skipped });
    if (failed > 0) return error.TestFailed;
}
