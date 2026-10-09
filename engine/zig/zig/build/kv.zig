//! KV memory's build: the family-neutral `sessions` module (zig/src/sessions) and DeepSeek-V4.1's `kv` module over it, with host tests on any OS.

const std = @import("std");

/// The CUDA runtime without the engine's kernels (kv needs the driver, memory and streams; tp's collectives need the same).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize }));
    return cuda;
}

pub fn sessions(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/sessions/sessions.zig"), .target = target, .optimize = optimize, .link_libc = true });
}

/// The `kv` module: imports sessions, cuda and tp (an engine passes its own runtime and tp modules).
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, tp: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/kv/kv.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("sessions", sessions(b, target, optimize));
    m.addImport("cuda", cuda);
    m.addImport("tp", tp);
    return m;
}

pub fn add(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, test_step: *std.Build.Step, tp_module: fn (*std.Build, std.Build.ResolvedTarget, std.builtin.OptimizeMode, *std.Build.Module) *std.Build.Module) void {
    const host = b.graph.host;
    const cuda = runtime(b, host, .debug);
    const s = b.addRunArtifact(b.addTest(.{ .root_module = sessions(b, host, .debug) }));
    const k = b.addRunArtifact(b.addTest(.{ .root_module = module(b, host, .debug, cuda, tp_module(b, host, .debug, cuda)) }));
    const step = b.step("test-kv", "KV memory's host tests: pool, sessions, NVMe tier on real files, split KV on in-process ranks (no GPU)");
    step.dependOn(&s.step);
    step.dependOn(&k.step);
    test_step.dependOn(&s.step);
    test_step.dependOn(&k.step);
    const bench = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/kv/bench.zig"), .target = target, .optimize = .ReleaseFast, .link_libc = true });
    bench.addImport("sessions", sessions(b, target, .ReleaseFast));
    const exe = b.addExecutable(.{ .name = "tf-kv-bench", .root_module = bench });
    b.step("tf-kv-bench", "Page ops and a 1M-token session parked to / restored from a real NVMe tier (host pool)").dependOn(&b.addInstallArtifact(exe, .{}).step);
    if (target.result.os.tag != .linux) return;
    const rt = runtime(b, target, optimize);
    const gpu = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/kv/gpu_main.zig"), .target = target, .optimize = optimize, .link_libc = true });
    gpu.addImport("sessions", sessions(b, target, optimize));
    gpu.addImport("cuda", rt);
    const gexe = b.addExecutable(.{ .name = "tf-kv-gpu-test", .root_module = gpu });
    b.step("tf-kv-gpu-test", "The device pool on one GPU: tables, a 1M-token session through device memory and the NVMe tier, copy on write").dependOn(&b.addInstallArtifact(gexe, .{}).step);
    const sg = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/kv/split_gpu.zig"), .target = target, .optimize = optimize, .link_libc = true });
    sg.addImport("sessions", sessions(b, target, optimize));
    sg.addImport("cuda", rt);
    sg.addImport("tp", tp_module(b, target, optimize, rt));
    const sexe = b.addExecutable(.{ .name = "tf-kv-split-test", .root_module = sg });
    b.step("tf-kv-split-test", "Split KV's exchange on two GPUs (one process a rank): dense, graph replays, unions vs the replicated rows").dependOn(&b.addInstallArtifact(sexe, .{}).step);
}
