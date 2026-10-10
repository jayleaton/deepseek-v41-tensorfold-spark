//! DeepSeek-V4.1's keyed sampling on the GPU path: `test-dsv41-sampling` (the pipeline on in-process ranks against the
//! Python engine's choices, any host) and, on Linux with kernels, `tf-dsv41-samp` (the device steps against the
//! host reference on one GPU).

const std = @import("std");
const server_build = @import("server.zig");
const tp_build = @import("tp.zig");

/// The CUDA runtime without the engine's kernels (tp's in-process ranks need only its types).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const options = b.addOptions();
    options.addOption(bool, "with_kernels", false);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    cuda.addImport("stagger", b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize }));
    return cuda;
}

/// DeepSeek-V4.1's serving module (the exact sampler) with the modules it needs, over `lanes`.
pub fn serve(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, lanes: *std.Build.Module) struct { serve: *std.Build.Module, json: *std.Build.Module } {
    const json = b.createModule(.{ .root_source_file = b.path("zig/src/server/json.zig"), .target = target, .optimize = optimize });
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    return .{ .serve = server_build.serveModule(b, target, optimize, json, tokenizer, lanes), .json = json };
}

pub fn add(b: *std.Build, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = host, .optimize = .debug, .link_libc = true });
    const s = serve(b, host, .debug, lanes);
    const m = b.createModule(.{
        .root_source_file = b.path("zig/src/families/deepseek_v41/sampling_test.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "tp", .module = tp_build.module(b, host, .debug, runtime(b, host, .debug)) },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "dsv41_serve", .module = s.serve },
            .{ .name = "json", .module = s.json },
        },
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-dsv41-sampling", "DeepSeek-V4.1's keyed sampling pipeline on in-process ranks vs the Python engine's choices (no GPU)").dependOn(&run.step);
    test_step.dependOn(&run.step);
}

/// `tf-dsv41-samp` (Linux): the device steps on one GPU against the host reference (zig/build/cuda.zig passes its
/// runtime and kernels modules).
pub fn gpu(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, kernels: *std.Build.Module) void {
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const m = b.createModule(.{
        .root_source_file = b.path("zig/src/families/deepseek_v41/sampling_check.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "cuda", .module = cuda },
            .{ .name = "dsv41_kernels", .module = kernels },
            .{ .name = "tp", .module = tp_build.module(b, target, optimize, cuda) },
            .{ .name = "dsv41_serve", .module = serve(b, target, optimize, lanes).serve },
        },
    });
    b.installArtifact(b.addExecutable(.{ .name = "tf-dsv41-samp", .root_module = m }));
}
