//! DeepSeek-V4.1's image front end (zig/src/families/deepseek_v41/vision, module `dsv41_vision`, host only): image
//! parts, PNG / JPEG decoding, the processor, virtual ids. `test-dsv41-vision`: its goldens against Pillow and the
//! prod engine's ``vision_prep`` (committed small set; TF_DSV41_VISION_GOLDEN: a fuller set from
//! tools/zig/dsv41_vision/gen_prep_golden.py).
const std = @import("std");

pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, json: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{
        .root_source_file = b.path("zig/src/families/deepseek_v41/vision/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "json", .module = json }},
    });
}

pub fn add(b: *std.Build, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    const json = b.createModule(.{ .root_source_file = b.path("zig/src/server/json.zig"), .target = host, .optimize = .debug });
    const t = b.addTest(.{ .root_module = module(b, host, .debug, json) });
    const run = b.addRunArtifact(t);
    run.setCwd(b.path("."));
    b.step("test-dsv41-vision", "DeepSeek-V4.1's image front end vs Pillow + the prod engine's vision_prep goldens (no GPU)").dependOn(&run.step);
    test_step.dependOn(&run.step);
}
