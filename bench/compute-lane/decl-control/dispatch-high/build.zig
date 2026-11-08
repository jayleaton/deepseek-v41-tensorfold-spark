const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    if (target.result.os.tag != .linux or target.result.abi != .gnu)
        @panic("declaration control requires Linux glibc for the spawn interposer");
    const opt: std.builtin.OptimizeMode = .ReleaseSafe;
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = opt, .link_libc = true });
    const root = @import("zig/build/server.zig").plain(b, "zig/src/server/decl_control_probe.zig", target, opt, lanes);
    root.addCSourceFile(.{ .file = b.path("dispatch_spy.c"), .flags = &.{ "-std=c11", "-Wall", "-Wextra", "-Werror" } });
    root.linkSystemLibrary("dl", .{});
    const sandbox = b.addExecutable(.{ .name = "declaration-control-sandbox", .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/compute_sandbox_main.zig"), .target = target, .optimize = opt, .link_libc = true }) });
    const options = b.addOptions();
    options.addOptionPath("sandbox_exe", sandbox.getEmittedBin());
    root.addOptions("decl_control_options", options);
    const probe = b.addTest(.{ .root_module = root, .filters = &.{"declaration control:"} });
    // Keep an explicit build-only step; the parent owns execution and confinement.
    b.step("compile", "Compile only the declaration control and real sandbox").dependOn(&probe.step);
    const run = b.addRunArtifact(probe);
    run.has_side_effects = true;
    b.step("check", "Run only declaration control forced-call checks").dependOn(&run.step);
    b.default_step.dependOn(&probe.step);
    @import("zig/build/sys.zig").wire(b);
}
