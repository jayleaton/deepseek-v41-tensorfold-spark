const std = @import("std");
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const opt: std.builtin.OptimizeMode = .ReleaseSafe;
    const root = b.option([]const u8, "source", "Pinned source tree") orelse @panic("-Dsource required");
    const tok = b.createModule(.{ .root_source_file = .{ .cwd_relative = b.fmt("{s}/zig/src/core/tokenizer/tokenizer.zig", .{root}) }, .target = target, .optimize = opt });
    const j = b.createModule(.{ .root_source_file = .{ .cwd_relative = b.fmt("{s}/zig/src/server/json.zig", .{root}) }, .target = target, .optimize = opt });
    const m = b.createModule(.{ .root_source_file = b.path("cap_main.zig"), .target = target, .optimize = opt, .link_libc = true, .imports = &.{ .{ .name = "json", .module = j }, .{ .name = "tokenizer", .module = tok } } });
    b.installArtifact(b.addExecutable(.{ .name = "native-cap", .root_module = m }));
}
