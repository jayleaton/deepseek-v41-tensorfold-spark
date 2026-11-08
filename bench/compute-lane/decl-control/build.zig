const std = @import("std");
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const opt: std.builtin.OptimizeMode = .ReleaseSafe;
    const j = b.createModule(.{ .root_source_file = b.path("native/json.zig"), .target = target, .optimize = opt });
    const u = b.createModule(.{ .root_source_file = b.path("native/unicode_root.zig"), .target = target, .optimize = opt });
    const api = b.createModule(.{ .root_source_file = b.path("native/api_stub.zig"), .target = target, .optimize = opt });
    const template = b.createModule(.{ .root_source_file = b.path("native/template.zig"), .target = target, .optimize = opt, .imports = &.{.{ .name = "json", .module = j }} });
    const tok = b.createModule(.{ .root_source_file = b.path("native/tokenizer.zig"), .target = target, .optimize = opt, .imports = &.{.{ .name = "tokenizer", .module = u }} });
    const turns = b.createModule(.{ .root_source_file = b.path("native/compute_turns.zig"), .target = target, .optimize = opt, .imports = &.{ .{ .name = "json", .module = j }, .{ .name = "engine_api", .module = api } } });
    const m = b.createModule(.{ .root_source_file = b.path("prompt_gate.zig"), .target = target, .optimize = opt, .link_libc = true, .imports = &.{ .{ .name = "json", .module = j }, .{ .name = "template", .module = template }, .{ .name = "native_tokenizer", .module = tok }, .{ .name = "turns", .module = turns } } });
    b.installArtifact(b.addExecutable(.{ .name = "prompt-gate", .root_module = m }));
}
