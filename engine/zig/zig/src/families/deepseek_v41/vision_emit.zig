//! `tf-dsv41-m1 vision-emit CAPTURE` (host only): the image routing bias's prefill program (block_prefill's moeSplit,
//! Python's vision.moe) against the programs the capture checks already hold to Python. For every captured prefill
//! segment of n rows and several image row counts ni, each layer's MoE must be: image_split, then the image rows'
//! call, then (ni < n) the text rows' call, then image_merge; and each call, launch by launch (names, grids, every
//! scalar, every tensor's dtype / shape / stride / offset and role), the MoE of a plain segment of ni (n - ni) rows:
//! prefill_moe.forward at that row count, the same function vision.moe calls. Only the input / output rows and the
//! bias (gate.bias_vl for the image call) differ by name; the text call's ticket is the layer's second.
const std = @import("std");
const calls = @import("calls.zig");
const block = @import("block.zig");
const block_prefill = @import("block_prefill.zig");
const check = @import("m1_check.zig");
const Config = @import("config.zig").Config;

const Region = struct { a: usize, b: usize }; // [route, moe_sum] inclusive

/// Each MoE call's span in `cs`: from its router launch to its moe_sum glue.
fn regions(a: std.mem.Allocator, cs: []const calls.Call) ![]Region {
    var out: std.ArrayList(Region) = .empty;
    var start: ?usize = null;
    for (cs, 0..) |c, i| {
        if (std.mem.eql(u8, c.name, "tf_dsv41_router_gemv_v2.route")) start = i;
        if (c.glue and std.mem.eql(u8, c.name, "glue.moe_sum")) {
            try out.append(a, .{ .a = start orelse return error.NoRouter, .b = i });
            start = null;
        }
    }
    return out.items;
}

fn canon(r: []const u8) []const u8 {
    for ([_][2][]const u8{ .{ "w.out.img", "w.out" }, .{ "w.out.txt", "w.out" }, .{ "L.moe16.img", "L.moe16" }, .{ "L.moe16.txt", "L.moe16" } }) |m|
        if (std.mem.eql(u8, r, m[0])) return m[1];
    return r;
}

fn sameRole(x: calls.Role, y: calls.Role) bool {
    return switch (x) {
        .empty => y == .empty,
        .buf => |b| y == .buf and std.mem.eql(u8, canon(b), canon(y.buf)),
        .weight => |w| y == .weight and (std.mem.eql(u8, w, y.weight) or
            (std.mem.endsWith(u8, w, ".moe.bias_vl") and std.mem.endsWith(u8, y.weight, ".moe.bias") and
            std.mem.eql(u8, w[0 .. w.len - 3], y.weight))),
    };
}

fn sameTensor(x: calls.Tensor, y: calls.Tensor) bool {
    return x.dt == y.dt and x.offset == y.offset and std.mem.eql(i64, x.shape, y.shape) and
        std.mem.eql(i64, x.stride, y.stride) and sameRole(x.role, y.role);
}

fn sameArg(x: calls.Arg, y: calls.Arg) bool {
    return switch (x) {
        .t => |t| y == .t and sameTensor(t, y.t),
        .opaque_table => |t| y == .opaque_table and sameTensor(t, y.opaque_table),
        .i => |v| y == .i and v == y.i,
        .f => |v| y == .f and @as(u64, @bitCast(v)) == @as(u64, @bitCast(y.f)),
        .b => |v| y == .b and v == y.b,
        .none => y == .none,
        .list => |l| y == .list and l.len == y.list.len and for (l, y.list) |p, q| {
            if (!sameArg(p, q)) break false;
        } else true,
    };
}

fn sameCall(x: calls.Call, y: calls.Call) bool {
    const nx = if (std.mem.eql(u8, x.name, "glue.gm_ticket_again")) "glue.gm_ticket" else x.name;
    if (!std.mem.eql(u8, nx, y.name) or x.triton != y.triton or x.glue != y.glue or !std.mem.eql(i64, &x.grid, &y.grid)) return false;
    if (x.args.len != y.args.len) return false;
    for (x.args, y.args) |p, q| if (!std.mem.eql(u8, p.name, q.name) or !sameArg(p.arg, q.arg)) return false;
    return true;
}

/// The split program's checks for one segment and image row count; returns the failures.
fn one(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, win: check.Window, ni: i64, log: *std.Io.Writer) !usize {
    var so = o;
    so.image_keep = true;
    so.image_rows = ni;
    const cs = try block_prefill.emitPrefill(a, cfg, w, so, win.layers, win.n, win.start, true);
    const nt = win.n - ni;
    const ref_img = try block_prefill.emitPrefill(a, cfg, w, o, win.layers, ni, win.start, true);
    const ref_txt: []const calls.Call = if (nt > 0) try block_prefill.emitPrefill(a, cfg, w, o, win.layers, nt, win.start, true) else &.{};
    const img = try regions(a, ref_img);
    const txt: []Region = if (nt > 0) try regions(a, ref_txt) else &.{};
    const got = try regions(a, cs);
    const per: usize = if (nt > 0) 2 else 1;
    var bad: usize = 0;
    if (got.len != per * win.layers.len or img.len != win.layers.len) {
        try log.print("  {s} n {d} ni {d}: {d} MoE calls, want {d}\n", .{ win.set, win.n, ni, got.len, per * win.layers.len });
        return 1;
    }
    for (win.layers, 0..) |L, li| {
        const gi = got[per * li];
        // image_split right before the image call's router, image_merge right after the last call's moe_sum
        if (gi.a == 0 or !std.mem.eql(u8, cs[gi.a - 1].name, "glue.image_split")) bad += 1;
        const last = got[per * li + per - 1];
        if (last.b + 1 >= cs.len or !std.mem.eql(u8, cs[last.b + 1].name, "glue.image_merge")) bad += 1;
        const pairs = [_]struct { g: Region, r: Region, ref: []const calls.Call }{
            .{ .g = gi, .r = img[li], .ref = ref_img },
            .{ .g = last, .r = if (nt > 0) txt[li] else img[li], .ref = if (nt > 0) ref_txt else ref_img },
        };
        for (pairs[0..per]) |p| {
            if (p.g.b - p.g.a != p.r.b - p.r.a) {
                try log.print("  {s} L{d} n {d} ni {d}: {d} launches, a plain segment's MoE {d}\n", .{ win.set, L, win.n, ni, p.g.b - p.g.a + 1, p.r.b - p.r.a + 1 });
                bad += 1;
                continue;
            }
            for (cs[p.g.a .. p.g.b + 1], p.ref[p.r.a .. p.r.b + 1]) |x, y| if (!sameCall(x, y)) {
                try log.print("  {s} L{d} n {d} ni {d}: {s} differs from a plain segment's", .{ win.set, L, win.n, ni, x.name });
                if (x.args.len == y.args.len) for (x.args, y.args, 0..) |q, r, k| if (!sameArg(q.arg, r.arg)) {
                    try log.print(" (argument {d}: {any} vs {any})", .{ k, q.arg, r.arg });
                    break;
                };
                try log.print("\n", .{});
                bad += 1;
                break;
            };
        }
        // the image call routes with gate.bias_vl, the text call with gate.bias
        const r0 = cs[gi.a].args[2].arg.t.role.weight;
        if (!std.mem.endsWith(u8, r0, ".moe.bias_vl")) bad += 1;
        if (nt > 0 and !std.mem.endsWith(u8, cs[last.a].args[2].arg.t.role.weight, ".moe.bias")) bad += 1;
    }
    return bad;
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, experts: []const u8, log: *std.Io.Writer) !usize {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const cwd = std.Io.Dir.cwd();
    var w = try check.widthsFromCapture(a, try cwd.readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "weights.json" }), a, .limited(1 << 28)), experts);
    const cfg: Config = .{};
    const ws = try check.windows(a, io, dir);
    try check.gmWidths(a, &w, ws);
    const o = try check.optionsOf(a, io, dir);
    var bad: usize = 0;
    var cases: usize = 0;
    for (ws) |win| if (win.prefill) {
        const n = win.n;
        for ([_]i64{ 1, 5, 16, 17, @divFloor(n, 3), n - 1, n }) |ni| {
            if (ni < 1 or ni > n) continue;
            const b = try one(a, &cfg, &w, o, win, ni, log);
            try log.print("{s} {s}: {d} rows, {d} image rows: {s}\n", .{ win.set, win.phase, n, ni, if (b == 0) "ok" else "FAIL" });
            bad += b;
            cases += 1;
        }
    };
    if (cases == 0) return error.NoPrefillWindows;
    // the plan program (image_rows < 0) names every split role
    var po = o;
    po.image_keep = true;
    po.image_rows = -1;
    for (ws) |win| if (win.prefill) {
        _ = try block_prefill.emitPrefill(a, &cfg, &w, po, win.layers, win.n, win.start, true);
    };
    try log.print("vision-emit: {d} split programs, {d} mismatches\n", .{ cases, bad });
    return bad;
}
