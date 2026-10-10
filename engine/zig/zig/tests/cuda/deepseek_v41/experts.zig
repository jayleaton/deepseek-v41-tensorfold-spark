//! Bit checks of the routed EXL3 expert kernels against the Python engine's on the oracle's inputs: the decode chain
//! (group, rot_in, upstream grouped / x3ld at every setting / x3pf, the epilogues, down_combine) and x3gm.

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");
const Fixture = @import("../fixture.zig").Fixture;

const exl3 = dsv41.exl3;
const Gpu = check.Gpu;

pub const Kind = struct {
    buf: [32]u8 = undefined,
    len: usize = 0,
    pub fn slice(k: *const Kind) []const u8 {
        return k.buf[0..k.len];
    }
};

/// The case's "kind" param (which check reads it).
pub fn caseKind(gpu: Gpu, dir: []const u8) !Kind {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    const s = try fx.string("kind");
    var k: Kind = .{};
    if (s.len > k.buf.len) return error.WrongType;
    @memcpy(k.buf[0..s.len], s);
    k.len = s.len;
    return k;
}

/// Device copies of a fixture's arrays, freed together.
const Bufs = struct {
    gpu: Gpu,
    fx: *const Fixture,
    list: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn deinit(b: *Bufs) void {
        for (b.list.items) |*x| x.free();
        b.list.deinit(b.gpu.gpa);
    }

    fn up(b: *Bufs, name: []const u8) !cuda.DeviceBuffer {
        const host = try b.fx.bytes(name);
        defer b.gpu.gpa.free(host);
        const buf = try cuda.DeviceBuffer.fromHost(b.gpu.d, host);
        try b.list.append(b.gpu.gpa, buf);
        return buf;
    }

    fn zeros(b: *Bufs, len: usize, byte: u8) !cuda.DeviceBuffer {
        const buf = try cuda.DeviceBuffer.alloc(b.gpu.d, @max(len, 4));
        try buf.fill8(byte, null);
        try b.list.append(b.gpu.gpa, buf);
        return buf;
    }

    /// int64 [E] pointer table: `base` + the fixture's byte offsets `name`.
    fn table(b: *Bufs, base: cuda.DeviceBuffer, name: []const u8) !cuda.DeviceBuffer {
        const host = try b.fx.bytes(name);
        defer b.gpu.gpa.free(host);
        const offs = std.mem.bytesAsSlice(u64, host);
        const ptrs = try b.gpu.gpa.alloc(u64, offs.len);
        defer b.gpu.gpa.free(ptrs);
        for (offs, ptrs) |o, *p| p.* = base.ptr + o;
        const buf = try cuda.DeviceBuffer.fromHost(b.gpu.d, std.mem.sliceAsBytes(ptrs));
        try b.list.append(b.gpu.gpa, buf);
        return buf;
    }
};

fn size(fx: Fixture, name: []const u8) !usize {
    return @intCast(try fx.int(name));
}

/// `got` (the first `want.len` bytes of the device buffer) against the fixture's array `name`.
fn same(gpu: Gpu, fx: Fixture, buf: cuda.DeviceBuffer, name: []const u8, what: []const u8) !void {
    const want = try fx.bytes(name);
    defer gpu.gpa.free(want);
    const got = try gpu.gpa.alloc(u8, want.len);
    defer gpu.gpa.free(got);
    try buf.download(0, got);
    try check.sameBytes(what, got, want);
}

fn has(fx: Fixture, name: []const u8) bool {
    const arrays = fx.parsed.value.object.get("arrays") orelse return false;
    return arrays.object.get(name) != null;
}

/// The decode chain of one routed layer at R rows, every stage against Python's output of the same stage.
pub fn decodeChain(gpu: Gpu, k: *const dsv41.Kernels, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    var b: Bufs = .{ .gpu = gpu, .fx = &fx };
    defer b.deinit();
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: ordered after the uploads and memsets on the default stream
    defer stream.deinit();
    const o = k.experts(stream);

    const R = try size(fx, "R");
    const slots = try size(fx, "slots");
    const E = try size(fx, "E");
    const D = try size(fx, "D");
    const I = try size(fx, "I");
    const P = R * slots;
    const maxu = @min(P, E);
    const input: exl3.Input = if (std.mem.eql(u8, try fx.string("input"), "bf16")) .bf16 else .f16;
    const limit: f32 = if ((try fx.int("limit_inf")) != 0) std.math.inf(f32) else @floatCast(try fx.float("limit"));
    const act_mode: c_int = @intCast(try fx.int("act_mode"));

    const x = try b.up("x");
    const pick = try b.up("pick");
    const wts = try b.up("wts");
    const words = try b.up("words");
    const tg = try b.table(words, "off_g");
    const tu = try b.table(words, "off_u");
    const td = try b.table(words, "off_d");
    const k2g = try b.up("k2_g");
    const k2u = try b.up("k2_u");
    const k2d = try b.up("k2_d");
    const suh_g = try b.up("suh_g");
    const suh_u = try b.up("suh_u");
    const svh_g = try b.up("svh_g");
    const svh_u = try b.up("svh_u");
    const suh_d = try b.up("suh_d");
    const svh_d = try b.up("svh_d");

    const uids = try b.zeros(maxu * 4, 0);
    const ucount = try b.zeros(4, 0);
    const members = try b.zeros(maxu * R * 4, 0xff);
    try o.group(pick.ptr, uids.ptr, ucount.ptr, members.ptr, R, slots, E, R);
    try stream.synchronize();
    try same(gpu, fx, uids, "uids", "group uids");
    try same(gpu, fx, ucount, "ucount", "group ucount");
    try same(gpu, fx, members, "members", "group members");

    const xg = try b.zeros(P * D * 2, 0);
    const xu = try b.zeros(P * D * 2, 0);
    try o.rotIn(input, x.ptr, D, pick.ptr, suh_g.ptr, suh_u.ptr, xg.ptr, xu.ptr, R, D, slots, E);
    try stream.synchronize();
    try same(gpu, fx, xg, "xg", "rot_in gate");
    try same(gpu, fx, xu, "xu", "rot_in up");

    // Exercise the actual Zig ABI against the same Python fixture, including untouched invalid-pick rows.
    if (input == .bf16 and R <= 64 and D == 5120 and slots <= 9 and E <= 512) {
        const fg = try b.zeros(P * D * 2, 0);
        const fu = try b.zeros(P * D * 2, 0);
        const fi = try b.zeros(maxu * 4, 0);
        const fc = try b.zeros(4, 0);
        const fm = try b.zeros(maxu * R * 4, 0xff);
        try o.groupRotIn(x.ptr, D, pick.ptr, suh_g.ptr, suh_u.ptr, fg.ptr, fu.ptr, fi.ptr, fc.ptr, fm.ptr, R, D, slots, E, R);
        try stream.synchronize();
        try same(gpu, fx, fi, "uids", "group_rot ids");
        try same(gpu, fx, fc, "ucount", "group_rot count");
        try same(gpu, fx, fm, "members", "group_rot members");
        try same(gpu, fx, fg, "xg", "group_rot gate");
        try same(gpu, fx, fu, "xu", "group_rot up");
    }

    const lo_gu: u32 = @intCast(try fx.int("lo_gu"));
    const hi_gu: u32 = @intCast(try fx.int("hi_gu"));
    const lo_d: u32 = @intCast(try fx.int("lo_d"));
    const hi_d: u32 = @intCast(try fx.int("hi_d"));
    const sk_gu = try size(fx, "sk_gu");
    const sk_d = try size(fx, "sk_d");
    const gu: exl3.Grouped = .{ .x0 = xg.ptr, .x1 = xu.ptr, .tp0 = tg.ptr, .tp1 = tu.ptr, .k2_0 = k2g.ptr, .k2_1 = k2u.ptr, .uids = uids.ptr, .ucount = ucount.ptr, .members = members.ptr, .z = 0, .mats = 2, .K = D, .N = I, .P = P, .SK = sk_gu, .slots = slots, .maxm = R, .nexp = maxu, .lo = lo_gu, .hi = hi_gu };
    const z = try b.zeros(@max(2 * sk_gu * P * I, sk_d * P * D) * 4, 0);
    const zlen_gu = 2 * sk_gu * P * I * 4;
    const zlen_d = sk_d * P * D * 4;

    // upstream's grouped kernel at the shape's default setting: the Z the epilogue reads
    var g = gu;
    g.z = z.ptr;
    const ucfg_gu = [3]u32{ @intCast(try fx.int("nt_gu")), @intCast(try fx.int("w_gu")), @intCast(try fx.int("pf_gu")) };
    try zeroThen(z, zlen_gu);
    try o.grouped(g, ucfg_gu[0], ucfg_gu[1], ucfg_gu[2]);
    try stream.synchronize();
    try same(gpu, fx, z, "z_gu", "upstream grouped gate/up Z");
    var n_ld: usize = 0;
    var n_pf: usize = 0;
    try settings(gpu, fx, o, stream, z, g, zlen_gu, "gu", &n_ld, &n_pf);

    // the gate/up epilogue on upstream's Z (re-run: the settings above overwrote it with the same bits)
    try zeroThen(z, zlen_gu);
    try o.grouped(g, ucfg_gu[0], ucfg_gu[1], ucfg_gu[2]);
    const xd = try b.zeros(P * I * 2, 0);
    try o.gateupEpilogue(z.ptr, pick.ptr, svh_g.ptr, svh_u.ptr, suh_d.ptr, xd.ptr, R, P, I, sk_gu, slots, E, limit, act_mode);
    try stream.synchronize();
    try same(gpu, fx, xd, "xd", "gate/up epilogue Xd");

    var dn: exl3.Grouped = .{ .x0 = xd.ptr, .x1 = xd.ptr, .tp0 = td.ptr, .tp1 = td.ptr, .k2_0 = k2d.ptr, .k2_1 = k2d.ptr, .uids = uids.ptr, .ucount = ucount.ptr, .members = members.ptr, .z = z.ptr, .mats = 1, .K = I, .N = D, .P = P, .SK = sk_d, .slots = slots, .maxm = R, .nexp = maxu, .lo = lo_d, .hi = hi_d };
    const ucfg_d = [3]u32{ @intCast(try fx.int("nt_d")), @intCast(try fx.int("w_d")), @intCast(try fx.int("pf_d")) };
    try zeroThen(z, zlen_d);
    try o.grouped(dn, ucfg_d[0], ucfg_d[1], ucfg_d[2]);
    try stream.synchronize();
    try same(gpu, fx, z, "z_d", "upstream grouped down Z");
    try settings(gpu, fx, o, stream, z, dn, zlen_d, "d", &n_ld, &n_pf);

    try zeroThen(z, zlen_d);
    try o.grouped(dn, ucfg_d[0], ucfg_d[1], ucfg_d[2]);
    const y = try b.zeros(P * D * 4, 0);
    try o.downEpilogue(z.ptr, pick.ptr, svh_d.ptr, y.ptr, R, P, D, sk_d, slots, E);
    try stream.synchronize();
    try same(gpu, fx, y, "y", "down epilogue Y");
    const out = try b.zeros(R * D * 4, 0);
    try o.combine(y.ptr, wts.ptr, out.ptr, R, D, slots);
    try stream.synchronize();
    try same(gpu, fx, out, "out", "combine");
    try y.fill8(0, null);
    try out.fill8(0, null);
    try o.downCombine(z.ptr, pick.ptr, svh_d.ptr, y.ptr, wts.ptr, out.ptr, R, P, D, sk_d, slots, E, false);
    try stream.synchronize();
    try same(gpu, fx, out, "out", "down_combine");
    dn.z = 0;
    check.pass("BITEXACT experts {s}: R {d} x {d} slots, E {d}, D {d}, I {d}, K2 gu [{d},{d}] d [{d},{d}]: group, rot_in, grouped, {d} x3ld and {d} x3pf settings, epilogues, combine", .{ std.fs.path.basename(dir), R, slots, E, D, I, lo_gu, hi_gu, lo_d, hi_d, n_ld, n_pf });
}

/// Python zeroes the whole Z before each launch: rows a launch does not write stay 0 on both sides.
fn zeroThen(z: cuda.DeviceBuffer, len: usize) !void {
    _ = len;
    try z.fill8(0, null);
}

/// Every x3ld setting (and PDL) and every x3pf setting that fits this matrix: Z against Python's of the same setting.
fn settings(gpu: Gpu, fx: Fixture, o: exl3.Ops, stream: cuda.Stream, z: cuda.DeviceBuffer, g: exl3.Grouped, zlen: usize, tag: []const u8, n_ld: *usize, n_pf: *usize) !void {
    var name_buf: [64]u8 = undefined;
    for (exl3.ld_cfgs) |c| for ([_]u32{ 0, 3 }) |probe| {
        _ = exl3.x3ldCheck(g, c[0], c[1], probe) catch continue;
        if (probe == 3) continue; // timing only: wrong results by design
        for ([_]bool{ false, true }) |pdl| {
            const name = try std.fmt.bufPrint(&name_buf, "z_{s}_ld_{d}_{d}", .{ tag, c[0], c[1] });
            try zeroThen(z, zlen);
            try o.x3ld(g, c[0], c[1], 0, pdl);
            try stream.synchronize();
            if (!has(fx, name)) return error.FixtureMissingSetting;
            try same(gpu, fx, z, name, name);
            n_ld.* += 1;
        }
    };
    for (exl3.pf_cfgs) |c| {
        _ = exl3.x3pfCheck(g, c[0], c[1]) catch continue;
        const name = try std.fmt.bufPrint(&name_buf, "z_{s}_pf_{d}_{d}", .{ tag, c[0], c[1] });
        if (!has(fx, name)) continue; // the oracle runs x3pf at prefill row counts only
        try zeroThen(z, zlen);
        try o.x3pf(g, c[0], c[1]);
        try stream.synchronize();
        try same(gpu, fx, z, name, name);
        n_pf.* += 1;
    }
}

/// x3gm over one row block: rot, then per gate width a gate/up launch and per down width a down launch (the
/// oracle's plans), then combine; every configuration the oracle ran, each against its Python output.
pub fn x3gm(gpu: Gpu, k: *const dsv41.Kernels, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    var b: Bufs = .{ .gpu = gpu, .fx = &fx };
    defer b.deinit();
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: ordered after the uploads and memsets on the default stream
    defer stream.deinit();
    const o = k.experts(stream);

    const R = try size(fx, "R");
    const slots = try size(fx, "slots");
    const D = try size(fx, "D");
    const I = try size(fx, "I");
    const P = R * slots;
    const shx = (try fx.int("shx")) != 0;
    const mats_x: usize = if (shx) 1 else 2;
    const ragged = (try fx.int("ragged")) != 0;
    const limit: f32 = @floatCast(try fx.float("limit"));
    const input: exl3.Input = if (std.mem.eql(u8, try fx.string("input"), "bf16")) .bf16 else .f16;

    const x = try b.up("x");
    const pick = try b.up("pick");
    const wts = try b.up("wts");
    const words = try b.up("words");
    const suh_g = try b.up("suh_g");
    const suh_u = try b.up("suh_u");
    const svh_g = try b.up("svh_g");
    const svh_u = try b.up("svh_u");
    const suh_d = try b.up("suh_d");
    const svh_d = try b.up("svh_d");
    // the stacked form: base + 0 / base + gate bytes / ...; the ragged form: pointer tables
    const tg = if (ragged) try b.table(words, "off_g") else words;
    const tu = if (ragged) try b.table(words, "off_u") else words;
    const td = if (ragged) try b.table(words, "off_d") else words;
    const base_u: u64 = if (ragged) 0 else @intCast(try fx.int("base_u"));
    const base_d: u64 = if (ragged) 0 else @intCast(try fx.int("base_d"));

    const xs = try b.zeros(mats_x * P * D * 2, 0);
    const xg = xs.ptr;
    const xu = xs.ptr + (mats_x - 1) * P * D * 2;
    try o.gmRot(input, x.ptr, D, pick.ptr, suh_g.ptr, suh_u.ptr, xg, xu, D, slots, P, mats_x);
    try stream.synchronize();
    try same(gpu, fx, xs, "xs", "x3gm rot");

    const xd = try b.zeros(P * I * 2, 0);
    const y = try b.zeros(P * D * 4, 0);
    const ticket = try b.zeros(4, 0);
    const out = try b.zeros(R * D * 4, 0);
    var name_buf: [96]u8 = undefined;
    const runs = try fx.int("runs");
    var launches: usize = 0;
    for (0..@intCast(runs)) |r| {
        // run r: a configuration per (kind, width) run<r>_<gu|dn>_<k2> and its plan run<r>_<gu|dn>_<k2>_<order|...>
        const tk = (try fx.int(try std.fmt.bufPrint(&name_buf, "run{d}_ticket", .{r}))) != 0;
        try xd.fill8(0, null);
        try y.fill8(0, null);
        for (exl3.gm_k2s) |k2| {
            const pl = (try plan(&b, fx, r, "gu", k2)) orelse continue;
            const cg: usize = @intCast(try fx.int(try std.fmt.bufPrint(&name_buf, "run{d}_gu_{d}", .{ r, k2 })));
            try nativePlan(&b, fx, o, stream, r, "gu", k2, P, cg);
            try ticket.fill8(0, null);
            try o.gmGateup(xg, xu, tg.ptr, tu.ptr + base_u, pl[0], pl[1], pl[2], pl[3], pl[4], ticket.ptr, svh_g.ptr, svh_u.ptr, suh_d.ptr, xd.ptr, D, I, k2, shx, cg, tk, limit, ragged);
            launches += 1;
        }
        try stream.synchronize();
        try same(gpu, fx, xd, try std.fmt.bufPrint(&name_buf, "run{d}_xd", .{r}), "x3gm gate/up Xd");
        for (exl3.gm_k2s) |k2| {
            const pl = (try plan(&b, fx, r, "dn", k2)) orelse continue;
            const cd: usize = @intCast(try fx.int(try std.fmt.bufPrint(&name_buf, "run{d}_dn_{d}", .{ r, k2 })));
            try nativePlan(&b, fx, o, stream, r, "dn", k2, P, cd);
            try ticket.fill8(0, null);
            try o.gmDown(xd.ptr, td.ptr + base_d, pl[0], pl[1], pl[2], pl[3], pl[4], ticket.ptr, svh_d.ptr, y.ptr, I, D, k2, cd, tk, ragged);
            launches += 1;
        }
        try stream.synchronize();
        try same(gpu, fx, y, try std.fmt.bufPrint(&name_buf, "run{d}_y", .{r}), "x3gm down Y");
        try o.combine(y.ptr, wts.ptr, out.ptr, R, D, slots);
        try stream.synchronize();
        try same(gpu, fx, out, try std.fmt.bufPrint(&name_buf, "run{d}_out", .{r}), "x3gm combine");
    }
    var v2_launches: usize = 0;
    if (has(fx, "v2_xd") or has(fx, "v2_t1_xd")) {
        // x3gm v2: one plan over every pair, one gate/up and one down launch over every width
        const k2g = try b.up("v2_k2g");
        const k2d = try b.up("v2_k2d");
        const pl = [_]u64{ (try b.up("v2_order")).ptr, (try b.up("v2_pe")).ptr, (try b.up("v2_poff")).ptr, (try b.up("v2_pcnt")).ptr, (try b.up("v2_npass")).ptr };
        const cg: usize = @intCast(try fx.int("v2_gu"));
        const cd: usize = @intCast(try fx.int("v2_dn"));
        for ([_]bool{ true, false }) |tk| {
            try xd.fill8(0, null);
            try y.fill8(0, null);
            try ticket.fill8(0, null);
            try o.gm2Gateup(xg, xu, tg.ptr, tu.ptr + base_u, k2g.ptr, pl[0], pl[1], pl[2], pl[3], pl[4], ticket.ptr, svh_g.ptr, svh_u.ptr, suh_d.ptr, xd.ptr, D, I, shx, cg, tk, limit, ragged);
            try ticket.fill8(0, null);
            try o.gm2Down(xd.ptr, td.ptr + base_d, k2d.ptr, pl[0], pl[1], pl[2], pl[3], pl[4], ticket.ptr, svh_d.ptr, y.ptr, I, D, cd, tk, ragged);
            try o.combine(y.ptr, wts.ptr, out.ptr, R, D, slots);
            try stream.synchronize();
            const t: u8 = @intFromBool(tk);
            try same(gpu, fx, xd, try std.fmt.bufPrint(&name_buf, "v2_t{d}_xd", .{t}), "x3gm v2 gate/up Xd");
            try same(gpu, fx, y, try std.fmt.bufPrint(&name_buf, "v2_t{d}_y", .{t}), "x3gm v2 down Y");
            try same(gpu, fx, out, try std.fmt.bufPrint(&name_buf, "v2_t{d}_out", .{t}), "x3gm v2 combine");
            v2_launches += 2;
        }
    }
    if (v2_launches == 0) {
        // No v2 arrays (the oracle's Python twin predates x3gm v2): gm2 against the fixture's gm_kernel outputs
        // (run 0, ticket on), the same bits by v2's contract. Its one plan over every pair comes from x3gm_plan.cu
        // (bit-equal to x3gm.plan, the "plan" case), the width tables from the experts' K2.
        const E = try size(fx, "E");
        const T = exl3.planPasses(P, E, 64);
        const order = try b.zeros(P * 4, 0);
        const pe = try b.zeros(T * 4, 0);
        const poff = try b.zeros(T * 4, 0);
        const pcnt = try b.zeros(T * 4, 0);
        const npass = try b.zeros(4, 0);
        try o.gmPlan(pick.ptr, P, E, 64, order.ptr, pe.ptr, poff.ptr, pcnt.ptr, npass.ptr);
        const k2g = try b.up("k2_g");
        const k2d = try b.up("k2_d");
        const cfg = exl3.gmTuned2(null, null, shx);
        for ([_]bool{ true, false }) |tk| {
            try xd.fill8(0, null);
            try y.fill8(0, null);
            try ticket.fill8(0, null);
            try o.gm2Gateup(xg, xu, tg.ptr, tu.ptr + base_u, k2g.ptr, order.ptr, pe.ptr, poff.ptr, pcnt.ptr, npass.ptr, ticket.ptr, svh_g.ptr, svh_u.ptr, suh_d.ptr, xd.ptr, D, I, shx, cfg[0], tk, limit, ragged);
            try ticket.fill8(0, null);
            try o.gm2Down(xd.ptr, td.ptr + base_d, k2d.ptr, order.ptr, pe.ptr, poff.ptr, pcnt.ptr, npass.ptr, ticket.ptr, svh_d.ptr, y.ptr, I, D, cfg[1], tk, ragged);
            try o.combine(y.ptr, wts.ptr, out.ptr, R, D, slots);
            try stream.synchronize();
            try same(gpu, fx, xd, "run0_xd", "x3gm v2 Xd vs gm_kernel");
            try same(gpu, fx, y, "run0_y", "x3gm v2 Y vs gm_kernel");
            try same(gpu, fx, out, "run0_out", "x3gm v2 combine vs gm_kernel");
            v2_launches += 2;
        }
    }
    // x3gm v3 (gm3_kernel): every variant of this case's projections against the fixture's gm_kernel outputs (run 0),
    // both ticket modes; down reads the fixture's Xd, so a variant is checked alone. One plan (x3gm_plan.cu, bm 64).
    var v3_launches: usize = 0;
    {
        const E = try size(fx, "E");
        const T = exl3.planPasses(P, E, 64);
        const order = try b.zeros(P * 4, 0);
        const pe = try b.zeros(T * 4, 0);
        const poff = try b.zeros(T * 4, 0);
        const pcnt = try b.zeros(T * 4, 0);
        const npass = try b.zeros(4, 0);
        try o.gmPlan(pick.ptr, P, E, 64, order.ptr, pe.ptr, poff.ptr, pcnt.ptr, npass.ptr);
        const k2g = try b.up("k2_g");
        const k2d = try b.up("k2_d");
        const xd_want = try b.up("run0_xd");
        var vb: [exl3.v3_variants.len]usize = undefined;
        for (exl3.v3Of(2, @intCast(mats_x), &vb)) |v| for ([_]bool{ true, false }) |tk| {
            try xd.fill8(0, null);
            try ticket.fill8(0, null);
            try o.gm3Gateup(v, xg, xu, tg.ptr, tu.ptr + base_u, k2g.ptr, order.ptr, pe.ptr, poff.ptr, pcnt.ptr, npass.ptr, ticket.ptr, svh_g.ptr, svh_u.ptr, suh_d.ptr, xd.ptr, D, I, shx, tk, limit, ragged);
            try stream.synchronize();
            try same(gpu, fx, xd, "run0_xd", try std.fmt.bufPrint(&name_buf, "x3gm v3 gate/up variant {d} Xd vs gm_kernel", .{v}));
            v3_launches += 1;
        };
        for (exl3.v3Of(1, 1, &vb)) |v| for ([_]bool{ true, false }) |tk| {
            try y.fill8(0, null);
            try ticket.fill8(0, null);
            try o.gm3Down(v, xd_want.ptr, td.ptr + base_d, k2d.ptr, order.ptr, pe.ptr, poff.ptr, pcnt.ptr, npass.ptr, ticket.ptr, svh_d.ptr, y.ptr, I, D, tk, ragged);
            try stream.synchronize();
            try same(gpu, fx, y, "run0_y", try std.fmt.bufPrint(&name_buf, "x3gm v3 down variant {d} Y vs gm_kernel", .{v}));
            v3_launches += 1;
        };
    }
    if (has(fx, "dequant_in")) {
        const t = try b.up("dequant_in");
        const dk: usize = @intCast(try fx.int("dequant_k"));
        const dn: usize = @intCast(try fx.int("dequant_n"));
        const dk2: u32 = @intCast(try fx.int("dequant_k2"));
        const dq = try b.zeros(dk * dn * 2, 0);
        try o.gmDequant(t.ptr, dq.ptr, dk, dn, dk2);
        try stream.synchronize();
        try same(gpu, fx, dq, "dequant_gm", "x3gm dequant");
        try dq.fill8(0, null);
        try o.dequant(t.ptr, dq.ptr, dk, dn, dk2);
        try stream.synchronize();
        try same(gpu, fx, dq, "dequant_up", "upstream dequant");
    }
    check.pass("BITEXACT x3gm {s}: R {d} x {d} slots, D {d}, I {d}, {s}, shx {}: rot, {d} runs ({d} launches), v2 {d} launches, v3 {d} launches, combine", .{ std.fs.path.basename(dir), R, slots, D, I, if (ragged) "ragged" else "stacked", shx, runs, launches, v2_launches, v3_launches });
}

/// pointwise.cu against torch: the shared expert's SwiGLU with the limit, and the partial adds (fp32, into bf16).
pub fn pointwiseCases(gpu: Gpu, k: *const dsv41.Kernels, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    var b: Bufs = .{ .gpu = gpu, .fx = &fx };
    defer b.deinit();
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: ordered after the uploads and memsets on the default stream
    defer stream.deinit();
    const o = k.others(stream);
    const R = try size(fx, "R");
    const n = try size(fx, "n");
    const D = try size(fx, "D");
    const g = try b.up("g");
    const u = try b.up("u");
    const act = try b.zeros(R * n * 4, 0xee);
    try o.swiglu(g.ptr, n, u.ptr, n, act.ptr, n, R, n, @floatCast(try fx.float("limit")));
    try stream.synchronize();
    try same(gpu, fx, act, "act", "swiglu vs torch clamp_ / silu / mul_");
    const a = try b.up("a");
    const bb = try b.up("b");
    const y16 = try b.zeros(R * D * 2, 0xee);
    try o.addBf16(a.ptr, bb.ptr, y16.ptr, R * D);
    try o.addInto(a.ptr, bb.ptr, R * D);
    try stream.synchronize();
    try same(gpu, fx, a, "sum32", "out += shared");
    try same(gpu, fx, y16, "sum16", "torch.add(out=bf16)");
    check.pass("BITEXACT pointwise.cu == torch: swiglu with limit ({d} x {d}, NaN / inf / -0.0), adds into fp32 and bf16", .{ R, n });
}

/// glue.cu against torch: the window's embedding send / sum, the casts, positions, gather_cols and the carry.
pub fn glueCases(gpu: Gpu, k: *const dsv41.Kernels, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    var b: Bufs = .{ .gpu = gpu, .fx = &fx };
    defer b.deinit();
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: ordered after the uploads on the default stream
    defer stream.deinit();
    const o = k.others(stream);
    const V = try size(fx, "V");
    const D = try size(fx, "D");
    const W = try size(fx, "W");
    const n = try size(fx, "n");
    const c = try size(fx, "c");

    const embed = try b.up("embed");
    const ids = try b.up("ids");
    const send = try b.zeros(n * D * 2, 0xee);
    try o.embedSend(ids.ptr, n, embed.ptr, try fx.int("lo"), V, D, send.ptr);
    const recv = try b.up("recv");
    const summed = try b.zeros(n * 4 * D * 2, 0xee);
    try o.embedSum(recv.ptr, W, n, D, summed.ptr);
    const x = try b.up("x");
    const cast = try b.zeros(n * D * 2, 0xee);
    try o.castBf16(x.ptr, cast.ptr, n * D);
    const kw = try b.zeros(n * D * 4, 0xee);
    try o.kitWeights(x.ptr, kw.ptr, n * D);
    const kl = try b.up("x");
    try o.kitLogits(kl.ptr, n * D);
    const gc = try b.up("gc");
    const cols = try b.zeros(n * W * c * 2, 0xee);
    try o.gatherCols(gc.ptr, W, n, c, cols.ptr);
    const src = try b.up("src");
    const row: usize = @intCast(try fx.int("row"));
    const carry = try b.zeros(1024 * 4, 0xee);
    try o.carry(src.ptr, @intCast(try fx.int("ld")), 0, row, 1024, carry.ptr);
    // the device-held row (graph-safe form): the same bits
    const acc = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&@as(i32, @intCast(row))));
    try b.list.append(gpu.gpa, acc);
    const carry2 = try b.zeros(1024 * 4, 0xee);
    try o.carry(src.ptr, @intCast(try fx.int("ld")), acc.ptr, 0, 1024, carry2.ptr);
    const start: i32 = @intCast(try fx.int("start"));
    const pos = try b.zeros(4, 0xee);
    const pos64 = try b.zeros(n * 8, 0xee);
    const lo = try b.zeros(n * 4, 0xee);
    try o.positions(start, n, pos.ptr, pos64.ptr, lo.ptr);
    // the graphed window's form: start read from the device, the same values
    const start_dev = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&start));
    try b.list.append(gpu.gpa, start_dev);
    const pos_d = try b.zeros(4, 0xee);
    const pos64_d = try b.zeros(n * 8, 0xee);
    const lo_d = try b.zeros(n * 4, 0xee);
    try o.positionsDev(start_dev.ptr, n, pos_d.ptr, pos64_d.ptr, lo_d.ptr);
    // widen_f32: the recv rows (bf16, NaN / inf / -0 among them) to fp32, against the bits shifted up on the host
    const wide = try b.zeros(W * n * D * 4, 0xee);
    try o.widenF32(recv.ptr, wide.ptr, W * n * D);
    try stream.synchronize();
    {
        const recv_host = try fx.bytes("recv");
        defer gpu.gpa.free(recv_host);
        const want = try gpu.gpa.alloc(u32, recv_host.len / 2);
        defer gpu.gpa.free(want);
        for (want, 0..) |*w, i| w.* = @as(u32, std.mem.readInt(u16, recv_host[2 * i ..][0..2], .little)) << 16;
        const got = try gpu.gpa.alloc(u8, 4 * want.len);
        defer gpu.gpa.free(got);
        try wide.download(0, got);
        try check.sameBytes("glue widen_f32", got, std.mem.sliceAsBytes(want));
    }

    try same(gpu, fx, send, "send", "glue embed_send");
    try same(gpu, fx, summed, "summed", "glue embed_sum");
    try same(gpu, fx, cast, "cast", "glue cast_bf16");
    try same(gpu, fx, kw, "kw", "glue kit_weights");
    try same(gpu, fx, kl, "kl", "glue kit_logits");
    try same(gpu, fx, cols, "cols", "glue gather_cols");
    try same(gpu, fx, carry, "carry", "glue carry (host row)");
    try same(gpu, fx, carry2, "carry", "glue carry (device row)");
    // positions: no torch op to record; Win.make's values
    var want_pos: [1]i32 = .{start};
    const want64 = try gpu.gpa.alloc(i64, n);
    defer gpu.gpa.free(want64);
    const want_lo = try gpu.gpa.alloc(i32, n);
    defer gpu.gpa.free(want_lo);
    for (want64, want_lo, 0..) |*p, *l, i| {
        p.* = start + @as(i64, @intCast(i));
        l.* = 0;
    }
    for ([_]struct { cuda.DeviceBuffer, []const u8 }{ .{ pos, std.mem.sliceAsBytes(&want_pos) }, .{ pos64, std.mem.sliceAsBytes(want64) }, .{ lo, std.mem.sliceAsBytes(want_lo) }, .{ pos_d, std.mem.sliceAsBytes(&want_pos) }, .{ pos64_d, std.mem.sliceAsBytes(want64) }, .{ lo_d, std.mem.sliceAsBytes(want_lo) } }) |pw| {
        const got = try gpu.gpa.alloc(u8, pw[1].len);
        defer gpu.gpa.free(got);
        try pw[0].download(0, got);
        try check.sameBytes("glue positions", got, pw[1]);
    }
    check.pass("BITEXACT glue.cu == torch: embed send / sum (W {d}), cast_bf16, kit_weights, kit_logits, gather_cols, carry, positions (host and device start), widen_f32", .{W});
}

/// prefill_glue.cu against torch: block_prefill.zig's glue steps (top_positions, candidate_keys, visible_counts, the
/// MoE's kit picks, x3gm's width mask, the head weights' casts, the compressor projection, the staging ring's copies).
pub fn pfglueCases(gpu: Gpu, k: *const dsv41.Kernels, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    var b: Bufs = .{ .gpu = gpu, .fx = &fx };
    defer b.deinit();
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: ordered after the uploads on the default stream
    defer stream.deinit();
    const o = k.others(stream);
    var nb: [64]u8 = undefined;
    var nb2: [64]u8 = undefined;

    const ntp = try size(fx, "tp");
    for (0..ntp) |i| {
        const R = try size(fx, try std.fmt.bufPrint(&nb, "tp{d}_R", .{i}));
        const n = try size(fx, try std.fmt.bufPrint(&nb, "tp{d}_n", .{i}));
        const count = try size(fx, try std.fmt.bufPrint(&nb, "tp{d}_count", .{i}));
        const keys = try b.up(try std.fmt.bufPrint(&nb, "tp{d}_keys", .{i}));
        const got = try b.zeros(R * count * 4, 0xee);
        try o.topPositions(keys.ptr, n, R, n, count, got.ptr);
        try stream.synchronize();
        try same(gpu, fx, got, try std.fmt.bufPrint(&nb, "tp{d}_want", .{i}), try std.fmt.bufPrint(&nb2, "pfglue top_positions {d}", .{i}));
    }
    {
        const R = try size(fx, "ck_R");
        const cnb = try size(fx, "ck_nb");
        const bs = try size(fx, "ck_bs");
        const blocks = try b.up("ck_blocks");
        const got = try b.zeros(R * cnb * bs * 4, 0xee);
        try o.candKeys(blocks.ptr, cnb, R, cnb, bs, got.ptr);
        try stream.synchronize();
        try same(gpu, fx, got, "ck_want", "pfglue candidate_keys");
    }
    {
        const sel = try b.up("tp0_want");
        const pos = try b.up("vc_pos");
        const R = try size(fx, "tp0_R");
        const K = try size(fx, "tp0_count");
        for ([_]u32{ 4, 128 }) |ratio| {
            const got = try b.zeros(R * 4, 0xee);
            try o.visibleCounts(sel.ptr, K, pos.ptr, ratio, R, got.ptr);
            try stream.synchronize();
            try same(gpu, fx, got, try std.fmt.bufPrint(&nb, "vc{d}_want", .{ratio}), "pfglue visible_counts");
        }
    }
    {
        const n = try size(fx, "gp_n");
        const ld = try size(fx, "gp_slots");
        const topk = try size(fx, "gp_topk");
        const pick = try b.up("gp_pick");
        const wts = try b.up("gp_wts");
        const pk = try b.zeros(n * topk * 4, 0xee);
        const w6 = try b.zeros(n * topk * 4, 0xee);
        try o.gmPicks(pick.ptr, wts.ptr, ld, topk, n, pk.ptr, w6.ptr);
        try stream.synchronize();
        try same(gpu, fx, pk, "gp_pk", "pfglue gm_picks picks");
        try same(gpu, fx, w6, "gp_w6", "pfglue gm_picks kit weights");
    }
    {
        const E = try size(fx, "wm_E");
        const P = try size(fx, "wm_P");
        const pick = try b.up("wm_pick");
        const tab = try b.up("wm_tab");
        for ([_]u32{ 4, 10 }) |k2| {
            const got = try b.zeros(P * 4, 0xee);
            try o.widthMask(pick.ptr, P, tab.ptr, E, k2, got.ptr);
            try stream.synchronize();
            try same(gpu, fx, got, try std.fmt.bufPrint(&nb, "wm{d}_want", .{k2}), "pfglue width mask");
        }
    }
    {
        const x = try b.up("f64_in");
        const n = x.len / 8;
        const got = try b.zeros(n * 4, 0xee);
        try o.f64Bf16F32(x.ptr, got.ptr, n);
        try stream.synchronize();
        try same(gpu, fx, got, "f64_want", "pfglue fp64 -> bf16 -> fp32");
    }
    {
        const n = try size(fx, "pj_n");
        const hd = try size(fx, "pj_hd");
        const kv = try b.up("pj_kv");
        const gate = try b.up("pj_gate");
        const two = try b.zeros(n * 2 * hd * 4, 0xee);
        try o.widenCat(kv.ptr, gate.ptr, n, hd, two.ptr);
        const one = try b.zeros(n * hd * 4, 0xee);
        try o.widenCat(kv.ptr, 0, n, hd, one.ptr);
        try stream.synchronize();
        try same(gpu, fx, two, "pj_two", "pfglue projection [kv | gate]");
        try same(gpu, fx, one, "pj_one", "pfglue projection kv");
    }
    for (0..2) |i| {
        const sr = try size(fx, try std.fmt.bufPrint(&nb, "rc{d}_sr", .{i}));
        const dr = try size(fx, try std.fmt.bufPrint(&nb, "rc{d}_dr", .{i}));
        const rb = try size(fx, try std.fmt.bufPrint(&nb, "rc{d}_rb", .{i}));
        const lo: u64 = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "rc{d}_lo", .{i})));
        const hi: u64 = @intCast(try fx.int(try std.fmt.bufPrint(&nb, "rc{d}_hi", .{i})));
        const src = try b.up(try std.fmt.bufPrint(&nb, "rc{d}_src", .{i}));
        const dst = try b.up(try std.fmt.bufPrint(&nb, "rc{d}_dst", .{i}));
        try o.ringCopy(src.ptr, sr, dst.ptr, dr, rb, lo, hi);
        try stream.synchronize();
        try same(gpu, fx, dst, try std.fmt.bufPrint(&nb, "rc{d}_want", .{i}), "pfglue staging ring copy");
    }
    check.pass("BITEXACT prefill_glue.cu == torch: top_positions ({d} shapes, counts 512 / 2,048), candidate_keys, visible_counts, gm_picks, width mask, fp64 casts, projection, ring copies", .{ntp});
}

/// topk_keys.cu against pick.top (torch.topk on pick.keys) at k 1 / 5 / 64 / 1,024 over 64,640 columns.
pub fn topkCases(gpu: Gpu, k: *const dsv41.Kernels, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    var b: Bufs = .{ .gpu = gpu, .fx = &fx };
    defer b.deinit();
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: ordered after the uploads and memsets on the default stream
    defer stream.deinit();
    const o = k.others(stream);
    const R = try size(fx, "R");
    const C = try size(fx, "C");
    const lg = try b.up("lg");
    var name_buf: [32]u8 = undefined;
    for (0..try size(fx, "ks")) |i| {
        const kk = try size(fx, try std.fmt.bufPrint(&name_buf, "k{d}", .{i}));
        const vals = try b.zeros(R * kk * 4, 0xee);
        const cols = try b.zeros(R * kk * 8, 0xee);
        try o.topKeys(lg.ptr, C, R, C, kk, vals.ptr, cols.ptr);
        try stream.synchronize();
        try same(gpu, fx, cols, try std.fmt.bufPrint(&name_buf, "cols{d}", .{i}), "topk_keys cols");
        try same(gpu, fx, vals, try std.fmt.bufPrint(&name_buf, "vals{d}", .{i}), "topk_keys vals");
    }
    check.pass("BITEXACT topk_keys.cu == pick.top on {d} rows x {d} columns, 4 k", .{ R, C });
}

/// x3gm_plan.cu against torch's x3gm.plan on prod-sized picks (the "plan" case).
pub fn planCases(gpu: Gpu, k: *const dsv41.Kernels, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    var b: Bufs = .{ .gpu = gpu, .fx = &fx };
    defer b.deinit();
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: ordered after the uploads and memsets on the default stream
    defer stream.deinit();
    const o = k.experts(stream);
    var name_buf: [64]u8 = undefined;
    const n = try size(fx, "cases");
    for (0..n) |i| {
        const P = try size(fx, try std.fmt.bufPrint(&name_buf, "c{d}_P", .{i}));
        const E = try size(fx, try std.fmt.bufPrint(&name_buf, "c{d}_E", .{i}));
        const bm = try size(fx, try std.fmt.bufPrint(&name_buf, "c{d}_bm", .{i}));
        const T = exl3.planPasses(P, E, bm);
        const pk = try b.up(try std.fmt.bufPrint(&name_buf, "c{d}_pk", .{i}));
        const order = try b.zeros(P * 4, 0xee);
        const pe = try b.zeros(T * 4, 0xee);
        const poff = try b.zeros(T * 4, 0xee);
        const pcnt = try b.zeros(T * 4, 0xee);
        const npass = try b.zeros(4, 0xee);
        try o.gmPlan(pk.ptr, P, E, bm, order.ptr, pe.ptr, poff.ptr, pcnt.ptr, npass.ptr);
        try stream.synchronize();
        for ([_][]const u8{ "order", "pe", "poff", "pcnt", "npass" }, [_]cuda.DeviceBuffer{ order, pe, poff, pcnt, npass }) |part, buf| {
            const name = try std.fmt.bufPrint(&name_buf, "c{d}_{s}", .{ i, part });
            try same(gpu, fx, buf, name, name);
        }
    }
    check.pass("BITEXACT x3gm_plan.cu == torch x3gm.plan on {d} cases (up to 24,576 pairs, 384 experts)", .{n});
}

/// x3gm_plan.cu on the plan's input picks against torch's x3gm.plan (the arrays the launches then use).
fn nativePlan(b: *Bufs, fx: Fixture, o: exl3.Ops, stream: cuda.Stream, r: usize, kind: []const u8, k2: u32, P: usize, cfg: usize) !void {
    var name_buf: [64]u8 = undefined;
    const pk = try b.up(try std.fmt.bufPrint(&name_buf, "run{d}_{s}_{d}_pk", .{ r, kind, k2 }));
    const E = try size(fx, "E");
    const bm: usize = exl3.gmMembers(if (kind[0] == 'g') .gu else .dn, cfg);
    const T = exl3.planPasses(P, E, bm);
    const order = try b.zeros(P * 4, 0xee);
    const pe = try b.zeros(T * 4, 0xee);
    const poff = try b.zeros(T * 4, 0xee);
    const pcnt = try b.zeros(T * 4, 0xee);
    const npass = try b.zeros(4, 0xee);
    try o.gmPlan(pk.ptr, P, E, bm, order.ptr, pe.ptr, poff.ptr, pcnt.ptr, npass.ptr);
    try stream.synchronize();
    for ([_][]const u8{ "order", "pe", "poff", "pcnt", "npass" }, [_]cuda.DeviceBuffer{ order, pe, poff, pcnt, npass }) |part, buf| {
        const name = try std.fmt.bufPrint(&name_buf, "run{d}_{s}_{d}_{s}", .{ r, kind, k2, part });
        try same(b.gpu, fx, buf, name, "x3gm_plan.cu vs x3gm.plan");
    }
}

/// The plan arrays of (run, kind, width), uploaded; null when that width has no launch in the run.
fn plan(b: *Bufs, fx: Fixture, r: usize, kind: []const u8, k2: u32) !?[5]u64 {
    var out: [5]u64 = undefined;
    var name_buf: [64]u8 = undefined;
    for ([_][]const u8{ "order", "pe", "poff", "pcnt", "npass" }, 0..) |part, i| {
        const name = try std.fmt.bufPrint(&name_buf, "run{d}_{s}_{d}_{s}", .{ r, kind, k2, part });
        if (!has(fx, name)) return null;
        out[i] = (try b.up(name)).ptr;
    }
    return out;
}
