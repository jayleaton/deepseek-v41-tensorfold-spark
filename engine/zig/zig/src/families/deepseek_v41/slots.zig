//! Several live slots on the GPU path (Python forward.Forward's `slots` with rowmode.stack, G8): every slot's bounded
//! state stacked in one buffer a role, so a row's slot is an offset (row mode, batch.zig) and a one-slot program (a
//! prefill, a session operation, a release) reads one slot through a view.
//! - Stacked roles: each layer's SWA ring "s.L<i>.swa.v / .s" (slot s at s x ring rows), each ratio-2 carry
//!   "s.L<i>.carry" (row s), the pool's page tables "s.kv.pt" / "s.kv.ct" (row s of kv.DevicePool's [S, pts]). The
//!   buffer plan sizes them from the row programs (rowmode.zig), so Runner.bind allocates S slots' worth.
//! - `activate(s)`: the one-slot roles point at slot s's views, the forward's `slot` (position, Engram tail, pending
//!   window) is slot s's, and the pool's current slot is s. Every one-slot operation (Forward.prefill / window / keep /
//!   release) then runs on slot s unchanged. The leader sends `op_select` first, so every rank activates the same slot
//!   before the same operation.
//! - `deactivate()`: the roles back at the stacked bases (row mode's view), the active slot's state stored.
//! - Swapped roles (`swap`: CED's stash ring "s.ced.*", one buffer the active slot's prompt uses): each slot's bytes kept
//!   aside while another slot's prompt runs, swapped in by `activate` on every rank alike (Python keeps a stash a slot,
//!   replay.py `encode`). Only prompts prefilled in pieces between decode rounds (TF_DSV41_PREFILL_PIECES) interleave
//!   two slots' prompts; a whole prompt in one call never needed it.
//! Row mode needs the pool (Python: rowmode.ready); the contiguous slot is refused by name (error.RowsNeedPool).
//! The session store (sessions_gpu.zig) runs on the activated slot: its views of the stacked roles (a slot's bytes),
//! its `Forward.Slot`, the pool's current slot.

const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("forward.zig");
const buffers = @import("buffers.zig");
const rowmode = @import("rowmode.zig");

pub const Error = error{ RowsNeedPool, SlotRange, NotBound };

/// Plan-link operations of this workstream (docs: 16-19).
pub const op_select: i64 = 16;
pub const op_rows: i64 = 17;
pub const op_rows_keep: i64 = 18;
pub const op_rows_drop: i64 = 19;

/// TF_DSV41_SLOTS: live slots (1: today's one-slot path, nothing stacked). Every rank must set the same.
pub fn countFromEnv() !u32 {
    const v = std.c.getenv("TF_DSV41_SLOTS") orelse return 1;
    const n = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    if (n < 1 or n > 16) return error.SlotRange;
    return n;
}

const Stacked = struct { base: u64, bytes: u64 };

pub const SlotSet = struct {
    gpa: std.mem.Allocator,
    f: *fwd.Forward,
    n: u32,
    states: []fwd.Slot,
    /// the slot whose views are bound (its state lives in f.slot meanwhile); null: the stacked bases
    active: ?u32 = null,
    roles: std.StringArrayHashMapUnmanaged(Stacked) = .empty,
    /// roles one buffer holds for one slot at a time, each slot's copy aside (`swap`); `owner`: whose bytes are in
    swapped: std.ArrayList(Swapped) = .empty,
    owner: ?u32 = null,

    const Swapped = struct { addr: u64, len: u64, aside: cuda.DeviceBuffer };

    pub fn init(gpa: std.mem.Allocator, f: *fwd.Forward, n: u32) !*SlotSet {
        if (f.kv == null) return error.RowsNeedPool;
        if (f.kv.?.slots.len != n) return error.SlotRange;
        const ss = try gpa.create(SlotSet);
        errdefer gpa.destroy(ss);
        const states = try gpa.alloc(fwd.Slot, n);
        @memset(states, .{});
        ss.* = .{ .gpa = gpa, .f = f, .n = n, .states = states };
        f.pf_switch = .{ .ctx = ss, .run = viewFn };
        return ss;
    }

    pub fn deinit(ss: *SlotSet) void {
        for (ss.swapped.items) |*x| x.aside.free();
        ss.swapped.deinit(ss.gpa);
        ss.roles.deinit(ss.gpa);
        ss.gpa.free(ss.states);
        ss.gpa.destroy(ss);
    }

    /// After Runner.bind (and Kv.bind): each stacked role's base and a slot's bytes. The roles stay at their bases.
    pub fn bindBases(ss: *SlotSet, plan: *const buffers.Plan) !void {
        const r = ss.f.runner;
        for (plan.sizes.keys(), plan.sizes.values()) |role, size| {
            if (!(rowmode.isRing(role) or rowmode.isCarry(role)) or !ss.backbone(role)) continue;
            if (size % ss.n != 0) return error.NotBound;
            try ss.roles.put(ss.gpa, role, .{ .base = r.addressOf(role) orelse return error.NotBound, .bytes = size / ss.n });
        }
        const k = ss.f.kv.?;
        const tb: u64 = 4 * @as(u64, k.dev.pts);
        try ss.roles.put(ss.gpa, "s.kv.pt", .{ .base = k.dev.table.ptr, .bytes = tb });
        try ss.roles.put(ss.gpa, "s.kv.ct", .{ .base = k.dev.local.ptr, .bytes = tb });
    }

    /// After Runner.bind: `roles` (planned ones; others skipped) become per-slot by swapping (CED's stash ring with
    /// prompts in pieces). Each slot's copy is set aside in its own buffer (S x the role's bytes).
    pub fn swap(ss: *SlotSet, plan: *const buffers.Plan, roles: []const []const u8) !void {
        const r = ss.f.runner;
        for (roles) |role| {
            const len = plan.sizes.get(role) orelse continue;
            const addr = r.addressOf(role) orelse return error.NotBound;
            try ss.swapped.append(ss.gpa, .{ .addr = addr, .len = len, .aside = try cuda.DeviceBuffer.alloc(r.d, len * ss.n) });
        }
    }

    /// The swapped roles hold slot s's bytes (the owner's set aside first), on the runner's stream.
    fn swapIn(ss: *SlotSet, s: u32) !void {
        if (ss.swapped.items.len == 0 or ss.owner == s) return;
        const r = ss.f.runner;
        for (ss.swapped.items) |x| {
            if (ss.owner) |o| try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(x.aside.ptr + o * x.len, x.addr, x.len, r.stream.handle), "cuMemcpyDtoDAsync");
            try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(x.addr, x.aside.ptr + s * x.len, x.len, r.stream.handle), "cuMemcpyDtoDAsync");
        }
        ss.owner = s;
    }

    /// "s.L<i>.…" of a backbone layer (DSpark's blocks keep their own one-slot rings).
    fn backbone(ss: *const SlotSet, role: []const u8) bool {
        const rest = role["s.L".len..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return false;
        const L = std.fmt.parseInt(u32, rest[0..dot], 10) catch return false;
        return L < ss.f.cfg.layers;
    }

    /// A stacked role's base (the row programs' view), e.g. "s.L2.carry".
    pub fn base(ss: *const SlotSet, role: []const u8) ?Stacked {
        return ss.roles.get(role);
    }

    pub fn leads(ss: *const SlotSet) bool {
        const f = ss.f;
        return f.link != null and !f.quiet and f.comm.rank() == 0;
    }

    /// Slot s for one-slot operations on every rank (the leader sends op_select first).
    pub fn activate(ss: *SlotSet, s: u32) !void {
        if (s >= ss.n) return error.SlotRange;
        if (ss.leads()) try ss.f.link.?.send(&.{ op_select, s });
        try ss.swapIn(s);
        if (ss.active == s) return;
        ss.store();
        ss.f.slot = ss.states[s];
        try ss.point(s);
        ss.active = s;
    }

    /// Slot s's views and state current on this rank alone (no op_select: a multi-segment prefill run's segments,
    /// which every rank runs in the same order; forward_prefill.promptMulti).
    pub fn view(ss: *SlotSet, s: u32) !void {
        if (s >= ss.n) return error.SlotRange;
        try ss.swapIn(s);
        if (ss.active == s) return;
        ss.store();
        ss.f.slot = ss.states[s];
        try ss.point(s);
        ss.active = s;
    }

    fn viewFn(ctx: *anyopaque, s: u32) anyerror!void {
        const ss: *SlotSet = @ptrCast(@alignCast(ctx));
        return ss.view(s);
    }

    /// The stacked bases again (row mode), the active slot's state kept.
    pub fn deactivate(ss: *SlotSet) !void {
        if (ss.active == null) return;
        ss.store();
        ss.active = null;
        ss.f.slot = .{};
        for (ss.roles.keys(), ss.roles.values()) |role, st| try ss.f.runner.external(role, st.base);
    }

    /// A slot's state: the forward's own while it is active, else the stored one.
    pub fn state(ss: *SlotSet, s: u32) *fwd.Slot {
        return if (ss.active == s) &ss.f.slot else &ss.states[s];
    }

    fn store(ss: *SlotSet) void {
        if (ss.active) |a| ss.states[a] = ss.f.slot;
    }

    fn point(ss: *SlotSet, s: u32) !void {
        for (ss.roles.keys(), ss.roles.values()) |role, st| try ss.f.runner.external(role, st.base + s * st.bytes);
        ss.f.kv.?.select(s);
    }
};
