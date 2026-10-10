//! TF_DSV41_PROFILE=<file>: each eager window's calls timed on the GPU (a CUDA event pair a call, the glue's
//! collectives included) and the window's host wall time; summed by call name over every profiled window and written
//! to <file> as JSON when the runner closes. The decode gap's split (the Spark window's 46.7 vs prod's tok/s):
//! GPU time by kernel family against Python's (dsv41_m2b_ref.py --profile), and the host's share (wall - GPU busy).
//! Graphed windows are one launch: their wall time comes from the caller (m2b's M2B_PROF_ROWS timing).
//! Prefill segments (longpf.window) are timed the same way into their own totals (the JSON's "prefill": segments,
//! wall, GPU and calls by name), the full prefill's split against dsv41_m2b_ref.py --profile-prefill.

const std = @import("std");
const cuda = @import("cuda");
const calls = @import("calls.zig");

pub const Profile = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    path: []const u8,
    events: std.ArrayList(cuda.Event) = .empty,
    /// this window's calls: name and the index of its start event
    open: std.ArrayList(struct { name: []const u8, at: usize }) = .empty,
    used: usize = 0,
    totals: std.StringArrayHashMapUnmanaged(Total) = .empty,
    windows: u64 = 0,
    wall_ns: u64 = 0,
    gpu_ns: u64 = 0,
    t0: u64 = 0,
    /// the open window is a prefill segment (beginPrefill): its times go to `pf`
    prefill: bool = false,
    pf: Totals = .{},

    pub const Totals = struct {
        calls: std.StringArrayHashMapUnmanaged(Total) = .empty,
        windows: u64 = 0,
        wall_ns: u64 = 0,
        gpu_ns: u64 = 0,
        /// prompt rows the encoder / full segments covered, and the CED decoder replays' (forward_prefill: the
        /// per-1K-row split of gap.py prefill-ab)
        rows: u64 = 0,
        tail_rows: u64 = 0,
    };

    pub const Total = struct { calls: u64 = 0, ns: u64 = 0 };

    pub fn fromEnv(gpa: std.mem.Allocator, d: *const cuda.Driver) !?*Profile {
        const p = std.c.getenv("TF_DSV41_PROFILE") orelse return null;
        const x = try gpa.create(Profile);
        x.* = .{ .gpa = gpa, .d = d, .path = std.mem.span(p) };
        return x;
    }

    fn event(p: *Profile) !*cuda.Event {
        if (p.used == p.events.items.len) try p.events.append(p.gpa, try cuda.Event.init(p.d, true));
        p.used += 1;
        return &p.events.items[p.used - 1];
    }

    pub fn begin(p: *Profile) void {
        p.prefill = false;
        p.used = 0;
        p.open.clearRetainingCapacity();
        p.t0 = nowNs();
    }

    /// A prefill segment's window (its end adds to the prefill totals).
    pub fn beginPrefill(p: *Profile) void {
        p.begin();
        p.prefill = true;
    }

    /// The rows the last prefill window covered (`tail`: a CED decoder replay's).
    pub fn prefillRows(p: *Profile, rows: u64, tail: bool) void {
        if (tail) p.pf.tail_rows += rows else p.pf.rows += rows;
    }

    /// Before a call: its start event on the stream.
    pub fn before(p: *Profile, s: cuda.Stream, c: *const calls.Call) !void {
        const at = p.used;
        try (try p.event()).record(s);
        try p.open.append(p.gpa, .{ .name = c.name, .at = at });
    }

    /// After a call: its end event.
    pub fn after(p: *Profile, s: cuda.Stream) !void {
        try (try p.event()).record(s);
    }

    /// The window's end: wait for the stream, add every call's time and the window's wall time.
    pub fn end(p: *Profile, s: cuda.Stream) !void {
        try s.synchronize();
        const wall = nowNs() -| p.t0;
        const map = if (p.prefill) &p.pf.calls else &p.totals;
        if (p.prefill) {
            p.pf.wall_ns += wall;
            p.pf.windows += 1;
        } else {
            p.wall_ns += wall;
            p.windows += 1;
        }
        for (p.open.items) |o| {
            const ms = try p.events.items[o.at].elapsedMs(p.events.items[o.at + 1]);
            const ns: u64 = @intFromFloat(@max(0.0, ms) * 1e6);
            const g = try map.getOrPut(p.gpa, o.name);
            if (!g.found_existing) {
                g.key_ptr.* = try p.gpa.dupe(u8, o.name);
                g.value_ptr.* = .{};
            }
            g.value_ptr.calls += 1;
            g.value_ptr.ns += ns;
            if (p.prefill) p.pf.gpu_ns += ns else p.gpu_ns += ns;
        }
        p.prefill = false;
    }

    /// Writes the totals (JSON) and frees everything.
    pub fn close(p: *Profile, io: std.Io) void {
        var w: std.Io.Writer.Allocating = .init(p.gpa);
        defer w.deinit();
        w.writer.print("{{\"windows\": {d}, \"wall_ms\": {d:.3}, \"gpu_ms\": {d:.3}, \"calls\": {{", .{ p.windows, @as(f64, @floatFromInt(p.wall_ns)) / 1e6, @as(f64, @floatFromInt(p.gpu_ns)) / 1e6 }) catch {};
        for (p.totals.keys(), p.totals.values(), 0..) |k, v, i| w.writer.print("{s}\"{s}\": {{\"calls\": {d}, \"ms\": {d:.4}}}", .{ if (i == 0) "" else ", ", k, v.calls, @as(f64, @floatFromInt(v.ns)) / 1e6 }) catch {};
        w.writer.print("}}, \"prefill\": {{\"segments\": {d}, \"rows\": {d}, \"tail_rows\": {d}, \"wall_ms\": {d:.3}, \"gpu_ms\": {d:.3}, \"calls\": {{", .{ p.pf.windows, p.pf.rows, p.pf.tail_rows, @as(f64, @floatFromInt(p.pf.wall_ns)) / 1e6, @as(f64, @floatFromInt(p.pf.gpu_ns)) / 1e6 }) catch {};
        for (p.pf.calls.keys(), p.pf.calls.values(), 0..) |k, v, i| w.writer.print("{s}\"{s}\": {{\"calls\": {d}, \"ms\": {d:.4}}}", .{ if (i == 0) "" else ", ", k, v.calls, @as(f64, @floatFromInt(v.ns)) / 1e6 }) catch {};
        w.writer.print("}}}}}}\n", .{}) catch {};
        if (std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p.path, .data = w.written() })) |_| {
            std.log.info("profile: {s} written ({d} windows, {d} prefill segments)", .{ p.path, p.windows, p.pf.windows });
        } else |e| std.log.err("profile: {s} not written ({t})", .{ p.path, e });
        for (p.events.items) |*e| e.deinit();
        p.events.deinit(p.gpa);
        p.open.deinit(p.gpa);
        for (p.totals.keys()) |k| p.gpa.free(k);
        p.totals.deinit(p.gpa);
        for (p.pf.calls.keys()) |k| p.gpa.free(k);
        p.pf.calls.deinit(p.gpa);
        p.gpa.destroy(p);
    }
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
