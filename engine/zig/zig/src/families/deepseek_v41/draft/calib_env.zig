//! The served engine's draft pricing from prod's own calibration (Python calib.costs_for, depth.env_mode / env_skip,
//! joint.env_*), with Python's knob names:
//!
//! - **TF_DSV41_CALIB** (`cached`, the default | `real` | `default`): `cached` reads a `calib-<key>.json` from
//!   **TF_DSV41_CALIB_DIR** (default ~/.cache/tensorfold/dsv41-calib), the file Python's boot measured and stored;
//!   `default` prices from depth.default_costs. `real` measures at every boot in Python (prod: `real`, the dir
//!   /cache/dsv41-calib), and the boot stores the table; Zig does not measure, so `real` reads the newest stored table
//!   as `cached` does: prod's last boot's.
//! - Python's key hashes its own image digest, which Zig cannot reproduce, so the entry is chosen by the shape Python
//!   stores beside it (`meta.shape`): the same world and DSpark, at least our slots (a table timed on more slots
//!   prices a lone window's rows the same: prod's 4-slot table for a 1-slot engine); the same slots, then the same
//!   context preferred, then the newest (`meta.time`). **TF_DSV41_CALIB_FILE** names one entry outright (a Zig knob:
//!   no Python twin).
//! - The table is used as Python's ranks use it: rank 0's entry goes through `Costs.encode` / `decode` (microsecond
//!   ints, round half to even) before the depth sees it, so a cached table prices with the shared bits, not the
//!   file's decimals.
//! - Nothing found (no directory, no matching entry): depth.default_costs, as before this knob, and a log line.
//! - The depth's knobs: TF_DSV41_DEPTH (cost | static), TF_DSV41_DRAFT_DEPTH (cap 5 / static k 3; 0-15),
//!   TF_DSV41_DRAFT_SKIP, TF_DSV41_DEPTH_JOINT (0 | 1 | 2), TF_DSV41_DEPTH_FAIR, TF_DSV41_DEPTH_HOST_MS,
//!   TF_DSV41_GRAPH_ROWS_MAX.
const std = @import("std");
const Allocator = std.mem.Allocator;
const costs_mod = @import("costs.zig");
const depth = @import("depth.zig");
const joint = @import("joint.zig");
const Costs = costs_mod.Costs;

pub const Mode = enum { cached, default, real };

/// The engine the table must have been measured on (Python's `shape`: slots, world, DSpark; context preferred).
pub const Shape = struct { slots: u32, world: u32, dspark: bool, context: ?u64 = null };

pub const Source = enum { default, cached };

pub const Loaded = struct {
    costs: Costs,
    source: Source,
    /// the entry read (gpa-owned; null for the defaults)
    path: ?[]u8 = null,

    pub fn deinit(l: *Loaded, gpa: Allocator) void {
        l.costs.deinit(gpa);
        if (l.path) |p| gpa.free(p);
    }
};

/// What the environment asks for (strings as read; null: unset).
pub const Knobs = struct {
    mode: ?[]const u8 = null,
    dir: ?[]const u8 = null,
    file: ?[]const u8 = null,
    home: ?[]const u8 = null,

    pub fn fromEnv() Knobs {
        return .{ .mode = env("TF_DSV41_CALIB"), .dir = env("TF_DSV41_CALIB_DIR"), .file = env("TF_DSV41_CALIB_FILE"), .home = env("HOME") };
    }
};

pub fn mode(raw: ?[]const u8) !Mode {
    const v = trimmed(raw) orelse return .cached;
    var buf: [16]u8 = undefined;
    if (v.len > buf.len) return error.CalibMode;
    return std.meta.stringToEnum(Mode, std.ascii.lowerString(&buf, v)) orelse error.CalibMode;
}

/// The cost table for `want` per `k`; `default_rows`: the defaults' rows (depth.default_costs: 16; the served
/// engine's lanes have priced 64 since M4).
pub fn load(gpa: Allocator, io: std.Io, k: Knobs, want: Shape, default_rows: usize) !Loaded {
    switch (try mode(k.mode)) {
        .default => return defaults(gpa, default_rows),
        .cached, .real => {}, // real: the newest stored measurement (calib.measure runs in Python only)
    }
    if (trimmed(k.file)) |f| return .{ .costs = try readEntry(gpa, io, f), .source = .cached, .path = try gpa.dupe(u8, f) };
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = trimmed(k.dir) orelse std.fmt.bufPrint(&dir_buf, "{s}/.cache/tensorfold/dsv41-calib", .{trimmed(k.home) orelse return defaults(gpa, default_rows)}) catch return defaults(gpa, default_rows);
    const path = (try pick(gpa, io, dir, want)) orelse return defaults(gpa, default_rows);
    errdefer gpa.free(path);
    return .{ .costs = try readEntry(gpa, io, path), .source = .cached, .path = path };
}

fn defaults(gpa: Allocator, rows: usize) !Loaded {
    return .{ .costs = try costs_mod.defaults(gpa, rows), .source = .default };
}

/// One entry's table as the ranks share it (calib.load, then Costs.encode / decode).
pub fn readEntry(gpa: Allocator, io: std.Io, path: []const u8) !Costs {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(bytes);
    var raw = try Costs.parseJson(gpa, bytes);
    defer raw.deinit(gpa);
    return shared(gpa, raw);
}

/// `c` after rank 0's share: microsecond ints and back (Python `Costs.decode(costs.encode())`).
pub fn shared(gpa: Allocator, c: Costs) !Costs {
    const ints = try c.encode(gpa);
    defer gpa.free(ints);
    return Costs.decode(gpa, ints);
}

const Meta = struct {
    meta: struct {
        shape: struct { slots: u32 = 0, world: u32 = 0, dspark: bool = false, context: ?u64 = null } = .{},
        time: []const u8 = "",
    } = .{},
};

/// The directory's `calib-*.json` measured on `want`'s engine (see the top); null: none.
pub fn pick(gpa: Allocator, io: std.Io, dir: []const u8, want: Shape) !?[]u8 {
    var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return null;
    defer d.close(io);
    var best: ?[]u8 = null;
    errdefer if (best) |b| gpa.free(b);
    var best_rank: u8 = 0;
    var best_time: [32]u8 = undefined;
    var best_time_len: usize = 0;
    var it = d.iterate();
    while (try it.next(io)) |e| {
        if (e.kind != .file and e.kind != .sym_link) continue;
        if (!std.mem.startsWith(u8, e.name, "calib-") or !std.mem.endsWith(u8, e.name, ".json")) continue;
        const path = try std.fs.path.join(gpa, &.{ dir, e.name });
        var keep = false;
        defer if (!keep) gpa.free(path);
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24)) catch continue;
        defer gpa.free(bytes);
        const parsed = std.json.parseFromSlice(Meta, gpa, bytes, .{ .ignore_unknown_fields = true }) catch continue;
        defer parsed.deinit();
        const m = parsed.value.meta;
        if (m.shape.world != want.world or m.shape.slots < want.slots or m.shape.dspark != want.dspark) continue;
        // a usable table (calib.load's rule) before it can win
        var c = Costs.parseJson(gpa, bytes) catch continue;
        c.deinit(gpa);
        const rank: u8 = 1 + @as(u8, if (m.shape.slots == want.slots) 2 else 0) + @intFromBool(want.context != null and m.shape.context == want.context);
        const t = m.time[0..@min(m.time.len, best_time.len)];
        const newer = std.mem.order(u8, t, best_time[0..best_time_len]) == .gt;
        if (best == null or rank > best_rank or (rank == best_rank and newer)) {
            if (best) |b| gpa.free(b);
            best = path;
            keep = true;
            best_rank = rank;
            @memcpy(best_time[0..t.len], t);
            best_time_len = t.len;
        }
    }
    return best;
}

/// The depth's knobs as Python's depth.env_mode / env_skip and joint.env_mode / env_fair / env_host_ms / rows_cap.
pub fn depthSettings(get: *const fn ([*:0]const u8) ?[]const u8) !depth.Settings {
    var s: depth.Settings = .{};
    if (trimmed(get("TF_DSV41_CODE_ACCEPT"))) |v| s.code_accept = try flag(v, error.CodeAccept);
    if (s.code_accept) if (trimmed(get("TF_DSV41_GRAPH_BUCKET"))) |v| {
        s.context_bucket = try std.fmt.parseInt(u32, v, 10);
        if (s.context_bucket == 0) return error.BadGraphBucket;
    };
    const m = trimmed(get("TF_DSV41_DEPTH")) orelse "cost";
    s.mode = if (std.ascii.eqlIgnoreCase(m, "cost")) .cost else if (std.ascii.eqlIgnoreCase(m, "static")) .static else return error.DepthMode;
    if (trimmed(get("TF_DSV41_DRAFT_DEPTH"))) |v| {
        s.cap = try std.fmt.parseInt(usize, v, 10);
        if (s.cap > 15) return error.DraftDepth;
    } else s.cap = if (s.mode == .cost) 5 else 3;
    if (trimmed(get("TF_DSV41_DRAFT_SKIP"))) |v| s.skip = try flag(v, error.DraftSkip);
    if (trimmed(get("TF_DSV41_DEPTH_JOINT"))) |v| s.joint = if (std.mem.eql(u8, v, "2")) 2 else @intFromBool(try flag(v, error.DepthJoint));
    s.fair = try float(get("TF_DSV41_DEPTH_FAIR"), joint.fair, 0.0, 4.0);
    s.host_ms = try float(get("TF_DSV41_DEPTH_HOST_MS"), joint.host_ms, 0.0, 100.0);
    if (trimmed(get("TF_DSV41_GRAPH_ROWS_MAX"))) |v| s.max_rows = @max(1, @min(try std.fmt.parseInt(u64, v, 10), costs_mod.max_rows));
    return s;
}

fn flag(v: []const u8, bad: anyerror) !bool {
    for ([_][]const u8{ "1", "on", "true" }) |t| if (std.ascii.eqlIgnoreCase(v, t)) return true;
    for ([_][]const u8{ "0", "off", "false" }) |t| if (std.ascii.eqlIgnoreCase(v, t)) return false;
    return bad;
}

fn float(raw: ?[]const u8, default: f64, lo: f64, hi: f64) !f64 {
    const v = trimmed(raw) orelse return default;
    const x = try std.fmt.parseFloat(f64, v);
    if (x < lo or x > hi) return error.DepthKnob;
    return x;
}

pub fn env(name: [*:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

fn trimmed(v: ?[]const u8) ?[]const u8 {
    const t = std.mem.trim(u8, v orelse return null, " \t\r\n");
    return if (t.len == 0) null else t;
}
