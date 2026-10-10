//! TF_DSV41_CALIB=measure's host side: Python prod's boot calibration (tensorfold-decode1 8474f31
//! deepseek_v41/cuda/calib.py `measure`, `method`, `statistic`, `fit`, `monotone`, `table_of`, `split`, `extend`,
//! `wide_buckets`) without the GPU. `Plan` lists the timed runs in Python's order (rows 1..cycle as warmed interleaved
//! cycles, the rest of rows 1..8 a row at a time, rows 9..16 one warm-up + DEEP_REPS, the 2-slot 4 + 4 window, the wide
//! buckets split over SLOT_ROWS-row slots, rows 1..recheck again, the DSpark pass); `Plan.build` turns every rank's
//! samples (ms, one a run, warm-ups included and skipped) into the `Costs` Python's `measure` returns: each rank's
//! statistic a row (the lower of sweep and recheck), the per-entry maximum over ranks of the microsecond ints
//! (`gather_max`), `table_of`, the per-slot overhead and the wide points extended. calib_gpu.zig runs the plan.
const std = @import("std");
const Allocator = std.mem.Allocator;
const costs_mod = @import("costs.zig");
const Costs = costs_mod.Costs;

pub const max_rows = 16; // depth.MAX_ROWS: rows 1..16 each timed
pub const early = 8; // EARLY: rows timed with WARM warm-ups and REPS runs
pub const light_reps = 3; // REPS: the wide buckets and the DSpark pass, after one warm-up
pub const cut = 7; // CUT: slot i's text is the calibration text less its last CUT x i tokens
pub const slot_rows = costs_mod.slot_rows; // SLOT_ROWS
pub const wide = costs_mod.wide; // WIDE
pub const deep_from = 8; // glm5_next.spark.calib.DEEP_FROM (the `fit` shape)
pub const max_samples = 64; // a statistic's runs at most (REPS / DEEP_REPS <= 50)

pub const Shape = enum { raw, fit };
pub const Stat = enum { median, min };

/// calib.method(): the timing method (TF_DSV41_CALIB_SHAPE / _REPS / _WARM / _STAT / _RECHECK / _CYCLE / _DEEP_REPS).
pub const Method = struct {
    reps: u32 = 5,
    warm: u32 = 3,
    stat: Stat = .median,
    recheck: u32 = 4,
    shape: Shape = .raw,
    cycle: u32 = 4,
    deep: u32 = 5,
};

/// Python's DEFAULTS by shape (raw: G15; fit: G14, VERSION 3), then the explicit knobs, each range-checked.
pub fn method(get: *const fn ([*:0]const u8) ?[]const u8) !Method {
    const shape_s = trimmed(get("TF_DSV41_CALIB_SHAPE")) orelse "raw";
    const shape: Shape = if (std.ascii.eqlIgnoreCase(shape_s, "raw")) .raw else if (std.ascii.eqlIgnoreCase(shape_s, "fit")) .fit else return error.CalibShape;
    const d: Method = if (shape == .raw) .{} else .{ .shape = .fit, .warm = 2, .recheck = 2, .cycle = 0, .deep = 3 };
    const stat_s = trimmed(get("TF_DSV41_CALIB_STAT")) orelse "median";
    const stat: Stat = if (std.ascii.eqlIgnoreCase(stat_s, "median")) .median else if (std.ascii.eqlIgnoreCase(stat_s, "min")) .min else return error.CalibStat;
    return .{
        .reps = try num(get("TF_DSV41_CALIB_REPS"), 5, 1, 50),
        .warm = try num(get("TF_DSV41_CALIB_WARM"), d.warm, 0, 20),
        .stat = stat,
        .recheck = try num(get("TF_DSV41_CALIB_RECHECK"), d.recheck, 0, early),
        .shape = shape,
        .cycle = try num(get("TF_DSV41_CALIB_CYCLE"), d.cycle, 0, early),
        .deep = try num(get("TF_DSV41_CALIB_DEEP_REPS"), d.deep, 1, 50),
    };
}

fn num(raw: ?[]const u8, default: u32, lo: u32, hi: u32) !u32 {
    const v: i64 = if (trimmed(raw)) |s| try std.fmt.parseInt(i64, s, 10) else default;
    if (v < lo or v > hi) return error.CalibKnob;
    return @intCast(v);
}

fn trimmed(v: ?[]const u8) ?[]const u8 {
    const t = std.mem.trim(u8, v orelse return null, " \t\r\n");
    return if (t.len == 0) null else t;
}

/// calib.statistic: the median (the mean of the middle two when even) or the minimum of a window's timed runs.
pub fn statistic(samples: []const f64, stat: Stat) !f64 {
    if (samples.len == 0) return error.NoRuns;
    if (samples.len > max_samples) return error.TooManyRuns;
    var buf: [max_samples]f64 = undefined;
    const xs = buf[0..samples.len];
    @memcpy(xs, samples);
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    if (stat == .min) return xs[0];
    const m = xs.len / 2;
    return if (xs.len % 2 == 1) xs[m] else (xs[m - 1] + xs[m]) / 2.0;
}

/// statistics.median of a slice (sorted in place).
fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    const m = xs.len / 2;
    return if (xs.len % 2 == 1) xs[m] else (xs[m - 1] + xs[m]) / 2.0;
}

/// glm5_next.spark.calib.fit_line: the median of pairwise slopes, then the median intercept.
pub fn fitLine(gpa: Allocator, rows: []const f64, ys: []const f64) !struct { base: f64, slope: f64 } {
    var slopes: std.ArrayList(f64) = .empty;
    defer slopes.deinit(gpa);
    for (0..rows.len) |i| for (i + 1..rows.len) |j| try slopes.append(gpa, (ys[j] - ys[i]) / (rows[j] - rows[i]));
    const slope = median(slopes.items);
    const bases = try gpa.alloc(f64, rows.len);
    defer gpa.free(bases);
    for (bases, rows, ys) |*b, r, y| b.* = y - slope * r;
    return .{ .base = median(bases), .slope = slope };
}

/// calib.fit (VERSION 3's table): 1 row as measured, 2..8 on their fitted line, 9+ on a line of their own through
/// the 8-row value, non-decreasing.
pub fn fit(gpa: Allocator, v: []const f64) ![]f64 {
    if (v.len < 3) return gpa.dupe(f64, v);
    const first = @min(v.len, deep_from);
    var xs: [max_rows]f64 = undefined;
    var ys: [max_rows]f64 = undefined;
    if (v.len > max_rows) return error.TooManyRows;
    for (2..first + 1, 0..) |r, i| {
        xs[i] = @floatFromInt(r);
        ys[i] = v[r - 1];
    }
    const l = try fitLine(gpa, xs[0 .. first - 1], ys[0 .. first - 1]);
    const out = try gpa.alloc(f64, v.len);
    errdefer gpa.free(out);
    out[0] = v[0];
    for (2..first + 1) |r| out[r - 1] = l.base + l.slope * @as(f64, @floatFromInt(r));
    if (v.len > first) {
        for (first..v.len + 1, 0..) |r, i| {
            xs[i] = @floatFromInt(r);
            ys[i] = v[r - 1];
        }
        const l2 = try fitLine(gpa, xs[0 .. v.len - first + 1], ys[0 .. v.len - first + 1]);
        for (first + 1..v.len + 1) |r| out[r - 1] = out[first - 1] + @max(l2.slope, 0.0) * @as(f64, @floatFromInt(r - first));
    }
    for (1..out.len) |i| out[i] = @max(out[i], out[i - 1]);
    return out;
}

/// calib.table_of: `raw` = monotone, `fit` = fit.
pub fn tableOf(gpa: Allocator, times: []const f64, shape: Shape) ![]f64 {
    if (shape == .fit) return fit(gpa, times);
    const t = try gpa.dupe(f64, times);
    errdefer gpa.free(t);
    try costs_mod.monotone(gpa, t);
    return t;
}

/// calib.split: `total` rows over `out.len` slots, as even as can be (the first slots one more).
pub fn split(total: u32, out: []u32) void {
    const n: u32 = @intCast(out.len);
    for (out, 0..) |*o, i| o.* = total / n + @intFromBool(i < total % n);
}

/// calib.wide_buckets: the wide buckets `slots` slots can time (at most SLOT_ROWS rows a slot), past `rows`.
pub fn wideBuckets(slots: u32, rows_cap: u64, rows: u32, out: *[wide.len]u32) []u32 {
    var n: usize = 0;
    const top = @min(rows_cap, @as(u64, slot_rows) * @max(1, slots));
    for (wide) |b| if (rows < b and b <= top) {
        out[n] = b;
        n += 1;
    };
    return out[0..n];
}

fn ceilDiv(a: u32, b: u32) u32 {
    return (a + b - 1) / b;
}

/// One timed run's segments (slot, rows), a DSpark pass when `draft`.
pub const Seg = struct { slot: u32, rows: u32 };
pub const What = union(enum) { sweep: u32, recheck: u32, shared: u32, draft };
pub const Run = struct {
    segs: [4]Seg = undefined,
    nsegs: u8 = 0,
    warm: bool,
    what: What,

    pub fn segments(r: *const Run) []const Seg {
        return r.segs[0..r.nsegs];
    }
};

/// The measurement's runs in Python's order (calib.measure).
pub const Plan = struct {
    rows: u32,
    /// the slots the windows use (calib.measure's `n`)
    n: u32,
    wide_buf: [wide.len]u32 = undefined,
    nwide: u32 = 0,
    /// the 2-slot 4 + 4 window (Costs.slot) is timed
    per_slot: bool,
    draft: bool,
    m: Method,
    runs: std.ArrayList(Run) = .empty,

    /// `slots`: the engine's slots (Python's shape slots); `rows_cap`: joint.rows_cap (TF_DSV41_GRAPH_ROWS_MAX, 64);
    /// `rows`: the widest one-slot window (MAX_ROWS, 16); `draft`: the engine has a DSpark pass.
    pub fn init(gpa: Allocator, slots: u32, rows_cap: u64, rows: u32, m: Method, draft: bool) !Plan {
        if (rows == 0 or rows > max_rows) return error.BadRows;
        var p: Plan = .{ .rows = rows, .n = 1, .per_slot = false, .draft = draft, .m = m };
        errdefer p.runs.deinit(gpa);
        const w = wideBuckets(slots, rows_cap, rows, &p.wide_buf);
        p.nwide = @intCast(w.len);
        const want: u32 = if (w.len > 0) @max(2, ceilDiv(w[w.len - 1], slot_rows)) else 2;
        p.n = @min(@max(1, slots), want);
        p.per_slot = p.n >= 2 and rows >= 8;
        try p.block(gpa, @min(rows, early), false);
        var r: u32 = early + 1;
        while (r <= rows) : (r += 1) try p.add(gpa, &.{.{ .slot = 0, .rows = r }}, .{ .sweep = r }, 1, m.deep);
        var j: u32 = 0;
        if (p.per_slot) {
            try p.add(gpa, &.{ .{ .slot = 0, .rows = 4 }, .{ .slot = 1, .rows = 4 } }, .{ .shared = j }, 1, light_reps);
            j += 1;
        }
        for (w) |b| {
            const k = ceilDiv(b, slot_rows);
            var parts: [4]u32 = undefined;
            split(b, parts[0..k]);
            var segs: [4]Seg = undefined;
            for (segs[0..k], parts[0..k], 0..) |*s, x, i| s.* = .{ .slot = @intCast(i), .rows = x };
            try p.add(gpa, segs[0..k], .{ .shared = j }, 1, light_reps);
            j += 1;
        }
        try p.block(gpa, @min(m.recheck, rows), true);
        if (draft) try p.add(gpa, &.{}, .draft, 1, light_reps);
        return p;
    }

    pub fn deinit(p: *Plan, gpa: Allocator) void {
        p.runs.deinit(gpa);
    }

    pub fn wideBucketsOf(p: *const Plan) []const u32 {
        return p.wide_buf[0..p.nwide];
    }

    /// Entries a rank gathers: [draft, rows 1..rows, the shared windows...].
    pub fn entries(p: *const Plan) usize {
        return 1 + p.rows + p.nwide + @intFromBool(p.per_slot);
    }

    /// calib.measure's `block`: rows 1..min(cycle, hi) as interleaved cycles (`warm` discarded, `reps` timed), the
    /// other rows up to `hi` a row at a time with the same plan.
    fn block(p: *Plan, gpa: Allocator, hi: u32, recheck: bool) !void {
        const cyc = @min(p.m.cycle, hi);
        for (0..p.m.warm + p.m.reps) |k| {
            var r: u32 = 1;
            while (r <= cyc) : (r += 1) try p.runs.append(gpa, one(r, k < p.m.warm, recheck));
        }
        var r: u32 = cyc + 1;
        while (r <= hi) : (r += 1) for (0..p.m.warm + p.m.reps) |k| try p.runs.append(gpa, one(r, k < p.m.warm, recheck));
    }

    fn one(r: u32, warm: bool, recheck: bool) Run {
        var x: Run = .{ .warm = warm, .what = if (recheck) .{ .recheck = r } else .{ .sweep = r } };
        x.segs[0] = .{ .slot = 0, .rows = r };
        x.nsegs = 1;
        return x;
    }

    fn add(p: *Plan, gpa: Allocator, segs: []const Seg, what: What, warm: u32, reps: u32) !void {
        for (0..warm + reps) |k| {
            var x: Run = .{ .warm = k < warm, .what = what, .nsegs = @intCast(segs.len) };
            @memcpy(x.segs[0..segs.len], segs);
            try p.runs.append(gpa, x);
        }
    }

    /// One rank's kept values [draft, rows 1..rows (the lower of sweep and recheck), shared...] in ms from its
    /// samples (one a run, in `runs` order; warm-ups skipped). No DSpark pass: draft 0 (Python's draft = 0.0).
    pub fn values(p: *const Plan, samples: []const f64, out: []f64) !void {
        if (samples.len != p.runs.items.len or out.len != p.entries()) return error.BadSamples;
        var buf: [max_samples]f64 = undefined;
        // each entry's timed samples in run order: rows (sweep), rows (recheck), shared, draft
        out[0] = 0.0;
        for (0..p.rows) |i| out[1 + i] = (try p.stat(samples, .{ .sweep = @intCast(i + 1) }, &buf)) orelse return error.NoRuns;
        for (0..p.rows) |i| if (try p.stat(samples, .{ .recheck = @intCast(i + 1) }, &buf)) |x| {
            out[1 + i] = @min(out[1 + i], x);
        };
        const ns = p.nwide + @intFromBool(p.per_slot);
        for (0..ns) |j| out[1 + p.rows + j] = (try p.stat(samples, .{ .shared = @intCast(j) }, &buf)) orelse return error.NoRuns;
        if (p.draft) out[0] = (try p.stat(samples, .draft, &buf)) orelse return error.NoRuns;
    }

    fn stat(p: *const Plan, samples: []const f64, what: What, buf: *[max_samples]f64) !?f64 {
        var n: usize = 0;
        for (p.runs.items, samples) |r, x| if (!r.warm and std.meta.eql(r.what, what)) {
            if (n == buf.len) return error.TooManyRuns;
            buf[n] = x;
            n += 1;
        };
        if (n == 0) return null;
        return try statistic(buf[0..n], p.m.stat);
    }

    /// The table every rank holds: each rank's values as microsecond ints (Python round, half to even), the maximum
    /// over ranks a entry (`gather_max`), then table_of, Costs.slot from the 4 + 4 window against the 8-row entry, the
    /// wide buckets less the further slots' overhead, extended. `ranks[r]`: rank r's `values`.
    pub fn build(p: *const Plan, gpa: Allocator, ranks: []const []const f64) !Costs {
        const e = p.entries();
        const both = try gpa.alloc(f64, e);
        defer gpa.free(both);
        for (0..e) |i| {
            var hi: i64 = std.math.minInt(i64);
            for (ranks) |v| {
                if (v.len != e) return error.BadSamples;
                hi = @max(hi, costs_mod.roundEven(v[i] * 1000.0));
            }
            both[i] = @as(f64, @floatFromInt(hi)) / 1000.0;
        }
        const table = try tableOf(gpa, both[1 .. p.rows + 1], p.m.shape);
        defer gpa.free(table);
        var rest = both[p.rows + 1 ..];
        var per_slot: f64 = 0.0;
        if (p.per_slot) {
            per_slot = @max(0.0, rest[0] - table[7]);
            rest = rest[1..];
        }
        var pts: [wide.len]costs_mod.Point = undefined;
        for (p.wideBucketsOf(), rest, pts[0..p.nwide]) |b, t, *pt| pt.* = .{ .rows = b, .ms = t - per_slot * @as(f64, @floatFromInt(ceilDiv(b, slot_rows) - 1)) };
        return .{ .verify = try costs_mod.extend(gpa, table, pts[0..p.nwide]), .draft = both[0], .slot = per_slot };
    }
};

/// glm5_next.spark.calib.TEXT: the calibration text (prose, a numbered explanation and code), tokenized by the
/// checkpoint's tokenizer.json (calib_gpu.zig), the first PROMPT_MAX ids.
pub const text =
    \\The committee met on Tuesday to review the quarterly results. Revenue grew by eleven percent, driven mostly
    \\by the new subscription plans, while operating costs stayed flat. Two questions remained open: whether to expand
    \\the support team before the summer, and how to price the enterprise tier for customers with more than fifty seats.
    \\
    \\Here is a short function that answers the second question from the usage data:
    \\
    \\```python
    \\def enterprise_price(seats: int, base: float = 12.0, discount: float = 0.15) -> float:
    \\    """Monthly price for a team: the first fifty seats at the base rate, the rest discounted."""
    \\    if seats <= 50:
    \\        return seats * base
    \\    return 50 * base + (seats - 50) * base * (1.0 - discount)
    \\
    \\
    \\for team in (10, 50, 120):
    \\    print(f"{team} seats: ${enterprise_price(team):,.2f} a month")
    \\```
    \\
    \\In summary, the pricing keeps small teams unchanged and rewards larger ones, and the
;
pub const prompt_max = 256;

/// glm5_next.spark.calib.prompt_ids after the tokenizer: the ids below `vocab`, the first PROMPT_MAX, or
/// `fallback_ids` when fewer than 16 remain. `ids` is filtered in place; the result is a slice of it or of `fb`.
pub fn promptIds(ids: []u32, vocab: u32, fb: *[64]u32) []const u32 {
    var n: usize = 0;
    for (ids) |t| if (t < vocab and n < prompt_max) {
        ids[n] = t;
        n += 1;
    };
    if (n >= 16) return ids[0..n];
    return fallbackIds(vocab, fb);
}

/// glm5_next.spark.calib.fallback_ids(vocab, 64): ids spread over the vocabulary by a fixed rule.
pub fn fallbackIds(vocab: u32, out: *[64]u32) []const u32 {
    const m: u64 = @intCast(@max(@as(i64, vocab) - 2, 1));
    for (out, 0..) |*o, i| o.* = @intCast((7919 * (@as(u64, i) + 1) + 104729) % m + 1);
    return out;
}

/// calib.measure's texts: slot i's prompt is `prompt` less its last CUT x i tokens (when more than 8 remain).
pub fn cutOf(prompt_len: usize, i: u32) usize {
    const c = @as(usize, cut) * i;
    return if (prompt_len > c + 8) prompt_len - c else prompt_len;
}
