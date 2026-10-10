//! A prefill segment on the prod fast-prefill path (TF_DSV41_PREFILL_KERNELS=fast with prod-knobs.env), emitted as
//! calls (calls.zig) by block.zig's Emitter: every launch with its arguments and its tensors named by role, in the
//! Python engine's order (forward._run / _layers, blocks.attention's serial path, prefill_moe.forward, mhc_pf). M1
//! checks the calls against the capture's prefill phase launch by launch; the forward issues the same calls.
//!
//! What differs from a decode window (block.zig):
//! - a dense projection is prefill_mm.matmul: upstream's rot_in into the workspace, W_q unpacked from the words
//!   (dense3's lanes, upstream's strips, or the shared expert's stored tiles), the Triton `_gemm`, in column blocks
//!   whose W_q fits the 64 MiB workspace;
//! - an mHC site is mhc_pf.run + `_finish_k` from 256 rows (Triton `_site` below, and for the post-only / final sites);
//! - the SWA rows go through the staging ring (forward.staging) when the slot's ring cannot hold the segment;
//! - the indexer materialises its scores (`_scores`, `_keys`, the candidate source's `_block_keys`; in row blocks
//!   under the index budget) and torch picks the top-k (glue);
//! - the attention core is the fused prefill kernel (`_fused`), group-major for wo_a's groups (TF_DSV41_PF_COPIES);
//! - the routed experts are x3gm (rot, then gate/up and down once a K2 width present, upstream's combine), the shared
//!   expert three GEMMs, the partial bf16(routed + shared) before its exchange;
//! - the head is the decode path's linear over every row, 128 rows a launch.
//!
//! Roles as block.zig's ("s." persistent, "w." the segment, "L." a layer), plus the workspace "s.pf.*", the staging
//! ring "s.stage.*" and x3gm's pointer tables "s.L<i>.gm.*". The attention's exchange lands in "L.recv.attn": the
//! caching allocator hands each layer another storage for it.
//!
//! Glue steps (no captured launch; the forward runs them): positions, embed, ix_w (the head weights' fp64 -> bf16 ->
//! fp32 casts), proj (a compressor projection to fp32, ratio 2: [kv | gate], then [carry | rows]), zeros, stage_in /
//! stage_out, block_pos (a row block's first position), cand_keys, topk (index.top_positions), counts
//! (visible_counts), exchange, gm_picks (kit weights, the shared slot masked, the routed slots' contiguous copies),
//! gm_ticket, gm_plan (x3gm.plan of one width's picks), swiglu (the shared expert's clamped SiLU product), moe_sum
//! (bf16(routed + shared)), engram_rows, trace, kit_logits.
//!
//! A short segment (at most 16 rows: a prompt's ragged tail, or a whole short prompt) is Python's mix (pod 24's
//! captures at 1, 12 and 16 rows): the prefill projections, indexer scores, fused attention, x3gm experts and shared
//! GEMMs as above, but every kernel that has a row path for at most 16 rows takes it, as in a decode window
//! (block.zig, block_moe.zig): mhc_cuda for every site, post-only and the final norm (no `_finish_k`), the L2 prefetch
//! (o / f / x sites), attn_cuda's top-k (a Reindex layer scoring over its candidate blocks, no `_keys` or torch top-k)
//! and the narrow router. The exchanges land in the decode window's "w.recv.*" that mhc_cuda reads.
//!
//! From TF_DSV41_INDEX_STREAM_MIN visible keys a full-mode layer without the candidates' scores (2 / 8 / 14) selects
//! with stream_topk (`_stream` + the merge glue, longpf.zig): no materialised scores, so any prompt up to the limit.
//!
//! Not taken here (error.Unsupported): uniform (not ragged) expert stacks, gate / up sharing their sign vector (x3gm
//! shx).

const std = @import("std");
const dk = @import("dsv41_kernels");
const calls = @import("calls.zig");
const block = @import("block.zig");
const config = @import("config.zig");
const moe = @import("block_moe.zig");
const longpf = @import("longpf.zig");
const gm2pf = @import("gm2pf.zig");
const ced = @import("ced.zig");
const Config = config.Config;
const E = block.Emitter;
const Arg = calls.Arg;
const Dt = calls.Dt;
const dense = dk.dense;
const PfDense = @import("prod_knobs.zig").PfDense;

// prod knobs and the Python engine's constants
const ws_limit: i64 = 64 << 20; // TF_DSV41_PREFILL_WS_MIB: a workspace W_q's bytes at most (column blocks)
const had_scale: f64 = 0.08838834764831845; // exl3.prefill.HAD_SCALE (the literal: 1 / sqrt(128) rounds the other way)
const pf_rows: i64 = 256; // TF_DSV41_MHC_PF_ROWS / TF_DSV41_MHC_ROWS: mhc_pf and the wide `_site` tile from here
const mhc_bm: i64 = 64; // TF_DSV41_MHC_BM
const split_rows: i64 = 32; // TF_DSV41_MHC_SPLIT_ROWS: out-of-place boundaries, a program a stream
pub const stream_min: i64 = 4096; // TF_DSV41_INDEX_STREAM_MIN (block.zig: a decode window past it would stream)
const bmq: i64 = 32; // TF_DSV41_PREFILL_ATTN_BMQ
const kt: i64 = 32; // csa2.attn.KT
const router_chunk: i64 = 2048; // router_gemv.CHUNK: rows a route launch
const gm_bm: i64 = 64; // x3gm: members a pass of the gate/up (cfg 1) and down (cfg 0) tiles below 3,072 rows
const gm_widths: u32 = 7; // x3gm.MAX_WIDTHS: the ticket's down half starts there
const index_budget: i64 = 256 << 20; // TF_DSV41_INDEX_BUDGET_MIB's default: block.Options.index_budget is the run's
const dtopk_rows: i64 = 64; // dtopk.MAX_ROWS: windows up to this many rows select with attn_cuda's top-k (backend._dt)
const overlap_min: i64 = 256; // TF_DSV41_PF_OVERLAP: rows a piece at least (a smaller GEMM's waves cost more than it hides)

fn size(dt: Dt) i64 {
    return switch (dt) {
        .bf16, .f16, .i16 => 2,
        .f32, .i32 => 4,
        .f64, .i64 => 8,
        .i8, .u8, .bool => 1,
    };
}

fn cdiv(a: i64, b: i64) i64 {
    return @divFloor(a + b - 1, b);
}

/// How a matrix's words are kept: dense3's lanes, upstream's strips, or the checkpoint's tiles as stored (the shared
/// expert: prefill_moe.shared_linears, layout "stored").
const Form = enum { lanes, strips, stored };

/// Where a matmul's [m, N] result goes: `role`'s rows of `stride` elements from byte `offset`.
const Out = struct { role: calls.Role, dt: Dt, stride: i64, offset: i64 = 0 };

/// One prefill segment: the emitter, the SWA ring its rows go through, and what the segment's layers left (layer
/// 20's candidates).
const Seg = struct {
    e: *E,
    /// the SWA ring the layers store into and attend over: the slot's (2 x window rows) or the staging ring
    staged: bool,
    ring: i64,
    /// the candidate source ran in this segment (reindex layers read its candidates)
    cand: bool = false,
    /// at most 16 rows: Python's short segment (the module doc)
    short: bool = false,
    /// CED replay's passes (`emitReplay`, ced.zig); `.whole`: a full segment
    part: Part = .whole,
    /// a multi-segment run (`emitMulti`): its segments; the attention core and the compressor run a segment at a
    /// time on that segment's slot (glue "pf_seg"), every other step over all the run's rows
    multi: ?[]const Span = null,
    /// TF_DSV41_PF_OVERLAP: this layer's attention / MoE partial already went out a piece at a time (`pieces`), so
    /// `run` emits no whole exchange after it
    sent_attn: bool = false,
    sent_moe: bool = false,
    /// TF_DSV41_PF_OVERLAP_SITE: the last exchange's k pieces are all on the side stream, marks 1..k; the next
    /// boundary site runs in those pieces (`sitePieces`), anything else joins first. 0: none pending
    pend: i64 = 0,

    // -----------------------------------------------------------------------------------------------------------
    // TF_DSV41_PF_OVERLAP: the exchanges in row pieces, overlapped with their producers' next piece

    /// Row pieces of this segment's exchanges: Options.pf_overlap when it applies (branches' side stream, TP=2, not a
    /// short segment, at least `overlap_min` rows a piece), else 1 (one exchange after its producer).
    fn pieces(s: *const Seg) i64 {
        const e = s.e;
        const k = e.o.pf_overlap;
        if (k < 2 or s.short or !e.o.branches or e.o.pf_tbo or e.o.world != 2 or e.n < k * overlap_min) return 1;
        return k;
    }

    /// Piece `p` of `k` over `n` rows: [r0, r1) on the 16-row grid, the last to `n`.
    fn pieceAt(n: i64, k: i64, p: i64) [2]i64 {
        const cut = struct {
            fn at(nn: i64, kk: i64, q: i64) i64 {
                return if (q >= kk) nn else @divFloor(@divFloor(nn * q, kk), 16) * 16;
            }
        };
        return .{ cut.at(n, k, p), cut.at(n, k, p + 1) };
    }

    /// Exchange of rows [r0, r0 + rows) of `part` [n, D] into `into` [world, n, D] (glue exchange_rows: this rank's
    /// rows copied, the peer's received: the whole exchange's bytes at the same places, a piece at a time). Every piece
    /// but the last goes on the side stream once the main stream issued its producer (fork); the last runs on the main
    /// stream after the side stream's pieces (join), so the consumer (the next mHC site) reads a whole buffer.
    /// TF_DSV41_PF_OVERLAP_SITE: every piece p (the last too) on the side, marking p + 1 for the next site's piece p;
    /// the call after the last piece joins unless that site takes the pieces (`pend`).
    fn sendPiece(s: *Seg, part: Arg, into: Arg, r0: i64, rows: i64, p: i64, k: i64) !void {
        const e = s.e;
        const send = try rowsOf(e, part, .bf16, r0, rows, e.cfg.hidden);
        const last = p + 1 == k;
        if (last and !e.o.pf_overlap_site) e.br_join = true else {
            e.br_fork = true;
            e.br_side = true;
            if (e.o.pf_overlap_site) e.br_mark = @intCast(p + 1);
        }
        defer e.br_side = false;
        // the piece's all-gather lands in "L.xrows" [world, rows, D] (a planned role: in the 4K workspace and its
        // price), then each rank's rows are copied into `into`: the collective's own channels, no NCCL p2p state
        const tmp = try e.buf("L.xrows", .{}, .bf16, &.{ e.o.world, rows, e.cfg.hidden });
        try e.glue("exchange_rows", &.{ send, into, .{ .i = r0 }, tmp });
        if (last and e.o.pf_overlap_site) {
            s.pend = k;
            e.br_join = true;
        }
    }

    // -----------------------------------------------------------------------------------------------------------
    // dense EXL3 projections (prefill_mm.matmul)

    /// y = x [m, K] @ W [K, N] (`w`'s prefix): per column block, upstream's rot_in, the W_q unpack, `_gemm`.
    fn matmul(s: *Seg, w: []const u8, form: Form, x: Arg, k: i64, n: i64, y: Out) !void {
        const e = s.e;
        const m = e.n;
        const k2: i64 = try e.w.k2(w);
        const tw = 4 * k2;
        const nb = @divExact(n, 128);
        if (e.o.pfd) |p| if (PfDense.eligible(k, n, @intCast(k2), form == .lanes)) return s.fused(p, w, form, x, k, n, y);
        // prefill_mm.column_blocks: whole 128-column blocks whose W_q fits the workspace (strips / lanes only)
        const per: i64 = if (k * n * 2 <= ws_limit or form == .stored) nb else @max(1, @divFloor(ws_limit, k * 2 * 128));
        const st = dense.strides(@intCast(k), @intCast(k2));
        var b0: i64 = 0;
        while (b0 < nb) : (b0 += per) {
            const b1 = @min(nb, b0 + per);
            const c0 = 128 * b0;
            const nc = 128 * (b1 - b0);
            const xh = try e.buf("s.pf.xh", .{}, .f16, &.{ m, k });
            try e.ext("tensorfold_exl3_linear_v4.rot_in", &.{ x, try e.weight("{s}.suh", .{w}, .f16, &.{k}), xh });
            const wq = try e.buf("s.pf.w", .{}, .f16, &.{ k, nc });
            const blk_words = @divExact(k, 16) * 8 * tw; // words of a 128-column block
            switch (form) {
                .lanes => try e.ext("tf_dsv41_dense3_v1.unpack", &.{
                    try e.view(.{ .buf = try e.fmt("s.{s}.lanes", .{w}) }, .i32, &.{ b1 - b0, @divExact(k, 16), 8, tw }, &.{ blk_words, 8 * tw, tw, 1 }, b0 * blk_words * 4),
                    wq, .{ .i = st[0] }, .{ .i = st[1] }, .{ .i = k2 },
                }),
                .strips => try e.ext("tensorfold_exl3_linear_v4.unpack", &.{
                    // one block of 128 columns: the stored trellis is the strips (block.zig's upstream)
                    if (n == 128)
                        try e.view(.{ .weight = try e.fmt("{s}.trellis", .{w}) }, .i32, &.{ 1, @divExact(k, 16), 8, tw }, &.{ 8 * tw, 8 * tw, tw, 1 }, 0)
                    else
                        try e.view(.{ .buf = try e.fmt("s.{s}.T", .{w}) }, .i32, &.{ b1 - b0, @divExact(k, 16), 8, tw }, &.{ blk_words, 8 * tw, tw, 1 }, b0 * blk_words * 4),
                    wq, .{ .i = st[0] }, .{ .i = st[1] }, .{ .i = k2 }, .{ .i = 2 },
                }),
                // Exl3Linear.strides of the stored layout: (words between k tiles, between 16-column tiles)
                .stored => try e.ext("tensorfold_exl3_linear_v4.unpack", &.{
                    try e.view(.{ .weight = try e.fmt("{s}.trellis", .{w}) }, .i32, &.{ @divExact(k, 16), @divExact(n, 16), tw }, &.{ @divExact(n, 16) * tw, tw, 1 }, 0),
                    wq, .{ .i = @divExact(n, 16) * tw }, .{ .i = 8 * tw }, .{ .i = k2 }, .{ .i = 2 },
                }),
            }
            const svh = try e.view(.{ .weight = try e.fmt("{s}.svh", .{w}) }, .f16, &.{nc}, &.{1}, c0 * 2);
            const out = try e.view(y.role, y.dt, &.{ m, nc }, &.{ y.stride, 1 }, y.offset + c0 * size(y.dt));
            // exl3.prefill.tiles: BM 128, BK 32, group 8 (the shape's alone)
            try e.triton("_gemm", .{ cdiv(m, 128) * @divExact(nc, 128), 1, 1 }, &.{
                .{ .name = "X", .arg = xh },                                   .{ .name = "W", .arg = wq },
                .{ .name = "H", .arg = try e.buf("s.pf.had", .{}, .bf16, &.{ 128, 128 }) }, .{ .name = "SVH", .arg = svh },
                .{ .name = "BIAS", .arg = svh },                               .{ .name = "OUT", .arg = out },
                .{ .name = "M", .arg = .{ .i = m } },                          .{ .name = "o_stride", .arg = .{ .i = y.stride } },
                .{ .name = "K", .arg = .{ .i = k } },                          .{ .name = "N", .arg = .{ .i = nc } },
                .{ .name = "BM", .arg = .{ .i = 128 } },                       .{ .name = "BK", .arg = .{ .i = 32 } },
                .{ .name = "GROUP", .arg = .{ .i = 8 } },                      .{ .name = "HAS_BIAS", .arg = .{ .b = false } },
                .{ .name = "SCALE", .arg = .{ .f = had_scale } },
            });
        }
    }

    /// pfdense.matmul (TF_DSV41_PF_DENSE=fused): upstream's rot_in into the workspace, then one fused GEMM over the
    /// whole layer (no W_q workspace, so no column blocks) reading the same words the unpack would.
    fn fused(s: *Seg, p: *const PfDense, w: []const u8, form: Form, x: Arg, k: i64, n: i64, y: Out) !void {
        const e = s.e;
        const m = e.n;
        const k2: u32 = try e.w.k2(w);
        const tw: i64 = 4 * @as(i64, k2);
        const nb = @divExact(n, 128);
        const blk_words = @divExact(k, 16) * 8 * tw;
        const xh = try e.buf("s.pf.xh", .{}, .f16, &.{ m, k });
        try e.ext("tensorfold_exl3_linear_v4.rot_in", &.{ x, try e.weight("{s}.suh", .{w}, .f16, &.{k}), xh });
        const st = dense.strides(@intCast(k), k2);
        const words: Arg, const sk: i64, const snb: i64 = switch (form) {
            .lanes => .{ try e.view(.{ .buf = try e.fmt("s.{s}.lanes", .{w}) }, .i32, &.{ nb, @divExact(k, 16), 8, tw }, &.{ blk_words, 8 * tw, tw, 1 }, 0), st[0], st[1] },
            .strips => .{ if (n == 128)
                try e.view(.{ .weight = try e.fmt("{s}.trellis", .{w}) }, .i32, &.{ 1, @divExact(k, 16), 8, tw }, &.{ 8 * tw, 8 * tw, tw, 1 }, 0)
            else
                try e.view(.{ .buf = try e.fmt("s.{s}.T", .{w}) }, .i32, &.{ nb, @divExact(k, 16), 8, tw }, &.{ blk_words, 8 * tw, tw, 1 }, 0), st[0], st[1] },
            .stored => .{ try e.view(.{ .weight = try e.fmt("{s}.trellis", .{w}) }, .i32, &.{ @divExact(k, 16), @divExact(n, 16), tw }, &.{ @divExact(n, 16) * tw, tw, 1 }, 0), @divExact(n, 16) * tw, 8 * tw },
        };
        const c = p.pick(@intCast(k), @intCast(n), k2, @intCast(m));
        try e.ext("tf_dsv41_pfdense_v1.gemm", &.{
            xh,                                                          words,
            .{ .i = sk },                                                .{ .i = snb },
            try e.weight("{s}.svh", .{w}, .f16, &.{n}),                  .none,
            try e.view(y.role, y.dt, &.{ m, n }, &.{ y.stride, 1 }, y.offset), .{ .i = k2 },
            .{ .b = form == .lanes },                                    .{ .i = c[0] },
            .{ .i = c[1] },
        });
    }

    /// A projection into a whole contiguous buffer.
    fn proj(s: *Seg, w: []const u8, form: Form, x: Arg, k: i64, n: i64, y: Arg) !void {
        return s.matmul(w, form, x, k, n, .{ .role = y.t.role, .dt = y.t.dt, .stride = n, .offset = y.t.offset });
    }

    // -----------------------------------------------------------------------------------------------------------
    // mHC (mhc.py with mhc_pf)

    fn coefs(e: *E, set: u1) ![3]Arg {
        return .{
            try e.buf("w.c{d}.pre", .{set}, .f32, &.{ e.n, 4 }),
            try e.buf("w.c{d}.post", .{set}, .f32, &.{ e.n, 4 }),
            try e.buf("w.c{d}.comb", .{set}, .f32, &.{ e.n, 16 }),
        };
    }

    /// DSpark's tap (forward._layers): the new streams' mean at a boundary entering a target layer.
    fn tapOf(e: *E, L: u32, which: []const u8, mode: moe.Mode) !?Arg {
        if (!e.o.taps or mode != .boundary or !std.mem.eql(u8, which, "hc_attn")) return null;
        const t = e.cfg.dspark_targets.items();
        const j = std.mem.indexOfScalar(u32, t, L) orelse return null;
        const D: i64 = e.cfg.hidden;
        return try e.view(.{ .buf = "w.taps" }, .bf16, &.{ e.n, D }, &.{ @as(i64, @intCast(t.len)) * D, 1 }, @as(i64, @intCast(j)) * D * 2);
    }

    /// The exchange a post reads: the attention's partials at the FFN site, the MoE's anywhere else. Both are
    /// layer-scoped: Python's allocator gives each exchange its own storage (pod 18's capture moved the MoE's between
    /// layers); the role stays one buffer here, read by the next layer's first site.
    fn gathered(e: *E, which: []const u8) !Arg {
        const D: i64 = e.cfg.hidden;
        if (std.mem.eql(u8, which, "hc_ffn")) return e.buf("L.recv.attn", .{}, .bf16, &.{ e.o.world, e.n, D });
        return e.buf("L.recv.moe", .{}, .bf16, &.{ e.o.world, e.n, D });
    }

    /// One mixing site of layer L (`which`: hc_attn / hc_ffn): mhc.site / mhc.boundary, then `_finish_k` (a short
    /// segment: mhc_cuda alone).
    fn site(s: *Seg, L: u32, which: []const u8, mode: moe.Mode) !void {
        const e = s.e;
        const pend = s.pend;
        s.pend = 0;
        // the decoder replay's boundaries write the streams in place (replay.finish's Streams has no `alt`)
        if (s.short) return moe.mhcInto(e, L, which, mode, s.part == .decoder);
        const n = e.n;
        if (pend > 1 and mode == .boundary and n >= pend * pf_rows and n > split_rows) return s.sitePieces(L, which, pend, true);
        // TF_DSV41_MHC_SITE_ROWS: the boundary in row pieces on the main stream, each piece's partials and normed
        // input still in L2 when its `_finish_k` reads them
        if (mode == .boundary and e.o.mhc_site_rows > 0 and n > e.o.mhc_site_rows) {
            const k = cdiv(n, e.o.mhc_site_rows);
            if (n >= k * pf_rows) return s.sitePieces(L, which, k, false);
        }
        const D: i64 = e.cfg.hidden;
        const post = mode == .boundary;
        const x = try e.streams();
        const prev = try coefs(e, e.coef);
        const next = try coefs(e, 1 - e.coef);
        const b0 = try e.empty(.bf16);
        const f0 = try e.empty(.f32);
        const fnw = try e.buf("s.L{d}.{s}.fn16", .{ L, which }, .bf16, &.{ 6 * @as(i64, e.cfg.hc_mult), 4 * D });
        const part = try e.buf("L.mhc.part", .{}, .f32, &.{ n, 4, 40, 32 });
        const c = try e.buf("L.mhc.c", .{}, .bf16, &.{ n, D });
        const tap = try tapOf(e, L, which, mode);
        // the forward's second streams buffer (mhc.out_of_place): a boundary of at most 32 rows writes it
        const spare = post and n <= split_rows and s.part != .decoder;
        const xout = if (spare) try e.spare() else x;
        if (n >= pf_rows) {
            try e.ext("tf_dsv41_mhc_pf_v1.run", &.{
                x,                            if (post) xout else b0,
                if (post) try gathered(e, which) else b0, if (post) prev[1] else f0,
                if (post) prev[2] else f0,    if (mode != .site1) prev[0] else f0,
                fnw,                          part,
                c,                            tap orelse b0,
                .{ .i = n },                  .{ .i = @intFromEnum(mode) },
            });
        } else try s.siteTriton(x, if (post) xout else null, if (post) try gathered(e, which) else null, prev, if (mode != .site1) prev[0] else null, fnw, part, c, tap, if (mode == .site1) 1 else 2, true);
        try s.finish(.{ L, which }, next, try e.weight("L{d}.{s}", .{ L, if (std.mem.eql(u8, which, "hc_attn")) "attn_norm" else "ffn_norm" }, .f32, &.{D}), part, c, 0);
        e.coef = 1 - e.coef;
        if (spare) e.cur = 1 - e.cur;
    }

    /// TF_DSV41_PF_OVERLAP_SITE: a boundary site over the last exchange's k row pieces (`pieceAt`), piece p on the main
    /// stream once exchange piece p landed (its mark; the last after the whole side: join), so the site's earlier
    /// pieces run beside the later pieces' exchanges. mhc_pf.run and `_finish_k` are row-wise (a row's streams in place,
    /// its gathered rows, its coefficients, its partials; no row reads another, and each piece is >= 256 rows, so
    /// mhc_pf as for the whole): the site's calls over row views, every row the same bits.
    /// `overlapped` false (TF_DSV41_MHC_SITE_ROWS): the pieces one after another on the main stream, no marks.
    fn sitePieces(s: *Seg, L: u32, which: []const u8, k: i64, overlapped: bool) !void {
        const e = s.e;
        const n = e.n;
        const D: i64 = e.cfg.hidden;
        const W: i64 = e.o.world;
        const x = try e.streams();
        const prev = try coefs(e, e.coef);
        const next = try coefs(e, 1 - e.coef);
        const b0 = try e.empty(.bf16);
        const fnw = try e.buf("s.L{d}.{s}.fn16", .{ L, which }, .bf16, &.{ 6 * @as(i64, e.cfg.hc_mult), 4 * D });
        const part = try e.buf("L.mhc.part", .{}, .f32, &.{ n, 4, 40, 32 });
        const c = try e.buf("L.mhc.c", .{}, .bf16, &.{ n, D });
        const g = try gathered(e, which);
        const tap = try tapOf(e, L, which, .boundary);
        const nw = try e.weight("L{d}.{s}", .{ L, if (std.mem.eql(u8, which, "hc_attn")) "attn_norm" else "ffn_norm" }, .f32, &.{D});
        if (overlapped) e.br_join = false; // piece p waits for its own exchange piece; the last joins
        defer e.n = n;
        var p: i64 = 0;
        while (p < k) : (p += 1) {
            const r = pieceAt(n, k, p);
            const m = r[1] - r[0];
            e.n = m;
            if (overlapped) {
                if (p + 1 < k) e.br_wait = @intCast(p + 1) else e.br_join = true;
            }
            const xr = try rowsOf(e, x, .bf16, r[0], m, 4 * D);
            const nx: [3]Arg = .{ try rowsOf(e, next[0], .f32, r[0], m, 4), try rowsOf(e, next[1], .f32, r[0], m, 4), try rowsOf(e, next[2], .f32, r[0], m, 16) };
            const pr = try e.view(part.t.role, .f32, &.{ m, 4, 40, 32 }, &.{ 4 * 40 * 32, 40 * 32, 32, 1 }, r[0] * 4 * 40 * 32 * 4);
            const cr = try rowsOf(e, c, .bf16, r[0], m, D);
            try e.ext("tf_dsv41_mhc_pf_v1.run", &.{
                xr,                                                                                   xr,
                try e.view(g.t.role, .bf16, &.{ W, m, D }, &.{ n * D, D, 1 }, r[0] * D * 2),          try rowsOf(e, prev[1], .f32, r[0], m, 4),
                try rowsOf(e, prev[2], .f32, r[0], m, 16),                                            try rowsOf(e, prev[0], .f32, r[0], m, 4),
                fnw,                                                                                  pr,
                cr,                                                                                   if (tap) |t| try e.view(t.t.role, .bf16, &.{ m, D }, t.t.stride, t.t.offset + r[0] * t.t.stride[0] * 2) else b0,
                .{ .i = m },                                                                          .{ .i = @intFromEnum(moe.Mode.boundary) },
            });
            try s.finish(.{ L, which }, nx, nw, pr, cr, r[0]);
        }
        e.coef = 1 - e.coef;
    }

    /// mhc._launch's Triton `_site`: `xout` / `g` with the post, `pre` with collapse 2, `fnw` with the mix.
    fn siteTriton(s: *Seg, x: Arg, xout: ?Arg, g: ?Arg, prev: [3]Arg, pre: ?Arg, fnw: ?Arg, part: Arg, c: Arg, tap: ?Arg, collapse: i64, mix: bool) !void {
        const e = s.e;
        const n = e.n;
        const D: i64 = e.cfg.hidden;
        const post = g != null;
        const bm: i64 = if (n >= pf_rows) mhc_bm else 16;
        // a program a stream for small mixing windows, never in place with a post
        const split = mix and n <= split_rows and !(post and std.mem.eql(u8, xout.?.t.role.buf, x.t.role.buf));
        try e.triton("_site", .{ cdiv(n, bm), 40, if (split) 4 else 1 }, &.{
            .{ .name = "X", .arg = x },                                    .{ .name = "x_stride", .arg = .{ .i = 4 * D } },
            .{ .name = "XOUT", .arg = xout orelse x },                     .{ .name = "G", .arg = g orelse x },
            .{ .name = "g_rank", .arg = .{ .i = if (post) n * D else 0 } }, .{ .name = "POST", .arg = if (post) prev[1] else x },
            .{ .name = "COMB", .arg = if (post) prev[2] else x },          .{ .name = "PRE", .arg = pre orelse x },
            .{ .name = "FN", .arg = fnw orelse x },                        .{ .name = "PART", .arg = part },
            .{ .name = "C", .arg = c },                                    .{ .name = "TAP", .arg = tap orelse x },
            .{ .name = "tap_stride", .arg = .{ .i = if (tap) |t| t.t.stride[0] else 0 } }, .{ .name = "R", .arg = .{ .i = n } },
            .{ .name = "D", .arg = .{ .i = D } },                          .{ .name = "NB", .arg = .{ .i = 40 } },
            .{ .name = "BM", .arg = .{ .i = bm } },                        .{ .name = "BK", .arg = .{ .i = 16 } },
            .{ .name = "WORLD", .arg = .{ .i = if (post) e.o.world else 1 } }, .{ .name = "POST_ON", .arg = .{ .b = post } },
            .{ .name = "COLLAPSE", .arg = .{ .i = collapse } },           .{ .name = "MIX", .arg = .{ .b = mix } },
            .{ .name = "TAP_ON", .arg = .{ .b = tap != null } },           .{ .name = "SPLIT", .arg = .{ .b = split } },
        });
    }

    /// mhc._finish: the partials' sums, the next coefficients (a layer's site) and the normed input into "w.out".
    /// `r0`: the first of the e.n rows of "w.out" it writes (a site piece's; 0 for a whole site).
    fn finish(s: *Seg, hc: ?struct { u32, []const u8 }, next: [3]Arg, nw: Arg, part: Arg, c: Arg, r0: i64) !void {
        const e = s.e;
        const D: i64 = e.cfg.hidden;
        const coef = hc != null;
        const base = if (hc) |h| try e.weight("L{d}.{s}.base", .{ h[0], h[1] }, .f32, &.{6 * @as(i64, e.cfg.hc_mult)}) else part;
        const scale = if (hc) |h| try e.weight("L{d}.{s}.scale", .{ h[0], h[1] }, .f32, &.{3}) else part;
        try e.triton("_finish_k", .{ e.n, 1, 1 }, &.{
            .{ .name = "PART", .arg = part },                              .{ .name = "BASE", .arg = base },
            .{ .name = "SCALE", .arg = scale },                            .{ .name = "PRE", .arg = if (coef) next[0] else part },
            .{ .name = "POST", .arg = if (coef) next[1] else part },       .{ .name = "COMB", .arg = if (coef) next[2] else part },
            .{ .name = "C", .arg = c },                                    .{ .name = "NW", .arg = nw },
            .{ .name = "OUT", .arg = try rowsOf(e, try e.buf("w.out", .{}, .bf16, &.{ e.n, D }), .bf16, r0, e.n, D) }, .{ .name = "eps", .arg = .{ .f = E.dec(e.cfg.eps) } },
            .{ .name = "hc_eps", .arg = .{ .f = E.dec(e.cfg.hc_eps) } },  .{ .name = "post_alpha", .arg = .{ .f = E.dec(e.cfg.hc_post_alpha) } },
            .{ .name = "D", .arg = .{ .i = D } },                          .{ .name = "NB", .arg = .{ .i = 40 } },
            .{ .name = "ITERS", .arg = .{ .i = e.cfg.hc_sinkhorn_iters } }, .{ .name = "COEF", .arg = .{ .b = coef } },
            .{ .name = "BLOCK", .arg = .{ .i = @intCast(std.math.gcd(@as(u64, 1024), @as(u64, @intCast(D)))) } }, .{ .name = "UNROLL", .arg = .{ .b = true } },
            .{ .name = "NORM_FIRST", .arg = .{ .b = true } },
        });
    }

    /// mhc.post_only: the last exchange's post into `xout` (the streams in place, or the trace's copy).
    fn postOnly(s: *Seg, xout: Arg) !void {
        const e = s.e;
        s.pend = 0; // reads the whole gathered buffer: it joined (sendPiece)
        if (s.short) return moe.postOnly(e, xout);
        const prev = try coefs(e, e.coef);
        const x = try e.streams();
        try s.siteTriton(x, xout, try gathered(e, ""), prev, null, null, x, x, null, 0, false);
    }

    /// mhc.final: the last post, collapse 2, the final norm; then the head's columns over every row.
    fn final(s: *Seg) !void {
        const e = s.e;
        s.pend = 0;
        if (s.short) return moe.finish(e); // mhc_cuda's final, then the same head
        const n = e.n;
        const D: i64 = e.cfg.hidden;
        const x = try e.streams();
        const prev = try coefs(e, e.coef);
        const part = try e.buf("L.mhc.part", .{}, .f32, &.{ n, 4, 40, 32 });
        const c = try e.buf("L.mhc.c", .{}, .bf16, &.{ n, D });
        try s.siteTriton(x, x, try gathered(e, ""), prev, prev[0], null, part, c, null, 2, false);
        try s.finish(null, prev, try e.weight("norm", .{}, .f32, &.{D}), part, c, 0);
        // the head: the decode path's linear (prefill_mm.fast off), upstream's 128 rows a launch
        const V: i64 = e.cfg.vocab / e.o.world;
        var a: i64 = 0;
        while (a < n) : (a += 128) {
            const m = @min(n - a, 128);
            e.n = m;
            defer e.n = n;
            try e.upstream("head", try e.view(.{ .buf = "w.out" }, .bf16, &.{ m, D }, &.{ D, 1 }, a * D * 2), D, V, try e.view(.{ .buf = "w.logits" }, .f32, &.{ m, V }, &.{ V, 1 }, a * V * 4));
        }
        try e.glue("kit_logits", &.{try e.buf("w.logits", .{}, .f32, &.{ n, V })});
    }

    // -----------------------------------------------------------------------------------------------------------
    // attention (blocks.attention's serial path, _segment with PF_COPIES)

    fn pos(e: *E) !Arg {
        return e.buf("w.pos", .{}, .i32, &.{1});
    }

    fn rope(e: *E, x: Arg, heads: i64, dim: i64) !void {
        return ropeAt(e, x, heads, dim, "w.pos64");
    }

    /// `rope` reading each row's position from `positions` (a multi-segment run: every segment's, "w.mpos64").
    fn ropeAt(e: *E, x: Arg, heads: i64, dim: i64, positions: []const u8) !void {
        const n = e.n;
        const xv = try e.view(x.t.role, .bf16, &.{ n, heads, dim }, &.{ heads * dim, dim, 1 }, 0);
        try e.triton("_rope", .{ n, @divExact(heads, 4), 1 }, &.{
            .{ .name = "X", .arg = xv },                               .{ .name = "x_rs", .arg = .{ .i = heads * dim } },
            .{ .name = "x_hs", .arg = .{ .i = dim } },                 .{ .name = "OUT", .arg = xv },
            .{ .name = "o_rs", .arg = .{ .i = heads * dim } },         .{ .name = "o_hs", .arg = .{ .i = dim } },
            .{ .name = "CS", .arg = try e.ropeTable() },              .{ .name = "cs_rs", .arg = .{ .i = 64 } },
            .{ .name = "POS", .arg = try e.t(.{ .buf = positions }, .i64, &.{n}) }, .{ .name = "H", .arg = .{ .i = heads } },
            .{ .name = "NOPE", .arg = .{ .i = dim - 64 } },            .{ .name = "NB", .arg = .{ .i = 64 } },
            .{ .name = "HALF", .arg = .{ .i = 32 } },                  .{ .name = "BH", .arg = .{ .i = 4 } },
            .{ .name = "INV", .arg = .{ .b = false } },                .{ .name = "COPY", .arg = .{ .b = false } },
            .{ .name = "PDL", .arg = .{ .b = false } },
        });
    }

    /// The SWA ring this segment's layer L stores into (values, scales): the staging ring or the slot's.
    fn swa(s: *Seg, L: u32) ![2]Arg {
        const e = s.e;
        if (s.staged) return .{ try e.buf("s.stage.v", .{}, .u8, &.{ s.ring, 576 }), try e.buf("s.stage.s", .{}, .u8, &.{ s.ring, 8 }) };
        return slotRing(e, L);
    }

    fn slotRing(e: *E, L: u32) ![2]Arg {
        const r = 2 * @as(i64, e.cfg.window);
        return .{ try e.buf("s.L{d}.swa.v", .{L}, .u8, &.{ r, 576 }), try e.buf("s.L{d}.swa.s", .{L}, .u8, &.{ r, 8 }) };
    }

    /// A KV source's compressed rows (values, scales): the slot's contiguous ones, or the pool's (block.zig).
    fn pool(e: *E, L: u32) ![2]Arg {
        return e.pool(L, e.cfg.compressRatio(L));
    }

    fn indexKeys(e: *E, L: u32) !Arg {
        return e.indexKeys(L);
    }

    /// compress.kv_store: rows into the SWA ring (ratio 0) or a compressed family (pool mode: through the slot's
    /// table, the split table under split KV).
    fn kvStore(s: *Seg, lat: Arg, rows: [2]Arg, ratio: i64) !void {
        const e = s.e;
        const pg = try e.paging(ratio, true);
        try e.triton("_kv_store", .{ e.n, 1, 1 }, &.{
            .{ .name = "LAT", .arg = lat },                          .{ .name = "l_stride", .arg = .{ .i = e.cfg.head_dim } },
            .{ .name = "CS", .arg = try e.ropeTable() },             .{ .name = "cs_stride", .arg = .{ .i = 64 } },
            .{ .name = "V", .arg = rows[0] },                        .{ .name = "S", .arg = rows[1] },
            .{ .name = "v_stride", .arg = .{ .i = pg.vs } },         .{ .name = "s_stride", .arg = .{ .i = pg.ss } },
            .{ .name = "POS", .arg = try pos(e) },                   .{ .name = "RATIO", .arg = .{ .i = ratio } },
            .{ .name = "RING", .arg = .{ .i = if (ratio == 0) s.ring else 1 } }, .{ .name = "PT", .arg = pg.pt },
            .{ .name = "PSH", .arg = .{ .i = pg.psh } },             .{ .name = "SL", .arg = .none },
            .{ .name = "PTS", .arg = .{ .i = 0 } },                  .{ .name = "ROWS", .arg = .{ .b = false } },
            .{ .name = "PDL", .arg = .{ .b = false } },
        });
    }

    /// A KV source's compressor (blocks._compress): pooled rows normed into its pool, the index keys from them.
    fn compress(s: *Seg, L: u32, projection: Arg) !void {
        const e = s.e;
        const n = e.n;
        const hd: i64 = e.cfg.head_dim;
        const ID: i64 = e.cfg.index_dim;
        const ratio: i64 = e.cfg.compressRatio(L);
        // ratio 2: [the slot's carry (position start - 1) | the rows]
        const buf = if (ratio == 2) try e.buf("L.cbuf", .{}, .f32, &.{ n + 1, 2 * hd }) else projection;
        if (ratio == 2) try e.glue("proj_carry", &.{ try e.buf("s.L{d}.carry", .{L}, .f32, &.{ 1, 2 * hd }), projection, buf });
        const lat = try e.buf("L.lat", .{}, .bf16, &.{ n, hd });
        try e.glue("zeros", &.{lat});
        try e.triton("_pool_norm", .{ n, 1, 1 }, &.{
            .{ .name = "BUF", .arg = buf },                                                   .{ .name = "b_stride", .arg = .{ .i = ratio * hd } },
            .{ .name = "W", .arg = try e.weight("L{d}.attn.comp_norm", .{L}, .f32, &.{hd}) }, .{ .name = "LAT", .arg = lat },
            .{ .name = "POS", .arg = try pos(e) },                                            .{ .name = "n", .arg = .{ .i = n } },
            .{ .name = "EPS", .arg = .{ .f = E.dec(e.cfg.eps) } },                            .{ .name = "RATIO", .arg = .{ .i = ratio } },
            .{ .name = "SL", .arg = .none },                                                  .{ .name = "PREV", .arg = .none },
            .{ .name = "OFF", .arg = .{ .i = 0 } },                                           .{ .name = "ROWS", .arg = .{ .b = false } },
            .{ .name = "PDL", .arg = .{ .b = false } },                                       .{ .name = "CARRY", .arg = .none },
            .{ .name = "c_stride", .arg = .{ .i = 0 } },                                      .{ .name = "SPLIT", .arg = .{ .b = false } },
        });
        try s.kvStore(lat, try pool(e, L), ratio);
        const kp = try e.buf("L.ix_k", .{}, .bf16, &.{ n, ID });
        try s.proj(try e.fmt("L{d}.attn.ix_wk", .{L}), .strips, lat, hd, ID, kp);
        const kpg = try e.paging(ratio, false);
        try e.triton("_index_k", .{ n, 1, 1 }, &.{
            .{ .name = "KP", .arg = kp },                                                    .{ .name = "k_stride", .arg = .{ .i = ID } },
            .{ .name = "W", .arg = try e.weight("L{d}.attn.ix_knorm", .{L}, .f32, &.{ID}) }, .{ .name = "CS", .arg = try e.ropeTable() },
            .{ .name = "cs_stride", .arg = .{ .i = 64 } },                                   .{ .name = "IK", .arg = try indexKeys(e, L) },
            .{ .name = "POS", .arg = try pos(e) },                                           .{ .name = "EPS", .arg = .{ .f = E.dec(e.cfg.eps) } },
            .{ .name = "RATIO", .arg = .{ .i = ratio } },                                    .{ .name = "PT", .arg = kpg.pt },
            .{ .name = "PSH", .arg = .{ .i = kpg.psh } },                                    .{ .name = "KFP8", .arg = .{ .b = true } },
            .{ .name = "SL", .arg = .none },                                                 .{ .name = "PTS", .arg = .{ .i = 0 } },
            .{ .name = "ROWS", .arg = .{ .b = false } },                                     .{ .name = "PDL", .arg = .{ .b = false } },
        });
    }

    /// The segment's selection of index layer `src` (reuse layers read the latest source's).
    fn selection(e: *E, src: u32) ![2]Arg {
        const k: i64 = e.cfg.index_topk;
        return .{
            try e.view(.{ .buf = try e.fmt("w.sel{d}", .{src}) }, .i32, &.{ e.n, k }, &.{ k, 1 }, e.row0 * k * 4),
            // the counts' 4-byte rows: a run's segment from a row a multiple of 4 (cnt0), so the slice is 16-aligned
            try e.view(.{ .buf = try e.fmt("w.cnt{d}", .{src}) }, .i32, &.{e.n}, &.{1}, e.cnt0 * 4),
        };
    }

    /// The segment's rows of the run's per-row candidates (layer 20's, read by the reindex layers).
    fn cands(e: *E) !Arg {
        const cb: i64 = e.cfg.candidate_blocks;
        return e.view(.{ .buf = "w.cand" }, .i32, &.{ e.n, cb }, &.{ cb, 1 }, e.row0 * cb * 4);
    }

    /// Rows [a, a + rows) of a row-major [n, cols] role.
    fn rowsOf(e: *E, x: Arg, dt: Dt, a: i64, rows: i64, cols: i64) !Arg {
        return e.view(x.t.role, dt, &.{ rows, cols }, &.{ cols, 1 }, x.t.offset + a * cols * size(dt));
    }

    /// index.scores over rows [a, a + rows) of the segment: dense (key i = position i) or over `keys`' positions
    /// (`cbs` > 0: `keys` are candidate blocks of `cbs` positions, a short segment's Reindex layer).
    fn scores(s: *Seg, keys_of: u32, ratio: i64, a: i64, rows: i64, p: Arg, nk: i64, keys: ?Arg, cbs: i64) !Arg {
        const e = s.e;
        const IH: i64 = e.cfg.index_heads;
        const ID: i64 = e.cfg.index_dim;
        const out = try e.buf("L.ix.scores", .{}, .f32, &.{ rows, nk });
        const spg = try e.paging(ratio, false);
        try e.triton("_scores", .{ rows, cdiv(nk, 64), 1 }, &.{
            .{ .name = "QI", .arg = try e.view(.{ .buf = "L.qi" }, .bf16, &.{ rows, IH, ID }, &.{ IH * ID, ID, 1 }, (e.row0 + a) * IH * ID * 2) },
            .{ .name = "W", .arg = try e.view(.{ .buf = "L.ix_w" }, .f32, &.{ rows, IH }, &.{ IH, 1 }, (e.row0 + a) * IH * 4) },
            .{ .name = "w_stride", .arg = .{ .i = IH } },               .{ .name = "IK", .arg = try indexKeys(e, keys_of) },
            .{ .name = "OUT", .arg = out },                             .{ .name = "POS", .arg = p },
            .{ .name = "KEYS", .arg = keys orelse out },                .{ .name = "k_stride", .arg = .{ .i = if (cbs > 0) keys.?.t.shape[1] else nk } },
            .{ .name = "NK", .arg = .{ .i = nk } },                     .{ .name = "o_stride", .arg = .{ .i = nk } },
            .{ .name = "RATIO", .arg = .{ .i = ratio } },               .{ .name = "H", .arg = .{ .i = IH } },
            .{ .name = "D", .arg = .{ .i = ID } },                      .{ .name = "BP", .arg = .{ .i = 64 } },
            .{ .name = "WS", .arg = .{ .f = 1.0 / @sqrt(@as(f64, @floatFromInt(IH))) } },
            .{ .name = "SCALE", .arg = .{ .f = 1.0 / @sqrt(@as(f64, @floatFromInt(ID))) } },
            .{ .name = "GATHER", .arg = .{ .b = keys != null } },       .{ .name = "PT", .arg = spg.pt },
            .{ .name = "PSH", .arg = .{ .i = spg.psh } },               .{ .name = "KFP8", .arg = .{ .b = true } },
            .{ .name = "SL", .arg = .none },                            .{ .name = "PTS", .arg = .{ .i = 0 } },
            .{ .name = "ROWS", .arg = .{ .b = false } },                .{ .name = "CBS", .arg = .{ .i = cbs } },
        });
        return out;
    }

    /// index.sort_keys: a unique int64 key a score (`posn`: the gathered keys' positions).
    fn sortKeys(s: *Seg, sc: Arg, rows: i64, nk: i64, posn: ?Arg) !Arg {
        const e = s.e;
        const k = try e.buf("L.ix.keys", .{}, .i64, &.{ rows, nk });
        try e.triton("_keys", .{ rows, cdiv(nk, 1024), 1 }, &.{
            .{ .name = "S", .arg = sc },                    .{ .name = "s_stride", .arg = .{ .i = nk } },
            .{ .name = "K", .arg = k },                     .{ .name = "k_stride", .arg = .{ .i = nk } },
            .{ .name = "POSN", .arg = posn orelse sc },     .{ .name = "p_stride", .arg = .{ .i = nk } },
            .{ .name = "NK", .arg = .{ .i = nk } },         .{ .name = "BLOCK", .arg = .{ .i = 1024 } },
            .{ .name = "GATHER", .arg = .{ .b = posn != null } },
        });
        return k;
    }

    /// index.candidate_blocks: the blocks' keys from the scores, then torch's top-k (glue) into rows [a, ..) of the
    /// candidates.
    fn candidates(s: *Seg, sc: Arg, a: i64, rows: i64, nk: i64, p: Arg, ratio: i64) !void {
        const e = s.e;
        const cbs: i64 = e.cfg.candidate_block_size;
        const cb: i64 = e.cfg.candidate_blocks;
        const nb = cdiv(nk, cbs);
        const k = try e.buf("L.ix.bkeys", .{}, .i64, &.{ rows, nb });
        try e.triton("_block_keys", .{ rows, cdiv(nb, 256), 1 }, &.{
            .{ .name = "S", .arg = sc },                  .{ .name = "s_stride", .arg = .{ .i = nk } },
            .{ .name = "K", .arg = k },                   .{ .name = "k_stride", .arg = .{ .i = nb } },
            .{ .name = "POS", .arg = p },                 .{ .name = "NB", .arg = .{ .i = nb } },
            .{ .name = "RATIO", .arg = .{ .i = ratio } }, .{ .name = "BS", .arg = .{ .i = cbs } },
            .{ .name = "TB", .arg = .{ .i = 256 } },      .{ .name = "ROWS", .arg = .{ .b = false } },
        });
        try e.glue("topk", &.{ k, try rowsOf(e, try cands(e), .i32, a, rows, cb), .{ .i = cb } });
    }

    /// Row blocks of the materialised indexer (backend._row_blocks): `per_row` bytes a row under the index budget.
    fn step(e: *E, per_row: i64) i64 {
        return if (e.n > 16) @max(16, @divFloor(e.o.index_budget, @max(1, per_row))) else e.n;
    }

    /// An index layer's selection (blocks._select over backend.select / select_cand / reindex).
    fn select(s: *Seg, L: u32) !void {
        const e = s.e;
        const n = e.n;
        const md = e.cfg.mode(L);
        const src = e.cfg.kvSource(L).?;
        const ratio: i64 = e.cfg.compressRatio(L);
        const topk: i64 = e.cfg.index_topk;
        const cbs: i64 = e.cfg.candidate_block_size;
        const cb: i64 = e.cfg.candidate_blocks;
        const sel = try selection(e, L);
        // backend._dt: a segment of at most 64 rows whose materialised scores fit the index budget selects with
        // attn_cuda's top-k (the decode window's job list), not `_keys` + torch's top-k (pod 25: Python's 45-row
        // prompt launches no `_keys`); past the stream threshold the stream top-k (below)
        const reindexing = md == .reindex and e.cfg.usesCandidates(L) and s.cand;
        const nvis = @max(@divFloor(e.end(), ratio), 1);
        const streamed = !e.cfg.isCandidateSource(L) and !reindexing and n > 16 and nvis >= stream_min and @popCount(topk) == 1;
        const fits = if (reindexing) n * cb * cbs * 16 <= e.o.index_budget else n * nvis * 12 <= e.o.index_budget;
        if (n <= dtopk_rows and fits and !streamed) {
            // the whole segment's scores, then attn_cuda's top-k (its job list picks a source's candidate blocks; a
            // Reindex layer scores over them) and visible counts
            const cand = try cands(e);
            if (reindexing) {
                const sc = try s.scores(src, ratio, 0, n, try pos(e), cb * cbs, cand, cbs);
                return e.topk(L, sc, cand, cb * cbs, ratio);
            }
            const sc = try s.scores(src, ratio, 0, n, try pos(e), nvis, null, 0);
            try e.topk(L, sc, cand, nvis, ratio);
            if (e.cfg.isCandidateSource(L)) s.cand = true;
            return;
        }
        if (md == .reindex and e.cfg.usesCandidates(L) and s.cand) {
            // backend.reindex: the row's own scores over its candidates' positions, in row blocks under the budget
            const NK = cb * cbs;
            const whole = n * NK * 16 <= e.o.index_budget;
            const st = if (whole) n else step(e, NK * 16);
            var a: i64 = 0;
            var j: usize = 0;
            while (a < n) : ({
                a += st;
                j += 1;
            }) {
                const rows = @min(st, n - a);
                const p = if (whole) try pos(e) else try s.blockPos(j, a);
                const ck = try e.buf("L.ix.ckeys", .{}, .i32, &.{ rows, NK });
                try e.glue("cand_keys", &.{ try rowsOf(e, try cands(e), .i32, a, rows, cb), ck, .{ .i = cbs } });
                const sc = try s.scores(src, ratio, a, rows, p, NK, ck, 0);
                try e.glue("topk", &.{ try s.sortKeys(sc, rows, NK, ck), try rowsOf(e, sel[0], .i32, a, rows, topk), .{ .i = topk } });
            }
        } else {
            const cand = e.cfg.isCandidateSource(L);
            // backend.select's stream path (full-mode layers without the candidates' scores): stream_topk.select
            if (streamed) {
                try s.streamSelect(src, ratio, topk, sel[0]);
                try e.glue("counts", &.{ sel[0], sel[1], try e.buf("w.pos64", .{}, .i64, &.{n}), .{ .i = ratio } });
                return;
            }
            const whole = n * nvis * 12 <= e.o.index_budget;
            const st = if (whole) n else step(e, nvis * 12);
            var a: i64 = 0;
            var j: usize = 0;
            while (a < n) : ({
                a += st;
                j += 1;
            }) {
                const rows = @min(st, n - a);
                const p = if (whole) try pos(e) else try s.blockPos(j, a);
                const nk = if (whole) nvis else @max(@divFloor(e.start + a + rows, ratio), 1);
                const sc = try s.scores(src, ratio, a, rows, p, nk, null, 0);
                try e.glue("topk", &.{ try s.sortKeys(sc, rows, nk, null), try rowsOf(e, sel[0], .i32, a, rows, topk), .{ .i = topk } });
                if (cand) try s.candidates(sc, a, rows, nk, p, ratio);
            }
            if (cand) s.cand = true;
        }
        try e.glue("counts", &.{ sel[0], sel[1], try e.buf("w.pos64", .{}, .i64, &.{n}), .{ .i = ratio } });
    }

    /// backend._stream_select: stream_topk.select in row blocks whose scratch fits stream_topk.BUDGET; each block's
    /// `_stream` (a program a row and key split: the split's best `topk` keys in its buffer's first `topk` slots), then
    /// the merge (glue: the splits' first `topk` keys a row gathered, index.top_positions) into the selection's rows.
    fn streamSelect(s: *Seg, src: u32, ratio: i64, topk: i64, out: Arg) !void {
        const e = s.e;
        const n = e.n;
        const IH: i64 = e.cfg.index_heads;
        const ID: i64 = e.cfg.index_dim;
        const st = longpf.plan(n, @max(@divFloor(e.end(), ratio), 1), topk);
        const spg = try e.paging(ratio, false);
        var a: i64 = 0;
        var j: usize = 0;
        while (a < n) : ({
            a += st;
            j += 1;
        }) {
            const rows = @min(st, n - a);
            const p = if (st >= n) try pos(e) else try s.blockPos(j, a);
            const nk = @max(@divFloor(e.start + a + rows, ratio), 1);
            const ns = longpf.splits(nk);
            const buf = try e.buf("L.ix.stream", .{}, .i64, &.{ rows, ns, 2 * topk });
            if (e.o.stream_rb == 1 or e.o.stream_rb == 2 or e.o.stream_rb == 4) {
                // TF_DSV41_STREAM_RB: `_stream_pf` (1) or `_stream_rb<N>`, N rows a program (the same buffers, the
                // same merge)
                const rb = e.o.stream_rb;
                try e.triton(switch (rb) {
                    1 => "_stream_pf",
                    2 => "_stream_rb2",
                    else => "_stream_rb4",
                }, .{ cdiv(rows, rb), ns, 1 }, &.{
                    .{ .name = "QI", .arg = try e.view(.{ .buf = "L.qi" }, .bf16, &.{ rows, IH, ID }, &.{ IH * ID, ID, 1 }, (e.row0 + a) * IH * ID * 2) },
                    .{ .name = "W", .arg = try e.view(.{ .buf = "L.ix_w" }, .f32, &.{ rows, IH }, &.{ IH, 1 }, (e.row0 + a) * IH * 4) },
                    .{ .name = "w_stride", .arg = .{ .i = IH } },           .{ .name = "IK", .arg = try indexKeys(e, src) },
                    .{ .name = "POS", .arg = p },                           .{ .name = "R", .arg = .{ .i = rows } },
                    .{ .name = "BUF", .arg = buf },                         .{ .name = "nsplit", .arg = .{ .i = ns } },
                    .{ .name = "RATIO", .arg = .{ .i = ratio } },           .{ .name = "H", .arg = .{ .i = IH } },
                    .{ .name = "D", .arg = .{ .i = ID } },                  .{ .name = "BP", .arg = .{ .i = 64 } },
                    .{ .name = "WS", .arg = .{ .f = 1.0 / @sqrt(@as(f64, @floatFromInt(IH))) } },
                    .{ .name = "SCALE", .arg = .{ .f = 1.0 / @sqrt(@as(f64, @floatFromInt(ID))) } },
                    .{ .name = "SPLIT", .arg = .{ .i = longpf.split_keys } }, .{ .name = "K", .arg = .{ .i = topk } },
                    .{ .name = "CAP", .arg = .{ .i = 2 * topk } },          .{ .name = "PT", .arg = spg.pt },
                    .{ .name = "PSH", .arg = .{ .i = spg.psh } },           .{ .name = "KFP8", .arg = .{ .b = true } },
                });
                try e.glue("stream_merge", &.{ buf, try rowsOf(e, out, .i32, a, rows, topk), .{ .i = topk }, try e.buf("L.ix.keys", .{}, .i64, &.{ rows, ns * topk }) });
                continue;
            }
            try e.triton("_stream", .{ rows, ns, 1 }, &.{
                .{ .name = "QI", .arg = try e.view(.{ .buf = "L.qi" }, .bf16, &.{ rows, IH, ID }, &.{ IH * ID, ID, 1 }, (e.row0 + a) * IH * ID * 2) },
                .{ .name = "W", .arg = try e.view(.{ .buf = "L.ix_w" }, .f32, &.{ rows, IH }, &.{ IH, 1 }, (e.row0 + a) * IH * 4) },
                .{ .name = "w_stride", .arg = .{ .i = IH } },           .{ .name = "IK", .arg = try indexKeys(e, src) },
                .{ .name = "POS", .arg = p },                           .{ .name = "KEYS", .arg = buf },
                .{ .name = "k_stride", .arg = .{ .i = 0 } },            .{ .name = "NK", .arg = .{ .i = nk } },
                .{ .name = "BUF", .arg = buf },                         .{ .name = "nsplit", .arg = .{ .i = ns } },
                .{ .name = "RATIO", .arg = .{ .i = ratio } },           .{ .name = "H", .arg = .{ .i = IH } },
                .{ .name = "D", .arg = .{ .i = ID } },                  .{ .name = "BP", .arg = .{ .i = 64 } },
                .{ .name = "WS", .arg = .{ .f = 1.0 / @sqrt(@as(f64, @floatFromInt(IH))) } },
                .{ .name = "SCALE", .arg = .{ .f = 1.0 / @sqrt(@as(f64, @floatFromInt(ID))) } },
                .{ .name = "MODE", .arg = .{ .i = 0 } },                .{ .name = "SPLIT", .arg = .{ .i = longpf.split_keys } },
                .{ .name = "K", .arg = .{ .i = topk } },                .{ .name = "CAP", .arg = .{ .i = 2 * topk } },
                .{ .name = "BS", .arg = .{ .i = 8 } }, // index.BLOCK
                .{ .name = "PT", .arg = spg.pt },
                .{ .name = "PSH", .arg = .{ .i = spg.psh } },           .{ .name = "KFP8", .arg = .{ .b = true } },
            });
            // the merge's gathered keys [rows, ns x topk] share the materialised path's sort keys' storage
            try e.glue("stream_merge", &.{ buf, try rowsOf(e, out, .i32, a, rows, topk), .{ .i = topk }, try e.buf("L.ix.keys", .{}, .i64, &.{ rows, ns * topk }) });
        }
    }

    /// Win.sub's first position (int32 [1]) of the row block starting at row `a`.
    fn blockPos(s: *Seg, j: usize, a: i64) !Arg {
        const e = s.e;
        const p = try e.buf("L.ix.pos{d}", .{j}, .i32, &.{1});
        try e.glue("block_pos", &.{ p, .{ .i = e.start + a } });
        return p;
    }

    /// blocks._projection: a kv source's compressor projection of "w.out" as fp32 rows (ratio 2: [kv | gate], kept for
    /// the commit's carry).
    fn compProj(s: *Seg, L: u32) !Arg {
        const e = s.e;
        const n = e.n;
        const D: i64 = e.cfg.hidden;
        const hd: i64 = e.cfg.head_dim;
        const x = try e.buf("w.out", .{}, .bf16, &.{ n, D });
        const cw = try e.buf("L.comp", .{}, .bf16, &.{ n, hd });
        try s.proj(try e.fmt("L{d}.attn.comp_wkv", .{L}), .lanes, x, D, hd, cw);
        if (e.cfg.compressRatio(L) == 2) {
            const cg = try e.buf("L.comp_gate", .{}, .bf16, &.{ n, hd });
            try s.proj(try e.fmt("L{d}.attn.comp_wgate", .{L}), .lanes, x, D, hd, cg);
            const p = try e.buf("w.L{d}.proj", .{L}, .f32, &.{ n, 2 * hd });
            try e.glue("proj", &.{ cw, cg, p });
            return p;
        }
        const p = try e.buf("L.proj", .{}, .f32, &.{ n, hd });
        try e.glue("proj", &.{ cw, p });
        return p;
    }

    /// CED's encoder pass at the decoder's first layer, after its attention site (blocks.compress_rows): the kv
    /// source's compressor alone (its compressed rows and index keys for every row), then the stash: the rows entering
    /// the layer (the streams, the normed input, the site's coefficients) into the slot's ring (replay.encode).
    fn encoderEnd(s: *Seg, L: u32) !void {
        const e = s.e;
        if (e.cfg.mode(L) != .full) return error.Unsupported;
        const p = try s.compProj(L);
        const segs = s.multi orelse {
            try s.compress(L, p);
            return e.glue("ced_stash", &try stashArgs(e));
        };
        // blocks.compress_rows: the projection over every row, then each segment's compressor and stash (replay.encode)
        const total = e.n;
        for (segs, 0..) |sp, j| {
            try s.enter(sp, j);
            try s.compress(L, try rowsOf(e, p, .f32, e.row0, e.n, p.t.stride[0]));
            try e.glue("ced_stash", &try stashArgs(e));
        }
        try s.leave(total);
    }

    fn attention(s: *Seg, L: u32) !void {
        const e = s.e;
        const n = e.n;
        const D: i64 = e.cfg.hidden;
        const md = e.cfg.mode(L);
        const H: i64 = e.cfg.heads / e.o.world;
        const hd: i64 = e.cfg.head_dim;
        const IH: i64 = e.cfg.index_heads;
        const ID: i64 = e.cfg.index_dim;
        const x = try e.buf("w.out", .{}, .bf16, &.{ n, D });
        const indexer = md == .full or md == .reindex;
        // the decoder replay skips the kv source's compressor (blocks.Context.comp_done: the encoder pass stored it)
        const own_kv = md == .full and s.part != .decoder;
        // the x projections (wq_a, wkv, a KV source's compressor), each its own matmul
        const qa = try e.buf("L.qa", .{}, .bf16, &.{ n, e.cfg.q_lora });
        const kva = try e.buf("L.kva", .{}, .bf16, &.{ n, hd });
        try s.proj(try e.fmt("L{d}.attn.wq_a", .{L}), .lanes, x, D, e.cfg.q_lora, qa);
        try s.proj(try e.fmt("L{d}.attn.wkv", .{L}), .lanes, x, D, hd, kva);
        const cproj: Arg = if (own_kv) try s.compProj(L) else undefined;
        const qn = try e.buf("L.qn", .{}, .bf16, &.{ n, e.cfg.q_lora });
        const kv = try e.buf("L.kv", .{}, .bf16, &.{ n, hd });
        try e.triton("_rms2", .{ n, 2, 1 }, &.{
            .{ .name = "XA", .arg = qa },                                                      .{ .name = "xa_rs", .arg = .{ .i = e.cfg.q_lora } },
            .{ .name = "WA", .arg = try e.weight("L{d}.attn.q_norm", .{L}, .f32, &.{e.cfg.q_lora}) }, .{ .name = "OA", .arg = qn },
            .{ .name = "oa_rs", .arg = .{ .i = e.cfg.q_lora } },                              .{ .name = "KA", .arg = .{ .i = e.cfg.q_lora } },
            .{ .name = "inv_ka", .arg = .{ .f = 1.0 / @as(f64, @floatFromInt(e.cfg.q_lora)) } }, .{ .name = "XB", .arg = kva },
            .{ .name = "xb_rs", .arg = .{ .i = hd } },                                        .{ .name = "WB", .arg = try e.weight("L{d}.attn.kv_norm", .{L}, .f32, &.{hd}) },
            .{ .name = "OB", .arg = kv },                                                     .{ .name = "ob_rs", .arg = .{ .i = hd } },
            .{ .name = "KB", .arg = .{ .i = hd } },                                           .{ .name = "inv_kb", .arg = .{ .f = 1.0 / @as(f64, @floatFromInt(hd)) } },
            .{ .name = "eps", .arg = .{ .f = E.dec(e.cfg.eps) } },                            .{ .name = "BK", .arg = .{ .i = 512 } },
            .{ .name = "NARROW", .arg = .{ .b = true } },                                     .{ .name = "PDL", .arg = .{ .b = false } },
        });
        const q = try e.buf("L.q", .{}, .bf16, &.{ n, H * hd });
        try s.proj(try e.fmt("L{d}.attn.wq_b", .{L}), .lanes, qn, e.cfg.q_lora, H * hd, q);
        const qi = try e.buf("L.qi", .{}, .bf16, &.{ n, IH * ID });
        if (indexer) try s.proj(try e.fmt("L{d}.attn.ix_wq_b", .{L}), .lanes, qn, e.cfg.q_lora, IH * ID, qi);
        if (s.short) try e.prefetchO(L);
        const positions = if (s.multi != null) "w.mpos64" else "w.pos64";
        try ropeAt(e, q, H, hd, positions);
        if (indexer) {
            try ropeAt(e, qi, IH, ID, positions);
            // the head weights: `_plain`'s fp64 sums, cast to bf16 then fp32 by torch
            const w64 = try e.buf("L.ix_w64", .{}, .f64, &.{ n, IH });
            try e.triton("_plain", .{ n, IH, 1 }, &.{
                .{ .name = "X", .arg = x },                                                      .{ .name = "x_rs", .arg = .{ .i = D } },
                .{ .name = "W", .arg = try e.weight("L{d}.attn.ix_wp", .{L}, .f32, &.{ IH, D }) }, .{ .name = "w_rs", .arg = .{ .i = D } },
                .{ .name = "OUT", .arg = w64 },                                                  .{ .name = "K", .arg = .{ .i = D } },
                .{ .name = "N", .arg = .{ .i = IH } },                                           .{ .name = "BK", .arg = .{ .i = 256 } },
                .{ .name = "PDL", .arg = .{ .b = false } },                                      .{ .name = "BF16", .arg = .{ .b = false } },
            });
            try e.glue("ix_w", &.{ w64, try e.buf("L.ix_w", .{}, .f32, &.{ n, IH }) });
        }
        // the CSA2 steps a segment at a time (blocks.attention's loop over its contexts), into the group-major core output
        const groups: i64 = e.cfg.o_groups / e.o.world;
        if (@mod(H, bmq) != 0 or @mod(H, groups) != 0) return error.Unsupported;
        const hg = @divExact(H, groups);
        const o = try e.buf("L.o", .{}, .bf16, &.{ groups, n, hg * hd });
        if (s.multi) |segs| {
            for (segs, 0..) |sp, j| {
                try s.enter(sp, j);
                try s.core(L, q, kv, cproj, own_kv, indexer, o, n);
            }
            try s.leave(n);
        } else try s.core(L, q, kv, cproj, own_kv, indexer, o, n);
        // _out_pf: each wo_a group's GEMM over its contiguous slice into its columns of z, then wo_b (bf16 partial)
        const ol: i64 = e.cfg.o_lora;
        const gk = hg * hd;
        const z: Out = .{ .role = .{ .buf = "L.z" }, .dt = .bf16, .stride = groups * ol };
        const k = s.pieces();
        if (k == 1) {
            var g: i64 = 0;
            while (g < groups) : (g += 1) {
                var zg = z;
                zg.offset = g * ol * 2;
                try s.matmul(try e.fmt("L{d}.attn.wo_a.{d}", .{ L, g }), .lanes, try e.view(o.t.role, .bf16, &.{ n, gk }, &.{ gk, 1 }, g * n * gk * 2), gk, ol, zg);
            }
            return s.proj(try e.fmt("L{d}.attn.wo_b", .{L}), .lanes, try e.buf("L.z", .{}, .bf16, &.{ n, groups * ol }), groups * ol, D, try e.buf("L.part", .{}, .bf16, &.{ n, D }));
        }
        // TF_DSV41_PF_OVERLAP: wo_a's groups and wo_b a row piece at a time (each GEMM row-independent, the same bits at
        // any M), each piece's partial sent while the next piece's GEMMs run
        const part = try e.buf("L.part", .{}, .bf16, &.{ n, D });
        const zall = try e.buf("L.z", .{}, .bf16, &.{ n, groups * ol });
        const gath = try s.recv("attn");
        defer e.n = n;
        var p: i64 = 0;
        while (p < k) : (p += 1) {
            const r = pieceAt(n, k, p);
            const m = r[1] - r[0];
            e.n = m;
            var g: i64 = 0;
            while (g < groups) : (g += 1) {
                var zg = z;
                zg.offset = g * ol * 2 + r[0] * groups * ol * 2;
                try s.matmul(try e.fmt("L{d}.attn.wo_a.{d}", .{ L, g }), .lanes, try e.view(o.t.role, .bf16, &.{ m, gk }, &.{ gk, 1 }, (g * n + r[0]) * gk * 2), gk, ol, zg);
            }
            try s.matmul(try e.fmt("L{d}.attn.wo_b", .{L}), .lanes, try rowsOf(e, zall, .bf16, r[0], m, groups * ol), groups * ol, D, .{ .role = part.t.role, .dt = .bf16, .stride = D, .offset = r[0] * D * 2 });
            e.n = n;
            try s.sendPiece(part, gath, r[0], m, p, k);
        }
        s.sent_attn = true;
    }

    /// The CSA2 steps of the current segment (blocks._segment): its SWA rows stored (through the staging ring when its
    /// slot's ring cannot hold them), a kv source's compressor, an index layer's selection, the fused core into rows
    /// [row0, row0 + n) of the run's group-major output `o` (`total` rows a group). `q`, `kv`, `cproj`: the run's rows.
    fn core(s: *Seg, L: u32, q: Arg, kv: Arg, cproj: Arg, own_kv: bool, indexer: bool, o: Arg, total: i64) !void {
        const e = s.e;
        const n = e.n;
        const ratio: i64 = e.cfg.compressRatio(L);
        const H: i64 = e.cfg.heads / e.o.world;
        const hd: i64 = e.cfg.head_dim;
        const groups: i64 = e.cfg.o_groups / e.o.world;
        const hg = @divExact(H, groups);
        // blocks._segment: a segment the slot's ring cannot hold runs on the staging ring, the window before it in
        const rows = try s.swa(L);
        const slot = try slotRing(e, L);
        const W: i64 = e.cfg.window;
        const ring0 = 2 * W;
        if (s.staged) try e.glue("stage_in", &.{ rows[0], rows[1], .{ .i = s.ring }, slot[0], slot[1], .{ .i = ring0 }, .{ .i = @max(0, e.start - W + 1) }, .{ .i = e.start } });
        try s.kvStore(try rowsOf(e, kv, .bf16, e.row0, n, hd), rows, 0);
        if (own_kv) try s.compress(L, try rowsOf(e, cproj, .f32, e.row0, n, cproj.t.stride[0]));
        if (indexer) try s.select(L);
        // the fused prefill core over the SWA ring and (a compressed layer) the source's pool at the latest selection
        const lo = try e.buf("w.lo", .{}, .i32, &.{n});
        var cv = rows[0];
        var cs = rows[1];
        var tok = lo;
        var cnt = lo;
        var t_stride: i64 = 0;
        var fpg: E.Paging = .{ .pt = .none, .psh = 0, .vs = 0, .ss = 0 };
        if (ratio > 0) {
            const src = e.cfg.kvSource(L).?;
            const sel = try selection(e, e.cfg.indexSource(L).?);
            cnt = sel[1];
            t_stride = e.cfg.index_topk;
            if (e.o.pool != null and e.o.pool.?.split) {
                // split: an index layer's selection brings its rows over the exchange; its reuse layers read them too
                const got = if (indexer) try e.exchange(src, e.cfg.compressRatio(src), sel[0]) else try e.received(src, e.cfg.compressRatio(src));
                cv = got.cv;
                cs = got.cs;
                tok = got.tok;
            } else {
                const p = try pool(e, src);
                cv = p[0];
                cs = p[1];
                tok = sel[0];
                fpg = try e.paging(e.cfg.compressRatio(src), false);
            }
        }
        try e.triton("_fused", .{ n, @divExact(H, bmq), 1 }, &.{
            .{ .name = "Q", .arg = try e.view(q.t.role, .bf16, &.{ n, H, hd }, &.{ H * hd, hd, 1 }, e.row0 * H * hd * 2) },
            .{ .name = "CV", .arg = cv },                         .{ .name = "CSC", .arg = cs },
            .{ .name = "cvs", .arg = .{ .i = cv.t.stride[0] } },  .{ .name = "css", .arg = .{ .i = cs.t.stride[0] } },
            .{ .name = "TOK", .arg = tok },                       .{ .name = "t_stride", .arg = .{ .i = t_stride } },
            .{ .name = "CNT", .arg = cnt },                       .{ .name = "SV", .arg = rows[0] },
            .{ .name = "SSC", .arg = rows[1] },                   .{ .name = "svs", .arg = .{ .i = 576 } },
            .{ .name = "sss", .arg = .{ .i = 8 } },               .{ .name = "LO", .arg = lo },
            .{ .name = "POS", .arg = try pos(e) },                .{ .name = "SINK", .arg = try e.weight("L{d}.attn.sink", .{L}, .f32, &.{H}) },
            .{ .name = "CS", .arg = try e.ropeTable() },          .{ .name = "cs_stride", .arg = .{ .i = 64 } },
            .{ .name = "OUT", .arg = try e.view(o.t.role, .bf16, &.{ groups, n, hg * hd }, &.{ total * hg * hd, hg * hd, 1 }, e.row0 * hg * hd * 2) },
            .{ .name = "H", .arg = .{ .i = H } },
            .{ .name = "KT", .arg = .{ .i = kt } },               .{ .name = "BMQ", .arg = .{ .i = bmq } },
            // Python's D ** -0.5 (libm pow, correctly rounded): sqrt of the exact 1 / 512 rounds the same way
            .{ .name = "SCALE", .arg = .{ .f = @sqrt(1.0 / @as(f64, @floatFromInt(hd))) } }, .{ .name = "RING", .arg = .{ .i = s.ring } },
            .{ .name = "WINDOW", .arg = .{ .i = W } },            .{ .name = "COMP", .arg = .{ .b = ratio > 0 } },
            .{ .name = "PT", .arg = fpg.pt },                     .{ .name = "PSH", .arg = .{ .i = fpg.psh } },
            .{ .name = "HG", .arg = .{ .i = hg } },               .{ .name = "OR", .arg = .{ .i = total } },
        });
        if (s.staged) try e.glue("stage_out", &.{ slot[0], slot[1], .{ .i = ring0 }, rows[0], rows[1], .{ .i = s.ring }, .{ .i = @max(0, e.end() - W) }, .{ .i = e.end() } });
    }

    /// A multi-segment run's segment `j` (`sp`, its first row in the run): the emitter at its rows and position, the
    /// glue on its slot (pf_seg: the slot's views, the segment's positions and window starts).
    fn enter(s: *Seg, sp: Span, j: usize) !void {
        const e = s.e;
        e.row0 = sp.row0;
        e.cnt0 = sp.cnt0;
        e.n = sp.n;
        e.start = sp.start;
        s.setRing();
        // the decoder replay: every row's window from the segment's first row (replay.finish's ctx.lo)
        try e.glue("pf_seg", &.{ .{ .i = @intCast(j) }, try pos(e), try e.buf("w.pos64", .{}, .i64, &.{sp.n}), try e.buf("w.lo", .{}, .i32, &.{sp.n}), .{ .b = s.part == .decoder } });
    }

    /// Back at the whole run (`total` rows from its first segment's start): the glue's state the run's again.
    fn leave(s: *Seg, total: i64) !void {
        const e = s.e;
        const first = s.multi.?[0];
        e.row0 = 0;
        e.cnt0 = 0;
        e.n = total;
        e.start = first.start;
        s.setRing();
        try e.glue("pf_seg", &.{.{ .i = -1 }});
    }

    /// forward.staging for `e.n` rows: the slot's ring (2 x window rows), or the staging ring past it.
    fn setRing(s: *Seg) void {
        const e = s.e;
        const W: i64 = e.cfg.window;
        s.staged = e.n + W - 1 > 2 * W;
        s.ring = if (s.staged) stageRows(e) else 2 * W;
    }

    // -----------------------------------------------------------------------------------------------------------
    // MoE (prefill_moe.forward: router, x3gm's ragged launches, the shared expert's GEMMs)

    fn moeBlock(s: *Seg, L: u32) !void {
        const e = s.e;
        if (e.o.image_rows != 0) return s.moeSplit(L);
        const D: i64 = e.cfg.hidden;
        return s.moeRows(L, e.n, try e.buf("w.out", .{}, .bf16, &.{ e.n, D }), "bias", try e.buf("L.moe16", .{}, .bf16, &.{ e.n, D }), false);
    }

    /// vision.moe (TF_DSV41_BIAS_VL): the segment's image rows through the MoE as their own call, routed with
    /// gate.bias_vl, then the text rows with gate.bias, each a whole prefill_moe.forward at its own row count; the
    /// partials go back to their rows (glue image_split / image_merge: the positions from the segment's ids).
    /// image_rows < 0 (the buffer plan): both calls at the segment's rows, so every role is sized for any split.
    fn moeSplit(s: *Seg, L: u32) !void {
        const e = s.e;
        const D: i64 = e.cfg.hidden;
        const n = e.n;
        const ni: i64 = if (e.o.image_rows < 0) n else e.o.image_rows;
        const nt: i64 = if (e.o.image_rows < 0) n else n - ni;
        const x = try e.buf("w.out", .{}, .bf16, &.{ n, D });
        const xi = try e.buf("w.out.img", .{}, .bf16, &.{ ni, D });
        const xt: Arg = if (nt > 0) try e.buf("w.out.txt", .{}, .bf16, &.{ nt, D }) else try e.empty(.bf16);
        try e.glue("image_split", &.{ x, xi, xt });
        const pi = try e.buf("L.moe16.img", .{}, .bf16, &.{ ni, D });
        try s.moeRows(L, ni, xi, "bias_vl", pi, false);
        var pt = try e.empty(.bf16);
        if (nt > 0) {
            pt = try e.buf("L.moe16.txt", .{}, .bf16, &.{ nt, D });
            try s.moeRows(L, nt, xt, "bias", pt, true);
        }
        try e.glue("image_merge", &.{ pi, pt, try e.buf("L.moe16", .{}, .bf16, &.{ n, D }) });
    }

    /// prefill_moe.forward over `n` rows at `x`: the router with gate.<bias>, x3gm's ragged launches, the shared
    /// expert, bf16(routed + shared) into `part`; `again`: the layer's second call (its MoE ticket, not the next layer's)
    fn moeRows(s: *Seg, L: u32, n: i64, x: Arg, comptime bias: []const u8, part: Arg, again: bool) !void {
        const e = s.e;
        // the call's own rows are the emitter's rows (the projections' M, their scratch): a plain n-row segment's MoE
        const seg_rows = e.n;
        e.n = n;
        defer e.n = seg_rows;
        const D: i64 = e.cfg.hidden;
        const ex = e.cfg.expertsOf(L);
        const topk: i64 = ex.topk;
        const slots: i64 = topk + 1;
        const Et: i64 = ex.count;
        const I: i64 = e.cfg.expert_width / e.o.world;
        const P = n * topk;
        const pick = try e.buf("L.pick", .{}, .i32, &.{ n, slots });
        const wts = try e.buf("L.wts", .{}, .f32, &.{ n, slots });
        // router_gemv.route, a launch a CHUNK of rows (one at 2,048 rows or fewer; a 4K segment's two halves), prefill's
        // config (8 warps, 2 experts a warp); a call of at most 16 rows the narrow kernel (TF_DSV41_RG_NARROW 2, as a
        // decode window's); a launch of at most 16 rows in a longer call config(E, rows)'s (8, 1) (route's use_narrow
        // is the call's)
        const logits = try e.buf("L.logits", .{}, .f32, &.{ n, Et });
        var c0: i64 = 0;
        while (c0 < n) : (c0 += router_chunk) {
            const m = @min(router_chunk, n - c0);
            const whole = m == n;
            const tile: i64 = for ([_]i64{ 1, 2, 4, 8, 16 }) |t| {
                if (t >= @min(m, 16)) break t;
            } else 16;
            const cfg: [2]i64 = if (n <= 16) .{ 2, 0 } else if (m > 16) .{ 8, 2 } else .{ 8, 1 };
            try e.ext("tf_dsv41_router_gemv_v2.route", &.{
                if (whole) x else try rowsOf(e, x, .bf16, c0, m, D),                   try e.weight("L{d}.moe.gate", .{L}, .bf16, &.{ Et, D }),
                try e.weight("L{d}.moe." ++ bias, .{L}, .f32, &.{Et}),                if (whole) pick else try rowsOf(e, pick, .i32, c0, m, slots),
                if (whole) wts else try rowsOf(e, wts, .f32, c0, m, slots),            if (whole) logits else try rowsOf(e, logits, .f32, c0, m, Et),
                try e.buf("s.router.cnt", .{}, .i32, &.{128}),                        .{ .i = topk },
                .{ .i = slots },                                                       .{ .f = E.dec(e.cfg.routed_scale) },
                .{ .i = tile },                                                        .{ .i = cfg[0] },
                .{ .i = cfg[1] },                                                      .{ .b = true },
                .{ .b = false },                                                       try e.empty(.i32),
                try e.empty(.i32),                                                     try e.t(.empty, .i32, &.{ 0, 1 }),
                .{ .i = 0 },                                                           .{ .b = false },
            });
            if (e.o.r1) {
                // the R1 twin's route always takes the prune list (router_gemv.route: list(prune or [])); prefill's is
                // empty (the prune runs after it, prefill_moe's rule)
                const last = &e.out.items[e.out.items.len - 1];
                const more = try e.a.alloc(calls.Named, last.args.len + 1);
                @memcpy(more[0..last.args.len], last.args);
                more[last.args.len] = .{ .arg = .{ .list = &.{} } };
                last.args = more;
            } else if (e.o.r1_sig) try e.appendLast(&.{.{ .list = &.{} }});
        }
        // the kit's fp16 weights, the shared slot masked, the routed slots' contiguous copies (one row: torch's
        // .contiguous() keeps the [1, topk] view of the [1, slots] rows, the same bytes)
        const rs: i64 = if (n == 1) slots else topk;
        const pk = try e.view(.{ .buf = "L.gm.pick" }, .i32, &.{ n, topk }, &.{ rs, 1 }, 0);
        const w6 = try e.view(.{ .buf = "L.gm.wts" }, .f32, &.{ n, topk }, &.{ rs, 1 }, 0);
        try e.glue("gm_picks", &.{ pick, wts, pk, w6 });
        // x3gm._run_ragged: one row block (X / Y in the experts' scratch z, Xd in its xg), rot, a gate/up launch a
        // gate width present, a down launch a down width present, upstream's combine
        const widths = e.w.gm.get(L);
        const rows: i64 = e.o.expert_rows;
        const zb = rows * (topk + 1) * @max(4 * 2 * I, D); // the scratch z's floats (block_moe)
        // x3gm.Buffers: past what the 1,024-row scratch holds, rows halve (Python streams the trellis twice); under 4K
        // the block is the segment (pf4k.gm_rows: moe.arena_scratch holds it; here the views below size s.ex.z / xg in
        // the buffer plan, one rank's 4,096 x 6 pairs: 503 MB of X / Y, 57 MB of Xd)
        if (!e.o.pf4k and (P * D > zb or P * I > rows * (topk + 1) * D)) return error.Unsupported;
        const pre = try e.fmt("L{d}.moe", .{L});
        const xg = try e.view(.{ .buf = "s.ex.z" }, .f16, &.{ P, D }, &.{ D, 1 }, 0);
        const xu = try e.view(.{ .buf = "s.ex.z" }, .f16, &.{ P, D }, &.{ D, 1 }, P * D * 2);
        const xd = try e.view(.{ .buf = "s.ex.xg" }, .f16, &.{ P, I }, &.{ I, 1 }, 0);
        const y = try e.view(.{ .buf = "s.ex.z" }, .f32, &.{ P, D }, &.{ D, 1 }, 0);
        try e.ext("tf_dsv41_x3gm_v1.rot", &.{
            x,  .{ .i = D }, pk, try e.weight("{s}.w1.suh", .{pre}, .f16, &.{ Et, D }), try e.weight("{s}.w3.suh", .{pre}, .f16, &.{ Et, D }),
            xg, xu,          .{ .i = D }, .{ .i = topk }, .{ .i = P }, .{ .i = 2 },
        });
        const ticket = try e.buf("L.gm.ticket", .{}, .i32, &.{2 * gm_widths});
        try e.glue(if (again) "gm_ticket_again" else "gm_ticket", &.{ticket});
        const tmax = @divFloor(P, gm_bm) + Et + 1;
        const lim: f64 = E.dec(e.cfg.swiglu_limit);
        // TF_DSV41_GM_V2 (gm2pf.zig): x3gm._run_v2's gm2_kernel launches in place of the loop below
        if (e.o.gm_v2 != .off) try gm2pf.emit(e, e.o.gm_v2, .{ .L = L, .pre = pre, .P = P, .Et = Et, .D = D, .I = I, .pk = pk, .xg = xg, .xu = xu, .xd = xd, .y = y, .ticket = ticket, .limit = lim, .widths = widths }) else for (0..2) |down| {
            const wd = (widths orelse return error.NoWidth)[down];
            var j: i64 = 0;
            for (0..32) |kk| {
                const k2: i64 = @intCast(kk);
                if (wd >> @intCast(kk) & 1 == 0) continue;
                const tag = if (down == 1) "d" else "g";
                const plan = [5]Arg{
                    try e.buf("L.gm.{s}{d}.order", .{ tag, j }, .i32, &.{P}),
                    try e.buf("L.gm.{s}{d}.pe", .{ tag, j }, .i32, &.{tmax}),
                    try e.buf("L.gm.{s}{d}.poff", .{ tag, j }, .i32, &.{tmax}),
                    try e.buf("L.gm.{s}{d}.pcnt", .{ tag, j }, .i32, &.{tmax}),
                    try e.buf("L.gm.{s}{d}.npass", .{ tag, j }, .i32, &.{1}),
                };
                try e.glue("gm_plan", &.{ pk, plan[0], plan[1], plan[2], plan[3], plan[4], .{ .i = k2 }, .{ .b = down == 1 }, .{ .i = gm_bm }, .{ .i = Et } });
                const t = try e.view(ticket.t.role, .i32, &.{1}, &.{1}, (j + if (down == 1) @as(i64, gm_widths) else 0) * 4);
                if (down == 0) {
                    try e.ext("tf_dsv41_x3gm_v1.gateup", &.{
                        xg,                                                  xu,
                        try e.buf("s.L{d}.gm.tg", .{L}, .i64, &.{Et}),      try e.buf("s.L{d}.gm.tu", .{L}, .i64, &.{Et}),
                        plan[0],                                             plan[1],
                        plan[2],                                             plan[3],
                        plan[4],                                             t,
                        try e.weight("{s}.w1.svh", .{pre}, .f16, &.{ Et, I }), try e.weight("{s}.w3.svh", .{pre}, .f16, &.{ Et, I }),
                        try e.weight("{s}.w2.suh", .{pre}, .f16, &.{ Et, I }), xd,
                        .{ .i = D },                                         .{ .i = I },
                        .{ .i = k2 },                                        .{ .b = false },
                        .{ .i = 1 },                                         .{ .b = true },
                        .{ .f = lim },                                       .{ .b = true },
                    });
                } else {
                    try e.ext("tf_dsv41_x3gm_v1.down", &.{
                        xd,          try e.buf("s.L{d}.gm.td", .{L}, .i64, &.{Et}), plan[0],     plan[1],
                        plan[2],     plan[3],                                       plan[4],     t,
                        try e.weight("{s}.w2.svh", .{pre}, .f16, &.{ Et, D }),      y,           .{ .i = I },
                        .{ .i = D }, .{ .i = k2 },                                  .{ .i = 0 }, .{ .b = true },
                        .{ .b = true },
                    });
                }
                j += 1;
            }
        }
        const routed = try e.buf("L.routed", .{}, .f32, &.{ n, D });
        try e.ext("tensorfold_exl3_experts_v1.combine", &.{ y, w6, routed, .{ .i = n }, .{ .i = D }, .{ .i = topk } });
        // the shared expert: gate and up GEMMs, the clamped SiLU product (torch), the down GEMM
        // TF_DSV41_PF_OVERLAP (a whole segment's MoE call): the shared expert and bf16(routed + shared) a row piece at a
        // time (row-independent GEMMs and row-wise glue: the same bits), each piece's partial sent while the next runs
        const k: i64 = if (e.cfg.shared_experts > 0 and n == seg_rows and part.t.role == .buf and std.mem.eql(u8, part.t.role.buf, "L.moe16")) s.pieces() else 1;
        if (k > 1) {
            const sh = try e.fmt("{s}.shared.0", .{pre});
            const g = try e.buf("L.sh.g", .{}, .f32, &.{ n, I });
            const u = try e.buf("L.sh.u", .{}, .f32, &.{ n, I });
            const act = try e.buf("L.sh.act", .{}, .f32, &.{ n, I });
            const sy = try e.buf("L.sh.y", .{}, .f32, &.{ n, D });
            const gath = try s.recv("moe");
            var p: i64 = 0;
            while (p < k) : (p += 1) {
                const r = pieceAt(n, k, p);
                const m = r[1] - r[0];
                e.n = m;
                const xr = try rowsOf(e, x, .bf16, r[0], m, D);
                const gr = try rowsOf(e, g, .f32, r[0], m, I);
                const ur = try rowsOf(e, u, .f32, r[0], m, I);
                const ar = try rowsOf(e, act, .f32, r[0], m, I);
                const yr = try rowsOf(e, sy, .f32, r[0], m, D);
                try s.proj(try e.fmt("{s}.w1", .{sh}), .stored, xr, D, I, gr);
                try s.proj(try e.fmt("{s}.w3", .{sh}), .stored, xr, D, I, ur);
                try e.glue("swiglu", &.{ gr, ur, ar, .{ .f = lim } });
                try s.proj(try e.fmt("{s}.w2", .{sh}), .stored, ar, I, D, yr);
                try e.glue("moe_sum", &.{ try rowsOf(e, routed, .f32, r[0], m, D), yr, try rowsOf(e, part, .bf16, r[0], m, D) });
                e.n = n;
                try s.sendPiece(part, gath, r[0], m, p, k);
            }
            s.sent_moe = true;
            return;
        }
        if (e.cfg.shared_experts > 0) {
            const sh = try e.fmt("{s}.shared.0", .{pre});
            const g = try e.buf("L.sh.g", .{}, .f32, &.{ n, I });
            const u = try e.buf("L.sh.u", .{}, .f32, &.{ n, I });
            const act = try e.buf("L.sh.act", .{}, .f32, &.{ n, I });
            const sy = try e.buf("L.sh.y", .{}, .f32, &.{ n, D });
            try s.proj(try e.fmt("{s}.w1", .{sh}), .stored, x, D, I, g);
            try s.proj(try e.fmt("{s}.w3", .{sh}), .stored, x, D, I, u);
            try e.glue("swiglu", &.{ g, u, act, .{ .f = lim } });
            try s.proj(try e.fmt("{s}.w2", .{sh}), .stored, act, I, D, sy);
            try e.glue("moe_sum", &.{ routed, sy, part });
        } else try e.glue("moe_sum", &.{ routed, part });
    }

    /// Engram before the layer's first site (blocks.engram): wkv over the hashed rows (a prefill matmul), the fusion.
    fn engram(s: *Seg, L: u32) !void {
        const e = s.e;
        const n = e.n;
        const D: i64 = e.cfg.hidden;
        const K: i64 = @as(i64, e.cfg.hashCols()) * e.cfg.engram_head_dim;
        const N: i64 = (@as(i64, e.cfg.engram_max_ngram) + 1) * D;
        const kv = try e.buf("L.engram.kv", .{}, .bf16, &.{ n, N });
        const rws = try e.buf("L.engram.rows", .{}, .bf16, &.{ n, K });
        try e.glue("engram_rows", &.{ rws, .{ .i = L } });
        try s.proj(try e.fmt("L{d}.engram.wkv", .{L}), .strips, rws, K, N, kv);
        const qk = try e.weight("L{d}.engram.qk", .{L}, .f32, &.{ e.cfg.hc_mult, D });
        // image positions (TF_DSV41_IMAGES=native): the gate times vision.window's keep (0 there)
        const keep: ?Arg = if (e.o.image_keep) try e.buf("w.engram.keep", .{}, .f32, &.{n}) else null;
        if (keep) |k| try e.glue("image_keep", &.{k});
        try e.triton("_fuse_dec", .{ n, 1, 1 }, &.{
            .{ .name = "X", .arg = try e.streams() },                               .{ .name = "x_stride", .arg = .{ .i = 4 * D } },
            .{ .name = "KV", .arg = kv },                                           .{ .name = "kv_stride", .arg = .{ .i = N } },
            .{ .name = "QK", .arg = qk },                                           .{ .name = "KEEP", .arg = keep orelse qk },
            .{ .name = "GATE", .arg = qk },                                         .{ .name = "eps", .arg = .{ .f = E.dec(e.cfg.eps) } },
            .{ .name = "clamp", .arg = .{ .f = E.dec(e.cfg.engram_gate_clamp) } }, .{ .name = "sqrt_d", .arg = .{ .f = @sqrt(@as(f64, @floatFromInt(D))) } },
            .{ .name = "D", .arg = .{ .i = D } },                                   .{ .name = "BK", .arg = .{ .i = 64 } },
            .{ .name = "UB", .arg = .{ .i = 1024 } },                               .{ .name = "HAS_KEEP", .arg = .{ .b = keep != null } },
            .{ .name = "HAS_GATE", .arg = .{ .b = false } },
        });
    }

    // -----------------------------------------------------------------------------------------------------------
    // the segment (forward._run / _layers / _finish)

    /// Where an exchange (`which`: attn / moe) lands: the layer's (`gathered`), or a short segment's "w.recv.*"
    /// (block_moe.mhc's).
    fn recv(s: *Seg, comptime which: []const u8) !Arg {
        const e = s.e;
        return e.buf("{s}.recv." ++ which, .{if (s.short) "w" else "L"}, .bf16, &.{ e.o.world, e.n, e.cfg.hidden });
    }

    fn run(s: *Seg, layers: []const u32, head: bool) !void {
        const e = s.e;
        const n = e.n;
        const D: i64 = e.cfg.hidden;
        if (s.part == .decoder and s.multi != null) {
            // several slots' decoder replays (`emitMulti`): every segment's positions (rope over the run's rows), then
            // each segment's rows entering the decoder from its slot's stash, at its rows of the run
            try e.glue("mpositions", &.{try e.buf("w.mpos64", .{}, .i64, &.{n})});
            for (s.multi.?, 0..) |sp, j| {
                try s.enter(sp, j);
                try e.glue("ced_load", &try stashArgs(e));
            }
            try s.leave(n);
        } else if (s.part == .decoder) {
            // replay.finish: every row's window starts at the pass's first row R0 (ctx.lo), and the rows entering the
            // decoder come from the stash (its Streams: the streams, the normed input, coefficient set 0)
            try e.glue("positions", &.{ try pos(e), try e.buf("w.pos64", .{}, .i64, &.{n}), try e.buf("w.lo", .{}, .i32, &.{n}), .{ .i = e.start } });
            const st = try stashArgs(e);
            try e.glue("ced_load", &st);
        } else if (s.multi != null) {
            // every segment's positions (rope over the run's rows); each segment's own at its CSA2 steps (pf_seg)
            try e.glue("mpositions", &.{try e.buf("w.mpos64", .{}, .i64, &.{n})});
            try e.glue("embed", &.{ try e.streams(), try e.buf("w.recv.moe", .{}, .bf16, &.{ e.o.world, n, D }) });
        } else {
            try e.glue("positions", &.{ try pos(e), try e.buf("w.pos64", .{}, .i64, &.{n}), try e.buf("w.lo", .{}, .i32, &.{n}) });
            try e.glue("embed", &.{ try e.streams(), try e.buf("w.recv.moe", .{}, .bf16, &.{ e.o.world, n, D }) });
        }
        for (layers, 0..) |L, i| {
            if (i > 0) e.begin = .layer;
            // TF_DSV41_PF_TBO: a layer's attention part starts here (the first's at the program's first call)
            if (i > 0) if (e.marks) |m| try m.append(e.a, e.out.items.len);
            e.layer_ratio = e.cfg.compressRatio(L);
            if (s.part == .decoder and i == 0) {
                // entered (forward._layers): the first block's attention site is the stash's
            } else if (i == 0 or e.cfg.isEngram(L)) {
                // the run's first block, or an Engram block: the last exchange's post alone, Engram, then a site that
                // collapses with the previous coefficients' pre
                if (i > 0) try s.postOnly(try e.streams());
                if (e.cfg.isEngram(L)) try s.engram(L);
                try s.site(L, "hc_attn", if (i == 0) .site1 else .site2);
            } else try s.site(L, "hc_attn", .boundary);
            // CED's encoder pass stops at the decoder's first layer once its site and compressor ran (`upto`)
            if (s.part == .encoder and i + 1 == layers.len) return s.encoderEnd(L);
            try s.attention(L);
            if (s.short) try e.prefetchF(L);
            // TF_DSV41_PF_OVERLAP: the partial went out a piece at a time behind its producer
            if (!s.sent_attn) try e.glue("exchange", &.{ try e.buf("L.part", .{}, .bf16, &.{ n, D }), try s.recv("attn") });
            s.sent_attn = false;
            if (e.marks) |m| try m.append(e.a, e.out.items.len); // its MoE part
            try s.site(L, "hc_ffn", .boundary);
            try s.moeBlock(L);
            if (s.short and i + 1 < layers.len) try e.prefetchX(layers[i + 1]);
            if (!s.sent_moe) try e.glue("exchange", &.{ try e.buf("L.moe16", .{}, .bf16, &.{ n, D }), try s.recv("moe") });
            s.sent_moe = false;
            if (e.o.trace) {
                const y = try e.buf("w.trace", .{}, .bf16, &.{ n, 4 * D });
                try s.postOnly(y);
                try e.glue("trace", &.{ y, .{ .i = L } });
            }
        }
        if (head) try s.final();
        s.unpend();
    }

    /// A program ending on an exchange's pieces (TF_DSV41_PF_OVERLAP_SITE, no consumer after them): its last piece
    /// back on the main stream after a join, as without the knob, so no fork stays open.
    fn unpend(s: *Seg) void {
        const e = s.e;
        if (s.pend == 0) return;
        s.pend = 0;
        e.br_join = false;
        const c = &e.out.items[e.out.items.len - 1];
        std.debug.assert(std.mem.eql(u8, c.name, "glue.exchange_rows") and c.side);
        c.side = false;
        c.fork = false;
        c.mark = 0;
        c.join = true;
    }
};

/// The CED stash glue's tensors (ced_stash / ced_load): the run's streams, normed input and current coefficient set,
/// then the ring's roles (ced.roles, `ced.ring` rows; position p at row p % ring).
fn stashArgs(e: *E) ![10]Arg {
    const D: i64 = e.cfg.hidden;
    const R: i64 = ced.ring;
    const c = try Seg.coefs(e, e.coef);
    const x = try e.streams();
    const r0 = e.row0;
    return .{
        try Seg.rowsOf(e, x, .bf16, r0, e.n, 4 * D),              try Seg.rowsOf(e, try e.buf("w.out", .{}, .bf16, &.{ e.n, D }), .bf16, r0, e.n, D),
        try Seg.rowsOf(e, c[0], .f32, r0, e.n, 4),                try Seg.rowsOf(e, c[1], .f32, r0, e.n, 4),
        try Seg.rowsOf(e, c[2], .f32, r0, e.n, 16),               try e.buf(ced.roles[0], .{}, .bf16, &.{ R, 4 * D }),
        try e.buf(ced.roles[1], .{}, .bf16, &.{ R, D }),      try e.buf(ced.roles[2], .{}, .f32, &.{ R, 4 }),
        try e.buf(ced.roles[3], .{}, .f32, &.{ R, 4 }),       try e.buf(ced.roles[4], .{}, .f32, &.{ R, 16 }),
    };
}

/// One segment of a multi-segment run (`emitMulti`): its slot's rows from position `start`, rows [row0, row0 + n) of
/// the run.
pub const Span = struct { slot: u32, start: i64, n: i64, row0: i64 = 0, cnt0: i64 = 0 };

/// The most rows a prefill program takes: one router CHUNK (Python prod's segment), 4,096 under TF_DSV41_PF_4K (the
/// route in CHUNK launches, x3gm one block).
fn rowsCap(o: block.Options) i64 {
    return if (o.pf4k) 2 * router_chunk else router_chunk;
}

/// forward.staging's ring rows for a prefill segment the slot's ring cannot hold.
fn stageRows(e: *const E) i64 {
    const W: i64 = e.cfg.window;
    return @intCast(std.math.ceilPowerOfTwo(u64, @intCast(e.o.prefill_rows + W - 1)) catch unreachable);
}

/// Several slots' prompt segments in one run (Python's slots.prefill_runs / forward._run over several contexts, GLM
/// 0560), or several slots' CED decoder replays (`.decoder`: each segment a slot's prefilled tail from its R0, its rows
/// entering the decoder from its slot's stash; Python replays a slot at a time): the embedding, the mHC sites, every dense projection, the experts and the exchanges over all the run's rows
/// (the segments' rows in order), and each segment's CSA2 steps (SWA store, compressor, selection, fused core) on its
/// own slot (glue "pf_seg" switches the slot's views), as blocks.attention's loop over its contexts. `part`: `.whole`
/// (full mode, no head: a prompt's pieces never take logits) or `.encoder` (CED replay's encoder pass, each segment's
/// stash). Every segment past 16 rows (a short segment's decode-kernel mix is one segment's), the run within one
/// prefill segment's rows, no image rows; error.Unsupported otherwise (the caller runs the segments one at a time).
pub fn emitMulti(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, layers: []const u32, spans_in: []const Span, part: Part) ![]calls.Call {
    if (spans_in.len == 0 or o.image_rows != 0) return error.Unsupported;
    if (o.pool) |pl| if (pl.split) return error.Unsupported; // split KV's exchanges: one segment a run
    const spans = try a.dupe(Span, spans_in);
    var total: i64 = 0;
    // the decoder replay's rows past the split mHC programs' 32 (a replay batched is the one-slot replay's arithmetic
    // row for row: the same kernel paths at every row count it takes); a prompt piece's past the short segment's 16
    const floor: i64 = if (part == .decoder) split_rows else 16;
    var cnt: i64 = 0;
    for (spans) |*sp| {
        if (sp.n <= floor) return error.Unsupported;
        sp.row0 = total;
        sp.cnt0 = cnt;
        total += sp.n;
        cnt = std.mem.alignForward(i64, cnt + sp.n, 4);
    }
    if (total > o.prefill_rows or total > rowsCap(o)) return error.Unsupported;
    var e: E = .{ .a = a, .cfg = cfg, .w = w, .o = o, .n = total, .start = spans[0].start };
    var s: Seg = .{ .e = &e, .staged = false, .ring = 0, .part = part, .multi = spans };
    s.setRing();
    try s.run(layers, false);
    return e.out.items;
}

/// The multi-segment runs' programs at their largest, for the buffer plan (forward_prefill.plan): a run reads roles no
/// one-segment program has (the run's positions "w.mpos64") and the run's rows' roles at up to `rows` rows. `enc`: a
/// prompt piece run's layers (CED's encoder, or every layer in full mode: `part`); `dec`: CED's decoder layers (null in
/// full mode), its runs of tails of 127 rows (ced.keep) up to `rows`. A run its options refuse (split KV) adds nothing.
pub fn planRuns(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, enc: []const u32, dec: ?[]const u32, rows: i64, part: Part) !std.ArrayList([]const calls.Call) {
    var out: std.ArrayList([]const calls.Call) = .empty;
    if (rows > 2 * 17) {
        const sp = [_]Span{ .{ .slot = 0, .start = 0, .n = rows - 17 }, .{ .slot = 1, .start = 0, .n = 17 } };
        if (emitMulti(a, cfg, w, o, enc, &sp, part)) |cs| try out.append(a, cs) else |e| if (e != error.Unsupported) return e;
    }
    // the counts' widest layout: 15 segments of 17 rows (each padded 3 to the next multiple of 4) and the rest
    if (rows > 16 * 17) {
        var sp: [16]Span = undefined;
        for (sp[0..15], 0..) |*x, i| x.* = .{ .slot = @intCast(i), .start = 0, .n = 17 };
        sp[15] = .{ .slot = 15, .start = 0, .n = rows - 15 * 17 };
        if (emitMulti(a, cfg, w, o, enc, &sp, part)) |cs| try out.append(a, cs) else |e| if (e != error.Unsupported) return e;
    }
    if (dec) |d| {
        var tails: [16]Span = undefined;
        const k: usize = @intCast(@min(16, @divFloor(rows, ced.keep)));
        if (k >= 2) {
            for (tails[0..k], 0..) |*t, i| t.* = .{ .slot = @intCast(i), .start = 0, .n = ced.keep };
            if (emitMulti(a, cfg, w, o, d, tails[0..k], .decoder)) |cs| try out.append(a, cs) else |e| if (e != error.Unsupported) return e;
        }
    }
    return out;
}

/// A prefill program's part: a whole segment, or one of CED replay's passes (ced.zig).
pub const Part = enum { whole, encoder, decoder };

/// CED replay's programs (TF_DSV41_PREFILL=replay, ced.zig), `n` rows from `start`:
/// - `.encoder`: a prompt segment over `layers` (0 .. D, D last: the decoder's first layer), D only to its attention
///   site and compressor, then the stash glue; no head (replay.encode: run(upto=D), logits off);
/// - `.decoder`: layers D .. from D's attention (`layers[0]` = D), its compressor skipped, every window from `start`
///   (= R0), the rows entering D from the stash, the boundaries in place, no final norm or head (replay.finish).
pub fn emitReplay(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, layers: []const u32, n: i64, start: i64, part: Part) ![]calls.Call {
    if (layers.len == 0) return error.Unsupported;
    return emitPart(a, cfg, w, o, layers, n, start, part == .whole, part);
}

/// The calls of a prefill segment of `n` rows from position `start` over `layers` (ascending, consecutive), and the
/// final norm + head when `head` (arena-owned by `a`).
pub fn emitPrefill(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, layers: []const u32, n: i64, start: i64, head: bool) ![]calls.Call {
    return emitPart(a, cfg, w, o, layers, n, start, head, .whole);
}

fn emitPart(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, layers: []const u32, n: i64, start: i64, head: bool, part: Part) ![]calls.Call {
    return emitMarked(a, cfg, w, o, layers, n, start, head, part, null);
}

/// TF_DSV41_PF_TBO: micro-batches of at least this many rows (smaller ones run alone: their GEMMs' waves and the
/// interleave's joins cost more than the overlap hides)
pub const tbo_min: i64 = 512;

/// The persistent roles a segment uses as scratch (rewritten every segment, never read by the next): a program in
/// a namespace of its own (`ownRole`: a TF_DSV41_PF_TBO pair's B, TF_DSV41_PF_4K's workspace) takes its own copy of
/// each. Every other "s." role is the slot's state (the KV pool, SWA rings, ratio-2 carries, index keys, the CED
/// stash) or a constant table, which every namespace shares.
pub const own_scratch = [_][]const u8{ "s.ex.", "s.pf.xh", "s.pf.w", "s.router.", "s.stage.", "s.attn." };

/// A role in namespace `tag`: "w.x" -> "w.<tag>x", "L.x" -> "L.<tag>x" (the same scope), a scratch "s.x" ->
/// "s.<tag>x"; the slot's state and the constants keep theirs.
pub fn ownRole(a: std.mem.Allocator, r: []const u8, comptime tag: []const u8) ![]const u8 {
    if (std.mem.startsWith(u8, r, "s.")) {
        // a role already in a namespace ("s.b~ex.z") is scratch by its bare name
        var bare = r[2..];
        while (std.mem.indexOf(u8, bare, "~")) |i| {
            if (std.mem.indexOfScalar(u8, bare[0..i], '.') != null) break;
            bare = bare[i + 1 ..];
        }
        for (own_scratch) |p| if (std.mem.startsWith(u8, bare, p[2..])) return std.fmt.allocPrint(a, "s." ++ tag ++ "{s}", .{r[2..]});
        return r;
    }
    const dot = std.mem.indexOfScalar(u8, r, '.') orelse return std.fmt.allocPrint(a, tag ++ "{s}", .{r});
    return std.fmt.allocPrint(a, "{s}" ++ tag ++ "{s}", .{ r[0 .. dot + 1], r[dot + 1 ..] });
}

/// Micro-batch B's name of a role (TF_DSV41_PF_TBO).
pub fn tboRole(a: std.mem.Allocator, r: []const u8) ![]const u8 {
    return ownRole(a, r, "b~");
}

/// The roles a namespace leaves as they are (ownCallsKeep): the forward's, which hold them already.
pub const Keep = std.StringHashMapUnmanaged(void);

fn ownArg(a: std.mem.Allocator, x: Arg, comptime tag: []const u8, keep: ?*const Keep) !Arg {
    return switch (x) {
        .t, .opaque_table => |t| blk: {
            if (t.role != .buf) break :blk x;
            if (keep) |k| if (k.contains(t.role.buf)) break :blk x;
            var u = t;
            u.role = .{ .buf = try ownRole(a, t.role.buf, tag) };
            break :blk if (x == .t) .{ .t = u } else .{ .opaque_table = u };
        },
        .list => |l| blk: {
            const out = try a.alloc(Arg, l.len);
            for (l, out) |y, *z| z.* = try ownArg(a, y, tag, keep);
            break :blk .{ .list = out };
        },
        else => x,
    };
}

fn tboArg(a: std.mem.Allocator, x: Arg) !Arg {
    return ownArg(a, x, "b~", null);
}

/// `cs` with every role in namespace `tag` (ownRole), the calls otherwise as they are.
pub fn ownCalls(a: std.mem.Allocator, cs: []const calls.Call, comptime tag: []const u8) ![]calls.Call {
    return ownCallsKeep(a, cs, tag, null);
}

/// `ownCalls` but for the roles in `keep`, which stay the forward's.
pub fn ownCallsKeep(a: std.mem.Allocator, cs: []const calls.Call, comptime tag: []const u8, keep: ?*const Keep) ![]calls.Call {
    const out = try a.alloc(calls.Call, cs.len);
    for (cs, out) |c, *x| {
        x.* = c;
        const args = try a.alloc(calls.Named, c.args.len);
        for (c.args, args) |y, *z| z.* = .{ .name = y.name, .arg = try ownArg(a, y.arg, tag, keep) };
        x.args = args;
    }
    return out;
}

/// TF_DSV41_PF_TBO: two consecutive CED encoder segments as one program on two streams (two micro-batches; DeepSeek's
/// two-batch overlap). A = rows [start, start + na), B = [start + na, start + na + nb), each its own segment's calls
/// exactly as `emitReplay(.encoder)` emits them (so each row's arithmetic is a consecutive segment's: segmentation-
/// invariant, as the 2,048-row path already relies on), B's on the branches' side stream with its own scratch
/// (`tboRole`), half a layer behind A:
///   A.attn(L), [fork] B.attn(L), A.moe(L), B.moe(L), A.attn(L + 1), [fork] B.attn(L + 1), ...
/// so A's routed experts run beside B's attention (and its dense GEMMs), and B's beside A's next attention. B reads
/// the slot's state of layer L (SWA ring rows, index keys and compressed rows in the pool, the ratio-2 carry) only
/// after A's layer L wrote it: each B attention part is forked behind A's, and A hands its ratio-2 carry of a layer to
/// the slot at the end of that layer's attention (glue copy_rows: the copy A's commit would have made), not at the
/// pair's end. A writes no state of a layer B is still reading (a layer's state is its own roles); the one state both
/// touch at once is a kv source's compressed rows, which A's later layers read up to A's positions while B's source
/// layer appends rows past them (disjoint rows: the pair's boundary is even, so no ratio-2 pair straddles it). A's CED stash is
/// left out (B's, of at least 128 rows, overwrites all of it). An Engram layer's A part joins B first when B ran an
/// Engram step since (the Engram glue's gathered buffer and pinned stages are one set), and the program ends with a join (glue tbo_join), so the commit sees both.
/// error.Unsupported: either part off the prefill path, a micro-batch under `tbo_min` rows, split KV, or images.
pub fn emitTbo(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, layers: []const u32, na: i64, nb: i64, start: i64) ![]calls.Call {
    if (!o.pf_tbo or !o.branches or na < tbo_min or nb < tbo_min or o.image_rows != 0) return error.Unsupported;
    // B's first row starts a ratio-2 pair: its compressed rows begin after A's last (A's later layers read a kv
    // source's compressed rows up to A's own positions while B's source layer appends past them)
    if (@mod(start, 2) != 0 or @mod(na, 2) != 0) return error.Unsupported;
    if (o.pool) |pl| if (pl.split) return error.Unsupported;
    var ma: std.ArrayList(usize) = .empty;
    var mb: std.ArrayList(usize) = .empty;
    const pa = try emitMarked(a, cfg, w, o, layers, na, start, false, .encoder, &ma);
    const pb = try emitMarked(a, cfg, w, o, layers, nb, start + na, false, .encoder, &mb);
    if (ma.items.len != mb.items.len) return error.Unsupported;
    // the phases: [0, m0), [m0, m1), ... alternating attention / MoE parts, the last the encoder's end (D's site + compressor)
    const nph = ma.items.len + 1;
    var out: std.ArrayList(calls.Call) = .empty;
    var started = false; // B issued any call since the last join (its next part forks)
    var b_engram = false; // B issued an Engram step since the last join (A's next Engram step joins it first)
    for (0..nph) |k| {
        const a0 = if (k == 0) 0 else ma.items[k - 1];
        const a1 = if (k < ma.items.len) ma.items[k] else pa.len;
        const b0 = if (k == 0) 0 else mb.items[k - 1];
        const b1 = if (k < mb.items.len) mb.items[k] else pb.len;
        const attn = k % 2 == 0;
        const L = layers[k / 2];
        // A's part on the main stream
        const first_a = out.items.len;
        for (pa[a0..a1]) |c| {
            if (std.mem.eql(u8, c.name, "glue.ced_stash")) continue;
            try out.append(a, c);
        }
        if (b_engram and attn and out.items.len > first_a) {
            for (pa[a0..a1]) |c| if (std.mem.eql(u8, c.name, "glue.engram_rows")) {
                out.items[first_a].join = true;
                started = false;
                b_engram = false;
                break;
            };
        }
        if (attn and cfg.isKvSource(L) and cfg.compressRatio(L) == 2) {
            // the slot's ratio-2 carry: A's last projection row, as A's commit (forward_prefill.commitAt) would copy it
            const row: i64 = 2 * @as(i64, cfg.head_dim);
            const src: Arg = .{ .t = .{ .role = .{ .buf = try std.fmt.allocPrint(a, "w.L{d}.proj", .{L}) }, .dt = .f32, .shape = try a.dupe(i64, &.{ 1, row }), .stride = try a.dupe(i64, &.{ row, 1 }), .offset = (na - 1) * row * 4 } };
            const dst: Arg = .{ .t = .{ .role = .{ .buf = try std.fmt.allocPrint(a, "s.L{d}.carry", .{L}) }, .dt = .f32, .shape = try a.dupe(i64, &.{ 1, row }), .stride = try a.dupe(i64, &.{ row, 1 }) } };
            const args = try a.alloc(calls.Named, 2);
            args[0] = .{ .arg = src };
            args[1] = .{ .arg = dst };
            try out.append(a, .{ .triton = false, .glue = true, .name = "glue.copy_rows", .args = args });
        }
        // B's part on the side stream, its attention parts forked behind A's
        for (pb[b0..b1], 0..) |c, j| {
            var x = c;
            const args = try a.alloc(calls.Named, c.args.len);
            for (c.args, args) |y, *z| z.* = .{ .name = y.name, .arg = try tboArg(a, y.arg) };
            x.args = args;
            x.side = true;
            x.fork = j == 0 and (attn or !started);
            x.join = false;
            x.begin = .none;
            try out.append(a, x);
            started = true;
            if (std.mem.eql(u8, c.name, "glue.engram_rows")) b_engram = true;
        }
    }
    try out.append(a, .{ .triton = false, .glue = true, .name = "glue.tbo_join", .args = &.{}, .join = true });
    return out.items;
}

fn emitMarked(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o: block.Options, layers: []const u32, n: i64, start: i64, head: bool, part: Part, marks: ?*std.ArrayList(usize)) ![]calls.Call {
    if (n < 1 or n > o.prefill_rows or n > rowsCap(o)) return error.Unsupported;
    var e: E = .{ .a = a, .cfg = cfg, .w = w, .o = o, .n = n, .start = start, .marks = marks };
    const W: i64 = cfg.window;
    // forward.staging: the slot's ring holds 2 x window rows; a longer segment goes through the staging ring
    const staged = n + W - 1 > 2 * W;
    const stage: i64 = @intCast(std.math.ceilPowerOfTwo(u64, @intCast(o.prefill_rows + W - 1)) catch return error.Unsupported);
    const ring = if (staged) stage else 2 * W;
    if (ring < n + W - 1) return error.Unsupported; // attn_pf: the ring must hold the rows and their windows
    var s: Seg = .{ .e = &e, .staged = staged, .ring = ring, .short = n <= 16, .part = part };
    try s.run(layers, head);
    if (o.holdsDeferred(n)) calls.holdDeferred(e.out.items);
    return e.out.items;
}
