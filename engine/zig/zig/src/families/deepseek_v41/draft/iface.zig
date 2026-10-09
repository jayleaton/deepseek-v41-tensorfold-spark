//! The contract between DeepSeek's lanes backend and the model: `Target` (the forward over verify windows, M2's
//! forward.zig on the GPU, twin.zig on the CPU) and `Pass` (the DSpark pass over its context rings). The drafter
//! never touches the target's pool: it reads the taps the target hands back and keeps its own rings.
const std = @import("std");
const lanes = @import("lanes");
const dspark = @import("dspark.zig");

/// One slot's rows in a verify window. Row 0 is the pending token at `start` (the slot's cache length); row r > 0
/// sits at start + its depth, attends the committed cache plus its ancestors in the window (`parents`), and its
/// choice is keyed at `draws[r]` (= its own position + 1).
pub const Segment = struct {
    slot: u32,
    start: u64,
    tokens: []const u32,
    parents: ?[]const i32, // each row's parent row (row 0: -1); null: a chain
    draws: []const u64,
    sampling: ?lanes.Sampling,
};

/// The taps of a slot's last rows (the streams' mean entering the DSpark target layers, bf16 [rows, 3 x hidden] on
/// the GPU): `address` a device address (a host one on the CPU twin), row i = the window's (or prompt's) row i.
/// `map`: a tree window resolved as chains (branches.zig): window row -> taps row (branches.unrun: no taps).
pub const Taps = struct { address: u64, rows: u32, stride: u32, map: ?[]const u32 = null };

pub const Target = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Run the prompt into the slot's state; the keyed choice after its last row (drawn at `draw`).
        prefill: *const fn (ptr: *anyopaque, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) anyerror!u32,
        /// One forward over every segment; `choices[i][r]`: segment i's row r's keyed choice. A segment's rows stay in
        /// its state until `keep` (a chain window that is never kept keeps every row).
        window: *const fn (ptr: *anyopaque, segments: []const Segment, choices: [][]u32) anyerror!void,
        /// Keep the listed rows of the slot's last window (row 0 first, each the next one's parent), drop the rest:
        /// pool truncation, SWA rings and ratio-2 carries roll back to the kept path.
        keep: *const fn (ptr: *anyopaque, slot: u32, path: []const u32) anyerror!void,
        /// The taps of the slot's last window (or its prompt, after prefill).
        taps: *const fn (ptr: *anyopaque, slot: u32) Taps,
        release: *const fn (ptr: *anyopaque, slot: u32) void,
        /// Before `prefill`: the request's reply budget (its max new tokens), for a target that admits requests by the
        /// pages they may write (the GPU target's pool admission, prod's batch.py `_admit`). null: nothing to do.
        admit: ?*const fn (ptr: *anyopaque, slot: u32, max_new: u64) void = null,
        /// A prompt in pieces between decode rounds (prod's batcher: kv/sched.zig), in place of `prefill`: `begin`
        /// takes the request into the slot (its pool pages, a saved prefix restored): the tokens its state holds and
        /// whether a damaged saved entry was dropped; `piece` runs prompt rows [start, end) (`save`: the prompt's
        /// snapshot right after them); `finish` runs the prompt's last token as the reply's first window: the keyed
        /// choice after it. null: the target runs whole prompts only.
        begin: ?*const fn (ptr: *anyopaque, slot: u32, ids: []const u32) anyerror!Begun = null,
        piece: ?*const fn (ptr: *anyopaque, slot: u32, ids: []const u32, start: u64, end: u64, save: bool) anyerror!void = null,
        finish: ?*const fn (ptr: *anyopaque, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) anyerror!u32 = null,
        /// `finish` up to the prompt's last token (TF_DSV41_TAIL_JOIN): the decoder replay and its taps to the
        /// drafter, the slot left at the last token's position with no window pending, so the reply's first window is
        /// a round's shared one (Python's batcher `finals`). null from the call: done so; a choice: the target ran the
        /// last row as `finish` (a grammar's first mask). null: `finish`.
        tail: ?*const fn (ptr: *anyopaque, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) anyerror!?u32 = null,
        /// A round's pieces of several slots (TF_DSV41_PIECE_RUNS): one forward over them where the target can, else
        /// `piece` a piece. null: `piece` a piece.
        pieces: ?*const fn (ptr: *anyopaque, list: []const Piece) anyerror!void = null,
        /// A round's finished prompts before their `finish` / `tail` (TF_DSV41_REPLAY_RUNS): the target may run their
        /// CED decoder replays together (those finishes then skip theirs). null: nothing.
        replays: ?*const fn (ptr: *anyopaque, list: []const Final) anyerror!void = null,
        /// Engram reads ahead of the next window (TF_DSV41_ENGRAM_WARM, Python's decode.prefetch_pending at a commit
        /// and its draft-end gate.warm): each ask's rows `ids` from position `start` on its slot. Reads only: a
        /// window's rows are its own reads' bytes whatever was warmed.
        warm: ?*const fn (ptr: *anyopaque, asks: []const Warm) void = null,
    };

    /// A slot's whole prompt whose rows but the last are in (its `finish` / `tail` follows).
    pub const Final = struct { slot: u32, ids: []const u32 };

    /// A slot's prompt `ids` (the whole prompt) rows [start, end) of a round; `save`: its snapshot right after them.
    pub const Piece = struct { slot: u32, ids: []const u32, start: u64, end: u64, save: bool };


    pub fn warm(t: Target, asks: []const Warm) void {
        if (t.vtable.warm) |f| f(t.ptr, asks);
    }

    pub const Begun = struct { at: u64 = 0, damaged: bool = false };

    pub fn admit(t: Target, slot: u32, max_new: u64) void {
        if (t.vtable.admit) |f| f(t.ptr, slot, max_new);
    }

    pub fn pieces(t: Target) bool {
        return t.vtable.begin != null and t.vtable.piece != null and t.vtable.finish != null;
    }
    pub fn begin(t: Target, slot: u32, ids: []const u32) !Begun {
        return (t.vtable.begin orelse return error.NoPieces)(t.ptr, slot, ids);
    }
    pub fn piece(t: Target, slot: u32, ids: []const u32, start: u64, end: u64, save: bool) !void {
        return (t.vtable.piece orelse return error.NoPieces)(t.ptr, slot, ids, start, end, save);
    }
    pub fn finish(t: Target, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) !u32 {
        return (t.vtable.finish orelse return error.NoPieces)(t.ptr, slot, ids, sampling, draw);
    }
    pub fn runReplays(t: Target, list: []const Final) !void {
        if (t.vtable.replays) |f| return f(t.ptr, list);
    }
    pub fn runPieces(t: Target, list: []const Piece) !void {
        if (t.vtable.pieces) |f| return f(t.ptr, list);
        for (list) |p| try t.piece(p.slot, p.ids, p.start, p.end, p.save);
    }
    pub fn tail(t: Target, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) !?u32 {
        const f = t.vtable.tail orelse return try t.finish(slot, ids, sampling, draw);
        return f(t.ptr, slot, ids, sampling, draw);
    }

    pub fn prefill(t: Target, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) !u32 {
        return t.vtable.prefill(t.ptr, slot, ids, sampling, draw);
    }
    pub fn window(t: Target, segments: []const Segment, choices: [][]u32) !void {
        return t.vtable.window(t.ptr, segments, choices);
    }
    pub fn keep(t: Target, slot: u32, path: []const u32) !void {
        return t.vtable.keep(t.ptr, slot, path);
    }
    pub fn taps(t: Target, slot: u32) Taps {
        return t.vtable.taps(t.ptr, slot);
    }
    pub fn release(t: Target, slot: u32) void {
        t.vtable.release(t.ptr, slot);
    }
};

/// One slot of a DSpark pass: its anchor (the pending token) at P and the keyed draft parameters.
pub const Ask = struct { slot: u32, anchor: u32, start: u64, params: dspark.Params };

/// A pass's results for one slot, in caller buffers: `block` drafts and confidences; for trees also the gathered
/// candidates and base logits [block][candidates] (before the Markov bias), when `cand` is set.
pub const Proposal = struct { drafts: []u32, conf: []f32, cand: ?[]i32 = null, base: ?[]f32 = null };

pub const Pass = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Committed rows into the slot's context rings: `taps` rows `rows[i]` land at positions start + i.
        ingest: *const fn (ptr: *anyopaque, slot: u32, start: u64, taps: Taps, rows: []const u32) anyerror!void,
        /// One pass over every ask (one launch a block over all slots); results into `out`.
        propose: *const fn (ptr: *anyopaque, asks: []const Ask, out: []Proposal) anyerror!void,
        /// A new request on the slot: its context restarts at position 0.
        reset: *const fn (ptr: *anyopaque, slot: u32) void,
        /// The Markov / confidence heads on the host, for trees (null: no trees).
        markov: *const fn (ptr: *anyopaque) ?dspark.Markov,
        /// Start a pass over `asks` whose results `collect` returns: the device drafts while the host goes on
        /// (spec.zig, TF_DSV41_SPEC_DRAFT). Between the two, only target work without collectives may run (`keep`).
        /// Null: `propose` only.
        begin: ?*const fn (ptr: *anyopaque, asks: []const Ask) anyerror!void = null,
        /// The begun pass's results, in its asks' order (as `propose` gives them).
        collect: ?*const fn (ptr: *anyopaque, out: []Proposal) anyerror!void = null,
        /// Before a verify window whose next pass would be speculated (spec.zig): the slot's window of `rows` rows at
        /// `start`, so the pass may start from the window's device pick, before the host has the tokens (Python
        /// spec.py's order); `begin` then takes it over. `sampled`: its rows are keyed (T > 0, the sampler's device
        /// choice), else greedy (the GPU pick). Null: the pass starts at `begin`.
        arm: ?*const fn (ptr: *anyopaque, slot: u32, start: u64, rows: u32, sampled: bool) void = null,
    };

    pub fn ingest(p: Pass, slot: u32, start: u64, taps: Taps, rows: []const u32) !void {
        return p.vtable.ingest(p.ptr, slot, start, taps, rows);
    }
    pub fn propose(p: Pass, asks: []const Ask, out: []Proposal) !void {
        return p.vtable.propose(p.ptr, asks, out);
    }
    pub fn reset(p: Pass, slot: u32) void {
        p.vtable.reset(p.ptr, slot);
    }
    pub fn markov(p: Pass) ?dspark.Markov {
        return p.vtable.markov(p.ptr);
    }
};

/// Which committed rows an ingest writes: only the last `window` (older ones are never read).
pub fn ingestSpan(n: usize, window: usize) struct { skip: usize, count: usize } {
    const skip = n -| window;
    return .{ .skip = skip, .count = n - skip };
}

/// An Engram warm ask (Target.warm).
pub const Warm = struct { slot: u32, start: u64, ids: []const u32 };
