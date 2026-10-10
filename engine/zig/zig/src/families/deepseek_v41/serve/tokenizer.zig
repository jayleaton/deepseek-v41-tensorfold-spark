//! DeepSeek-V4.1's tokenizer: the release tokenizer.json, encoded as Hugging Face ``tokenizers`` does it
//! (``encode(text, add_special_tokens=False)``) and decoded as its ByteLevel decoder does (``skip_special_tokens=False``).
//!
//! The release pipeline is fixed, so it is written out instead of interpreted:
//! 1. added tokens: the non-normalized ones (the specials, the image span tokens) split the text first, leftmost-longest;
//!    then the normalized ones (``<｜User｜>``, ``<think>``, ``｜DSML｜`` ...) split what is left (the normalizer is empty);
//! 2. ``\p{N}{1,3}``, then ``[一-龥぀-ゟ゠-ヿ]+``, then the word regex
//!    ``[!-/:-@[-`{-~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+``,
//!    each an Isolated split of the pieces before it, as hand-written scanners (Oniguruma's leftmost-first semantics);
//! 3. byte-level BPE on each piece with ``tokenizers``' merge order (lowest rank, then leftmost; stale merges skipped).
//!
//! ``Tokenizer`` is immutable and shared; an ``Encoder`` holds one thread's scratch and piece cache, so encoding a
//! prompt allocates nothing once its buffers have grown. ``Detokenizer`` turns ids into text a token at a time.
const std = @import("std");
const unicode = @import("tokenizer").unicode; // the core tokenizer's Unicode tables
const Allocator = std.mem.Allocator;

pub const Error = error{ InvalidTokenizer, OutOfMemory };

/// The 32-bit id of a token; ids past the vocabulary decode to nothing (as ``tokenizers`` skips unknown ids).
pub const Id = u32;

/// A byte the vocabulary has no token for.
const missing: Id = std.math.maxInt(Id) - 1;

// code point classes the pre-tokenizer reads
const L: u8 = 1;
const M: u8 = 2;
const N: u8 = 4;
const P: u8 = 8;
const S: u8 = 16;
const WS: u8 = 32;
const CJK: u8 = 64;
const CRLF: u8 = 128;

/// Classes of every code point below ``table_end``; the rest are looked up (planes 3+ are almost empty).
const table_end = 0x30000;

fn classOf(cp: u21) u8 {
    var f: u8 = 0;
    const cat = unicode.bit(unicode.category(cp));
    if (cat & unicode.letter != 0) f |= L;
    if (cat & unicode.mark != 0) f |= M;
    if (cat & unicode.number != 0) f |= N;
    if (cat & unicode.punctuation != 0) f |= P;
    if (cat & unicode.symbol != 0) f |= S;
    if (unicode.isSpace(cp)) f |= WS;
    if ((cp >= 0x4E00 and cp <= 0x9FA5) or (cp >= 0x3040 and cp <= 0x30FF)) f |= CJK;
    if (cp == '\r' or cp == '\n') f |= CRLF;
    return f;
}

/// GPT-2's byte alphabet: printable Latin-1 maps to itself, the other 68 bytes to U+0100 onwards.
const byte_chars: [256]u21 = blk: {
    var table: [256]u21 = undefined;
    var next: u21 = 256;
    for (0..256) |b| {
        const printable = (b >= '!' and b <= '~') or (b >= 0xA1 and b <= 0xAC) or b >= 0xAE;
        table[b] = if (printable) b else next;
        if (!printable) next += 1;
    }
    break :blk table;
};

/// A merge table entry: (left, right) -> rank and merged id, open addressing on a power-of-two table.
const Merges = struct {
    keys: []u64,
    vals: []u64, // rank << 32 | id
    mask: u64,

    const empty: u64 = std.math.maxInt(u64);

    inline fn slot(m: *const Merges, key: u64) u64 {
        return (key *% 0x9E3779B97F4A7C15) >> 40 & m.mask;
    }

    fn put(m: *Merges, left: Id, right: Id, rank: u32, id: Id) void {
        const key = @as(u64, left) << 32 | right;
        var i = m.slot(key);
        while (m.keys[i] != empty) : (i = (i + 1) & m.mask) if (m.keys[i] == key) return; // the first rank wins
        m.keys[i] = key;
        m.vals[i] = @as(u64, rank) << 32 | id;
    }

    inline fn get(m: *const Merges, left: Id, right: Id) ?u64 {
        const key = @as(u64, left) << 32 | right;
        var i = m.slot(key);
        while (true) : (i = (i + 1) & m.mask) {
            const k = m.keys[i];
            if (k == key) return m.vals[i];
            if (k == empty) return null;
        }
    }
};

/// Added tokens as a byte trie, matched leftmost-longest (``tokenizers``' Aho-Corasick split).
const Trie = struct {
    first: [256]u32 = @splat(0), // node after the first byte, 0: none
    edges: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    ends: std.ArrayList(?Id) = .empty, // node 0 is the root

    fn insert(t: *Trie, a: Allocator, s: []const u8, id: Id) Allocator.Error!void {
        if (s.len == 0) return;
        if (t.ends.items.len == 0) try t.ends.append(a, null);
        var node: u32 = 0;
        for (s, 0..) |b, i| {
            const slot = if (i == 0) &t.first[b] else (try t.edges.getOrPutValue(a, @as(u64, node) << 8 | b, 0)).value_ptr;
            if (slot.* == 0) {
                try t.ends.append(a, null);
                slot.* = @intCast(t.ends.items.len - 1);
            }
            node = slot.*;
        }
        if (t.ends.items[node] == null) t.ends.items[node] = id;
    }

    const Hit = struct { start: usize, end: usize, id: Id };

    /// The leftmost match at or after ``from``, longest at its start.
    fn next(t: *const Trie, s: []const u8, from: usize) ?Hit {
        var i = from;
        while (i < s.len) : (i += 1) {
            var node = t.first[s[i]];
            if (node == 0) continue;
            var best: ?Hit = null;
            var j = i + 1;
            while (true) {
                if (t.ends.items[node]) |id| best = .{ .start = i, .end = j, .id = id };
                if (j == s.len) break;
                node = t.edges.get(@as(u64, node) << 8 | s[j]) orelse break;
                j += 1;
            }
            if (best) |b| return b;
        }
        return null;
    }
};

pub const Tokenizer = struct {
    arena: std.heap.ArenaAllocator,
    /// Every id's decoded bytes: ``bytes[offsets[id]..offsets[id + 1]]``.
    bytes: []const u8,
    offsets: []const u32,
    byte_id: [256]Id,
    merges: Merges,
    raw: Trie = .{}, // added tokens matched on the text as given
    normalized: Trie = .{}, // added tokens matched after the (empty) normalizer
    special: []const bool, // added tokens flagged special, by id
    /// Token strings as tokenizer.json spells them (``convert_ids_to_tokens``), by id.
    strings: []const []const u8,
    by_string: std.StringHashMapUnmanaged(Id) = .empty,
    classes: []const u8, // classOf for code points below table_end
    pool_lock: std.atomic.Value(bool) = .init(false), // a spin lock: the pool is touched once a request
    pool: std.ArrayList(*Encoder) = .empty,
    gpa: Allocator,

    pub fn vocabSize(t: *const Tokenizer) u32 {
        return @intCast(t.offsets.len - 1);
    }

    /// Loads ``dir/tokenizer.json`` (or the file itself when ``path`` ends in .json).
    pub fn load(gpa: Allocator, io: std.Io, path: []const u8) !*Tokenizer {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const s = scratch.allocator();
        const file = if (std.mem.endsWith(u8, path, ".json")) path else try std.fs.path.join(s, &.{ path, "tokenizer.json" });
        const text = try std.Io.Dir.cwd().readFileAlloc(io, file, s, .limited(1 << 30));
        return parse(gpa, text);
    }

    pub fn parse(gpa: Allocator, json_text: []const u8) Error!*Tokenizer {
        var scratch = std.heap.ArenaAllocator.init(gpa);
        defer scratch.deinit();
        const s = scratch.allocator();
        const doc = std.json.parseFromSliceLeaky(std.json.Value, s, json_text, .{}) catch return error.InvalidTokenizer;
        if (doc != .object) return error.InvalidTokenizer;
        const model = doc.object.get("model") orelse return error.InvalidTokenizer;
        if (model != .object) return error.InvalidTokenizer;
        const vocab = model.object.get("vocab") orelse return error.InvalidTokenizer;
        const merges = model.object.get("merges") orelse return error.InvalidTokenizer;
        if (vocab != .object or merges != .array) return error.InvalidTokenizer;
        const added = doc.object.get("added_tokens") orelse std.json.Value{ .array = .init(s) };
        if (added != .array) return error.InvalidTokenizer;

        const t = try gpa.create(Tokenizer);
        errdefer gpa.destroy(t);
        t.* = .{ .arena = .init(gpa), .bytes = &.{}, .offsets = &.{}, .byte_id = undefined, .merges = undefined, .special = &.{}, .strings = &.{}, .classes = &.{}, .gpa = gpa };
        errdefer t.arena.deinit();
        const a = t.arena.allocator();

        var size: usize = 0;
        var it = vocab.object.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.* != .integer or e.value_ptr.integer < 0) return error.InvalidTokenizer;
            size = @max(size, @as(usize, @intCast(e.value_ptr.integer)) + 1);
        }
        for (added.array.items) |x| {
            const id = (x.object.get("id") orelse return error.InvalidTokenizer).integer;
            size = @max(size, @as(usize, @intCast(id)) + 1);
        }
        const strings = try a.alloc([]const u8, size);
        @memset(strings, "");
        it = vocab.object.iterator();
        try t.by_string.ensureTotalCapacity(a, @intCast(vocab.object.count() + added.array.items.len));
        while (it.next()) |e| {
            const piece = try a.dupe(u8, e.key_ptr.*);
            strings[@intCast(e.value_ptr.integer)] = piece;
            t.by_string.putAssumeCapacity(piece, @intCast(e.value_ptr.integer));
        }
        const special = try a.alloc(bool, size);
        @memset(special, false);
        for (added.array.items) |x| {
            const content = (x.object.get("content") orelse return error.InvalidTokenizer).string;
            const id: Id = @intCast(x.object.get("id").?.integer);
            const is_special = if (x.object.get("special")) |v| v == .bool and v.bool else false;
            const norm = if (x.object.get("normalized")) |v| v == .bool and v.bool else !is_special;
            if (x.object.get("single_word")) |v| if (v == .bool and v.bool) return error.InvalidTokenizer;
            const piece = try a.dupe(u8, content);
            strings[id] = piece;
            t.by_string.putAssumeCapacity(piece, id);
            special[id] = is_special;
            try (if (norm) &t.normalized else &t.raw).insert(a, piece, id);
        }
        t.strings = strings;
        t.special = special;

        // decoded bytes: a token whose characters are all in the byte alphabet maps through it, else its own UTF-8
        var char_byte: [0x144]i16 = @splat(-1);
        for (byte_chars, 0..) |cp, b| char_byte[cp] = @intCast(b);
        const offsets = try a.alloc(u32, size + 1);
        var out: std.ArrayList(u8) = .empty;
        for (strings, 0..) |piece, id| {
            offsets[id] = @intCast(out.items.len);
            const start = out.items.len;
            var mapped = std.unicode.Utf8View.init(piece) catch return error.InvalidTokenizer;
            var cps = mapped.iterator();
            var ok = true;
            while (cps.nextCodepoint()) |cp| {
                if (cp >= char_byte.len or char_byte[cp] < 0) {
                    ok = false;
                    break;
                }
                try out.append(a, @intCast(char_byte[cp]));
            }
            if (!ok) {
                out.shrinkRetainingCapacity(start);
                try out.appendSlice(a, piece);
            }
        }
        offsets[size] = @intCast(out.items.len);
        t.bytes = out.items;
        t.offsets = offsets;

        var buf: [4]u8 = undefined;
        for (byte_chars, 0..) |cp, b| {
            const n = std.unicode.utf8Encode(cp, &buf) catch unreachable;
            t.byte_id[b] = t.by_string.get(buf[0..n]) orelse missing; // a byte the vocabulary lacks is dropped, as BPE without unk does
        }

        var cap: u64 = 16;
        while (cap < merges.array.items.len * 2) cap *= 2;
        t.merges = .{ .keys = try a.alloc(u64, cap), .vals = try a.alloc(u64, cap), .mask = cap - 1 };
        @memset(t.merges.keys, Merges.empty);
        var joined: std.ArrayList(u8) = .empty;
        for (merges.array.items, 0..) |m, rank| {
            var left: []const u8 = undefined;
            var right: []const u8 = undefined;
            switch (m) {
                .string => |pair| {
                    const sp = std.mem.indexOfScalar(u8, pair, ' ') orelse return error.InvalidTokenizer;
                    left = pair[0..sp];
                    right = pair[sp + 1 ..];
                },
                .array => |pair| {
                    if (pair.items.len != 2 or pair.items[0] != .string or pair.items[1] != .string) return error.InvalidTokenizer;
                    left = pair.items[0].string;
                    right = pair.items[1].string;
                },
                else => return error.InvalidTokenizer,
            }
            joined.clearRetainingCapacity();
            try joined.appendSlice(s, left);
            try joined.appendSlice(s, right);
            const l = t.by_string.get(left) orelse return error.InvalidTokenizer;
            const r = t.by_string.get(right) orelse return error.InvalidTokenizer;
            const id = t.by_string.get(joined.items) orelse return error.InvalidTokenizer;
            t.merges.put(l, r, @intCast(rank), id);
        }

        const classes = try a.alloc(u8, table_end);
        for (classes, 0..) |*c, cp| c.* = if (cp >= 0xD800 and cp <= 0xDFFF) 0 else classOf(@intCast(cp));
        t.classes = classes;
        return t;
    }

    pub fn deinit(t: *Tokenizer) void {
        for (t.pool.items) |e| e.destroy();
        t.pool.deinit(t.gpa);
        const gpa = t.gpa;
        t.arena.deinit();
        gpa.destroy(t);
    }

    /// An id's bytes as the ByteLevel decoder gives them (possibly part of a character).
    pub inline fn tokenBytes(t: *const Tokenizer, id: Id) []const u8 {
        if (id >= t.offsets.len - 1) return "";
        return t.bytes[t.offsets[id]..t.offsets[id + 1]];
    }

    /// The id of a whole token string (``convert_tokens_to_ids``): a special or added token's content included.
    pub fn tokenId(t: *const Tokenizer, piece: []const u8) ?Id {
        return t.by_string.get(piece);
    }

    pub fn tokenString(t: *const Tokenizer, id: Id) []const u8 {
        return if (id < t.strings.len) t.strings[id] else "";
    }

    pub fn isSpecial(t: *const Tokenizer, id: Id) bool {
        return id < t.special.len and t.special[id];
    }

    inline fn class(t: *const Tokenizer, cp: u21) u8 {
        return if (cp < table_end) t.classes[cp] else classOf(cp);
    }

    /// An encoder from the shared pool (``release`` gives it back); one thread uses it at a time.
    pub fn acquire(t: *Tokenizer) Allocator.Error!*Encoder {
        {
            t.lock();
            defer t.unlock();
            if (t.pool.pop()) |e| return e;
        }
        return Encoder.create(t.gpa, t);
    }

    pub fn release(t: *Tokenizer, e: *Encoder) void {
        t.lock();
        defer t.unlock();
        t.pool.append(t.gpa, e) catch e.destroy();
    }

    fn lock(t: *Tokenizer) void {
        while (t.pool_lock.swap(true, .acquire)) std.atomic.spinLoopHint();
    }

    fn unlock(t: *Tokenizer) void {
        t.pool_lock.store(false, .release);
    }

    /// ``encode(text)`` into a new slice of ``a`` (through a pooled encoder).
    pub fn encodeAlloc(t: *Tokenizer, a: Allocator, text: []const u8) Allocator.Error![]Id {
        const e = try t.acquire();
        defer t.release(e);
        var out: std.ArrayList(Id) = .empty;
        errdefer out.deinit(a);
        try e.encode(a, text, &out);
        return out.toOwnedSlice(a);
    }

    /// ``decode(ids, skip_special_tokens=False)``: every id's bytes, then invalid UTF-8 as U+FFFD (from_utf8_lossy).
    pub fn decodeAlloc(t: *const Tokenizer, a: Allocator, ids: []const Id) Allocator.Error![]u8 {
        var d: Detokenizer = .{ .tok = t };
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(a);
        for (ids) |id| try d.push(a, id, &out);
        try d.flush(a, &out);
        return out.toOwnedSlice(a);
    }
};

/// Text from ids a token at a time, equal to decoding them all at once: a character split across tokens waits for
/// its last byte; each maximal invalid subpart becomes U+FFFD (Rust's from_utf8_lossy, Python's ``replace``).
pub const Detokenizer = struct {
    tok: *const Tokenizer,
    held: [4]u8 = undefined,
    n: u8 = 0,

    /// Appends the text ``id`` completes to ``out``.
    pub fn push(d: *Detokenizer, a: Allocator, id: Id, out: *std.ArrayList(u8)) Allocator.Error!void {
        const b = d.tok.tokenBytes(id);
        if (b.len == 0) return;
        if (d.n == 0 and std.unicode.utf8ValidateSlice(b)) return out.appendSlice(a, b); // the common case
        for (b) |x| try d.byte(a, x, out);
    }

    /// Whether a character is waiting for its next bytes.
    pub fn pending(d: *const Detokenizer) bool {
        return d.n > 0;
    }

    fn byte(d: *Detokenizer, a: Allocator, x: u8, out: *std.ArrayList(u8)) Allocator.Error!void {
        d.held[d.n] = x;
        d.n += 1;
        while (d.n > 0) {
            const got = scanPrefix(d.held[0..d.n]);
            switch (got) {
                .incomplete => return,
                .char => |len| {
                    try out.appendSlice(a, d.held[0..len]);
                    d.drop(len);
                },
                .invalid => |len| {
                    try out.appendSlice(a, "\u{FFFD}");
                    d.drop(len);
                },
            }
        }
    }

    fn drop(d: *Detokenizer, len: usize) void {
        std.mem.copyForwards(u8, d.held[0 .. d.n - len], d.held[len..d.n]);
        d.n -= @intCast(len);
    }

    /// The end of the stream: a character still incomplete is one U+FFFD.
    pub fn flush(d: *Detokenizer, a: Allocator, out: *std.ArrayList(u8)) Allocator.Error!void {
        if (d.n > 0) try out.appendSlice(a, "\u{FFFD}");
        d.n = 0;
    }
};

const Scan = union(enum) { char: usize, invalid: usize, incomplete };

/// The sequence at the start of ``s``: a whole character, an invalid maximal subpart, or a prefix still open.
fn scanPrefix(s: []const u8) Scan {
    const b = s[0];
    if (b < 0x80) return .{ .char = 1 };
    var lo: u8 = 0x80;
    var hi: u8 = 0xBF;
    const need: usize = switch (b) {
        0xC2...0xDF => 1,
        0xE0...0xEF => 2,
        0xF0...0xF4 => 3,
        else => return .{ .invalid = 1 },
    };
    if (b == 0xE0) lo = 0xA0;
    if (b == 0xED) hi = 0x9F;
    if (b == 0xF0) lo = 0x90;
    if (b == 0xF4) hi = 0x8F;
    for (1..need + 1) |j| {
        if (j >= s.len) return .incomplete;
        if (s[j] < lo or s[j] > hi) return .{ .invalid = j };
        lo = 0x80;
        hi = 0xBF;
    }
    return .{ .char = need + 1 };
}

/// One thread's encoding state: BPE scratch and a direct-mapped cache of short pieces' ids.
pub const Encoder = struct {
    gpa: Allocator,
    tok: *const Tokenizer,
    cache: []Slot,
    sym: std.ArrayList(Id) = .empty,
    prev: std.ArrayList(u32) = .empty,
    next: std.ArrayList(u32) = .empty,
    heap: std.ArrayList(Pending) = .empty,
    lossy: std.ArrayList(u8) = .empty,

    const cache_slots = 1 << 15;
    const key_max = 22;
    const ids_max = 8;
    const none: u32 = std.math.maxInt(u32);

    const Slot = struct {
        key: [key_max]u8 = undefined,
        key_len: u8 = 0xFF, // 0xFF: empty
        n: u8 = 0,
        ids: [ids_max]Id = undefined,
    };

    /// A merge waiting in the heap: ordered by rank, then position (``tokenizers``' ``Merge``).
    const Pending = struct { key: u64, id: Id };

    pub fn create(gpa: Allocator, tok: *const Tokenizer) Allocator.Error!*Encoder {
        const e = try gpa.create(Encoder);
        errdefer gpa.destroy(e);
        const cache = try gpa.alloc(Slot, cache_slots);
        @memset(cache, .{});
        e.* = .{ .gpa = gpa, .tok = tok, .cache = cache };
        return e;
    }

    pub fn destroy(e: *Encoder) void {
        const gpa = e.gpa;
        gpa.free(e.cache);
        e.sym.deinit(gpa);
        e.prev.deinit(gpa);
        e.next.deinit(gpa);
        e.heap.deinit(gpa);
        e.lossy.deinit(gpa);
        gpa.destroy(e);
    }

    /// Appends the ids of ``text`` to ``out`` (``a`` grows ``out``). Invalid UTF-8 reads as U+FFFD.
    pub fn encode(e: *Encoder, a: Allocator, text: []const u8, out: *std.ArrayList(Id)) Allocator.Error!void {
        var valid = text;
        if (!std.unicode.utf8ValidateSlice(text)) {
            e.lossy.clearRetainingCapacity();
            try unicode.appendLossy(&e.lossy, e.gpa, text);
            valid = e.lossy.items;
        }
        var done: usize = 0;
        var at: usize = 0;
        while (e.tok.raw.next(valid, at)) |hit| {
            try e.normalizedPass(a, valid[done..hit.start], out);
            try out.append(a, hit.id);
            done = hit.end;
            at = hit.end;
        }
        try e.normalizedPass(a, valid[done..], out);
    }

    fn normalizedPass(e: *Encoder, a: Allocator, s: []const u8, out: *std.ArrayList(Id)) Allocator.Error!void {
        var done: usize = 0;
        while (e.tok.normalized.next(s, done)) |hit| {
            try e.digits(a, s[done..hit.start], out);
            try out.append(a, hit.id);
            done = hit.end;
        }
        try e.digits(a, s[done..], out);
    }

    inline fn cpAt(s: []const u8, i: usize) unicode.Decoded {
        return unicode.decodeAt(s, i);
    }

    /// Split 1: ``\p{N}{1,3}`` isolated.
    fn digits(e: *Encoder, a: Allocator, s: []const u8, out: *std.ArrayList(Id)) Allocator.Error!void {
        const t = e.tok;
        var gap: usize = 0;
        var i: usize = 0;
        while (i < s.len) {
            const d = cpAt(s, i);
            if (t.class(d.cp) & N == 0) {
                i += d.len;
                continue;
            }
            try e.cjk(a, s[gap..i], out);
            var j = i;
            var count: usize = 0;
            while (j < s.len and count < 3) {
                const x = cpAt(s, j);
                if (t.class(x.cp) & N == 0) break;
                j += x.len;
                count += 1;
            }
            try e.piece(a, s[i..j], out); // a run of numbers has no further split (the word regex matches none of it)
            i = j;
            gap = j;
        }
        try e.cjk(a, s[gap..], out);
    }

    /// Split 2: the CJK and kana runs isolated.
    fn cjk(e: *Encoder, a: Allocator, s: []const u8, out: *std.ArrayList(Id)) Allocator.Error!void {
        if (s.len == 0) return;
        const t = e.tok;
        var gap: usize = 0;
        var i: usize = 0;
        while (i < s.len) {
            if (s[i] < 0xE3 or s[i] > 0xE9) { // every CJK and kana code point here is 3 bytes from E3 to E9
                i += unicode.decodeAt(s, i).len;
                continue;
            }
            const d = cpAt(s, i);
            if (t.class(d.cp) & CJK == 0) {
                i += d.len;
                continue;
            }
            try e.words(a, s[gap..i], out);
            var j = i + d.len;
            while (j < s.len) {
                const x = cpAt(s, j);
                if (t.class(x.cp) & CJK == 0) break;
                j += x.len;
            }
            try e.words(a, s[i..j], out);
            i = j;
            gap = j;
        }
        try e.words(a, s[gap..], out);
    }

    /// Split 3: the word regex, leftmost-first, as matches and the gaps between them.
    fn words(e: *Encoder, a: Allocator, s: []const u8, out: *std.ArrayList(Id)) Allocator.Error!void {
        if (s.len == 0) return;
        const t = e.tok;
        var gap: usize = 0;
        var i: usize = 0;
        while (i < s.len) {
            const end = matchAt(t, s, i) orelse {
                i += unicode.decodeAt(s, i).len;
                continue;
            };
            if (gap < i) try e.piece(a, s[gap..i], out);
            try e.piece(a, s[i..end], out);
            i = end;
            gap = end;
        }
        if (gap < s.len) try e.piece(a, s[gap..], out);
    }

    /// The end of the word regex's match starting at ``i``, or null.
    fn matchAt(t: *const Tokenizer, s: []const u8, i: usize) ?usize {
        const c0 = cpAt(s, i);
        const f0 = t.class(c0.cp);
        const j = i + c0.len;
        const c1: ?unicode.Decoded = if (j < s.len) cpAt(s, j) else null;
        const f1: u8 = if (c1) |c| t.class(c.cp) else 0;
        // [!-/:-@[-`{-~][A-Za-z]+
        if (c0.cp < 0x80 and std.ascii.isPrint(@intCast(c0.cp)) and !std.ascii.isAlphanumeric(@intCast(c0.cp)) and c0.cp != ' ') {
            if (c1 != null and c1.?.cp < 0x80 and std.ascii.isAlphabetic(@intCast(c1.?.cp))) {
                var k = j + 1;
                while (k < s.len and std.ascii.isAlphabetic(s[k])) k += 1;
                return k;
            }
        }
        // [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
        if (f0 & (CRLF | L | P | S) == 0 and f1 & (L | M) != 0) return runOf(t, s, j + c1.?.len, L | M);
        if (f0 & (L | M) != 0) return runOf(t, s, j, L | M);
        // ' '?[\p{P}\p{S}]+[\r\n]*
        if (c0.cp == ' ' and f1 & (P | S) != 0) return runOf(t, s, runOf(t, s, j + c1.?.len, P | S), CRLF);
        if (f0 & (P | S) != 0) return runOf(t, s, runOf(t, s, j, P | S), CRLF);
        if (f0 & WS == 0) return null;
        // \s*[\r\n]+ | \s+(?!\S) | \s+
        var last_crlf: ?usize = if (f0 & CRLF != 0) i else null;
        var last_start = i;
        var k = j;
        while (k < s.len) {
            const x = cpAt(s, k);
            const fx = t.class(x.cp);
            if (fx & WS == 0) break;
            if (fx & CRLF != 0) last_crlf = k;
            last_start = k;
            k += x.len;
        }
        if (last_crlf) |p| return p + 1;
        if (k == s.len) return k;
        if (last_start > i) return last_start; // give the last space back to the next word
        return k;
    }

    inline fn runOf(t: *const Tokenizer, s: []const u8, from: usize, mask: u8) usize {
        var k = from;
        while (k < s.len) {
            const x = cpAt(s, k);
            if (t.class(x.cp) & mask == 0) break;
            k += x.len;
        }
        return k;
    }

    fn hashOf(s: []const u8) u64 {
        return std.hash.Wyhash.hash(0x6473763431, s);
    }

    /// A piece's BPE ids, through the cache.
    fn piece(e: *Encoder, a: Allocator, s: []const u8, out: *std.ArrayList(Id)) Allocator.Error!void {
        if (s.len == 1) return if (e.tok.byte_id[s[0]] != missing) out.append(a, e.tok.byte_id[s[0]]);
        if (s.len <= key_max) {
            const slot = &e.cache[@intCast(hashOf(s) & (cache_slots - 1))];
            if (slot.key_len == s.len and std.mem.eql(u8, slot.key[0..s.len], s)) return out.appendSlice(a, slot.ids[0..slot.n]);
            const before = out.items.len;
            try e.bpe(a, s, out);
            const got = out.items[before..];
            if (got.len <= ids_max) {
                @memcpy(slot.key[0..s.len], s);
                slot.key_len = @intCast(s.len);
                slot.n = @intCast(got.len);
                @memcpy(slot.ids[0..got.len], got);
            }
            return;
        }
        return e.bpe(a, s, out);
    }

    /// ``tokenizers``' ``Word::merge_all``: a min-heap of (rank, position) merges, stale ones skipped.
    fn bpe(e: *Encoder, a: Allocator, s: []const u8, out: *std.ArrayList(Id)) Allocator.Error!void {
        const gpa = e.gpa;
        const n = s.len;
        const m = &e.tok.merges;
        try e.sym.resize(gpa, n);
        try e.prev.resize(gpa, n);
        try e.next.resize(gpa, n);
        const sym = e.sym.items;
        const prev = e.prev.items;
        const next = e.next.items;
        for (s, 0..) |b, i| {
            sym[i] = e.tok.byte_id[b];
            prev[i] = if (i == 0) none else @intCast(i - 1);
            next[i] = if (i + 1 == n) none else @intCast(i + 1);
        }
        e.heap.clearRetainingCapacity();
        for (0..n - 1) |i| if (m.get(sym[i], sym[i + 1])) |v| try e.push(.{ .key = (v >> 32) << 32 | i, .id = @truncate(v) });
        while (e.pop()) |top| {
            const pos: u32 = @truncate(top.key);
            if (sym[pos] == none) continue; // merged into its left neighbour
            const right = next[pos];
            if (right == none) continue;
            const v = m.get(sym[pos], sym[right]) orelse continue;
            if (@as(Id, @truncate(v)) != top.id) continue; // the pair changed since this merge was queued
            sym[pos] = top.id;
            sym[right] = none;
            next[pos] = next[right];
            if (next[right] != none) prev[next[right]] = pos;
            if (prev[pos] != none) if (m.get(sym[prev[pos]], sym[pos])) |pv| try e.push(.{ .key = (pv >> 32) << 32 | prev[pos], .id = @truncate(pv) });
            if (next[pos] != none) if (m.get(sym[pos], sym[next[pos]])) |nv| try e.push(.{ .key = (nv >> 32) << 32 | pos, .id = @truncate(nv) });
        }
        var i: u32 = 0;
        while (i != none) : (i = next[i]) if (sym[i] != missing) try out.append(a, sym[i]);
    }

    fn push(e: *Encoder, p: Pending) Allocator.Error!void {
        try e.heap.append(e.gpa, p);
        const h = e.heap.items;
        var i = h.len - 1;
        while (i > 0) {
            const parent = (i - 1) / 2;
            if (h[parent].key <= p.key) break;
            h[i] = h[parent];
            i = parent;
        }
        h[i] = p;
    }

    fn pop(e: *Encoder) ?Pending {
        const h = e.heap.items;
        if (h.len == 0) return null;
        const top = h[0];
        const last = h[h.len - 1];
        e.heap.items.len -= 1;
        const n = h.len - 1;
        if (n == 0) return top;
        var i: usize = 0;
        while (true) {
            const l = 2 * i + 1;
            if (l >= n) break;
            const c = if (l + 1 < n and h[l + 1].key < h[l].key) l + 1 else l;
            if (h[c].key >= last.key) break;
            h[i] = h[c];
            i = c;
        }
        h[i] = last;
        return top;
    }
};

test "byte-level BPE with merges in rank order, added tokens in two passes, and decode" {
    const json =
        \\{"added_tokens":[{"id":9,"content":"<s>","special":true,"normalized":false},{"id":10,"content":"<u>","special":false,"normalized":true}],
        \\ "model":{"type":"BPE","vocab":{"a":0,"b":1,"Ġ":2,"ab":3,"Ġa":4,"Ġab":5,"1":6,"2":7,"12":8},"merges":["Ġ a","a b","Ġa b","1 2"]}}
    ;
    const a = std.testing.allocator;
    const t = try Tokenizer.parse(a, json);
    defer t.deinit();
    const ids = try t.encodeAlloc(a, "ab ab<s>121<u>");
    defer a.free(ids);
    try std.testing.expectEqualSlices(Id, &.{ 3, 5, 9, 8, 6, 10 }, ids);
    const text = try t.decodeAlloc(a, ids);
    defer a.free(text);
    try std.testing.expectEqualStrings("ab ab<s>121<u>", text);
}

test "the detokenizer holds a split character and replaces invalid bytes as from_utf8_lossy does" {
    var d: Detokenizer = .{ .tok = undefined };
    const a = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    for ("\xe4\xbd") |b| try d.byte(a, b, &out);
    try std.testing.expect(d.pending() and out.items.len == 0);
    try d.byte(a, 0xa0, &out);
    try std.testing.expectEqualStrings("你", out.items);
    for ("\xe4x\xff\xf0\x9f") |b| try d.byte(a, b, &out);
    try d.flush(a, &out);
    try std.testing.expectEqualStrings("你\u{FFFD}x\u{FFFD}\u{FFFD}", out.items);
}
