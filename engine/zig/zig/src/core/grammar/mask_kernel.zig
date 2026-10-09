//! The grammar mask's device kernel, compiled to PTX by Zig's NVPTX backend (zig/build/grammar.zig) and JIT-loaded
//! by the driver: grid (ceil(span / 128), rows), one thread a 32-token word of one row (rows.zig's `maskWord`).

const rows = @import("rows.zig");

/// logits rows idx[r] (stride ld floats) under bits rows r (`words` int32 a row, this rank's words from w0, n of them),
/// `width` columns.
export fn tf_grammar_mask(logits: [*]addrspace(.global) f32, ld: u64, idx: [*]addrspace(.global) const u32, bits: [*]addrspace(.global) const u32, words: u32, w0: u32, n: u32, width: u32) callconv(.kernel) void {
    const r = @workGroupId(1);
    const k = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    if (k >= n) return;
    const word = bits[@as(u64, r) * words + w0 + k];
    const base: [*]f32 = @addrSpaceCast(logits + @as(u64, idx[r]) * ld);
    rows.maskWord(base, word, 32 * k, width);
}
