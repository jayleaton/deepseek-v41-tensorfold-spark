//! DeepSeek-V4.1's image input, host side (``TF_DSV41_IMAGES=native``; Python: ``vision_prep.py`` and GLM's front
//! end it imports): image parts and data: URLs, Pillow-exact PNG / JPEG decoding, the processor, virtual ids, and
//! rank 0's prepared-image cache. The engine side (the tower on rank 0, the image rows, ``bias_vl``, Engram) reads
//! ``prep.Prepared`` and the ids.
pub const pixels = @import("pixels.zig");
pub const png = @import("png.zig");
pub const jpeg = @import("jpeg.zig");
pub const gif = @import("gif.zig");
pub const bmp = @import("bmp.zig");
pub const decode = @import("decode.zig");
pub const resample = @import("resample.zig");
pub const pyjson = @import("pyjson.zig");
pub const prep = @import("prep.zig");
pub const vids = @import("vids.zig");
pub const parts = @import("parts.zig");
pub const host = @import("host.zig");
pub const fetch = @import("fetch.zig");
pub const mode = @import("mode.zig");
pub const held = @import("held.zig");

test {
    @import("std").testing.refAllDecls(@This());
    _ = @import("jpeg_pix.zig");
    _ = @import("golden_test.zig");
    _ = @import("fetch_test.zig");
}
