//! The backend-neutral checkpoint reader alone, as `core` for a module that also lives beside the server's tokenizer
//! (tensorfold-dsv41's engine: DeepSeek-V4.1's pack reads safetensors headers through `core.safetensors`).

pub const safetensors = @import("safetensors.zig");
