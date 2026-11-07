//! Inert types only, to compile unmodified native request-side tools/arming functions. Not a dispatch test.
pub const compute_lane = struct { pub const Config = struct {}; pub const Pool = struct {}; };
pub const Request = struct { eos: []const u32, max_tokens: u32, background: bool = false, structure: ?u8 = null, call: ?u8 = null, logprobs: ?u8 = null, prompt_logprobs: ?u8 = null };
