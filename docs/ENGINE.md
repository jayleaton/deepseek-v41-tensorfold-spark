# The native engine

The current recipe already serves through the CUDA Zig engine in [`engine/zig`](../engine/zig).
See its [source provenance](../engine/zig/SOURCE-EXPORT.md) and [native serving guide](ZIG-SERVE.md).
The former Python v0.6.0 patch delivery and serving image have been removed.

The native source export, Python kernel/asset generators under `engine/zig/tools`, `engine/kernels` and independent
reference code remain unchanged pending an equivalence audit against upstream. Their presence does not select a
Python inference server. RoPE/Triton AOT still have build-time reference dependencies; those generators are retained
until a proved equivalent exists, rather than deleted merely because their language is Python.

Historical source, patch provenance and results remain at public commit
[60b6688](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/tree/60b66888a1a379fb38bf85c232b27ec60b007373).
The `results/` tree is unchanged. Credits and third-party notices remain available with that release and the retained
source. Future upstream wiring is a separate change; this cleanup does not claim a new engine qualification.
