# CUDA DeepSeek-V4.1 engine

This directory contains a sanitized, buildable TensorFold 1.0.2 source export with the CUDA DeepSeek-V4.1
family, kernels, TP transport, drafting, shared server integration and host tests.

Use Zig 0.17.0 from `.zig-version`:

```sh
zig build
zig build test
```

These default builds check the host code and embed empty GPU images. Serving requires compiled CUDA
fatbins and generated Triton, RoPE and Engram assets. Follow the [TP=2 serving quick start](../../docs/ZIG-SERVE.md)
for GPU builds, asset generation, model packing and startup on two DGX Sparks.

Read the [CUDA guide](docs/DEEPSEEK-V41-CUDA.md) for gates and measured performance, and
[source provenance](SOURCE-EXPORT.md) for the snapshot, export changes and exclusions.
The origin is [TensorFold](https://github.com/ashhart/TensorFold), licensed under
[Apache-2.0](LICENSE); `NOTICE`, `LICENSES/` and `THIRD_PARTY_NOTICES.md` retain its notices.
