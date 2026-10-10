# Engines and reference code

The buildable CUDA Zig engine lives in [`zig/`](zig), with [source provenance](zig/SOURCE-EXPORT.md) and
[build/run instructions](../docs/ZIG-SERVE.md). It is independent of the Python patch application below.

The remaining directories hold the code as it was written and tested before it was ported into the TensorFold family (`patches/0002`):
`reference/` is the pure-PyTorch reference model and checkpoint loader used as the correctness oracle, `kernels/`
and `serving/` the first versions of the kernels and the serving layer (`scripts/campaign/port_serving.py` copied the
latter into the engine). The tests are in `../tests/`. The Python engine that runs is the patched TensorFold; these directories are its earlier reference/staging code.
