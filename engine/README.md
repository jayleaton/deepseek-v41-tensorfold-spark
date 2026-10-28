# Native engine and build-time references

The existing native CUDA Zig source is in [`zig/`](zig), with [source provenance](zig/SOURCE-EXPORT.md) and
[serving instructions](../docs/ZIG-SERVE.md). It already runs independently of the retired Python patch delivery.

`kernels/`, `reference/` and the Python tools under `zig/tools/` remain as kernel generators, asset tools and
independent references. They are retained pending a separate upstream-equivalence audit; their language does not
make them inference-server code. Their applicable CPU checks remain under `../tests/`.

The staged Python serving layer and its coupled family/server tests were removed with the old Python engine path.
Historical source and results remain available in public release 60b6688.
