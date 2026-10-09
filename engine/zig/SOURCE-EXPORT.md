# Source export provenance

This is a sanitized source snapshot based on TensorFold 1.0.2, upstream revision
`f8fe17d24629aedabf90bbf78279dd776e6d62e7`, from
[ashhart/TensorFold](https://github.com/ashhart/TensorFold). The DeepSeek-V4.1 source snapshot revision is
`0d723a8275f5b7879a083e8042a7048242ee624c`.

The source derives from the public upstream tree and is added through logical source commits. Development
history is not imported. Source changes include removal of infrastructure references, conversion of
machine-specific benchmark configuration to environment inputs, and default-off controls for the new
family's drafting, native prefill, lookup, session retention, and Engram prefetch.

Apache-2.0 license headers, attribution, `LICENSE`, `NOTICE`, third-party notices, and applicable
third-party license texts are retained. Weight and dataset licenses remain separate.

Excluded material includes performance diaries and infrastructure notes; private benchmark results
and logs; profiler captures; databases; packaged kits; caches; scratch files; generated build outputs;
and large generated binary goldens. Small checked-in source fixtures remain; no huge binary fixtures are included. The source tools generate
checkpoint-dependent fixtures and AOT assets locally; the public serving recipe documents asset
generation. No checkpoint weights or generated AOT bundle are shipped in this source export.

The export is intended to make the source inspectable and buildable. Historical device receipts are
identified separately from checks on the exported tree. Host compilation does not qualify hardware,
models, arithmetic, or precision choices. See [the CUDA guide](docs/DEEPSEEK-V41-CUDA.md) and the pull
request for build/test outcomes, unrun gates, and the outstanding upstream lean-rule cleanup.
