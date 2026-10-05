# The development log (G1-G19)

The documents written while the DeepSeek-V4.1-Flash family was built and measured, 2026-10-01 to 10-05, kept as
they were except for machine names, link addresses, local paths and a few process words, which were replaced. They
describe the work in the order it happened, so later windows correct earlier plans; the summaries in
[`../RESULTS.md`](../RESULTS.md), [`../DECODE.md`](../DECODE.md) and [`../ARCHITECTURE.md`](../ARCHITECTURE.md) are
the current picture. Raw files: [`../../results/campaign/`](../../results/campaign/). Paths such as
`scripts/windows/*.sh` and `config/prod.env` refer to the development harness, which is not published (it managed
the private test windows and the switch with another production service); `scripts/serve.sh` and
`config/prod.env.example` are its public form.

| document | what |
| --- | --- |
| [LANDSCAPE.md](LANDSCAPE.md) | what others ran and measured on this model (1-4 Sparks and other hardware), quality vs bits |
| [ARCH-LEVERAGE.md](ARCH-LEVERAGE.md) | what DeepSeek built into V4.1 (CED, Engram, CSA2, DSpark, MoE, mHC) and how to use it on two Sparks |
| [ARCHITECTURE.md](ARCHITECTURE.md) | the model layer by layer, the TP=2 split, the byte budget |
| [TARGETS.md](TARGETS.md), [BASELINE-RESULTS.md](BASELINE-RESULTS.md) | the targets and the measured kit baseline |
| [ENGINE-PLAN.md](ENGINE-PLAN.md), [M1-STATUS.md](M1-STATUS.md) | the engine plan and its status through the windows |
| [G1-RESULTS.md](G1-RESULTS.md) ... [G13-RESULTS.md](G13-RESULTS.md) | each test window's results (G12: the host-memory growth, the stall and the segmentation dependence, diagnosed and fixed; G13: five decode rewrites, three adopted, and the strict-mode receipt) |
| [G14-RESULTS.md](G14-RESULTS.md), [G14a-RESULTS.md](G14a-RESULTS.md) | G14: the window levers (plan link over RoCE, BRANCHES on priority streams, paced L2 prefetch, the faster RoCE kernel) and calibration VERSION 3; G14a: the per-kernel timeline and the RoCE exchange split |
| [G15a-RESULTS.md](G15a-RESULTS.md), [G15b-RESULTS.md](G15b-RESULTS.md), [VISION.md](VISION.md) | image input: the placeholder mode, native vision's failure and fix, the vision design and battery; calibration VERSION 4, joint depth mode 2, the expert map |
| [G16-RELEASE.md](G16-RELEASE.md), [G16-CHUNK4K.md](G16-CHUNK4K.md) | native images in production, where the memory goes in the worst case, the memory cap, the vision store bug and the graph-cache leak; the 4,096-row prefill windows |
| [G17-LEVERS.md](G17-LEVERS.md), [G17-PREFILL.md](G17-PREFILL.md) | why C4 mixed sits below steady, the full-mode cone, SPEC_NUCLEUS; the prefill levers on the GPU, fail-fast and the priced floor |
| [G18-RESULTS.md](G18-RESULTS.md), [G19-RESULTS.md](G19-RESULTS.md) | the serving drift found (the NVMe session index) and fixed; adaptive 2,048-row prefill shipped |
| [REWRITE-PLAN.md](REWRITE-PLAN.md) | the rewrite study behind G13: where a 1-row window's time goes against its floor, and the ranked kernel rewrites |
| [DECODE-ROOFLINE.md](DECODE-ROOFLINE.md), [DRAFT-ACCEPTANCE.md](DRAFT-ACCEPTANCE.md) | the decode roofline and the draft-acceptance study |
| [PARKED-SAGE-1.59.md](PARKED-SAGE-1.59.md) | the evaluated (and parked) 1.59 bpw SAGE pack |
