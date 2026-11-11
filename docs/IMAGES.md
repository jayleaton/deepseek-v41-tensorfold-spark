# Native image input

The [Zig engine](ZIG-SERVE.md) serves images the same way: the tower and aligner on rank 0, image rows routed with
`gate.bias_vl` on both ranks, no Engram contribution at image positions. Requests take the same image parts and the
same limits (8 images, 20 MB and 64 M pixels an image, 1,024 positions an image); its preprocessed-image cache
(`TF_DSV41_VISION_PREP_MB`) defaults to 256 MB. The [serving profile](../config/prod-zig.env.example) turns it on:

```
TF_DSV41_IMAGES=native
TF_DSV41_BIAS_VL=/cache/dsv41-bias-vl
```

Use the selected pack's own image routing biases. The flash17 profile reads them from `/model` on both nodes;
do not use the retired Python serving image to fetch them or substitute another release's tensors.

If the variable names a folder without the tensors, `scripts/run-zig.sh` refuses to start (the engine would fail its
boot). Unset it to serve images with the text bias; the boot log then says the image routing bias is missing.
Turning images on adds the image segments' Triton variants to the AOT set: run `scripts/run-zig.sh N --check-aot`
on each node after changing either variable (see [ZIG-SERVE](ZIG-SERVE.md#generate-runtime-assets)).

### Long histories with many images

A request carries at most `TF_DSV41_VISION_MAX_IMAGES` images (8 by default; one image is at most 1,024 positions).
Session reuse resumes only from a stored entry that is an exact prefix of the new prompt. A client that drops an old
image from the middle of a long history, replacing it with a note, a summary or nothing, changes the prompt at that
image, so every entry past it stops matching and the next turn prefills from the last entry before the change. On a
long agent session that can be the whole context again: 216–278 s at ~500K tokens in
[#80](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/issues/80). Edited messages and compaction do the same.

- **Raise the ceiling** when an agent history keeps many screenshots: `TF_DSV41_VISION_MAX_IMAGES=32` costs at most
  ~32K pool positions and changes neither the AOT set nor the buffer plan. Check `TF_DSV41_VISION_FETCH_TOTAL_S`, one
  deadline for every image of a request.
- **Trim in batches.** When the history passes the ceiling, drop old images down to the ceiling minus 8, not minus 1,
  so one rewrite (one long prefill) covers eight new images instead of one.
- **Checkpoints** (`TF_DSV41_SESSION_LCP_RESUME=1`, off by default): the engine also saves the prompt's state every
  `TF_DSV41_SESSION_LCP_EVERY` tokens (default 32,768) while it prefills, and keeps those checkpoints as long as their
  conversation. A rewritten history then resumes from the last checkpoint before the change and prefills only the rest
  (a change inside an image resumes before that image). Each checkpoint holds a few MB of the RAM tier
  (`TF_DSV41_SESSION_RAM_MIB`) and no extra pool pages; checkpoints are dropped, never written to the NVMe tier.

Whatever the setting, a miss that loses at least 1,024 tokens of reuse is logged with where the prompt changed:
`sessions: slot S: a N-token prompt diverged at token D (stored M); it prefills from P`.

### Zig measurements

Pending: the Zig vision gate on two DGX Sparks.

| Check | Result |
| --- | --- |
| Vision tower vs DeepSeek's reference | pending |
| Visual questions | pending (x / 20) |
| Memory with images on (rank 0 MemAvailable at boot) | pending |
| Text decode speed with images on vs off | pending |
