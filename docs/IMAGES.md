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

### Zig measurements

Pending: the Zig vision gate on two DGX Sparks.

| Check | Result |
| --- | --- |
| Vision tower vs DeepSeek's reference | pending |
| Visual questions | pending (x / 20) |
| Memory with images on (rank 0 MemAvailable at boot) | pending |
| Text decode speed with images on vs off | pending |
