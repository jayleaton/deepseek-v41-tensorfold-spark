# STREAM_RB=4 arithmetic review receipt

2026-10-30; follow-up to PR #71 at `c140fb63f46250b6e964a5e04858b695c30f461a`.
The kernel arithmetic and launch options are unchanged by this follow-up.

## Independent score gate

`check_stream_rb_scores.py` executes the actual `_stream` and `_stream_rb4` kernels,
with the served source identified by SHA-256 in [scores.json](scores.json).
The reference is CPU FP64: independently decode the stored FP8 bytes and scales,
round the decoded key to BF16, then compute the dot, scaling, ReLU, weighted head sum
in FP64 on the same rounded input rows. BF16 keys are also tested.

The candidate buffer encodes the FP32 score and key position. To expose all columns
without instrumenting the kernels, use K=2048/CAP=4096, retaining all visible keys.
Invert the encoding and reorder by key column. The encoding canonicalizes signed
zero; these measurements do not distinguish +0 from -0.
For every case, also run production K=512/CAP=1024 and verify:

- Captured and production MMA layouts and score floating instruction counts match.
- Production top-512 packed candidates match the captured scores' top-512.
- `_stream` and rb4 select identical packed candidates.
- Every recovered score is finite; rb4's worst absolute error in each column is
  no greater than `_stream`'s, with no tolerance or averaged-column allowance.

The comparison covers the first 513 or 1024 columns of each row; later rows also
see up to three extra keys, included in the production selection and FP64 cutoff
comparison. Every per-column maximum and RMS, mismatched FP64 selection position,
cutoff margin, compiler layout and instruction count is recorded in `scores.json`.

48 cases: seeds 7/23/101; random, cancelling dot products, cancelling weighted
heads, and near-tie/repeated keys; BF16 contiguous and paged keys, FP8 paged keys;
1/3/7/8 rows, including ragged four-row blocks and a 513-column masked tile.
Total: 194,636 compared scores per implementation. **48 pass, 0 fail, 0 skip.**
Recovered scores and production top-512 candidates agree between the two kernels
in every case. **Zero worse columns.**

| Input class | Worst absolute error, both kernels | Worst case RMS, both kernels | Rows differing from FP64 top-512, each kernel |
| --- | ---: | ---: | ---: |
| Random | 2.966400146e-6 | 2.902269245e-7 | 0 |
| Dot cancellation | 3.384193406e-7 | 3.046784000e-8 | 47 |
| Head cancellation | 1.530039299e-6 | 8.457884776e-8 | 45 |
| Near tie | 9.951773201e-7 | 2.619052868e-7 | 7 |

The FP64 selection differences are inherited equally by both kernels. The
adversarial cases include zero cutoff margins; this receipt makes no claim of
FP64-exact selection or universal bit equivalence outside these tested inputs.
Decode does not launch rb4 and is unchanged. No model-fidelity, throughput or
end-to-end benchmark was repeated in this follow-up.

## Environment and reproduction

NVIDIA GB10, Linux, Python 3.12.3, PyTorch 2.13.0a0+9186a08b2c.nv26.07,
Triton 3.7.1. Separate temporary test containers, 3 GiB memory limit,
256 MiB shared memory, no network. The existing serving process stayed running;
no model process was added, restarted or reconfigured. Tests use small synthetic
inputs, release device allocations when the temporary process exits, and do not
modify model caches or assets. Production was idle at initial inspection;
performance is not measured on this shared device.

Run in an environment containing the served Python package and Triton:

```sh
python check_stream_rb_scores.py --src <served-package-root> --output scores.json
TRITON_PTXAS_PATH=<ptxas> TRITON_PTXAS_BLACKWELL_PATH=<ptxas> \
  python check_stream_rb.py --src <served-package-root>
```

Production scoring options: `_stream` uses 8 warps/3 stages; rb4 uses 4 warps/1
stage; H=32, D=128, BP=64, SPLIT=16384, RATIO=2, PSH=7, FP fusion enabled.
The independent reference hashes and checker hash are included in `scores.json`.
The offline checker passes all pf/rb2 MMA, dot, float-op and shuffle comparisons,
and builds rb4 with one dot. **9 checks pass, 0 fail, 0 skip.**

## Static checks against pinned base

Pinned base: `853310dab967480013eda7cef4e79c2fd041629a`.

- Engine Lean: 6,379 baseline findings; 6,375 after this follow-up; zero additions.
- Ruff 0.17.3, `check --line-length 120` on the two original changed Python files:
  5 baseline findings, 5 after; the added C408 is removed. The five inherited
  findings are two I001, two C408 and one FURB167.
- Ruff on the new score checker: zero findings.
- `git diff --check`: pass.
- `scripts/check-public.sh`: pass (full source tree plus a separate scan of the added receipt files).

No native engine or kernel arithmetic changed in this follow-up; native host
builds, model replay and long-context timing were not repeated. The new device
score gate and existing offline compiler checker cover the changed tooling.

## Design and historical performance moved from source

pf/rb2 retain per-row dots in separate control-flow regions so Triton's MMA warp
layout remains `_stream`'s [2,4]; the offline checker checks the corresponding
floating instruction and shuffle patterns. Each row owns its buffer and receives
keys in order. rb4 instead combines four queries into [4H,D] and scores two key
tiles per step with one [4H,D] x [D,2BP] dot, then reduces each row's heads and
feeds the row's own buffer. Its [2,2] warp layout requires the independent gate
above, rather than inferring precision from the pf/rb2 layout check.

Historical measurements in the original PR (not rerun here): the old source
comment reported 1.60–1.84x on the six production-shape A/B cases and 1.87–1.89x
at 300K/480K keys; the PR's detailed table records 1.55–1.85x across six cases.
The existing A/B compared split buffers slot for slot; two 304K/320K cold
prompts produced identical greedy replies and about 12% lower prefill time.
Those observations are performance/selection evidence, not a bound on score error.
