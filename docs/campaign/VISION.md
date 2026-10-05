# Image input for DeepSeek-V4.1-Flash (TF_DSV41_IMAGES)

Status 2026-10-04 (G15a, docs/G15a-RESULTS.md): `native` on 7954c1d FAILED on the GPU (the tower's SDPA gets 3-D
q/k/v -> "No available kernel", HTTP 500 on every image); a one-line 4-D fix passed the section 6 battery as a test-only
copy. Prod runs 7954c1d with `placeholder`. **tf `c872f17` (dsv41-060) commits that fix** (offline, CPU-tested;
section 8), makes the placeholder notice say not to guess, and makes `stats.vision` per request; **tf `8f10167`**
escapes a typed `<｜deepseek_image｜>` to plain text in every field and mode (`c872f17` refused it with HTTP 400 as
DeepSeek's reference does; reverted, since a refused literal breaks a coding agent's conversation for good). Its GPU window is `scripts/windows/g15vision.sh`
(G15b, NOT RUN). The GPU checks are in [section 6](#6-gpu-window-vision-not-run).

## 1. Can the model see? Yes, natively

| evidence | deepseek-ai/DeepSeek-V4.1-Flash | Mia-AiLab ...-EXL3-2.9bpw | dealignai ...-UNCENSORED-EXL3-2.9bpw |
| --- | --- | --- | --- |
| HF pipeline tag | `image-text-to-text` | `image-text-to-text`, `multimodal` | `image-text-to-text`, `multimodal` |
| `config.json` `vision_config` | `deepseek_v41_vision`: 32 layers x 1,024, 16 heads, MLP 2,816, patch 14, downsample 3, <= 1,024 tokens an image, `min_pixels` 295,936 | same | same |
| `image_token_id` | 129,264 (`<｜deepseek_image｜>` in tokenizer.json) | same | same |
| vision weights (index) | 262 `vision.*`, `aligner.w1/w2`, `image_start/end/newline` | all kept, BF16, shard 39 (0.97 GB: "tower left native") | all kept, shard 39 ("vision tower preserved") |
| `ffn.gate.bias_vl` (image routing bias, 40 layers + 3 DSpark) | 43 tensors, fp32 | **dropped** | **dropped** |
| reference code | `inference/vision.py` (ViT, aligner), `image_processor.py`, `model.py` (image span, `bias_vl`, Engram mask), `encoding.py` (image blocks) | the card points at vLLM `--quantization exl3`; the kit serves images ("vision on" in BASELINE-RESULTS) | card: "Vision + tools + DSpark" |

DeepSeek's card and tech report describe V4.1-Flash as a multimodal MoE. The reference encodes image content blocks
as `<｜deepseek_image｜>`. Mia's vLLM kit serves images through vLLM's DeepSeek-V4.1 multimodal path. Two caveats
from earlier notes: it routes image tokens with the text bias (the pack has no `bias_vl`), and on SM12x it turns off
the in-image bidirectional window.

`bias_vl` matters. Against the pack's text bias it is a different vector, not a small correction. At layer 5 the
mean |difference| is 3.6 and the correlation is -0.16. At layer 30 they are 9.4 and -0.12. Image rows routed with
the text bias pick different experts. `bias_vl_fetch.py` fetched the 43 tensors from the release with HTTP range
reads: 62,976 bytes (40 x 384 + 3 x 128 fp32 values). They are in `data/bias_vl/bias_vl.safetensors` here. The
`__metadata__` holds each tensor's SHA-256 and the source `deepseek-ai/DeepSeek-V4.1-Flash@main` (revision 2cba9e4).

## 2. What was built (tf `dsv41-060`, behind `TF_DSV41_IMAGES`)

| mode | what an image part does |
| --- | --- |
| `placeholder` (**default**) | It becomes the text `[image omitted: this server does not pass images to the model, so the model cannot see this image. Do not guess what it shows; say that it could not be seen.]` (`TF_DSV41_IMAGE_TEXT` overrides it; the second sentence is new in `c872f17`: with the first alone the model guessed, G15a). The reply runs normally, and the model can tell the user it saw no image. The server logs `images: N image part(s) replaced by the placeholder notice`, and `stats.images_omitted` = N. This fixes the coding-agent preview-browser failure (`messages[59].content: images are not served ...`) with no GPU and no memory. |
| `reject` | HTTP 400 `images are not served by this DeepSeek-V4.1 build`, as before. |
| `native` | The model sees the image (sections 3-5). Rank 0 loads the tower. Set `TF_DSV41_BIAS_VL=<folder>` on both ranks. |

Accepted part shapes, in user, tool and system messages: OpenAI `image_url` (object or string), `input_image`,
`image` (`image` / `url` / `image_url`), and Anthropic `source` (base64 or url). A request without image parts
renders and runs byte-identically to before in every mode, which is tested.

Files (tf):

- `cuda/encoding.py`: an `image` hook in `_text` / `normalize` / `encode` (None means refuse, the old behaviour);
  `escape_image_token` on `encode`'s output (`8f10167`: a typed `<｜deepseek_image｜>` stays text).
- `cuda/app.py`: the mode, the placeholder, native rendering / ids / `/tokenize` / context check, and
  `request.images` for the engine.
- `cuda/vision_prep.py`: the V4.1 processor, the span, the digests and virtual ids, the sentinel `Marker` (prompt
  order), `expand`, and the `Host` (fetch, decode and limits are GLM 0500/0600's code).
- `cuda/vision.py`: `Tower` (ViT + aligner + delimiter rows), `Encoder` (digest cache), `Store` (rank 0's rows
  for queued and running requests), the forward hooks (`window`, `fill`, `moe`), `attach_bias_vl`, `mode`.
- `cuda/forward.py`: `embed_rows` gets the rows; the `Streams.image` window; Engram `keep`; the MoE split.
- `cuda/replay.py`: the replayed tail's image rows route with `bias_vl` too.
- `cuda/engram_host.py`, `cuda/engram.py`: ids past the vocabulary are DEAD.
- `cuda/blocks.py`: `moe.bias_vl`.
- `cuda/engine.py`: rank 0 loads the tower (native); both ranks attach `bias_vl` and the ranks cross-check it; the
  request's rows are held around `generate`.
- `cuda/bias_vl_fetch.py`: the 63 KB sidecar, from HF range reads or a local release copy (`--src`).

Knobs: `TF_DSV41_IMAGES` (reject | placeholder | native), `TF_DSV41_IMAGE_TEXT`, `TF_DSV41_BIAS_VL` (a folder; must be
set on both ranks or neither: the boot gather refuses a mismatch), `TF_DSV41_VISION_CACHE_MB` (64: encoded spans by
digest, rank 0). The `TF_DSV41_VISION_*` limits: MAX_IMAGES (8), MAX_TOKENS (a cap below 1,024 positions), FETCH (1;
https only, public addresses, GLM 0600's hardening), FETCH_TIMEOUT (10 s), FETCH_TOTAL_S (30 s), MAX_BYTES (20 MB),
MAX_PIXELS (64 M), PREP_SLOTS (8), PREP_MB (256).

Not built: a `describe` mode (a second vision model captioning the image). The model sees natively, so a describer
would only be a fallback if `native` fails its GPU gate. GLM-5.3 cannot be the describer while V4.1 serves, because
the two models cannot be resident together on the Sparks (ENGINE-PLAN section 14, item 8).

## 3. Design: where an image goes

The design follows the reference (`inference/model.py`) and GLM 0500's host side.

1. **Placement.** The reference renders an image part as `<｜deepseek_image｜>`, joined to the other parts by blank
   lines. It is tested equal to the reference encoder, including image order. The sentinels keep prompt order even
   when tool results are re-sorted. Each placeholder becomes the image's span: `IMAGE_START, (IMAGE x w,
   IMAGE_NEW_LINE) x h, IMAGE_END` = h(w+1)+2 positions, at most 1,024. The grid h x w is the ViT patch grid / 3,
   rounded up.
2. **Virtual ids.** Each span position carries a virtual id (>= 2^24, past the 129,280-token vocabulary): a hash of
   (the image digest, the position). The digest is SHA-256 of a version tag, the processor and vision config,
   whether `bias_vl` is loaded, the grid and the bf16 patches. So every token-keyed cache (the session store's RAM
   and NVMe tiers, prefix shares, request-log hashes) keys the pixels without any change to that cache. The same
   image gives the same ids, so a follow-up turn hits the session. A different image gives different ids, so there
   are no false hits. `/tokenize` and `usage.prompt_tokens` count the span as the reference does (129,264 each).
3. **Embedding.** No rank's vocabulary range holds a virtual id, so `embed_rows` gives 0 there. Rank 0 writes the
   span row into its partial before the embedding all-sum, so rank 1 gets the rows through an exchange it already
   does. No new protocol message is needed, and row + 0 = row exactly. IMAGE positions get the aligner's rows in
   reading order, and the delimiters get the learned `image_start` / `image_newline` / `image_end` rows (bf16, as
   the reference's `h`).
4. **Engram.** Image positions are DEAD for the n-gram hash: look-back stops there, including across a chunk
   boundary inside a span, via the slot's lookback. Their Engram gate is 0, so the stream passes through as in the
   reference.
5. **Routing.** Image rows select experts with `bias_vl`. A window's image rows run the MoE as their own call with
   the bias swapped. Every MoE path (twin, grouped, fused, fast prefill, gm) computes a row from that row alone, so
   this equals the reference's per-row `torch.where(image_mask, bias_vl, bias)`. Text windows take the old single
   call. The DSpark blocks' `bias_vl` stays unused, as in the reference.
6. **Attention.** It is causal over the span, as in DeepSeek's reference `model.py`. The kit's SM12x path is causal
   too. In-image bidirectional attention is a separate quality experiment (ENGINE-PLAN section 9) and is not done.
7. **Replay prefill (`TF_DSV41_PREFILL=replay`, prod).** The encoder pass runs the hooks above. The decoder replay
   over the last 127 prompt rows reads the stash's ids, so image rows in the tail route with `bias_vl` there too.
   Replay equals full for n <= 128 with an image in the tail (tested). Prompt snapshots carry the ids (virtual ones
   included), so resumed equals fresh.
8. **Drafting.** DSpark reads the target's taps, never the prompt ids. The lookup drafter never proposes a virtual
   id (GLM `SuffixIndex`). The draft head's tracker clips ids.
9. **The tower** (rank 0 only, BF16, 0.97 GB). It is `inference/vision.py`: 2D RoPE, full attention inside the
   image (SDPA on [1, heads, n, 64], flash / memory-efficient only, so no N^2 buffer at ~8.9k patches; 7954c1d passed
   3-D q/k/v, which the fused kernels refuse, G15a), and the 3 x 3 unfold
   aligner. It runs on the HTTP thread that serves the request, before the job is queued: one image at a time (a
   lock), with a 64 MB digest cache.
10. **Rows' lifetime.** Rank 0's `vision.STORE` holds a request's span rows from before its job is queued until
    `generate` returns or raises. A prompt with virtual ids but no images is refused before anything reaches rank 1.

**Memory** (head = rank 0, ~10 GiB MemAvailable while serving):

| item | rank 0 | rank 1 |
| --- | --- | --- |
| tower + aligner + delimiters (BF16, resident with `native`) | 0.97 GB | 0 |
| `bias_vl` | 63 KB | 63 KB |
| encoder span cache (`TF_DSV41_VISION_CACHE_MB`) | <= 64 MB (10 MB an image of 1,024 rows) | 0 |
| tower activations, one image of ~8.9k patches (transient) | ~0.25 GB peak (MLP 100 MB, qkv 55 MB; SDPA without N^2) | 0 |
| preprocessed-image cache (`TF_DSV41_VISION_PREP_MB`, host) | <= 256 MB (~0.5 MB an image) | 0 |

So `native` costs ~1.0 GB resident plus ~0.25 GB peak on the head. `placeholder` and `reject` cost nothing. Set
`TF_DSV41_VISION_PREP_MB=64` on the head to keep the total under ~1.1 GB.

## 4. Exactness rules

- **Text requests are unchanged in every mode.** Same prompt text, same ids, no image hooks in the window. The only
  new text-path cost is a `max()` over a window's ids. The forward, replay and serving suites pass unchanged.
- **The forward with images is a function of (tokens, pixels).** Virtual ids are content hashes. The rows are the
  tower's output for those pixels, cached by digest, and the digest includes every setting the rows or the routing
  depend on. Session hits are therefore sound.
- **Image positions follow the reference's rules exactly** (`tests/family/test_vision_reference.py` here, against
  `engine/reference` extended with the image rows, the Engram mask and `bias_vl`). In exact numerics the logits
  agree to 1e-12 for the whole prompt, windows of 8, windows that cut the span, and TP=2 (rank 1 gets the rows only
  through the all-sum). Each rule moves the logits when left out. Replay with the span in the tail equals full.
  Rows are row-invariant as before, so batched equals alone.
- **The tower against the reference.** The processor matches bit for bit (patches, plans over 15 sizes x 3 budgets,
  span layout). The ViT + aligner match the reference modules at 1e-5 in fp32. On the GPU the tower runs in BF16, as
  the reference does (`generate.py` sets the default dtype to bf16), so its bits depend on the kernels (SDPA, GEMM).
  That is the one place outputs are not pinned. The digest cache makes it deterministic within a process. Across
  processes, the same image could give rows that differ in the last bf16 bit under different kernel selections.
  Section 6 measures that.

## 5. CPU tests (workstation)

- `tests/test_dsv41_vision.py` (23): processor against the reference, tower against the reference, digests and ids,
  `expand` and refusals, part shapes, the `Store`, `window` / `fill` (rank 0 only), the MoE split against a per-row
  `where`, `attach_bias_vl`, the modes, and Engram DEAD for virtual ids (hasher and round tables).
- `tests/test_dsv41_images.py` (13): text-only byte identity in all modes; reject; placeholder (3 part shapes, user +
  tool messages, the stats, the log line, the text knob); native equal to the reference encoder, tool-result
  ordering, ids / `/tokenize` / context check, cache keys by pixels, refusals; and the engine holding and releasing
  rows (also on failure) and refusing virtual ids without images.
- `tests/test_dsv41_serving_app.py`: one assertion changed. An image part no longer fails `check` in the default
  mode.
- Here: `tests/family/test_vision_reference.py` (6) and the full `tests/family` + `tests/reference` suites (102
  pass).

Run: `TF_DSV41_TOKENIZER_DIR=<dir with tokenizer.json + config.json> python -m pytest -q tests/test_dsv41_vision.py
tests/test_dsv41_images.py` in tf, and `TF_SRC=<engine checkout>/src python -m pytest -q
tests/family/test_vision_reference.py` here.

## 6. GPU window: vision (NOT RUN)

Before the window, copy `data/bias_vl/` to both nodes (e.g. `/cache/dsv41-bias-vl/`) and check the SHA-256 values
in its metadata. About 2 h in total. Every step is A/B against the same build with `TF_DSV41_IMAGES=placeholder`.

1. **Boot** with `TF_DSV41_IMAGES=native TF_DSV41_BIAS_VL=/cache/dsv41-bias-vl` on both ranks (rank 1 ignores
   `IMAGES`). Expect `bias_vl: 40 routers` on both ranks and `vision tower 0.97 GB` on rank 0. Record the memory
   snapshot after boot: head MemAvailable is expected about 1.0 GiB below the placeholder boot. **Stop if the head's
   MemAvailable floor (memory.py, TF_DSV41_GRAPH_FLOOR_GIB) is breached.** In that case lower `TF_DSV41_POOL_TOKENS`
   by ~1.1 GiB worth of pages, or ship `placeholder`.
2. **Tower parity** (rank 0, no serving): load the tower and run 3 test images (`inference/examples/images/*.jpeg`
   from the release plus one 1920x1080 screenshot). Compare against the reference `vision.py` in bf16 on the GPU:
   span rows max |diff| <= 1 bf16 ulp of the row max. Run it twice in two processes: are the bits identical? If
   not, record it (section 4's caveat) and see whether `torch.backends.cuda.enable_flash_sdp(False)` makes it
   repeatable. Time each image: expected well under 1 s for 8.9k patches.
3. **Smoke**: `curl` a chat with one data-URL image ("what is in this image?") at T=0. The answer must describe
   the image. Then run the release's two-image example (`inference/examples/example_harmony.json`, carrots / corn)
   and compare the answer and prompt token count against the kit (vLLM) on the same request. `prompt_tokens` must
   equal the kit's.
4. **Exactness on the GPU**: the same image request (a) cold, (b) with a session hit after a first turn, (c) next
   to 3 text streams, (d) with `TF_DSV41_PREFILL_CHUNK=128` (spans cut by chunks). The greedy replies must be
   identical. Then send a request with the image in the last 127 tokens (replay tail).
5. **Routing check**: the same request with `TF_DSV41_BIAS_VL` unset. The reply should differ (it routes as the
   kit), confirming the bias is live. Optionally score 20 VQA items (screenshots with known answers) with and
   without it.
6. **Agent flow**: a coding agent's preview browser takes a screenshot inside a tool result through the full agent loop
   (multi-turn, tools). Expect no 400s, session hits on later turns (`cached` in stats), and the head watchdog
   quiet.
7. **Concurrency**: 4 image requests at once while 2 text streams decode. Check the text streams' decode tok/s dip
   during tower runs (the tower shares the GPU) and that no graph capture failed (`capture_error_mode`
   thread_local: lazy captures beside the tower's kernels).
8. **Limits**: a 9-image request (400), a 25 MB image (400), an https URL (fetch policy), a typed
   `<｜deepseek_image｜>`: 200 with no image id 129,264 in `/tokenize`, alone, in a conversation's history (tool call,
   tool result, reply) and beside a real image whose image-token count is unchanged (from `8f10167`; G15a on
   7954c1d: 200 but read as the image id).

Pass: steps 1-4 and 6 clean and step 3 matches the kit. Then prod sets `TF_DSV41_IMAGES=native`. On any failure,
ship `placeholder`.

## 7. Production setting

- **Now / next deploy:** `placeholder`, the default, so nothing needs setting. It fixes the failing agent turns at
  no memory cost.
- **After the vision window passes:** add to `config/prod.env`:
  `TF_DSV41_IMAGES=native`, `TF_DSV41_BIAS_VL=/cache/dsv41-bias-vl` (both nodes), `TF_DSV41_VISION_PREP_MB=64`.

## 8. The G15a fixes (tf `c872f17` + `8f10167`, offline) and G15b

- **Tower attention.** `vision._sdpa` adds a batch dimension of 1 for SDPA and removes it after, inside
  `sdpa_kernel(flash, memory-efficient)` on every device, so the CPU tests run the same call. They also run the tower
  through a stand-in for CUDA's fused-kernel rules (math path off, q/k/v 4-D, else "No available kernel");
  7954c1d's call fails it. `tests/cuda/test_dsv41_vision_gpu.py` runs `_sdpa` at 8.4k patches on CUDA, the tower at
  real widths on real image tensors (shape, dtype, finite, delimiters, repeatable, activation peak < 1 GiB), and the
  checkpoint's tower against DeepSeek's reference (cosine).
- **Typed image token: escaped, not refused (`8f10167`).** DeepSeek's reference encoder (`encoding.py`:
  `_validate_no_image_sp_tokens`, `_process_image_blocks`) raises on `<｜deepseek_image｜>` in any message's content,
  text block or reasoning, and the sage kit's server uses that encoder (the vLLM kit's code is in its container on the
  Sparks, not checked offline). `c872f17` matched that with HTTP 400, but our clients are coding agents that read this
  engine's sources and logs in tool results, and a refused literal breaks every later turn of the conversation. So
  `encoding.encode` now passes its output through `escape_image_token`: every typed occurrence becomes
  `<` + U+200B (zero-width space) + `｜deepseek_image｜>`, which tokenizes as ordinary BPE pieces, never as id 129,264.
  That covers every field the encoder renders (message text, text parts, tool results, reasoning, tool-call
  arguments, tool schemas) in every mode. Nothing escapes other special-token literals typed by clients (a typed
  `<｜User｜>` still becomes its id), so there was no existing mechanism to follow. Real images are not affected: the
  native hook renders `vision_prep.Marker` sentinels, which become the placeholder only after `encode`. Text without
  the literal renders byte-identically.
- **`stats.vision`** is the calling request's encode (thread-local), not the last encode of any request.
- **G15b section** `scripts/windows/g15vision.sh` (reuses G15a.sh, `g15a_tower.py`, `g15a_vision.py`): check, images,
  tests (the GPU test + the CPU suites in the image), tower x2, the battery (s1 native, s2 chunk 128, s3 no bias_vl),
  the 299K stress with native on (plus image requests during the long prefill; head MemAvailable >= 5 GiB), verdict,
  and `ship` (on PASS: `TF_COMMIT`, `TF_DSV41_IMAGES=native`, `TF_DSV41_BIAS_VL=/cache/dsv41-bias-vl`,
  `TF_DSV41_VISION_PREP_MB=64` into config/prod.env). Stage the commit for prod first: `TF_COMMIT=8f101674302826b3f9668b527f21af976841efb3
  TF_REF=8f10167 scripts/serve.sh stage` (8f10167 also carries two G15b commits of other sections, `242664c` depth
  mode 2 and `5157976` m2bench, both default-off / bench-only).
