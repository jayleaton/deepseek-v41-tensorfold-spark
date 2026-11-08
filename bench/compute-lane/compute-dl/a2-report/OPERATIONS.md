# Completed collection — A2 operations

The final raw audit qualifies all 720 original P/D/L keys, with zero missing. The A2 turn's step log records QUALIFIED after normal G restoration. Its final dataset response was at 2026-11-06T15:36:08.553911Z; G was verified serving at 15:43:08.727298Z. Independent live verification at 15:43:45Z returned `/v1/models` HTTP 200 and verified both running G rank images. See `final-G.json` and `final-audit-v1/audit.json`.

## Authority and unchanged science

- Parent: `c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141`.
- A1: `0dca1fa52838388ea1e7afbc3a4c5e9747775f64e871640df7a66496369e5453`.
- A2: `8eaa9575592f519b504d08022d400602e4b552874c8424b02c59c34f4af174e2`.

A2 changes only the cumulative budget and permits one P restoration break. Tasks, keys, order, seeds, wires, image, caps, strict extraction and A1 analysis remain unchanged. Strict L/P is the amended primary comparator; D/L measures tool availability. All 342 retained replies passed clipping-aware revalidation. No outage-related exclusions or replacement requests were introduced.

## Extension and break

The extension was outcome-informed. JJ's reported decision point was 49 P replies at 11:53Z; A2 froze at 11:59:56Z with 52 durable P replies. The sealed A1 controller could not safely reload its cutoff. It therefore reached the original admission cutoff, settled its request, drained and restored G, using the one permitted break.

The final A1 P reply completed at 12:37:13Z. Admission then closed at cumulative 15,905.560 seconds. G restoration completed at 12:44:36Z, closing the ledger at 16,348.196950864 seconds. The exact unattempted 124-key P suffix was sealed in a fresh namespace, preserving all 116 observed P replies and the original order. No observed reply was retried. A2 resumed at 13:23:15Z after CPU qualification. The final report includes descriptive P-before/after-break batches; both remain in the primary analysis.

The additional A2 interval charged 8,393.054357156 seconds (2h19m53s), including boot, diagnostics, generation, drain and restoration. Total cumulative charged time is **24,741.251308020 seconds (6h52m21s)**, with zero overrun. Limits were 36,000 seconds total, 33,900 seconds admission and a 2,100-second reserve. The early 100–110 seconds/reply projection did not describe every task family; actual counters governed admission. Waiting and offline CPU work with G restored were excluded under the original definition; held time was never reset or subtracted.

## Outage recovery

The power outage rebooted both Sparks around 2026-11-05T07:30Z and removed `/tmp` lock files. JJ confirmed that no measured request was in flight. Recreated coordination files are checked for existence/readability in preflight; <rank1-host> GPU leasing uses the required read-only open.

Recovery conservatively charged the whole 2026-11-06T08:53:34.922223Z–09:42:05.092783Z interval (2,910.17056 seconds), retaining the failed ledger and restoration evidence. The idle Flash17 proxy was stopped to release port 8000; its unit remains enabled and its configuration was unchanged. The restoration wrapper uses process-only termination so detached G survives wrapper settlement. Normal G restoration was independently checked before collection resumed.

## Limits on interpretation and provenance

Cold boots/cache resets, mostly L-before-P ordering, the P restoration break and outcome-informed amendments are confounds. CPU prompt-array equality, D/L wire identity and server prompt-count reconciliation do not expose actual served arrays; same-length rendering differences remain possible. Multi-round processed-input totals remain withheld where unknown. Initial-prompt-plus-returned totals are subtotals, not full-compute or FLOP measurements. Strict grading remains primary; sensitivity grading never rescues it.

Historical evidence copies are hash-identical and live only in the fresh stdbench namespace. No engine changes, original W9/AB3 edits, mutable image tags, external posts or pushes were made. The generated six-family mixture has no matched DeepSeek published score; published benchmarks remain labelled historical references. No serving-default promotion is proposed.

The final raw auditor and publisher run on <rank0-host> under the required 2 GiB/no-swap, nice 19, ionice idle, one-worker scopes with timeout at least 7200 seconds. Original raw/native receipts remain on the Sparks. Final GO markers were revoked on both nodes. The full report and NEEDS-T5 handoff are in `../a2-results-v1/`.
