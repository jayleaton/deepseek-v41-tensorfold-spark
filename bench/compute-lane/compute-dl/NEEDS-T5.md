# NEEDS-T5: computation-demanding D/L screen

The frozen study is **INCOMPLETE: 342/480 replies retained, 138 uncollected**.
G is restored, both collection GO markers are revoked, and no benchmark/check job
is pending. No engine source, image, grader, analysis, task, reference answer or
observed reply was changed.

Protocol SHA256: `c3e6da463138fb82ae51245dd55d7033fedae56c5e4266f8aaacfb6b80a4a141`.
See [results](RESULTS.md), [frozen audit](report-v1/audit.json), and
[the exact remaining original requests](remaining-collection.json).

## What stopped collection

At `compute-dl-v1-aggregation-015`, seed 101, the frozen comparator compared a
native tool result with full independent CPython stdout. The native output
explicitly carries the engine's 512-token tool-result cap. Full stdout is 554
native tokens. Three fresh isolated CPU replays reproduced the full stdout.

[The pinned native-tokenizer CPU proof](diagnosis-v1/truncation-proof.json)
reconstructs the production `capTokens` operation and matches every delivered
byte: the first 512 tokens, UTF-8 boundary handling, and the truncation marker.
The full first output line also matches the independent task reference. This
diagnosis does not change final-answer grading or replace the failing receipt.
The tool-body cap is separate from the study's 8192 aggregate returned-ID cap.

The comparator omitted an existing native presentation policy. The frozen audit
therefore remains unqualified, and collection remains stopped. The source-aware
diagnosis supplies no completed-study efficacy claim.

## Requested prospective adjudication

1. Decide whether a source-aware verifier amendment can honestly qualify the
   retained clipped result. It should apply the pinned native body/status and
   token-clipping rules to independent stdout/stderr before exact byte comparison;
   a generic marker/prefix exemption is insufficient. Preserve source/native-frame
   attribution, the forced-call D0/L1 hard gate, replay provenance, and wrong-result
   stops. Pre-register clipped/unclipped, UTF-8, error, boundary and marker-spoof
   controls before any further collection.
2. Freeze an explicit completion layout for the **18 uncollected L replies and
   120 uncollected D replies**, retaining original tasks/seeds/wire bytes and every
   existing observation. The original two-turn layout has no restart path after
   draining L mid-turn. Any additional boot/cold-cache effect needs disclosure.
   If this cannot be justified, leave this study incomplete and freeze a new
   prospective study separately.
3. Retain the original grader, extraction, analysis and reference hashes. Do not
   rescue format rejections, retry observed replies, shrink the denominator, or
   pool older cohorts. Carry forward the held-time ledger: **3413.519 seconds**
   used, no overrun; the original 5-hour target and 4h25 admission cutoff remain
   the existing limits unless separately amended.

The CPU proof is diagnostic, not an effective amendment. No resume script or new
GPU turn has been authorized by this document. No engine edit is needed to
express the candidate reference policy.

Implementation note for any newly sealed launcher: systemd's inherited ignored
SIGPIPE left an idle CPU-state producer after successful protocol checks. Exact
idle producer PIDs were verified and terminated before each queue append; this
did not alter measured requests or held time. A fresh launcher can set
`IgnoreSIGPIPE=no` and CPU-check that lifecycle, keeping the original artifacts
unchanged.

**Default: preserve the stop until JJ steers a frozen T5 amendment SHA.**
