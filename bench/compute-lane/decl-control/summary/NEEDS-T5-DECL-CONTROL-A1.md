# NEEDS-T5 — declaration control A1

The fresh three-arm study is complete: 552/552 scheduled replies, 18 excluded diagnostics, three qualified GPU turns, no reserve turn. G was restored after every turn and directly verified on both ranks. All 18 experiment container names were absent at the final checks; no experiment job remains pending. No engine changes, pushes or external posts.

[RESULTS.md](RESULTS.md) records both frozen spec hashes, per-suite accuracy/native returned tokens/wall time/timeouts/caps, exact cluster sign-flip/Holm tests and paired bootstrap intervals. [The complete report](decl-control/RESULTS.md), [552-row CSV](decl-control/finalized-a1-v1/per-request.csv), [analysis](decl-control/finalized-a1-v1/analysis.json), [evidence audit](decl-control/finalized-a1-v1/audit.json), [export verification](decl-control/exports-a1-v1.json), and [ledger](LEDGER.md) are retained locally. The earlier two-arm study is preserved separately.

Review focus:

- All suites are **DECLARED BUT UNUSED**. Dataset calls were zero in P/D/L; all six L diagnostic results were authenticated and independently checked. The pinned-source forced-dispatch CPU control passed D=0/L=1, including a behavioral guard mutant. This does not instrument the deployed binary.
- D and L have identical strict accuracy and returned-token totals. Their wall ratio intervals include 1. Use **no detected incremental execution-enabled effect on this cohort**, with no equivalence, execution-benefit or serving-default claim.
- All 190 D/L wire/count pairs and 18 actual native rank environments passed. A1 permits only CPU-matched declaration prompts, identical request bodies and matching served prompt counts; same-length server rendering differences remain possible.
- A declaration adds 306 input tokens per reply. Lower returned tokens versus P already occur with execution OFF; full prompt+returned ratios do not support total-token savings. Dataset input coverage is complete; multi-round diagnostic totals are withheld.
- GSM strict results are 54/96 P and 52/96 D/L. Formatting sensitivity materially changes their interpretation; the preregistered one-numeral secondary remains secondary. HumanEval's one capped P reply dominates apparent savings, blocking an uncensored efficiency claim.
- All 120 MATH answers received two independent blinded reviewer agent reviews; no disagreements or unresolved labels. This is explicitly agent grading, without a human-review claim. All six primary Holm p-values are 1.
- Postcollection I/O/reporting wrappers sit outside the precollection seal. The first audit's tuple/list persistence comparison was repaired with its failed receipt preserved; all 570 corrected attestation checks and negative controls passed. Frozen grades were identical on rerun, and all 87 precollection file hashes remain unchanged.

Main results SHA256: `50fe83b8dc4f3dccd8d9a4a63d5a217d43fe703188a44bafd0347bdc2208f5c7`.

This file is the requested local **NEEDS-T5** handoff; no message was sent to Lead or posted elsewhere.
