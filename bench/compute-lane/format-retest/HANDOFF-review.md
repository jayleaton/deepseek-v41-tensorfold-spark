# Source semantic review — format retest v1

Reviewed 2026-11-07. Deliverables are this new file and `stdbench/format-retest/review-resolutions-v1.json`. No existing source, prompt, manifest, resolution, ledger or configuration was edited. No helpers, Python, Node, task generation, tests, CollisionIndex run, model requests, SSH, endpoints, engine work, commits, pushes or external posts were used. Local file reads and JSON/prose serialization only.

## Scope and result

All 334 complete prompts in `stdbench/format-retest/inventory-v1/review-pending.json` were read, including their final-format instructions, complete programming examples and full falling-block game API. The new JSON has an entry for each exact exported ID. Each entry binds the exported prompt SHA256 and actual authoritative source path/SHA256, and identifies the source record's original ID and original user-prompt location. For renamed historical variants, the location uses `historical-records.json`'s `source_original_id`; the JSON key remains the exported `historical-...` ID. Existing records and original IDs were not rewritten.

Disposition: 334 `decision: nonoverlap`, with individual input/operation reasons; zero unresolved pending originals. This is not a blanket classification by study name or shared concepts. These particular pending originals either specify different operations or input bounds, or ask for arbitrary-input programs, advice, explanations, language transformations or applications. There is no pending original requiring an invented signature. The 40 small `mul_big` products each quote their exact complete prompt in their reason and exclude the factor-length bounds; the seven pending `div_big` divisions likewise quote their exact operands and exclude the 80/29-digit bounds. Neither their names nor the requested output formatting establishes nonoverlap.

This disposition is confined to these 334 pending originals. It does not declare that all historical computations are nonoverlapping, nor seal source completeness or accept any new candidate. Potentially colliding historical operations already recognized by the canonical parser must remain in the index. Merge these entries with the inherited prompt-bound resolutions when building the parent's index; do not replace the entire historical inventory with these entries.

## Semantic basis

Read the full `in-model-runner/T5-FORMAT-RETEST-SPEC.md`, compute-DL base specification's distributions and fixed prompts, and `stdbench/compute-dl/disjointness.py`. Also consulted the complete pending prompt text, historical source records, source-audit source pins/retained variants and inherited resolutions. `stdbench/format-retest/inventory_sources.py` explains the export's separation of older compute-DL instances. These are original-source exports, not generated model answers. No historical prompt/code was executed.

The comparisons use the actual prospective inputs:

- Integer: positive 24–36-digit A and B, C in [-10^20,10^20], operation A*B+C. C=0 is allowed: a bare old product can collide. Small-factor products are excluded by operands, not by the absence of an offset. The very large square-difference expression `(10^80+123456789)^2-10^160` has neither the supplied factor lengths nor the bounded offset; algebraic expansion does not produce two allowed factors.
- Modular: 14–20-digit A, exponent 500–5000, 16–22-digit M. The pending powers of 7/11/13 have small bases/exponents/moduli; last 18 digits of 3^400 has an allowed-size implied modulus but excluded base/exponent. General power-query samples also use M<=10^9. Binomial, digit-sum, Fibonacci, LCM and stair-count operations are not renamed modular exponentiation.
- Division: positive 80-digit A and 29-digit B, both q and r. Every pending concrete division has shorter inputs; remainder/divisibility by 11 is not this input pair.
- DP: 7–10 rows and 7–10 columns, integer costs 0..99, right/down moves, minimum visited-cell sum including endpoints. String edit distance is not the same supplied matrix. Maze walls/four-direction moves and the 2x2 Harvest maximum-sum example differ in operation as well as inputs; the latter also has negative costs.
- Graph: 9–14 directed weighted vertices, positive weights 1..40, backbone and extra unique nonself edges, distance 0 to n-1. The concrete A..F unit-edge example has only six vertices. Undirected all-destination roads, components, topological ordering, spanning trees, tree depths, forward jumps and Euclidean geometry request different structures/quantities.
- Aggregation: 40–60 [A..F,quantity,unit_amount] records, quantity 1..20, signed integer cents -999..1999, all six categories, sum of quantity*unit_amount separately for each. Invoices with discounts/tax/rounding, short customer/city/status sums, averages and generic SQL/CSV/expense apps do not supply this fixed input multiset. Sample table/record content was read rather than discarded as a formatting suffix.

All format variants retain their own entries and prompt hashes. The final JSON sentence, FINAL_JSON key shapes, code-fence instructions and padding requirements were read but were not used as novelty evidence. Duplicate underlying small inputs receive the same mathematical exclusion reasoning without dropping distinct exported records.

## Canonical signatures and parent gate

Read the actual canonical implementation rather than guessing a representation: `integer_ast` returns `['integer-expression', node]`; literals are `['literal', integer]`. `_binary` orders commutative children by compact sorted-key JSON text, rewrites subtraction of literals to addition of negatives, and removes additive literal zero. Consequently A*B+0 and a bare A*B share a signature; mathematical operands, not English wording, determine the check. JSON integers must remain exact arbitrary integers, never binary floats or quoted operand strings in a signature.

The old 30-digit English multiplication prompts are outside this pending list because their literal `What is ... * ...? Give the exact integer.` form is recognized by `_prompt_signature`; W9's corresponding `spec.op='*'` rows are recognized by `prior_signature`. For example, W9 `w9-mul30-010` represents:

```json
["integer-expression",["*",["literal",142178118648781391961882344118],["literal",249077809251689575526424560768]]]
```

Do not classify those rows as nonoverlap. The parent must derive/index all such signatures from actual source inputs with exact integers, including the unstructured `mul30-*` rows, and test candidate A/B with C=0 against them. The example above documents the reviewed representation, not a replacement resolution or proof that the complete index was executed here.

`source-audit.json` reports 1040 retained historical records, 40 retained variants, 8767 deduplications and 366 discovered source entries. Importantly, `inventory_sources.py` explicitly skips IDs beginning `compute-dl-v1-` from `historical-records.json`. The discovered 120-task compute-DL source is pinned at `<rank0-work>/stdbench/compute-dl/dataset-v1/tasks-public.jsonl`, SHA256 `3e1e07d7a573eb194c0568afac2ef2112011b2b2cada9012fbbce4e6f47f7548`; the parent must include those 120 typed instances separately in the final gate. Discovery of that file alone is not indexing it.

Uncertainty among the 334 reviewed originals: none retained. Remaining verification is operational/provenance, not an assertion of semantic ambiguity: independently verify every source and prompt hash binding, full ID coverage and preserved aliases, inherited-resolution applicability, and the combined CollisionIndex on Sparks under the existing <2GiB scope. Include the 120 old compute-DL tasks, recognized signatures and generic-contract rows as well as these reviewed rows. Missing sources, unexpected schema/signature failures, unresolved originals or actual collisions still block collection. This review neither authorizes generation nor asserts that those parent checks passed.

## Standards and verification boundary

The parent task ledger `stdbench/format-retest/LEDGER.md` records that this artifact tree is outside Git and applicable AGENTS.md/LOCAL-AGENT.md/CONTRIBUTING and lint/format/build/type configuration were searched with none applying. No repository standards were imported from unrelated projects. There is no commit, export kit or PR in this task. The ledger reserves executable validation for Sparks: source pins, full historical disjointness, independent references, grader and request/provenance checks. No local project check command was run.

Applied the assigned `principle-prove-it-works` skill within the user restriction: directly read the produced JSON, including both endpoints, restored-original-ID locations and representative arithmetic/graph/aggregation decisions. JSON assembly copied path/hash strings from the export and IDs from the historical records without numerical conversion of operands. Executable hash recomputation, candidate generation and CollisionIndex validation are deliberately left to the parent as requested; source review is not their substitute.
