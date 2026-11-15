# Option: eight slots of 192K (2026-11-14)

The default profile serves four requests at once, each up to 1M tokens. Setting `TF_DSV41_SLOTS=8` and
`TF_DSV41_CONTEXT=196608` in the profile serves eight at once instead, each up to 196,608 tokens of prompt plus output. The 2.1M-token KV pool, the memory floors,
the row cap and slot drafts are unchanged, and no new kernels or assets are needed.

Measured on the pair (TP2, flash17 K6 MTP pack), greedy, 256 tokens per reply, distinct prompts:

| load | decode |
|---|---|
| 1 chat | 61–84 tok/s |
| 4 at once | 135 tok/s total, 34 each |
| 8 at once | 155 tok/s total, 19 each (steady-state medians 152.4 / 156.4 / 155.4 over three runs) |

With four slots, chats 5–8 of a burst of eight wait about 10 s for their first token, because they queue behind the
first four (9.7 s at p95). With eight slots they wait 1.3 s.

Concurrent replies can differ from the same request sent alone (raw completions and some prompts; chat replies to the
same prompt matched). Four slots behave the same way: the production profile's precision settings make the output
depend on the rest of the batch. The slot count does not cause it.

Pick eight slots when many people share the pair; keep the default when single sessions (agents) go past 192K.
Both use the same assets, so switching is a restart with the two values changed.
