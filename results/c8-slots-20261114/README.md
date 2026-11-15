# Eight slots of 192K by default (2026-11-14)

The flash17 profile serves eight requests at once, each up to 196,608 tokens of prompt plus output
(`TF_DSV41_SLOTS=8`, `TF_DSV41_CONTEXT=196608`), instead of four of up to 1M. The 2.1M-token KV pool, the memory floors,
the row cap and slot drafts are unchanged, and no new kernels or assets are needed.

Measured on the pair (TP2, flash17 K6 MTP pack), greedy, 256 tokens per reply, distinct prompts:

| load | decode |
|---|---|
| 1 chat | 61–84 tok/s |
| 4 at once | 135 tok/s total, 34 each |
| 8 at once | 155 tok/s total, 19 each (steady-state medians 152.4 / 156.4 / 155.4 over three runs) |

With four slots, chats 5–8 of a burst of eight wait about 10 s for their first token, because they queue behind the
first four. With eight slots they wait 1.3 s.

Concurrent replies can differ from the same request sent alone (raw completions and some prompts; chat replies to the
same prompt matched). Four slots behave the same way: the production profile's precision settings make the output
depend on the rest of the batch. The slot count does not cause it.

Long single contexts (agent sessions past 192K) need `TF_DSV41_CONTEXT=1048576` with `TF_DSV41_SLOTS=4`, the previous
default, which uses the same assets.
