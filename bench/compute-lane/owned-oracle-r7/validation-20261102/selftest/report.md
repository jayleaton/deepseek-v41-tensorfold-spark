# owned-oracle report

Replies (unique texts): 8; cells checked: 11.

| form | verdict | owner | cells |
|---|---|---|---:|
| fence | ok |  | 1 |
| fence | value | owned-form | 1 |
| inline | boundary | owned-form | 2 |
| inline | ok |  | 3 |
| repl | ok |  | 1 |
| repl | value | owned-form | 1 |
| trace | ok |  | 1 |
| trace | value | engine | 1 |

Verdicts: ok = written value equals the independent value of the visible expression (inline), the span (trace) or CPython stdout (fence/repl). boundary = the value of a strictly shorter span glued inside a larger expression (the engine-cell signature: `3^400 mod 10^18 = 10^18`, `φ(2^18) = 2^18`). value = a different value. unknown/unparsed = not decidable here. owner: engine (trace form), owned-form (the reply owned cells of this form), model (this form owned nothing in the reply, so the text is the model's).

- **value** [trace, engine] `2^10` = 1024; wrote 1000; span None
  - at <rank0-work>/owned-oracle/tool/selftest/receipts/bad-trace.json (reasoning offset 15; 1 receipt copies)
  - model: reappears later 0x, in final content False, questioned False; next ']\nOk.\n'
- **boundary** [inline, owned-form] `3^400%10^18` = 192443283334088001; wrote 1000000000000000000; span 10^18
  - at <rank0-work>/owned-oracle/tool/selftest/receipts/bad-inline.json (reasoning offset 21; 1 receipt copies)
  - model: reappears later 0x, in final content False, questioned True; next '? Wait.\nSince φ(2^18)= 262144/2=131072. 400.\n'
- **boundary** [inline, owned-form] `φ(2^18)` = 131072; wrote 262144; span (2^18)
  - at <rank0-work>/owned-oracle/tool/selftest/receipts/bad-inline.json (reasoning offset 63; 1 receipt copies)
  - model: reappears later 0x, in final content False, questioned False; next '/2=131072. 400.\n'
- **value** [fence, owned-form] code 'print(min([2, 3]))\nprint(sorted([2, 3]))'; wrote '4\n[4, 3]'; CPython '2\n[2, 3]\n'
  - at <rank0-work>/owned-oracle/tool/selftest/receipts/bad-list-fence.json (content offset 67; 1 receipt copies)
  - model: reappears later 0x, in final content False, questioned False; next '\n```\n\nDone.'
- **value** [repl, owned-form] code 'len([x for x in [7] if x > 0])'; wrote '0\nSo none.'; CPython '1\n'
  - at <rank0-work>/owned-oracle/tool/selftest/receipts/bad-list-repl.json (reasoning offset 45; 1 receipt copies)
  - model: reappears later 0x, in final content False, questioned False; next '\n'
