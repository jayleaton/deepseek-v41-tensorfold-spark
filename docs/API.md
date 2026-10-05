# API

OpenAI-compatible on `HOST:PORT` (`127.0.0.1:8000` by default: put your own proxy and authentication in front):
`/v1/chat/completions` (streaming, tool calls, `response_format`), `/v1/completions` (text or token ids),
`/tokenize`, `/v1/models` (`max_model_len`), `/health`, `/metrics`.

Thinking follows DeepSeek-V4.1's encoding (`Reasoning Effort: N (range 1-100)`), on by default
(`TF_DSV41_THINKING=0` turns it off). Both the top-level `reasoning_effort` and
`chat_template_kwargs.reasoning_effort` are read (the kwargs win):

| value | thinking | effort |
| --- | --- | ---: |
| `none`, `minimal` | off | - |
| `low` | on | 50 |
| `medium`, `high` (default: `TF_DSV41_DEFAULT_EFFORT`) | on | 75 |
| `xhigh`, `max` | on | 100 |
| an integer 1-100 | on | that |

Effort sets how long the model thinks, not how fast it decodes (one report measured ~38% fewer thinking tokens at
`low` than at `max`; tok/s unchanged).

`chat_template_kwargs.enable_thinking` (or `thinking`) true / false sets the mode directly. Note: the kit's vLLM
path renders `low` as 25; we keep DeepSeek's 50.

Images: see [Images](IMAGES.md).
