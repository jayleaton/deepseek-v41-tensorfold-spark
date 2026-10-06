# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
"""The OpenAI routes for DeepSeek-V4.1-Flash (``CUDA_APP``): our GLM Spark server (``glm5_next/spark/server.py``:
0150 health / metrics, 0490 context errors and ``/tokenize``, 0600 cancellation and request hardening, 0300
request log, 0210 prompt-token cache) with V4.1's prompt and reply formats.

- **Prompt**: DeepSeek's V4.1 encoding (``encoding.py``, MIT; not the kit's AGPL Jinja). Thinking on by default
  (``TF_DSV41_THINKING=0``: off); ``chat_template_kwargs.enable_thinking`` / ``thinking`` per request; OpenAI's
  ``reasoning_effort``: ``none`` / ``minimal`` thinking off, ``low`` 50, ``medium`` / ``high`` 75, ``xhigh`` / ``max``
  100, or an int 1-100 (``chat_template_kwargs.reasoning_effort`` the same); default ``TF_DSV41_DEFAULT_EFFORT``
  (high). ``response_format``'s schema joins the system block as the encoding's ``## Response Format``
  (``TF_DSV41_SCHEMA_PROMPT``, default 1).
- **Replies**: reasoning before ``</think>`` as ``reasoning_content`` (and ``reasoning``), DSML tool calls parsed
  (``dsml.py``) and, when streamed, sent as they complete: the call's name when its invoke opens, each parameter as
  an arguments fragment, ``}`` when it closes (OpenAI's streamed tool-call shape).
- **Multi-step tool calling** (GLM 0620's fixes, always on): history in any OpenAI shape (``content: null``, list
  parts, ``reasoning``, string arguments); the reasoning of earlier tool-call steps kept server-side and restored when
  a client drops it (V4.1 keeps every turn's reasoning when tools are present, so the model sees its own plan and the
  prompt stays a prefix of the session store's entries); ``tool_choice`` ``none`` (no tools offered) / named /
  ``required`` (held by the grammar when structured output is on); ``parallel_tool_calls: false`` keeps the first
  call; schema-typed arguments.
- **Structured output**: ``structured.py`` (GLM 0610's grammars, the DSML tool tag); the grammar starts after
  ``</think>``.
- ``/tokenize`` renders ``messages`` exactly as a chat request with the same body.

Knobs (``TF_DSV41_*``; the reused GLM modules' own names are set from them: ``alias_knobs``).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tensorfold.families.glm5_next.spark import reqlog as reqlog_mod
from tensorfold.families.glm5_next.spark import server
from tensorfold.families.glm5_next.spark.batchplan import is_title_request
from tensorfold.families.glm5_next.spark.prompt_tokens import Memo, PromptTokens
from tensorfold.server.cancellation import RequestCancelled

from . import dsml, encoding
from .structured import Host, schema_for_prompt

EFFORT = {"low": 50, "medium": 75, "high": 75, "max": 100}
FIELD_EFFORTS = {"none": None, "minimal": None, "low": 50, "medium": 75, "high": 75, "xhigh": 100, "max": 100}
ALIASES = {"TF_DSV41_REQUEST_LOG": "GLM53_TF_REQUEST_LOG", "TF_DSV41_TOKCACHE": "GLM53_TF_TOKCACHE",
           "TF_DSV41_DISCONNECT": "GLM53_TF_DISCONNECT", "TF_DSV41_HEALTH": "GLM53_TF_HEALTH",
           "TF_DSV41_STALL_S": "GLM53_TF_STALL_S", "TF_DSV41_REASONING_FIELDS": "GLM53_TF_REASONING_FIELDS",
           "TF_DSV41_GRAMMAR_THREADS": "GLM53_TF_GRAMMAR_THREADS"}


def alias_knobs(env=os.environ) -> None:
    """``TF_DSV41_X`` -> the reused GLM module's ``GLM53_TF_X`` (unless that is set explicitly)."""

    for ours, theirs in ALIASES.items():
        if ours in env and theirs not in env:
            env[theirs] = env[ours]


def effort(value) -> int:
    """``reasoning_effort`` -> the model's 1-100 effort."""

    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        v = int(value)
        if not 1 <= v <= 100:
            raise ValueError("reasoning_effort must be 1-100 or low / medium / high / max")
        return v
    if value not in EFFORT:
        raise ValueError("reasoning_effort must be 1-100 or low / medium / high / max")
    return EFFORT[value]


def _level(value) -> int | None:
    """A ``reasoning_effort`` value -> 1-100, or None for "thinking off" (``none`` / ``minimal``)."""

    key = value if isinstance(value, int) and not isinstance(value, bool) else str(value).strip().lower()
    if isinstance(key, str) and key in FIELD_EFFORTS:
        return FIELD_EFFORTS[key]
    return effort(key)


def options(body: dict[str, Any], default_thinking: bool, default_effort: int) -> tuple[bool, int]:
    """(thinking, effort 1-100) of a request; ValueError (HTTP 400) for unusable values. ``chat_template_kwargs``
    win over the top-level ``reasoning_effort``."""

    kwargs = server.template_kwargs(body)
    thinking, level = default_thinking, default_effort
    for value in (body.get("reasoning_effort"), kwargs.get("reasoning_effort")):
        if value is None:
            continue
        got = _level(value)
        if got is None:
            thinking = False
        else:
            thinking, level = True, got
    for name in ("enable_thinking", "thinking"):
        if name in kwargs:
            if not isinstance(kwargs[name], bool):
                raise ValueError(f"chat_template_kwargs.{name} must be true or false")
            thinking = kwargs[name]
    return thinking, level


class ReasoningMemory:
    """Reasoning of replies this server gave, by tool-call id and by content digest (GLM 0620 ``reasoning``): put
    back into a history message that comes without it."""

    def __init__(self, entries: int = 1024) -> None:
        self.entries = entries
        self.d: OrderedDict[str, str] = OrderedDict()
        self.lock = threading.Lock()

    @staticmethod
    def content_key(content: str) -> str:
        return "c:" + hashlib.sha256(content.encode()).hexdigest()

    def put(self, reasoning: str, content: str, calls) -> None:
        if not reasoning:
            return
        keys = [c["id"] for c in calls or []] or [self.content_key(content)]
        with self.lock:
            for k in keys:
                self.d[k] = reasoning
                self.d.move_to_end(k)
            while len(self.d) > self.entries:
                self.d.popitem(last=False)

    def restore(self, messages: list[dict]) -> int:
        n = 0
        with self.lock:
            for m in messages:
                if not isinstance(m, dict) or m.get("role") != "assistant":
                    continue
                if m.get("reasoning_content") or m.get("reasoning"):
                    continue
                calls = m.get("tool_calls") or []
                key = calls[0].get("id") if calls and isinstance(calls[0], dict) else None
                if not key:
                    content = m.get("content")
                    key = self.content_key(content) if isinstance(content, str) else None
                got = self.d.get(key) if key else None
                if got:
                    m["reasoning_content"] = got
                    n += 1
        return n


class Dsv41App(server.App):
    def __init__(self, engine, model_dir, served: str, **kwargs: Any) -> None:
        alias_knobs()
        model_dir = Path(model_dir)
        super().__init__(engine, model_dir, served, **kwargs)
        raw = os.environ.get("TF_DSV41_THINKING", "1").strip() or "1"
        if raw not in ("0", "1"):
            raise ValueError(f"TF_DSV41_THINKING={raw!r}: expected 0 or 1")
        self.default_thinking = raw == "1"
        raw = os.environ.get("TF_DSV41_DEFAULT_EFFORT", "high").strip().lower() or "high"
        try:
            self.default_effort = effort(raw)
        except ValueError:
            raise ValueError(f"TF_DSV41_DEFAULT_EFFORT={raw!r}: expected low (50), high (75), max (100) or an "
                             "integer 1-100") from None
        self.schema_prompt = os.environ.get("TF_DSV41_SCHEMA_PROMPT", "1").strip() != "0"
        host = getattr(engine, "grammar_host", None)
        self.grammar = host if host is not None else Host(model_dir, self._vocab(model_dir), tuple(engine.eos))
        self.memory = ReasoningMemory(int(os.environ.get("TF_DSV41_REASONING_MEMORY", "1024") or 1024))
        self.prompt_tokens = PromptTokens(self.tok, model_dir / "tokenizer.json")
        self.prompt_memo = Memo()
        self.reqlog = reqlog_mod.RequestLog.from_env(self.tok)
        self._rl = threading.local()
        if getattr(engine, "batch", None) is not None:
            self.lock = contextlib.nullcontext()            # the batcher queues concurrent requests itself
        greedy = float(self.sampling.get("temperature", 1.0)) <= 0
        print(f"[tensorfold] DeepSeek-V4.1 app: thinking {'on' if self.default_thinking else 'off'} by default, "
              f"effort {self.default_effort}; context {self.context_limit()} tokens"
              + ("; WARNING: default sampling is greedy (temperature <= 0) - a client that omits `temperature` "
                 "decodes greedily, which on a long agentic turn can loop in reasoning and return empty content"
                 if greedy else ""), flush=True)

    @staticmethod
    def _vocab(model_dir: Path) -> int:
        cfg = json.loads((model_dir / "config.json").read_text())
        return int(cfg.get("text_config", cfg).get("vocab_size", 129280))

    # -- prompts -------------------------------------------------------------------------------------------------
    def render(self, body: dict[str, Any]) -> tuple[str, bool]:
        """(the prompt text of a chat body, thinking)."""

        thinking, level = options(body, self.default_thinking, self.default_effort)
        tools = body.get("tools") or None
        if tools is not None and (not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools)):
            raise ValueError("tools must be a list of objects")
        if body.get("tool_choice") == "none":
            tools = None
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        self.memory.restore(messages)
        schema = schema_for_prompt(body) if self.schema_prompt else None
        return encoding.encode(messages, tools=tools, thinking=thinking, effort=level, response_format=schema), \
            thinking

    def prompt_ids(self, text: str) -> list[int]:
        ids = self.prompt_memo.take(text)
        ids = list(ids) if ids is not None else self.prompt_tokens.encode(text)
        if self.reqlog is not None:
            self._rl.ticket = self.reqlog.begin(ids)
        return ids

    def given_ids(self, ids: list[int]) -> list[int]:
        ids = super().given_ids(ids)
        if self.reqlog is not None:
            self._rl.ticket = self.reqlog.begin(ids)
        return ids

    def tokenize(self, body: dict[str, Any]) -> list[int]:
        if isinstance(body.get("messages"), list):
            text, _ = self.render(body)
            return list(self.tok.encode(text, add_special_tokens=False).ids)
        return super().tokenize(body)

    def check(self, body: dict[str, Any]) -> str | None:
        problem = super().check(body)
        if problem:
            return problem
        chat = "messages" in body
        try:
            if chat:
                text, _ = self.render(body)
                ids = self.prompt_tokens.encode(text)
                self.prompt_memo.put(text, ids)
                n = len(ids)
            elif isinstance(body.get("prompt"), list):
                n = len(body["prompt"])
            else:
                n = len(self.prompt_tokens.encode(str(body.get("prompt") or "")))
        except ValueError as exc:
            return str(exc)
        problem = self.grammar.check(body)
        if problem:
            return problem
        limit = self.context_limit()
        if limit is None:
            return None
        asked = body.get("max_tokens") or body.get("max_completion_tokens")
        return server.context_problem(n, int(asked) if asked else None, limit, chat=chat)

    # -- a reply -------------------------------------------------------------------------------------------------
    def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool], *,
            cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
        log = self.reqlog
        self._rl.ticket = None
        result, error = None, None
        thinking = None
        try:
            result, thinking = self._run(body, chat, emit, cancelled)
            return result
        except RequestCancelled as exc:
            result = exc.result
            raise
        except BaseException as exc:
            error = exc
            raise
        finally:
            if log is not None:
                eff = body.get("max_tokens") or body.get("max_completion_tokens") or self.max_tokens
                log.end(getattr(self._rl, "ticket", None), body=body, chat=chat, result=result, error=error,
                        thinking=thinking, max_tokens_eff=int(eff) if eff else None)

    def _run(self, body, chat, emit, cancelled):
        tools = (body.get("tools") or None) if chat and body.get("tool_choice") != "none" else None
        thinking = False
        if chat:
            text, thinking = self.render(body)
            prompt = self.prompt_ids(text)
        elif isinstance(body.get("prompt"), list):
            prompt = self.given_ids(body["prompt"])
        else:
            prompt = self.prompt_ids(str(body.get("prompt") or ""))
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or self.max_tokens)
        sampling = self.sampling_for(body, prompt)
        req = self.engine.request
        model = str(body.get("model") or "")
        req.policy = body.get("tf_policy") or (model.split("@", 1)[1] if "@" in model else None)
        req.stop_eos = not bool(body.get("ignore_eos", False))
        req.knobs = body.get("tf_knobs")
        req.grammar = self.grammar.for_request(body)
        req.background = body.get("priority") == "background" or (
            chat and is_title_request(body.get("messages"), body.get("tools")))
        streaming = bool(body.get("stream"))
        single = body.get("parallel_tool_calls", True) is False
        eos = frozenset(self.engine.eos)
        out: list[int] = []
        dec = server.StreamDecoder(self.tok, tuple(self.engine.eos))
        parser = dsml.Stream(thinking=thinking, tools=tools)
        stops = server.parse_stop(body.get("stop"))
        st = {"content": "", "sent": 0, "hit": False, "at": 0, "client": False, "first": True}
        failed: list[BaseException] = []

        def push(deltas: list[dict], finished: bool) -> None:
            for d in deltas:
                if "content" in d:
                    st["content"] += d["content"]
                elif "reasoning" in d:
                    send(server.reasoning_fields(d["reasoning"]))
                elif "tool_calls" in d and not (single and d["tool_calls"][0]["index"] > 0):
                    send(d)
            full = st["content"]
            cut = server.find_stop(full, stops, max(0, st["sent"] - max(map(len, stops), default=0) + 1)) \
                if stops else -1
            if cut >= 0:
                full, st["hit"], st["at"] = full[:cut], True, len(out)
            upto = len(full) if finished or st["hit"] else len(full) - server.stop_holdback(full, stops)
            if upto > st["sent"]:
                send({"content": full[st["sent"]:upto]})
                st["sent"] = upto

        def send(delta: dict) -> None:
            if not streaming or st["client"]:
                return
            if self.reqlog is not None and st["first"]:
                self.reqlog.first(getattr(self._rl, "ticket", None))
            st["first"] = False
            if not emit(delta):
                st["client"] = True

        def on_tokens(new: list[int]) -> bool:
            out.extend(new)
            if st["client"] or st["hit"] or failed:
                return True
            try:
                dec.add(new)
                if streaming:
                    push(parser.feed(dec.text), False)
                elif stops and not tools and server.find_stop(dec.text, stops) >= 0:
                    st["hit"], st["at"] = True, len(out)
                if not st["client"] and cancelled is not None and cancelled():
                    st["client"] = True
            except Exception as exc:          # noqa: BLE001  raised after generate, never into the engine
                failed.append(exc)
                return True
            return st["client"] or st["hit"]

        if cancelled is not None:
            on_tokens.cancelled = cancelled
        draft = body.get("draft", True) is not False
        with self.lock:
            if cancelled is not None and cancelled():
                raise RequestCancelled("the client left before the request started")
            stats = self.engine.generate(prompt, max_tokens, sampling, on_tokens,
                                         **({} if draft else {"draft": False}))
        if failed:
            raise failed[0]
        if st["client"]:
            raise RequestCancelled("the client left during the reply", result={
                "prompt_tokens": len(prompt), "completion_tokens": len(out), "finish": "cancelled",
                "stats": dict(stats or {}, cancelled=True)})
        text = self.tok.decode([t for t in out if t not in eos], skip_special_tokens=False)
        reply = dsml.parse(text, thinking=thinking, tools=tools)
        if streaming:
            push(parser.finish(text), True)
            calls = parser.calls[:1] if single else parser.calls
            content = st["content"][:st["sent"]]
        else:
            calls = reply.calls[:1] if single else reply.calls
            content = reply.content
            if stops:
                cut = server.find_stop(content, stops)
                if cut >= 0:
                    content, st["hit"] = content[:cut], True
                    st["at"] = st["at"] or len(out)
        call_dicts = [c.openai() for c in calls]
        if chat:
            self.memory.put(reply.reasoning, content, call_dicts)
        if call_dicts:
            finish = "tool_calls"
        elif st["hit"] or (out and out[-1] in eos):
            finish = "stop"
        else:
            finish = "length"
        used = st["at"] if st["hit"] and st["at"] else len(out)
        if body.get("return_token_ids"):
            stats = dict(stats or {}, token_ids=[int(t) for t in out[:used]])
        return {"final": {}, "calls": None if streaming else (call_dicts or None), "finish": finish,
                "content": content, "reasoning": reply.reasoning, "prompt_tokens": len(prompt),
                "completion_tokens": used, "stats": stats}, thinking
