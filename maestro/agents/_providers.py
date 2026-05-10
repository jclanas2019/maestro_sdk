"""
maestro.agents._providers — Native LLM adapters for Anthropic and OpenAI.

Provides a unified ``LLMAdapter`` interface that works with all Maestro
agent components (``NativeAgentWork``, ``AgentRecordProcessor``,
``AgentCondition``) without requiring LangChain or LangGraph.

    from maestro.agents import AnthropicAdapter, OpenAIAdapter
    from maestro.agents import AnthropicModels, OpenAIModels

    llm = AnthropicAdapter(model=AnthropicModels.CLAUDE_SONNET)
    # or
    llm = OpenAIAdapter(model=OpenAIModels.GPT_4O)

Both adapters support:

* Synchronous and async chat (``chat`` / ``achat``)
* Tool calling — pass Maestro tools; the adapter handles schema conversion
* Token usage tracking — integrates with ``maestro.observe``
* Rate limit retry — integrates with ``maestro.retry``
* Streaming callbacks for long responses

Install::

    pip install "maestro-sdk[anthropic]"    # pip install anthropic
    pip install "maestro-sdk[openai]"       # pip install openai
    pip install "maestro-sdk[llm]"          # both providers
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  Model name constants
# ════════════════════════════════════════════════════════════════════════════

class AnthropicModels:
    """Current Anthropic model identifiers."""
    CLAUDE_OPUS    = "claude-opus-4-6"
    CLAUDE_SONNET  = "claude-sonnet-4-6"
    CLAUDE_HAIKU   = "claude-haiku-4-5-20251001"
    # Aliases
    OPUS    = CLAUDE_OPUS
    SONNET  = CLAUDE_SONNET
    HAIKU   = CLAUDE_HAIKU
    DEFAULT = CLAUDE_SONNET


class OpenAIModels:
    """Current OpenAI model identifiers."""
    GPT_4O       = "gpt-4o"
    GPT_4O_MINI  = "gpt-4o-mini"
    O1           = "o1"
    O1_MINI      = "o1-mini"
    O3_MINI      = "o3-mini"
    DEFAULT      = GPT_4O


# ════════════════════════════════════════════════════════════════════════════
#  Core data types
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolCall:
    """A single tool invocation requested by the LLM."""
    id:        str
    name:      str
    arguments: dict[str, Any]


@dataclass
class TokenUsage:
    """Token counts from an LLM response."""
    input_tokens:  int  = 0
    output_tokens: int  = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMResponse:
    """
    Normalised response from any LLM provider.

    ``content``    — text response (may be empty when tool calls are present).
    ``tool_calls`` — list of tools the LLM wants to call.
    ``stop_reason``— ``"end_turn"`` | ``"tool_use"`` | ``"stop"`` | ``"length"``.
    ``usage``      — token counts.
    ``raw``        — original provider response object.
    """
    content:    str
    tool_calls: list[ToolCall]   = field(default_factory=list)
    stop_reason: str             = "end_turn"
    usage:      TokenUsage       = field(default_factory=TokenUsage)
    raw:        Any              = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_done(self) -> bool:
        return not self.has_tool_calls


@dataclass
class Message:
    """
    Provider-agnostic message for building conversation history.

    ``role``         — ``"user"`` | ``"assistant"`` | ``"system"`` | ``"tool"``
    ``content``      — text content
    ``tool_calls``   — tool calls in this message (set by LLM responses)
    ``tool_call_id`` — ID of the tool call this message responds to
    """
    role:         str
    content:      str
    tool_calls:   list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None

    @staticmethod
    def user(content: str) -> "Message":
        return Message(role="user", content=content)

    @staticmethod
    def system(content: str) -> "Message":
        return Message(role="system", content=content)

    @staticmethod
    def assistant(content: str, tool_calls: list[ToolCall] | None = None) -> "Message":
        return Message(role="assistant", content=content, tool_calls=tool_calls or [])

    @staticmethod
    def tool_result(tool_call_id: str, content: str) -> "Message":
        return Message(role="tool", content=content, tool_call_id=tool_call_id)

    def __repr__(self) -> str:
        snippet = self.content[:60].replace("\n", " ") if self.content else "(no content)"
        tcs = f" [{len(self.tool_calls)} tool call(s)]" if self.tool_calls else ""
        return f"Message(role={self.role!r}, content={snippet!r}{tcs})"


# ════════════════════════════════════════════════════════════════════════════
#  Tool schema conversion
# ════════════════════════════════════════════════════════════════════════════

def _maestro_tool_to_anthropic(tool) -> dict:
    """Convert a Maestro/LangChain tool to Anthropic's tool format."""
    schema = dict(tool.args_schema.model_json_schema())
    schema.pop("title", None)
    return {
        "name":         tool.name,
        "description":  tool.description,
        "input_schema": schema,
    }


def _maestro_tool_to_openai(tool) -> dict:
    """Convert a Maestro/LangChain tool to OpenAI's function tool format."""
    schema = dict(tool.args_schema.model_json_schema())
    schema.pop("title", None)
    return {
        "type":     "function",
        "function": {
            "name":        tool.name,
            "description": tool.description,
            "parameters":  schema,
        },
    }


def _execute_tool(tool, arguments: dict) -> str:
    """Execute a Maestro/LangChain tool and return the string result."""
    try:
        result = tool._run(**arguments)
        return str(result) if not isinstance(result, str) else result
    except Exception as exc:
        logger.warning("Tool '%s' raised: %s", tool.name, exc)
        return json.dumps({"error": str(exc)})


# ════════════════════════════════════════════════════════════════════════════
#  LLMAdapter — abstract base
# ════════════════════════════════════════════════════════════════════════════

class LLMAdapter(ABC):
    """
    Abstract base for all Maestro LLM adapters.

    Subclass this to add support for any LLM provider.

    Implementations: ``AnthropicAdapter``, ``OpenAIAdapter``.
    """

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools:    Optional[list] = None,
    ) -> LLMResponse:
        """Synchronous chat completion."""

    async def achat(
        self,
        messages: list[Message],
        tools:    Optional[list] = None,
    ) -> LLMResponse:
        """Async chat completion. Default: runs ``chat`` in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.chat, messages, tools)

    @property
    @abstractmethod
    def model(self) -> str:
        """The model identifier string."""

    @property
    @abstractmethod
    def provider(self) -> str:
        """The provider name, e.g. ``"anthropic"`` or ``"openai"``."""

    # ── Convenience decorators ──────────────────────────────────────────── #

    def with_retry(
        self,
        max_attempts: int = 3,
        on:           Optional[list] = None,
    ) -> "_RetryingAdapter":
        """
        Wrap this adapter with automatic retry on transient errors.

        By default retries on ``anthropic.RateLimitError``,
        ``openai.RateLimitError``, and ``ConnectionError``.

        Example::

            llm = AnthropicAdapter().with_retry(max_attempts=5)
        """
        return _RetryingAdapter(self, max_attempts=max_attempts, on=on)

    def with_observer(self, observer) -> "_ObservingAdapter":
        """
        Wrap this adapter to emit token usage and latency metrics.

        Example::

            from maestro.observe import InMemoryObserver
            obs = InMemoryObserver()
            llm = AnthropicAdapter().with_observer(obs)
        """
        return _ObservingAdapter(self, observer)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"


# ════════════════════════════════════════════════════════════════════════════
#  AnthropicAdapter
# ════════════════════════════════════════════════════════════════════════════

class AnthropicAdapter(LLMAdapter):
    """
    Native Anthropic Messages API adapter.

    Parameters
    ----------
    model:
        Model identifier (default: ``AnthropicModels.CLAUDE_SONNET``).
    api_key:
        Anthropic API key. If not provided, reads ``ANTHROPIC_API_KEY``
        environment variable.
    max_tokens:
        Maximum tokens in the response (required by Anthropic).
    temperature:
        Sampling temperature (0.0 – 1.0).
    timeout:
        Request timeout in seconds.
    extra_headers:
        Additional HTTP headers passed to every request.

    Example::

        from maestro.agents import AnthropicAdapter, AnthropicModels

        llm = AnthropicAdapter(model=AnthropicModels.HAIKU)
        resp = llm.chat([Message.user("What is 2 + 2?")])
        print(resp.content)   # "4"
    """

    def __init__(
        self,
        model:         str            = AnthropicModels.DEFAULT,
        api_key:       Optional[str]  = None,
        max_tokens:    int            = 4096,
        temperature:   float          = 1.0,
        timeout:       float          = 60.0,
        extra_headers: Optional[dict] = None,
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as e:
            raise ImportError(
                "pip install anthropic  (or: pip install 'maestro-sdk[anthropic]')"
            ) from e

        self._client       = _anthropic.Anthropic(api_key=api_key)
        self._model        = model
        self._max_tokens   = max_tokens
        self._temperature  = temperature
        self._timeout      = timeout
        self._extra_headers = extra_headers or {}
        self._anthropic    = _anthropic

    @property
    def model(self) -> str:    return self._model
    @property
    def provider(self) -> str: return "anthropic"

    def chat(
        self,
        messages: list[Message],
        tools:    Optional[list] = None,
    ) -> LLMResponse:
        system, api_messages = self._build_messages(messages)

        kwargs: dict = {
            "model":      self._model,
            "max_tokens": self._max_tokens,
            "messages":   api_messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [_maestro_tool_to_anthropic(t) for t in tools]
        if self._temperature != 1.0:
            kwargs["temperature"] = self._temperature

        logger.debug("Anthropic(%s): %d message(s), %d tool(s)",
                     self._model, len(api_messages), len(tools or []))

        response = self._client.messages.create(**kwargs)
        return self._parse_response(response)

    def _build_messages(self, messages: list[Message]) -> tuple[str, list[dict]]:
        """Split system message out; convert the rest to Anthropic format."""
        system_parts = []
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
                continue

            if msg.role == "tool":
                # Tool result — append to last user message or create new one
                tool_result_block = {
                    "type":        "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content":     msg.content,
                }
                if api_messages and api_messages[-1]["role"] == "user" and \
                   isinstance(api_messages[-1]["content"], list):
                    api_messages[-1]["content"].append(tool_result_block)
                else:
                    api_messages.append({
                        "role":    "user",
                        "content": [tool_result_block],
                    })
                continue

            if msg.role == "assistant" and msg.tool_calls:
                content_blocks = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type":  "tool_use",
                        "id":    tc.id,
                        "name":  tc.name,
                        "input": tc.arguments,
                    })
                api_messages.append({"role": "assistant", "content": content_blocks})
                continue

            api_messages.append({"role": msg.role, "content": msg.content})

        return "\n\n".join(system_parts), api_messages

    def _parse_response(self, response) -> LLMResponse:
        content = ""
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id        = block.id,
                    name      = block.name,
                    arguments = dict(block.input),
                ))

        usage = TokenUsage(
            input_tokens  = response.usage.input_tokens,
            output_tokens = response.usage.output_tokens,
        )
        stop_map = {"end_turn": "end_turn", "tool_use": "tool_use",
                    "max_tokens": "length", "stop_sequence": "stop"}
        return LLMResponse(
            content     = content,
            tool_calls  = tool_calls,
            stop_reason = stop_map.get(response.stop_reason, response.stop_reason),
            usage       = usage,
            raw         = response,
        )

    # ── Streaming ─────────────────────────────────────────────────────── #

    def stream(
        self,
        messages:  list[Message],
        tools:     Optional[list] = None,
        on_text:   Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        """
        Stream a response. ``on_text`` is called with each text chunk.

        Example::

            def print_chunk(chunk): print(chunk, end="", flush=True)
            resp = llm.stream([Message.user("Tell me a story.")], on_text=print_chunk)
        """
        system, api_messages = self._build_messages(messages)
        kwargs: dict = {
            "model":      self._model,
            "max_tokens": self._max_tokens,
            "messages":   api_messages,
        }
        if system: kwargs["system"] = system
        if tools:  kwargs["tools"] = [_maestro_tool_to_anthropic(t) for t in tools]

        full_text   = []
        tool_calls: list[ToolCall] = []
        usage       = TokenUsage()

        with self._client.messages.stream(**kwargs) as stream:
            for event in stream:
                if hasattr(event, "type"):
                    if event.type == "content_block_delta":
                        delta = event.delta
                        if hasattr(delta, "text"):
                            full_text.append(delta.text)
                            if on_text: on_text(delta.text)
            final = stream.get_final_message()
            return self._parse_response(final)


# ════════════════════════════════════════════════════════════════════════════
#  OpenAIAdapter
# ════════════════════════════════════════════════════════════════════════════

class OpenAIAdapter(LLMAdapter):
    """
    Native OpenAI Chat Completions API adapter.

    Parameters
    ----------
    model:
        Model identifier (default: ``OpenAIModels.GPT_4O``).
    api_key:
        OpenAI API key. If not provided, reads ``OPENAI_API_KEY``.
    base_url:
        Override the API base URL (useful for Azure OpenAI or local models).
    temperature:
        Sampling temperature.
    max_tokens:
        Maximum response tokens (``None`` = model default).
    timeout:
        Request timeout in seconds.
    response_format:
        Pass ``{"type": "json_object"}`` to enable JSON mode.

    Example::

        from maestro.agents import OpenAIAdapter, OpenAIModels

        llm = OpenAIAdapter(model=OpenAIModels.GPT_4O_MINI)
        resp = llm.chat([Message.user("What is 2 + 2?")])
        print(resp.content)   # "4"
    """

    def __init__(
        self,
        model:           str           = OpenAIModels.DEFAULT,
        api_key:         Optional[str] = None,
        base_url:        Optional[str] = None,
        temperature:     float         = 1.0,
        max_tokens:      Optional[int] = None,
        timeout:         float         = 60.0,
        response_format: Optional[dict]= None,
    ) -> None:
        try:
            import openai as _openai
        except ImportError as e:
            raise ImportError(
                "pip install openai  (or: pip install 'maestro-sdk[openai]')"
            ) from e

        client_kwargs: dict = {"api_key": api_key}
        if base_url: client_kwargs["base_url"] = base_url

        self._client          = _openai.OpenAI(**client_kwargs)
        self._model           = model
        self._temperature     = temperature
        self._max_tokens      = max_tokens
        self._timeout         = timeout
        self._response_format = response_format
        self._openai          = _openai

    @property
    def model(self) -> str:    return self._model
    @property
    def provider(self) -> str: return "openai"

    def chat(
        self,
        messages: list[Message],
        tools:    Optional[list] = None,
    ) -> LLMResponse:
        api_messages = self._build_messages(messages)
        kwargs: dict = {
            "model":       self._model,
            "messages":    api_messages,
            "temperature": self._temperature,
        }
        if self._max_tokens:
            kwargs["max_tokens"] = self._max_tokens
        if tools:
            kwargs["tools"]      = [_maestro_tool_to_openai(t) for t in tools]
            kwargs["tool_choice"] = "auto"
        if self._response_format:
            kwargs["response_format"] = self._response_format

        logger.debug("OpenAI(%s): %d message(s), %d tool(s)",
                     self._model, len(api_messages), len(tools or []))

        response = self._client.chat.completions.create(**kwargs)
        return self._parse_response(response)

    def _build_messages(self, messages: list[Message]) -> list[dict]:
        api_messages = []
        for msg in messages:
            if msg.role == "tool":
                api_messages.append({
                    "role":         "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content":      msg.content,
                })
            elif msg.role == "assistant" and msg.tool_calls:
                api_messages.append({
                    "role":       "assistant",
                    "content":    msg.content or None,
                    "tool_calls": [
                        {
                            "id":       tc.id,
                            "type":     "function",
                            "function": {
                                "name":      tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })
            else:
                api_messages.append({"role": msg.role, "content": msg.content})
        return api_messages

    def _parse_response(self, response) -> LLMResponse:
        choice  = response.choices[0]
        message = choice.message
        content = message.content or ""
        tool_calls: list[ToolCall] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id        = tc.id,
                    name      = tc.function.name,
                    arguments = args,
                ))

        usage = TokenUsage(
            input_tokens  = response.usage.prompt_tokens,
            output_tokens = response.usage.completion_tokens,
        )
        finish_map = {"stop": "end_turn", "tool_calls": "tool_use", "length": "length"}
        return LLMResponse(
            content     = content,
            tool_calls  = tool_calls,
            stop_reason = finish_map.get(choice.finish_reason, choice.finish_reason),
            usage       = usage,
            raw         = response,
        )

    # ── Streaming ─────────────────────────────────────────────────────── #

    def stream(
        self,
        messages: list[Message],
        tools:    Optional[list] = None,
        on_text:  Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        """
        Stream a response, calling ``on_text`` for each text chunk.

        Makes a single API call — usage stats are accumulated from stream events.
        """
        api_messages = self._build_messages(messages)
        kwargs: dict = {
            "model":    self._model,
            "messages": api_messages,
            "stream":   True,
            "stream_options": {"include_usage": True},
        }
        if tools: kwargs["tools"] = [_maestro_tool_to_openai(t) for t in tools]

        full_text:  list[str]    = []
        tool_calls: list[ToolCall] = []
        usage       = TokenUsage()
        finish_reason = "stop"

        for chunk in self._client.chat.completions.create(**kwargs):
            if not chunk.choices and hasattr(chunk, "usage") and chunk.usage:
                # Final usage chunk when stream_options.include_usage=True
                usage = TokenUsage(
                    input_tokens  = chunk.usage.prompt_tokens,
                    output_tokens = chunk.usage.completion_tokens,
                )
                continue
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                full_text.append(delta.content)
                if on_text: on_text(delta.content)
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        finish_map = {"stop": "end_turn", "tool_calls": "tool_use", "length": "length"}
        return LLMResponse(
            content     = "".join(full_text),
            tool_calls  = tool_calls,
            stop_reason = finish_map.get(finish_reason, finish_reason),
            usage       = usage,
        )


# ════════════════════════════════════════════════════════════════════════════
#  Wrapper adapters
# ════════════════════════════════════════════════════════════════════════════

class _RetryingAdapter(LLMAdapter):
    """Adapter that retries on rate limit and transient errors."""

    def __init__(
        self,
        base:         LLMAdapter,
        max_attempts: int = 3,
        on:           Optional[list] = None,
    ) -> None:
        from maestro.retry import ExponentialBackoff, RetryPolicy, execute_with_retry

        self._base = base
        self._execute = execute_with_retry

        # Default: retry on common transient errors
        default_on = []
        try:
            import anthropic
            default_on += [anthropic.RateLimitError, anthropic.InternalServerError,
                           anthropic.APIConnectionError]
        except ImportError: pass
        try:
            import openai
            default_on += [openai.RateLimitError, openai.InternalServerError,
                           openai.APIConnectionError]
        except ImportError: pass
        default_on.append(ConnectionError)

        self._policy = RetryPolicy(
            max_attempts = max_attempts,
            backoff      = ExponentialBackoff(base=2.0, multiplier=2.0, max_delay=60.0),
            on           = on or default_on,
        )

    @property
    def model(self) -> str:    return self._base.model
    @property
    def provider(self) -> str: return self._base.provider

    def chat(self, messages, tools=None):
        return self._execute(lambda: self._base.chat(messages, tools), self._policy)

    async def achat(self, messages, tools=None):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.chat, messages, tools)


class _ObservingAdapter(LLMAdapter):
    """Adapter that emits LLM metrics into a Maestro observer."""

    def __init__(self, base: LLMAdapter, observer) -> None:
        self._base = base
        self._obs  = observer

    @property
    def model(self) -> str:    return self._base.model
    @property
    def provider(self) -> str: return self._base.provider

    def _emit(self, response: LLMResponse, duration: float) -> None:
        from maestro.observe import MetricEvent
        labels = {"model": self._base.model, "provider": self._base.provider}
        self._obs.on_event(MetricEvent("agents", "llm_call",           1.0,                              labels, kind="counter"))
        self._obs.on_event(MetricEvent("agents", "tokens_input",       response.usage.input_tokens,      labels, kind="gauge"))
        self._obs.on_event(MetricEvent("agents", "tokens_output",      response.usage.output_tokens,     labels, kind="gauge"))
        self._obs.on_event(MetricEvent("agents", "llm_duration_seconds", duration,                      labels, kind="histogram"))
        self._obs.on_event(MetricEvent("agents", "tool_calls_per_turn", len(response.tool_calls),        labels, kind="gauge"))

    def chat(self, messages, tools=None):
        t0       = time.monotonic()
        response = self._base.chat(messages, tools)
        self._emit(response, time.monotonic() - t0)
        return response

    async def achat(self, messages, tools=None):
        t0       = time.monotonic()
        response = await self._base.achat(messages, tools)
        self._emit(response, time.monotonic() - t0)
        return response


# ════════════════════════════════════════════════════════════════════════════
#  Factory helpers
# ════════════════════════════════════════════════════════════════════════════

def make_anthropic(
    model:       Optional[str]  = None,
    api_key:     Optional[str]  = None,
    max_tokens:  Optional[int]  = None,
    temperature: Optional[float]= None,
    **kwargs,
) -> AnthropicAdapter:
    """
    Create an ``AnthropicAdapter``.

    All parameters fall back to the Maestro config (loaded from ``.env``) when
    not explicitly provided.

    Example::

        # Without .env — uses built-in defaults
        llm = make_anthropic()

        # With .env containing ANTHROPIC_API_KEY and MAESTRO_ANTHROPIC_MODEL
        maestro.load_env()
        llm = make_anthropic()    # api_key and model come from .env

        # Explicit override
        llm = make_anthropic(model="claude-sonnet-4-6", max_tokens=8192)
    """
    try:
        from maestro.config import get_config
        cfg = get_config()
    except Exception as _exc:
        logger.debug("config unavailable, using defaults: %s", _exc)
        cfg = None

    resolved_model       = model       or (cfg.anthropic_model       if cfg else AnthropicModels.DEFAULT)
    resolved_api_key     = api_key     or (cfg.anthropic_api_key     if cfg else None)
    resolved_max_tokens  = max_tokens  or (cfg.anthropic_max_tokens  if cfg else 4096)
    resolved_temperature = temperature if temperature is not None else (cfg.anthropic_temperature if cfg else 1.0)

    return AnthropicAdapter(
        model       = resolved_model,
        api_key     = resolved_api_key,
        max_tokens  = resolved_max_tokens,
        temperature = resolved_temperature,
        **kwargs,
    )


def make_openai(
    model:       Optional[str]  = None,
    api_key:     Optional[str]  = None,
    base_url:    Optional[str]  = None,
    temperature: Optional[float]= None,
    max_tokens:  Optional[int]  = None,
    **kwargs,
) -> OpenAIAdapter:
    """
    Create an ``OpenAIAdapter``.

    All parameters fall back to the Maestro config (loaded from ``.env``) when
    not explicitly provided.

    Example::

        # Without .env — uses built-in defaults (gpt-4o-mini)
        llm = make_openai()

        # With .env containing OPENAI_API_KEY and MAESTRO_OPENAI_MODEL
        maestro.load_env()
        llm = make_openai()    # api_key and model come from .env

        # Any OpenAI-compatible endpoint (Ollama, Azure, Groq…)
        llm = make_openai(base_url="http://localhost:11434/v1", model="llama3")
        # or via .env: MAESTRO_OPENAI_BASE_URL=http://localhost:11434/v1
    """
    try:
        from maestro.config import get_config
        cfg = get_config()
    except Exception as _exc:
        logger.debug("config unavailable, using defaults: %s", _exc)
        cfg = None

    resolved_model       = model    or (cfg.openai_model       if cfg else OpenAIModels.DEFAULT)
    resolved_api_key     = api_key  or (cfg.openai_api_key     if cfg else None)
    resolved_base_url    = base_url or (cfg.openai_base_url    if cfg else None)
    resolved_temperature = temperature if temperature is not None else (cfg.openai_temperature if cfg else 1.0)

    return OpenAIAdapter(
        model       = resolved_model,
        api_key     = resolved_api_key,
        base_url    = resolved_base_url,
        temperature = resolved_temperature,
        max_tokens  = max_tokens,
        **kwargs,
    )


__all__ = [
    "AnthropicModels", "OpenAIModels",
    "ToolCall", "TokenUsage", "LLMResponse", "Message",
    "LLMAdapter", "AnthropicAdapter", "OpenAIAdapter",
    "make_anthropic", "make_openai",
    "_maestro_tool_to_anthropic", "_maestro_tool_to_openai", "_execute_tool",
]
