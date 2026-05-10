"""
maestro.agents._native — Native agentic patterns using the Maestro LLM adapters.

Works entirely without LangChain or LangGraph — only requires an
``LLMAdapter`` (``AnthropicAdapter`` or ``OpenAIAdapter``) and
standard Maestro components.

Classes
-------
AgentResult         Structured output of a ``NativeReActAgent`` run.
NativeReActAgent    ReAct loop driven by a Maestro LLM adapter.
NativeAgentWork     Maestro Work that runs a NativeReActAgent.
NativeAgentBatchProcessor  LLM-powered batch record enrichment (native API).

Usage::

    from maestro.agents import AnthropicAdapter, NativeAgentWork
    from maestro.agents import RulesEngineTool, SagaTool

    llm = AnthropicAdapter(model="claude-haiku-4-5-20251001")
    work = NativeAgentWork(
        llm           = llm,
        maestro_tools = [RulesEngineTool(rules=pricing_rules)],
        system_prompt = "You are an order processing agent.",
        input_fn      = lambda ctx: f"Process order: {ctx.as_map()}",
        output_fn     = lambda result, ctx: ctx.put("agent_answer", result.final_response),
    )
    flow = maestro.aNewSequentialFlow().execute(work).build()
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from maestro.flows._work import DefaultWorkReport, Work, WorkContext, WorkReport, WorkStatus

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  AgentResult
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentResult:
    """
    Complete result of a ``NativeReActAgent`` run.

    Attributes
    ----------
    final_response:
        The last text response from the LLM after all tool calls are resolved.
    messages:
        Full conversation history (user → assistant → tool → assistant …).
    tool_calls_made:
        Ordered list of tool names called during the run.
    usage:
        Aggregated token usage across all LLM turns.
    iterations:
        Number of LLM turns taken.
    context:
        Final shared context dict (updated by tools via AgentCondition etc.).
    """
    from maestro.agents._providers import TokenUsage, Message

    final_response:  str
    messages:        list  = field(default_factory=list)
    tool_calls_made: list[str] = field(default_factory=list)
    usage:           Any   = None   # TokenUsage
    iterations:      int   = 0
    context:         dict  = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return bool(self.final_response)

    def __str__(self) -> str:
        return (
            f"AgentResult(\n"
            f"  final_response = {self.final_response[:100]!r}{'...' if len(self.final_response)>100 else ''}\n"
            f"  tool_calls     = {self.tool_calls_made}\n"
            f"  iterations     = {self.iterations}\n"
            f"  tokens         = {self.usage.total_tokens if self.usage else 'unknown'}\n"
            f")"
        )


# ════════════════════════════════════════════════════════════════════════════
#  NativeReActAgent
# ════════════════════════════════════════════════════════════════════════════

class NativeReActAgent:
    """
    A ReAct (Reason + Act) agent loop that uses Maestro's LLM adapters
    directly — no LangChain, no LangGraph.

    Supports both Anthropic and OpenAI APIs via the ``LLMAdapter`` interface.
    Maestro tools (``RulesEngineTool``, ``SagaTool``, etc.) are passed as-is;
    the agent handles schema conversion and tool execution internally.

    Parameters
    ----------
    llm:
        An ``AnthropicAdapter`` or ``OpenAIAdapter`` (or any ``LLMAdapter``).
    tools:
        List of Maestro/LangChain tools available to the agent.
    system_prompt:
        Injected as the first system message.
    max_iterations:
        Hard cap on the number of LLM turns (prevents infinite loops).
    on_tool_call:
        Optional callback invoked before each tool execution:
        ``(tool_name, arguments) → None``.
    on_tool_result:
        Optional callback invoked after each tool execution:
        ``(tool_name, result) → None``.

    Example::

        from maestro.agents import AnthropicAdapter, NativeReActAgent
        from maestro.agents import RulesEngineTool

        agent = NativeReActAgent(
            llm    = AnthropicAdapter(),
            tools  = [RulesEngineTool(rules=pricing_rules, name="pricing")],
            system_prompt = "You are a pricing agent.",
        )
        result = agent.run("Apply pricing for customer Alice, total $200, tier vip.")
        print(result.final_response)
        print(result.tool_calls_made)
    """

    def __init__(
        self,
        llm,
        tools:           Optional[list] = None,
        system_prompt:   str            = "",
        max_iterations:  int            = 15,
        on_tool_call:    Optional[Callable] = None,
        on_tool_result:  Optional[Callable] = None,
    ) -> None:
        from maestro.agents._providers import TokenUsage, _execute_tool

        self._llm           = llm
        self._tools         = tools or []
        self._system_prompt = system_prompt
        self._max_iter      = max_iterations
        self._on_call       = on_tool_call
        self._on_result     = on_tool_result
        self._tool_registry = {t.name: t for t in self._tools}
        self._exec_tool     = _execute_tool
        self._TokenUsage    = TokenUsage

    def run(
        self,
        query:   str,
        context: Optional[dict] = None,
        history: Optional[list] = None,
    ) -> "AgentResult":
        """
        Run the ReAct loop for a single query.

        Parameters
        ----------
        query:
            The user question or instruction.
        context:
            Optional dict of contextual facts prepended to the query.
        history:
            Prior conversation messages to include for multi-turn sessions.
        """
        from maestro.agents._providers import Message, TokenUsage

        messages: list[Message] = list(history or [])

        if self._system_prompt and not any(m.role == "system" for m in messages):
            messages.insert(0, Message.system(self._system_prompt))

        if context:
            user_content = f"Context:\n{json.dumps(context, indent=2)}\n\nTask: {query}"
        else:
            user_content = query
        messages.append(Message.user(user_content))

        tool_calls_made: list[str] = []
        total_usage = TokenUsage()
        iterations  = 0

        for _ in range(self._max_iter):
            response = self._llm.chat(messages, tools=self._tools or None)
            iterations += 1
            total_usage.input_tokens  += response.usage.input_tokens
            total_usage.output_tokens += response.usage.output_tokens

            logger.debug(
                "NativeReActAgent: turn %d — %d tool call(s), %d token(s)",
                iterations, len(response.tool_calls), response.usage.total_tokens,
            )

            # Add assistant message to history
            messages.append(Message.assistant(response.content, response.tool_calls))

            if not response.tool_calls:
                logger.info("NativeReActAgent: done after %d turn(s) — %d total token(s)",
                            iterations, total_usage.total_tokens)
                return AgentResult(
                    final_response  = response.content,
                    messages        = messages,
                    tool_calls_made = tool_calls_made,
                    usage           = total_usage,
                    iterations      = iterations,
                    context         = context or {},
                )

            # Execute all requested tool calls
            for tc in response.tool_calls:
                if self._on_call:
                    try: self._on_call(tc.name, tc.arguments)
                    except Exception: pass

                tool = self._tool_registry.get(tc.name)
                if tool:
                    result = self._exec_tool(tool, tc.arguments)
                    tool_calls_made.append(tc.name)
                else:
                    result = json.dumps({"error": f"Unknown tool: {tc.name}"})

                logger.debug("Tool '%s' → %s", tc.name, result[:100])

                if self._on_result:
                    try: self._on_result(tc.name, result)
                    except Exception: pass

                messages.append(Message.tool_result(tc.id, result))

        # Safety: max iterations reached
        logger.warning("NativeReActAgent: max_iterations (%d) reached — returning last response",
                       self._max_iter)
        last_text = next(
            (m.content for m in reversed(messages) if m.role == "assistant" and m.content),
            "Max iterations reached without a final answer.",
        )
        return AgentResult(
            final_response  = last_text,
            messages        = messages,
            tool_calls_made = tool_calls_made,
            usage           = total_usage,
            iterations      = iterations,
            context         = context or {},
        )

    def stream(
        self,
        query:    str,
        context:  Optional[dict] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> Iterator["AgentResult"]:
        """
        Run the agent with streaming text output.

        Yields intermediate text chunks via ``on_chunk``, then returns the
        final ``AgentResult``.  Requires a streaming-capable adapter.
        """
        if not hasattr(self._llm, "stream"):
            yield self.run(query, context)
            return

        from maestro.agents._providers import Message, TokenUsage

        messages: list[Message] = []
        if self._system_prompt:
            messages.insert(0, Message.system(self._system_prompt))

        content = (f"Context:\n{json.dumps(context)}\n\nTask: {query}"
                   if context else query)
        messages.append(Message.user(content))

        response = self._llm.stream(messages, tools=self._tools or None, on_text=on_chunk)
        messages.append(Message.assistant(response.content, response.tool_calls))

        if not response.tool_calls:
            yield AgentResult(
                final_response = response.content,
                messages       = messages,
                usage          = response.usage,
                iterations     = 1,
                context        = context or {},
            )
        else:
            # Fall back to non-streaming for tool-call resolution
            yield self.run(query, context)


# ════════════════════════════════════════════════════════════════════════════
#  NativeAgentWork — Maestro Work backed by a NativeReActAgent
# ════════════════════════════════════════════════════════════════════════════

class NativeAgentWork(Work):
    """
    A Maestro ``Work`` step that runs a ``NativeReActAgent``.

    Works without LangChain or LangGraph — only requires an
    ``LLMAdapter`` and standard Maestro tools.

    Parameters
    ----------
    llm:
        ``AnthropicAdapter`` or ``OpenAIAdapter``.
    maestro_tools:
        Maestro/LangChain tools available to the agent.
    system_prompt:
        System prompt injected at the start of every run.
    input_fn:
        ``(WorkContext) → str | dict`` — builds the agent's input query.
        If a dict with ``"query"`` and optional ``"context"`` keys is returned,
        both are passed to the agent.  If a string, passed as the query.
    output_fn:
        ``(AgentResult, WorkContext) → None`` — writes agent output into context.
        Default: writes ``final_response``, ``tool_calls_made``, and ``usage``
        (as a dict) into the WorkContext.
    max_iterations:
        Hard cap on the agent loop.
    name:
        Display name for logging.

    Example::

        from maestro.agents import AnthropicAdapter, NativeAgentWork, RulesEngineTool

        work = NativeAgentWork(
            llm           = AnthropicAdapter(model="claude-haiku-4-5-20251001"),
            maestro_tools = [
                RulesEngineTool(rules=pricing_rules, name="apply_pricing"),
                SagaTool(saga=checkout_saga, name="checkout"),
            ],
            system_prompt = "You are an order processing agent. Use tools to process orders.",
            input_fn      = lambda ctx: {
                "query":   f"Process order {ctx.get('order_id')}",
                "context": {"total": ctx.get("total"), "tier": ctx.get("tier")},
            },
            output_fn = lambda r, ctx: (
                ctx.put("answer",          r.final_response),
                ctx.put("tools_used",      r.tool_calls_made),
                ctx.put("tokens_used",     r.usage.total_tokens if r.usage else 0),
            ),
            name = "order-agent",
        )

        flow = maestro.aNewSequentialFlow().execute(work).then(persist_work).build()
    """

    def __init__(
        self,
        llm,
        maestro_tools:  Optional[list] = None,
        system_prompt:  str            = "",
        input_fn:       Optional[Callable] = None,
        output_fn:      Optional[Callable] = None,
        max_iterations: int            = 15,
        name:           str            = "native-agent-work",
    ) -> None:
        self._llm     = llm
        self._tools   = maestro_tools or []
        self._system  = system_prompt
        self._input   = input_fn  or (lambda ctx: ctx.as_map().get("query", "Process this."))
        self._output  = output_fn or _default_output_fn
        self._max_iter = max_iterations
        self._name    = name

    def get_name(self) -> str:
        return self._name

    def execute(self, work_context: WorkContext) -> WorkReport:
        agent = NativeReActAgent(
            llm           = self._llm,
            tools         = self._tools,
            system_prompt = self._system,
            max_iterations= self._max_iter,
        )

        raw_input = self._input(work_context)
        if isinstance(raw_input, dict):
            query   = raw_input.get("query", str(raw_input))
            context = raw_input.get("context") or work_context.as_map()
        else:
            query   = str(raw_input)
            context = work_context.as_map()

        try:
            result = agent.run(query, context=context)
            self._output(result, work_context)
            logger.info(
                "NativeAgentWork '%s': completed in %d turn(s), %d tool call(s)",
                self._name, result.iterations, len(result.tool_calls_made),
            )
            return DefaultWorkReport(WorkStatus.COMPLETED, work_context)
        except Exception as exc:
            logger.error("NativeAgentWork '%s': failed — %s", self._name, exc)
            return DefaultWorkReport(WorkStatus.FAILED, work_context, error=exc)


def _default_output_fn(result: "AgentResult", ctx: WorkContext) -> None:
    ctx.put("agent_response",    result.final_response)
    ctx.put("agent_tools_used",  result.tool_calls_made)
    ctx.put("agent_iterations",  result.iterations)
    if result.usage:
        ctx.put("agent_tokens", result.usage.total_tokens)


# ════════════════════════════════════════════════════════════════════════════
#  NativeAgentBatchProcessor — LLM enrichment without LangChain
# ════════════════════════════════════════════════════════════════════════════

from maestro.batch._processor import RecordProcessingException, RecordProcessor
from maestro.batch._record    import Record


class NativeAgentBatchProcessor(RecordProcessor):
    """
    Enrich batch records using a native LLM adapter (no LangChain).

    More efficient than ``AgentRecordProcessor`` for bulk processing because
    it uses the provider's native API directly and can be configured with
    the provider's native retry and observability decorators.

    Parameters
    ----------
    llm:
        ``AnthropicAdapter`` or ``OpenAIAdapter``.
    prompt_fn:
        ``(payload) → str`` — builds a prompt from the record payload.
    response_fn:
        ``(response_text, payload) → payload`` — merges LLM output into payload.
    system_prompt:
        Optional system message for every call.
    fail_on_error:
        Raise ``RecordProcessingException`` when the LLM call fails.

    Example::

        from maestro.agents import OpenAIAdapter, NativeAgentBatchProcessor

        proc = NativeAgentBatchProcessor(
            llm           = OpenAIAdapter(model="gpt-4o-mini"),
            system_prompt = "Classify text as POSITIVE, NEGATIVE, or NEUTRAL.",
            prompt_fn     = lambda p: f"Review: {p['text']}",
            response_fn   = lambda r, p: {**p, "sentiment": r.strip()},
        )
        job = maestro.JobBuilder().reader(...).processor(proc).writer(...).build()
    """

    def __init__(
        self,
        llm,
        prompt_fn:     Callable[[Any], str],
        response_fn:   Optional[Callable[[str, Any], Any]] = None,
        system_prompt: str  = "",
        fail_on_error: bool = False,
    ) -> None:
        from maestro.agents._providers import Message

        self._llm          = llm
        self._prompt_fn    = prompt_fn
        self._response_fn  = response_fn or (lambda r, p: ({**p, "llm_response": r}
                                                           if isinstance(p, dict) else r))
        self._system       = system_prompt
        self._fail         = fail_on_error
        self._Message      = Message

    def process_record(self, record: Record) -> Record:
        messages = []
        if self._system:
            messages.append(self._Message.system(self._system))
        messages.append(self._Message.user(self._prompt_fn(record.payload)))

        try:
            response = self._llm.chat(messages)
            record.payload = self._response_fn(response.content, record.payload)
        except Exception as exc:
            if self._fail:
                raise RecordProcessingException(
                    f"Native LLM failed on record #{record.header.number}: {exc}"
                ) from exc
            logger.warning("NativeAgentBatchProcessor: record #%d failed — %s",
                           record.header.number, exc)
        return record


__all__ = [
    "AgentResult",
    "NativeReActAgent",
    "NativeAgentWork",
    "NativeAgentBatchProcessor",
]
