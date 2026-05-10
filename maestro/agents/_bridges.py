"""
Bidirectional bridges between Maestro Work units and LangGraph graphs.

Maestro → LangGraph
--------------------
``AgentWork``               Run a compiled LangGraph graph as a Maestro Work.
``AsyncAgentWork``          Async version of AgentWork.

LangGraph → Maestro
--------------------
``AgentRecordProcessor``    Process batch records through an LLM agent.
``AgentCondition``          Use an LLM call as a Maestro rule condition.
``AgentEventHandler``       Use a LangGraph graph as a Maestro FSM event handler.

State bridging
--------------
All bridges accept ``input_fn`` / ``output_fn`` callables that translate
between a Maestro ``WorkContext`` and a LangGraph state dict::

    input_fn  : WorkContext → dict        (context → graph input state)
    output_fn : (dict, WorkContext) → None (graph output state → context mutations)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from maestro.flows._work import DefaultWorkReport, Work, WorkContext, WorkReport, WorkStatus

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

def _default_input(ctx: WorkContext) -> dict:
    """Default input_fn: expose WorkContext as 'context' key in graph state."""
    from langchain_core.messages import HumanMessage
    return {
        "context": ctx.as_map(),
        "facts":   ctx.as_map(),
    }


def _default_output(result: dict, ctx: WorkContext) -> None:
    """Default output_fn: write all non-message result keys back to WorkContext."""
    for key, value in result.items():
        if key != "messages":
            ctx.put(key, value)


# ════════════════════════════════════════════════════════════════════════════
#  AgentWork — LangGraph graph as a Maestro Work
# ════════════════════════════════════════════════════════════════════════════

class AgentWork(Work):
    """
    Run a compiled LangGraph graph as a step inside any Maestro workflow.

    The bridge translates ``WorkContext`` → graph state before execution and
    merges graph output state → ``WorkContext`` after.

    Parameters
    ----------
    graph:
        A compiled LangGraph graph (``graph.compile()``).
    input_fn:
        ``(WorkContext) → dict`` — builds the initial graph state.
        Default: exposes WorkContext as ``"context"`` and ``"facts"`` keys.
    output_fn:
        ``(result_dict, WorkContext) → None`` — reads graph output into context.
        Default: writes every non-``"messages"`` key back to WorkContext.
    config:
        Optional LangGraph run config (thread_id, recursion_limit, etc.).
    name:
        Display name for logging.

    Example::

        from maestro.agents import AgentWork, ReActWithRulesGraph
        from langchain_anthropic import ChatAnthropic

        graph = ReActWithRulesGraph(
            llm   = ChatAnthropic(model="claude-sonnet-4-5"),
            rules = maestro.Rules(classify_rule, route_rule),
        ).build()

        flow = (maestro.aNewSequentialFlow()
                .execute(validate_input_work)
                .then(AgentWork(
                    graph,
                    input_fn  = lambda ctx: {"messages": [HumanMessage(ctx.get("query"))]},
                    output_fn = lambda r, ctx: ctx.put("answer", r.get("result")),
                ))
                .then(persist_work)
                .build())
    """

    def __init__(
        self,
        graph,
        input_fn:  Optional[Callable[[WorkContext], dict]] = None,
        output_fn: Optional[Callable[[dict, WorkContext], None]] = None,
        config:    Optional[dict] = None,
        name:      str = "agent-work",
    ) -> None:
        self._graph     = graph
        self._input_fn  = input_fn  or _default_input
        self._output_fn = output_fn or _default_output
        self._config    = config or {}
        self._name      = name

    def get_name(self) -> str:
        return self._name

    def execute(self, work_context: WorkContext) -> WorkReport:
        inputs = self._input_fn(work_context)
        try:
            result = self._graph.invoke(inputs, config=self._config)
            self._output_fn(result, work_context)
            logger.info("AgentWork '%s': completed — messages=%d",
                        self._name, len(result.get("messages", [])))
            return DefaultWorkReport(WorkStatus.COMPLETED, work_context)
        except Exception as exc:
            logger.error("AgentWork '%s': failed — %s", self._name, exc)
            return DefaultWorkReport(WorkStatus.FAILED, work_context, error=exc)


# ════════════════════════════════════════════════════════════════════════════
#  AsyncAgentWork — async variant
# ════════════════════════════════════════════════════════════════════════════

class AsyncAgentWork:
    """
    Run a compiled LangGraph graph asynchronously inside a Maestro async flow.

    Implements the ``AsyncWork`` protocol from ``maestro.async_``.

    Example::

        from maestro.async_ import AsyncSequentialFlow, AsyncWorkFlowEngine

        flow = (AsyncSequentialFlow.Builder()
                .execute(AsyncAgentWork(graph, input_fn=..., output_fn=...))
                .build())
        report = await AsyncWorkFlowEngine().run(flow, WorkContext())
    """

    def __init__(
        self,
        graph,
        input_fn:  Optional[Callable] = None,
        output_fn: Optional[Callable] = None,
        config:    Optional[dict] = None,
        name:      str = "async-agent-work",
    ) -> None:
        self._graph     = graph
        self._input_fn  = input_fn  or _default_input
        self._output_fn = output_fn or _default_output
        self._config    = config or {}
        self._name      = name

    def get_name(self) -> str:
        return self._name

    async def execute(self, work_context: WorkContext) -> WorkReport:
        inputs = self._input_fn(work_context)
        try:
            result = await self._graph.ainvoke(inputs, config=self._config)
            self._output_fn(result, work_context)
            return DefaultWorkReport(WorkStatus.COMPLETED, work_context)
        except Exception as exc:
            logger.error("AsyncAgentWork '%s': failed — %s", self._name, exc)
            return DefaultWorkReport(WorkStatus.FAILED, work_context, error=exc)


# ════════════════════════════════════════════════════════════════════════════
#  AgentRecordProcessor — LLM to process batch records
# ════════════════════════════════════════════════════════════════════════════

from maestro.batch._processor import RecordProcessingException, RecordProcessor
from maestro.batch._record import Record


class AgentRecordProcessor(RecordProcessor):
    """
    Process each batch record by running it through a LangGraph agent or an LLM.

    The record payload is converted to a prompt via ``prompt_fn``; the LLM
    response is written back to the record via ``response_fn``.

    Parameters
    ----------
    llm:
        Any LangChain ``BaseChatModel``.
    prompt_fn:
        ``(payload) → str`` — builds the prompt from the record payload.
    response_fn:
        ``(response_text, payload) → payload`` — merges the LLM response into
        the payload. Default: adds ``"agent_response"`` key.
    fail_on_error:
        Raise ``RecordProcessingException`` when the LLM call fails.

    Example::

        from langchain_anthropic import ChatAnthropic
        from maestro.agents import AgentRecordProcessor

        proc = AgentRecordProcessor(
            llm         = ChatAnthropic(model="claude-haiku-4-5"),
            prompt_fn   = lambda p: f"Classify this review as POSITIVE/NEGATIVE: {p['text']}",
            response_fn = lambda r, p: {**p, "sentiment": r.strip()},
        )
        job = maestro.JobBuilder().reader(...).processor(proc).writer(...).build()
    """

    def __init__(
        self,
        llm,
        prompt_fn:   Callable[[Any], str],
        response_fn: Optional[Callable[[str, Any], Any]] = None,
        fail_on_error: bool = False,
    ) -> None:
        self._llm          = llm
        self._prompt_fn    = prompt_fn
        self._response_fn  = response_fn or (lambda r, p: ({**p, "agent_response": r}
                                                            if isinstance(p, dict) else r))
        self._fail_on_err  = fail_on_error

    def process_record(self, record: Record) -> Record:
        from langchain_core.messages import HumanMessage
        prompt = self._prompt_fn(record.payload)
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            text = response.content if hasattr(response, "content") else str(response)
            record.payload = self._response_fn(text, record.payload)
        except Exception as exc:
            if self._fail_on_err:
                raise RecordProcessingException(
                    f"Agent failed on record #{record.header.number}: {exc}"
                ) from exc
            logger.warning("AgentRecordProcessor: record #%d failed — %s",
                           record.header.number, exc)
        return record


# ════════════════════════════════════════════════════════════════════════════
#  AgentCondition — LLM as a Maestro rule condition
# ════════════════════════════════════════════════════════════════════════════

class AgentCondition:
    """
    Use an LLM call as the condition of a Maestro rule.

    Allows mixing deterministic rules with LLM-powered soft rules in the
    same ``Rules`` collection.  The LLM is prompted to respond with a
    boolean-like answer ("yes"/"no"/"true"/"false").

    Warning: LLM calls in rule conditions run on every engine iteration —
    use sparingly and consider caching for high-throughput scenarios.

    Example::

        from maestro.agents import AgentCondition

        sentiment_condition = AgentCondition(
            llm         = ChatAnthropic(model="claude-haiku-4-5"),
            prompt_fn   = lambda f: f"Is '{f.get('review')}' positive? Answer yes or no.",
        )

        positive_review_rule = (
            maestro.RuleBuilder()
            .name("positive-sentiment")
            .when(sentiment_condition)
            .then(lambda f: f.put("action", "highlight"))
            .build()
        )
    """

    def __init__(
        self,
        llm,
        prompt_fn: Callable,
        true_tokens: tuple[str, ...] = ("yes", "true", "1", "positive"),
    ) -> None:
        self._llm         = llm
        self._prompt_fn   = prompt_fn
        self._true_tokens = tuple(t.lower() for t in true_tokens)

    def __call__(self, facts) -> bool:
        from langchain_core.messages import HumanMessage
        from maestro.rules import Facts
        f = facts if isinstance(facts, Facts) else facts
        prompt   = self._prompt_fn(f)
        response = self._llm.invoke([HumanMessage(content=prompt)])
        text     = (response.content if hasattr(response, "content")
                    else str(response)).strip().lower()
        return any(text.startswith(t) for t in self._true_tokens)


# ════════════════════════════════════════════════════════════════════════════
#  AgentEventHandler — LangGraph graph as an FSM event handler
# ════════════════════════════════════════════════════════════════════════════

from maestro.states._state import EventHandler


class AgentEventHandler(EventHandler):
    """
    Use a LangGraph graph (or any callable returning a response) as a
    Maestro FSM ``EventHandler``.

    When the FSM fires an event, this handler runs the agent with the
    event metadata as input.  The agent can publish bus messages, update
    databases, or trigger other workflows.

    Example::

        handler = AgentEventHandler(
            graph     = notification_agent,
            input_fn  = lambda event: {
                "messages": [HumanMessage(f"Order state changed: {type(event).__name__}")]
            },
        )

        transition = (
            maestro.TransitionBuilder()
            .source_state(paid).event_type(ShipEvent)
            .event_handler(handler)
            .target_state(shipped)
            .build()
        )
    """

    def __init__(
        self,
        graph,
        input_fn:  Optional[Callable] = None,
        config:    Optional[dict] = None,
    ) -> None:
        self._graph    = graph
        self._input_fn = input_fn or (lambda e: {"messages": []})
        self._config   = config or {}

    def handle(self, event) -> None:
        inputs = self._input_fn(event)
        try:
            self._graph.invoke(inputs, config=self._config)
        except Exception as exc:
            logger.error("AgentEventHandler: graph failed — %s", exc)


__all__ = [
    "AgentWork", "AsyncAgentWork",
    "AgentRecordProcessor", "AgentCondition", "AgentEventHandler",
]
