"""
maestro.agents — agentic capabilities powered by LangGraph.

Bidirectional integration between Maestro SDK and LangGraph so that:

* Maestro components (rules, FSM, batch, saga) become LangChain **tools**
  that an LLM agent can call.
* A compiled LangGraph graph becomes a Maestro **Work** step embedded in
  any Sequential / Parallel / Repeat flow.
* LangGraph **nodes** are pre-built from Maestro components (rules node,
  FSM node, batch node, saga node).
* Three **pre-built graph patterns** cover the most common agentic workflows:
  ReAct + rules, human-in-the-loop, and multi-agent orchestration.

Install
-------
::

    pip install "maestro-sdk[agents]"
    # or
    pip install langgraph langchain-core langchain-anthropic

Quick start
-----------
::

    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage
    import maestro
    from maestro.agents import (
        ReActWithRulesGraph, RulesEngineTool, SagaTool,
        AgentWork,
    )

    # 1. Expose Maestro rules as an LLM tool
    pricing_tool = RulesEngineTool(
        rules       = maestro.Rules(vip_rule, promo_rule),
        name        = "apply_pricing",
        description = "Apply VIP and promotional pricing rules.",
    )

    # 2. Build a ReAct agent that uses Maestro for structured decisions
    graph = ReActWithRulesGraph(
        llm           = ChatAnthropic(model="claude-sonnet-4-5"),
        maestro_tools = [pricing_tool],
        system_prompt = "You are an order processing agent.",
    ).build()

    # 3a. Run standalone
    result = graph.invoke({
        "messages": [HumanMessage("Process order for Alice, total $200, tier vip.")],
        "facts":    {"customer": "Alice", "total": 200.0, "tier": "vip"},
    })

    # 3b. Or embed inside a Maestro flow
    flow = (maestro.aNewSequentialFlow()
            .execute(validate_work)
            .then(AgentWork(
                graph,
                input_fn  = lambda ctx: {
                    "messages": [HumanMessage(ctx.get("request"))],
                    "facts":    ctx.as_map(),
                },
                output_fn = lambda r, ctx: ctx.put("discount", r.get("facts", {}).get("discount")),
            ))
            .then(charge_work)
            .build())
"""
from __future__ import annotations

# ── State ────────────────────────────────────────────────────────────────── #
from maestro.agents._state import MaestroAgentState

# ── Tools ────────────────────────────────────────────────────────────────── #
from maestro.agents._tools import (
    RulesEngineTool,
    FSMTransitionTool,
    FSMStatusTool,
    BatchJobTool,
    SagaTool,
    EventPublisherTool,
    ValidatorTool,
)

# ── Bridges ──────────────────────────────────────────────────────────────── #
from maestro.agents._bridges import (
    AgentWork,
    AsyncAgentWork,
    AgentRecordProcessor,
    AgentCondition,
    AgentEventHandler,
)

# ── Node factories ───────────────────────────────────────────────────────── #
from maestro.agents._nodes import (
    make_rules_node,
    make_fsm_node,
    make_batch_node,
    make_saga_node,
    make_validator_node,
    make_observer_node,
)

# ── Pre-built graph patterns ─────────────────────────────────────────────── #
from maestro.agents._graphs import (
    ReActWithRulesGraph,
    HumanInTheLoopGraph,
    MultiAgentOrchestratorGraph,
)

__all__ = [
    # state
    "MaestroAgentState",
    # tools
    "RulesEngineTool", "FSMTransitionTool", "FSMStatusTool",
    "BatchJobTool", "SagaTool", "EventPublisherTool", "ValidatorTool",
    # bridges
    "AgentWork", "AsyncAgentWork",
    "AgentRecordProcessor", "AgentCondition", "AgentEventHandler",
    # node factories
    "make_rules_node", "make_fsm_node", "make_batch_node",
    "make_saga_node", "make_validator_node", "make_observer_node",
    # pre-built graphs
    "ReActWithRulesGraph", "HumanInTheLoopGraph", "MultiAgentOrchestratorGraph",
]

# ── Provider adapters ────────────────────────────────────────────────────── #
from maestro.agents._providers import (
    AnthropicModels, OpenAIModels,
    ToolCall, TokenUsage, LLMResponse, Message,
    LLMAdapter, AnthropicAdapter, OpenAIAdapter,
    make_anthropic, make_openai,
)

# ── Native agent patterns ────────────────────────────────────────────────── #
from maestro.agents._native import (
    AgentResult,
    NativeReActAgent,
    NativeAgentWork,
    NativeAgentBatchProcessor,
)
