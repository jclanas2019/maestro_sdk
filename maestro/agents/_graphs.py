"""
Pre-built LangGraph graph patterns for common agentic workflows.

Patterns
--------
ReActWithRulesGraph
    Standard ReAct (Reason + Act) loop where the LLM reasons and calls tools.
    Maestro rules are exposed as a tool for deterministic sub-decisions.
    The LLM handles flexible reasoning; rules handle structured logic.

HumanInTheLoopGraph
    Agent that pauses at a checkpoint and waits for human approval before
    continuing.  A Maestro FSM tracks the workflow state (PENDING →
    UNDER_REVIEW → APPROVED / REJECTED).  The ``MemorySaver`` checkpointer
    persists state across the pause.

MultiAgentOrchestratorGraph
    Supervisor + specialist pattern.  A supervisor LLM reads the user
    request and routes it to one of several specialist sub-agents.  Each
    specialist can use its own set of Maestro tools.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Optional, Sequence

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  ReActWithRulesGraph
# ════════════════════════════════════════════════════════════════════════════

class ReActWithRulesGraph:
    """
    ReAct agent augmented with Maestro tools for structured sub-decisions.

    The LLM reasons and decides which tools to call.  Maestro tools
    (``RulesEngineTool``, ``FSMTransitionTool``, etc.) give the agent
    access to deterministic, auditable business logic that should NOT
    be left to LLM judgement alone.

    Parameters
    ----------
    llm:
        Any LangChain ``BaseChatModel``.
    maestro_tools:
        List of Maestro tools (``RulesEngineTool``, ``SagaTool``, etc.).
    extra_tools:
        Any additional LangChain tools.
    system_prompt:
        System prompt injected as the first message.
    max_iterations:
        Hard limit on agent loop iterations (safety valve).
    checkpointer:
        Optional LangGraph checkpointer for persistence.  Pass a
        ``MemorySaver`` for in-memory state across interrupts.

    Example::

        from langchain_anthropic import ChatAnthropic
        from maestro.agents import ReActWithRulesGraph, RulesEngineTool, SagaTool

        graph = ReActWithRulesGraph(
            llm = ChatAnthropic(model="claude-sonnet-4-5"),
            maestro_tools = [
                RulesEngineTool(rules=pricing_rules, name="apply_pricing",
                                description="Apply pricing and discount rules."),
                SagaTool(saga=checkout_saga, name="checkout",
                         description="Run the checkout saga (reserve + charge + ship)."),
            ],
            system_prompt = "You are an order processing agent.",
        ).build()

        result = graph.invoke({
            "messages": [HumanMessage("Process order ORD-42 for customer Alice.")],
            "facts":    {"customer": "Alice", "total": 150.0, "tier": "vip"},
        })
    """

    def __init__(
        self,
        llm,
        maestro_tools:  Sequence = (),
        extra_tools:    Sequence = (),
        system_prompt:  str = "",
        max_iterations: int = 20,
        checkpointer    = None,
    ) -> None:
        self._llm            = llm
        self._maestro_tools  = list(maestro_tools)
        self._extra_tools    = list(extra_tools)
        self._system_prompt  = system_prompt
        self._max_iter       = max_iterations
        self._checkpointer   = checkpointer

    def build(self):
        from langchain_core.messages import SystemMessage
        from langgraph.graph import END, START, StateGraph
        from langgraph.prebuilt import ToolNode

        from maestro.agents._state import MaestroAgentState

        all_tools = self._maestro_tools + self._extra_tools
        llm_bound = self._llm.bind_tools(all_tools) if all_tools else self._llm
        sys_msg   = self._system_prompt

        def agent_node(state: dict) -> dict:
            messages = list(state.get("messages") or [])
            if sys_msg and (not messages or not hasattr(messages[0], "type")
                            or messages[0].type != "system"):
                messages = [SystemMessage(content=sys_msg)] + messages
            iteration = (state.get("iteration") or 0) + 1
            response  = llm_bound.invoke(messages)
            logger.debug("ReAct agent: iteration %d, tool_calls=%d",
                         iteration, len(getattr(response, "tool_calls", []) or []))
            return {"messages": [response], "iteration": iteration}

        def should_continue(state: dict) -> str:
            last      = (state.get("messages") or [None])[-1]
            iteration = state.get("iteration") or 0
            if iteration >= self._max_iter:
                logger.warning("ReActWithRulesGraph: max iterations (%d) reached", self._max_iter)
                return END
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        graph = StateGraph(MaestroAgentState)
        graph.add_node("agent", agent_node)
        if all_tools:
            graph.add_node("tools", ToolNode(all_tools))
            graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
            graph.add_edge("tools", "agent")
        else:
            graph.add_edge("agent", END)
        graph.set_entry_point("agent")

        compile_kwargs: dict = {}
        if self._checkpointer:
            compile_kwargs["checkpointer"] = self._checkpointer
        return graph.compile(**compile_kwargs)


# ════════════════════════════════════════════════════════════════════════════
#  HumanInTheLoopGraph
# ════════════════════════════════════════════════════════════════════════════

class HumanInTheLoopGraph:
    """
    Agent workflow that pauses for human approval, backed by a Maestro FSM.

    Flow
    ----
    1. Agent analyses the request (LLM).
    2. Agent calls Maestro tools to gather data / apply rules.
    3. If the agent sets ``next_action = "needs_review"``, execution pauses
       (``interrupt``).  The Maestro FSM transitions to ``UNDER_REVIEW``.
    4. A human resumes execution by supplying ``human_input`` in the state.
    5. The FSM transitions to ``APPROVED`` or ``REJECTED`` based on the
       human decision; the agent continues or stops accordingly.

    Parameters
    ----------
    llm:
        Any LangChain ``BaseChatModel``.
    fsm:
        Maestro FSM tracking approval lifecycle (must have UNDER_REVIEW,
        APPROVED, REJECTED states and the corresponding event classes).
    review_event_class:
        Event to fire when entering review (e.g. ``FlagEvent``).
    approve_event_class:
        Event to fire on approval.
    reject_event_class:
        Event to fire on rejection.
    maestro_tools:
        Maestro tools available to the agent.
    system_prompt:
        System prompt for the agent.
    checkpointer:
        Checkpointer for persisting state across the human interrupt.
        Default: in-memory ``MemorySaver``.

    Example::

        from langchain_anthropic import ChatAnthropic
        from langgraph.checkpoint.memory import MemorySaver

        graph = HumanInTheLoopGraph(
            llm                  = ChatAnthropic(model="claude-sonnet-4-5"),
            fsm                  = approval_fsm,
            review_event_class   = FlagForReviewEvent,
            approve_event_class  = ApproveEvent,
            reject_event_class   = RejectEvent,
            maestro_tools        = [validator_tool, rules_tool],
            checkpointer         = MemorySaver(),
        ).build()

        config  = {"configurable": {"thread_id": "review-42"}}
        # First run — pauses at review checkpoint
        result  = graph.invoke({"messages": [HumanMessage("Approve this refund.")]}, config)

        # Human decision → resume
        result2 = graph.invoke({"human_input": "approved"}, config)
    """

    def __init__(
        self,
        llm,
        fsm,
        review_event_class,
        approve_event_class,
        reject_event_class,
        maestro_tools:  Sequence = (),
        system_prompt:  str = "",
        checkpointer    = None,
    ) -> None:
        self._llm            = llm
        self._fsm            = fsm
        self._review_ev      = review_event_class
        self._approve_ev     = approve_event_class
        self._reject_ev      = reject_event_class
        self._maestro_tools  = list(maestro_tools)
        self._system_prompt  = system_prompt
        self._checkpointer   = checkpointer

    def build(self):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from langgraph.graph import END, START, StateGraph
        from langgraph.prebuilt import ToolNode
        from langgraph.types import interrupt

        from maestro.agents._state import MaestroAgentState
        from maestro.agents._nodes import make_fsm_node

        all_tools = self._maestro_tools
        llm_bound = self._llm.bind_tools(all_tools) if all_tools else self._llm
        sys_msg   = self._system_prompt
        fsm_ref   = self._fsm

        def agent_node(state: dict) -> dict:
            messages = list(state.get("messages") or [])
            if sys_msg and (not messages or getattr(messages[0], "type", None) != "system"):
                messages = [SystemMessage(content=sys_msg)] + messages
            response = llm_bound.invoke(messages)
            return {"messages": [response], "iteration": (state.get("iteration") or 0) + 1}

        def should_continue(state: dict) -> str:
            last = (state.get("messages") or [None])[-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            content = getattr(last, "content", "") or ""
            if "needs_review" in content.lower() or state.get("next_action") == "needs_review":
                return "review"
            return END

        def review_node(state: dict) -> dict:
            """Pause here for human input."""
            fsm_ref.fire(self._review_ev())
            history = list(state.get("fsm_history") or [])
            history.append(fsm_ref.current_state.name)

            human_decision = interrupt({
                "question": "Please review and respond with 'approved' or 'rejected'.",
                "context":  state.get("facts") or {},
            })
            return {
                "human_input": human_decision,
                "fsm_state":   fsm_ref.current_state.name,
                "fsm_history": history,
            }

        def resolve_decision(state: dict) -> str:
            decision = (state.get("human_input") or "").strip().lower()
            return "approve" if decision.startswith("appro") else "reject"

        def approve_node(state: dict) -> dict:
            fsm_ref.fire(self._approve_ev())
            history = list(state.get("fsm_history") or [])
            history.append(fsm_ref.current_state.name)
            return {
                "fsm_state":   fsm_ref.current_state.name,
                "fsm_history": history,
                "next_action": "approved",
                "messages":    [AIMessage(content="Request approved by human reviewer.")],
            }

        def reject_node(state: dict) -> dict:
            fsm_ref.fire(self._reject_ev())
            history = list(state.get("fsm_history") or [])
            history.append(fsm_ref.current_state.name)
            return {
                "fsm_state":   fsm_ref.current_state.name,
                "fsm_history": history,
                "next_action": "rejected",
                "messages":    [AIMessage(content="Request rejected by human reviewer.")],
            }

        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = self._checkpointer or MemorySaver()

        graph = StateGraph(MaestroAgentState)
        graph.add_node("agent",  agent_node)
        graph.add_node("review", review_node)
        graph.add_node("approve", approve_node)
        graph.add_node("reject",  reject_node)
        if all_tools:
            graph.add_node("tools", ToolNode(all_tools))
            graph.add_edge("tools", "agent")
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent",  should_continue,
                                    {"tools": "tools", "review": "review", END: END})
        graph.add_conditional_edges("review", resolve_decision,
                                    {"approve": "approve", "reject": "reject"})
        graph.add_edge("approve", END)
        graph.add_edge("reject",  END)
        return graph.compile(checkpointer=checkpointer, interrupt_before=["review"])


# ════════════════════════════════════════════════════════════════════════════
#  MultiAgentOrchestratorGraph
# ════════════════════════════════════════════════════════════════════════════

class MultiAgentOrchestratorGraph:
    """
    Supervisor + specialist pattern for multi-agent coordination.

    A supervisor LLM reads the user request and routes it to the most
    appropriate specialist.  Each specialist is an independent agent with
    its own Maestro tools and system prompt.

    Architecture
    ------------
    ``user → supervisor → specialist_A / specialist_B / … → supervisor → …``

    The supervisor decides routing by replying with a JSON object like::

        {"next": "billing-agent"}

    When the specialist finishes, control returns to the supervisor which
    can route again or signal completion with ``{"next": "FINISH"}``.

    Parameters
    ----------
    supervisor_llm:
        LLM for the routing supervisor (should support JSON mode or structured output).
    specialists:
        Dict mapping specialist name → ``(llm, tools, system_prompt)`` tuples.
    supervisor_prompt:
        System prompt for the supervisor.  Should name the available specialists
        and explain when to use each.
    max_rounds:
        Maximum number of supervisor↔specialist round trips.

    Example::

        graph = MultiAgentOrchestratorGraph(
            supervisor_llm = ChatAnthropic(model="claude-sonnet-4-5"),
            specialists = {
                "billing": (
                    ChatAnthropic(model="claude-haiku-4-5"),
                    [RulesEngineTool(pricing_rules, name="pricing")],
                    "You are the billing specialist. Apply pricing rules.",
                ),
                "fulfillment": (
                    ChatAnthropic(model="claude-haiku-4-5"),
                    [SagaTool(checkout_saga, name="checkout")],
                    "You are the fulfillment specialist. Execute checkout sagas.",
                ),
            },
            supervisor_prompt = (
                "Route requests to the appropriate specialist:\\n"
                "- 'billing': pricing, discounts, invoices\\n"
                "- 'fulfillment': orders, shipping, inventory\\n"
                "Reply with JSON: {\"next\": \"<specialist_name>\"} or {\"next\": \"FINISH\"}."
            ),
        ).build()
    """

    def __init__(
        self,
        supervisor_llm,
        specialists:       dict[str, tuple],
        supervisor_prompt: str = "",
        max_rounds:        int = 10,
    ) -> None:
        self._sup_llm    = supervisor_llm
        self._specialists = specialists
        self._sup_prompt  = supervisor_prompt
        self._max_rounds  = max_rounds

    def build(self):
        import json as _json
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from langgraph.graph import END, StateGraph
        from langgraph.prebuilt import ToolNode

        from maestro.agents._state import MaestroAgentState

        specialist_names = list(self._specialists.keys())
        sup_llm  = self._sup_llm
        sup_msg  = self._sup_prompt
        max_r    = self._max_rounds

        def supervisor_node(state: dict) -> dict:
            iteration = (state.get("iteration") or 0) + 1
            messages  = list(state.get("messages") or [])
            prefix    = ([SystemMessage(content=sup_msg)] if sup_msg else [])
            response  = sup_llm.invoke(prefix + messages)
            content   = getattr(response, "content", "") or ""
            try:
                parsed = _json.loads(content)
                next_  = parsed.get("next", "FINISH")
            except Exception as _exc:
                logger.debug("supervisor routing parse failed: %s — defaulting to FINISH", _exc)
                next_  = "FINISH"
            logger.info("supervisor: iteration=%d routing → %s", iteration, next_)
            return {
                "messages":   [response],
                "next_action": next_,
                "iteration":   iteration,
            }

        def route_supervisor(state: dict) -> str:
            if (state.get("iteration") or 0) >= max_r:
                return END
            nxt = state.get("next_action", "FINISH")
            if nxt in specialist_names:
                return nxt
            return END

        graph = StateGraph(MaestroAgentState)
        graph.add_node("supervisor", supervisor_node)
        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            route_supervisor,
            {name: name for name in specialist_names} | {END: END},
        )

        for spec_name, spec_config in self._specialists.items():
            if len(spec_config) == 3:
                spec_llm, spec_tools, spec_prompt = spec_config
            else:
                spec_llm, spec_tools = spec_config[:2]
                spec_prompt = ""

            _llm   = spec_llm
            _tools = list(spec_tools)
            _msg   = spec_prompt
            _lbound = _llm.bind_tools(_tools) if _tools else _llm

            def make_specialist(llm_b, tools, sys_p):
                def _specialist(state: dict) -> dict:
                    msgs = list(state.get("messages") or [])
                    pfx  = [SystemMessage(content=sys_p)] if sys_p else []
                    resp = llm_b.invoke(pfx + msgs)
                    inner_msgs = [resp]
                    if hasattr(resp, "tool_calls") and resp.tool_calls and tools:
                        tn     = ToolNode(tools)
                        t_resp = tn.invoke({"messages": [resp]})
                        inner_msgs += t_resp.get("messages", [])
                    return {"messages": inner_msgs}
                return _specialist

            graph.add_node(spec_name, make_specialist(_lbound, _tools, _msg))
            graph.add_edge(spec_name, "supervisor")

        return graph.compile()


__all__ = [
    "ReActWithRulesGraph",
    "HumanInTheLoopGraph",
    "MultiAgentOrchestratorGraph",
]
