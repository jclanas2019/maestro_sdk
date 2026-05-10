"""Shared state TypedDict for LangGraph ↔ Maestro interoperability."""
from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class MaestroAgentState(TypedDict, total=False):
    """
    Shared state between a LangGraph graph and the Maestro SDK.

    Every field is optional so graphs can use only what they need.

    Fields
    ------
    messages:
        Conversation history (LangChain messages). Merged via ``add_messages``
        so concurrent nodes can safely append without overwriting each other.
    facts:
        Maestro ``Facts`` serialised as a plain dict. Any rules node can read
        and update this; changes are visible to the LLM on the next turn.
    context:
        Maestro ``WorkContext`` serialised as a plain dict. Allows flow steps
        and agent nodes to share structured data.
    fsm_state:
        Current Maestro FSM state name. Updated by ``make_fsm_node`` and
        ``FSMTransitionTool``.
    fsm_history:
        List of FSM state names visited so far.
    saga_status:
        Latest ``SagaStatus`` value (string) from the most recent saga execution.
    batch_metrics:
        Metrics dict from the last batch job (written_count, filtered_count, etc.).
    next_action:
        Routing hint set by nodes; inspected by conditional edges.
    iteration:
        Loop counter incremented by the agent node. Use to implement safety
        limits (e.g. bail out after N iterations).
    human_input:
        Populated when a human-in-the-loop node resumes execution.
    error:
        Last error message; set when a node catches an exception.
    result:
        Final structured output from the agent run.
    """
    messages:     Annotated[list[BaseMessage], add_messages]
    facts:         dict[str, Any]
    context:       dict[str, Any]
    fsm_state:     Optional[str]
    fsm_history:   list[str]
    saga_status:   Optional[str]
    batch_metrics: dict[str, Any]
    next_action:   Optional[str]
    iteration:     int
    human_input:   Optional[str]
    error:         Optional[str]
    result:        Optional[Any]
