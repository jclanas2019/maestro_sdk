"""
Maestro components exposed as LangChain tools for LLM agents.

Each tool follows the LangChain ``BaseTool`` protocol so an agent can call
it via function-calling, tool-use, or ReAct prompting.

Available tools
---------------
RulesEngineTool     Execute a Maestro rules engine against a facts dict
FSMTransitionTool   Fire a named event on a Maestro FSM
FSMStatusTool       Query the current state of a Maestro FSM
BatchJobTool        Run a Maestro batch job and return its metrics
SagaTool            Execute a Maestro saga, return status + compensations
EventPublisherTool  Publish a message to a Maestro EventBus topic
ValidatorTool       Validate a dict against a Maestro Schema

Usage::

    from maestro.agents import RulesEngineTool, FSMTransitionTool

    tools = [
        RulesEngineTool(rules=pricing_rules, name="apply_pricing"),
        FSMTransitionTool(fsm=order_fsm, event_map={"pay": PayEvent, "cancel": CancelEvent}),
    ]
    agent = create_react_agent(llm, tools)
"""
from __future__ import annotations

import json
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field


# ════════════════════════════════════════════════════════════════════════════
#  Input schemas (Pydantic v2)
# ════════════════════════════════════════════════════════════════════════════

class _FactsInput(BaseModel):
    facts: dict[str, Any] = Field(description="Facts to evaluate. Keys are field names, values can be any JSON-serializable type.")

class _FSMInput(BaseModel):
    event_name: str = Field(description="Name of the event to fire (must be one of the registered event names).")

class _BatchInput(BaseModel):
    parameters: dict[str, Any] = Field(default_factory=dict, description="Optional job parameters to override (batch_size, etc.).")

class _SagaInput(BaseModel):
    context: dict[str, Any] = Field(default_factory=dict, description="Initial context values for the saga execution.")

class _PublishInput(BaseModel):
    topic:   str        = Field(description="Topic name to publish to.")
    payload: Any        = Field(description="Message payload (any JSON-serializable value).")

class _ValidateInput(BaseModel):
    data: dict[str, Any] = Field(description="Data dict to validate against the schema.")


# ════════════════════════════════════════════════════════════════════════════
#  RulesEngineTool
# ════════════════════════════════════════════════════════════════════════════

class RulesEngineTool(BaseTool):
    """
    Execute a Maestro rules engine against a facts dict.

    The LLM can call this tool to apply deterministic business rules —
    pricing, eligibility, classification — without reasoning about them
    itself.  Returns the updated facts dict after all rules have fired.

    Example::

        tool = RulesEngineTool(
            rules       = maestro.Rules(vip_rule, promo_rule),
            name        = "apply_pricing",
            description = "Apply pricing rules to a customer order.",
        )
        agent = create_react_agent(llm, [tool])
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str        = "rules_engine"
    description: str = (
        "Execute business rules against a facts dictionary. "
        "Returns the updated facts after all matching rules have fired. "
        "Use this for deterministic decisions: pricing, eligibility, classification."
    )
    args_schema: Type[BaseModel] = _FactsInput

    rules:  Any
    engine: Any = None

    def model_post_init(self, __context: Any) -> None:
        if self.engine is None:
            from maestro.rules import DefaultRulesEngine
            object.__setattr__(self, "engine", DefaultRulesEngine())

    def _run(self, facts: dict[str, Any]) -> str:
        from maestro.rules import Facts
        f = Facts(**facts)
        result = self.engine.fire(self.rules, f)
        fired  = [r.name for r, v in result.items() if v]
        return json.dumps({
            "facts":       f.as_map(),
            "rules_fired": fired,
            "count":       len(fired),
        })


# ════════════════════════════════════════════════════════════════════════════
#  FSMTransitionTool
# ════════════════════════════════════════════════════════════════════════════

class FSMTransitionTool(BaseTool):
    """
    Fire a named event on a Maestro FSM and return the new state.

    The LLM uses event names (strings) and the tool handles the
    class instantiation internally via ``event_map``.

    Example::

        tool = FSMTransitionTool(
            fsm       = order_fsm,
            event_map = {
                "pay":    PayEvent,
                "ship":   ShipEvent,
                "cancel": CancelEvent,
            },
            name = "transition_order",
        )
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name:        str             = "fsm_transition"
    description: str             = (
        "Fire a state machine event. Provide the event_name (one of the registered events). "
        "Returns the new state name. Raises an error if the transition is not defined."
    )
    args_schema: Type[BaseModel] = _FSMInput

    fsm:       Any
    event_map: dict[str, Any]   = Field(default_factory=dict)

    def _run(self, event_name: str) -> str:
        if event_name not in self.event_map:
            available = list(self.event_map.keys())
            return json.dumps({"error": f"Unknown event '{event_name}'. Available: {available}"})
        try:
            event_cls = self.event_map[event_name]
            new_state = self.fsm.fire(event_cls())
            return json.dumps({
                "event":     event_name,
                "new_state": new_state.name,
                "success":   True,
            })
        except Exception as exc:
            return json.dumps({"error": str(exc), "success": False})


class FSMStatusTool(BaseTool):
    """Query the current state and available transitions of a Maestro FSM."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name:        str             = "fsm_status"
    description: str             = "Query the current FSM state and which events can be fired from it."
    args_schema: Type[BaseModel] = _FactsInput   # accepts empty dict {}

    fsm:       Any
    event_map: dict[str, Any] = Field(default_factory=dict)

    def _run(self, facts: dict[str, Any]) -> str:
        current = self.fsm.current_state.name
        available = [name for name, ecls in self.event_map.items()
                     if any(t.source_state == self.fsm.current_state
                            and isinstance(ecls(), type(ecls()))
                            for t in getattr(self.fsm, "_transitions", {}).values())]
        return json.dumps({
            "current_state":      current,
            "available_events":   list(self.event_map.keys()),
        })


# ════════════════════════════════════════════════════════════════════════════
#  BatchJobTool
# ════════════════════════════════════════════════════════════════════════════

class BatchJobTool(BaseTool):
    """
    Run a Maestro batch job and return execution metrics.

    The LLM can trigger ETL pipelines, data processing or report
    generation as part of an agent workflow.

    Example::

        tool = BatchJobTool(
            job         = orders_etl_job,
            name        = "run_etl",
            description = "Process the daily orders CSV and enrich records.",
        )
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name:        str             = "batch_job"
    description: str             = (
        "Run a batch processing pipeline. Returns metrics: "
        "records read, written, filtered, skipped, failed and duration."
    )
    args_schema: Type[BaseModel] = _BatchInput

    job: Any

    def _run(self, parameters: dict[str, Any]) -> str:
        report = self.job.call()
        m      = report.metrics
        return json.dumps({
            "status":        report.status.value,
            "total_count":   m.total_count,
            "written_count": m.written_count,
            "filtered_count":m.filtered_count,
            "skipped_count": m.skipped_count,
            "failed_count":  m.failed_count,
            "duration_s":    round(m.duration_seconds, 3),
        })


# ════════════════════════════════════════════════════════════════════════════
#  SagaTool
# ════════════════════════════════════════════════════════════════════════════

class SagaTool(BaseTool):
    """
    Execute a Maestro saga as an agent tool, with automatic compensation.

    Returns saga status, completed steps and any compensated steps so
    the LLM can decide how to proceed on partial failure.

    Example::

        tool = SagaTool(
            saga        = checkout_saga,
            name        = "checkout",
            description = "Book flight + hotel + charge card with auto-rollback.",
        )
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name:        str             = "saga"
    description: str             = (
        "Execute a multi-step saga with automatic compensation on failure. "
        "Returns the saga status (COMPLETED, COMPENSATED, PARTIALLY_COMPENSATED, FAILED) "
        "and lists of completed and compensated steps."
    )
    args_schema: Type[BaseModel] = _SagaInput

    saga: Any

    def _run(self, context: dict[str, Any]) -> str:
        from maestro.flows._work import WorkContext
        ctx    = WorkContext(**context)
        report = self.saga.execute(ctx)
        return json.dumps({
            "saga_status":        report.saga_status.value,
            "succeeded_steps":    report.succeeded_steps,
            "compensated_steps":  report.compensated_steps,
            "failed_step":        report.failed_step,
            "failed_compensations": report.failed_compensations,
            "context":            ctx.as_map(),
        })


# ════════════════════════════════════════════════════════════════════════════
#  EventPublisherTool
# ════════════════════════════════════════════════════════════════════════════

class EventPublisherTool(BaseTool):
    """
    Publish a message to a Maestro EventBus topic from inside an agent.

    Allows an agent to notify other services or trigger downstream
    workflows reactively.

    Example::

        tool = EventPublisherTool(
            bus  = shared_bus,
            name = "notify_ops",
        )
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name:        str             = "event_publisher"
    description: str             = (
        "Publish a message to the event bus. Provide 'topic' and 'payload'. "
        "Returns the message id."
    )
    args_schema: Type[BaseModel] = _PublishInput

    bus: Any

    def _run(self, topic: str, payload: Any) -> str:
        msg = self.bus.publish(topic, payload, source="agent")
        return json.dumps({"message_id": msg.id, "topic": topic})


# ════════════════════════════════════════════════════════════════════════════
#  ValidatorTool
# ════════════════════════════════════════════════════════════════════════════

class ValidatorTool(BaseTool):
    """
    Validate a data dict against a Maestro Schema.

    Lets the LLM verify whether data is valid before acting on it,
    and see exactly which fields are invalid and why.

    Example::

        tool = ValidatorTool(
            schema      = order_schema,
            name        = "validate_order",
            description = "Check that an order dict is valid before processing.",
        )
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name:        str             = "validator"
    description: str             = (
        "Validate a data dictionary against the defined schema. "
        "Returns ok=True if valid, otherwise lists field-level errors."
    )
    args_schema: Type[BaseModel] = _ValidateInput

    maestro_schema: Any

    def _run(self, data: dict[str, Any]) -> str:
        result = self.maestro_schema.validate(data)
        return json.dumps({
            "ok":     result.ok,
            "errors": [{"field": e.field, "message": e.message} for e in result.errors],
        })


__all__ = [
    "RulesEngineTool", "FSMTransitionTool", "FSMStatusTool",
    "BatchJobTool", "SagaTool", "EventPublisherTool", "ValidatorTool",
]
