"""
LangGraph node factory functions.

Each factory returns a plain function suitable for ``graph.add_node(…)``.
The returned node reads from and writes to ``MaestroAgentState``.

Factories
---------
make_rules_node     Evaluate a Maestro rules engine; update facts in state
make_fsm_node       Fire an FSM event; update fsm_state in state
make_batch_node     Run a batch job; store metrics in state
make_saga_node      Execute a saga; store status and updated context in state
make_validator_node Validate facts against a schema; set error or continue
make_observer_node  Emit metrics from the agent state into a MaestroObserver
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Type

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  make_rules_node
# ════════════════════════════════════════════════════════════════════════════

def make_rules_node(
    rules,
    engine=None,
    facts_key: str = "facts",
    name:       str = "rules-node",
) -> Callable:
    """
    Return a LangGraph node that evaluates a Maestro rules engine.

    The node reads ``state[facts_key]`` (default ``"facts"``), fires all
    matching rules, then returns the updated facts dict.  Downstream nodes
    can read rule-derived values (tier, discount, risk_flag …) from state.

    Parameters
    ----------
    rules:      Maestro ``Rules`` collection.
    engine:     Maestro rules engine (default: ``DefaultRulesEngine``).
    facts_key:  State key to read/write facts (default: ``"facts"``).
    name:       Label used in logging.

    Usage::

        graph.add_node("apply-pricing", make_rules_node(pricing_rules))
        graph.add_edge("apply-pricing", "decide")
    """
    from maestro.rules import DefaultRulesEngine, Facts

    _engine = engine or DefaultRulesEngine()

    def _node(state: dict) -> dict:
        raw   = state.get(facts_key) or {}
        facts = Facts(**raw)
        result = _engine.fire(rules, facts)
        fired  = [r.name for r, v in result.items() if v]
        logger.debug("rules-node '%s': %d rule(s) fired: %s", name, len(fired), fired)
        return {facts_key: facts.as_map()}

    _node.__name__ = name
    return _node


# ════════════════════════════════════════════════════════════════════════════
#  make_fsm_node
# ════════════════════════════════════════════════════════════════════════════

def make_fsm_node(
    fsm,
    event_class: Type,
    name: str = "fsm-node",
) -> Callable:
    """
    Return a LangGraph node that fires a single FSM event.

    Updates ``state["fsm_state"]`` and appends to ``state["fsm_history"]``.

    Parameters
    ----------
    fsm:         Maestro ``FiniteStateMachine``.
    event_class: The event class to instantiate and fire.
    name:        Label used in logging.

    Usage::

        graph.add_node("pay", make_fsm_node(order_fsm, PayEvent))
        graph.add_node("ship", make_fsm_node(order_fsm, ShipEvent))
    """
    from maestro.states._state import NoSuchTransitionException

    def _node(state: dict) -> dict:
        try:
            new_state = fsm.fire(event_class())
            history   = list(state.get("fsm_history") or [])
            history.append(new_state.name)
            logger.debug("fsm-node '%s': → %s", name, new_state.name)
            return {"fsm_state": new_state.name, "fsm_history": history}
        except NoSuchTransitionException as exc:
            logger.warning("fsm-node '%s': no transition — %s", name, exc)
            return {"error": str(exc)}

    _node.__name__ = name
    return _node


# ════════════════════════════════════════════════════════════════════════════
#  make_batch_node
# ════════════════════════════════════════════════════════════════════════════

def make_batch_node(
    job,
    metrics_key: str = "batch_metrics",
    name:        str = "batch-node",
) -> Callable:
    """
    Return a LangGraph node that runs a Maestro batch job.

    Stores job metrics in ``state[metrics_key]`` so downstream nodes and
    the LLM can inspect throughput, error counts, etc.

    Parameters
    ----------
    job:         Maestro ``Job`` (built with ``JobBuilder``).
    metrics_key: State key to store metrics dict (default: ``"batch_metrics"``).
    name:        Label used in logging.

    Usage::

        graph.add_node("run-etl", make_batch_node(daily_etl_job))
        graph.add_edge("run-etl", "analyze-results")
    """
    def _node(state: dict) -> dict:
        report = job.call()
        m      = report.metrics
        metrics = {
            "status":        report.status.value,
            "total_count":   m.total_count,
            "written_count": m.written_count,
            "filtered_count":m.filtered_count,
            "skipped_count": m.skipped_count,
            "failed_count":  m.failed_count,
            "duration_s":    round(m.duration_seconds, 3),
        }
        logger.info("batch-node '%s': %d written, %d filtered in %.2fs",
                    name, m.written_count, m.filtered_count, m.duration_seconds)
        return {metrics_key: metrics}

    _node.__name__ = name
    return _node


# ════════════════════════════════════════════════════════════════════════════
#  make_saga_node
# ════════════════════════════════════════════════════════════════════════════

def make_saga_node(
    saga,
    context_key:     str = "context",
    saga_status_key: str = "saga_status",
    name:            str = "saga-node",
) -> Callable:
    """
    Return a LangGraph node that executes a Maestro saga.

    Reads initial context from ``state[context_key]``, runs the saga, then
    writes the updated context and saga status back to state.

    On compensation (partial or full), sets ``state["error"]`` so the LLM
    or a conditional edge can detect failure.

    Parameters
    ----------
    saga:            Maestro ``Saga`` object.
    context_key:     State key for the WorkContext dict.
    saga_status_key: State key for the SagaStatus string.
    name:            Label used in logging.

    Usage::

        graph.add_node("checkout", make_saga_node(checkout_saga))
        graph.add_conditional_edges(
            "checkout",
            lambda s: "succeeded" if s.get("saga_status") == "COMPLETED" else "compensated",
        )
    """
    from maestro.flows._work import WorkContext

    def _node(state: dict) -> dict:
        ctx    = WorkContext(**(state.get(context_key) or {}))
        report = saga.execute(ctx)
        out: dict = {
            saga_status_key: report.saga_status.value,
            context_key:     ctx.as_map(),
        }
        if report.saga_status.value != "COMPLETED":
            out["error"] = (
                f"Saga failed at '{report.failed_step}'; "
                f"compensated: {report.compensated_steps}"
            )
        logger.info("saga-node '%s': %s", name, report.saga_status.value)
        return out

    _node.__name__ = name
    return _node


# ════════════════════════════════════════════════════════════════════════════
#  make_validator_node
# ════════════════════════════════════════════════════════════════════════════

def make_validator_node(
    schema,
    facts_key: str = "facts",
    name:      str = "validator-node",
) -> Callable:
    """
    Return a LangGraph node that validates ``state[facts_key]`` against a schema.

    If invalid, sets ``state["error"]`` with a summary of validation errors
    and sets ``state["next_action"] = "fix"`` so the LLM can decide to
    request corrections from the user.

    Parameters
    ----------
    schema:    Maestro ``Schema``.
    facts_key: State key containing the dict to validate.
    name:      Label used in logging.

    Usage::

        graph.add_node("validate", make_validator_node(order_schema))
        graph.add_conditional_edges(
            "validate",
            lambda s: "fix" if s.get("error") else "process",
        )
    """
    def _node(state: dict) -> dict:
        data   = state.get(facts_key) or {}
        result = schema.validate(data)
        if result.ok:
            return {"error": None, "next_action": None}
        error_summary = "; ".join(f"[{e.field}] {e.message}" for e in result.errors)
        logger.warning("validator-node '%s': %d error(s): %s", name, len(result.errors), error_summary)
        return {"error": error_summary, "next_action": "fix"}

    _node.__name__ = name
    return _node


# ════════════════════════════════════════════════════════════════════════════
#  make_observer_node
# ════════════════════════════════════════════════════════════════════════════

def make_observer_node(
    observer,
    module:    str = "agents",
    name:      str = "observer-node",
) -> Callable:
    """
    Return a LangGraph node that emits agent metrics into a Maestro observer.

    Emits counters for iteration count, message count, and an event for
    each distinct next_action value seen — allowing the Maestro observability
    dashboard to include agent loop metrics alongside batch and rules metrics.

    Parameters
    ----------
    observer:  Maestro ``InMemoryObserver`` or any ``Observer`` subclass.
    module:    Module label in metric events (default: ``"agents"``).
    name:      Label used in logging.

    Usage::

        obs = InMemoryObserver()
        graph.add_node("observe", make_observer_node(obs))
        graph.add_edge("agent", "observe")
        graph.add_edge("observe", "tools")
    """
    from maestro.observe import MetricEvent
    import time

    def _node(state: dict) -> dict:
        iteration = (state.get("iteration") or 0) + 1
        msg_count = len(state.get("messages") or [])

        observer.on_event(MetricEvent("agents", "iteration",    iteration, {}, kind="gauge"))
        observer.on_event(MetricEvent("agents", "message_count", msg_count, {}, kind="gauge"))

        action = state.get("next_action")
        if action:
            observer.on_event(MetricEvent("agents", "action", 1.0, {"action": action}, kind="counter"))

        return {"iteration": iteration}

    _node.__name__ = name
    return _node


__all__ = [
    "make_rules_node", "make_fsm_node", "make_batch_node",
    "make_saga_node", "make_validator_node", "make_observer_node",
]
