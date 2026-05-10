"""
maestro.integration — cross-module bridges.

These classes combine the four Maestro modules into higher-order compositions
that are impossible with any single module alone.

+---------------------+--------------------------------------+
| Class               | Combines                             |
+---------------------+--------------------------------------+
| RuleSetWork         | flows + rules  (rules as a Work step)|
| BatchWork           | flows + batch  (batch job as a step) |
| FSMGuardWork        | flows + states (FSM gate in workflow) |
| FSMTransitionWork   | flows + states (fire events in flow) |
| RuleBasedFilter     | batch + rules  (rules filter records)|
| RuleBasedProcessor  | batch + rules  (rules enrich records)|
+---------------------+--------------------------------------+
"""
from __future__ import annotations

import logging
from typing import Optional, Set

from maestro.batch._filter    import RecordFilter
from maestro.batch._processor import RecordProcessingException, RecordProcessor
from maestro.batch._record    import Record
from maestro.flows._work      import (DefaultWorkReport, Work, WorkContext,
                                       WorkReport, WorkStatus)
from maestro.rules            import DefaultRulesEngine, Facts, Rules
from maestro.states._state    import Event, NoSuchTransitionException, State
from maestro.states._fsm      import FiniteStateMachine

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  flows + rules
# ════════════════════════════════════════════════════════════════════════════

class RuleSetWork(Work):
    """
    Run a rules engine as a :class:`~maestro.flows.Work` step inside a workflow.

    The ``Facts`` are populated from the ``WorkContext`` before evaluation and
    written back afterwards, so downstream steps can read rule-derived values.

    Succeeds (COMPLETED) if at least one rule fired; fails (FAILED) otherwise —
    unless ``fail_when_no_rule_fired=False``.

    Example::

        from maestro.flows import SequentialFlow, WorkContext
        from maestro.rules import Rules, RuleBuilder, Facts
        from maestro.integration import RuleSetWork

        discount_rules = Rules(RuleBuilder().name("vip").when(...).then(...).build())
        step = RuleSetWork(rules=discount_rules, fail_when_no_rule_fired=False)

        flow = SequentialFlow.Builder().execute(step).then(charge_step).build()
    """

    def __init__(
        self,
        rules: Rules,
        fail_when_no_rule_fired: bool = False,
        engine: Optional[DefaultRulesEngine] = None,
        name: str = "rule-set-work",
    ) -> None:
        self._rules  = rules
        self._fail   = fail_when_no_rule_fired
        self._engine = engine or DefaultRulesEngine()
        self._name   = name

    def get_name(self) -> str:
        return self._name

    def execute(self, work_context: WorkContext) -> WorkReport:
        facts = Facts(**work_context.as_map())
        result = self._engine.fire(self._rules, facts)

        # Merge any fact changes back into the context
        for k, v in facts.as_map().items():
            work_context.put(k, v)

        any_fired = any(result.values())
        logger.debug("RuleSetWork '%s': %d rule(s) fired", self._name, sum(result.values()))

        status = WorkStatus.COMPLETED
        if self._fail and not any_fired:
            status = WorkStatus.FAILED

        return DefaultWorkReport(status, work_context)


# ════════════════════════════════════════════════════════════════════════════
#  flows + batch
# ════════════════════════════════════════════════════════════════════════════

class BatchWork(Work):
    """
    Execute a batch :class:`~maestro.batch.Job` as a
    :class:`~maestro.flows.Work` step inside a workflow.

    The job report is stored in the context under ``"batch_report"``
    (configurable via *report_key*).

    Returns COMPLETED when the job status is COMPLETED; FAILED otherwise.

    Example::

        from maestro.batch import JobBuilder, StringRecordReader, CollectionRecordWriter
        from maestro.flows import SequentialFlow, WorkContext
        from maestro.integration import BatchWork

        sink = []
        job  = JobBuilder().named("etl").reader(...).writer(CollectionRecordWriter(sink)).build()
        step = BatchWork(job, report_key="etl_report")

        flow = SequentialFlow.Builder().execute(step).then(validate_step).build()
    """

    def __init__(self, job, name: str = "batch-work", report_key: str = "batch_report") -> None:
        from maestro.batch._job import JobStatus
        self._job        = job
        self._name       = name
        self._report_key = report_key
        self._JobStatus  = JobStatus

    def get_name(self) -> str:
        return self._name

    def execute(self, work_context: WorkContext) -> WorkReport:
        logger.info("BatchWork '%s': starting job '%s'", self._name, self._job._params.name)
        try:
            report = self._job.call()
        except Exception as exc:
            logger.error("BatchWork '%s': job raised %s", self._name, exc)
            return DefaultWorkReport(WorkStatus.FAILED, work_context, error=exc)

        work_context.put(self._report_key, report)
        status = (WorkStatus.COMPLETED
                  if report.status == self._JobStatus.COMPLETED
                  else WorkStatus.FAILED)

        logger.info("BatchWork '%s': job finished — %s (written=%d)",
                    self._name, report.status.value, report.metrics.written_count)
        return DefaultWorkReport(status, work_context)


# ════════════════════════════════════════════════════════════════════════════
#  flows + states
# ════════════════════════════════════════════════════════════════════════════

class FSMGuardWork(Work):
    """
    Fire an :class:`~maestro.states.Event` against an FSM and use the resulting
    state to decide success or failure of this :class:`~maestro.flows.Work` step.

    Parameters
    ----------
    fsm:
        The :class:`~maestro.states.FiniteStateMachine` to fire against.
    event:
        The event to fire (or a callable ``(WorkContext) → Event`` for dynamic events).
    success_states:
        Set of states that count as COMPLETED. If ``None``, any non-error state succeeds.
    error_states:
        Set of states that count as FAILED. Takes priority over ``success_states``.
    ignore_undefined:
        When True, silently keep current state if no transition is registered.

    Example::

        from maestro.states import State, FiniteStateMachineBuilder
        from maestro.integration import FSMGuardWork

        fsm    = FiniteStateMachineBuilder(states={pending, approved, rejected},
                                           initial_state=pending).register_transition(...).build()
        guard  = FSMGuardWork(fsm=fsm, event=SubmitEvent(),
                              success_states={approved}, error_states={rejected})

        flow = SequentialFlow.Builder().execute(guard).then(process_step).build()
    """

    def __init__(
        self,
        fsm: FiniteStateMachine,
        event,
        success_states: Optional[Set[State]] = None,
        error_states:   Optional[Set[State]] = None,
        name: str = "fsm-guard",
    ) -> None:
        self._fsm            = fsm
        self._event          = event
        self._success_states = success_states
        self._error_states   = error_states or set()
        self._name           = name

    def get_name(self) -> str:
        return self._name

    def execute(self, work_context: WorkContext) -> WorkReport:
        event = self._event(work_context) if callable(self._event) else self._event
        try:
            new_state = self._fsm.fire(event)
        except NoSuchTransitionException as exc:
            logger.warning("FSMGuardWork '%s': no transition — %s", self._name, exc)
            return DefaultWorkReport(WorkStatus.FAILED, work_context, error=exc)

        work_context.put("fsm_state", new_state.name)
        logger.debug("FSMGuardWork '%s': transitioned to '%s'", self._name, new_state.name)

        if new_state in self._error_states:
            return DefaultWorkReport(WorkStatus.FAILED, work_context)
        if self._success_states and new_state not in self._success_states:
            return DefaultWorkReport(WorkStatus.FAILED, work_context)
        return DefaultWorkReport(WorkStatus.COMPLETED, work_context)


class FSMTransitionWork(Work):
    """
    Fire a **sequence** of FSM events as a single workflow step.

    Stops and returns FAILED on the first undefined transition.
    All intermediate states are appended to ``work_context["fsm_history"]``.

    Example::

        step = FSMTransitionWork(fsm, events=[PayEvent(), ShipEvent()], name="process-order")
    """

    def __init__(
        self,
        fsm: FiniteStateMachine,
        events: list,
        name: str = "fsm-transition",
    ) -> None:
        self._fsm    = fsm
        self._events = events
        self._name   = name

    def get_name(self) -> str:
        return self._name

    def execute(self, work_context: WorkContext) -> WorkReport:
        history = work_context.get("fsm_history", [])
        for event in self._events:
            ev = event(work_context) if callable(event) else event
            try:
                new_state = self._fsm.fire(ev)
                history.append(new_state.name)
            except NoSuchTransitionException as exc:
                work_context.put("fsm_history", history)
                return DefaultWorkReport(WorkStatus.FAILED, work_context, error=exc)

        work_context.put("fsm_state",   self._fsm.current_state.name)
        work_context.put("fsm_history", history)
        return DefaultWorkReport(WorkStatus.COMPLETED, work_context)


# ════════════════════════════════════════════════════════════════════════════
#  batch + rules
# ════════════════════════════════════════════════════════════════════════════

class RuleBasedFilter(RecordFilter):
    """
    Use a :class:`~maestro.rules.Rules` engine as a batch record filter.

    A record is **accepted** (passes through) when at least one rule fires
    against the record's payload turned into ``Facts``.

    The payload can be a ``dict``, a dataclass, or any object with ``__dict__``.

    Example::

        from maestro.rules import Rules, RuleBuilder
        from maestro.integration import RuleBasedFilter

        age_rule = RuleBuilder().name("adult").when(lambda f: int(f.get("age",0)) >= 18).build()
        job = JobBuilder().filter(RuleBasedFilter(Rules(age_rule))).build()
    """

    def __init__(
        self,
        rules: Rules,
        engine: Optional[DefaultRulesEngine] = None,
        accept_on_no_match: bool = False,
    ) -> None:
        self._rules    = rules
        self._engine   = engine or DefaultRulesEngine()
        self._no_match = accept_on_no_match

    def accept(self, record: Record) -> bool:
        facts = self._payload_to_facts(record.payload)
        result = self._engine.fire(self._rules, facts)
        any_fired = any(result.values())
        return any_fired if not self._no_match else (not any_fired or any_fired)

    def _payload_to_facts(self, payload) -> Facts:
        if isinstance(payload, dict):
            return Facts(**payload)
        if hasattr(payload, "__dict__"):
            return Facts(**{k: v for k, v in vars(payload).items() if not k.startswith("_")})
        # scalar — expose as "value"
        return Facts(value=payload)


class RuleBasedProcessor(RecordProcessor):
    """
    Apply a :class:`~maestro.rules.Rules` engine to enrich each record's payload.

    Rules can mutate ``Facts`` during execution; those changes are written back
    to the record payload (if payload is a dict or has ``__dict__``).

    Raises :exc:`~maestro.batch.RecordProcessingException` when ``fail_on_no_match=True``
    and no rule fired.

    Example::

        enrich_rule = RuleBuilder().name("tag-vip").when(lambda f: f.get("total",0)>1000)\\
                          .then(lambda f: f.put("tier","vip")).build()
        job = JobBuilder().processor(RuleBasedProcessor(Rules(enrich_rule))).build()
    """

    def __init__(
        self,
        rules: Rules,
        engine: Optional[DefaultRulesEngine] = None,
        fail_on_no_match: bool = False,
    ) -> None:
        self._rules    = rules
        self._engine   = engine or DefaultRulesEngine()
        self._fail     = fail_on_no_match

    def process_record(self, record: Record) -> Record:
        facts = self._payload_to_facts(record.payload)
        result = self._engine.fire(self._rules, facts)
        any_fired = any(result.values())

        if self._fail and not any_fired:
            raise RecordProcessingException(
                f"No rule fired for record #{record.header.number}"
            )

        # Write enriched facts back into the payload
        if isinstance(record.payload, dict):
            record.payload = facts.as_map()
        elif hasattr(record.payload, "__dict__"):
            for k, v in facts.as_map().items():
                if hasattr(record.payload, k):
                    setattr(record.payload, k, v)

        return record

    def _payload_to_facts(self, payload) -> Facts:
        if isinstance(payload, dict): return Facts(**payload)
        if hasattr(payload, "__dict__"):
            return Facts(**{k: v for k, v in vars(payload).items() if not k.startswith("_")})
        return Facts(value=payload)


__all__ = [
    "RuleSetWork", "BatchWork",
    "FSMGuardWork", "FSMTransitionWork",
    "RuleBasedFilter", "RuleBasedProcessor",
]
