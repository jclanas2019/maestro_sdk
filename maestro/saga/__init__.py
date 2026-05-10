"""
maestro.saga — distributed saga pattern with compensation actions.

A Saga is a sequence of steps where every completed step has an optional
*compensating action* that undoes its effects. If any step fails, the Saga
automatically runs compensations in reverse order — rolling back successfully
completed steps and leaving the system in a consistent state.

    from maestro.saga import SagaBuilder, SagaStatus

    saga = (
        SagaBuilder()
        .named("create-order")
        .step("reserve-inventory",
              work         = ReserveInventoryWork(),
              compensation = ReleaseInventoryWork())
        .step("charge-payment",
              work         = ChargePaymentWork(),
              compensation = RefundPaymentWork())
        .step("update-shipping",
              work         = UpdateShippingWork())   # no compensation needed
        .build()
    )

    report = saga.execute(WorkContext(order_id="ORD-42"))
    # On success:  SagaStatus.COMPLETED
    # On failure:  SagaStatus.COMPENSATED  (all compensations ran)
    #              SagaStatus.PARTIALLY_COMPENSATED (some compensations failed)

Execution model (Erlang-style)
-------------------------------
    T1 → T2 → T3 → ... → Tn          (forward)
    Failure at T3:
    T3(FAIL) → C2 → C1               (backward)

Use as a Work inside any flow::

    flow = SequentialFlow.Builder()
           .execute(SagaWork(saga))
           .then(notify_step)
           .build()
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from maestro.flows._work import DefaultWorkReport, Work, WorkContext, WorkReport, WorkStatus

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  SagaStatus
# ════════════════════════════════════════════════════════════════════════════

class SagaStatus(enum.Enum):
    COMPLETED             = "COMPLETED"              # all steps succeeded
    FAILED                = "FAILED"                 # step failed, no compensation defined
    COMPENSATED           = "COMPENSATED"             # failed + all compensations ran OK
    PARTIALLY_COMPENSATED = "PARTIALLY_COMPENSATED"  # failed + some compensations also failed


# ════════════════════════════════════════════════════════════════════════════
#  SagaStep
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SagaStep:
    """
    A single step in a saga.

    Attributes:
        name:         Unique identifier for logging and reporting.
        work:         The forward :class:`~maestro.flows.Work` to execute.
        compensation: Optional :class:`~maestro.flows.Work` that undoes *work*
                      if a later step fails.
    """
    name:         str
    work:         Work
    compensation: Optional[Work] = None

    @property
    def is_compensatable(self) -> bool:
        return self.compensation is not None


# ════════════════════════════════════════════════════════════════════════════
#  Step results
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class StepResult:
    """Execution result of a single forward step."""
    name:        str
    status:      WorkStatus
    report:      Optional[WorkReport] = None
    error:       Optional[Exception]  = None
    duration:    float                = 0.0

    @property
    def succeeded(self) -> bool: return self.status == WorkStatus.COMPLETED
    @property
    def failed(self) -> bool:    return self.status == WorkStatus.FAILED


@dataclass
class CompensationResult:
    """Result of a single compensation (rollback) step."""
    step_name:   str
    status:      WorkStatus
    report:      Optional[WorkReport] = None
    error:       Optional[Exception]  = None
    duration:    float                = 0.0

    @property
    def succeeded(self) -> bool: return self.status == WorkStatus.COMPLETED


# ════════════════════════════════════════════════════════════════════════════
#  SagaReport
# ════════════════════════════════════════════════════════════════════════════

class SagaReport(WorkReport):
    """
    Complete execution report for a :class:`Saga` run.

    Carries per-step forward results, per-step compensation results,
    and the final :class:`SagaStatus`.
    """

    def __init__(
        self,
        saga_status:         SagaStatus,
        work_context:        WorkContext,
        step_results:        list[StepResult],
        compensation_results: list[CompensationResult],
        failed_step:         Optional[str] = None,
        error:               Optional[Exception] = None,
    ) -> None:
        self._saga_status          = saga_status
        self._work_context         = work_context
        self._step_results         = step_results
        self._compensation_results = compensation_results
        self._failed_step          = failed_step
        self._error                = error

    @property
    def status(self) -> WorkStatus:
        return (WorkStatus.COMPLETED
                if self._saga_status == SagaStatus.COMPLETED
                else WorkStatus.FAILED)

    @property
    def saga_status(self) -> SagaStatus:
        return self._saga_status

    @property
    def work_context(self) -> WorkContext:
        return self._work_context

    @property
    def error(self) -> Optional[Exception]:
        return self._error

    @property
    def step_results(self) -> list[StepResult]:
        return list(self._step_results)

    @property
    def compensation_results(self) -> list[CompensationResult]:
        return list(self._compensation_results)

    @property
    def failed_step(self) -> Optional[str]:
        return self._failed_step

    @property
    def succeeded_steps(self) -> list[str]:
        return [r.name for r in self._step_results if r.succeeded]

    @property
    def compensated_steps(self) -> list[str]:
        return [r.step_name for r in self._compensation_results if r.succeeded]

    @property
    def failed_compensations(self) -> list[str]:
        return [r.step_name for r in self._compensation_results if not r.succeeded]

    def __str__(self) -> str:
        lines = [f"SagaReport — {self._saga_status.value}"]
        for sr in self._step_results:
            sym = "✓" if sr.succeeded else "✗"
            lines.append(f"  {sym} [{sr.name}] {sr.status.value} ({sr.duration*1000:.1f}ms)")
        if self._compensation_results:
            lines.append("  ── compensations ──")
            for cr in self._compensation_results:
                sym = "↩" if cr.succeeded else "⚠"
                lines.append(f"  {sym} [{cr.step_name}] {cr.status.value} ({cr.duration*1000:.1f}ms)")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
#  SagaListener
# ════════════════════════════════════════════════════════════════════════════

class SagaListener:
    """Override the methods you need to observe saga execution."""

    def on_step_started(self, step: SagaStep, ctx: WorkContext) -> None: pass
    def on_step_completed(self, step: SagaStep, result: StepResult) -> None: pass
    def on_step_failed(self, step: SagaStep, result: StepResult) -> None: pass
    def on_compensation_started(self, step: SagaStep, ctx: WorkContext) -> None: pass
    def on_compensation_completed(self, step: SagaStep, result: CompensationResult) -> None: pass
    def on_compensation_failed(self, step: SagaStep, result: CompensationResult) -> None: pass
    def on_saga_completed(self, report: SagaReport) -> None: pass
    def on_saga_compensated(self, report: SagaReport) -> None: pass


class LoggingSagaListener(SagaListener):
    """Logs every saga event at INFO / WARNING level."""

    def on_step_started(self, step, ctx):
        logger.info("Saga step '%s': starting", step.name)

    def on_step_completed(self, step, result):
        logger.info("Saga step '%s': COMPLETED (%.1fms)", step.name, result.duration * 1000)

    def on_step_failed(self, step, result):
        logger.warning("Saga step '%s': FAILED — %s", step.name, result.error)

    def on_compensation_started(self, step, ctx):
        logger.info("Saga compensation '%s': starting", step.name)

    def on_compensation_completed(self, step, result):
        logger.info("Saga compensation '%s': COMPLETED", step.name)

    def on_compensation_failed(self, step, result):
        logger.error("Saga compensation '%s': FAILED — %s", step.name, result.error)

    def on_saga_completed(self, report):
        logger.info("Saga COMPLETED — %d steps succeeded", len(report.succeeded_steps))

    def on_saga_compensated(self, report):
        logger.warning("Saga COMPENSATED — failed at '%s', rolled back %d step(s)",
                       report.failed_step, len(report.compensated_steps))


# ════════════════════════════════════════════════════════════════════════════
#  Saga
# ════════════════════════════════════════════════════════════════════════════

import time as _time


class Saga(Work):
    """
    A sequence of :class:`SagaStep` objects with automatic compensation on failure.

    Also implements :class:`~maestro.flows.Work` so it can be embedded
    directly into any workflow::

        flow = SequentialFlow.Builder()
               .execute(saga)            # Saga IS a Work
               .then(post_process_step)
               .build()

    Create via :class:`SagaBuilder`.
    """

    def __init__(
        self,
        name:                 str,
        steps:                list[SagaStep],
        listeners:            Optional[list[SagaListener]] = None,
        fail_on_compensation_error: bool = False,
    ) -> None:
        self._name       = name
        self._steps      = steps
        self._listeners  = listeners or [LoggingSagaListener()]
        self._fail_comp  = fail_on_compensation_error

    def get_name(self) -> str:
        return self._name

    # ── Forward execution ─────────────────────────────────────────────── #

    def execute(self, work_context: WorkContext) -> SagaReport:
        logger.info("Saga '%s': starting %d step(s)", self._name, len(self._steps))

        step_results:  list[StepResult]        = []
        completed:     list[SagaStep]          = []   # steps eligible for compensation
        failed_step:   Optional[str]           = None

        for step in self._steps:
            for l in self._listeners: l.on_step_started(step, work_context)

            t0 = _time.monotonic()
            try:
                report = step.work.execute(work_context)
                status = report.status
                error  = report.error
            except Exception as exc:
                status = WorkStatus.FAILED
                report = DefaultWorkReport(WorkStatus.FAILED, work_context, error=exc)
                error  = exc

            duration = _time.monotonic() - t0
            sr = StepResult(name=step.name, status=status, report=report,
                            error=error, duration=duration)
            step_results.append(sr)

            if status == WorkStatus.COMPLETED:
                for l in self._listeners: l.on_step_completed(step, sr)
                if step.is_compensatable:
                    completed.append(step)
            else:
                for l in self._listeners: l.on_step_failed(step, sr)
                failed_step = step.name
                break

        # All steps completed
        if failed_step is None:
            report = SagaReport(
                saga_status=SagaStatus.COMPLETED,
                work_context=work_context,
                step_results=step_results,
                compensation_results=[],
            )
            for l in self._listeners: l.on_saga_completed(report)
            return report

        # Run compensations in reverse order
        comp_results = self._compensate(list(reversed(completed)), work_context)
        failed_comps = [r for r in comp_results if not r.succeeded]

        saga_status = (SagaStatus.PARTIALLY_COMPENSATED
                       if failed_comps
                       else (SagaStatus.COMPENSATED
                             if completed
                             else SagaStatus.FAILED))

        report = SagaReport(
            saga_status=saga_status,
            work_context=work_context,
            step_results=step_results,
            compensation_results=comp_results,
            failed_step=failed_step,
            error=step_results[-1].error,
        )
        for l in self._listeners: l.on_saga_compensated(report)
        return report

    def _compensate(
        self, steps: list[SagaStep], ctx: WorkContext
    ) -> list[CompensationResult]:
        results: list[CompensationResult] = []
        for step in steps:
            if not step.is_compensatable:
                continue
            for l in self._listeners: l.on_compensation_started(step, ctx)
            t0 = _time.monotonic()
            try:
                rep    = step.compensation.execute(ctx)
                status = rep.status
                error  = rep.error
            except Exception as exc:
                status = WorkStatus.FAILED
                error  = exc

            duration = _time.monotonic() - t0
            cr = CompensationResult(step_name=step.name, status=status,
                                    error=error, duration=duration)
            results.append(cr)

            if status == WorkStatus.COMPLETED:
                for l in self._listeners: l.on_compensation_completed(step, cr)
            else:
                for l in self._listeners: l.on_compensation_failed(step, cr)
                if self._fail_comp:
                    logger.error("Compensation failed for '%s' — stopping compensation chain",
                                 step.name)
                    break
        return results


# ════════════════════════════════════════════════════════════════════════════
#  SagaBuilder
# ════════════════════════════════════════════════════════════════════════════

class SagaBuilder:
    """
    Fluent builder for :class:`Saga`.

    Example::

        saga = (
            SagaBuilder()
            .named("book-trip")
            .step("book-flight",  book_flight_work,  cancel_flight_work)
            .step("book-hotel",   book_hotel_work,   cancel_hotel_work)
            .step("charge-card",  charge_work,       refund_work)
            .on_failure(notify_ops_work)     # optional: always runs on failure
            .build()
        )
    """

    def __init__(self) -> None:
        self._name      = "saga"
        self._steps:    list[SagaStep]       = []
        self._listeners: list[SagaListener]  = [LoggingSagaListener()]
        self._fail_comp = False
        self._on_failure: Optional[Work]     = None

    def named(self, name: str) -> "SagaBuilder":
        self._name = name; return self

    def step(
        self,
        name:         str,
        work:         Work,
        compensation: Optional[Work] = None,
    ) -> "SagaBuilder":
        """
        Add a step.

        Args:
            name:         Unique identifier for this step.
            work:         Forward action.
            compensation: Optional rollback action.
        """
        self._steps.append(SagaStep(name=name, work=work, compensation=compensation))
        return self

    def fail_on_compensation_error(self, value: bool = True) -> "SagaBuilder":
        """Stop compensating if a compensation itself fails."""
        self._fail_comp = value; return self

    def add_listener(self, listener: SagaListener) -> "SagaBuilder":
        self._listeners.append(listener); return self

    def quiet(self) -> "SagaBuilder":
        """Remove the default LoggingSagaListener."""
        self._listeners = [l for l in self._listeners
                           if not isinstance(l, LoggingSagaListener)]
        return self

    def build(self) -> Saga:
        if not self._steps:
            raise ValueError("A Saga requires at least one step.")
        return Saga(
            name=self._name,
            steps=list(self._steps),
            listeners=list(self._listeners),
            fail_on_compensation_error=self._fail_comp,
        )


# ════════════════════════════════════════════════════════════════════════════
#  Convenience: lambda-based steps
# ════════════════════════════════════════════════════════════════════════════

def saga_step(
    name: str,
    fn:   Callable[[WorkContext], Any],
    compensation_fn: Optional[Callable[[WorkContext], Any]] = None,
) -> SagaStep:
    """
    Create a :class:`SagaStep` from plain callables.

    Example::

        step = saga_step(
            "reserve",
            fn              = lambda ctx: reserve_inventory(ctx.get("item_id")),
            compensation_fn = lambda ctx: release_inventory(ctx.get("item_id")),
        )
    """
    from maestro.flows._work import LambdaWork
    work = LambdaWork(fn, name=name)
    comp = LambdaWork(compensation_fn, name=f"compensate-{name}") if compensation_fn else None
    return SagaStep(name=name, work=work, compensation=comp)


__all__ = [
    "SagaStatus", "SagaStep", "StepResult", "CompensationResult",
    "SagaReport", "SagaListener", "LoggingSagaListener",
    "Saga", "SagaBuilder", "saga_step",
]
