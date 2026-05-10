"""
RepeatFlow — repeats a work unit a fixed number of times or until a predicate
is satisfied.

Two modes
---------
* **Fixed repetitions** — ``.times(N)``: run N times regardless of outcome.
* **Until predicate**   — ``.until(predicate)``: repeat *while* predicate is
  False, stop when it becomes True (or ``max_times`` is reached).

Example::

    # Repeat exactly 3 times
    flow = RepeatFlow.Builder().named("retry-3x").repeat(work).times(3).build()

    # Repeat until the report is COMPLETED (or 10 tries max)
    flow = (
        RepeatFlow.Builder()
        .named("retry-until-done")
        .repeat(work)
        .until(WorkReportPredicate.COMPLETED)
        .times(10)
        .build()
    )
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from maestro.flows._predicate import WorkReportPredicate
from maestro.flows._work import (
    DefaultWorkReport, Work, WorkContext, WorkReport, WorkStatus,
)

logger = logging.getLogger(__name__)

_MAX_LOOP_GUARD = 10_000   # safety ceiling


class RepeatFlow(Work):
    """
    Repeats a single work unit according to a fixed count and/or an
    until-predicate.

    If ``until`` is set:
        * Run the work, check ``until(report)``.
        * If True → stop and return the last report.
        * If False → repeat (up to ``times`` iterations).

    If only ``times`` is set (no predicate):
        * Run the work exactly ``times`` times.
        * Stop early on FAILED.
    """

    def __init__(
        self,
        name: str,
        work: Work,
        times: int = 1,
        until: Optional[WorkReportPredicate] = None,
    ) -> None:
        self._name  = name
        self._work  = work
        self._times = max(1, times)
        self._until = until

    def get_name(self) -> str:
        return self._name

    def execute(self, work_context: WorkContext) -> WorkReport:
        logger.info(
            "Running repeat flow '%s' (times=%d, until=%s)",
            self._name, self._times, self._until,
        )
        report: WorkReport = DefaultWorkReport(WorkStatus.COMPLETED, work_context)
        iterations = 0

        while iterations < self._times:
            try:
                report = self._work.execute(work_context)
            except Exception as exc:
                logger.error(
                    "Work '%s' raised on iteration %d: %s",
                    self._work.get_name(), iterations + 1, exc,
                )
                report = DefaultWorkReport(WorkStatus.FAILED, work_context, error=exc)

            iterations += 1
            logger.debug(
                "Repeat flow '%s' — iteration %d/%d — status=%s",
                self._name, iterations, self._times, report.status.value,
            )

            # Stop early on failure (only in fixed-repetition mode)
            if self._until is None and report.status == WorkStatus.FAILED:
                logger.info(
                    "Repeat flow '%s' stopped early — work failed on iteration %d",
                    self._name, iterations,
                )
                break

            # Until-predicate check
            if self._until is not None and self._until.test(report):
                logger.info(
                    "Repeat flow '%s' — until predicate satisfied after %d iterations",
                    self._name, iterations,
                )
                break

        logger.info(
            "Repeat flow '%s' finished after %d iteration(s) — status=%s",
            self._name, iterations, report.status.value,
        )
        return report

    # ─────────────────────────────── Builder ─────────────────────────── #

    class Builder:
        def __init__(self) -> None:
            self._name  = f"repeat-flow-{uuid.uuid4().hex[:8]}"
            self._work: Optional[Work] = None
            self._times = 1
            self._until: Optional[WorkReportPredicate] = None

        def named(self, name: str) -> "RepeatFlow.Builder":
            self._name = name
            return self

        def repeat(self, work: Work) -> "RepeatFlow.Builder":
            """Set the work unit to repeat."""
            self._work = work
            return self

        def times(self, n: int) -> "RepeatFlow.Builder":
            """Repeat exactly *n* times (or up to *n* times when ``until`` is set)."""
            self._times = max(1, n)
            return self

        def until(self, predicate: WorkReportPredicate) -> "RepeatFlow.Builder":
            """Repeat until *predicate* is True."""
            self._until = predicate
            return self

        def build(self) -> "RepeatFlow":
            if self._work is None:
                raise ValueError("RepeatFlow requires a work unit (.repeat(…)).")
            return RepeatFlow(
                name=self._name,
                work=self._work,
                times=self._times,
                until=self._until,
            )
