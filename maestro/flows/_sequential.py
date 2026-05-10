"""
SequentialFlow — executes work units one after another.

If any unit returns FAILED the flow stops immediately and returns
that failed report.

Example::

    flow = (
        SequentialFlow.Builder()
        .named("my-pipeline")
        .execute(step1)
        .then(step2)
        .then(step3)
        .build()
    )
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from maestro.flows._work import (
    DefaultWorkReport, Work, WorkContext, WorkReport, WorkStatus,
)

logger = logging.getLogger(__name__)


class SequentialFlow(Work):
    """
    Executes a list of :class:`~easy_flows.work.Work` units in order.

    * Stops on the first FAILED result.
    * The :class:`~easy_flows.work.WorkContext` is shared and threaded through
      every step.
    * Returns the last executed step's report.
    """

    def __init__(self, name: str, works: list[Work]) -> None:
        self._name = name
        self._works = works

    def get_name(self) -> str:
        return self._name

    def execute(self, work_context: WorkContext) -> WorkReport:
        logger.info("Running sequential flow '%s'", self._name)
        report: WorkReport = DefaultWorkReport(WorkStatus.COMPLETED, work_context)

        for work in self._works:
            logger.debug("Executing work '%s'", work.get_name())
            try:
                report = work.execute(work_context)
            except Exception as exc:
                logger.error("Work '%s' raised: %s", work.get_name(), exc)
                report = DefaultWorkReport(WorkStatus.FAILED, work_context, error=exc)

            if report.status == WorkStatus.FAILED:
                logger.info(
                    "Sequential flow '%s' stopped — '%s' failed: %s",
                    self._name, work.get_name(), report.error,
                )
                return report

        logger.info("Sequential flow '%s' completed", self._name)
        return report

    # ─────────────────────────────── Builder ─────────────────────────── #

    class Builder:
        """Fluent builder for :class:`SequentialFlow`."""

        def __init__(self) -> None:
            self._name = f"sequential-flow-{uuid.uuid4().hex[:8]}"
            self._works: list[Work] = []

        def named(self, name: str) -> "SequentialFlow.Builder":
            self._name = name
            return self

        def execute(self, work: Work) -> "SequentialFlow.Builder":
            """Set the first (or only) work unit."""
            self._works.append(work)
            return self

        def then(self, work: Work) -> "SequentialFlow.Builder":
            """Append subsequent work units."""
            self._works.append(work)
            return self

        def build(self) -> "SequentialFlow":
            if not self._works:
                raise ValueError("A SequentialFlow needs at least one work unit.")
            return SequentialFlow(self._name, list(self._works))
