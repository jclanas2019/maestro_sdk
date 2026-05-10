"""
ConditionalFlow — executes a work unit then branches based on a predicate.

Flow logic::

    execute(initial_work)
    if predicate(report):
        execute(then_work)
    else:
        execute(otherwise_work)   # optional

Example::

    flow = (
        ConditionalFlow.Builder()
        .named("validate-then-process")
        .execute(validation_work)
        .when(WorkReportPredicate.COMPLETED)
        .then(processing_work)
        .otherwise(error_handler_work)
        .build()
    )
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from maestro.flows._predicate import WorkReportPredicate
from maestro.flows._work import (
    DefaultWorkReport, NoOpWork, Work, WorkContext, WorkReport, WorkStatus,
)

logger = logging.getLogger(__name__)


class ConditionalFlow(Work):
    """
    Executes *initial* work, then based on a :class:`WorkReportPredicate`
    either runs *then_work* or *otherwise_work*.

    * *otherwise_work* is optional (defaults to a no-op).
    * The returned report is from whichever branch ran (or the initial work
      if neither branch is defined and the predicate is False).
    """

    def __init__(
        self,
        name: str,
        initial_work: Work,
        predicate: WorkReportPredicate,
        then_work: Work,
        otherwise_work: Optional[Work] = None,
    ) -> None:
        self._name         = name
        self._initial_work = initial_work
        self._predicate    = predicate
        self._then_work    = then_work
        self._otherwise    = otherwise_work or NoOpWork(name="no-op-otherwise")

    def get_name(self) -> str:
        return self._name

    def execute(self, work_context: WorkContext) -> WorkReport:
        logger.info("Running conditional flow '%s'", self._name)

        try:
            initial_report = self._initial_work.execute(work_context)
        except Exception as exc:
            logger.error("Initial work '%s' raised: %s", self._initial_work.get_name(), exc)
            return DefaultWorkReport(WorkStatus.FAILED, work_context, error=exc)

        if self._predicate.test(initial_report):
            logger.debug(
                "Predicate TRUE — executing 'then' branch: '%s'",
                self._then_work.get_name(),
            )
            branch = self._then_work
        else:
            logger.debug(
                "Predicate FALSE — executing 'otherwise' branch: '%s'",
                self._otherwise.get_name(),
            )
            branch = self._otherwise

        try:
            report = branch.execute(work_context)
        except Exception as exc:
            logger.error("Branch work '%s' raised: %s", branch.get_name(), exc)
            report = DefaultWorkReport(WorkStatus.FAILED, work_context, error=exc)

        logger.info("Conditional flow '%s' completed with status %s", self._name, report.status)
        return report

    # ─────────────────────────────── Builder ─────────────────────────── #

    class Builder:
        def __init__(self) -> None:
            self._name = f"conditional-flow-{uuid.uuid4().hex[:8]}"
            self._initial_work: Optional[Work] = None
            self._predicate: Optional[WorkReportPredicate] = None
            self._then_work: Optional[Work] = None
            self._otherwise_work: Optional[Work] = None

        def named(self, name: str) -> "ConditionalFlow.Builder":
            self._name = name
            return self

        def execute(self, work: Work) -> "ConditionalFlow.Builder":
            """Set the initial work unit to execute before branching."""
            self._initial_work = work
            return self

        def when(self, predicate: WorkReportPredicate) -> "ConditionalFlow.Builder":
            """Set the predicate that decides which branch to take."""
            self._predicate = predicate
            return self

        def then(self, work: Work) -> "ConditionalFlow.Builder":
            """Set the work to execute when the predicate is True."""
            self._then_work = work
            return self

        def otherwise(self, work: Work) -> "ConditionalFlow.Builder":
            """Set the work to execute when the predicate is False."""
            self._otherwise_work = work
            return self

        def build(self) -> "ConditionalFlow":
            if self._initial_work is None:
                raise ValueError("ConditionalFlow requires an initial work (.execute(…)).")
            if self._predicate is None:
                raise ValueError("ConditionalFlow requires a predicate (.when(…)).")
            if self._then_work is None:
                raise ValueError("ConditionalFlow requires a 'then' branch (.then(…)).")
            return ConditionalFlow(
                name=self._name,
                initial_work=self._initial_work,
                predicate=self._predicate,
                then_work=self._then_work,
                otherwise_work=self._otherwise_work,
            )
