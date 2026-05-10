"""
WorkFlowEngine — the entry point for running workflows.

Example::

    engine = WorkFlowEngine.Builder().build()
    report = engine.run(my_workflow, WorkContext())
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from maestro.flows._work import Work, WorkContext, WorkReport

logger = logging.getLogger(__name__)


class WorkFlowEngine:
    """
    Runs a :class:`~easy_flows.work.Work` (or any
    :class:`~easy_flows.workflow.*Flow`) against a :class:`~easy_flows.work.WorkContext`.

    The engine is intentionally thin — it delegates everything to the
    workflow's own ``execute`` method and simply adds logging.

    Example::

        engine = WorkFlowEngine.Builder().build()
        ctx    = WorkContext()
        report = engine.run(workflow, ctx)
        print(report.status)
    """

    def __init__(self, name: str = "") -> None:
        self._name = name or f"workflow-engine-{uuid.uuid4().hex[:8]}"

    def run(self, workflow: Work, work_context: WorkContext) -> WorkReport:
        """
        Execute *workflow* with *work_context*.

        Args:
            workflow:     Any :class:`~easy_flows.work.Work` — including
                          SequentialFlow, ConditionalFlow, ParallelFlow or
                          RepeatFlow.
            work_context: The shared execution context.

        Returns:
            A :class:`~easy_flows.work.WorkReport`.
        """
        logger.info(
            "Engine '%s' running workflow '%s'",
            self._name, workflow.get_name(),
        )
        report = workflow.execute(work_context)
        logger.info(
            "Engine '%s' finished — status=%s",
            self._name, report.status.value,
        )
        return report

    # ─────────────────────────────── Builder ─────────────────────────── #

    class Builder:
        def __init__(self) -> None:
            self._name = ""

        def named(self, name: str) -> "WorkFlowEngine.Builder":
            self._name = name
            return self

        def build(self) -> "WorkFlowEngine":
            return WorkFlowEngine(name=self._name)
