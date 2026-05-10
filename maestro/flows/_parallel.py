"""
ParallelFlow — executes work units concurrently using a thread pool.

Example::

    import concurrent.futures

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    flow = (
        ParallelFlow.Builder()
        .named("parallel-download")
        .execute(download_a, download_b, download_c)
        .with_executor(executor)
        .build()
    )
    report = flow.execute(WorkContext())
    executor.shutdown()
"""
from __future__ import annotations

import concurrent.futures
import logging
import uuid
from typing import Optional

from maestro.flows._work import (
    DefaultWorkReport, Work, WorkContext, WorkReport, WorkStatus,
)

logger = logging.getLogger(__name__)


class ParallelFlowReport(WorkReport):
    """
    Aggregate report for a :class:`ParallelFlow`.

    * Status is **COMPLETED** if every work unit completed successfully.
    * Status is **FAILED** if any work unit failed.
    """

    def __init__(self, reports: list[WorkReport], work_context: WorkContext) -> None:
        self._reports = reports
        self._work_context = work_context

    @property
    def status(self) -> WorkStatus:
        failed = any(r.status == WorkStatus.FAILED for r in self._reports)
        return WorkStatus.FAILED if failed else WorkStatus.COMPLETED

    @property
    def work_context(self) -> WorkContext:
        return self._work_context

    @property
    def error(self):
        for r in self._reports:
            if r.error is not None:
                return r.error
        return None

    @property
    def reports(self) -> list[WorkReport]:
        return list(self._reports)

    def __repr__(self) -> str:
        return (
            f"ParallelFlowReport(status={self.status.value}, "
            f"reports={len(self._reports)})"
        )


class ParallelFlow(Work):
    """
    Executes all work units **concurrently** using a
    :class:`~concurrent.futures.Executor`.

    Each work unit receives a **copy** of the WorkContext so they don't
    interfere with each other.  After all units complete, their contexts
    are merged (last-writer-wins per key) back into the original context.

    Returns a :class:`ParallelFlowReport`.
    """

    def __init__(
        self,
        name: str,
        works: list[Work],
        executor: Optional[concurrent.futures.Executor] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._name     = name
        self._works    = works
        self._executor = executor
        self._timeout  = timeout
        self._owns_executor = executor is None

    def get_name(self) -> str:
        return self._name

    def execute(self, work_context: WorkContext) -> ParallelFlowReport:
        logger.info("Running parallel flow '%s' with %d work units", self._name, len(self._works))

        executor = self._executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self._works) or 1
        )
        try:
            futures: dict[concurrent.futures.Future, Work] = {}
            # Each work gets its own context snapshot so they don't race
            contexts: dict[concurrent.futures.Future, WorkContext] = {}

            for work in self._works:
                ctx_copy = WorkContext(**work_context.as_map())
                future = executor.submit(work.execute, ctx_copy)
                futures[future] = work
                contexts[future] = ctx_copy

            reports: list[WorkReport] = []
            for future in concurrent.futures.as_completed(futures, timeout=self._timeout):
                work = futures[future]
                try:
                    report = future.result()
                    reports.append(report)
                    # Merge worker context back
                    for k, v in contexts[future].as_map().items():
                        work_context.put(k, v)
                except Exception as exc:
                    logger.error("Work '%s' raised in parallel: %s", work.get_name(), exc)
                    reports.append(
                        DefaultWorkReport(WorkStatus.FAILED, work_context, error=exc)
                    )
        finally:
            if self._owns_executor:
                executor.shutdown(wait=False)

        result = ParallelFlowReport(reports, work_context)
        logger.info(
            "Parallel flow '%s' completed with status %s", self._name, result.status
        )
        return result

    # ─────────────────────────────── Builder ─────────────────────────── #

    class Builder:
        def __init__(self) -> None:
            self._name     = f"parallel-flow-{uuid.uuid4().hex[:8]}"
            self._works:   list[Work] = []
            self._executor: Optional[concurrent.futures.Executor] = None
            self._timeout:  Optional[float] = None

        def named(self, name: str) -> "ParallelFlow.Builder":
            self._name = name
            return self

        def execute(self, *works: Work) -> "ParallelFlow.Builder":
            """Add one or more work units to run in parallel."""
            self._works.extend(works)
            return self

        def with_executor(
            self, executor: concurrent.futures.Executor
        ) -> "ParallelFlow.Builder":
            """Supply an external executor (caller is responsible for shutdown)."""
            self._executor = executor
            return self

        def timeout(self, seconds: float) -> "ParallelFlow.Builder":
            """Maximum seconds to wait for all parallel units to complete."""
            self._timeout = seconds
            return self

        def build(self) -> "ParallelFlow":
            if not self._works:
                raise ValueError("ParallelFlow needs at least one work unit.")
            return ParallelFlow(
                name=self._name,
                works=list(self._works),
                executor=self._executor,
                timeout=self._timeout,
            )
