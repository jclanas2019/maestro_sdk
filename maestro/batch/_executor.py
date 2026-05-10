"""
JobExecutor — submits and runs :class:`Job` objects.

Two execution modes:
* **Sequential** — ``execute(job)`` runs a single job in the calling thread.
* **Parallel**   — ``execute_all(jobs)`` runs multiple jobs concurrently using
  a :class:`~concurrent.futures.ThreadPoolExecutor`.

Example::

    executor = JobExecutor()
    report   = executor.execute(job)
    executor.shutdown()

    # or parallel
    reports = executor.execute_all([job1, job2, job3])
"""
from __future__ import annotations

import concurrent.futures
import logging
from typing import Optional

from maestro.batch._job import Job, JobReport

logger = logging.getLogger(__name__)


class JobExecutor:
    """
    Manages the execution of one or more batch jobs.

    Args:
        max_workers: Thread pool size for :meth:`execute_all`.
                     Defaults to the number of jobs (one thread per job).
    """

    def __init__(self, max_workers: Optional[int] = None) -> None:
        self._max_workers = max_workers
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    # ------------------------------------------------------------------ #
    #  Sequential                                                          #
    # ------------------------------------------------------------------ #

    def execute(self, job: Job) -> JobReport:
        """Run *job* synchronously in the calling thread."""
        logger.info("Starting job '%s'", job._params.name)
        return job.call()

    # ------------------------------------------------------------------ #
    #  Parallel                                                            #
    # ------------------------------------------------------------------ #

    def execute_all(self, jobs: list[Job]) -> list[JobReport]:
        """
        Submit all *jobs* to a thread pool and block until all complete.
        Returns a list of :class:`JobReport` objects in the same order as *jobs*.
        """
        workers = self._max_workers or len(jobs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(job.call): idx for idx, job in enumerate(jobs)}
            reports: list[Optional[JobReport]] = [None] * len(jobs)
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    reports[idx] = future.result()
                except Exception as exc:
                    logger.error("Job %d raised an unhandled exception: %s", idx, exc)
        return reports  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def shutdown(self) -> None:
        """Shut down the internal thread pool (no-op in sequential mode)."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
