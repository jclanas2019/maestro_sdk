"""
maestro.graph — DAG-based workflow with automatic dependency resolution.

Unlike ``SequentialFlow`` (A then B then C) or ``ParallelFlow`` (A and B and C),
``GraphFlow`` lets you declare *exactly which steps depend on which* and
automatically extracts the maximum possible parallelism.

    from maestro.graph import GraphFlow, GraphBuilder, GraphReport

    flow = (
        GraphBuilder()
        .add("fetch-a",    fetch_a_work)
        .add("fetch-b",    fetch_b_work)
        .add("transform",  transform_work,  depends_on=["fetch-a", "fetch-b"])
        .add("validate",   validate_work,   depends_on=["transform"])
        .add("persist",    persist_work,    depends_on=["validate"])
        .add("notify",     notify_work,     depends_on=["validate"])
        .build()
    )
    # Execution order:
    #   [fetch-a, fetch-b]  → parallel
    #   [transform]         → after both fetches complete
    #   [validate]          → after transform
    #   [persist, notify]   → parallel, after validate

    engine = WorkFlowEngine()
    report = engine.run(flow, WorkContext())
    print(report.node_reports)        # per-node status and duration

Design
------
* Uses Kahn's algorithm: nodes become "ready" as their dependencies satisfy.
* Ready nodes are submitted to a ``ThreadPoolExecutor`` immediately.
* ``fail_fast=True``  (default): stop on the first failed node.
* ``fail_fast=False``: run all non-blocked nodes, collect all failures.
* ``NodeReport`` carries the Work's report, wall-clock duration, and order of completion.
* Cycle detection raises ``CyclicDependencyError`` at build time.
"""
from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from maestro.flows._work import DefaultWorkReport, Work, WorkContext, WorkReport, WorkStatus

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  Exceptions
# ════════════════════════════════════════════════════════════════════════════

class CyclicDependencyError(Exception):
    """Raised when the declared graph contains a cycle."""


class UnknownDependencyError(Exception):
    """Raised when a node lists a dependency that does not exist."""


# ════════════════════════════════════════════════════════════════════════════
#  NodeReport — per-node execution result
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class NodeReport:
    """Execution result for a single graph node."""
    name:          str
    status:        WorkStatus
    duration:      float                  = 0.0
    completion_order: int                 = 0
    work_report:   Optional[WorkReport]   = None
    error:         Optional[Exception]    = None

    @property
    def succeeded(self) -> bool: return self.status == WorkStatus.COMPLETED
    @property
    def failed(self) -> bool:    return self.status == WorkStatus.FAILED

    def __repr__(self) -> str:
        return f"NodeReport({self.name!r}, {self.status.value}, {self.duration:.3f}s)"


# ════════════════════════════════════════════════════════════════════════════
#  GraphReport — aggregate result
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class GraphReport(WorkReport):
    """Aggregate execution report for the entire DAG."""
    _status:        WorkStatus
    _work_context:  WorkContext
    node_reports:   dict[str, NodeReport] = field(default_factory=dict)
    total_duration: float = 0.0

    @property
    def status(self) -> WorkStatus: return self._status
    @property
    def work_context(self) -> WorkContext: return self._work_context

    @property
    def succeeded_nodes(self) -> list[str]:
        return [n for n, r in self.node_reports.items() if r.succeeded]
    @property
    def failed_nodes(self) -> list[str]:
        return [n for n, r in self.node_reports.items() if r.failed]

    def __repr__(self) -> str:
        return (
            f"GraphReport(status={self._status.value}, "
            f"nodes={len(self.node_reports)}, "
            f"failed={self.failed_nodes}, "
            f"duration={self.total_duration:.3f}s)"
        )


# ════════════════════════════════════════════════════════════════════════════
#  Internal node descriptor
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class _NodeDef:
    name:        str
    work:        Work
    depends_on:  list[str]


# ════════════════════════════════════════════════════════════════════════════
#  GraphFlow
# ════════════════════════════════════════════════════════════════════════════

class GraphFlow(Work):
    """
    A DAG workflow that executes nodes respecting declared dependencies
    and parallelises independent steps automatically.

    Do not instantiate directly — use :class:`GraphBuilder`.
    """

    def __init__(
        self,
        name:      str,
        nodes:     list[_NodeDef],
        executor:  Optional[concurrent.futures.Executor] = None,
        fail_fast: bool = True,
        timeout:   Optional[float] = None,
    ) -> None:
        self._name      = name
        self._nodes     = {n.name: n for n in nodes}
        self._executor  = executor
        self._fail_fast = fail_fast
        self._timeout   = timeout
        self._owns_exec = executor is None
        self._validate()

    def get_name(self) -> str:
        return self._name

    # ── Validation ────────────────────────────────────────────────────── #

    def _validate(self) -> None:
        """Check for unknown references and cycles at build time."""
        for node in self._nodes.values():
            for dep in node.depends_on:
                if dep not in self._nodes:
                    raise UnknownDependencyError(
                        f"Node {node.name!r} depends on {dep!r} which does not exist."
                    )
        order = self._topological_order()
        if order is None:
            raise CyclicDependencyError(
                f"Graph {self._name!r} contains a cycle — check dependency declarations."
            )

    def _topological_order(self) -> Optional[list[str]]:
        """Kahn's algorithm — returns None if a cycle is detected."""
        in_degree = {n: 0 for n in self._nodes}
        for node in self._nodes.values():
            for dep in node.depends_on:
                in_degree[node.name] += 1

        queue   = [n for n, d in in_degree.items() if d == 0]
        visited: list[str] = []

        while queue:
            name = queue.pop(0)
            visited.append(name)
            for other in self._nodes.values():
                if name in other.depends_on:
                    in_degree[other.name] -= 1
                    if in_degree[other.name] == 0:
                        queue.append(other.name)

        return visited if len(visited) == len(self._nodes) else None

    # ── Execution ─────────────────────────────────────────────────────── #

    def execute(self, work_context: WorkContext) -> GraphReport:
        logger.info("GraphFlow '%s': executing %d nodes", self._name, len(self._nodes))
        t_start = time.monotonic()

        executor = self._executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(self._nodes), 16) or 1
        )
        try:
            result = self._run(executor, work_context)
        finally:
            if self._owns_exec:
                executor.shutdown(wait=False)

        result.total_duration = time.monotonic() - t_start
        return result

    def _run(self, executor: concurrent.futures.Executor,
             work_context: WorkContext) -> GraphReport:

        node_reports: dict[str, NodeReport] = {}
        completed:    set[str]              = set()
        aborted:      bool                  = False
        completion_counter                  = 0
        lock                                = threading.Lock()

        # in-degree tracking
        in_degree  = {n: len(self._nodes[n].depends_on) for n in self._nodes}
        futures:   dict[concurrent.futures.Future, str] = {}

        def submit_ready() -> None:
            for name, deg in list(in_degree.items()):
                if deg == 0 and name not in completed and name not in [futures[f] for f in futures]:
                    fut = executor.submit(_execute_node, name)
                    futures[fut] = name

        def _execute_node(name: str) -> NodeReport:
            node = self._nodes[name]
            t0   = time.monotonic()
            try:
                report = node.work.execute(work_context)
                status = report.status
                error  = report.error
            except Exception as exc:
                status = WorkStatus.FAILED
                report = DefaultWorkReport(WorkStatus.FAILED, work_context, error=exc)
                error  = exc
                logger.error("GraphFlow: node %r raised: %s", name, exc)

            duration = time.monotonic() - t0
            with lock:
                nonlocal completion_counter
                completion_counter += 1
                nr = NodeReport(name=name, status=status, duration=duration,
                                completion_order=completion_counter,
                                work_report=report, error=error)
            logger.debug("Node %r → %s (%.3fs)", name, status.value, duration)
            return nr

        # Initial submission
        submit_ready()

        while futures:
            done, _ = concurrent.futures.wait(
                list(futures.keys()),
                timeout=self._timeout,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            if not done:
                aborted = True
                logger.error("GraphFlow '%s': timeout — cancelling remaining nodes", self._name)
                for f in futures: f.cancel()
                break

            for fut in done:
                name = futures.pop(fut)
                try:
                    nr = fut.result()
                except Exception as exc:
                    nr = NodeReport(name=name, status=WorkStatus.FAILED, error=exc)

                node_reports[name] = nr
                with lock:
                    completed.add(name)
                    del in_degree[name]

                if nr.failed:
                    if self._fail_fast:
                        logger.info("GraphFlow: node %r FAILED — fail_fast, stopping", name)
                        aborted = True
                        # Cancel pending futures
                        for f in list(futures): f.cancel()
                        futures.clear()
                        break
                else:
                    # Decrement dependents
                    with lock:
                        for other_name, other_node in self._nodes.items():
                            if name in other_node.depends_on and other_name in in_degree:
                                in_degree[other_name] -= 1

                    if not aborted: submit_ready()

            if aborted: break

        overall = (WorkStatus.FAILED
                   if any(r.failed for r in node_reports.values()) or aborted
                   else WorkStatus.COMPLETED)

        return GraphReport(
            _status=overall,
            _work_context=work_context,
            node_reports=node_reports,
        )

    # ── Introspection ─────────────────────────────────────────────────── #

    def to_dot(self) -> str:
        """Export the dependency graph as a Graphviz DOT string."""
        lines = [f'digraph "{self._name}" {{', '  rankdir=LR;']
        for node in self._nodes.values():
            for dep in node.depends_on:
                lines.append(f'  "{dep}" -> "{node.name}";')
        if not any(n.depends_on for n in self._nodes.values()):
            for n in self._nodes:
                lines.append(f'  "{n}";')
        lines.append("}")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
#  GraphBuilder
# ════════════════════════════════════════════════════════════════════════════

class GraphBuilder:
    """
    Fluent builder for :class:`GraphFlow`.

    Example::

        flow = (
            GraphBuilder()
            .named("data-pipeline")
            .add("extract-users",   extract_users_work)
            .add("extract-orders",  extract_orders_work)
            .add("join",            join_work,     depends_on=["extract-users", "extract-orders"])
            .add("aggregate",       agg_work,      depends_on=["join"])
            .add("export-csv",      csv_work,      depends_on=["aggregate"])
            .add("export-json",     json_work,     depends_on=["aggregate"])
            .fail_fast(True)
            .build()
        )
    """

    def __init__(self) -> None:
        self._name      = "graph-flow"
        self._nodes:    list[_NodeDef] = []
        self._executor: Optional[concurrent.futures.Executor] = None
        self._fail_fast = True
        self._timeout:  Optional[float] = None

    def named(self, name: str) -> "GraphBuilder":
        self._name = name; return self

    def add(self, name: str, work: Work,
            depends_on: Optional[list[str]] = None) -> "GraphBuilder":
        """
        Register a node.

        Args:
            name:       Unique node identifier.
            work:       The :class:`~maestro.flows.Work` to execute.
            depends_on: Names of nodes that must complete before this one starts.
        """
        self._nodes.append(_NodeDef(name=name, work=work,
                                     depends_on=depends_on or []))
        return self

    def with_executor(self, executor: concurrent.futures.Executor) -> "GraphBuilder":
        """Supply an external executor (caller is responsible for shutdown)."""
        self._executor = executor; return self

    def fail_fast(self, fail_fast: bool = True) -> "GraphBuilder":
        self._fail_fast = fail_fast; return self

    def timeout(self, seconds: float) -> "GraphBuilder":
        self._timeout = seconds; return self

    def build(self) -> GraphFlow:
        if not self._nodes:
            raise ValueError("GraphFlow requires at least one node.")
        return GraphFlow(
            name=self._name,
            nodes=list(self._nodes),
            executor=self._executor,
            fail_fast=self._fail_fast,
            timeout=self._timeout,
        )


__all__ = [
    "CyclicDependencyError", "UnknownDependencyError",
    "NodeReport", "GraphReport",
    "GraphFlow", "GraphBuilder",
]
