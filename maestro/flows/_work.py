"""
Core work abstractions for easy-flows.

Key types
---------
* :class:`WorkStatus`       — COMPLETED | FAILED
* :class:`WorkContext`      — shared mutable key/value store passed to every work unit
* :class:`WorkReport`       — result returned by every work execution
* :class:`DefaultWorkReport`— standard implementation of WorkReport
* :class:`Work`             — abstract unit of work (implement ``execute``)
* :class:`NoOpWork`         — a work that always completes and does nothing
* :class:`LambdaWork`       — wraps a plain callable as a Work
"""
from __future__ import annotations

import enum
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


# ═══════════════════════════════════════════════════════════════════════════ #
#  WorkStatus                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

class WorkStatus(enum.Enum):
    """Result status of a work execution."""
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"


# ═══════════════════════════════════════════════════════════════════════════ #
#  WorkContext                                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

class WorkContext:
    """
    A mutable key/value store shared across all work units in a workflow.

    Think of it as the "blackboard" that every Work can read from and write to.

    Example::

        ctx = WorkContext()
        ctx.put("order_id", 42)
        order_id = ctx.get("order_id")
    """

    def __init__(self, **kwargs: Any) -> None:
        self._data: dict[str, Any] = dict(kwargs)

    def put(self, key: str, value: Any) -> "WorkContext":
        """Store *value* under *key*. Returns self for chaining."""
        self._data[key] = value
        return self

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if absent."""
        return self._data.get(key, default)

    def remove(self, key: str) -> None:
        """Remove *key* (no-op if absent)."""
        self._data.pop(key, None)

    def copy(self) -> "WorkContext":
        """
        Return a new ``WorkContext`` with a shallow copy of all current entries.

        Used by ``ParallelFlow`` and ``GraphFlow`` to give each concurrent
        work unit an isolated snapshot so they cannot race on shared state.
        """
        return WorkContext(**self._data)

    def merge(self, other: "WorkContext") -> "WorkContext":
        """
        Merge all entries from *other* into this context.

        *other*'s values overwrite any existing entries with the same key.
        Returns ``self`` for method-chaining.

        A snapshot of *other* is taken first so concurrent modifications
        to *other* cannot cause a ``RuntimeError`` during iteration.
        """
        snapshot = dict(other._data)   # atomic snapshot — safe under CPython GIL
        self._data.update(snapshot)
        return self

    def contains(self, key: str) -> bool:
        return key in self._data

    def as_map(self) -> dict[str, Any]:
        return dict(self._data)

    # Python niceties
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"WorkContext({self._data!r})"


# ═══════════════════════════════════════════════════════════════════════════ #
#  WorkReport                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

class WorkReport:
    """
    Abstract result of a work execution.  Subclass or use :class:`DefaultWorkReport`.
    """

    @property
    def status(self) -> WorkStatus:
        raise NotImplementedError

    @property
    def work_context(self) -> WorkContext:
        raise NotImplementedError

    @property
    def error(self) -> Optional[Exception]:
        return None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"status={self.status.value}, "
            f"error={self.error!r})"
        )


class DefaultWorkReport(WorkReport):
    """
    Standard :class:`WorkReport` implementation.

    Example::

        return DefaultWorkReport(WorkStatus.COMPLETED, work_context)
        return DefaultWorkReport(WorkStatus.FAILED, work_context, error=exc)
    """

    def __init__(
        self,
        status: WorkStatus,
        work_context: WorkContext,
        error: Optional[Exception] = None,
    ) -> None:
        self._status = status
        self._work_context = work_context
        self._error = error

    @property
    def status(self) -> WorkStatus:
        return self._status

    @property
    def work_context(self) -> WorkContext:
        return self._work_context

    @property
    def error(self) -> Optional[Exception]:
        return self._error


# ═══════════════════════════════════════════════════════════════════════════ #
#  Work                                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

class Work(ABC):
    """
    Abstract unit of work.

    Implement :meth:`execute` to define the work logic.
    Every :class:`~easy_flows.workflow.WorkFlow` is also a ``Work``, so
    workflows can be nested freely.

    Example::

        class SendEmailWork(Work):
            def get_name(self) -> str:
                return "send-email"

            def execute(self, work_context: WorkContext) -> WorkReport:
                # ... send email ...
                return DefaultWorkReport(WorkStatus.COMPLETED, work_context)
    """

    def get_name(self) -> str:
        """Human-readable name for this work unit (default: class name)."""
        return type(self).__name__

    @abstractmethod
    def execute(self, work_context: WorkContext) -> WorkReport:
        """
        Execute this work unit.

        Args:
            work_context: shared context — read/write freely.

        Returns:
            A :class:`WorkReport` whose status is either COMPLETED or FAILED.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.get_name()!r})"


# ═══════════════════════════════════════════════════════════════════════════ #
#  Built-in Work implementations                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

class NoOpWork(Work):
    """
    A work unit that always completes and does nothing.
    Useful as a placeholder.
    """

    def __init__(self, name: str = "no-op") -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name

    def execute(self, work_context: WorkContext) -> WorkReport:
        return DefaultWorkReport(WorkStatus.COMPLETED, work_context)


class LambdaWork(Work):
    """
    Wraps a plain callable as a :class:`Work`.

    The callable receives the :class:`WorkContext` and must return either:
    * A :class:`WorkReport`, or
    * ``None`` / any non-exception value (treated as COMPLETED).

    Example::

        work = LambdaWork(lambda ctx: print("hello"), name="print-hello")
    """

    def __init__(
        self,
        fn: Callable[[WorkContext], Any],
        name: str = "",
        fail_on_exception: bool = True,
    ) -> None:
        self._fn = fn
        self._name = name or getattr(fn, "__name__", "lambda-work")
        self._fail_on_exception = fail_on_exception

    def get_name(self) -> str:
        return self._name

    def execute(self, work_context: WorkContext) -> WorkReport:
        try:
            result = self._fn(work_context)
            if isinstance(result, WorkReport):
                return result
            return DefaultWorkReport(WorkStatus.COMPLETED, work_context)
        except Exception as exc:
            if self._fail_on_exception:
                return DefaultWorkReport(WorkStatus.FAILED, work_context, error=exc)
            raise
