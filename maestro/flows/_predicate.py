"""
WorkReportPredicate — boolean conditions evaluated on a :class:`WorkReport`.

Used by :class:`~easy_flows.workflow.ConditionalFlow` to decide which
branch to take, and by :class:`~easy_flows.workflow.RepeatFlow` to decide
when to stop repeating.

Built-in predicates
-------------------
* :data:`COMPLETED`        — report status is COMPLETED
* :data:`FAILED`           — report status is FAILED
* :data:`ALWAYS_TRUE`      — always True (repeat forever until `times` limit)
* :data:`ALWAYS_FALSE`     — always False (never repeat)

Combining predicates
--------------------
Predicates support ``&`` (AND), ``|`` (OR), and ``~`` (NOT)::

    pred = WorkReportPredicate.COMPLETED & ~WorkReportPredicate.FAILED
"""
from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .work import WorkReport


class WorkReportPredicate:
    """
    A callable predicate over a :class:`WorkReport`.

    Instantiate with any callable that takes a ``WorkReport`` and returns bool::

        my_pred = WorkReportPredicate(lambda r: r.status == WorkStatus.COMPLETED)

    Or use the pre-built class attributes: :data:`COMPLETED`, :data:`FAILED`,
    :data:`ALWAYS_TRUE`, :data:`ALWAYS_FALSE`.
    """

    def __init__(
        self,
        fn: "Callable[[WorkReport], bool]",
        name: str = "",
    ) -> None:
        self._fn = fn
        self._name = name or getattr(fn, "__name__", "predicate")

    def test(self, report: "WorkReport") -> bool:
        """Evaluate the predicate against *report*."""
        return bool(self._fn(report))

    # Allow direct call syntax: predicate(report)
    def __call__(self, report: "WorkReport") -> bool:
        return self.test(report)

    # Composition operators
    def __and__(self, other: "WorkReportPredicate") -> "WorkReportPredicate":
        return WorkReportPredicate(
            lambda r: self.test(r) and other.test(r),
            name=f"({self._name} AND {other._name})",
        )

    def __or__(self, other: "WorkReportPredicate") -> "WorkReportPredicate":
        return WorkReportPredicate(
            lambda r: self.test(r) or other.test(r),
            name=f"({self._name} OR {other._name})",
        )

    def __invert__(self) -> "WorkReportPredicate":
        return WorkReportPredicate(
            lambda r: not self.test(r),
            name=f"NOT {self._name}",
        )

    def __repr__(self) -> str:
        return f"WorkReportPredicate({self._name!r})"


# ─────────────────────── Built-in predicates ────────────────────────────── #

from maestro.flows._work import WorkStatus  # noqa: E402 (avoids circular import at module level)

WorkReportPredicate.COMPLETED = WorkReportPredicate(
    lambda r: r.status == WorkStatus.COMPLETED,
    name="COMPLETED",
)

WorkReportPredicate.FAILED = WorkReportPredicate(
    lambda r: r.status == WorkStatus.FAILED,
    name="FAILED",
)

WorkReportPredicate.ALWAYS_TRUE = WorkReportPredicate(
    lambda r: True,
    name="ALWAYS_TRUE",
)

WorkReportPredicate.ALWAYS_FALSE = WorkReportPredicate(
    lambda r: False,
    name="ALWAYS_FALSE",
)
