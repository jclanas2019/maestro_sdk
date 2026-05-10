"""
Record filters — decide whether a record should continue through the pipeline.

A filter returns ``True``  to **keep** the record (process it normally) or
``False`` to **discard** it (the record is counted as "filtered" in the job report).

Built-in filters
----------------
* :class:`HeaderRecordFilter`          — discards record number 1 (the CSV header line)
* :class:`PredicateRecordFilter`       — discards records that satisfy a predicate
* :class:`PoisonRecordFilter`          — discards a configurable "poison" sentinel payload
* :class:`RecordNumberRangeFilter`     — keeps only records within a number range
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Optional, TypeVar

from maestro.batch._record import Record

P = TypeVar("P")


class RecordFilter(ABC, Generic[P]):
    """Abstract base for all record filters."""

    @abstractmethod
    def accept(self, record: Record[P]) -> bool:
        """
        Return ``True`` to let *record* continue, ``False`` to discard it.
        """


# ═══════════════════════════════════════════════════════════════════════════ #
#  HeaderRecordFilter                                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

class HeaderRecordFilter(RecordFilter[Any]):
    """
    Discards the very first record (record number 1).

    Typically used to skip a CSV header line::

        .filter(HeaderRecordFilter())
    """

    def accept(self, record: Record) -> bool:
        return record.header.number != 1


# ═══════════════════════════════════════════════════════════════════════════ #
#  PredicateRecordFilter                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

class PredicateRecordFilter(RecordFilter[P]):
    """
    Discards records whose payload satisfies a *predicate*.

    ``accept`` returns ``True`` when ``predicate(record)`` is **False**.

    Example — skip blank CSV lines::

        .filter(PredicateRecordFilter(lambda r: r.payload.strip() == ""))
    """

    def __init__(self, predicate: Callable[[Record[P]], bool]) -> None:
        self._predicate = predicate

    def accept(self, record: Record[P]) -> bool:
        return not self._predicate(record)


# ═══════════════════════════════════════════════════════════════════════════ #
#  PoisonRecordFilter                                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

class PoisonRecordFilter(RecordFilter[Any]):
    """
    Discards records whose payload equals a configured *sentinel* value.

    Useful for pipeline termination patterns::

        .filter(PoisonRecordFilter(sentinel="__STOP__"))
    """

    def __init__(self, sentinel: Any) -> None:
        self._sentinel = sentinel

    def accept(self, record: Record) -> bool:
        return record.payload != self._sentinel


# ═══════════════════════════════════════════════════════════════════════════ #
#  RecordNumberRangeFilter                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

class RecordNumberRangeFilter(RecordFilter[Any]):
    """
    Keeps only records whose number falls in [start, end] (inclusive).
    Records outside the range are discarded.

    Example — keep records 10 to 50::

        .filter(RecordNumberRangeFilter(start=10, end=50))
    """

    def __init__(self, start: int = 1, end: int = 2 ** 31 - 1) -> None:
        self._start = start
        self._end = end

    def accept(self, record: Record) -> bool:
        n = record.header.number
        return self._start <= n <= self._end
