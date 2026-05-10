"""
Record — the fundamental data unit in easy-batch.

Every piece of data flowing through the pipeline is wrapped in a
:class:`Record`.  A record carries both a :class:`Header` (metadata) and a
payload (the actual data).

Key types
---------
* :class:`Header`  — record number, source name, creation timestamp
* :class:`Record`  — header + typed payload
* :class:`Batch`   — ordered list of records flushed to the writer together
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Generic, Iterable, Iterator, TypeVar

P = TypeVar("P")   # payload type


# ═══════════════════════════════════════════════════════════════════════════ #
#  Header                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass(frozen=True)
class Header:
    """
    Immutable metadata attached to every record.

    Attributes:
        number:       1-based ordinal position in the data source.
        source:       Human-readable name of the data source (e.g. file path).
        created_at:   UTC timestamp when the record was read.
    """
    number: int
    source: str = "unknown"
    created_at: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    def __str__(self) -> str:
        return f"Header(number={self.number}, source={self.source!r})"


# ═══════════════════════════════════════════════════════════════════════════ #
#  Record                                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

class Record(Generic[P]):
    """
    A single unit of data flowing through the processing pipeline.

    Example::

        header  = Header(number=1, source="tweets.csv")
        record  = Record(header, payload="1,alice,hello world")
    """

    __slots__ = ("_header", "_payload")

    def __init__(self, header: Header, payload: P) -> None:
        self._header: Header = header
        self._payload: P = payload

    @property
    def header(self) -> Header:
        return self._header

    @property
    def payload(self) -> P:
        return self._payload

    @payload.setter
    def payload(self, value: P) -> None:
        self._payload = value

    def __repr__(self) -> str:
        return f"Record(header={self._header}, payload={self._payload!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Record):
            return NotImplemented
        return self._header == other._header and self._payload == other._payload

    def __hash__(self) -> int:
        return hash(self._header.number)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Batch                                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

class Batch(Generic[P]):
    """
    An ordered collection of records accumulated before being flushed to a
    :class:`~easy_batch.core.writer.RecordWriter`.

    Example::

        batch: Batch[str] = Batch()
        batch.add(record)
        batch.add(record2)
        print(len(batch))   # 2
    """

    def __init__(self) -> None:
        self._records: list[Record[P]] = []

    def add(self, record: Record[P]) -> None:
        self._records.append(record)

    def clear(self) -> None:
        self._records.clear()

    def is_empty(self) -> bool:
        return len(self._records) == 0

    @property
    def records(self) -> list[Record[P]]:
        return list(self._records)

    def payloads(self) -> list[P]:
        """Convenience: return the payload of every record in the batch."""
        return [r.payload for r in self._records]

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[Record[P]]:
        return iter(self._records)

    def __repr__(self) -> str:
        return f"Batch(size={len(self._records)})"
