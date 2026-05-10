"""
RecordProcessor — transforms or enriches records mid-pipeline.
RecordMarshaller — serialises domain objects back to a writable format.

RecordProcessor
---------------
* :class:`CompositeRecordProcessor`   — chains multiple processors in sequence
* :class:`LambdaRecordProcessor`      — applies a callable to the payload
* :class:`FilteringRecordProcessor`   — raises :exc:`RecordProcessingException` to skip bad records

RecordMarshaller
----------------
* :class:`ToStringMarshaller`    — calls ``str()`` on the payload
* :class:`LambdaMarshaller`      — applies a callable to produce the output string
* :class:`JsonMarshaller`        — serialises to a JSON string
* :class:`CsvMarshaller`         — serialises a dict/dataclass to a CSV row string
"""
from __future__ import annotations

import csv
import io
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Generic, TypeVar

from maestro.batch._record import Record

P = TypeVar("P")
O = TypeVar("O")


# ─────────────────────────────────── Exceptions ──────────────────────────── #

class RecordProcessingException(Exception):
    """
    Raised by a processor to signal that the current record should be
    counted as *failed* and skipped.
    """


# ═══════════════════════════════════════════════════════════════════════════ #
#  RecordProcessor                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

class RecordProcessor(ABC, Generic[P, O]):
    """Abstract base for record processors."""

    @abstractmethod
    def process_record(self, record: Record[P]) -> Record[O]:
        """
        Process *record* and return the (possibly transformed) record.
        Raise :exc:`RecordProcessingException` to discard the record as failed.
        """


class LambdaRecordProcessor(RecordProcessor[P, O]):
    """
    Applies a callable to each record's payload.

    Example::

        .processor(LambdaRecordProcessor(lambda p: p.upper()))
    """

    def __init__(self, fn: Callable[[P], O]) -> None:
        self._fn = fn

    def process_record(self, record: Record[P]) -> Record[O]:
        record.payload = self._fn(record.payload)   # type: ignore[assignment]
        return record  # type: ignore[return-value]


class CompositeRecordProcessor(RecordProcessor[Any, Any]):
    """
    Chains multiple processors in sequence.

    Example::

        .processor(CompositeRecordProcessor([validator, enricher, transformer]))
    """

    def __init__(self, processors: list[RecordProcessor]) -> None:
        self._processors = processors

    def process_record(self, record: Record) -> Record:
        for processor in self._processors:
            record = processor.process_record(record)
        return record


class FilteringRecordProcessor(RecordProcessor[P, P]):
    """
    Raises :exc:`RecordProcessingException` when a predicate returns ``True``.

    Use this to reject invalid records without stopping the job::

        .processor(FilteringRecordProcessor(lambda p: p.get("age", 0) < 0,
                                             reason="negative age"))
    """

    def __init__(
        self,
        predicate: Callable[[P], bool],
        reason: str = "predicate matched",
    ) -> None:
        self._predicate = predicate
        self._reason = reason

    def process_record(self, record: Record[P]) -> Record[P]:
        if self._predicate(record.payload):
            raise RecordProcessingException(
                f"Record #{record.header.number} rejected: {self._reason}"
            )
        return record


# ═══════════════════════════════════════════════════════════════════════════ #
#  RecordMarshaller                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

class RecordMarshaller(ABC, Generic[P, O]):
    """Abstract base for record marshallers."""

    @abstractmethod
    def marshal_record(self, record: Record[P]) -> Record[O]:
        """Serialise *record*'s payload and return the transformed record."""


class ToStringMarshaller(RecordMarshaller[Any, str]):
    """Converts the payload to a string using ``str()``."""

    def marshal_record(self, record: Record[Any]) -> Record[str]:
        record.payload = str(record.payload)
        return record  # type: ignore[return-value]


class LambdaMarshaller(RecordMarshaller[P, O]):
    """Applies a callable to produce the marshalled output."""

    def __init__(self, fn: Callable[[P], O]) -> None:
        self._fn = fn

    def marshal_record(self, record: Record[P]) -> Record[O]:
        record.payload = self._fn(record.payload)   # type: ignore[assignment]
        return record  # type: ignore[return-value]


class JsonMarshaller(RecordMarshaller[Any, str]):
    """
    Serialises the payload to a JSON string.

    Works with dicts, lists, dataclasses, and anything supported by
    ``json.dumps``.  Dataclasses are converted via :func:`dataclasses.asdict`.

    Example::

        .marshaller(JsonMarshaller(indent=2))
    """

    def __init__(self, indent: int | None = None, ensure_ascii: bool = False) -> None:
        self._indent = indent
        self._ensure_ascii = ensure_ascii

    def marshal_record(self, record: Record[Any]) -> Record[str]:
        payload = record.payload
        if is_dataclass(payload) and not isinstance(payload, type):
            payload = asdict(payload)
        record.payload = json.dumps(payload, indent=self._indent, ensure_ascii=self._ensure_ascii)
        return record  # type: ignore[return-value]


class CsvMarshaller(RecordMarshaller[Any, str]):
    """
    Serialises a ``dict`` or dataclass payload to a CSV row string.

    Example::

        .marshaller(CsvMarshaller(field_names=["id","user","message"]))
    """

    def __init__(
        self,
        field_names: list[str] | None = None,
        delimiter: str = ",",
    ) -> None:
        self._field_names = field_names
        self._delimiter = delimiter

    def marshal_record(self, record: Record[Any]) -> Record[str]:
        payload = record.payload
        if is_dataclass(payload) and not isinstance(payload, type):
            payload = asdict(payload)

        buf = io.StringIO()
        writer = csv.writer(buf, delimiter=self._delimiter)

        if isinstance(payload, dict):
            names = self._field_names or list(payload.keys())
            writer.writerow([payload.get(n, "") for n in names])
        else:
            writer.writerow(list(payload))

        record.payload = buf.getvalue().rstrip("\r\n")
        return record  # type: ignore[return-value]
