"""
Record mappers — transform the raw payload of a record into a domain object.

Built-in mappers
----------------
* :class:`DelimitedRecordMapper`  — splits a delimited string and maps to a dataclass / dict
* :class:`FieldSetMapper`         — maps a ``dict`` payload to a dataclass / object
* :class:`PassThroughRecordMapper`— leaves the payload unchanged (identity)
* :class:`LambdaRecordMapper`     — applies an arbitrary callable to the payload
"""
from __future__ import annotations

import csv
import io
from abc import ABC, abstractmethod
from dataclasses import fields as dataclass_fields, is_dataclass
from typing import Any, Callable, Generic, Optional, Type, TypeVar

from maestro.batch._record import Record

P = TypeVar("P")   # input payload
O = TypeVar("O")   # output (mapped) type


class RecordMapper(ABC, Generic[P, O]):
    """Abstract base for all record mappers."""

    @abstractmethod
    def map_record(self, record: Record[P]) -> Record[O]:
        """
        Transform *record* — typically by replacing its payload with a
        domain object — and return the transformed record.
        """


# ═══════════════════════════════════════════════════════════════════════════ #
#  PassThroughRecordMapper                                                    #
# ═══════════════════════════════════════════════════════════════════════════ #

class PassThroughRecordMapper(RecordMapper[P, P]):
    """Returns the record unchanged (useful when no mapping is needed)."""

    def map_record(self, record: Record[P]) -> Record[P]:
        return record


# ═══════════════════════════════════════════════════════════════════════════ #
#  LambdaRecordMapper                                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

class LambdaRecordMapper(RecordMapper[P, O]):
    """
    Applies an arbitrary callable to the record's payload.

    Example::

        .mapper(LambdaRecordMapper(lambda p: p.upper()))
    """

    def __init__(self, fn: Callable[[P], O]) -> None:
        self._fn = fn

    def map_record(self, record: Record[P]) -> Record[O]:
        record.payload = self._fn(record.payload)   # type: ignore[assignment]
        return record  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════ #
#  DelimitedRecordMapper                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

class DelimitedRecordMapper(RecordMapper[str, Any]):
    """
    Splits a delimited string (CSV) and maps the tokens to either:

    * A **dataclass** (fields matched by name or by positional order), or
    * A plain **dict** (when no target class is provided).

    Example — map to a dict::

        .mapper(DelimitedRecordMapper(delimiter=",", field_names=["id","user","msg"]))

    Example — map to a dataclass::

        @dataclass
        class Tweet:
            id: int
            user: str
            message: str

        .mapper(DelimitedRecordMapper(Tweet, field_names=["id","user","message"]))
    """

    def __init__(
        self,
        target_class: Optional[Type] = None,
        field_names: Optional[list[str]] = None,
        delimiter: str = ",",
        type_converters: Optional[dict[str, Callable[[str], Any]]] = None,
    ) -> None:
        self._target_class = target_class
        self._field_names = field_names
        self._delimiter = delimiter
        self._type_converters = type_converters or {}

        # Auto-discover field names from dataclass if not provided
        if self._target_class and not self._field_names and is_dataclass(self._target_class):
            self._field_names = [f.name for f in dataclass_fields(self._target_class)]

    def map_record(self, record: Record[str]) -> Record[Any]:
        row = next(csv.reader([record.payload], delimiter=self._delimiter))
        names = self._field_names or [str(i) for i in range(len(row))]

        if len(names) != len(row):
            raise ValueError(
                f"Record #{record.header.number}: expected {len(names)} fields "
                f"but got {len(row)} — payload={record.payload!r}"
            )

        kwargs: dict[str, Any] = {}
        for name, raw_value in zip(names, row):
            converter = self._type_converters.get(name)
            kwargs[name] = converter(raw_value) if converter else raw_value

        if self._target_class:
            # Auto-convert types based on dataclass annotations
            if is_dataclass(self._target_class):
                for f in dataclass_fields(self._target_class):
                    if f.name in kwargs and f.name not in self._type_converters:
                        try:
                            origin = getattr(f.type, "__origin__", None)
                            if f.type in (int, float, bool) or (
                                isinstance(f.type, type) and issubclass(f.type, (int, float, bool))
                            ):
                                kwargs[f.name] = f.type(kwargs[f.name])
                        except (TypeError, ValueError):
                            pass
            record.payload = self._target_class(**kwargs)
        else:
            record.payload = kwargs

        return record  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════ #
#  FieldSetMapper (dict → object)                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

class FieldSetMapper(RecordMapper[dict, Any]):
    """
    Maps a ``dict`` payload to an instance of *target_class*.

    Works with dataclasses or any class whose ``__init__`` accepts keyword
    arguments matching the dict keys.

    Example::

        .mapper(FieldSetMapper(Tweet))
    """

    def __init__(
        self,
        target_class: Type,
        field_mapping: Optional[dict[str, str]] = None,
        type_converters: Optional[dict[str, Callable[[str], Any]]] = None,
    ) -> None:
        self._target_class = target_class
        self._field_mapping = field_mapping or {}   # dict-key → class-attr
        self._type_converters = type_converters or {}

    def map_record(self, record: Record[dict]) -> Record[Any]:
        raw: dict = record.payload
        kwargs: dict[str, Any] = {}
        for k, v in raw.items():
            attr = self._field_mapping.get(k, k)
            converter = self._type_converters.get(attr)
            kwargs[attr] = converter(v) if converter else v

        record.payload = self._target_class(**kwargs)
        return record  # type: ignore[return-value]
