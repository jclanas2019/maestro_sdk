"""
Record readers — sources of :class:`Record` objects.

Built-in readers
----------------
* :class:`IterableRecordReader`   — wraps any Python iterable
* :class:`FlatFileRecordReader`   — reads lines from a text file
* :class:`StringRecordReader`     — reads lines from a multi-line string
* :class:`CsvDictRecordReader`    — reads a CSV file and yields ``dict`` payloads
* :class:`JsonLinesRecordReader`  — reads a JSON-Lines file (one JSON object per line)
* :class:`InMemoryRecordReader`   — wraps an already-materialised list (useful in tests)
"""
from __future__ import annotations

import csv
import io
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Iterable, Iterator, Optional, TypeVar, Union

from maestro.batch._record import Header, Record

P = TypeVar("P")


# ═══════════════════════════════════════════════════════════════════════════ #
#  Abstract base                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

class RecordReader(ABC, Generic[P]):
    """
    Abstract base for all record readers.

    Implement :meth:`open`, :meth:`read_record`, and :meth:`close`
    (or use the context-manager protocol via :meth:`__enter__` / :meth:`__exit__`).

    Alternatively subclass and implement ``__iter__`` if you prefer a generator style.
    """

    @abstractmethod
    def open(self) -> None:
        """Open the underlying data source."""

    @abstractmethod
    def read_record(self) -> Optional[Record[P]]:
        """
        Return the next record or ``None`` when exhausted.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the reader."""

    # Convenience: allow use as an iterator
    def __iter__(self) -> Iterator[Record[P]]:
        self.open()
        try:
            while True:
                record = self.read_record()
                if record is None:
                    break
                yield record
        finally:
            self.close()

    def __enter__(self) -> "RecordReader[P]":
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @property
    def source_name(self) -> str:
        return type(self).__name__


# ═══════════════════════════════════════════════════════════════════════════ #
#  IterableRecordReader                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

class IterableRecordReader(RecordReader[P]):
    """
    Wraps any Python iterable as a record source.

    Example::

        reader = IterableRecordReader([1, 2, 3], source="numbers")
    """

    def __init__(self, data: Iterable[P], source: str = "iterable") -> None:
        self._data = data
        self._source = source
        self._iter: Optional[Iterator[P]] = None
        self._counter = 0

    @property
    def source_name(self) -> str:
        return self._source

    def open(self) -> None:
        self._iter = iter(self._data)
        self._counter = 0

    def read_record(self) -> Optional[Record[P]]:
        assert self._iter is not None, "Call open() before read_record()"
        try:
            payload = next(self._iter)
            self._counter += 1
            return Record(Header(number=self._counter, source=self._source), payload)
        except StopIteration:
            return None

    def close(self) -> None:
        self._iter = None


# alias for convenience
InMemoryRecordReader = IterableRecordReader


# ═══════════════════════════════════════════════════════════════════════════ #
#  FlatFileRecordReader                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

class FlatFileRecordReader(RecordReader[str]):
    """
    Reads lines from a text file.  Each non-empty line becomes a record payload.

    Example::

        reader = FlatFileRecordReader("tweets.csv")
        for record in reader:
            print(record.payload)
    """

    def __init__(
        self,
        path: Union[str, Path],
        encoding: str = "utf-8",
        skip_empty_lines: bool = True,
    ) -> None:
        self._path = Path(path)
        self._encoding = encoding
        self._skip_empty_lines = skip_empty_lines
        self._file = None
        self._counter = 0

    @property
    def source_name(self) -> str:
        return str(self._path)

    def open(self) -> None:
        self._file = self._path.open("r", encoding=self._encoding)
        self._counter = 0

    def read_record(self) -> Optional[Record[str]]:
        assert self._file is not None, "Call open() before read_record()"
        while True:
            line = self._file.readline()
            if not line:               # EOF
                return None
            stripped = line.rstrip("\n\r")
            if self._skip_empty_lines and not stripped.strip():
                continue
            self._counter += 1
            return Record(
                Header(number=self._counter, source=self.source_name),
                stripped,
            )

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


# ═══════════════════════════════════════════════════════════════════════════ #
#  StringRecordReader                                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

class StringRecordReader(FlatFileRecordReader):
    """
    Reads records from a multi-line string (handy for tests / inline data).

    Example::

        reader = StringRecordReader("id,user\\n1,alice\\n2,bob")
    """

    def __init__(self, text: str, source: str = "string") -> None:
        self._text = text
        self._encoding = "utf-8"
        self._skip_empty_lines = True
        self._file = None
        self._counter = 0
        self._source_name = source

    @property
    def source_name(self) -> str:
        return self._source_name

    def open(self) -> None:
        self._file = io.StringIO(self._text)
        self._counter = 0


# ═══════════════════════════════════════════════════════════════════════════ #
#  CsvDictRecordReader                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

class CsvDictRecordReader(RecordReader[dict]):
    """
    Reads a CSV file using :mod:`csv.DictReader`.
    Each row becomes a ``dict`` payload.

    Example::

        reader = CsvDictRecordReader("tweets.csv")
        for record in reader:
            print(record.payload["user"])
    """

    def __init__(
        self,
        path: Union[str, Path, io.StringIO],
        encoding: str = "utf-8",
        delimiter: str = ",",
        fieldnames: Optional[list[str]] = None,
    ) -> None:
        if isinstance(path, io.StringIO):
            self._path = None
            self._stream: Optional[io.StringIO] = path
        else:
            self._path = Path(path)
            self._stream = None
        self._encoding = encoding
        self._delimiter = delimiter
        self._fieldnames = fieldnames
        self._file = None
        self._reader: Optional[csv.DictReader] = None
        self._counter = 0

    @property
    def source_name(self) -> str:
        return str(self._path) if self._path else "csv-string"

    def open(self) -> None:
        if self._path:
            self._file = self._path.open("r", encoding=self._encoding, newline="")
            stream = self._file
        else:
            self._stream.seek(0)
            stream = self._stream
        self._reader = csv.DictReader(
            stream, fieldnames=self._fieldnames, delimiter=self._delimiter
        )
        self._counter = 0

    def read_record(self) -> Optional[Record[dict]]:
        assert self._reader is not None
        try:
            row = next(self._reader)
            self._counter += 1
            return Record(
                Header(number=self._counter, source=self.source_name),
                dict(row),
            )
        except StopIteration:
            return None

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


# ═══════════════════════════════════════════════════════════════════════════ #
#  JsonLinesRecordReader                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

class JsonLinesRecordReader(RecordReader[dict]):
    """
    Reads a JSON-Lines file (one JSON object per line).

    Example::

        reader = JsonLinesRecordReader("events.jsonl")
    """

    def __init__(self, path: Union[str, Path], encoding: str = "utf-8") -> None:
        self._path = Path(path)
        self._encoding = encoding
        self._file = None
        self._counter = 0

    @property
    def source_name(self) -> str:
        return str(self._path)

    def open(self) -> None:
        self._file = self._path.open("r", encoding=self._encoding)
        self._counter = 0

    def read_record(self) -> Optional[Record[dict]]:
        assert self._file is not None
        while True:
            line = self._file.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            self._counter += 1
            return Record(
                Header(number=self._counter, source=self.source_name),
                json.loads(line),
            )

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
