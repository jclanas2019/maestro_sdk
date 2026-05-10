"""
Record writers — flush batches of records to a data sink.

Built-in writers
----------------
* :class:`StandardOutputRecordWriter`  — prints payloads to stdout
* :class:`FileRecordWriter`            — appends payloads to a text file
* :class:`CollectionRecordWriter`      — appends payloads to a Python list
* :class:`StringIORecordWriter`        — writes to an in-memory ``io.StringIO``
* :class:`DevNullRecordWriter`         — discards all records (useful for dry-runs)
"""
from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar, Union

from maestro.batch._record import Batch

P = TypeVar("P")


class RecordWriter(ABC, Generic[P]):
    """Abstract base for all record writers."""

    def open(self) -> None:
        """Called once before the job starts writing."""

    @abstractmethod
    def write_records(self, batch: Batch[P]) -> None:
        """Write all records in *batch* to the data sink."""

    def close(self) -> None:
        """Called once after the job finishes (flush / release resources)."""

    def __enter__(self) -> "RecordWriter[P]":
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ═══════════════════════════════════════════════════════════════════════════ #
#  StandardOutputRecordWriter                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

class StandardOutputRecordWriter(RecordWriter[Any]):
    """Prints each record's payload to stdout."""

    def write_records(self, batch: Batch[Any]) -> None:
        for record in batch:
            print(record.payload)


# ═══════════════════════════════════════════════════════════════════════════ #
#  FileRecordWriter                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

class FileRecordWriter(RecordWriter[str]):
    """
    Appends string payloads to a text file, one line per record.

    Example::

        .writer(FileRecordWriter("output.txt"))
    """

    def __init__(
        self,
        path: Union[str, Path],
        encoding: str = "utf-8",
        newline: str = "\n",
        mode: str = "w",
    ) -> None:
        self._path = Path(path)
        self._encoding = encoding
        self._newline = newline
        self._mode = mode
        self._file = None

    def open(self) -> None:
        self._file = self._path.open(self._mode, encoding=self._encoding)

    def write_records(self, batch: Batch[str]) -> None:
        assert self._file is not None
        for record in batch:
            self._file.write(str(record.payload) + self._newline)

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


# ═══════════════════════════════════════════════════════════════════════════ #
#  CollectionRecordWriter                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

class CollectionRecordWriter(RecordWriter[P]):
    """
    Appends record payloads to a Python list.  Useful for testing and
    in-memory pipelines.

    Example::

        sink: list = []
        .writer(CollectionRecordWriter(sink))
        # after job.execute() → sink contains all processed payloads
    """

    def __init__(self, collection: list | None = None) -> None:
        self._collection: list = collection if collection is not None else []

    @property
    def collection(self) -> list:
        return self._collection

    def write_records(self, batch: Batch[P]) -> None:
        for record in batch:
            self._collection.append(record.payload)


# ═══════════════════════════════════════════════════════════════════════════ #
#  StringIORecordWriter                                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

class StringIORecordWriter(RecordWriter[str]):
    """Writes string payloads to an in-memory :class:`io.StringIO`."""

    def __init__(self) -> None:
        self._buffer = io.StringIO()

    @property
    def buffer(self) -> io.StringIO:
        return self._buffer

    def getvalue(self) -> str:
        return self._buffer.getvalue()

    def write_records(self, batch: Batch[str]) -> None:
        for record in batch:
            self._buffer.write(str(record.payload) + "\n")


# ═══════════════════════════════════════════════════════════════════════════ #
#  DevNullRecordWriter                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

class DevNullRecordWriter(RecordWriter[Any]):
    """Silently discards all records.  Useful for dry-runs."""

    def write_records(self, batch: Batch[Any]) -> None:
        pass
