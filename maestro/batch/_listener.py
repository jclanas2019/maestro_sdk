"""
Listener interfaces for observing the easy-batch job lifecycle.

* :class:`JobListener`          — fired before/after the entire job
* :class:`BatchListener`        — fired before/after each batch is written
* :class:`RecordReaderListener` — fired before/after each record is read
* :class:`PipelineListener`     — fired before/after each record traverses the pipeline
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maestro.batch._record import Batch, Record
    from .job import JobParameters, JobReport


class JobListener:
    """Override the methods you need."""

    def before_job_start(self, parameters: "JobParameters") -> None:
        """Called before the job starts."""

    def after_job_end(self, report: "JobReport") -> None:
        """Called after the job completes (success or failure)."""


class BatchListener:
    """Override the methods you need."""

    def before_batch_writing(self, batch: "Batch") -> None:
        """Called before a batch is handed to the writer."""

    def after_batch_writing(self, batch: "Batch") -> None:
        """Called after the writer has processed a batch."""

    def on_batch_writing_exception(self, batch: "Batch", exc: Exception) -> None:
        """Called when the writer raises an exception."""


class RecordReaderListener:
    """Override the methods you need."""

    def before_record_reading(self) -> None:
        """Called before each ``read_record()`` call."""

    def after_record_reading(self, record: "Record") -> None:
        """Called after a record has been successfully read."""

    def on_record_reading_exception(self, exc: Exception) -> None:
        """Called when ``read_record()`` raises an exception."""


class PipelineListener:
    """Override the methods you need."""

    def before_record_processing(self, record: "Record") -> None:
        """Called before a record enters the pipeline (filter → map → process → marshal)."""

    def after_record_processing(self, record: "Record") -> None:
        """Called after a record has successfully traversed the pipeline."""

    def on_record_processing_exception(self, record: "Record", exc: Exception) -> None:
        """Called when an exception occurs during pipeline processing."""
