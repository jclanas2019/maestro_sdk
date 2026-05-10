"""
Core job abstractions for easy-batch.

Key types
---------
* :class:`JobParameters`  — configuration (batch size, error threshold, …)
* :class:`JobStatus`      — enum: STARTING / RUNNING / COMPLETED / FAILED / ABORTED
* :class:`JobMetrics`     — counters collected during execution
* :class:`JobReport`      — full execution summary returned after a job finishes
* :class:`Job`            — the processing pipeline (created via :class:`JobBuilder`)
* :class:`JobBuilder`     — fluent builder for :class:`Job`
"""
from __future__ import annotations

import datetime
import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generic, List, Optional, TypeVar

from maestro.batch._filter import RecordFilter
from maestro.batch._listener import (
    BatchListener,
    JobListener,
    PipelineListener,
    RecordReaderListener,
)
from maestro.batch._mapper import PassThroughRecordMapper, RecordMapper
from maestro.batch._processor import RecordProcessingException, RecordProcessor
from maestro.batch._processor import RecordMarshaller, ToStringMarshaller
from maestro.batch._reader import RecordReader
from maestro.batch._record import Batch, Record
from maestro.batch._writer import RecordWriter, DevNullRecordWriter

logger = logging.getLogger(__name__)

P = TypeVar("P")
O = TypeVar("O")


# ═══════════════════════════════════════════════════════════════════════════ #
#  JobParameters                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class JobParameters:
    """
    Configuration for a batch job.

    Attributes:
        name:            Human-readable job name.
        batch_size:      Number of records accumulated before flushing to the writer.
        error_threshold: Maximum number of failed records before the job is aborted.
                         Use -1 (default) for unlimited.
        jmx_enabled:     Reserved — no-op in the Python port.
    """
    name: str = "easy-batch-job"
    batch_size: int = 100
    error_threshold: int = -1   # -1 = unlimited


# ═══════════════════════════════════════════════════════════════════════════ #
#  JobStatus                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

class JobStatus(enum.Enum):
    STARTING   = "STARTING"
    RUNNING    = "RUNNING"
    COMPLETED  = "COMPLETED"
    FAILED     = "FAILED"
    ABORTED    = "ABORTED"


# ═══════════════════════════════════════════════════════════════════════════ #
#  JobMetrics                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class JobMetrics:
    """Counters collected during a job run."""
    start_time: Optional[datetime.datetime] = None
    end_time:   Optional[datetime.datetime] = None
    total_count:    int = 0
    filtered_count: int = 0
    skipped_count:  int = 0   # discarded by processor (RecordProcessingException)
    failed_count:   int = 0
    written_count:  int = 0

    @property
    def processed_count(self) -> int:
        """Records that successfully traversed the entire pipeline and were written."""
        return self.written_count

    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def __str__(self) -> str:
        return (
            f"JobMetrics("
            f"total={self.total_count}, "
            f"written={self.written_count}, "
            f"filtered={self.filtered_count}, "
            f"skipped={self.skipped_count}, "
            f"failed={self.failed_count}, "
            f"duration={self.duration_seconds:.3f}s)"
        )


# ═══════════════════════════════════════════════════════════════════════════ #
#  JobReport                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class JobReport:
    """Full summary of a completed job run."""
    parameters: JobParameters
    metrics:    JobMetrics
    status:     JobStatus
    last_error: Optional[Exception] = None

    def __str__(self) -> str:
        lines = [
            f"Job Report",
            f"  Name    : {self.parameters.name}",
            f"  Status  : {self.status.value}",
            f"  Metrics : {self.metrics}",
        ]
        if self.last_error:
            lines.append(f"  Error   : {self.last_error}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Job                                                                        #
# ═══════════════════════════════════════════════════════════════════════════ #

class Job:
    """
    A processing pipeline. Constructed exclusively via :class:`JobBuilder`.

    Do not instantiate directly.
    """

    def __init__(
        self,
        parameters: JobParameters,
        reader: RecordReader,
        filters: list[RecordFilter],
        mapper: RecordMapper,
        processors: list[RecordProcessor],
        marshaller: Optional[RecordMarshaller],
        writer: RecordWriter,
        job_listeners: list[JobListener],
        batch_listeners: list[BatchListener],
        reader_listeners: list[RecordReaderListener],
        pipeline_listeners: list[PipelineListener],
    ) -> None:
        self._params    = parameters
        self._reader    = reader
        self._filters   = filters
        self._mapper    = mapper
        self._processors= processors
        self._marshaller= marshaller
        self._writer    = writer
        self._jlisteners   = job_listeners
        self._blisteners   = batch_listeners
        self._rlisteners   = reader_listeners
        self._plisteners   = pipeline_listeners

    # ------------------------------------------------------------------ #
    #  Execution                                                           #
    # ------------------------------------------------------------------ #

    def call(self) -> "JobReport":
        """Execute the job and return the :class:`JobReport`."""
        metrics = JobMetrics(start_time=datetime.datetime.now(datetime.timezone.utc))
        status  = JobStatus.STARTING
        last_error: Optional[Exception] = None

        for listener in self._jlisteners:
            try:
                listener.before_job_start(self._params)
            except Exception as exc:
                logger.warning("JobListener.before_job_start raised: %s", exc)

        try:
            status = JobStatus.RUNNING
            self._reader.open()
            self._writer.open()

            batch: Batch = Batch()

            while True:
                # ── Read ──
                for listener in self._rlisteners:
                    listener.before_record_reading()

                try:
                    record = self._reader.read_record()
                except Exception as exc:
                    for listener in self._rlisteners:
                        listener.on_record_reading_exception(exc)
                    logger.error("Error reading record: %s", exc)
                    metrics.failed_count += 1
                    last_error = exc
                    if self._params.error_threshold != -1 and metrics.failed_count > self._params.error_threshold:
                        status = JobStatus.ABORTED
                        break
                    continue

                if record is None:
                    break

                metrics.total_count += 1

                for listener in self._rlisteners:
                    listener.after_record_reading(record)

                for listener in self._plisteners:
                    listener.before_record_processing(record)

                # ── Filter ──
                try:
                    if not all(f.accept(record) for f in self._filters):
                        metrics.filtered_count += 1
                        logger.debug("Record #%d filtered", record.header.number)
                        continue
                except Exception as exc:
                    logger.error("Error filtering record #%d: %s", record.header.number, exc)
                    metrics.failed_count += 1
                    last_error = exc
                    continue

                # ── Map ──
                try:
                    record = self._mapper.map_record(record)
                except Exception as exc:
                    logger.error("Error mapping record #%d: %s", record.header.number, exc)
                    metrics.failed_count += 1
                    last_error = exc
                    for listener in self._plisteners:
                        listener.on_record_processing_exception(record, exc)
                    if self._params.error_threshold != -1 and metrics.failed_count > self._params.error_threshold:
                        status = JobStatus.ABORTED
                        break
                    continue

                # ── Process ──
                skip = False
                for processor in self._processors:
                    try:
                        record = processor.process_record(record)
                    except RecordProcessingException as exc:
                        logger.debug("Record #%d skipped by processor: %s", record.header.number, exc)
                        metrics.skipped_count += 1
                        skip = True
                        break
                    except Exception as exc:
                        logger.error("Processor error on record #%d: %s", record.header.number, exc)
                        metrics.failed_count += 1
                        last_error = exc
                        for listener in self._plisteners:
                            listener.on_record_processing_exception(record, exc)
                        skip = True
                        if self._params.error_threshold != -1 and metrics.failed_count > self._params.error_threshold:
                            status = JobStatus.ABORTED
                        break
                if skip:
                    if status == JobStatus.ABORTED:
                        break
                    continue

                # ── Marshal ──
                if self._marshaller is not None:
                    try:
                        record = self._marshaller.marshal_record(record)
                    except Exception as exc:
                        logger.error("Marshaller error on record #%d: %s", record.header.number, exc)
                        metrics.failed_count += 1
                        last_error = exc
                        continue

                for listener in self._plisteners:
                    listener.after_record_processing(record)

                batch.add(record)

                # ── Flush batch ──
                if len(batch) >= self._params.batch_size:
                    self._flush_batch(batch, metrics)
                    batch.clear()

            # flush remaining
            if not batch.is_empty():
                self._flush_batch(batch, metrics)

            if status != JobStatus.ABORTED:
                status = JobStatus.COMPLETED

        except Exception as exc:
            logger.exception("Unexpected job error: %s", exc)
            status = JobStatus.FAILED
            last_error = exc

        finally:
            try:
                self._reader.close()
            except Exception as exc:
                logger.warning("Error closing reader: %s", exc)
            try:
                self._writer.close()
            except Exception as exc:
                logger.warning("Error closing writer: %s", exc)

        metrics.end_time = datetime.datetime.now(datetime.timezone.utc)
        report = JobReport(parameters=self._params, metrics=metrics, status=status, last_error=last_error)

        for listener in self._jlisteners:
            try:
                listener.after_job_end(report)
            except Exception as exc:
                logger.warning("JobListener.after_job_end raised: %s", exc)

        logger.info("%s", report)
        return report

    def _flush_batch(self, batch: Batch, metrics: JobMetrics) -> None:
        for listener in self._blisteners:
            listener.before_batch_writing(batch)
        try:
            self._writer.write_records(batch)
            metrics.written_count += len(batch)
            for listener in self._blisteners:
                listener.after_batch_writing(batch)
        except Exception as exc:
            logger.error("Batch writing error: %s", exc)
            metrics.failed_count += len(batch)
            for listener in self._blisteners:
                listener.on_batch_writing_exception(batch, exc)

    # allow concurrent.futures to call this directly
    __call__ = call


# ═══════════════════════════════════════════════════════════════════════════ #
#  JobBuilder                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

class JobBuilder:
    """
    Fluent builder for :class:`Job`.

    Example::

        job = (
            JobBuilder()
            .named("csv-to-json")
            .reader(FlatFileRecordReader("tweets.csv"))
            .filter(HeaderRecordFilter())
            .mapper(DelimitedRecordMapper(Tweet, field_names=["id","user","message"]))
            .marshaller(JsonMarshaller())
            .writer(FileRecordWriter("tweets.jsonl"))
            .batch_size(50)
            .build()
        )
    """

    def __init__(self) -> None:
        self._params      = JobParameters()
        self._reader: Optional[RecordReader] = None
        self._filters:    list[RecordFilter]       = []
        self._mapper:     RecordMapper             = PassThroughRecordMapper()
        self._processors: list[RecordProcessor]    = []
        self._marshaller: Optional[RecordMarshaller] = None
        self._writer:     RecordWriter             = DevNullRecordWriter()

        self._jlisteners: list[JobListener]            = []
        self._blisteners: list[BatchListener]          = []
        self._rlisteners: list[RecordReaderListener]   = []
        self._plisteners: list[PipelineListener]       = []

    # ── config ──

    def named(self, name: str) -> "JobBuilder":
        self._params.name = name
        return self

    def batch_size(self, size: int) -> "JobBuilder":
        self._params.batch_size = size
        return self

    def error_threshold(self, threshold: int) -> "JobBuilder":
        self._params.error_threshold = threshold
        return self

    # ── pipeline ──

    def reader(self, reader: RecordReader) -> "JobBuilder":
        self._reader = reader
        return self

    def filter(self, *filters: RecordFilter) -> "JobBuilder":
        self._filters.extend(filters)
        return self

    def mapper(self, mapper: RecordMapper) -> "JobBuilder":
        self._mapper = mapper
        return self

    def processor(self, *processors: RecordProcessor) -> "JobBuilder":
        self._processors.extend(processors)
        return self

    def marshaller(self, marshaller: RecordMarshaller) -> "JobBuilder":
        self._marshaller = marshaller
        return self

    def writer(self, writer: RecordWriter) -> "JobBuilder":
        self._writer = writer
        return self

    # ── listeners ──

    def job_listener(self, *listeners: JobListener) -> "JobBuilder":
        self._jlisteners.extend(listeners)
        return self

    def batch_listener(self, *listeners: BatchListener) -> "JobBuilder":
        self._blisteners.extend(listeners)
        return self

    def reader_listener(self, *listeners: RecordReaderListener) -> "JobBuilder":
        self._rlisteners.extend(listeners)
        return self

    def pipeline_listener(self, *listeners: PipelineListener) -> "JobBuilder":
        self._plisteners.extend(listeners)
        return self

    def build(self) -> Job:
        if self._reader is None:
            raise ValueError("A RecordReader must be provided via .reader(…)")
        return Job(
            parameters=self._params,
            reader=self._reader,
            filters=list(self._filters),
            mapper=self._mapper,
            processors=list(self._processors),
            marshaller=self._marshaller,
            writer=self._writer,
            job_listeners=list(self._jlisteners),
            batch_listeners=list(self._blisteners),
            reader_listeners=list(self._rlisteners),
            pipeline_listeners=list(self._plisteners),
        )
