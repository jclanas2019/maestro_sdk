"""
maestro.async_ — AsyncIO-native versions of flows and batch processing.

Drop-in async replacements for IO-bound pipelines — HTTP calls, database
queries, file streams — that would block threads in the synchronous API.

Flows
-----
    from maestro.async_ import (
        AsyncWork, AsyncLambdaWork, AsyncNoOpWork,
        AsyncSequentialFlow, AsyncConditionalFlow,
        AsyncParallelFlow, AsyncRepeatFlow,
        AsyncWorkFlowEngine,
    )

    async def fetch(ctx: WorkContext) -> WorkReport:
        data = await httpx.get("https://api.example.com/data")
        ctx.put("data", data.json())
        return DefaultWorkReport(WorkStatus.COMPLETED, ctx)

    flow = (AsyncSequentialFlow.Builder()
            .execute(AsyncLambdaWork(fetch))
            .then(AsyncLambdaWork(process))
            .build())

    engine = AsyncWorkFlowEngine()
    report = await engine.run(flow, WorkContext())

Batch
-----
    from maestro.async_ import AsyncRecordReader, AsyncRecordWriter, AsyncJob, AsyncJobBuilder

    class HttpRecordReader(AsyncRecordReader):
        async def read_record(self): ...

    job = (AsyncJobBuilder()
           .named("async-etl")
           .reader(reader)
           .writer(writer)
           .build())
    report = await job.call()

Adapters
--------
    from maestro.async_ import sync_to_async, async_to_sync

    async_work = sync_to_async(my_sync_work)   # run sync Work in thread pool
    sync_work  = async_to_sync(my_async_work)  # wrap async Work for sync engine
"""
from __future__ import annotations

import asyncio
import datetime
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Optional

from maestro.flows._work import (
    DefaultWorkReport, Work, WorkContext, WorkReport, WorkStatus,
)
from maestro.flows._predicate import WorkReportPredicate

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  AsyncWork — the async counterpart of Work
# ════════════════════════════════════════════════════════════════════════════

class AsyncWork(ABC):
    """
    Base class for async work units.

    Implement ``execute`` as a coroutine::

        class FetchDataWork(AsyncWork):
            async def execute(self, ctx: WorkContext) -> WorkReport:
                data = await some_async_call()
                ctx.put("data", data)
                return DefaultWorkReport(WorkStatus.COMPLETED, ctx)
    """

    def get_name(self) -> str:
        return type(self).__name__

    @abstractmethod
    async def execute(self, work_context: WorkContext) -> WorkReport: ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.get_name()!r})"


class AsyncNoOpWork(AsyncWork):
    """Always completes, does nothing."""
    def __init__(self, name: str = "async-no-op"): self._name = name
    def get_name(self) -> str: return self._name
    async def execute(self, ctx: WorkContext) -> WorkReport:
        return DefaultWorkReport(WorkStatus.COMPLETED, ctx)


class AsyncLambdaWork(AsyncWork):
    """
    Wraps a coroutine function (or plain function) as an :class:`AsyncWork`.

    If *fn* is a regular function it is run in the default executor
    (thread pool) so it does not block the event loop.

    Example::

        work = AsyncLambdaWork(
            lambda ctx: asyncio.sleep(0.1),
            name="wait-100ms",
        )
    """

    def __init__(self, fn: Callable, name: str = "") -> None:
        self._fn   = fn
        self._name = name or getattr(fn, "__name__", "async-lambda")

    def get_name(self) -> str:
        return self._name

    async def execute(self, ctx: WorkContext) -> WorkReport:
        try:
            if asyncio.iscoroutinefunction(self._fn):
                result = await self._fn(ctx)
            else:
                loop   = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self._fn, ctx)

            if isinstance(result, WorkReport):
                return result
            return DefaultWorkReport(WorkStatus.COMPLETED, ctx)
        except Exception as exc:
            logger.error("AsyncLambdaWork %r raised: %s", self._name, exc)
            return DefaultWorkReport(WorkStatus.FAILED, ctx, error=exc)


# ════════════════════════════════════════════════════════════════════════════
#  Async flow types
# ════════════════════════════════════════════════════════════════════════════

async def _run(work: "AsyncWork | Work", ctx: WorkContext) -> WorkReport:
    """Execute either an AsyncWork or a sync Work."""
    if isinstance(work, AsyncWork):
        return await work.execute(ctx)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, work.execute, ctx)


class AsyncSequentialFlow(AsyncWork):
    """
    Async version of :class:`~maestro.flows.SequentialFlow`.

    Steps run one after another; stops on first FAILED.

    Example::

        flow = (AsyncSequentialFlow.Builder()
                .named("pipeline")
                .execute(step_a)
                .then(step_b)
                .build())
        report = await AsyncWorkFlowEngine().run(flow, WorkContext())
    """

    def __init__(self, name: str, works: list) -> None:
        self._name  = name
        self._works = works

    def get_name(self) -> str: return self._name

    async def execute(self, ctx: WorkContext) -> WorkReport:
        logger.info("AsyncSequentialFlow '%s': starting %d steps", self._name, len(self._works))
        report: WorkReport = DefaultWorkReport(WorkStatus.COMPLETED, ctx)
        for work in self._works:
            report = await _run(work, ctx)
            if report.status == WorkStatus.FAILED:
                return report
        return report

    class Builder:
        def __init__(self) -> None:
            self._name  = "async-sequential"
            self._works = []
        def named(self, n: str) -> "AsyncSequentialFlow.Builder":
            self._name = n; return self
        def execute(self, w) -> "AsyncSequentialFlow.Builder":
            self._works.append(w); return self
        def then(self, w) -> "AsyncSequentialFlow.Builder":
            self._works.append(w); return self
        def build(self) -> "AsyncSequentialFlow":
            if not self._works: raise ValueError("At least one step required.")
            return AsyncSequentialFlow(self._name, list(self._works))


class AsyncConditionalFlow(AsyncWork):
    """
    Async version of :class:`~maestro.flows.ConditionalFlow`.

    Executes *initial*, evaluates *predicate*, then runs *then_work* or *otherwise_work*.
    """

    def __init__(self, name: str, initial, predicate: WorkReportPredicate,
                 then_work, otherwise_work=None) -> None:
        self._name      = name
        self._initial   = initial
        self._predicate = predicate
        self._then      = then_work
        self._otherwise = otherwise_work

    def get_name(self) -> str: return self._name

    async def execute(self, ctx: WorkContext) -> WorkReport:
        try:
            report = await _run(self._initial, ctx)
        except Exception as exc:
            return DefaultWorkReport(WorkStatus.FAILED, ctx, error=exc)

        branch = self._then if self._predicate.test(report) else self._otherwise
        if branch is None:
            return DefaultWorkReport(WorkStatus.COMPLETED, ctx)
        return await _run(branch, ctx)

    class Builder:
        def __init__(self) -> None:
            self._name = "async-conditional"
            self._initial = self._predicate = self._then = self._otherwise = None
        def named(self, n: str) -> "AsyncConditionalFlow.Builder":
            self._name = n; return self
        def execute(self, w) -> "AsyncConditionalFlow.Builder":
            self._initial = w; return self
        def when(self, p: WorkReportPredicate) -> "AsyncConditionalFlow.Builder":
            self._predicate = p; return self
        def then(self, w) -> "AsyncConditionalFlow.Builder":
            self._then = w; return self
        def otherwise(self, w) -> "AsyncConditionalFlow.Builder":
            self._otherwise = w; return self
        def build(self) -> "AsyncConditionalFlow":
            if not self._initial:   raise ValueError("Need .execute(…)")
            if not self._predicate: raise ValueError("Need .when(…)")
            if not self._then:      raise ValueError("Need .then(…)")
            return AsyncConditionalFlow(self._name, self._initial, self._predicate,
                                        self._then, self._otherwise)


class AsyncParallelFlow(AsyncWork):
    """
    Async version of :class:`~maestro.flows.ParallelFlow`.

    All work units run concurrently via ``asyncio.gather``.
    Each gets its own context copy; results are merged back.
    """

    def __init__(self, name: str, works: list) -> None:
        self._name  = name
        self._works = works

    def get_name(self) -> str: return self._name

    async def execute(self, ctx: WorkContext) -> WorkReport:
        from maestro.flows._parallel import ParallelFlowReport

        async def run_one(work) -> WorkReport:
            ctx_copy = WorkContext(**ctx.as_map())
            report   = await _run(work, ctx_copy)
            # merge back
            for k, v in ctx_copy.as_map().items():
                ctx.put(k, v)
            return report

        reports = await asyncio.gather(
            *[run_one(w) for w in self._works],
            return_exceptions=False,
        )
        return ParallelFlowReport(list(reports), ctx)

    class Builder:
        def __init__(self) -> None:
            self._name = "async-parallel"; self._works = []
        def named(self, n: str) -> "AsyncParallelFlow.Builder":
            self._name = n; return self
        def execute(self, *works) -> "AsyncParallelFlow.Builder":
            self._works.extend(works); return self
        def build(self) -> "AsyncParallelFlow":
            if not self._works: raise ValueError("Need at least one work unit.")
            return AsyncParallelFlow(self._name, list(self._works))


class AsyncRepeatFlow(AsyncWork):
    """
    Async version of :class:`~maestro.flows.RepeatFlow`.

    Repeats *work* up to *times* iterations or until *until* predicate is True.
    """

    def __init__(self, name: str, work, times: int = 1,
                 until: Optional[WorkReportPredicate] = None) -> None:
        self._name  = name
        self._work  = work
        self._times = max(1, times)
        self._until = until

    def get_name(self) -> str: return self._name

    async def execute(self, ctx: WorkContext) -> WorkReport:
        report: WorkReport = DefaultWorkReport(WorkStatus.COMPLETED, ctx)
        for i in range(self._times):
            report = await _run(self._work, ctx)
            if self._until is None and report.status == WorkStatus.FAILED:
                break
            if self._until is not None and self._until.test(report):
                break
        return report

    class Builder:
        def __init__(self) -> None:
            self._name = "async-repeat"; self._work = None
            self._times = 1; self._until = None
        def named(self, n: str) -> "AsyncRepeatFlow.Builder":
            self._name = n; return self
        def repeat(self, w) -> "AsyncRepeatFlow.Builder":
            self._work = w; return self
        def times(self, n: int) -> "AsyncRepeatFlow.Builder":
            self._times = n; return self
        def until(self, p: WorkReportPredicate) -> "AsyncRepeatFlow.Builder":
            self._until = p; return self
        def build(self) -> "AsyncRepeatFlow":
            if not self._work: raise ValueError("Need .repeat(…)")
            return AsyncRepeatFlow(self._name, self._work, self._times, self._until)


# ════════════════════════════════════════════════════════════════════════════
#  AsyncWorkFlowEngine
# ════════════════════════════════════════════════════════════════════════════

class AsyncWorkFlowEngine:
    """
    Async counterpart of :class:`~maestro.flows.WorkFlowEngine`.

    Example::

        engine = AsyncWorkFlowEngine()
        report = await engine.run(flow, WorkContext())

    Running from sync code::

        report = asyncio.run(engine.run(flow, WorkContext()))
    """

    async def run(self, workflow, work_context: WorkContext) -> WorkReport:
        logger.info("AsyncWorkFlowEngine: running '%s'", workflow.get_name())
        return await _run(workflow, work_context)

    def run_sync(self, workflow, work_context: WorkContext) -> WorkReport:
        """Convenience for calling from non-async code."""
        return asyncio.run(self.run(workflow, work_context))


# ════════════════════════════════════════════════════════════════════════════
#  Async batch
# ════════════════════════════════════════════════════════════════════════════

from maestro.batch._record import Batch, Header, Record
from maestro.batch._filter    import RecordFilter
from maestro.batch._mapper    import RecordMapper, PassThroughRecordMapper
from maestro.batch._processor import RecordMarshaller, RecordProcessor, RecordProcessingException
from maestro.batch._listener  import BatchListener, JobListener, PipelineListener, RecordReaderListener
from maestro.batch._job       import JobMetrics, JobParameters, JobReport, JobStatus


class AsyncRecordReader(ABC):
    """Abstract base for async record readers."""
    async def open(self) -> None: pass

    @abstractmethod
    async def read_record(self) -> Optional[Record]: ...

    async def close(self) -> None: pass

    @property
    def source_name(self) -> str: return type(self).__name__

    async def __aenter__(self) -> "AsyncRecordReader":
        await self.open(); return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def __aiter__(self):
        await self.open()
        try:
            while True:
                record = await self.read_record()
                if record is None: break
                yield record
        finally:
            await self.close()


class AsyncIterableReader(AsyncRecordReader):
    """Async reader that wraps a Python iterable (useful for tests)."""
    def __init__(self, data, source: str = "iterable"):
        self._data = data; self._source = source
        self._iter = None; self._counter = 0

    @property
    def source_name(self) -> str: return self._source

    async def open(self) -> None:
        self._iter = iter(self._data); self._counter = 0

    async def read_record(self) -> Optional[Record]:
        try:
            payload = next(self._iter)
            self._counter += 1
            return Record(Header(self._counter, self._source), payload)
        except StopIteration:
            return None

    async def close(self) -> None:
        self._iter = None


class AsyncRecordWriter(ABC):
    """Abstract base for async record writers."""
    async def open(self) -> None: pass

    @abstractmethod
    async def write_records(self, batch: Batch) -> None: ...

    async def close(self) -> None: pass


class AsyncCollectionWriter(AsyncRecordWriter):
    """Writes to a Python list (useful for tests)."""
    def __init__(self, collection: Optional[list] = None):
        self._col = collection if collection is not None else []

    @property
    def collection(self) -> list: return self._col

    async def write_records(self, batch: Batch) -> None:
        for record in batch: self._col.append(record.payload)


class AsyncDevNullWriter(AsyncRecordWriter):
    """Discards all records."""
    async def write_records(self, batch: Batch) -> None: pass


# ════════════════════════════════════════════════════════════════════════════
#  AsyncJob — the async ETL pipeline
# ════════════════════════════════════════════════════════════════════════════

class AsyncJob:
    """
    An async batch processing pipeline.

    Built via :class:`AsyncJobBuilder`. Supports both async readers/writers
    and synchronous ones (sync is run in thread executor to avoid blocking).
    """

    def __init__(self, params: JobParameters, reader,
                 filters: list, mapper, processors: list,
                 marshaller, writer,
                 jlisteners: list, blisteners: list,
                 rlisteners: list, plisteners: list) -> None:
        self._params      = params
        self._reader      = reader
        self._filters     = filters
        self._mapper      = mapper
        self._processors  = processors
        self._marshaller  = marshaller
        self._writer      = writer
        self._jl, self._bl, self._rl, self._pl = jlisteners, blisteners, rlisteners, plisteners

    async def call(self) -> JobReport:
        import datetime as _dt
        metrics   = JobMetrics(start_time=_dt.datetime.now(_dt.timezone.utc))
        status    = JobStatus.STARTING
        last_err  = None

        for l in self._jl:
            try: l.before_job_start(self._params)
            except Exception as _exc:
                logger.debug("async lifecycle error (ignored): %s", _exc)

        try:
            status = JobStatus.RUNNING
            await self._open(self._reader)
            await self._open(self._writer)
            batch = Batch()

            while True:
                try:
                    record = await self._read(self._reader)
                except Exception as exc:
                    metrics.failed_count += 1; last_err = exc; continue

                if record is None: break
                metrics.total_count += 1

                # Filter
                try:
                    if not all(f.accept(record) for f in self._filters):
                        metrics.filtered_count += 1; continue
                except Exception as exc:
                    metrics.failed_count += 1; last_err = exc; continue

                # Map
                try:
                    record = self._mapper.map_record(record)
                except Exception as exc:
                    metrics.failed_count += 1; last_err = exc
                    if self._params.error_threshold != -1 and metrics.failed_count > self._params.error_threshold:
                        status = JobStatus.ABORTED; break
                    continue

                # Process
                skip = False
                for proc in self._processors:
                    try:
                        record = proc.process_record(record)
                    except RecordProcessingException:
                        metrics.skipped_count += 1; skip = True; break
                    except Exception as exc:
                        metrics.failed_count += 1; last_err = exc; skip = True; break
                if skip:
                    if status == JobStatus.ABORTED: break
                    continue

                # Marshal
                if self._marshaller:
                    try: record = self._marshaller.marshal_record(record)
                    except Exception as exc:
                        metrics.failed_count += 1; last_err = exc; continue

                batch.add(record)
                if len(batch) >= self._params.batch_size:
                    await self._flush(batch, metrics)
                    batch.clear()

            if not batch.is_empty():
                await self._flush(batch, metrics)

            if status != JobStatus.ABORTED:
                status = JobStatus.COMPLETED

        except Exception as exc:
            status = JobStatus.FAILED; last_err = exc
        finally:
            await self._close(self._reader)
            await self._close(self._writer)

        import datetime as _dt
        metrics.end_time = _dt.datetime.now(_dt.timezone.utc)
        report = JobReport(parameters=self._params, metrics=metrics,
                           status=status, last_error=last_err)
        for l in self._jl:
            try: l.after_job_end(report)
            except Exception as _exc:
                logger.debug("async lifecycle error (ignored): %s", _exc)
        return report

    async def _open(self, obj) -> None:
        if hasattr(obj, "open"):
            if asyncio.iscoroutinefunction(obj.open): await obj.open()
            else: obj.open()

    async def _close(self, obj) -> None:
        if hasattr(obj, "close"):
            if asyncio.iscoroutinefunction(obj.close): await obj.close()
            else: obj.close()

    async def _read(self, reader) -> Optional[Record]:
        if asyncio.iscoroutinefunction(reader.read_record):
            return await reader.read_record()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, reader.read_record)

    async def _flush(self, batch: Batch, metrics: JobMetrics) -> None:
        try:
            if asyncio.iscoroutinefunction(self._writer.write_records):
                await self._writer.write_records(batch)
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._writer.write_records, batch)
            metrics.written_count += len(batch)
        except Exception as exc:
            logger.error("AsyncJob: batch write error: %s", exc)
            metrics.failed_count += len(batch)


class AsyncJobBuilder:
    """
    Fluent builder for :class:`AsyncJob`. Mirrors :class:`~maestro.batch.JobBuilder`.

    Example::

        job = (AsyncJobBuilder()
               .named("async-pipeline")
               .reader(AsyncIterableReader(range(100)))
               .writer(AsyncCollectionWriter(sink))
               .batch_size(20)
               .build())
        report = await job.call()
    """

    def __init__(self) -> None:
        self._params      = JobParameters()
        self._reader      = None
        self._filters:    list = []
        self._mapper      = PassThroughRecordMapper()
        self._processors: list = []
        self._marshaller  = None
        self._writer      = AsyncDevNullWriter()
        self._jl, self._bl, self._rl, self._pl = [], [], [], []

    def named(self, n: str) -> "AsyncJobBuilder": self._params.name = n; return self
    def batch_size(self, n: int) -> "AsyncJobBuilder": self._params.batch_size = n; return self
    def error_threshold(self, n: int) -> "AsyncJobBuilder": self._params.error_threshold = n; return self
    def reader(self, r) -> "AsyncJobBuilder": self._reader = r; return self
    def filter(self, *fs) -> "AsyncJobBuilder": self._filters.extend(fs); return self
    def mapper(self, m) -> "AsyncJobBuilder": self._mapper = m; return self
    def processor(self, *ps) -> "AsyncJobBuilder": self._processors.extend(ps); return self
    def marshaller(self, m) -> "AsyncJobBuilder": self._marshaller = m; return self
    def writer(self, w) -> "AsyncJobBuilder": self._writer = w; return self
    def job_listener(self, *ls) -> "AsyncJobBuilder": self._jl.extend(ls); return self

    def build(self) -> AsyncJob:
        if not self._reader: raise ValueError("A reader is required.")
        return AsyncJob(self._params, self._reader,
                        list(self._filters), self._mapper,
                        list(self._processors), self._marshaller, self._writer,
                        self._jl, self._bl, self._rl, self._pl)


# ════════════════════════════════════════════════════════════════════════════
#  Adapters — bridge sync ↔ async
# ════════════════════════════════════════════════════════════════════════════

class SyncWorkAdapter(AsyncWork):
    """
    Run a synchronous :class:`~maestro.flows.Work` inside an async flow
    without blocking the event loop (uses thread executor).

    Example::

        flow = AsyncSequentialFlow.Builder()
               .execute(sync_to_async(my_heavy_sync_work))
               .build()
    """
    def __init__(self, work: Work) -> None:
        self._work = work
    def get_name(self) -> str: return f"sync({self._work.get_name()})"
    async def execute(self, ctx: WorkContext) -> WorkReport:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._work.execute, ctx)


class AsyncWorkAdapter(Work):
    """
    Run an :class:`AsyncWork` from a synchronous flow engine.
    Uses ``asyncio.run`` (creates a new event loop per call).

    Example::

        flow = SequentialFlow.Builder()
               .execute(async_to_sync(my_async_work))
               .build()
    """
    def __init__(self, async_work: AsyncWork) -> None:
        self._async_work = async_work
    def get_name(self) -> str: return f"sync_wrap({self._async_work.get_name()})"
    def execute(self, ctx: WorkContext) -> WorkReport:
        return asyncio.run(self._async_work.execute(ctx))


def sync_to_async(work: Work) -> SyncWorkAdapter:
    """Wrap a sync Work for use in an async flow (runs in thread pool)."""
    return SyncWorkAdapter(work)


def async_to_sync(work: AsyncWork) -> AsyncWorkAdapter:
    """Wrap an async Work for use in a sync flow (via asyncio.run)."""
    return AsyncWorkAdapter(work)


__all__ = [
    "AsyncWork", "AsyncNoOpWork", "AsyncLambdaWork",
    "AsyncSequentialFlow", "AsyncConditionalFlow",
    "AsyncParallelFlow", "AsyncRepeatFlow",
    "AsyncWorkFlowEngine",
    "AsyncRecordReader", "AsyncIterableReader",
    "AsyncRecordWriter", "AsyncCollectionWriter", "AsyncDevNullWriter",
    "AsyncJob", "AsyncJobBuilder",
    "SyncWorkAdapter", "AsyncWorkAdapter",
    "sync_to_async", "async_to_sync",
]
