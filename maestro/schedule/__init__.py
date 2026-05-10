"""
maestro.schedule — cron-style scheduler for flows and batch jobs.

No external dependencies. Uses a background thread and a lightweight
cron-expression parser.

    from maestro.schedule import Scheduler, CronTrigger, IntervalTrigger

    scheduler = Scheduler()

    # Every 5 minutes
    scheduler.add("hourly-etl",
                  job     = my_batch_job,
                  trigger = CronTrigger("*/5 * * * *"))

    # Every 30 seconds
    scheduler.add("heartbeat",
                  work    = LambdaWork(lambda c: ping()),
                  trigger = IntervalTrigger(seconds=30))

    scheduler.start()
    ...
    scheduler.stop()

Trigger types
-------------
* ``CronTrigger(expression)``    — standard 5-field cron (min hr dom mon dow)
* ``IntervalTrigger(seconds=N)`` — every N seconds
* ``OnceTrigger(at=datetime)``   — fire once at a specific moment
* ``ImmediateTrigger()``         — fire once as soon as the scheduler starts

Supports::

    scheduler.pause("hourly-etl")
    scheduler.resume("hourly-etl")
    scheduler.remove("hourly-etl")
    scheduler.status()            # → list of task status dicts
    scheduler.next_runs()         # → dict[name, next_datetime]
"""
from __future__ import annotations

import datetime
import enum
import logging
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  Cron expression parser
# ════════════════════════════════════════════════════════════════════════════

def _parse_cron_field(expr: str, lo: int, hi: int) -> frozenset[int]:
    """Parse one cron field (e.g. ``"*/5"``, ``"1,3,5"``, ``"0-6/2"``). """
    values: set[int] = set()
    for part in expr.split(","):
        if "/" in part:
            range_part, step_s = part.rsplit("/", 1)
            step = int(step_s)
            if range_part == "*":
                seq = range(lo, hi + 1)
            elif "-" in range_part:
                a, b = range_part.split("-", 1)
                seq = range(int(a), int(b) + 1)
            else:
                seq = range(int(range_part), hi + 1)
            values.update(seq[::step])
        elif part == "*":
            values.update(range(lo, hi + 1))
        elif "-" in part:
            a, b = part.split("-", 1)
            values.update(range(int(a), int(b) + 1))
        else:
            v = int(part)
            if not (lo <= v <= hi):
                raise ValueError(f"Cron value {v} out of range [{lo}, {hi}]")
            values.add(v)
    return frozenset(values)


class _CronExpr:
    """Parsed cron expression (5-field: minute hour dom month dow)."""
    __slots__ = ("minutes", "hours", "days", "months", "weekdays", "_raw")

    def __init__(self, expression: str) -> None:
        self._raw = expression
        parts = expression.split()
        if len(parts) != 5:
            raise ValueError(
                f"Cron expression must have 5 fields (got {len(parts)}): {expression!r}"
            )
        self.minutes  = _parse_cron_field(parts[0], 0, 59)
        self.hours    = _parse_cron_field(parts[1], 0, 23)
        self.days     = _parse_cron_field(parts[2], 1, 31)
        self.months   = _parse_cron_field(parts[3], 1, 12)
        self.weekdays = _parse_cron_field(parts[4], 0, 6)   # 0=Sunday

    def matches(self, dt: datetime.datetime) -> bool:
        dow = dt.isoweekday() % 7   # isoweekday: Mon=1 … Sun=7 → Sun=0
        return (
            dt.minute  in self.minutes
            and dt.hour   in self.hours
            and dt.day    in self.days
            and dt.month  in self.months
            and dow        in self.weekdays
        )

    def next_after(self, dt: datetime.datetime) -> datetime.datetime:
        """Return the next datetime >= dt+1min that matches this expression."""
        candidate = dt.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
        # search forward, max 2 years
        limit = candidate + datetime.timedelta(days=366 * 2)
        while candidate < limit:
            dow = candidate.isoweekday() % 7
            if (candidate.month  in self.months
                    and candidate.day    in self.days
                    and dow               in self.weekdays
                    and candidate.hour   in self.hours
                    and candidate.minute in self.minutes):
                return candidate
            candidate += datetime.timedelta(minutes=1)
        raise RuntimeError(f"Cannot find next run for cron {self._raw!r}")

    def __str__(self) -> str:
        return self._raw


# ════════════════════════════════════════════════════════════════════════════
#  Trigger types
# ════════════════════════════════════════════════════════════════════════════

class Trigger(ABC):
    """Abstract base for all triggers."""

    @abstractmethod
    def next_fire_time(self, last_fire: Optional[datetime.datetime],
                       now: datetime.datetime) -> Optional[datetime.datetime]:
        """Return the next datetime to fire, or None if the trigger is exhausted."""


class CronTrigger(Trigger):
    """
    Fire on a 5-field cron schedule.

    Fields: ``minute hour day-of-month month day-of-week``

    Examples::

        CronTrigger("0 * * * *")       # every hour at minute 0
        CronTrigger("*/15 * * * *")    # every 15 minutes
        CronTrigger("0 9 * * 1-5")     # weekdays at 09:00
        CronTrigger("30 6 1 * *")      # 1st of every month at 06:30

    Predefined shortcuts::

        CronTrigger.HOURLY      = "0 * * * *"
        CronTrigger.DAILY       = "0 0 * * *"
        CronTrigger.WEEKLY      = "0 0 * * 0"
        CronTrigger.MONTHLY     = "0 0 1 * *"
    """
    HOURLY  = "0 * * * *"
    DAILY   = "0 0 * * *"
    WEEKLY  = "0 0 * * 0"
    MONTHLY = "0 0 1 * *"

    def __init__(self, expression: str) -> None:
        self._expr = _CronExpr(expression)

    def next_fire_time(self, last_fire, now):
        base = last_fire or now
        return self._expr.next_after(base)

    def __repr__(self) -> str:
        return f"CronTrigger({self._expr!s})"


class IntervalTrigger(Trigger):
    """
    Fire every *seconds* seconds after the previous run.

    Example::

        IntervalTrigger(seconds=30)    # every 30s
        IntervalTrigger(seconds=3600)  # every hour
    """

    def __init__(self, seconds: float, start_immediately: bool = True) -> None:
        self._interval = datetime.timedelta(seconds=seconds)
        self._start    = start_immediately

    def next_fire_time(self, last_fire, now):
        if last_fire is None:
            return now if self._start else now + self._interval
        return last_fire + self._interval

    def __repr__(self) -> str:
        return f"IntervalTrigger({self._interval.total_seconds()}s)"


class OnceTrigger(Trigger):
    """
    Fire once at a specific :class:`datetime.datetime`.

    Example::

        OnceTrigger(datetime.datetime(2025, 12, 31, 23, 59))
    """

    def __init__(self, at: datetime.datetime) -> None:
        self._at   = at
        self._fired = False

    def next_fire_time(self, last_fire, now):
        if self._fired or last_fire is not None:
            return None   # already fired or has been fired
        return self._at

    def __repr__(self) -> str:
        return f"OnceTrigger(at={self._at})"


class ImmediateTrigger(Trigger):
    """Fire once as soon as possible, then stop."""

    def next_fire_time(self, last_fire, now):
        return None if last_fire else now

    def __repr__(self) -> str:
        return "ImmediateTrigger()"


# ════════════════════════════════════════════════════════════════════════════
#  Task state
# ════════════════════════════════════════════════════════════════════════════

class TaskState(enum.Enum):
    WAITING  = "WAITING"
    RUNNING  = "RUNNING"
    PAUSED   = "PAUSED"
    FINISHED = "FINISHED"   # for one-shot triggers that have fired


@dataclass
class TaskRun:
    """Record of a single execution of a scheduled task."""
    started_at:  datetime.datetime
    finished_at: Optional[datetime.datetime] = None
    success:     bool = False
    error:       Optional[Exception] = None

    @property
    def duration(self) -> Optional[float]:
        if self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None


@dataclass
class ScheduledTask:
    """
    Handle for a task registered with :class:`Scheduler`.

    Use to pause, resume, or cancel a scheduled task.
    """
    name:      str
    trigger:   Trigger
    _scheduler: Any    # ref to Scheduler, avoid circular import
    id:        str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    state:     TaskState = TaskState.WAITING
    history:   list[TaskRun] = field(default_factory=list)
    last_fire: Optional[datetime.datetime] = None
    next_fire: Optional[datetime.datetime] = None

    def pause(self) -> None:
        """Suspend this task until ``resume()`` is called."""
        if self.state == TaskState.WAITING:
            self.state = TaskState.PAUSED
            logger.info("Task '%s' paused.", self.name)

    def resume(self) -> None:
        """Resume a paused task."""
        if self.state == TaskState.PAUSED:
            self.state = TaskState.WAITING
            self._refresh_next_fire()
            logger.info("Task '%s' resumed.", self.name)

    def cancel(self) -> None:
        """Cancel and remove this task from the scheduler."""
        self._scheduler.remove(self.name)

    def _refresh_next_fire(self) -> None:
        now = datetime.datetime.now()
        self.next_fire = self.trigger.next_fire_time(self.last_fire, now)

    @property
    def run_count(self) -> int: return len(self.history)

    @property
    def last_run(self) -> Optional[TaskRun]:
        return self.history[-1] if self.history else None


# ════════════════════════════════════════════════════════════════════════════
#  ScheduleListener
# ════════════════════════════════════════════════════════════════════════════

class ScheduleListener:
    """Override the methods you need."""
    def on_task_started(self, task: ScheduledTask) -> None: pass
    def on_task_succeeded(self, task: ScheduledTask, run: TaskRun) -> None: pass
    def on_task_failed(self, task: ScheduledTask, run: TaskRun) -> None: pass


# ════════════════════════════════════════════════════════════════════════════
#  Scheduler
# ════════════════════════════════════════════════════════════════════════════

class Scheduler:
    """
    Background-thread scheduler that fires registered tasks according to their triggers.

    Usage::

        scheduler = Scheduler()
        scheduler.add("etl", job=my_batch_job, trigger=CronTrigger("0 * * * *"))
        scheduler.add("ping", work=ping_work,  trigger=IntervalTrigger(seconds=60))
        scheduler.start()
        ...
        scheduler.stop()

    Context manager::

        with Scheduler() as s:
            s.add("task", work=my_work, trigger=IntervalTrigger(seconds=5))
            time.sleep(30)
    """

    def __init__(
        self,
        tick_seconds:   float = 1.0,
        max_workers:    int   = 4,
        listeners:      Optional[list[ScheduleListener]] = None,
    ) -> None:
        self._tick      = tick_seconds
        self._max_w     = max_workers
        self._listeners = listeners or []
        self._tasks:    dict[str, ScheduledTask] = {}
        self._callables: dict[str, Callable]     = {}
        self._lock      = threading.RLock()
        self._thread:   Optional[threading.Thread] = None
        self._running   = False
        self._executor: Optional[Any] = None

    # ── Registration ──────────────────────────────────────────────────── #

    def add(
        self,
        name:    str,
        trigger: Trigger,
        work=None,
        job=None,
        context_factory: Optional[Callable[[], Any]] = None,
    ) -> ScheduledTask:
        """
        Register a task.

        Supply exactly one of *work* (a :class:`~maestro.flows.Work`) or
        *job* (a :class:`~maestro.batch.Job`).

        *context_factory* is called each time to produce a fresh
        :class:`~maestro.flows.WorkContext` for Work tasks.

        Returns a :class:`ScheduledTask` handle.
        """
        if work is None and job is None:
            raise ValueError("Provide either 'work' or 'job'.")
        if work is not None and job is not None:
            raise ValueError("Provide only one of 'work' or 'job'.")

        task = ScheduledTask(name=name, trigger=trigger, _scheduler=self)
        now  = datetime.datetime.now()
        task.next_fire = trigger.next_fire_time(None, now)

        def _callable():
            if job is not None:
                return job.call()
            else:
                from maestro.flows._work import WorkContext as WC
                ctx = context_factory() if context_factory else WC()
                return work.execute(ctx)

        with self._lock:
            self._tasks[name]     = task
            self._callables[name] = _callable

        logger.info("Scheduler: task '%s' registered (trigger=%s, next=%s)",
                    name, trigger, task.next_fire)
        return task

    def remove(self, name: str) -> None:
        with self._lock:
            self._tasks.pop(name, None)
            self._callables.pop(name, None)
        logger.info("Scheduler: task '%s' removed.", name)

    def pause(self, name: str) -> None:
        with self._lock:
            if name in self._tasks:
                self._tasks[name].pause()

    def resume(self, name: str) -> None:
        with self._lock:
            if name in self._tasks:
                self._tasks[name].resume()

    # ── Lifecycle ─────────────────────────────────────────────────────── #

    def start(self) -> "Scheduler":
        if self._running:
            return self
        import concurrent.futures
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_w, thread_name_prefix="maestro-scheduler"
        )
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True,
                                          name="maestro-scheduler-ticker")
        self._thread.start()
        logger.info("Scheduler started (tick=%.1fs, workers=%d).", self._tick, self._max_w)
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=timeout)
        if self._executor:
            self._executor.shutdown(wait=False)
        logger.info("Scheduler stopped.")

    def _loop(self) -> None:
        while self._running:
            now = datetime.datetime.now()
            with self._lock:
                due = [(name, task, self._callables[name])
                       for name, task in list(self._tasks.items())
                       if task.state == TaskState.WAITING
                       and task.next_fire is not None
                       and task.next_fire <= now]

            for name, task, fn in due:
                self._executor.submit(self._run_task, name, task, fn)

            time.sleep(self._tick)

    def _run_task(self, name: str, task: ScheduledTask, fn: Callable) -> None:
        now = datetime.datetime.now()
        run = TaskRun(started_at=now)

        with self._lock:
            if task.state != TaskState.WAITING:
                return
            task.state = TaskState.RUNNING

        for l in self._listeners: l.on_task_started(task)
        logger.info("Scheduler: running task '%s'", name)

        try:
            fn()
            run.success     = True
            run.finished_at = datetime.datetime.now()
            for l in self._listeners: l.on_task_succeeded(task, run)
        except Exception as exc:
            run.error       = exc
            run.success     = False
            run.finished_at = datetime.datetime.now()
            logger.error("Scheduler: task '%s' raised: %s", name, exc)
            logger.debug(traceback.format_exc())
            for l in self._listeners: l.on_task_failed(task, run)

        with self._lock:
            if name not in self._tasks:
                return   # removed during execution
            task.history.append(run)
            task.last_fire = now
            task.next_fire = task.trigger.next_fire_time(now, datetime.datetime.now())

            if task.next_fire is None:
                task.state = TaskState.FINISHED
                logger.info("Scheduler: task '%s' finished (one-shot trigger exhausted).", name)
            else:
                task.state = TaskState.WAITING

    # ── Inspection ────────────────────────────────────────────────────── #

    def status(self) -> list[dict]:
        """Return a list of status dicts for all registered tasks."""
        with self._lock:
            return [
                {
                    "name":       name,
                    "state":      task.state.value,
                    "trigger":    repr(task.trigger),
                    "next_fire":  str(task.next_fire) if task.next_fire else "—",
                    "last_fire":  str(task.last_fire) if task.last_fire else "never",
                    "run_count":  task.run_count,
                    "last_ok":    task.last_run.success if task.last_run else None,
                }
                for name, task in self._tasks.items()
            ]

    def next_runs(self) -> dict[str, Optional[datetime.datetime]]:
        with self._lock:
            return {name: task.next_fire for name, task in self._tasks.items()}

    def get_task(self, name: str) -> Optional[ScheduledTask]:
        with self._lock: return self._tasks.get(name)

    def __enter__(self) -> "Scheduler":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def __repr__(self) -> str:
        return f"Scheduler(tasks={len(self._tasks)}, running={self._running})"


__all__ = [
    "Trigger", "CronTrigger", "IntervalTrigger", "OnceTrigger", "ImmediateTrigger",
    "TaskState", "TaskRun", "ScheduledTask", "ScheduleListener",
    "Scheduler",
]
