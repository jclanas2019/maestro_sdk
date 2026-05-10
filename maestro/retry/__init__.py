"""
maestro.retry — resilience policies for Work units and batch jobs.

    from maestro.retry import retry, RetryPolicy, ExponentialBackoff, CircuitBreaker

    resilient = retry(
        work        = call_api_work,
        max_attempts= 5,
        backoff     = ExponentialBackoff(base=0.5, multiplier=2.0, max_delay=30.0),
        on          = [ConnectionError, TimeoutError],
        circuit_breaker = CircuitBreaker(threshold=3, reset_after=60),
    )

Works transparently with every Maestro module:

- ``RetryWork``          — wrap any Work in retry logic
- ``@retryable``         — class decorator for Work subclasses
- ``RetryableReader``    — retry-on-error wrapper for RecordReaders
- ``CircuitBreaker``     — tracks failures; blocks calls when open
- Backoff strategies: ``NoBackoff``, ``ConstantBackoff``,
  ``LinearBackoff``, ``ExponentialBackoff``, ``JitteredBackoff``
"""
from __future__ import annotations

import enum
import functools
import logging
import math
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type

from maestro.flows._work import DefaultWorkReport, Work, WorkContext, WorkReport, WorkStatus

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  Exceptions
# ════════════════════════════════════════════════════════════════════════════

class MaxAttemptsExceeded(Exception):
    """Raised when all retry attempts are exhausted."""
    def __init__(self, attempts: int, last_error: Exception):
        super().__init__(f"Failed after {attempts} attempt(s): {last_error}")
        self.attempts   = attempts
        self.last_error = last_error


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the circuit breaker is open."""
    def __init__(self, name: str, reset_after: float):
        super().__init__(
            f"Circuit '{name}' is OPEN — too many failures. "
            f"Resets in ~{reset_after:.0f}s."
        )


# ════════════════════════════════════════════════════════════════════════════
#  Backoff strategies
# ════════════════════════════════════════════════════════════════════════════

class BackoffStrategy(ABC):
    """Abstract base for delay calculation between retry attempts."""

    @abstractmethod
    def delay(self, attempt: int) -> float:
        """Return the number of seconds to wait before *attempt* (1-based)."""


class NoBackoff(BackoffStrategy):
    """Zero delay — retry immediately."""
    def delay(self, attempt: int) -> float: return 0.0


class ConstantBackoff(BackoffStrategy):
    """Fixed delay between every attempt."""
    def __init__(self, delay_seconds: float = 1.0) -> None:
        self._delay = max(0.0, delay_seconds)
    def delay(self, attempt: int) -> float: return self._delay


class LinearBackoff(BackoffStrategy):
    """Delay grows linearly: ``base + (attempt-1) × increment``."""
    def __init__(self, base: float = 1.0, increment: float = 1.0,
                 max_delay: float = 60.0) -> None:
        self._base = base; self._inc = increment; self._max = max_delay
    def delay(self, attempt: int) -> float:
        return min(self._base + (attempt - 1) * self._inc, self._max)


class ExponentialBackoff(BackoffStrategy):
    """Delay grows exponentially: ``base × multiplier^(attempt-1)``."""
    def __init__(self, base: float = 1.0, multiplier: float = 2.0,
                 max_delay: float = 60.0) -> None:
        self._base = base; self._mul = multiplier; self._max = max_delay
    def delay(self, attempt: int) -> float:
        return min(self._base * (self._mul ** (attempt - 1)), self._max)


class JitteredBackoff(BackoffStrategy):
    """
    Exponential backoff with full jitter — recommended for distributed systems
    to avoid thundering herds.  Actual delay is random in ``[0, exp_delay]``.
    """
    def __init__(self, base: float = 1.0, multiplier: float = 2.0,
                 max_delay: float = 60.0) -> None:
        self._exp = ExponentialBackoff(base, multiplier, max_delay)
    def delay(self, attempt: int) -> float:
        return random.uniform(0, self._exp.delay(attempt))


# ════════════════════════════════════════════════════════════════════════════
#  Circuit Breaker
# ════════════════════════════════════════════════════════════════════════════

class CircuitState(enum.Enum):
    CLOSED    = "CLOSED"      # normal — calls pass through
    OPEN      = "OPEN"        # tripped — calls blocked
    HALF_OPEN = "HALF_OPEN"   # one probe call allowed


class CircuitBreaker:
    """
    Classic three-state circuit breaker.

    Args:
        threshold:    Number of consecutive failures before the circuit opens.
        reset_after:  Seconds to wait before moving OPEN → HALF_OPEN.
        name:         Optional label for logging.

    Usage::

        cb = CircuitBreaker(threshold=3, reset_after=30)
        with cb:
            risky_call()
    """

    def __init__(self, threshold: int = 5, reset_after: float = 60.0,
                 name: str = "circuit") -> None:
        self._threshold   = threshold
        self._reset_after = reset_after
        self._name        = name
        self._state       = CircuitState.CLOSED
        self._failures    = 0
        self._opened_at: Optional[float] = None
        self._lock        = threading.Lock()

    @property
    def state(self) -> CircuitState: return self._state

    @property
    def failure_count(self) -> int: return self._failures

    def allow_call(self) -> bool:
        """Return True if the call should proceed."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.OPEN:
                if (time.monotonic() - self._opened_at) >= self._reset_after:
                    logger.info("Circuit '%s': OPEN → HALF_OPEN (probe allowed)", self._name)
                    self._state = CircuitState.HALF_OPEN
                    return True
                return False
            # HALF_OPEN — allow exactly one probe
            return True

    def on_success(self) -> None:
        with self._lock:
            self._failures  = 0
            self._opened_at = None
            if self._state != CircuitState.CLOSED:
                logger.info("Circuit '%s': %s → CLOSED", self._name, self._state.value)
            self._state = CircuitState.CLOSED

    def on_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == CircuitState.HALF_OPEN or \
               (self._state == CircuitState.CLOSED and self._failures >= self._threshold):
                self._state      = CircuitState.OPEN
                self._opened_at  = time.monotonic()
                logger.warning(
                    "Circuit '%s': OPEN after %d failure(s). Will try again in %.0fs.",
                    self._name, self._failures, self._reset_after,
                )

    # Context-manager for non-Work code
    def __enter__(self) -> "CircuitBreaker":
        if not self.allow_call():
            raise CircuitOpenError(self._name, self._reset_after)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self.on_success()
        else:
            self.on_failure()
        return False  # do not suppress exceptions


# ════════════════════════════════════════════════════════════════════════════
#  RetryPolicy
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class RetryPolicy:
    """
    Configures retry behaviour.

    Args:
        max_attempts:    Maximum number of total attempts (1 = no retry).
        backoff:         Delay strategy between attempts.
        on:              Exception types that trigger a retry.
                         ``None`` → retry on any ``Exception``.
        circuit_breaker: Optional circuit breaker; ``None`` disables it.
        timeout:         Per-attempt timeout in seconds; ``None`` disables.
        on_retry:        Optional callback ``(attempt, delay, error) → None``.
    """
    max_attempts:    int                          = 3
    backoff:         BackoffStrategy              = field(default_factory=ExponentialBackoff)
    on:              Optional[list[Type[Exception]]] = None
    circuit_breaker: Optional[CircuitBreaker]     = None
    timeout:         Optional[float]              = None
    on_retry:        Optional[Callable]           = None


# ════════════════════════════════════════════════════════════════════════════
#  Core retry execution
# ════════════════════════════════════════════════════════════════════════════

def _should_retry(exc: Exception, on: Optional[list[Type[Exception]]]) -> bool:
    if on is None: return True
    return isinstance(exc, tuple(on))


def execute_with_retry(fn: Callable[[], Any], policy: RetryPolicy) -> Any:
    """
    Execute *fn* according to *policy*.  Returns the result of *fn* on success,
    raises :exc:`MaxAttemptsExceeded` when all attempts fail,
    raises :exc:`CircuitOpenError` when the circuit is open.
    """
    cb         = policy.circuit_breaker
    last_error: Optional[Exception] = None

    for attempt in range(1, policy.max_attempts + 1):
        # Circuit check
        if cb and not cb.allow_call():
            raise CircuitOpenError(cb._name, cb._reset_after)

        try:
            if policy.timeout is not None:
                result = _call_with_timeout(fn, policy.timeout)
            else:
                result = fn()

            if cb: cb.on_success()
            return result

        except CircuitOpenError:
            raise

        except Exception as exc:
            last_error = exc
            if cb: cb.on_failure()

            if not _should_retry(exc, policy.on):
                raise  # non-retryable

            if attempt == policy.max_attempts:
                break  # exhausted

            delay = policy.backoff.delay(attempt)
            logger.warning(
                "Attempt %d/%d failed: %s. Retrying in %.2fs…",
                attempt, policy.max_attempts, exc, delay,
            )
            if policy.on_retry:
                try: policy.on_retry(attempt, delay, exc)
                except Exception as _exc:
                    logger.debug("retry on_retry callback raised: %s", _exc)

            if delay > 0:
                time.sleep(delay)

    raise MaxAttemptsExceeded(policy.max_attempts, last_error)


def _call_with_timeout(fn: Callable, timeout: float) -> Any:
    """Run *fn* in a thread with a timeout (best-effort — thread may linger)."""
    result    = [None]
    error     = [None]
    completed = threading.Event()

    def _run():
        try: result[0] = fn()
        except Exception as e: error[0] = e
        finally: completed.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    if not completed.wait(timeout):
        raise TimeoutError(f"Call timed out after {timeout}s")
    if error[0]: raise error[0]
    return result[0]


# ════════════════════════════════════════════════════════════════════════════
#  RetryWork — wraps any Work
# ════════════════════════════════════════════════════════════════════════════

class RetryWork(Work):
    """
    Wraps a :class:`~maestro.flows.Work` with retry logic.

    The wrapped work is re-executed according to *policy* when it either
    raises an exception or returns a FAILED :class:`~maestro.flows.WorkReport`.
    On final failure raises :exc:`MaxAttemptsExceeded`.

    Example::

        from maestro.retry import RetryWork, RetryPolicy, ExponentialBackoff

        resilient = RetryWork(
            work   = unreliable_api_work,
            policy = RetryPolicy(
                max_attempts = 4,
                backoff      = ExponentialBackoff(base=0.5, multiplier=2.0),
                on           = [ConnectionError],
            ),
        )
    """

    def __init__(self, work: Work, policy: Optional[RetryPolicy] = None) -> None:
        self._work   = work
        self._policy = policy or RetryPolicy()

    def get_name(self) -> str:
        return f"retry({self._work.get_name()})"

    def execute(self, work_context: WorkContext) -> WorkReport:
        attempt_log: list[tuple[int, str]] = []

        def attempt() -> WorkReport:
            report = self._work.execute(work_context)
            if report.status == WorkStatus.FAILED:
                err = report.error or Exception(f"Work '{self._work.get_name()}' returned FAILED")
                if not _should_retry(err, self._policy.on):
                    return report  # non-retryable failure — return as-is
                raise err
            return report

        try:
            return execute_with_retry(attempt, self._policy)
        except MaxAttemptsExceeded as e:
            logger.error("RetryWork: '%s' exhausted %d attempt(s): %s",
                         self._work.get_name(), e.attempts, e.last_error)
            return DefaultWorkReport(WorkStatus.FAILED, work_context, error=e)
        except CircuitOpenError as e:
            logger.error("RetryWork: '%s' circuit open: %s", self._work.get_name(), e)
            return DefaultWorkReport(WorkStatus.FAILED, work_context, error=e)


# ════════════════════════════════════════════════════════════════════════════
#  Convenience factory
# ════════════════════════════════════════════════════════════════════════════

def retry(
    work:            Work,
    max_attempts:    int                          = 3,
    backoff:         Optional[BackoffStrategy]    = None,
    on:              Optional[list[Type[Exception]]] = None,
    circuit_breaker: Optional[CircuitBreaker]     = None,
    timeout:         Optional[float]              = None,
    on_retry:        Optional[Callable]           = None,
) -> RetryWork:
    """
    Convenience function — wraps *work* in a :class:`RetryWork`.

    Example::

        step = retry(fetch_work, max_attempts=3, backoff=ExponentialBackoff())
    """
    policy = RetryPolicy(
        max_attempts    = max_attempts,
        backoff         = backoff or ExponentialBackoff(),
        on              = on,
        circuit_breaker = circuit_breaker,
        timeout         = timeout,
        on_retry        = on_retry,
    )
    return RetryWork(work, policy)


# ════════════════════════════════════════════════════════════════════════════
#  @retryable decorator for Work subclasses
# ════════════════════════════════════════════════════════════════════════════

def retryable(
    max_attempts:    int                          = 3,
    backoff:         Optional[BackoffStrategy]    = None,
    on:              Optional[list[Type[Exception]]] = None,
    circuit_breaker: Optional[CircuitBreaker]     = None,
    timeout:         Optional[float]              = None,
):
    """
    Class decorator that makes a Work subclass automatically retried.

    Example::

        @retryable(max_attempts=4, backoff=ExponentialBackoff(base=1.0))
        class FetchDataWork(Work):
            def execute(self, ctx):
                ...
    """
    policy = RetryPolicy(max_attempts=max_attempts,
                         backoff=backoff or ExponentialBackoff(),
                         on=on, circuit_breaker=circuit_breaker, timeout=timeout)

    def decorator(cls):
        original_execute = cls.execute

        @functools.wraps(original_execute)
        def patched_execute(self_work, work_context: WorkContext) -> WorkReport:
            def _attempt():
                report = original_execute(self_work, work_context)
                if report.status == WorkStatus.FAILED:
                    err = report.error or Exception(f"{cls.__name__} returned FAILED")
                    raise err
                return report

            try:
                return execute_with_retry(_attempt, policy)
            except MaxAttemptsExceeded as e:
                return DefaultWorkReport(WorkStatus.FAILED, work_context, error=e)
            except CircuitOpenError as e:
                return DefaultWorkReport(WorkStatus.FAILED, work_context, error=e)

        cls.execute = patched_execute
        return cls

    return decorator


# ════════════════════════════════════════════════════════════════════════════
#  RetryableReader — retry-on-error wrapper for batch RecordReaders
# ════════════════════════════════════════════════════════════════════════════

class RetryableReader:
    """
    Wraps any :class:`~maestro.batch.RecordReader` so that transient read errors
    are retried according to *policy*.

    Conforms to the RecordReader protocol — open/read_record/close.

    Example::

        from maestro.retry import RetryableReader, RetryPolicy, ConstantBackoff

        reader = RetryableReader(
            reader = HttpRecordReader("https://api.example.com/events"),
            policy = RetryPolicy(max_attempts=3, backoff=ConstantBackoff(2.0),
                                 on=[ConnectionError]),
        )
        job = JobBuilder().reader(reader).build()
    """

    def __init__(self, reader, policy: Optional[RetryPolicy] = None) -> None:
        self._reader = reader
        self._policy = policy or RetryPolicy()

    @property
    def source_name(self) -> str:
        return getattr(self._reader, "source_name", type(self._reader).__name__)

    def open(self) -> None:
        execute_with_retry(self._reader.open, self._policy)

    def read_record(self):
        return execute_with_retry(self._reader.read_record, self._policy)

    def close(self) -> None:
        try: self._reader.close()
        except Exception as e:
            logger.warning("RetryableReader: error during close: %s", e)

    def __iter__(self):
        self.open()
        try:
            while True:
                record = self.read_record()
                if record is None: break
                yield record
        finally:
            self.close()


__all__ = [
    "BackoffStrategy", "NoBackoff", "ConstantBackoff", "LinearBackoff",
    "ExponentialBackoff", "JitteredBackoff",
    "CircuitBreaker", "CircuitState", "CircuitOpenError",
    "RetryPolicy", "RetryWork", "RetryableReader",
    "MaxAttemptsExceeded",
    "retry", "retryable", "execute_with_retry",
]
