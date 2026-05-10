"""tests/test_p1_retry.py — retry module tests"""
import sys, os, time, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import maestro
from maestro.retry import (
    NoBackoff, ConstantBackoff, LinearBackoff, ExponentialBackoff, JitteredBackoff,
    CircuitBreaker, CircuitState, CircuitOpenError,
    RetryPolicy, RetryWork, RetryableReader,
    MaxAttemptsExceeded, retry, retryable, execute_with_retry,
)


# ── Backoff strategies ────────────────────────────────────────────────────── #

class TestBackoffStrategies:
    def test_no_backoff(self):
        assert NoBackoff().delay(1) == 0.0
        assert NoBackoff().delay(5) == 0.0

    def test_constant_backoff(self):
        b = ConstantBackoff(2.5)
        assert b.delay(1) == 2.5
        assert b.delay(10) == 2.5

    def test_linear_backoff(self):
        b = LinearBackoff(base=1.0, increment=2.0, max_delay=10.0)
        assert b.delay(1) == 1.0
        assert b.delay(2) == 3.0
        assert b.delay(3) == 5.0
        assert b.delay(10) == 10.0  # capped

    def test_exponential_backoff(self):
        b = ExponentialBackoff(base=1.0, multiplier=2.0, max_delay=60.0)
        assert b.delay(1) == 1.0
        assert b.delay(2) == 2.0
        assert b.delay(3) == 4.0
        assert b.delay(4) == 8.0

    def test_exponential_capped(self):
        b = ExponentialBackoff(base=1.0, multiplier=10.0, max_delay=30.0)
        assert b.delay(5) == 30.0

    def test_jittered_in_range(self):
        b = JitteredBackoff(base=1.0, multiplier=2.0, max_delay=60.0)
        for attempt in range(1, 6):
            d = b.delay(attempt)
            assert 0 <= d <= ExponentialBackoff(1.0, 2.0, 60.0).delay(attempt)


# ── CircuitBreaker ────────────────────────────────────────────────────────── #

class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(threshold=3, reset_after=60)
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_call() is True

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(threshold=3, reset_after=60)
        cb.on_failure(); cb.on_failure(); cb.on_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_call() is False

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(threshold=3, reset_after=60)
        cb.on_failure(); cb.on_failure()
        cb.on_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_transitions_to_half_open_after_reset(self):
        cb = CircuitBreaker(threshold=1, reset_after=0.01)
        cb.on_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.05)
        assert cb.allow_call() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_closes_on_success(self):
        cb = CircuitBreaker(threshold=1, reset_after=0.01)
        cb.on_failure()
        time.sleep(0.05)
        cb.allow_call()  # move to HALF_OPEN
        cb.on_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self):
        cb = CircuitBreaker(threshold=1, reset_after=0.01)
        cb.on_failure()
        time.sleep(0.05)
        cb.allow_call()  # HALF_OPEN
        cb.on_failure()
        assert cb.state == CircuitState.OPEN

    def test_context_manager_success(self):
        cb = CircuitBreaker(threshold=3)
        with cb:
            pass  # no error
        assert cb.state == CircuitState.CLOSED

    def test_context_manager_failure(self):
        cb = CircuitBreaker(threshold=1)
        with pytest.raises(ValueError):
            with cb:
                raise ValueError("oops")
        assert cb.state == CircuitState.OPEN

    def test_context_manager_open_raises(self):
        cb = CircuitBreaker(threshold=1, reset_after=999)
        cb.on_failure()
        with pytest.raises(CircuitOpenError):
            with cb:
                pass


# ── execute_with_retry ────────────────────────────────────────────────────── #

class TestExecuteWithRetry:
    def test_succeeds_on_first_attempt(self):
        result = execute_with_retry(lambda: 42, RetryPolicy(max_attempts=3, backoff=NoBackoff()))
        assert result == 42

    def test_succeeds_after_failures(self):
        calls = [0]
        def fn():
            calls[0] += 1
            if calls[0] < 3: raise ValueError("not yet")
            return "done"
        result = execute_with_retry(fn, RetryPolicy(max_attempts=5, backoff=NoBackoff()))
        assert result == "done"
        assert calls[0] == 3

    def test_raises_after_max_attempts(self):
        def fn(): raise ValueError("always fails")
        with pytest.raises(MaxAttemptsExceeded) as exc:
            execute_with_retry(fn, RetryPolicy(max_attempts=3, backoff=NoBackoff()))
        assert exc.value.attempts == 3

    def test_respects_on_filter(self):
        def fn(): raise TypeError("wrong type")
        # TypeError not in the retry list → should not retry
        with pytest.raises(TypeError):
            execute_with_retry(fn, RetryPolicy(max_attempts=5, backoff=NoBackoff(),
                                               on=[ValueError]))

    def test_on_retry_callback(self):
        log = []
        def fn():
            if len(log) < 2: raise ValueError("retry me")
            return "ok"
        execute_with_retry(fn, RetryPolicy(
            max_attempts=5, backoff=NoBackoff(),
            on_retry=lambda attempt, delay, exc: log.append(attempt),
        ))
        assert log == [1, 2]

    def test_circuit_breaker_blocks_when_open(self):
        cb = CircuitBreaker(threshold=1, reset_after=999)
        cb.on_failure()  # force open
        with pytest.raises(CircuitOpenError):
            execute_with_retry(lambda: 1, RetryPolicy(max_attempts=3, backoff=NoBackoff(),
                                                       circuit_breaker=cb))


# ── RetryWork ─────────────────────────────────────────────────────────────── #

class TestRetryWork:
    def _failing_then_ok(self, fail_times: int):
        calls = [0]
        def execute(ctx):
            calls[0] += 1
            if calls[0] <= fail_times:
                raise ValueError(f"attempt {calls[0]} failed")
            return maestro.DefaultWorkReport(maestro.WorkStatus.COMPLETED, ctx)
        return maestro.LambdaWork(execute, name="flaky"), calls

    def test_succeeds_after_retries(self):
        work, calls = self._failing_then_ok(fail_times=2)
        rw = RetryWork(work, RetryPolicy(max_attempts=5, backoff=NoBackoff()))
        report = rw.execute(maestro.WorkContext())
        assert report.status == maestro.WorkStatus.COMPLETED
        assert calls[0] == 3

    def test_failed_after_exhausting_attempts(self):
        work = maestro.LambdaWork(lambda ctx: (_ for _ in ()).throw(RuntimeError("always")),
                                   name="always-fails")
        # rewrite without generator trick
        def always_fail(ctx): raise RuntimeError("always")
        work2 = maestro.LambdaWork(always_fail, name="always-fails")
        rw = RetryWork(work2, RetryPolicy(max_attempts=3, backoff=NoBackoff()))
        report = rw.execute(maestro.WorkContext())
        assert report.status == maestro.WorkStatus.FAILED
        assert isinstance(report.error, MaxAttemptsExceeded)

    def test_failed_work_report_triggers_retry(self):
        calls = [0]
        def execute(ctx):
            calls[0] += 1
            if calls[0] < 3:
                return maestro.DefaultWorkReport(maestro.WorkStatus.FAILED, ctx,
                                                  error=ValueError("not yet"))
            return maestro.DefaultWorkReport(maestro.WorkStatus.COMPLETED, ctx)
        work = maestro.LambdaWork(execute, name="flaky")
        rw = RetryWork(work, RetryPolicy(max_attempts=5, backoff=NoBackoff()))
        report = rw.execute(maestro.WorkContext())
        assert report.status == maestro.WorkStatus.COMPLETED

    def test_convenience_retry_function(self):
        calls = [0]
        def execute(ctx):
            calls[0] += 1
            if calls[0] < 2: raise ValueError("retry")
            return maestro.DefaultWorkReport(maestro.WorkStatus.COMPLETED, ctx)
        rw = retry(maestro.LambdaWork(execute, "w"), max_attempts=3, backoff=NoBackoff())
        report = rw.execute(maestro.WorkContext())
        assert report.status == maestro.WorkStatus.COMPLETED

    def test_inside_sequential_flow(self):
        calls = [0]
        def execute(ctx):
            calls[0] += 1
            if calls[0] < 3: raise ValueError("retry")
            return maestro.DefaultWorkReport(maestro.WorkStatus.COMPLETED, ctx)
        rw = retry(maestro.LambdaWork(execute, "w"), max_attempts=5, backoff=NoBackoff())
        flow = maestro.SequentialFlow.Builder().execute(rw).build()
        report = maestro.WorkFlowEngine().run(flow, maestro.WorkContext())
        assert report.status == maestro.WorkStatus.COMPLETED


# ── @retryable decorator ──────────────────────────────────────────────────── #

class TestRetryableDecorator:
    def test_decorator_retries_on_exception(self):
        calls = [0]

        @retryable(max_attempts=4, backoff=NoBackoff())
        class FlickyWork(maestro.Work):
            def execute(self, ctx):
                calls[0] += 1
                if calls[0] < 3: raise RuntimeError("not yet")
                return maestro.DefaultWorkReport(maestro.WorkStatus.COMPLETED, ctx)

        report = FlickyWork().execute(maestro.WorkContext())
        assert report.status == maestro.WorkStatus.COMPLETED
        assert calls[0] == 3

    def test_decorator_returns_failed_after_exhaustion(self):
        @retryable(max_attempts=2, backoff=NoBackoff())
        class AlwaysFail(maestro.Work):
            def execute(self, ctx):
                raise RuntimeError("gone")

        report = AlwaysFail().execute(maestro.WorkContext())
        assert report.status == maestro.WorkStatus.FAILED


# ── RetryableReader ───────────────────────────────────────────────────────── #

class TestRetryableReader:
    def test_reads_successfully(self):
        reader = maestro.IterableRecordReader([1, 2, 3])
        rr = RetryableReader(reader, RetryPolicy(max_attempts=2, backoff=NoBackoff()))
        records = list(rr)
        assert [r.payload for r in records] == [1, 2, 3]

    def test_retries_transient_read_error(self):
        calls = [0]
        class FlakyReader(maestro.RecordReader):
            def __init__(self):
                self._inner = maestro.IterableRecordReader([10, 20])
                self._inner.open()
            def open(self): pass
            def read_record(self):
                calls[0] += 1
                if calls[0] == 1: raise ConnectionError("transient")
                return self._inner.read_record()
            def close(self): self._inner.close()

        rr = RetryableReader(FlakyReader(), RetryPolicy(
            max_attempts=3, backoff=NoBackoff(), on=[ConnectionError]
        ))
        records = [r for r in rr]
        assert calls[0] >= 2
