"""
tests/test_easy_flows.py — comprehensive unit tests for the easy_flows port.

Run:  python -m pytest tests/test_easy_flows.py -v
"""
import sys, os


import concurrent.futures
import threading
import time

import pytest

from maestro.flows import (
    WorkContext, WorkStatus, DefaultWorkReport,
    Work, LambdaWork, NoOpWork, WorkReportPredicate,
    SequentialFlow, ConditionalFlow, ParallelFlow, RepeatFlow,
    WorkFlowEngine,
    aNewSequentialFlow, aNewConditionalFlow, aNewParallelFlow,
    aNewRepeatFlow, aNewWorkFlowEngine,
)
from maestro.flows._parallel import ParallelFlowReport


# ─────────────────────────────────────────────────────────────────── #
#  Test work helpers                                                  #
# ─────────────────────────────────────────────────────────────────── #

def completed_work(name="ok", side_effect=None):
    def _exec(ctx):
        if side_effect:
            side_effect()
        return DefaultWorkReport(WorkStatus.COMPLETED, ctx)
    return LambdaWork(_exec, name=name)


def failed_work(name="fail"):
    return LambdaWork(
        lambda ctx: DefaultWorkReport(WorkStatus.FAILED, ctx, error=Exception(name)),
        name=name,
    )


def context_setter(key, value, name="setter"):
    return LambdaWork(lambda ctx: ctx.put(key, value), name=name)


# ─────────────────────────────────────────────────────────────────── #
#  WorkContext                                                         #
# ─────────────────────────────────────────────────────────────────── #

class TestWorkContext:
    def test_put_get(self):
        ctx = WorkContext()
        ctx.put("x", 10)
        assert ctx.get("x") == 10

    def test_get_default(self):
        assert WorkContext().get("missing", 42) == 42

    def test_remove(self):
        ctx = WorkContext(a=1)
        ctx.remove("a")
        assert not ctx.contains("a")

    def test_as_map(self):
        ctx = WorkContext(a=1, b=2)
        assert ctx.as_map() == {"a": 1, "b": 2}

    def test_kwargs_init(self):
        ctx = WorkContext(x=5, y="hello")
        assert ctx.get("x") == 5

    def test_dict_style(self):
        ctx = WorkContext()
        ctx["k"] = "v"
        assert ctx["k"] == "v"
        assert "k" in ctx


# ─────────────────────────────────────────────────────────────────── #
#  WorkReport                                                         #
# ─────────────────────────────────────────────────────────────────── #

class TestWorkReport:
    def test_completed(self):
        ctx = WorkContext()
        r = DefaultWorkReport(WorkStatus.COMPLETED, ctx)
        assert r.status == WorkStatus.COMPLETED
        assert r.error is None

    def test_failed_with_error(self):
        ctx = WorkContext()
        exc = RuntimeError("boom")
        r = DefaultWorkReport(WorkStatus.FAILED, ctx, error=exc)
        assert r.status == WorkStatus.FAILED
        assert r.error is exc


# ─────────────────────────────────────────────────────────────────── #
#  LambdaWork / NoOpWork                                              #
# ─────────────────────────────────────────────────────────────────── #

class TestLambdaWork:
    def test_returns_completed_on_none(self):
        w = LambdaWork(lambda ctx: None, name="noop")
        r = w.execute(WorkContext())
        assert r.status == WorkStatus.COMPLETED

    def test_returns_report_directly(self):
        ctx = WorkContext()
        w = LambdaWork(
            lambda c: DefaultWorkReport(WorkStatus.FAILED, c),
            name="fail"
        )
        assert w.execute(ctx).status == WorkStatus.FAILED

    def test_exception_becomes_failed(self):
        w = LambdaWork(lambda ctx: (_ for _ in ()).throw(RuntimeError("x")), name="exc")
        # rewrite without generator trick
        def boom(ctx): raise RuntimeError("x")
        w2 = LambdaWork(boom, name="boom", fail_on_exception=True)
        r = w2.execute(WorkContext())
        assert r.status == WorkStatus.FAILED

    def test_no_op(self):
        r = NoOpWork().execute(WorkContext())
        assert r.status == WorkStatus.COMPLETED


# ─────────────────────────────────────────────────────────────────── #
#  WorkReportPredicate                                                #
# ─────────────────────────────────────────────────────────────────── #

class TestWorkReportPredicate:
    def _report(self, status):
        return DefaultWorkReport(status, WorkContext())

    def test_completed(self):
        p = WorkReportPredicate.COMPLETED
        assert p(self._report(WorkStatus.COMPLETED)) is True
        assert p(self._report(WorkStatus.FAILED)) is False

    def test_failed(self):
        p = WorkReportPredicate.FAILED
        assert p(self._report(WorkStatus.FAILED)) is True
        assert p(self._report(WorkStatus.COMPLETED)) is False

    def test_always_true(self):
        assert WorkReportPredicate.ALWAYS_TRUE(self._report(WorkStatus.FAILED)) is True

    def test_always_false(self):
        assert WorkReportPredicate.ALWAYS_FALSE(self._report(WorkStatus.COMPLETED)) is False

    def test_not(self):
        not_failed = ~WorkReportPredicate.FAILED
        assert not_failed(self._report(WorkStatus.COMPLETED)) is True
        assert not_failed(self._report(WorkStatus.FAILED)) is False

    def test_and(self):
        p = WorkReportPredicate.COMPLETED & WorkReportPredicate.ALWAYS_TRUE
        assert p(self._report(WorkStatus.COMPLETED)) is True
        assert p(self._report(WorkStatus.FAILED)) is False

    def test_or(self):
        p = WorkReportPredicate.COMPLETED | WorkReportPredicate.FAILED
        assert p(self._report(WorkStatus.COMPLETED)) is True
        assert p(self._report(WorkStatus.FAILED)) is True

    def test_custom_predicate(self):
        p = WorkReportPredicate(lambda r: r.work_context.get("x", 0) > 5)
        ctx_yes = WorkContext(x=10)
        ctx_no  = WorkContext(x=2)
        assert p(DefaultWorkReport(WorkStatus.COMPLETED, ctx_yes)) is True
        assert p(DefaultWorkReport(WorkStatus.COMPLETED, ctx_no)) is False


# ─────────────────────────────────────────────────────────────────── #
#  SequentialFlow                                                     #
# ─────────────────────────────────────────────────────────────────── #

class TestSequentialFlow:
    def test_all_steps_run(self):
        log = []
        flow = (
            aNewSequentialFlow()
            .execute(LambdaWork(lambda ctx: log.append(1), name="w1"))
            .then(LambdaWork(lambda ctx: log.append(2), name="w2"))
            .then(LambdaWork(lambda ctx: log.append(3), name="w3"))
            .build()
        )
        r = WorkFlowEngine().run(flow, WorkContext())
        assert r.status == WorkStatus.COMPLETED
        assert log == [1, 2, 3]

    def test_stops_on_failure(self):
        log = []
        flow = (
            aNewSequentialFlow()
            .execute(LambdaWork(lambda ctx: log.append("a"), name="a"))
            .then(failed_work())
            .then(LambdaWork(lambda ctx: log.append("c"), name="c"))
            .build()
        )
        r = WorkFlowEngine().run(flow, WorkContext())
        assert r.status == WorkStatus.FAILED
        assert log == ["a"]
        assert "c" not in log

    def test_context_threaded_through(self):
        flow = (
            aNewSequentialFlow()
            .execute(LambdaWork(lambda ctx: ctx.put("step", 1), name="s1"))
            .then(LambdaWork(lambda ctx: ctx.put("step", ctx.get("step") + 1), name="s2"))
            .build()
        )
        ctx = WorkContext()
        WorkFlowEngine().run(flow, ctx)
        assert ctx.get("step") == 2

    def test_single_step(self):
        r = aNewSequentialFlow().execute(completed_work()).build().execute(WorkContext())
        assert r.status == WorkStatus.COMPLETED

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aNewSequentialFlow().build()

    def test_name(self):
        flow = aNewSequentialFlow().named("my-flow").execute(NoOpWork()).build()
        assert flow.get_name() == "my-flow"


# ─────────────────────────────────────────────────────────────────── #
#  ConditionalFlow                                                    #
# ─────────────────────────────────────────────────────────────────── #

class TestConditionalFlow:
    def test_then_branch_on_completed(self):
        log = []
        flow = (
            aNewConditionalFlow()
            .execute(completed_work())
            .when(WorkReportPredicate.COMPLETED)
            .then(LambdaWork(lambda ctx: log.append("then"), name="then"))
            .otherwise(LambdaWork(lambda ctx: log.append("otherwise"), name="otherwise"))
            .build()
        )
        r = WorkFlowEngine().run(flow, WorkContext())
        assert r.status == WorkStatus.COMPLETED
        assert log == ["then"]

    def test_otherwise_branch_on_failed(self):
        log = []
        flow = (
            aNewConditionalFlow()
            .execute(failed_work())
            .when(WorkReportPredicate.COMPLETED)
            .then(LambdaWork(lambda ctx: log.append("then"), name="then"))
            .otherwise(LambdaWork(lambda ctx: log.append("otherwise"), name="otherwise"))
            .build()
        )
        WorkFlowEngine().run(flow, WorkContext())
        assert log == ["otherwise"]

    def test_no_otherwise_is_noop(self):
        flow = (
            aNewConditionalFlow()
            .execute(failed_work())
            .when(WorkReportPredicate.COMPLETED)
            .then(completed_work("then"))
            .build()
        )
        r = WorkFlowEngine().run(flow, WorkContext())
        assert r.status == WorkStatus.COMPLETED  # NoOpWork is used

    def test_missing_initial_raises(self):
        with pytest.raises(ValueError):
            aNewConditionalFlow().when(WorkReportPredicate.COMPLETED).then(NoOpWork()).build()

    def test_missing_predicate_raises(self):
        with pytest.raises(ValueError):
            aNewConditionalFlow().execute(NoOpWork()).then(NoOpWork()).build()

    def test_missing_then_raises(self):
        with pytest.raises(ValueError):
            aNewConditionalFlow().execute(NoOpWork()).when(WorkReportPredicate.COMPLETED).build()

    def test_custom_predicate(self):
        log = []
        ctx = WorkContext(score=90)
        high_score = WorkReportPredicate(lambda r: r.work_context.get("score", 0) >= 80)
        flow = (
            aNewConditionalFlow()
            .execute(completed_work())
            .when(high_score)
            .then(LambdaWork(lambda c: log.append("high"), name="high"))
            .otherwise(LambdaWork(lambda c: log.append("low"), name="low"))
            .build()
        )
        WorkFlowEngine().run(flow, ctx)
        assert log == ["high"]


# ─────────────────────────────────────────────────────────────────── #
#  RepeatFlow                                                         #
# ─────────────────────────────────────────────────────────────────── #

class TestRepeatFlow:
    def test_fixed_times(self):
        log = []
        flow = (
            aNewRepeatFlow()
            .repeat(LambdaWork(lambda ctx: log.append(1), name="w"))
            .times(4)
            .build()
        )
        r = WorkFlowEngine().run(flow, WorkContext())
        assert r.status == WorkStatus.COMPLETED
        assert len(log) == 4

    def test_stops_early_on_failure(self):
        log = []
        def work_fn(ctx):
            log.append(1)
            return DefaultWorkReport(WorkStatus.FAILED, ctx)
        flow = aNewRepeatFlow().repeat(LambdaWork(work_fn, name="w")).times(5).build()
        r = WorkFlowEngine().run(flow, WorkContext())
        assert r.status == WorkStatus.FAILED
        assert len(log) == 1  # stops after first failure

    def test_until_predicate(self):
        ctx = WorkContext(count=0)
        def increment(c):
            c.put("count", c.get("count") + 1)
        stop_at_3 = WorkReportPredicate(lambda r: r.work_context.get("count", 0) >= 3)
        flow = (
            aNewRepeatFlow()
            .repeat(LambdaWork(increment, name="inc"))
            .until(stop_at_3)
            .times(100)
            .build()
        )
        WorkFlowEngine().run(flow, ctx)
        assert ctx.get("count") == 3

    def test_until_with_max_guard(self):
        """until predicate never fires — stop at times limit."""
        ctx = WorkContext(count=0)
        def increment(c): c.put("count", c.get("count") + 1)
        flow = (
            aNewRepeatFlow()
            .repeat(LambdaWork(increment, "inc"))
            .until(WorkReportPredicate.ALWAYS_FALSE)
            .times(5)
            .build()
        )
        WorkFlowEngine().run(flow, ctx)
        assert ctx.get("count") == 5

    def test_missing_work_raises(self):
        with pytest.raises(ValueError):
            aNewRepeatFlow().times(3).build()

    def test_name(self):
        flow = aNewRepeatFlow().named("repeat-test").repeat(NoOpWork()).build()
        assert flow.get_name() == "repeat-test"


# ─────────────────────────────────────────────────────────────────── #
#  ParallelFlow                                                       #
# ─────────────────────────────────────────────────────────────────── #

class TestParallelFlow:
    def test_all_work_runs(self):
        log = []
        lock = threading.Lock()

        def make_work(val):
            def fn(ctx):
                with lock:
                    log.append(val)
            return LambdaWork(fn, name=str(val))

        flow = (
            aNewParallelFlow()
            .execute(make_work(1), make_work(2), make_work(3))
            .build()
        )
        r = WorkFlowEngine().run(flow, WorkContext())
        assert isinstance(r, ParallelFlowReport)
        assert r.status == WorkStatus.COMPLETED
        assert sorted(log) == [1, 2, 3]

    def test_failed_if_any_fails(self):
        flow = (
            aNewParallelFlow()
            .execute(completed_work("a"), failed_work("b"), completed_work("c"))
            .build()
        )
        r = WorkFlowEngine().run(flow, WorkContext())
        assert r.status == WorkStatus.FAILED

    def test_reports_count(self):
        flow = aNewParallelFlow().execute(NoOpWork(), NoOpWork()).build()
        r = WorkFlowEngine().run(flow, WorkContext())
        assert len(r.reports) == 2

    def test_with_external_executor(self):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        flow = (
            aNewParallelFlow()
            .execute(completed_work("a"), completed_work("b"))
            .with_executor(executor)
            .build()
        )
        r = WorkFlowEngine().run(flow, WorkContext())
        executor.shutdown(wait=True)
        assert r.status == WorkStatus.COMPLETED

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aNewParallelFlow().build()

    def test_name(self):
        flow = aNewParallelFlow().named("par").execute(NoOpWork()).build()
        assert flow.get_name() == "par"


# ─────────────────────────────────────────────────────────────────── #
#  WorkFlowEngine                                                     #
# ─────────────────────────────────────────────────────────────────── #

class TestWorkFlowEngine:
    def test_engine_runs_workflow(self):
        engine = aNewWorkFlowEngine().named("test-engine").build()
        flow = aNewSequentialFlow().execute(completed_work()).build()
        r = engine.run(flow, WorkContext())
        assert r.status == WorkStatus.COMPLETED

    def test_engine_returns_failed_report(self):
        engine = WorkFlowEngine()
        r = engine.run(failed_work(), WorkContext())
        assert r.status == WorkStatus.FAILED


# ─────────────────────────────────────────────────────────────────── #
#  Composition — nested flows                                         #
# ─────────────────────────────────────────────────────────────────── #

class TestComposedFlows:
    def test_sequential_inside_repeat(self):
        """RepeatFlow repeating a SequentialFlow."""
        log = []
        inner = (
            aNewSequentialFlow()
            .execute(LambdaWork(lambda ctx: log.append("a"), name="a"))
            .then(LambdaWork(lambda ctx: log.append("b"), name="b"))
            .build()
        )
        flow = aNewRepeatFlow().repeat(inner).times(2).build()
        WorkFlowEngine().run(flow, WorkContext())
        assert log == ["a", "b", "a", "b"]

    def test_canonical_tutorial(self):
        """
        print foo 3×  →  parallel(hello, world)  →  if COMPLETED → done else nok
        """
        log = []
        lock = threading.Lock()

        def printer(msg):
            return LambdaWork(lambda ctx: (lock.acquire(), log.append(msg), lock.release()), name=msg)

        exec_svc = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        try:
            workflow = (
                aNewSequentialFlow()
                .execute(aNewRepeatFlow().repeat(printer("foo")).times(3).build())
                .then(
                    aNewConditionalFlow()
                    .execute(
                        aNewParallelFlow()
                        .execute(printer("hello"), printer("world"))
                        .with_executor(exec_svc)
                        .build()
                    )
                    .when(WorkReportPredicate.COMPLETED)
                    .then(printer("done"))
                    .otherwise(printer("nok"))
                    .build()
                )
                .build()
            )
            r = WorkFlowEngine().run(workflow, WorkContext())
            assert r.status == WorkStatus.COMPLETED
            assert log.count("foo") == 3
            assert "hello" in log
            assert "world" in log
            assert "done" in log
            assert "nok" not in log
        finally:
            exec_svc.shutdown(wait=True)

    def test_parallel_inside_sequential(self):
        log = []
        lock = threading.Lock()
        exec_svc = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        try:
            flow = (
                aNewSequentialFlow()
                .execute(LambdaWork(lambda ctx: log.append("before"), name="before"))
                .then(
                    aNewParallelFlow()
                    .execute(
                        LambdaWork(lambda ctx: (lock.acquire(), log.append("p1"), lock.release()), name="p1"),
                        LambdaWork(lambda ctx: (lock.acquire(), log.append("p2"), lock.release()), name="p2"),
                    )
                    .with_executor(exec_svc)
                    .build()
                )
                .then(LambdaWork(lambda ctx: log.append("after"), name="after"))
                .build()
            )
            WorkFlowEngine().run(flow, WorkContext())
            assert log[0] == "before"
            assert log[-1] == "after"
            assert "p1" in log and "p2" in log
        finally:
            exec_svc.shutdown(wait=True)

    def test_conditional_wrapping_repeat(self):
        ctx = WorkContext(count=0)
        def inc(c): c.put("count", c.get("count") + 1)
        repeat = aNewRepeatFlow().repeat(LambdaWork(inc, "inc")).times(3).build()
        flow = (
            aNewConditionalFlow()
            .execute(completed_work())
            .when(WorkReportPredicate.COMPLETED)
            .then(repeat)
            .build()
        )
        WorkFlowEngine().run(flow, ctx)
        assert ctx.get("count") == 3
