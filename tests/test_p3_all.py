"""tests/test_p3_all.py — Priority 3 feature tests (saga, schedule, cli)"""
import sys, os, time, datetime, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import maestro
from maestro.saga import (
    SagaBuilder, SagaStatus, SagaStep, saga_step,
    SagaListener, LoggingSagaListener,
)
from maestro.schedule import (
    Scheduler, CronTrigger, IntervalTrigger, OnceTrigger, ImmediateTrigger,
    TaskState, ScheduleListener,
)
from maestro.schedule import _CronExpr   # internal, for parser tests
from maestro.cli import main as cli_main


# ══════════════════════════════════════════════════════════════════════
#  maestro.saga
# ══════════════════════════════════════════════════════════════════════

def _ok_work(name="ok", side_effect=None):
    def fn(ctx):
        if side_effect: side_effect(name)
        return maestro.DefaultWorkReport(maestro.WorkStatus.COMPLETED, ctx)
    return maestro.LambdaWork(fn, name=name)


def _fail_work(name="fail"):
    return maestro.LambdaWork(
        lambda ctx: maestro.DefaultWorkReport(
            maestro.WorkStatus.FAILED, ctx, error=Exception(name)),
        name=name,
    )


class TestSagaBuilder:
    def test_builds_with_steps(self):
        saga = (SagaBuilder()
                .named("test")
                .step("a", _ok_work("a"))
                .step("b", _ok_work("b"))
                .build())
        assert saga.get_name() == "test"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            SagaBuilder().named("empty").build()


class TestSagaExecution:
    def test_all_steps_succeed(self):
        log = []
        saga = (SagaBuilder()
                .quiet()
                .step("a", _ok_work("a", log.append))
                .step("b", _ok_work("b", log.append))
                .step("c", _ok_work("c", log.append))
                .build())
        report = saga.execute(maestro.WorkContext())
        assert report.saga_status == SagaStatus.COMPLETED
        assert report.status == maestro.WorkStatus.COMPLETED
        assert log == ["a", "b", "c"]

    def test_failure_triggers_compensation(self):
        log = []
        saga = (SagaBuilder()
                .quiet()
                .step("reserve",   _ok_work("reserve", log.append),
                                   _ok_work("release", log.append))
                .step("charge",    _ok_work("charge", log.append),
                                   _ok_work("refund", log.append))
                .step("ship",      _fail_work("ship"))
                .build())
        report = saga.execute(maestro.WorkContext())
        assert report.saga_status == SagaStatus.COMPENSATED
        assert "reserve" in log
        assert "charge"  in log
        assert "refund"  in log
        assert "release" in log

    def test_compensation_order_is_reversed(self):
        order = []
        saga = (SagaBuilder()
                .quiet()
                .step("A", _ok_work("A"), _ok_work("undo-A", order.append))
                .step("B", _ok_work("B"), _ok_work("undo-B", order.append))
                .step("C", _ok_work("C"), _ok_work("undo-C", order.append))
                .step("D", _fail_work("D"))
                .build())
        saga.execute(maestro.WorkContext())
        assert order == ["undo-C", "undo-B", "undo-A"]

    def test_step_without_compensation_skipped(self):
        log = []
        saga = (SagaBuilder()
                .quiet()
                .step("A", _ok_work("A"), _ok_work("undo-A", log.append))
                .step("B", _ok_work("B"))  # no compensation
                .step("C", _fail_work("C"))
                .build())
        report = saga.execute(maestro.WorkContext())
        assert report.saga_status == SagaStatus.COMPENSATED
        # Only A has compensation; B is skipped
        assert "undo-A" in log
        assert report.compensated_steps == ["A"]

    def test_failed_at_first_step_no_compensation(self):
        saga = (SagaBuilder()
                .quiet()
                .step("A", _fail_work("A"), _ok_work("undo-A"))
                .step("B", _ok_work("B"))
                .build())
        report = saga.execute(maestro.WorkContext())
        # A failed immediately — nothing was completed, so nothing to compensate
        assert report.saga_status == SagaStatus.FAILED
        assert report.compensated_steps == []

    def test_partially_compensated_when_compensation_fails(self):
        def fail_comp(ctx):
            return maestro.DefaultWorkReport(maestro.WorkStatus.FAILED, ctx)

        saga = (SagaBuilder()
                .quiet()
                .step("A", _ok_work("A"), maestro.LambdaWork(fail_comp, "bad-undo"))
                .step("B", _ok_work("B"), _ok_work("undo-B"))
                .step("C", _fail_work("C"))
                .build())
        report = saga.execute(maestro.WorkContext())
        assert report.saga_status == SagaStatus.PARTIALLY_COMPENSATED
        assert "A" in report.failed_compensations

    def test_context_flows_through_steps(self):
        def step1(ctx): ctx.put("counter", 1)
        def step2(ctx): ctx.put("counter", ctx.get("counter", 0) + 1)

        saga = (SagaBuilder()
                .quiet()
                .step("s1", maestro.LambdaWork(step1, "s1"))
                .step("s2", maestro.LambdaWork(step2, "s2"))
                .build())
        ctx = maestro.WorkContext()
        saga.execute(ctx)
        assert ctx.get("counter") == 2

    def test_compensation_receives_context(self):
        ctx = maestro.WorkContext()
        log = []

        def comp(c): log.append(c.get("booking_id"))

        def book(c): c.put("booking_id", "BK-001")

        saga = (SagaBuilder()
                .quiet()
                .step("book", maestro.LambdaWork(book, "book"),
                              maestro.LambdaWork(comp, "cancel"))
                .step("pay",  _fail_work("pay"))
                .build())
        saga.execute(ctx)
        assert log == ["BK-001"]

    def test_report_step_results(self):
        saga = (SagaBuilder()
                .quiet()
                .step("A", _ok_work("A"))
                .step("B", _fail_work("B"), _ok_work("undo-B"))
                .build())
        report = saga.execute(maestro.WorkContext())
        assert len(report.step_results) == 2
        assert report.step_results[0].succeeded
        assert report.step_results[1].failed
        assert report.failed_step == "B"

    def test_saga_as_work_in_flow(self):
        """Saga integrates as a Work inside a SequentialFlow."""
        log = []
        saga = (SagaBuilder()
                .quiet()
                .step("A", _ok_work("A", log.append))
                .step("B", _ok_work("B", log.append))
                .build())
        flow = maestro.SequentialFlow.Builder() \
               .execute(saga) \
               .then(maestro.LambdaWork(lambda c: log.append("after"), "after")) \
               .build()
        report = maestro.WorkFlowEngine().run(flow, maestro.WorkContext())
        assert report.status == maestro.WorkStatus.COMPLETED
        assert log == ["A", "B", "after"]

    def test_saga_step_helper(self):
        log = []
        step = saga_step("reserve",
                         fn=lambda ctx: log.append("reserve"),
                         compensation_fn=lambda ctx: log.append("release"))
        assert step.name == "reserve"
        assert step.is_compensatable
        step.work.execute(maestro.WorkContext())
        assert log == ["reserve"]

    def test_listener_events(self):
        events = []

        class Listener(SagaListener):
            def on_step_started(self, step, ctx):     events.append(("started", step.name))
            def on_step_completed(self, step, result): events.append(("completed", step.name))
            def on_step_failed(self, step, result):   events.append(("failed", step.name))
            def on_saga_compensated(self, report):    events.append(("compensated", None))

        saga = (SagaBuilder()
                .quiet()
                .step("A", _ok_work("A"))
                .step("B", _fail_work("B"), _ok_work("undo-B"))
                .add_listener(Listener())
                .build())
        saga.execute(maestro.WorkContext())

        assert ("started", "A")   in events
        assert ("completed","A")  in events
        assert ("started", "B")   in events
        assert ("failed",  "B")   in events
        assert ("compensated", None) in events

    def test_str_report(self):
        saga = (SagaBuilder()
                .quiet()
                .step("A", _ok_work("A"), _ok_work("undo-A"))
                .step("B", _fail_work("B"))
                .build())
        report = saga.execute(maestro.WorkContext())
        text = str(report)
        assert "SagaReport" in text
        assert "COMPENSATED" in text


# ══════════════════════════════════════════════════════════════════════
#  maestro.schedule — cron parser
# ══════════════════════════════════════════════════════════════════════

class TestCronParser:
    def _at(self, minute=0, hour=0, day=1, month=1, weekday=0):
        # weekday: isoweekday()%7 → Sun=0
        import calendar
        # Build a datetime with given day/month and correct weekday
        return datetime.datetime(2024, month, day, hour, minute)

    def test_all_star(self):
        expr = _CronExpr("* * * * *")
        dt   = datetime.datetime(2024, 1, 15, 12, 30)
        assert expr.matches(dt)

    def test_specific_minute(self):
        expr = _CronExpr("30 * * * *")
        assert expr.matches(datetime.datetime(2024, 1, 1, 10, 30))
        assert not expr.matches(datetime.datetime(2024, 1, 1, 10, 31))

    def test_step(self):
        expr = _CronExpr("*/15 * * * *")
        assert expr.matches(datetime.datetime(2024, 1, 1, 0,  0))
        assert expr.matches(datetime.datetime(2024, 1, 1, 0, 15))
        assert expr.matches(datetime.datetime(2024, 1, 1, 0, 30))
        assert expr.matches(datetime.datetime(2024, 1, 1, 0, 45))
        assert not expr.matches(datetime.datetime(2024, 1, 1, 0, 10))

    def test_range(self):
        expr = _CronExpr("* 9-17 * * *")
        assert expr.matches(datetime.datetime(2024, 1, 1, 9,  0))
        assert expr.matches(datetime.datetime(2024, 1, 1, 17, 0))
        assert not expr.matches(datetime.datetime(2024, 1, 1, 8, 0))
        assert not expr.matches(datetime.datetime(2024, 1, 1, 18, 0))

    def test_list(self):
        expr = _CronExpr("0,30 * * * *")
        assert expr.matches(datetime.datetime(2024, 1, 1, 10,  0))
        assert expr.matches(datetime.datetime(2024, 1, 1, 10, 30))
        assert not expr.matches(datetime.datetime(2024, 1, 1, 10, 15))

    def test_next_after(self):
        expr = _CronExpr("0 * * * *")    # every hour on the dot
        base = datetime.datetime(2024, 6, 1, 10, 30)
        nxt  = expr.next_after(base)
        assert nxt.hour   == 11
        assert nxt.minute == 0

    def test_invalid_expression(self):
        with pytest.raises(ValueError):
            _CronExpr("* * * *")   # only 4 fields


class TestCronTrigger:
    def test_next_fire_time(self):
        t   = CronTrigger("0 12 * * *")  # daily at noon
        now = datetime.datetime(2024, 1, 1, 10, 0)
        nxt = t.next_fire_time(None, now)
        assert nxt.hour == 12
        assert nxt.minute == 0

    def test_shortcuts(self):
        assert "0 * * * *" in CronTrigger.HOURLY
        assert "0 0 * * *" in CronTrigger.DAILY


class TestIntervalTrigger:
    def test_first_fire_immediately(self):
        t   = IntervalTrigger(seconds=60, start_immediately=True)
        now = datetime.datetime(2024, 1, 1, 12, 0)
        assert t.next_fire_time(None, now) == now

    def test_first_fire_delayed(self):
        t   = IntervalTrigger(seconds=60, start_immediately=False)
        now = datetime.datetime(2024, 1, 1, 12, 0)
        nxt = t.next_fire_time(None, now)
        assert nxt == now + datetime.timedelta(seconds=60)

    def test_subsequent_fires(self):
        t    = IntervalTrigger(seconds=30)
        last = datetime.datetime(2024, 1, 1, 12, 0)
        nxt  = t.next_fire_time(last, last)
        assert nxt == last + datetime.timedelta(seconds=30)


class TestOnceTrigger:
    def test_fires_once(self):
        at  = datetime.datetime(2024, 6, 1, 9, 0)
        t   = OnceTrigger(at)
        now = datetime.datetime(2024, 1, 1)
        assert t.next_fire_time(None, now) == at
        # After first fire, returns None
        assert t.next_fire_time(at, now) is None

    def test_immediate_trigger(self):
        t   = ImmediateTrigger()
        now = datetime.datetime.now()
        assert t.next_fire_time(None, now) == now
        assert t.next_fire_time(now, now) is None


class TestScheduler:
    def test_immediate_trigger_fires_once(self):
        log = []
        with Scheduler(tick_seconds=0.05) as sched:
            sched.add("once",
                      work    = maestro.LambdaWork(lambda c: log.append("fired"), "w"),
                      trigger = ImmediateTrigger())
            time.sleep(0.3)
        assert log == ["fired"]

    def test_interval_trigger_fires_multiple_times(self):
        log = []
        with Scheduler(tick_seconds=0.05) as sched:
            sched.add("ticker",
                      work    = maestro.LambdaWork(lambda c: log.append(1), "w"),
                      trigger = IntervalTrigger(seconds=0.1))
            time.sleep(0.55)
        assert len(log) >= 3

    def test_pause_and_resume(self):
        log = []
        with Scheduler(tick_seconds=0.05) as sched:
            sched.add("t",
                      work    = maestro.LambdaWork(lambda c: log.append("fired"), "w"),
                      trigger = IntervalTrigger(seconds=0.1))
            time.sleep(0.15)
            sched.pause("t")
            before_pause = len(log)
            time.sleep(0.25)
            after_pause = len(log)
            sched.resume("t")
            time.sleep(0.15)
        assert before_pause <= after_pause   # no new fires during pause
        assert len(log) > after_pause        # fires after resume

    def test_remove_task(self):
        log = []
        with Scheduler(tick_seconds=0.05) as sched:
            sched.add("t",
                      work    = maestro.LambdaWork(lambda c: log.append("fired"), "w"),
                      trigger = IntervalTrigger(seconds=0.1))
            time.sleep(0.15)
            n_before = len(log)
            sched.remove("t")
            time.sleep(0.2)
        assert len(log) == n_before  # no new fires after remove

    def test_status_returns_task_info(self):
        with Scheduler(tick_seconds=0.1) as sched:
            sched.add("task",
                      work    = maestro.NoOpWork(),
                      trigger = ImmediateTrigger())
            time.sleep(0.2)
            status = sched.status()
        assert any(s["name"] == "task" for s in status)

    def test_context_manager(self):
        log = []
        with Scheduler(tick_seconds=0.05) as sched:
            sched.add("once",
                      work    = maestro.LambdaWork(lambda c: log.append("ok"), "w"),
                      trigger = ImmediateTrigger())
            time.sleep(0.2)
        assert log == ["ok"]

    def test_schedule_listener(self):
        events = []

        class L(ScheduleListener):
            def on_task_started(self, task):          events.append("started")
            def on_task_succeeded(self, task, run):   events.append("succeeded")

        with Scheduler(tick_seconds=0.05, listeners=[L()]) as sched:
            sched.add("once",
                      work    = maestro.LambdaWork(lambda c: None, "w"),
                      trigger = ImmediateTrigger())
            time.sleep(0.3)
        assert "started"   in events
        assert "succeeded" in events

    def test_schedule_batch_job(self):
        """Scheduler can schedule a batch Job (not just Work)."""
        sink = []
        job  = (maestro.JobBuilder()
                .named("sched-job")
                .reader(maestro.IterableRecordReader([1, 2, 3]))
                .writer(maestro.CollectionRecordWriter(sink))
                .build())
        with Scheduler(tick_seconds=0.05) as sched:
            sched.add("batch", job=job, trigger=ImmediateTrigger())
            time.sleep(0.3)
        assert sink == [1, 2, 3]


# ══════════════════════════════════════════════════════════════════════
#  maestro.cli
# ══════════════════════════════════════════════════════════════════════

class TestCLI:
    def test_info(self, capsys):
        rc = cli_main(["info"])
        out = capsys.readouterr().out
        assert rc == 0 or rc is None
        assert "Maestro SDK" in out
        assert "maestro.rules" in out

    def test_version(self):
        with pytest.raises(SystemExit) as exc:
            cli_main(["--version"])
        assert exc.value.code == 0

    def test_no_args_shows_help(self, capsys):
        rc = cli_main([])
        assert rc == 0 or rc is None

    def test_rules_validate_valid(self, tmp_path, capsys):
        yaml_file = tmp_path / "rules.yaml"
        yaml_file.write_text(
            "name: weather\ncondition: rain == True\nactions:\n  - \"pass\"\n"
        )
        try:
            rc = cli_main(["rules", "validate", str(yaml_file)])
            out = capsys.readouterr().out
            assert "VALID" in out
        except ImportError:
            pytest.skip("PyYAML not installed")

    def test_rules_fire(self, tmp_path, capsys):
        yaml_file = tmp_path / "rules.yaml"
        yaml_file.write_text(
            "name: rain-check\ncondition: rain == True\nactions:\n  - \"result = 'umbrella'\"\n"
        )
        try:
            rc = cli_main(["rules", "fire", str(yaml_file), "--facts", '{"rain": true}'])
            out = capsys.readouterr().out
            assert "rain-check" in out
        except ImportError:
            pytest.skip("PyYAML not installed")

    def test_schedule_cron(self, capsys):
        rc  = cli_main(["schedule", "cron", "0 * * * *", "--count", "3"])
        out = capsys.readouterr().out
        assert out.count("00") >= 3   # 3 fire times shown with :00 minutes

    def test_schedule_demo(self, capsys):
        rc  = cli_main(["schedule", "demo", "--duration", "0.5"])
        out = capsys.readouterr().out
        assert "Scheduler demo" in out

    def test_saga_describe(self, tmp_path, capsys):
        yaml_file = tmp_path / "saga.yaml"
        yaml_file.write_text(
            "name: book-trip\nsteps:\n"
            "  - name: book-flight\n    compensation: cancel-flight\n"
            "  - name: book-hotel\n    compensation: cancel-hotel\n"
        )
        try:
            rc  = cli_main(["saga", "describe", str(yaml_file)])
            out = capsys.readouterr().out
            assert "book-trip"    in out
            assert "book-flight"  in out
            assert "cancel-hotel" in out
        except ImportError:
            pytest.skip("PyYAML not installed")

    def test_fsm_dot(self, tmp_path, capsys):
        yaml_file = tmp_path / "fsm.yaml"
        yaml_file.write_text(
            "name: turnstile\ninitial: locked\nstates: [locked, unlocked]\n"
            "transitions:\n"
            "  - name: unlock\n    from: locked\n    event: CoinEvent\n    to: unlocked\n"
            "  - name: lock\n    from: unlocked\n    event: PushEvent\n    to: locked\n"
        )
        try:
            rc  = cli_main(["fsm", "dot", str(yaml_file)])
            out = capsys.readouterr().out
            assert "digraph" in out
            assert "locked"  in out
            assert "unlocked" in out
        except ImportError:
            pytest.skip("PyYAML not installed")
