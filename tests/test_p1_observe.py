"""tests/test_p1_observe.py — observability module tests"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import maestro
pytestmark = pytest.mark.core

from maestro.observe import (
    MetricEvent, InMemoryObserver, LoggingObserver, CompositeObserver,
    MaestroObserver, timed,
)


# ── InMemoryObserver ──────────────────────────────────────────────────────── #

class TestInMemoryObserver:
    def test_records_counter_events(self):
        obs = InMemoryObserver()
        obs.on_event(MetricEvent("rules", "rule_fired", 1.0, {}, kind="counter"))
        obs.on_event(MetricEvent("rules", "rule_fired", 1.0, {}, kind="counter"))
        assert obs.counter("rules", "rule_fired") == 2.0

    def test_records_gauge_events(self):
        obs = InMemoryObserver()
        obs.on_event(MetricEvent("batch", "records_written", 42.0, {}, kind="gauge"))
        assert obs.gauge("batch", "records_written") == 42.0

    def test_histogram_aggregation(self):
        obs = InMemoryObserver()
        for v in [0.1, 0.2, 0.3]:
            obs.on_event(MetricEvent("flows", "duration", v, {}, kind="histogram"))
        snap = obs.histogram("flows", "duration")
        assert snap["count"] == 3
        assert abs(snap["sum"] - 0.6) < 1e-9
        assert abs(snap["mean"] - 0.2) < 1e-9

    def test_labels_differentiate_metrics(self):
        obs = InMemoryObserver()
        obs.on_event(MetricEvent("rules", "rule_fired", 3.0, {"rule": "r1"}, kind="counter"))
        obs.on_event(MetricEvent("rules", "rule_fired", 7.0, {"rule": "r2"}, kind="counter"))
        assert obs.counter("rules", "rule_fired", rule="r1") == 3.0
        assert obs.counter("rules", "rule_fired", rule="r2") == 7.0

    def test_events_filter(self):
        obs = InMemoryObserver()
        obs.on_event(MetricEvent("rules", "fired", 1.0, {}, kind="counter"))
        obs.on_event(MetricEvent("batch", "written", 5.0, {}, kind="gauge"))
        rules_evs = obs.events(module="rules")
        assert len(rules_evs) == 1 and rules_evs[0].module == "rules"

    def test_clear_resets_all(self):
        obs = InMemoryObserver()
        obs.on_event(MetricEvent("rules", "fired", 1.0, {}, kind="counter"))
        obs.clear()
        assert obs.counter("rules", "fired") == 0.0
        assert obs.events() == []

    def test_summary_string(self):
        obs = InMemoryObserver()
        obs.on_event(MetricEvent("rules", "rule_fired", 5.0, {}, kind="counter"))
        summary = obs.summary()
        assert "rule_fired" in summary
        assert "5" in summary

    def test_prometheus_export(self):
        obs = InMemoryObserver()
        obs.on_event(MetricEvent("rules", "fired", 10.0, {}, kind="counter"))
        prom = obs.export_prometheus()
        assert "maestro_rules_fired_total" in prom
        assert "10" in prom


# ── CompositeObserver ─────────────────────────────────────────────────────── #

class TestCompositeObserver:
    def test_fans_out_to_all_children(self):
        obs1 = InMemoryObserver()
        obs2 = InMemoryObserver()
        comp = CompositeObserver(obs1, obs2)
        comp.on_event(MetricEvent("rules", "fired", 1.0, {}, kind="counter"))
        assert obs1.counter("rules", "fired") == 1.0
        assert obs2.counter("rules", "fired") == 1.0

    def test_error_in_one_child_doesnt_break_others(self):
        class Broken(InMemoryObserver):
            def on_event(self, event):
                raise RuntimeError("broken")
        good = InMemoryObserver()
        comp = CompositeObserver(Broken(), good)
        comp.on_event(MetricEvent("x", "y", 1.0, {}, kind="counter"))
        assert good.counter("x", "y") == 1.0


# ── timed context manager ─────────────────────────────────────────────────── #

class TestTimedContextManager:
    def test_records_duration(self):
        obs = InMemoryObserver()
        with timed("flows", "my_step", obs):
            time.sleep(0.01)
        snap = obs.histogram("flows", "my_step")
        assert snap["count"] == 1
        assert snap["mean"] >= 0.005  # at least 5ms

    def test_with_labels(self):
        obs = InMemoryObserver()
        with timed("flows", "step_dur", obs, flow="my-flow"):
            pass
        assert obs.histogram("flows", "step_dur", flow="my-flow")["count"] == 1


# ── MaestroObserver — rules integration ──────────────────────────────────── #

class TestMaestroObserverRules:
    def test_instruments_rules_engine(self):
        obs = InMemoryObserver()
        mo  = MaestroObserver(observers=[obs])

        r = maestro.RuleBuilder().name("always").when(lambda f: True).then(lambda f: None).build()
        engine = mo.instrument_rules_engine(maestro.DefaultRulesEngine())
        engine.fire(maestro.Rules(r), maestro.Facts())

        assert obs.counter("rules", "rule_evaluated", rule="always") >= 1
        assert obs.counter("rules", "rule_fired",     rule="always") >= 1
        assert obs.counter("rules", "engine_fire") >= 1

    def test_failed_rule_counted(self):
        obs = InMemoryObserver()
        mo  = MaestroObserver(observers=[obs])

        def bad_action(f): raise RuntimeError("action error")
        r = maestro.RuleBuilder().name("bad").when(lambda f: True).then(bad_action).build()
        engine = mo.instrument_rules_engine(maestro.DefaultRulesEngine())
        engine.fire(maestro.Rules(r), maestro.Facts())

        assert obs.counter("rules", "rule_action_error", rule="bad") >= 1

    def test_rule_duration_histogram_populated(self):
        obs = InMemoryObserver()
        mo  = MaestroObserver(observers=[obs])
        r = maestro.RuleBuilder().name("timed-rule").when(lambda f: True).then(lambda f: None).build()
        engine = mo.instrument_rules_engine(maestro.DefaultRulesEngine())
        engine.fire(maestro.Rules(r), maestro.Facts())
        snap = obs.histogram("rules", "rule_duration_seconds", rule="timed-rule")
        assert snap["count"] >= 1


# ── MaestroObserver — batch integration ───────────────────────────────────── #

class TestMaestroObserverBatch:
    def test_instruments_job(self):
        obs = InMemoryObserver()
        mo  = MaestroObserver(observers=[obs])
        sink = []
        job = (maestro.JobBuilder()
               .named("obs-test")
               .reader(maestro.IterableRecordReader([1, 2, 3]))
               .writer(maestro.CollectionRecordWriter(sink))
               .build())
        job = mo.instrument_job(job)
        maestro.JobExecutor().execute(job)

        assert obs.counter("batch", "job_started", job="obs-test") >= 1
        assert obs.gauge("batch", "records_written", job="obs-test") == 3
        assert obs.gauge("batch", "records_total",   job="obs-test") == 3
        snap = obs.histogram("batch", "job_duration_seconds", job="obs-test", status="COMPLETED")
        assert snap["count"] == 1

    def test_batch_size_tracked(self):
        obs = InMemoryObserver()
        mo  = MaestroObserver(observers=[obs])
        job = (maestro.JobBuilder()
               .named("batch-size-test")
               .reader(maestro.IterableRecordReader(range(10)))
               .writer(maestro.DevNullRecordWriter())
               .batch_size(3)
               .build())
        mo.instrument_job(job).call()
        snap = obs.histogram("batch", "batch_size")
        assert snap["count"] == 4  # 3+3+3+1


# ── MaestroObserver — flows integration ───────────────────────────────────── #

class TestMaestroObserverFlows:
    def test_instruments_flow(self):
        obs = InMemoryObserver()
        mo  = MaestroObserver(observers=[obs])

        flow = maestro.SequentialFlow.Builder().execute(maestro.NoOpWork()).build()
        observed = mo.instrument_flow(flow)
        observed.execute(maestro.WorkContext())

        evs = obs.events(module="flows", name="flow_duration_seconds")
        assert len(evs) >= 1


# ── MaestroObserver — states integration ──────────────────────────────────── #

class TestMaestroObserverStates:
    def test_instruments_fsm(self):
        obs = InMemoryObserver()
        mo  = MaestroObserver(observers=[obs])

        locked, unlocked = maestro.State("locked"), maestro.State("unlocked")
        class Coin(maestro.Event): pass

        fsm = (maestro.FiniteStateMachineBuilder(states={locked, unlocked}, initial_state=locked)
               .register_transition(
                   maestro.TransitionBuilder().source_state(locked).event_type(Coin).target_state(unlocked).build())
               .build())
        fsm = mo.instrument_fsm(fsm)
        fsm.fire(Coin())

        assert len(obs.events(module="states", name="transition_completed")) >= 1
        assert len(obs.events(module="states", name="state_entered")) >= 1
        assert len(obs.events(module="states", name="transition_duration_seconds")) >= 1

    def test_undefined_transition_counted(self):
        obs = InMemoryObserver()
        mo  = MaestroObserver(observers=[obs])

        locked = maestro.State("locked")
        class Unknown(maestro.Event): pass

        fsm = (maestro.FiniteStateMachineBuilder(states={locked}, initial_state=locked)
               .ignore_undefined_transitions(True)
               .build())
        fsm = mo.instrument_fsm(fsm)
        fsm.fire(Unknown())
        assert len(obs.events(module="states", name="undefined_transition")) >= 1
