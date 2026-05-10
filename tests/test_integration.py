"""
tests/test_integration.py — Maestro SDK integration tests.

These tests exercise the maestro.integration bridges: combinations of
rules + flows, batch + flows, states + flows, and batch + rules.

Run: cd maestro_sdk && python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
import pytest

import maestro
from maestro.integration import (
    RuleSetWork, BatchWork, FSMGuardWork, FSMTransitionWork,
    RuleBasedFilter, RuleBasedProcessor,
)


# ─────────────────────────────────────────────────────────────────── #
#  Helpers                                                            #
# ─────────────────────────────────────────────────────────────────── #

def _rule(name, cond, then_fn=None):
    b = maestro.RuleBuilder().name(name).when(cond)
    if then_fn: b = b.then(then_fn)
    return b.build()


def _ok_work(name="ok"):
    return maestro.LambdaWork(lambda ctx: None, name=name)


# ─────────────────────────────────────────────────────────────────── #
#  RuleSetWork — flows + rules                                        #
# ─────────────────────────────────────────────────────────────────── #

class TestRuleSetWork:
    def test_context_populated_before_rules(self):
        """Rules can read values placed in WorkContext."""
        log = []
        r = _rule("check", lambda f: f.get("x", 0) > 5, lambda f: log.append("fired"))
        work = RuleSetWork(rules=maestro.Rules(r), name="check-x")

        ctx = maestro.WorkContext(x=10)
        report = work.execute(ctx)
        assert report.status == maestro.WorkStatus.COMPLETED
        assert log == ["fired"]

    def test_fact_changes_written_back_to_context(self):
        """Actions that mutate Facts → changes appear in WorkContext."""
        r = _rule("set-tier", lambda f: f.get("total", 0) > 1000,
                  lambda f: f.put("tier", "vip"))
        work = RuleSetWork(rules=maestro.Rules(r))

        ctx = maestro.WorkContext(total=1500)
        work.execute(ctx)
        assert ctx.get("tier") == "vip"

    def test_completed_when_rule_fires(self):
        r = _rule("always", lambda f: True)
        report = RuleSetWork(maestro.Rules(r)).execute(maestro.WorkContext())
        assert report.status == maestro.WorkStatus.COMPLETED

    def test_completed_when_no_rule_fires_by_default(self):
        r = _rule("never", lambda f: False)
        report = RuleSetWork(maestro.Rules(r)).execute(maestro.WorkContext())
        assert report.status == maestro.WorkStatus.COMPLETED  # default: don't fail

    def test_failed_when_no_rule_fires_and_flag_set(self):
        r = _rule("never", lambda f: False)
        work = RuleSetWork(maestro.Rules(r), fail_when_no_rule_fired=True)
        report = work.execute(maestro.WorkContext())
        assert report.status == maestro.WorkStatus.FAILED

    def test_inside_sequential_flow(self):
        """RuleSetWork chained in a SequentialFlow."""
        log = []
        r = _rule("log", lambda f: True, lambda f: log.append("rule-fired"))
        flow = (maestro.SequentialFlow.Builder()
                .execute(RuleSetWork(maestro.Rules(r), name="rule-step"))
                .then(maestro.LambdaWork(lambda ctx: log.append("next-step")))
                .build())
        maestro.WorkFlowEngine().run(flow, maestro.WorkContext())
        assert log == ["rule-fired", "next-step"]

    def test_conditional_flow_branches_on_rule_result(self):
        """ConditionalFlow uses RuleSetWork result to decide branch."""
        log = []
        never_rule = _rule("never", lambda f: False)
        work = RuleSetWork(maestro.Rules(never_rule), fail_when_no_rule_fired=True)

        flow = (maestro.ConditionalFlow.Builder()
                .execute(work)
                .when(maestro.WorkReportPredicate.COMPLETED)
                .then(maestro.LambdaWork(lambda ctx: log.append("then")))
                .otherwise(maestro.LambdaWork(lambda ctx: log.append("otherwise")))
                .build())
        maestro.WorkFlowEngine().run(flow, maestro.WorkContext())
        assert log == ["otherwise"]


# ─────────────────────────────────────────────────────────────────── #
#  BatchWork — flows + batch                                          #
# ─────────────────────────────────────────────────────────────────── #

class TestBatchWork:
    def _make_job(self, data, sink):
        return (maestro.JobBuilder()
                .named("test-job")
                .reader(maestro.IterableRecordReader(data))
                .writer(maestro.CollectionRecordWriter(sink))
                .build())

    def test_completed_on_success(self):
        sink = []
        job  = self._make_job([1, 2, 3], sink)
        work = BatchWork(job)
        report = work.execute(maestro.WorkContext())
        assert report.status == maestro.WorkStatus.COMPLETED
        assert sink == [1, 2, 3]

    def test_batch_report_stored_in_context(self):
        sink = []
        job  = self._make_job(range(5), sink)
        ctx  = maestro.WorkContext()
        BatchWork(job, report_key="my_report").execute(ctx)
        assert ctx.contains("my_report")
        assert ctx.get("my_report").metrics.written_count == 5

    def test_inside_sequential_flow(self):
        log = []
        sink = []
        job = self._make_job(["x", "y"], sink)
        flow = (maestro.SequentialFlow.Builder()
                .execute(BatchWork(job, name="etl"))
                .then(maestro.LambdaWork(lambda ctx: log.append(ctx.get("batch_report").metrics.written_count)))
                .build())
        maestro.WorkFlowEngine().run(flow, maestro.WorkContext())
        assert sink == ["x", "y"]
        assert log == [2]

    def test_failed_job_returns_failed_work_report(self):
        def explode(p): raise RuntimeError("boom")

        job = (maestro.JobBuilder()
               .named("fail-job")
               .reader(maestro.IterableRecordReader([1]))
               .processor(maestro.LambdaRecordProcessor(explode))
               .error_threshold(0)
               .writer(maestro.DevNullRecordWriter())
               .build())
        report = BatchWork(job).execute(maestro.WorkContext())
        assert report.status == maestro.WorkStatus.FAILED


# ─────────────────────────────────────────────────────────────────── #
#  FSMGuardWork — flows + states                                      #
# ─────────────────────────────────────────────────────────────────── #

class TestFSMGuardWork:
    def _turnstile(self):
        locked   = maestro.State("locked")
        unlocked = maestro.State("unlocked")
        class Coin(maestro.Event): pass
        class Push(maestro.Event): pass
        fsm = (maestro.FiniteStateMachineBuilder(states={locked, unlocked}, initial_state=locked)
               .register_transition(maestro.TransitionBuilder().source_state(locked).event_type(Coin).target_state(unlocked).build())
               .register_transition(maestro.TransitionBuilder().source_state(unlocked).event_type(Push).target_state(locked).build())
               .register_transition(maestro.TransitionBuilder().source_state(locked).event_type(Push).target_state(locked).build())
               .build())
        return fsm, locked, unlocked, Coin, Push

    def test_completed_when_in_success_state(self):
        fsm, locked, unlocked, Coin, Push = self._turnstile()
        guard = FSMGuardWork(fsm, Coin(), success_states={unlocked})
        report = guard.execute(maestro.WorkContext())
        assert report.status == maestro.WorkStatus.COMPLETED

    def test_failed_when_in_error_state(self):
        fsm, locked, unlocked, Coin, Push = self._turnstile()
        # locked after push = error
        guard = FSMGuardWork(fsm, Push(), error_states={locked})
        report = guard.execute(maestro.WorkContext())
        assert report.status == maestro.WorkStatus.FAILED

    def test_context_updated_with_state_name(self):
        fsm, locked, unlocked, Coin, Push = self._turnstile()
        ctx = maestro.WorkContext()
        FSMGuardWork(fsm, Coin()).execute(ctx)
        assert ctx.get("fsm_state") == "unlocked"

    def test_failed_on_undefined_transition(self):
        fsm, locked, unlocked, Coin, Push = self._turnstile()
        class Unknown(maestro.Event): pass
        guard = FSMGuardWork(fsm, Unknown())
        report = guard.execute(maestro.WorkContext())
        assert report.status == maestro.WorkStatus.FAILED

    def test_dynamic_event_from_context(self):
        """Event computed at runtime from WorkContext."""
        fsm, locked, unlocked, Coin, Push = self._turnstile()
        def make_event(ctx):
            return Coin() if ctx.get("has_coin") else Push()
        guard = FSMGuardWork(fsm, make_event, success_states={unlocked})
        report = guard.execute(maestro.WorkContext(has_coin=True))
        assert report.status == maestro.WorkStatus.COMPLETED

    def test_inside_conditional_flow(self):
        """FSMGuardWork drives a ConditionalFlow branch."""
        log = []
        fsm, locked, unlocked, Coin, Push = self._turnstile()
        guard = FSMGuardWork(fsm, Coin(), success_states={unlocked})
        flow = (maestro.ConditionalFlow.Builder()
                .execute(guard)
                .when(maestro.WorkReportPredicate.COMPLETED)
                .then(maestro.LambdaWork(lambda ctx: log.append("unlocked!"), name="ok"))
                .otherwise(maestro.LambdaWork(lambda ctx: log.append("still locked"), name="nok"))
                .build())
        maestro.WorkFlowEngine().run(flow, maestro.WorkContext())
        assert log == ["unlocked!"]


# ─────────────────────────────────────────────────────────────────── #
#  FSMTransitionWork — flows + states (sequence of events)           #
# ─────────────────────────────────────────────────────────────────── #

class TestFSMTransitionWork:
    def _order_fsm(self):
        pending, paid, shipped = maestro.State("PENDING"), maestro.State("PAID"), maestro.State("SHIPPED")
        class Pay(maestro.Event): pass
        class Ship(maestro.Event): pass
        class Cancel(maestro.Event): pass
        fsm = (maestro.FiniteStateMachineBuilder(states={pending, paid, shipped}, initial_state=pending)
               .register_transition(maestro.TransitionBuilder().source_state(pending).event_type(Pay).target_state(paid).build())
               .register_transition(maestro.TransitionBuilder().source_state(paid).event_type(Ship).target_state(shipped).build())
               .build())
        return fsm, pending, paid, shipped, Pay, Ship, Cancel

    def test_fires_all_events_in_sequence(self):
        fsm, _, _, shipped, Pay, Ship, _ = self._order_fsm()
        work = FSMTransitionWork(fsm, [Pay(), Ship()], name="order-flow")
        ctx  = maestro.WorkContext()
        report = work.execute(ctx)
        assert report.status == maestro.WorkStatus.COMPLETED
        assert fsm.current_state == shipped
        assert ctx.get("fsm_state") == "SHIPPED"

    def test_history_recorded(self):
        fsm, _, _, _, Pay, Ship, _ = self._order_fsm()
        work = FSMTransitionWork(fsm, [Pay(), Ship()])
        ctx  = maestro.WorkContext()
        work.execute(ctx)
        assert ctx.get("fsm_history") == ["PAID", "SHIPPED"]

    def test_failed_on_undefined_transition(self):
        fsm, _, _, _, Pay, Ship, Cancel = self._order_fsm()
        work = FSMTransitionWork(fsm, [Pay(), Cancel()]) # Cancel not registered from PAID
        ctx  = maestro.WorkContext()
        report = work.execute(ctx)
        assert report.status == maestro.WorkStatus.FAILED


# ─────────────────────────────────────────────────────────────────── #
#  RuleBasedFilter — batch + rules                                    #
# ─────────────────────────────────────────────────────────────────── #

class TestRuleBasedFilter:
    def test_accepts_matching_records(self):
        rule = _rule("adult", lambda f: int(f.get("age", 0)) >= 18)
        filt = RuleBasedFilter(maestro.Rules(rule))

        from maestro.batch._record import Header, Record
        accept = Record(Header(1, "test"), {"name": "Alice", "age": 25})
        reject = Record(Header(2, "test"), {"name": "Bob",   "age": 15})

        assert filt.accept(accept) is True
        assert filt.accept(reject) is False

    def test_scalar_payload(self):
        rule = _rule("positive", lambda f: f.get("value", 0) > 0)
        filt = RuleBasedFilter(maestro.Rules(rule))

        from maestro.batch._record import Header, Record
        assert filt.accept(Record(Header(1, "t"), 5))  is True
        assert filt.accept(Record(Header(2, "t"), -1)) is False

    def test_in_batch_pipeline(self):
        """RuleBasedFilter integrates cleanly with JobBuilder."""
        age_rule = _rule("adult", lambda f: int(f.get("age", 0)) >= 18)
        filt     = RuleBasedFilter(maestro.Rules(age_rule))

        data = [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 15},
                {"name": "Carol", "age": 30}]
        sink = []
        job  = (maestro.JobBuilder()
                .named("filter-adults")
                .reader(maestro.IterableRecordReader(data))
                .filter(filt)
                .writer(maestro.CollectionRecordWriter(sink))
                .build())
        report = maestro.JobExecutor().execute(job)
        assert len(sink) == 2
        assert {r["name"] for r in sink} == {"Alice", "Carol"}
        assert report.metrics.filtered_count == 1

    def test_dataclass_payload(self):
        @dataclass
        class Person:
            name: str
            age: int

        rule = _rule("adult", lambda f: f.get("age", 0) >= 18)
        filt = RuleBasedFilter(maestro.Rules(rule))

        from maestro.batch._record import Header, Record
        assert filt.accept(Record(Header(1, "t"), Person("Alice", 25))) is True
        assert filt.accept(Record(Header(2, "t"), Person("Bob", 15)))   is False


# ─────────────────────────────────────────────────────────────────── #
#  RuleBasedProcessor — batch + rules                                 #
# ─────────────────────────────────────────────────────────────────── #

class TestRuleBasedProcessor:
    def test_enriches_dict_payload(self):
        r = _rule("tag-vip", lambda f: f.get("total", 0) > 500,
                  lambda f: f.put("tier", "vip"))
        proc = RuleBasedProcessor(maestro.Rules(r))

        from maestro.batch._record import Header, Record
        record = Record(Header(1, "t"), {"name": "Alice", "total": 700})
        result = proc.process_record(record)
        assert result.payload["tier"] == "vip"

    def test_no_change_when_rule_doesnt_fire(self):
        r = _rule("tag-vip", lambda f: f.get("total", 0) > 500,
                  lambda f: f.put("tier", "vip"))
        proc = RuleBasedProcessor(maestro.Rules(r))

        from maestro.batch._record import Header, Record
        record = Record(Header(1, "t"), {"name": "Bob", "total": 100})
        result = proc.process_record(record)
        assert "tier" not in result.payload

    def test_fails_when_no_match_and_flag_set(self):
        r = _rule("never", lambda f: False)
        proc = RuleBasedProcessor(maestro.Rules(r), fail_on_no_match=True)

        from maestro.batch._record import Header, Record
        record = Record(Header(1, "t"), {"x": 1})
        with pytest.raises(maestro.RecordProcessingException):
            proc.process_record(record)

    def test_in_batch_pipeline_enriches_records(self):
        """RuleBasedProcessor enriches records in a full batch job."""
        tier_rule = _rule("set-tier",
                          lambda f: f.get("total", 0) > 1000,
                          lambda f: f.put("tier", "gold"))
        proc = RuleBasedProcessor(maestro.Rules(tier_rule))

        data = [{"id": 1, "total": 1500}, {"id": 2, "total": 200}, {"id": 3, "total": 2000}]
        sink = []
        job  = (maestro.JobBuilder()
                .named("enrich")
                .reader(maestro.IterableRecordReader(data))
                .processor(proc)
                .writer(maestro.CollectionRecordWriter(sink))
                .build())
        maestro.JobExecutor().execute(job)
        tiers = {r["id"]: r.get("tier") for r in sink}
        assert tiers[1] == "gold"
        assert tiers[2] is None
        assert tiers[3] == "gold"


# ─────────────────────────────────────────────────────────────────── #
#  Full cross-module scenario: order processing                       #
# ─────────────────────────────────────────────────────────────────── #

class TestFullCrossModuleScenario:
    """
    End-to-end: batch ETL → rules enrichment → FSM state guard → flow orchestration.
    Models a simplified order processing pipeline.
    """

    def test_order_pipeline(self):
        """
        1. Batch reads raw order dicts.
        2. RuleBasedProcessor tags VIP orders (total > 500).
        3. FSM validates order state transitions (PENDING → PAID → SHIPPED).
        4. Flow orchestrates steps 1–3 sequentially.
        5. Final sink contains only orders that passed all gates.
        """
        # ── States ──────────────────────────────────────────────────
        pending  = maestro.State("PENDING")
        paid     = maestro.State("PAID")
        shipped  = maestro.State("SHIPPED")
        rejected = maestro.State("REJECTED")

        class PayEvent(maestro.Event): pass
        class RejectEvent(maestro.Event): pass

        fsm = (maestro.FiniteStateMachineBuilder(
                   states={pending, paid, shipped, rejected},
                   initial_state=pending)
               .register_transition(
                   maestro.TransitionBuilder().source_state(pending).event_type(PayEvent).target_state(paid).build())
               .register_transition(
                   maestro.TransitionBuilder().source_state(pending).event_type(RejectEvent).target_state(rejected).build())
               .build())

        # ── Batch: ETL + enrichment ──────────────────────────────────
        tier_rule = _rule("vip", lambda f: f.get("total", 0) > 500,
                          lambda f: f.put("tier", "vip"))
        tier_proc = RuleBasedProcessor(maestro.Rules(tier_rule))

        orders_raw = [
            {"id": 1, "total": 750},
            {"id": 2, "total": 200},
            {"id": 3, "total": 1100},
        ]
        enriched_sink = []
        etl_job = (maestro.JobBuilder()
                   .named("enrich-orders")
                   .reader(maestro.IterableRecordReader(orders_raw))
                   .processor(tier_proc)
                   .writer(maestro.CollectionRecordWriter(enriched_sink))
                   .build())

        # ── Integration: BatchWork + FSMGuardWork + RuleSetWork ──────
        log_tiers = []
        log_rule  = []

        review_rules = maestro.Rules(
            _rule("log-tier", lambda f: f.get("tier") == "vip",
                  lambda f: log_rule.append("vip-detected"))
        )

        flow = (maestro.SequentialFlow.Builder()
                .named("order-pipeline")
                # Step 1: run the ETL batch job
                .execute(BatchWork(etl_job, name="etl-step", report_key="etl_report"))
                # Step 2: fire rules on enriched context metadata
                .then(RuleSetWork(review_rules, name="review-step"))
                # Step 3: FSM guard — fire PayEvent, expect PAID state
                .then(FSMGuardWork(fsm, PayEvent(), success_states={paid}, name="payment-gate"))
                .build())

        ctx = maestro.WorkContext(tier="vip")  # simulates context from previous steps
        report = maestro.WorkFlowEngine().run(flow, ctx)

        assert report.status == maestro.WorkStatus.COMPLETED
        assert len(enriched_sink) == 3
        assert enriched_sink[0]["tier"] == "vip"   # id 1: total 750
        assert "tier" not in enriched_sink[1]       # id 2: total 200
        assert enriched_sink[2]["tier"] == "vip"    # id 3: total 1100
        assert log_rule == ["vip-detected"]
        assert ctx.get("fsm_state") == "PAID"
        assert fsm.current_state == paid
