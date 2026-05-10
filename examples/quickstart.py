"""
examples/quickstart.py — Maestro SDK complete walkthrough.

Demonstrates all four modules and the integration layer in a single script.
Run: python examples/quickstart.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import concurrent.futures
from dataclasses import dataclass

import maestro
from maestro.integration import (
    RuleSetWork, BatchWork, FSMGuardWork, FSMTransitionWork,
    RuleBasedFilter, RuleBasedProcessor,
)

SEP = "═" * 62


# ════════════════════════════════════════════════════════════════════
#  1.  maestro.rules — declarative rules engine
# ════════════════════════════════════════════════════════════════════
print(SEP); print("1. maestro.rules"); print(SEP)

# Decorator style
@maestro.rule(name="premium discount", priority=1)
class PremiumDiscount:
    @maestro.condition
    def is_premium(self, facts): return facts.get("tier") == "premium"
    @maestro.action
    def apply(self, facts):
        facts.put("discount", 0.20)
        print(f"  → 20% discount for {facts.get('customer')}")

# Builder style
free_shipping = (
    maestro.RuleBuilder()
    .name("free shipping")
    .priority(2)
    .when(lambda f: f.get("total", 0) > 100)
    .then(lambda f: f.put("shipping", 0.0) or print(f"  → Free shipping for {f.get('customer')}"))
    .build()
)

facts  = maestro.Facts(customer="Alice", tier="premium", total=150)
rules  = maestro.Rules(PremiumDiscount(), free_shipping)
engine = maestro.DefaultRulesEngine()
engine.fire(rules, facts)
print(f"  discount={facts.get('discount')}, shipping={facts.get('shipping')}")

# Inference engine
print("\n  Inference engine (count down from 3):")
ctx = maestro.Facts(n=3)
countdown = (maestro.RuleBuilder()
             .name("count")
             .when(lambda f: f.get("n", 0) > 0)
             .then(lambda f: (f.put("n", f.get("n") - 1), print(f"    n={f.get('n')}")) and None)
             .build())
maestro.InferenceRulesEngine().fire(maestro.Rules(countdown), ctx)


# ════════════════════════════════════════════════════════════════════
#  2.  maestro.batch — ETL pipeline
# ════════════════════════════════════════════════════════════════════
print(); print(SEP); print("2. maestro.batch"); print(SEP)

CSV = "id,name,total\n1,Alice,750\n2,Bob,200\n3,Carol,1200"

@dataclass
class Order:
    id: int
    name: str
    total: int

sink: list = []
job = (
    maestro.JobBuilder()
    .named("orders-etl")
    .reader(maestro.StringRecordReader(CSV))
    .filter(maestro.HeaderRecordFilter())
    .mapper(maestro.DelimitedRecordMapper(Order, field_names=["id","name","total"],
                                          type_converters={"id": int, "total": int}))
    .marshaller(maestro.JsonMarshaller())
    .writer(maestro.CollectionRecordWriter(sink))
    .batch_size(10)
    .build()
)
report = maestro.JobExecutor().execute(job)
print(f"  Processed {report.metrics.written_count} orders in {report.metrics.duration_seconds:.3f}s")
for line in sink:
    print(f"  {line}")


# ════════════════════════════════════════════════════════════════════
#  3.  maestro.flows — workflow orchestration
# ════════════════════════════════════════════════════════════════════
print(); print(SEP); print("3. maestro.flows"); print(SEP)

work1 = maestro.LambdaWork(lambda ctx: print("  step 1: validate"), name="validate")
work2 = maestro.LambdaWork(lambda ctx: print("  step 2: enrich"), name="enrich")
work3 = maestro.LambdaWork(lambda ctx: print("  step 3: notify"), name="notify")
work_fail = maestro.LambdaWork(
    lambda ctx: maestro.DefaultWorkReport(maestro.WorkStatus.FAILED, ctx),
    name="fail"
)

# Sequential
print("  Sequential flow:")
maestro.WorkFlowEngine().run(
    maestro.aNewSequentialFlow().execute(work1).then(work2).then(work3).build(),
    maestro.WorkContext()
)

# Conditional
print("\n  Conditional flow (balance > 1000 → premium, otherwise → standard):")
check = maestro.LambdaWork(
    lambda ctx: maestro.DefaultWorkReport(
        maestro.WorkStatus.COMPLETED if ctx.get("balance",0) > 1000 else maestro.WorkStatus.FAILED, ctx),
    name="check-balance"
)
for balance in (1500, 500):
    flow = (maestro.aNewConditionalFlow()
            .execute(check)
            .when(maestro.WorkReportPredicate.COMPLETED)
            .then(maestro.LambdaWork(lambda ctx: print(f"    balance={ctx.get('balance')} → premium"), name="premium"))
            .otherwise(maestro.LambdaWork(lambda ctx: print(f"    balance={ctx.get('balance')} → standard"), name="standard"))
            .build())
    maestro.WorkFlowEngine().run(flow, maestro.WorkContext(balance=balance))

# Parallel
print("\n  Parallel flow (concurrent downloads):")
exec_svc = concurrent.futures.ThreadPoolExecutor(max_workers=3)
par_flow = (maestro.aNewParallelFlow()
            .execute(
                maestro.LambdaWork(lambda ctx: print("    downloading A..."), name="A"),
                maestro.LambdaWork(lambda ctx: print("    downloading B..."), name="B"),
                maestro.LambdaWork(lambda ctx: print("    downloading C..."), name="C"),
            )
            .with_executor(exec_svc)
            .build())
r = maestro.WorkFlowEngine().run(par_flow, maestro.WorkContext())
exec_svc.shutdown(wait=True)
print(f"    → all 3 completed: {r.status.value}")

# Repeat
print("\n  Repeat flow (retry until success, max 5 attempts):")
attempt = [0]
def flaky(ctx):
    attempt[0] += 1
    if attempt[0] < 3:
        return maestro.DefaultWorkReport(maestro.WorkStatus.FAILED, ctx)
    print(f"    succeeded on attempt {attempt[0]}")
retry_flow = (maestro.aNewRepeatFlow()
              .repeat(maestro.LambdaWork(flaky, "flaky"))
              .until(maestro.WorkReportPredicate.COMPLETED)
              .times(5)
              .build())
maestro.WorkFlowEngine().run(retry_flow, maestro.WorkContext())


# ════════════════════════════════════════════════════════════════════
#  4.  maestro.states — finite state machine
# ════════════════════════════════════════════════════════════════════
print(); print(SEP); print("4. maestro.states"); print(SEP)

pending   = maestro.State("PENDING")
paid      = maestro.State("PAID")
shipped   = maestro.State("SHIPPED")
delivered = maestro.State("DELIVERED")
cancelled = maestro.State("CANCELLED")

class PayEvent(maestro.Event): pass
class ShipEvent(maestro.Event): pass
class DeliverEvent(maestro.Event): pass
class CancelEvent(maestro.Event): pass

def log_t(msg): return maestro.LambdaEventHandler(lambda e: print(f"  → {msg}"))

order_fsm = (
    maestro.FiniteStateMachineBuilder(
        states={pending, paid, shipped, delivered, cancelled},
        initial_state=pending
    )
    .register_transition(maestro.TransitionBuilder().name("pay").source_state(pending).event_type(PayEvent).event_handler(log_t("payment received")).target_state(paid).build())
    .register_transition(maestro.TransitionBuilder().name("ship").source_state(paid).event_type(ShipEvent).event_handler(log_t("order shipped")).target_state(shipped).build())
    .register_transition(maestro.TransitionBuilder().name("deliver").source_state(shipped).event_type(DeliverEvent).event_handler(log_t("order delivered")).target_state(delivered).build())
    .register_transition(maestro.TransitionBuilder().name("cancel").source_state(pending).event_type(CancelEvent).event_handler(log_t("order cancelled")).target_state(cancelled).build())
    .build()
)

print(f"  Initial: {order_fsm.current_state}")
order_fsm.fire(PayEvent())
order_fsm.fire(ShipEvent())
order_fsm.fire(DeliverEvent())
print(f"  Final:   {order_fsm.current_state}")
print()
print(order_fsm.to_dot("order_lifecycle"))


# ════════════════════════════════════════════════════════════════════
#  5.  maestro.integration — cross-module composition
# ════════════════════════════════════════════════════════════════════
print(); print(SEP); print("5. maestro.integration — cross-module pipelines"); print(SEP)

# ── 5a. RuleBasedFilter in ETL pipeline ──────────────────────────
print("  5a. RuleBasedFilter — filter records with a rules engine:")

adult_rule = (maestro.RuleBuilder()
              .name("adult").when(lambda f: int(f.get("age", 0)) >= 18).build())
persons = [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 16},
           {"name": "Carol", "age": 30}]
adults: list = []
(maestro.JobBuilder()
 .named("adults")
 .reader(maestro.IterableRecordReader(persons))
 .filter(RuleBasedFilter(maestro.Rules(adult_rule)))
 .writer(maestro.CollectionRecordWriter(adults))
 .build()).call()
print(f"  Adults: {[p['name'] for p in adults]}")

# ── 5b. RuleBasedProcessor enrichment ────────────────────────────
print("\n  5b. RuleBasedProcessor — enrich records with rules engine:")
tier_rule = (maestro.RuleBuilder()
             .name("vip").when(lambda f: f.get("total", 0) > 500)
             .then(lambda f: f.put("tier", "vip")).build())
orders2 = [{"id": 1, "total": 750}, {"id": 2, "total": 100}, {"id": 3, "total": 900}]
enriched: list = []
(maestro.JobBuilder()
 .named("enrich")
 .reader(maestro.IterableRecordReader(orders2))
 .processor(RuleBasedProcessor(maestro.Rules(tier_rule)))
 .writer(maestro.CollectionRecordWriter(enriched))
 .build()).call()
for o in enriched:
    print(f"    id={o['id']} total={o['total']} tier={o.get('tier','standard')}")

# ── 5c. RuleSetWork inside a flow ─────────────────────────────────
print("\n  5c. RuleSetWork — rules engine as a workflow step:")
discount_rule = (maestro.RuleBuilder()
                 .name("discount").when(lambda f: f.get("tier") == "vip")
                 .then(lambda f: f.put("discount", 0.15) or print(f"    VIP discount applied!")).build())
ctx5 = maestro.WorkContext(tier="vip", customer="Dave")
flow5 = (maestro.aNewSequentialFlow()
         .execute(RuleSetWork(maestro.Rules(discount_rule), name="apply-discount"))
         .then(maestro.LambdaWork(lambda c: print(f"    discount={c.get('discount')}"), name="log"))
         .build())
maestro.WorkFlowEngine().run(flow5, ctx5)

# ── 5d. BatchWork + FSMGuardWork in one flow ──────────────────────
print("\n  5d. BatchWork + FSMGuardWork — batch job then FSM gate in one flow:")

pending2 = maestro.State("PENDING")
approved = maestro.State("APPROVED")

class ApproveEvent(maestro.Event): pass

approval_fsm = (maestro.FiniteStateMachineBuilder(states={pending2, approved}, initial_state=pending2)
               .register_transition(
                   maestro.TransitionBuilder().source_state(pending2).event_type(ApproveEvent).target_state(approved).build())
               .build())

batch_sink2: list = []
batch_job2 = (maestro.JobBuilder()
              .named("load-data")
              .reader(maestro.IterableRecordReader(["record-1","record-2","record-3"]))
              .writer(maestro.CollectionRecordWriter(batch_sink2))
              .build())

flow5d = (maestro.aNewSequentialFlow()
          .named("batch-then-approve")
          .execute(BatchWork(batch_job2, name="load"))
          .then(FSMGuardWork(approval_fsm, ApproveEvent(), success_states={approved}, name="approve"))
          .then(maestro.LambdaWork(lambda c: print(f"    Approved! Records loaded: {len(batch_sink2)}. FSM state: {c.get('fsm_state')}"), name="done"))
          .build())
r5d = maestro.WorkFlowEngine().run(flow5d, maestro.WorkContext())
print(f"    Flow status: {r5d.status.value}")

# ── 5e. FSMTransitionWork — fire a sequence of FSM events ─────────
print("\n  5e. FSMTransitionWork — fire a sequence of FSM events as one step:")
pending3   = maestro.State("P"); paid3 = maestro.State("PAID3"); shipped3 = maestro.State("S")

class P3(maestro.Event): pass
class S3(maestro.Event): pass

fsm3 = (maestro.FiniteStateMachineBuilder(states={pending3, paid3, shipped3}, initial_state=pending3)
        .register_transition(maestro.TransitionBuilder().source_state(pending3).event_type(P3).target_state(paid3).build())
        .register_transition(maestro.TransitionBuilder().source_state(paid3).event_type(S3).target_state(shipped3).build())
        .build())

ctx5e = maestro.WorkContext()
maestro.WorkFlowEngine().run(
    (maestro.aNewSequentialFlow()
     .execute(FSMTransitionWork(fsm3, [P3(), S3()], name="order-transitions"))
     .then(maestro.LambdaWork(lambda c: print(f"    Final state: {c.get('fsm_state')}, history: {c.get('fsm_history')}"), name="log"))
     .build()),
    ctx5e
)

print(); print(SEP); print("All Maestro SDK examples completed."); print(SEP)
