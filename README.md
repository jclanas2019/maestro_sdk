# Maestro SDK

**The simple, unified automation SDK for Python.**

Maestro unifies four automation disciplines — rules, batch processing, workflow orchestration and state machines — into a single coherent package. Each module is useful on its own; combined through the integration layer, they cover the full spectrum of enterprise automation patterns.

```bash
pip install maestro-sdk                   # core (no extra deps)
pip install "maestro-sdk[yaml]"           # + YAML rule and FSM files
pip install "maestro-sdk[dev]"            # + pytest, coverage
```

421 tests · Python 3.9+ · Zero required dependencies · MIT license

---

## Contents

1. [Architecture](#architecture)
2. [Modules at a glance](#modules-at-a-glance)
3. [Core modules](#core-modules)
4. [Integration layer](#integration-layer)
5. [P1 — resilience, observability, validation](#p1-modules)
6. [P2 — concurrency and reactivity](#p2-modules)
7. [P3 — operations](#p3-modules)
8. [Real-world use cases](#real-world-use-cases)
9. [CLI reference](#cli-reference)
10. [Project layout](#project-layout)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          maestro SDK                             │
├────────────────┬────────────────┬───────────────┬───────────────┤
│ maestro.rules  │ maestro.batch  │ maestro.flows │ maestro.states│
│                │                │               │               │
│ Facts          │ Record         │ Work          │ State         │
│ Rules          │ JobBuilder     │ WorkContext   │ Event         │
│ DefaultRules-  │ JobExecutor    │ Sequential-   │ FiniteState-  │
│ Engine         │ readers        │ Parallel-     │ Machine       │
│ @rule          │ writers        │ Conditional-  │ Transition-   │
│ @condition     │ filters        │ RepeatFlow    │ Builder       │
│ @action        │ mappers        │               │               │
├────────────────┴────────────────┴───────────────┴───────────────┤
│                      maestro.integration                        │
│  RuleSetWork · BatchWork · FSMGuardWork · FSMTransitionWork     │
│  RuleBasedFilter · RuleBasedProcessor                           │
├──────────────────────────────────────────────────────────────────┤
│  P1: maestro.retry · maestro.observe · maestro.validate         │
│  P2: maestro.events · maestro.graph · maestro.async_            │
│  P3: maestro.saga  · maestro.schedule · maestro.cli             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Modules at a glance

| Module | Key capability |
|---|---|
| `maestro.rules` | Declarative rules engine: decorator, builder and YAML APIs |
| `maestro.batch` | Record-oriented ETL: read → filter → map → process → write |
| `maestro.flows` | Sequential, Parallel, Conditional and Repeat workflow composition |
| `maestro.states` | Deterministic FSM with typed events, handlers and listeners |
| `maestro.integration` | Bridges: rules in flows, batch in flows, FSM gates |
| `maestro.retry` | Backoff strategies, circuit breaker, timeout, `@retryable` |
| `maestro.observe` | Counters, histograms and Prometheus export across all modules |
| `maestro.validate` | Schema validation and coercion for Facts, Records and WorkContext |
| `maestro.events` | Pub/sub bus: typed topics, wildcards, FSM bridge, async delivery |
| `maestro.graph` | DAG workflow: declare dependencies, automatic parallelism |
| `maestro.async_` | AsyncIO-native flows and batch, sync↔async adapters |
| `maestro.saga` | Distributed saga with automatic reverse compensation |
| `maestro.schedule` | Cron scheduler with no external dependencies |
| `maestro.cli` | `maestro` command: validate rules, run jobs, inspect FSMs |

---

## Core modules

### maestro.rules

Three styles for defining rules — pick the one that fits your team:

**Decorator style** (closest to business language):

```python
import maestro

@maestro.rule(name="premium discount", priority=1)
class PremiumDiscount:
    @maestro.condition
    def qualifies(self, facts):
        return facts.get("tier") == "premium" and facts.get("total", 0) > 100

    @maestro.action
    def apply(self, facts):
        facts.put("discount", 0.20)
        facts.put("free_shipping", True)

facts  = maestro.Facts(customer="Alice", tier="premium", total=250.0)
rules  = maestro.Rules(PremiumDiscount())
engine = maestro.DefaultRulesEngine()
engine.fire(rules, facts)
# facts.get("discount") == 0.20
```

**Builder style** (programmatic):

```python
free_shipping = (
    maestro.RuleBuilder()
    .name("free shipping")
    .priority(2)
    .when(lambda f: f.get("total", 0) > 100)
    .then(lambda f: f.put("shipping_cost", 0.0))
    .build()
)
```

**YAML style** (external configuration):

```yaml
# rules/pricing.yaml
name: vip-discount
condition: "tier == 'vip' and total > 50"
actions:
  - "discount = 0.15"
```

```python
from maestro.rules import YamlRuleFactory
rules = maestro.Rules(*YamlRuleFactory().from_file("rules/pricing.yaml"))
```

**Three engines:**

```python
maestro.DefaultRulesEngine().fire(rules, facts)          # fires every match
maestro.InferenceRulesEngine().fire(rules, facts)        # forward-chaining
maestro.FirstApplicableRulesEngine().fire(rules, facts)  # first match only
```

**Composite rules:**

```python
# All conditions must hold (atomic unit)
eligibility = maestro.UnitRuleGroup("eligibility")
eligibility.add_rule(age_rule)
eligibility.add_rule(country_rule)

# Mutually exclusive — first match wins (priority order)
tier = maestro.ActivationRuleGroup("discount tier")
tier.add_rule(gold_rule)    # priority 1 — checked first
tier.add_rule(silver_rule)  # priority 2
tier.add_rule(bronze_rule)  # priority 3
```

---

### maestro.batch

A declarative ETL pipeline: **Reader → Filter → Mapper → Processor → Marshaller → Writer**

```python
from dataclasses import dataclass

@dataclass
class Order:
    id: int
    customer: str
    total: float

sink = []
report = (
    maestro.JobBuilder()
    .named("orders-etl")
    .reader(maestro.FlatFileRecordReader("data/orders.csv"))
    .filter(maestro.HeaderRecordFilter())
    .filter(maestro.PredicateRecordFilter(lambda r: not r.payload.strip()))
    .mapper(maestro.DelimitedRecordMapper(
        Order,
        field_names=["id", "customer", "total"],
        type_converters={"id": int, "total": float},
    ))
    .processor(maestro.FilteringRecordProcessor(
        lambda p: p.total < 0, reason="negative total"
    ))
    .marshaller(maestro.JsonMarshaller())
    .writer(maestro.CollectionRecordWriter(sink))
    .batch_size(500)
    .error_threshold(10)
    .build()
    .call()
)
print(f"Written={report.metrics.written_count}  "
      f"Filtered={report.metrics.filtered_count}  "
      f"Duration={report.metrics.duration_seconds:.2f}s")
```

**Built-in readers:**

| Reader | Source |
|---|---|
| `FlatFileRecordReader` | Text file — one line per record |
| `StringRecordReader` | Multi-line string (useful in tests) |
| `CsvDictRecordReader` | CSV with header → `dict` per row |
| `JsonLinesRecordReader` | JSON-Lines file |
| `IterableRecordReader` | Any Python iterable |

**Parallel execution:**

```python
executor = maestro.JobExecutor()
reports  = executor.execute_all([job_a, job_b, job_c])
executor.shutdown()
```

---

### maestro.flows

Four flow types — all implement `Work` so they compose and nest freely:

```python
import concurrent.futures

executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

workflow = (
    maestro.aNewSequentialFlow()
    .named("order-pipeline")
    .execute(validate_work)
    .then(
        maestro.aNewParallelFlow()
        .execute(fetch_inventory, fetch_pricing, fetch_customer)
        .with_executor(executor)
        .build()
    )
    .then(
        maestro.aNewConditionalFlow()
        .execute(check_stock_work)
        .when(maestro.WorkReportPredicate.COMPLETED)
        .then(reserve_and_charge_work)
        .otherwise(backorder_work)
        .build()
    )
    .then(
        maestro.aNewRepeatFlow()
        .repeat(send_notification_work)
        .until(maestro.WorkReportPredicate.COMPLETED)
        .times(3)
        .build()
    )
    .build()
)

report = maestro.WorkFlowEngine().run(workflow, maestro.WorkContext(order_id="ORD-42"))
executor.shutdown()
```

**Sharing data between steps** via `WorkContext`:

```python
def step_a(ctx): ctx.put("result", compute())
def step_b(ctx): use(ctx.get("result"))   # data from step_a

flow = (maestro.aNewSequentialFlow()
        .execute(maestro.LambdaWork(step_a, "a"))
        .then(maestro.LambdaWork(step_b, "b"))
        .build())
```

---

### maestro.states

Implements the Erlang formula: **`State(S) × Event(E) → Actions(A), State(S')`**

```python
locked, unlocked = maestro.State("locked"), maestro.State("unlocked")

class CoinEvent(maestro.Event): pass
class PushEvent(maestro.Event): pass

fsm = (
    maestro.FiniteStateMachineBuilder(states={locked, unlocked}, initial_state=locked)
    .register_transition(
        maestro.TransitionBuilder().name("unlock")
        .source_state(locked).event_type(CoinEvent)
        .event_handler(maestro.LambdaEventHandler(lambda e: print("Unlocked!")))
        .target_state(unlocked).build()
    )
    .register_transition(
        maestro.TransitionBuilder().name("lock")
        .source_state(unlocked).event_type(PushEvent).target_state(locked).build()
    )
    .build()
)

fsm.fire(CoinEvent())                # locked → unlocked
fsm.fire(PushEvent())                # unlocked → locked
print(fsm.current_state)             # State('locked')
print(fsm.to_dot("turnstile"))       # Graphviz DOT export
```

---

## Integration layer

`maestro.integration` connects the four core modules. These are the only classes that require more than one module.

```python
from maestro.integration import (
    RuleSetWork,        # rules engine as a flow step
    BatchWork,          # batch job as a flow step
    FSMGuardWork,       # FSM transition as a conditional gate
    FSMTransitionWork,  # fire a sequence of FSM events as one step
    RuleBasedFilter,    # rules engine as a batch record filter
    RuleBasedProcessor, # rules engine to enrich batch records
)
```

**Rules inside a flow** (facts ↔ context are automatically bridged):

```python
discount_rules = maestro.Rules(vip_rule, promo_rule, bulk_rule)

flow = (maestro.aNewSequentialFlow()
        .execute(RuleSetWork(discount_rules, name="apply-discounts"))
        .then(charge_step)       # WorkContext has discount, shipping_cost, etc.
        .build())
```

**Batch records filtered by rules:**

```python
job = (maestro.JobBuilder()
       .reader(maestro.CsvDictRecordReader("orders.csv"))
       .filter(RuleBasedFilter(maestro.Rules(adult_rule, valid_email_rule)))
       .processor(RuleBasedProcessor(maestro.Rules(tier_rule, discount_rule)))
       .writer(maestro.FileRecordWriter("enriched.jsonl"))
       .build())
```

**FSM as a flow gate:**

```python
guard = FSMGuardWork(
    fsm=order_fsm, event=PayEvent(),
    success_states={paid}, error_states={cancelled},
)

flow = (maestro.aNewConditionalFlow()
        .execute(guard)
        .when(maestro.WorkReportPredicate.COMPLETED)
        .then(fulfillment_flow)
        .otherwise(refund_flow)
        .build())
```

---

## P1 modules

### maestro.retry

```python
from maestro.retry import retry, RetryPolicy, ExponentialBackoff, CircuitBreaker, retryable

# Wrap any Work
resilient = retry(
    work            = call_api_work,
    max_attempts    = 4,
    backoff         = ExponentialBackoff(base=0.5, multiplier=2.0, max_delay=30.0),
    on              = [ConnectionError, TimeoutError],
    circuit_breaker = CircuitBreaker(threshold=5, reset_after=60),
)

# Decorate a Work class
@retryable(max_attempts=3, backoff=ExponentialBackoff())
class FetchWork(maestro.Work):
    def execute(self, ctx): ...

# Retry batch readers
from maestro.retry import RetryableReader
reader = RetryableReader(
    HttpRecordReader("https://api.example.com/events"),
    policy=RetryPolicy(max_attempts=3, on=[ConnectionError]),
)
```

| Strategy | Delay |
|---|---|
| `NoBackoff` | 0s — retry immediately |
| `ConstantBackoff(N)` | always N seconds |
| `LinearBackoff(base, increment)` | `base + (n−1) × increment` |
| `ExponentialBackoff(base, multiplier)` | `base × multiplier^(n−1)` |
| `JitteredBackoff(...)` | random in `[0, exp_delay]` — prevents thundering herds |

---

### maestro.observe

```python
from maestro.observe import MaestroObserver, InMemoryObserver

mem = InMemoryObserver()
obs = MaestroObserver(observers=[mem])

engine = obs.instrument_rules_engine(maestro.DefaultRulesEngine())
job    = obs.instrument_job(job)
flow   = obs.instrument_flow(flow)
fsm    = obs.instrument_fsm(fsm)

# After execution
print(mem.summary())
print(mem.counter("rules", "rule_fired", rule="vip-discount"))
print(mem.histogram("batch", "job_duration_seconds", job="daily-etl"))
print(mem.export_prometheus())          # Prometheus text format

from maestro.observe import timed
with timed("flows", "custom-step", mem):
    heavy_computation()
```

---

### maestro.validate

```python
from maestro.validate import Schema, field, Required, Range, Length, Pattern, OneOf

order_schema = Schema(
    id     = field(int,   Required()),
    status = field(str,   OneOf("pending", "paid", "shipped")),
    total  = field(float, Range(min=0), coerce=True),
    email  = field(str,   Pattern(r"^[^@]+@[^@]+\.[^@]+$"), required=False),
    items  = field(list,  Length(min=1)),
)

result = order_schema.validate(data)
if not result.ok:
    for err in result.errors: print(err)  # [field] message

# Coerce string → target type
coerced = order_schema.coerce({"id": 1, "status": "paid", "total": "99.5", "items": ["x"]})

# ValidatedFacts
from maestro.validate import ValidatedFacts
facts = ValidatedFacts(order_schema, strict=True, id=1, status="pending", total=50.0, items=["a"])

# In batch pipelines
from maestro.validate import SchemaFilter, SchemaProcessor
job = (maestro.JobBuilder()
       .filter(SchemaFilter(order_schema))
       .processor(SchemaProcessor(order_schema, coerce=True))
       .build())

# In flows — validate WorkContext before/after a step
from maestro.validate import ValidatedWork
step = ValidatedWork(
    work        = process_order,
    pre_schema  = Schema(order_id=field(str, Required()), amount=field(float)),
    post_schema = Schema(receipt_id=field(str, Required())),
)
```

---

## P2 modules

### maestro.events

```python
from maestro.events import EventBus, Topic, FSMEventBridge, RuleEventRouter

bus    = EventBus()
orders = Topic[dict]("orders.created")

# Subscribe / unsubscribe
with orders.subscribe(bus, lambda m: process(m.payload)):
    orders.publish(bus, {"id": 1, "total": 150.0})

# Block until a message arrives (with timeout)
msg = bus.wait_for("payment.confirmed", timeout=30.0)

# Fire exactly once
bus.subscribe_once("order.ready", lambda m: notify_warehouse(m.payload))

# Wildcard — every topic
bus.subscribe_fn("*", lambda m: logger.debug("[%s] %s", m.topic, m.payload))

# FSM transitions → bus messages
bridge = FSMEventBridge(bus, topic_prefix="order.fsm")
fsm    = FiniteStateMachineBuilder(...).register_listener(bridge).build()
# Every fsm.fire(X) publishes to "order.fsm.transitioned" and "order.fsm.entered.<STATE>"

# Rules-based routing
router = RuleEventRouter(bus, maestro.Rules(vip_rule), default_topic="orders.standard")
bus.subscribe("orders.all", router)

# Consume bus messages as batch records
from maestro.events import BusRecordReader
job = (maestro.JobBuilder()
       .reader(BusRecordReader(bus, "orders.created", max_records=1000))
       .writer(maestro.FileRecordWriter("output.jsonl"))
       .build())
```

---

### maestro.graph

```python
from maestro.graph import GraphBuilder

flow = (
    GraphBuilder()
    .named("data-pipeline")
    .add("fetch-users",    fetch_users_work)
    .add("fetch-orders",   fetch_orders_work)
    .add("fetch-catalog",  fetch_catalog_work)
    .add("join",           join_work,      depends_on=["fetch-users",  "fetch-orders"])
    .add("enrich",         enrich_work,    depends_on=["join",         "fetch-catalog"])
    .add("export-db",      db_write_work,  depends_on=["enrich"])
    .add("export-s3",      s3_write_work,  depends_on=["enrich"])
    .add("notify",         notify_work,    depends_on=["export-db",    "export-s3"])
    .fail_fast(True)
    .build()
)
# Execution: [fetch-users, fetch-orders, fetch-catalog] → join → enrich → [export-db, export-s3] → notify

report = maestro.WorkFlowEngine().run(flow, maestro.WorkContext())
for name, nr in report.node_reports.items():
    print(f"{name}: {nr.status.value} ({nr.duration*1000:.0f}ms)")
print(flow.to_dot())   # Graphviz DOT for visualisation
```

---

### maestro.async_

```python
import asyncio
from maestro.async_ import (
    AsyncLambdaWork, AsyncSequentialFlow, AsyncParallelFlow,
    AsyncWorkFlowEngine, AsyncJobBuilder, AsyncIterableReader,
    sync_to_async, async_to_sync,
)

async def main():
    # Three I/O calls run concurrently
    flow = (
        AsyncSequentialFlow.Builder()
        .execute(validate_work)
        .then(
            AsyncParallelFlow.Builder()
            .execute(
                AsyncLambdaWork(fetch_inventory_async, "inventory"),
                AsyncLambdaWork(fetch_pricing_async,   "pricing"),
                AsyncLambdaWork(check_fraud_async,     "fraud"),
            )
            .build()
        )
        .then(AsyncLambdaWork(process_async, "process"))
        .build()
    )
    return await AsyncWorkFlowEngine().run(flow, maestro.WorkContext())

report = asyncio.run(main())

# Mix sync and async freely
async_step = sync_to_async(heavy_sync_work)   # thread pool — no blocking
sync_step  = async_to_sync(async_work)         # asyncio.run — works in sync engine
```

---

## P3 modules

### maestro.saga

```python
from maestro.saga import SagaBuilder, SagaStatus

saga = (
    SagaBuilder()
    .named("book-trip")
    .step("book-flight",
          work         = BookFlightWork(),
          compensation = CancelFlightWork())
    .step("book-hotel",
          work         = BookHotelWork(),
          compensation = CancelHotelWork())
    .step("charge-card",
          work         = ChargeCardWork(),
          compensation = RefundCardWork())
    .step("send-confirmation", work=SendEmailWork())   # no compensation
    .build()
)

report = saga.execute(maestro.WorkContext(trip_id="TRIP-99"))
match report.saga_status:
    case SagaStatus.COMPLETED:             print("Trip booked!")
    case SagaStatus.COMPENSATED:           print(f"Rolled back: {report.compensated_steps}")
    case SagaStatus.PARTIALLY_COMPENSATED: print(f"Partial rollback: {report.failed_compensations}")

# Saga IS a Work — embeds in any flow
flow = maestro.aNewSequentialFlow().execute(saga).then(update_crm_work).build()
```

---

### maestro.schedule

```python
from maestro.schedule import Scheduler, CronTrigger, IntervalTrigger, ImmediateTrigger

with Scheduler(tick_seconds=1.0, max_workers=4) as scheduler:
    scheduler.add("daily-etl",
                  job=daily_etl_job,
                  trigger=CronTrigger("0 2 * * *"))          # 2am daily

    scheduler.add("health-check",
                  work=maestro.LambdaWork(ping, "ping"),
                  trigger=IntervalTrigger(seconds=30))

    scheduler.add("startup",
                  work=migrate_work,
                  trigger=ImmediateTrigger())

    scheduler.pause("daily-etl")
    scheduler.resume("daily-etl")

    for s in scheduler.status():
        print(f"{s['name']}: next={s['next_fire']}  runs={s['run_count']}")
```

| Trigger | When it fires |
|---|---|
| `CronTrigger("*/5 * * * *")` | Every 5 minutes (5-field cron, no external deps) |
| `CronTrigger(CronTrigger.DAILY)` | Daily at midnight |
| `IntervalTrigger(seconds=30)` | Every 30 seconds |
| `OnceTrigger(at=datetime(...))` | Once at a specific moment |
| `ImmediateTrigger()` | Once, as soon as scheduler starts |

---

### maestro.cli

```bash
maestro info                                       # SDK version + module status
maestro rules validate rules/pricing.yaml         # validate YAML rules
maestro rules fire rules/pricing.yaml \
    --facts '{"tier":"vip","total":200}'           # fire rules, show results
maestro batch run jobs/orders.yaml                # run a batch job from YAML
maestro saga describe sagas/book-trip.yaml        # document saga steps
maestro schedule cron "*/15 * * * *" --count 5   # preview next fire times
maestro schedule demo --duration 10               # live scheduler demo
maestro fsm dot fsm/order.yaml                    # generate Graphviz DOT
maestro fsm run fsm/order.yaml \
    --events PayEvent ShipEvent                   # fire events, show transitions
```

---

## Real-world use cases

### Use case 1: E-commerce order processing

Rules determine pricing, a state machine tracks the order lifecycle, a saga handles payment and inventory with compensation, and events notify downstream services.

```python
import maestro
from maestro.saga import SagaBuilder, SagaStatus
from maestro.events import EventBus, FSMEventBridge
from maestro.integration import FSMGuardWork, RuleSetWork
from maestro.retry import retry, ExponentialBackoff, CircuitBreaker
from maestro.validate import Schema, field, Required, Range, OneOf, ValidatedWork
from maestro.observe import MaestroObserver, InMemoryObserver

# ── Observability ──────────────────────────────────────────────────
metrics = InMemoryObserver()
obs     = MaestroObserver(observers=[metrics])

# ── Event bus ──────────────────────────────────────────────────────
bus = EventBus()
bus.subscribe_fn("order.completed", lambda m: send_confirmation(m.payload))
bus.subscribe_fn("order.cancelled", lambda m: notify_support(m.payload))

# ── Input validation ───────────────────────────────────────────────
order_schema = Schema(
    order_id = field(str,   Required()),
    items    = field(list,  Required()),
    total    = field(float, Range(min=0.01), coerce=True),
    tier     = field(str,   OneOf("standard", "premium", "vip"), required=False),
)

# ── Pricing rules ──────────────────────────────────────────────────
@maestro.rule(name="vip-discount", priority=1)
class VIPDiscount:
    @maestro.condition
    def qualifies(self, f): return f.get("tier") == "vip" and f.get("total", 0) > 100
    @maestro.action
    def apply(self, f):     f.put("discount", 0.20); f.put("priority_shipping", True)

@maestro.rule(name="premium-discount", priority=2)
class PremiumDiscount:
    @maestro.condition
    def qualifies(self, f): return f.get("tier") == "premium"
    @maestro.action
    def apply(self, f):     f.put("discount", max(f.get("discount", 0), 0.10))

pricing_rules = maestro.Rules(VIPDiscount(), PremiumDiscount())

# ── Order FSM ──────────────────────────────────────────────────────
pending, paid, shipped, delivered, cancelled = (
    maestro.State(s) for s in ("PENDING", "PAID", "SHIPPED", "DELIVERED", "CANCELLED")
)
class PayEvent(maestro.Event):     pass
class ShipEvent(maestro.Event):    pass
class DeliverEvent(maestro.Event): pass
class CancelEvent(maestro.Event):  pass

order_fsm = (
    maestro.FiniteStateMachineBuilder(
        states={pending, paid, shipped, delivered, cancelled},
        initial_state=pending,
    )
    .register_transition(maestro.TransitionBuilder().name("pay")
        .source_state(pending).event_type(PayEvent).target_state(paid).build())
    .register_transition(maestro.TransitionBuilder().name("ship")
        .source_state(paid).event_type(ShipEvent).target_state(shipped).build())
    .register_transition(maestro.TransitionBuilder().name("cancel")
        .source_state(pending).event_type(CancelEvent).target_state(cancelled).build())
    .register_listener(FSMEventBridge(bus, topic_prefix="order.fsm"))
    .build()
)
obs.instrument_fsm(order_fsm)

# ── Payment saga ───────────────────────────────────────────────────
payment_cb = CircuitBreaker(threshold=5, reset_after=60, name="payment-cb")

order_saga = (
    SagaBuilder()
    .named("process-order")
    .step("reserve-inventory",
          maestro.LambdaWork(lambda ctx: warehouse_api.reserve(ctx), "reserve"),
          maestro.LambdaWork(lambda ctx: warehouse_api.release(ctx), "release"))
    .step("charge-payment",
          retry(maestro.LambdaWork(lambda ctx: payment_gateway.charge(ctx), "charge"),
                max_attempts=3, backoff=ExponentialBackoff(base=1.0),
                circuit_breaker=payment_cb, on=[ConnectionError]),
          maestro.LambdaWork(lambda ctx: payment_gateway.refund(ctx), "refund"))
    .step("update-state",
          maestro.LambdaWork(lambda ctx: order_fsm.fire(PayEvent()), "transition"))
    .step("publish-event",
          maestro.LambdaWork(
              lambda ctx: bus.publish("order.completed", {"order_id": ctx.get("order_id")}),
              "publish"))
    .build()
)

# ── Main order flow ────────────────────────────────────────────────
def process_order(order: dict):
    result = order_schema.validate(order)
    if not result.ok:
        raise ValueError(str(result))

    ctx = maestro.WorkContext(**order_schema.coerce(order))

    flow = (
        maestro.aNewSequentialFlow()
        .named(f"order-{order['order_id']}")
        .execute(RuleSetWork(pricing_rules, name="pricing"))   # apply discounts
        .then(ValidatedWork(order_saga,                        # saga with pre/post check
              pre_schema=Schema(total=field(float, Range(min=0)))))
        .build()
    )
    return obs.instrument_flow(maestro.WorkFlowEngine()).run(flow, ctx)
```

---

### Use case 2: Data quality pipeline

Daily ETL with schema validation, rule-based classification, retry on transient errors, and scheduled execution.

```python
import maestro
from maestro.validate import Schema, field, Required, Range, Pattern, SchemaFilter, SchemaProcessor
from maestro.integration import RuleBasedProcessor
from maestro.observe import MaestroObserver, InMemoryObserver
from maestro.schedule import Scheduler, CronTrigger
from maestro.retry import RetryableReader, RetryPolicy, ExponentialBackoff

# ── Schema ─────────────────────────────────────────────────────────
txn_schema = Schema(
    txn_id   = field(str,   Required(), Pattern(r"^TXN-\d+")),
    amount   = field(float, Range(min=0.01, max=1_000_000), coerce=True),
    currency = field(str,   Required()),
    country  = field(str,   Required()),
)

# ── Classification rules ───────────────────────────────────────────
large_txn = (
    maestro.RuleBuilder()
    .name("large-transaction")
    .when(lambda f: f.get("amount", 0) > 10_000)
    .then(lambda f: f.put("risk_flag", "HIGH"))
    .build()
)
foreign_txn = (
    maestro.RuleBuilder()
    .name("foreign-currency")
    .when(lambda f: f.get("currency") != "USD")
    .then(lambda f: f.put("requires_review", True))
    .build()
)

# ── Observability ──────────────────────────────────────────────────
metrics = InMemoryObserver()
obs     = MaestroObserver(observers=[metrics])

# ── Job factory ────────────────────────────────────────────────────
def build_job():
    reader = RetryableReader(
        maestro.FlatFileRecordReader("data/transactions.csv"),
        policy=RetryPolicy(max_attempts=3, backoff=ExponentialBackoff()),
    )
    return obs.instrument_job(
        maestro.JobBuilder()
        .named("daily-transaction-etl")
        .reader(reader)
        .filter(maestro.HeaderRecordFilter())
        .filter(SchemaFilter(txn_schema))                               # drop malformed
        .processor(SchemaProcessor(txn_schema, coerce=True))            # coerce types
        .processor(RuleBasedProcessor(maestro.Rules(large_txn, foreign_txn)))  # classify
        .marshaller(maestro.JsonMarshaller())
        .writer(maestro.FileRecordWriter("output/transactions_enriched.jsonl"))
        .batch_size(1000)
        .error_threshold(50)
        .build()
    )

# ── Schedule ───────────────────────────────────────────────────────
def run_etl(ctx):
    report = maestro.JobExecutor().execute(build_job())
    print(f"ETL: written={report.metrics.written_count}  "
          f"filtered={report.metrics.filtered_count}  "
          f"duration={report.metrics.duration_seconds:.1f}s")
    if report.metrics.failed_count > 10:
        send_alert(f"High ETL failure rate: {report.metrics.failed_count}")

scheduler = Scheduler()
scheduler.add("daily-etl",
              work    = maestro.LambdaWork(run_etl, "etl"),
              trigger = CronTrigger("0 2 * * *"))    # 2am every day
scheduler.start()
```

---

### Use case 3: Fraud detection system

Transactions arrive via event bus; rules flag suspicious ones; FSM tracks account suspension states; observer feeds the compliance dashboard.

```python
import maestro
from maestro.events import EventBus, Topic, BusRecordReader, RuleEventRouter
from maestro.integration import RuleBasedProcessor
from maestro.observe import MaestroObserver, InMemoryObserver

bus          = EventBus()
metrics      = InMemoryObserver()
obs          = MaestroObserver(observers=[metrics])
engine       = obs.instrument_rules_engine(maestro.DefaultRulesEngine())

# ── Fraud rules ────────────────────────────────────────────────────
velocity_rule = (
    maestro.RuleBuilder()
    .name("velocity-check")
    .when(lambda f: f.get("txn_count_1h", 0) > 20)
    .then(lambda f: f.put("fraud_flag", "VELOCITY"))
    .build()
)
amount_rule = (
    maestro.RuleBuilder()
    .name("high-amount-new-merchant")
    .when(lambda f: f.get("amount", 0) > 5000 and f.get("is_new_merchant", False))
    .then(lambda f: f.put("fraud_flag", "HIGH_AMOUNT"))
    .build()
)
geo_rule = (
    maestro.RuleBuilder()
    .name("geo-anomaly")
    .when(lambda f: f.get("distance_km", 0) > 5000 and f.get("minutes_since_last", 999) < 30)
    .then(lambda f: f.put("fraud_flag", "GEO_ANOMALY"))
    .build()
)

fraud_rules = maestro.Rules(velocity_rule, amount_rule, geo_rule)

# ── Account FSM ────────────────────────────────────────────────────
normal, under_review, frozen = (
    maestro.State(s) for s in ("NORMAL", "UNDER_REVIEW", "FROZEN")
)
class FlagEvent(maestro.Event):    pass
class FreezeEvent(maestro.Event):  pass
class ClearEvent(maestro.Event):   pass

account_fsm = (
    maestro.FiniteStateMachineBuilder(
        states={normal, under_review, frozen}, initial_state=normal)
    .register_transition(maestro.TransitionBuilder()
        .source_state(normal).event_type(FlagEvent).target_state(under_review).build())
    .register_transition(maestro.TransitionBuilder()
        .source_state(under_review).event_type(FreezeEvent).target_state(frozen).build())
    .register_transition(maestro.TransitionBuilder()
        .source_state(under_review).event_type(ClearEvent).target_state(normal).build())
    .ignore_undefined_transitions(True)
    .build()
)
obs.instrument_fsm(account_fsm)

# ── Route flagged transactions ─────────────────────────────────────
route_rule = (
    maestro.RuleBuilder()
    .name("route-flagged")
    .when(lambda f: "fraud_flag" in f)
    .then(lambda f: f.put("route", "fraud.alerts"))
    .build()
)
router = RuleEventRouter(bus, maestro.Rules(route_rule))
bus.subscribe("transactions.incoming", router)
bus.subscribe_fn("fraud.alerts", lambda m: account_fsm.fire(FlagEvent()))

# ── Stream-to-batch processing ─────────────────────────────────────
def start_detection():
    clean_sink = []
    job = obs.instrument_job(
        maestro.JobBuilder()
        .named("fraud-detection")
        .reader(BusRecordReader(bus, "transactions.incoming", max_records=10_000))
        .processor(RuleBasedProcessor(fraud_rules, engine=engine))
        .processor(maestro.FilteringRecordProcessor(lambda p: bool(p.get("fraud_flag"))))
        .writer(maestro.CollectionRecordWriter(clean_sink))
        .build()
    )
    maestro.JobExecutor().execute(job)

    # Dashboard metrics
    fired = metrics.counter("rules", "rule_fired", rule="velocity-check")
    print(f"Velocity violations: {fired:.0f}")
    print(metrics.export_prometheus())
```

---

### Use case 4: API orchestration with resilience

Five independent APIs called in parallel (via GraphFlow), each with retry and circuit breaker. Output is validated before the next stage.

```python
import asyncio
import maestro
from maestro.graph import GraphBuilder
from maestro.retry import retry, ExponentialBackoff, CircuitBreaker
from maestro.validate import ValidatedWork, Schema, field, Required
from maestro.async_ import AsyncLambdaWork, AsyncSequentialFlow, AsyncWorkFlowEngine, sync_to_async
from maestro.observe import MaestroObserver, InMemoryObserver

metrics = InMemoryObserver()
obs     = MaestroObserver(observers=[metrics])

# One circuit breaker per external service
cbs = {svc: CircuitBreaker(threshold=3, reset_after=30, name=svc)
       for svc in ("user-api", "inventory-api", "pricing-api", "shipping-api", "loyalty-api")}

# Async fetchers
async def fetch_user(ctx):      ctx.put("user",      await user_api.get(ctx.get("user_id")))
async def fetch_inventory(ctx): ctx.put("inventory", await inventory_api.get(ctx.get("items")))
async def fetch_pricing(ctx):   ctx.put("pricing",   await pricing_api.get(ctx.get("items")))
async def fetch_shipping(ctx):  ctx.put("shipping",  await shipping_api.get(ctx.get("dest")))
async def fetch_loyalty(ctx):   ctx.put("loyalty",   await loyalty_api.get(ctx.get("user_id")))

# Wrap each in retry + circuit breaker
def resilient(fn, service):
    return retry(
        maestro.LambdaWork(lambda ctx: asyncio.run(fn(ctx)), service),
        max_attempts=3, backoff=ExponentialBackoff(base=0.5),
        circuit_breaker=cbs[service], on=[ConnectionError, TimeoutError],
    )

# Validate after aggregation
aggregated_schema = Schema(
    user      = field(dict, Required()),
    inventory = field(dict, Required()),
    pricing   = field(dict, Required()),
    shipping  = field(dict, Required()),
    loyalty   = field(int,  Required()),
)

# DAG: five independent fetches, one aggregate step
def build_dag():
    return (
        GraphBuilder()
        .named("data-aggregation")
        .add("user",      resilient(fetch_user,      "user-api"))
        .add("inventory", resilient(fetch_inventory,  "inventory-api"))
        .add("pricing",   resilient(fetch_pricing,    "pricing-api"))
        .add("shipping",  resilient(fetch_shipping,   "shipping-api"))
        .add("loyalty",   resilient(fetch_loyalty,    "loyalty-api"))
        .add("aggregate",
             ValidatedWork(
                 maestro.LambdaWork(compute_final_offer, "aggregate"),
                 post_schema=aggregated_schema,
             ),
             depends_on=["user", "inventory", "pricing", "shipping", "loyalty"])
        .build()
    )

async def get_order_data(user_id, items, destination):
    ctx  = maestro.WorkContext(user_id=user_id, items=items, dest=destination)
    dag  = obs.instrument_flow(build_dag())
    flow = (AsyncSequentialFlow.Builder()
            .execute(sync_to_async(dag))
            .then(AsyncLambdaWork(build_offer_async, "build-offer"))
            .build())
    return await AsyncWorkFlowEngine().run(flow, ctx)
```

---

### Use case 5: Event-driven microservice coordination

Multiple services communicate through a bus. Each service has an FSM for internal state. A saga coordinates the cross-service transaction. The scheduler triggers reconciliation.

```python
import maestro
from maestro.events import EventBus, Topic, AsyncEventBus, FSMEventBridge
from maestro.saga import SagaBuilder, SagaStatus
from maestro.integration import FSMTransitionWork
from maestro.schedule import Scheduler, IntervalTrigger

bus = AsyncEventBus().start()

order_topic   = Topic[dict]("orders.incoming")
payment_topic = Topic[dict]("payments.status")
notify_topic  = Topic[dict]("notifications")

# Payment service — FSM tracks each payment lifecycle
p_init, p_auth, p_captured, p_declined = (
    maestro.State(s) for s in ("INITIATED", "AUTHORISED", "CAPTURED", "DECLINED")
)
class AuthoriseEvent(maestro.Event): pass
class CaptureEvent(maestro.Event):   pass
class DeclineEvent(maestro.Event):   pass

payment_fsm = (
    maestro.FiniteStateMachineBuilder(
        states={p_init, p_auth, p_captured, p_declined}, initial_state=p_init)
    .register_transition(maestro.TransitionBuilder()
        .source_state(p_init).event_type(AuthoriseEvent).target_state(p_auth)
        .event_handler(maestro.LambdaEventHandler(
            lambda e: payment_topic.publish(bus, {"status": "authorised"})))
        .build())
    .register_transition(maestro.TransitionBuilder()
        .source_state(p_auth).event_type(CaptureEvent).target_state(p_captured)
        .event_handler(maestro.LambdaEventHandler(
            lambda e: payment_topic.publish(bus, {"status": "captured"})))
        .build())
    .register_transition(maestro.TransitionBuilder()
        .source_state(p_auth).event_type(DeclineEvent).target_state(p_declined)
        .event_handler(maestro.LambdaEventHandler(
            lambda e: payment_topic.publish(bus, {"status": "declined"})))
        .build())
    .register_listener(FSMEventBridge(bus, "payment.fsm"))
    .build()
)

# Checkout saga — coordinates inventory + payment across services
checkout_saga = (
    SagaBuilder()
    .named("checkout")
    .step("authorise-payment",
          FSMTransitionWork(payment_fsm, [AuthoriseEvent()], name="authorise"),
          maestro.LambdaWork(lambda ctx: payment_fsm.fire(DeclineEvent()), "decline"))
    .step("reserve-stock",
          maestro.LambdaWork(reserve_stock, "reserve"),
          maestro.LambdaWork(release_stock, "release"))
    .step("capture-payment",
          FSMTransitionWork(payment_fsm, [CaptureEvent()], name="capture"))
    .step("dispatch",
          maestro.LambdaWork(dispatch_order, "dispatch"))
    .build()
)

# Each incoming order triggers the saga
def on_order_received(msg):
    ctx    = maestro.WorkContext(**msg.payload)
    report = checkout_saga.execute(ctx)
    topic  = "notifications"
    status = "ORDER_CONFIRMED" if report.saga_status == SagaStatus.COMPLETED else "ORDER_FAILED"
    notify_topic.publish(bus, {"status": status, "order_id": ctx.get("order_id")})

order_topic.subscribe(bus, on_order_received)

# Periodic reconciliation — find stuck orders, auto-decline
def reconcile(ctx):
    for order_id in find_stuck_orders(older_than_minutes=30):
        payment_fsm.fire(DeclineEvent())

scheduler = Scheduler()
scheduler.add("reconcile", work=maestro.LambdaWork(reconcile, "reconcile"),
              trigger=IntervalTrigger(seconds=1800))
scheduler.start()
```

---

## CLI reference

```
Usage: maestro [--version] [--debug] <command> [options]

Commands
────────
  info                         SDK version and module status

  rules validate <yaml>        Validate a YAML rule file
  rules fire     <yaml>        Fire rules against JSON facts
    --facts '{"k":"v"}'          Input as JSON string
    --engine default|inference|first

  batch run   <yaml>           Run a batch job from YAML config
  batch stats <json>           Show metrics from a saved report

  saga describe <yaml>         Document saga steps and compensations

  schedule cron <expr>         Preview next N fire times for a cron expression
    --count N                    Number of times to preview (default: 5)
  schedule demo                Live scheduler demo
    --duration N                 Duration in seconds (default: 10)

  fsm dot <yaml>               Generate Graphviz DOT from a YAML FSM
  fsm run <yaml>               Fire events against a YAML-defined FSM interactively
    --events Event1 Event2       Event names to fire
```

---

## Running the examples

```bash
pip install -e ".[yaml,dev]"

python examples/quickstart.py     # core modules + integration
python examples/p1_features.py    # retry, observe, validate
python examples/p2_features.py    # events, graph, async
python examples/p3_features.py    # saga, schedule, cli

python -m pytest tests/ -v        # 421 tests
```

---

## Project layout

```
maestro/
├── __init__.py              ← single import namespace
├── rules/                   ← rules engine
├── batch/                   ← ETL batch processing
├── flows/                   ← workflow orchestration
├── states/                  ← finite state machine
├── integration/             ← cross-module bridges
├── retry/                   ← resilience policies
├── observe/                 ← unified observability
├── validate/                ← schema validation
├── events/                  ← reactive event bus
├── graph/                   ← DAG workflow
├── async_/                  ← AsyncIO-native support
├── saga/                    ← distributed saga
├── schedule/                ← cron scheduler
└── cli/                     ← command-line interface

examples/
├── quickstart.py
├── p1_features.py
├── p2_features.py
└── p3_features.py

tests/                       ← 421 tests across all modules
pyproject.toml               ← installable: pip install maestro-sdk
```

---

## License

MIT — inspired by the [j-easy](https://github.com/j-easy) suite of Java libraries.
