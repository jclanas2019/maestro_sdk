"""
examples/p1_features.py — Priority 1 features: retry, observe, validate.

Run: python examples/p1_features.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import maestro
from maestro.retry import (
    retry, retryable, RetryPolicy, RetryWork, RetryableReader,
    ExponentialBackoff, ConstantBackoff, JitteredBackoff,
    CircuitBreaker, CircuitOpenError, MaxAttemptsExceeded,
)
from maestro.observe import (
    MaestroObserver, InMemoryObserver, CompositeObserver, LoggingObserver, timed,
)
from maestro.validate import (
    Schema, field, Required, Range, Length, Pattern, OneOf, Custom, NotEmpty,
    ValidatedFacts, SchemaFilter, SchemaProcessor, ValidatedWork, SchemaViolation,
)

SEP = "═" * 62


# ════════════════════════════════════════════════════════════════════
#  1.  maestro.retry — resilience policies
# ════════════════════════════════════════════════════════════════════
print(SEP); print("1. maestro.retry"); print(SEP)

# ── 1a. Exponential backoff (no actual sleep for demo) ─────────────
print("  1a. Exponential backoff delays (simulated, no sleep):")
eb = ExponentialBackoff(base=0.5, multiplier=2.0, max_delay=30.0)
for attempt in range(1, 7):
    print(f"    attempt {attempt}: wait {eb.delay(attempt):.2f}s")

# ── 1b. RetryWork: succeed on 3rd attempt ──────────────────────────
print("\n  1b. RetryWork — succeed after 2 failures:")
calls = [0]
def flaky_execute(ctx):
    calls[0] += 1
    if calls[0] < 3:
        print(f"    attempt {calls[0]}: FAILED (simulating transient error)")
        raise ConnectionError("service temporarily unavailable")
    print(f"    attempt {calls[0]}: OK")
    return maestro.DefaultWorkReport(maestro.WorkStatus.COMPLETED, ctx)

work = maestro.LambdaWork(flaky_execute, name="flaky-api")
resilient = retry(work, max_attempts=5, backoff=ConstantBackoff(0.0),
                  on=[ConnectionError])
report = resilient.execute(maestro.WorkContext())
print(f"    → status: {report.status.value}  total calls: {calls[0]}")

# ── 1c. Circuit breaker ────────────────────────────────────────────
print("\n  1c. Circuit breaker — opens after 2 failures:")
cb = CircuitBreaker(threshold=2, reset_after=0.05, name="payment-service")
print(f"    Initial state: {cb.state.value}")

for i in range(3):
    if cb.allow_call():
        cb.on_failure()
        print(f"    Call {i+1}: failed, circuit state: {cb.state.value}")
    else:
        print(f"    Call {i+1}: BLOCKED — circuit is {cb.state.value}")

time.sleep(0.06)
print(f"    After reset_after: allow_call={cb.allow_call()}, state={cb.state.value}")
cb.on_success()
print(f"    After success: {cb.state.value}")

# ── 1d. @retryable decorator ───────────────────────────────────────
print("\n  1d. @retryable decorator:")
attempt_count = [0]

@retryable(max_attempts=4, backoff=ConstantBackoff(0.0))
class UnstableApiWork(maestro.Work):
    def execute(self, ctx):
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise TimeoutError(f"timeout on attempt {attempt_count[0]}")
        ctx.put("api_result", "success")
        return maestro.DefaultWorkReport(maestro.WorkStatus.COMPLETED, ctx)

ctx = maestro.WorkContext()
r = UnstableApiWork().execute(ctx)
print(f"    Completed in {attempt_count[0]} attempts. Result: {ctx.get('api_result')}")

# ── 1e. MaxAttemptsExceeded ────────────────────────────────────────
print("\n  1e. MaxAttemptsExceeded after exhaustion:")
def always_fail(ctx): raise RuntimeError("always broken")
rw = retry(maestro.LambdaWork(always_fail, "broken"), max_attempts=3,
           backoff=ConstantBackoff(0.0))
r = rw.execute(maestro.WorkContext())
print(f"    Status: {r.status.value}  error type: {type(r.error).__name__}")
print(f"    Original error: {r.error.last_error}")

# ── 1f. Inside a flow with ConditionalFlow ─────────────────────────
print("\n  1f. RetryWork inside a ConditionalFlow:")
service_calls = [0]
def call_service(ctx):
    service_calls[0] += 1
    if service_calls[0] < 2: raise ValueError("service down")
    return maestro.DefaultWorkReport(maestro.WorkStatus.COMPLETED, ctx)

flow = (maestro.aNewConditionalFlow()
        .execute(retry(maestro.LambdaWork(call_service, "svc"),
                       max_attempts=3, backoff=ConstantBackoff(0.0)))
        .when(maestro.WorkReportPredicate.COMPLETED)
        .then(maestro.LambdaWork(lambda c: print("    → service call succeeded!"), "ok"))
        .otherwise(maestro.LambdaWork(lambda c: print("    → service call FAILED"), "fail"))
        .build())
maestro.WorkFlowEngine().run(flow, maestro.WorkContext())


# ════════════════════════════════════════════════════════════════════
#  2.  maestro.observe — unified observability
# ════════════════════════════════════════════════════════════════════
print(); print(SEP); print("2. maestro.observe"); print(SEP)

mem = InMemoryObserver()
obs = MaestroObserver(observers=[mem])

# ── 2a. Instrument rules engine ────────────────────────────────────
print("  2a. Observe rules engine:")

@maestro.rule(name="high-value", priority=1)
class HighValueRule:
    @maestro.condition
    def check(self, f): return f.get("total", 0) > 500
    @maestro.action
    def tag(self, f): f.put("tier", "premium")

@maestro.rule(name="free-shipping", priority=2)
class FreeShippingRule:
    @maestro.condition
    def check(self, f): return f.get("total", 0) > 100
    @maestro.action
    def apply(self, f): f.put("shipping", 0.0)

rules   = maestro.Rules(HighValueRule(), FreeShippingRule())
engine  = obs.instrument_rules_engine(maestro.DefaultRulesEngine())
engine.fire(rules, maestro.Facts(total=750))
engine.fire(rules, maestro.Facts(total=50))

print(f"    rule evaluations: {mem.counter('rules', 'rule_evaluated', rule='high-value'):.0f} (high-value)")
print(f"    rule firings:     {mem.counter('rules', 'rule_fired',     rule='high-value'):.0f} (high-value fired)")
print(f"    engine fires:     {mem.counter('rules', 'engine_fire'):.0f}")

# ── 2b. Instrument batch job ───────────────────────────────────────
print("\n  2b. Observe batch job:")
sink = []
job = (maestro.JobBuilder()
       .named("observed-job")
       .reader(maestro.IterableRecordReader(range(1, 11)))
       .filter(maestro.PredicateRecordFilter(lambda r: r.payload % 2 == 0))
       .writer(maestro.CollectionRecordWriter(sink))
       .batch_size(3)
       .build())
job = obs.instrument_job(job)
maestro.JobExecutor().execute(job)

print(f"    records total:    {mem.gauge('batch', 'records_total',    job='observed-job'):.0f}")
print(f"    records written:  {mem.gauge('batch', 'records_written',  job='observed-job'):.0f}")
print(f"    records filtered: {mem.gauge('batch', 'records_filtered', job='observed-job'):.0f}")
snap = mem.histogram("batch", "batch_size")
print(f"    batch sizes:      count={snap['count']:.0f} mean={snap['mean']:.1f}")

# ── 2c. Instrument workflow ────────────────────────────────────────
print("\n  2c. Observe flow:")
flow2 = (maestro.aNewSequentialFlow()
         .named("observed-flow")
         .execute(maestro.LambdaWork(lambda c: time.sleep(0.001), "step-1"))
         .then(maestro.LambdaWork(lambda c: None, "step-2"))
         .build())
observed_flow = obs.instrument_flow(flow2)
observed_flow.execute(maestro.WorkContext())

flow_evs = mem.events(module="flows", name="flow_duration_seconds")
print(f"    flow durations recorded: {len(flow_evs)}")
print(f"    completed events:        {len(mem.events(module='flows', name='flow_COMPLETED')) + mem.counter('flows', 'flow_completed'):.0f}")

# ── 2d. Instrument FSM ────────────────────────────────────────────
print("\n  2d. Observe FSM:")
pending   = maestro.State("PENDING")
paid      = maestro.State("PAID")
shipped   = maestro.State("SHIPPED")

class PayEvt(maestro.Event): pass
class ShipEvt(maestro.Event): pass

fsm = (maestro.FiniteStateMachineBuilder(
           states={pending, paid, shipped}, initial_state=pending)
       .register_transition(maestro.TransitionBuilder().source_state(pending).event_type(PayEvt).target_state(paid).build())
       .register_transition(maestro.TransitionBuilder().source_state(paid).event_type(ShipEvt).target_state(shipped).build())
       .build())
fsm = obs.instrument_fsm(fsm)
fsm.fire(PayEvt())
fsm.fire(ShipEvt())

transition_evs = mem.events(module="states", name="transition_completed")
print(f"    transitions fired: {len(transition_evs)}")
state_evs = mem.events(module="states", name="state_entered")
states_visited = [e.labels.get("state") for e in state_evs]
print(f"    states entered:    {states_visited}")

# ── 2e. timed context manager ─────────────────────────────────────
print("\n  2e. timed context manager:")
with timed("flows", "custom_computation", mem, operation="matrix-multiply"):
    time.sleep(0.005)

snap2 = mem.histogram("flows", "custom_computation", operation="matrix-multiply")
print(f"    custom timing: {snap2['mean']*1000:.1f}ms")

# ── 2f. Prometheus export ─────────────────────────────────────────
print("\n  2f. Prometheus export (excerpt):")
prom = mem.export_prometheus()
for line in prom.split('\n')[:8]:
    if line: print(f"    {line}")

# ── 2g. Summary ───────────────────────────────────────────────────
print("\n  2g. Full summary:")
for line in mem.summary().split('\n'):
    print(f"    {line}")


# ════════════════════════════════════════════════════════════════════
#  3.  maestro.validate — schema validation
# ════════════════════════════════════════════════════════════════════
print(); print(SEP); print("3. maestro.validate"); print(SEP)

order_schema = Schema(
    id       = field(int,   Required()),
    status   = field(str,   OneOf("pending", "paid", "shipped", "cancelled")),
    total    = field(float, Range(min=0.0, max=1_000_000.0), coerce=True),
    email    = field(str,   Pattern(r"^[^@]+@[^@]+\.[^@]+$"), required=False),
    items    = field(list,  Length(min=1), required=True),
)

# ── 3a. Valid order ────────────────────────────────────────────────
print("  3a. Valid order:")
valid_order = {"id": 1, "status": "paid", "total": 299.99,
               "email": "alice@example.com", "items": ["book", "pen"]}
result = order_schema.validate(valid_order)
print(f"    Result: {'✓ valid' if result.ok else '✗ invalid'}")

# ── 3b. Invalid order ─────────────────────────────────────────────
print("\n  3b. Invalid order (multiple errors):")
bad_order = {"id": None, "status": "unknown", "total": -10,
             "email": "not-an-email", "items": []}
result = order_schema.validate(bad_order)
print(f"    Result: {'✓ valid' if result.ok else f'✗ invalid ({len(result.errors)} errors)'}")
for err in result.errors:
    print(f"      {err}")

# ── 3c. Type coercion ─────────────────────────────────────────────
print("\n  3c. Type coercion (string → float):")
raw = {"id": 1, "status": "pending", "total": "149.5", "items": ["laptop"]}
try:
    coerced = order_schema.coerce(raw)
    print(f"    total coerced: {raw['total']!r} → {coerced['total']!r} (type: {type(coerced['total']).__name__})")
except SchemaViolation as e:
    print(f"    coercion failed: {e}")

# ── 3d. ValidatedFacts ────────────────────────────────────────────
print("\n  3d. ValidatedFacts — validation on every put:")
person_schema = Schema(
    name = field(str,   NotEmpty(), Length(max=100)),
    age  = field(int,   Range(min=0, max=150)),
    role = field(str,   OneOf("admin", "user", "guest"), required=False, default="guest"),
)
vf = ValidatedFacts(person_schema, name="Alice", age=30)
vf.put("role", "admin")
print(f"    name={vf.get('name')} age={vf.get('age')} role={vf.get('role')}")
vf.put("age", -1)   # warns but stores (non-strict mode)
print(f"    after invalid put: age={vf.get('age')} (stored despite warning)")

# ── 3e. SchemaFilter in ETL pipeline ─────────────────────────────
print("\n  3e. SchemaFilter — drop invalid records in batch pipeline:")
raw_orders = [
    {"id": 1, "status": "paid",    "total": 100.0, "items": ["book"]},
    {"id": 2, "status": "INVALID", "total": 200.0, "items": ["pen"]},   # bad status
    {"id": 3, "status": "pending", "total": -50.0, "items": ["lamp"]},  # negative total
    {"id": 4, "status": "shipped", "total": 99.0,  "items": ["mug"]},
]
valid_sink = []
report = (maestro.JobBuilder()
          .named("validated-orders")
          .reader(maestro.IterableRecordReader(raw_orders))
          .filter(SchemaFilter(order_schema))
          .writer(maestro.CollectionRecordWriter(valid_sink))
          .build()).call()
print(f"    input:    {len(raw_orders)} records")
print(f"    valid:    {len(valid_sink)} records")
print(f"    filtered: {report.metrics.filtered_count} records")

# ── 3f. SchemaProcessor — coerce + validate in pipeline ──────────
print("\n  3f. SchemaProcessor — coerce types and validate in-place:")
coerce_schema = Schema(
    id    = field(int,   Required(),    coerce=True),
    total = field(float, Range(min=0),  coerce=True),
    tag   = field(str,   required=False, default="standard"),
)
raw_records = [
    {"id": "10", "total": "250.0"},           # valid after coercion
    {"id": "11", "total": "bad-number"},      # will fail coercion → skipped
    {"id": "12", "total": "75.5", "tag": "express"},
]
coerced_sink = []
report2 = (maestro.JobBuilder()
           .named("coerce-pipeline")
           .reader(maestro.IterableRecordReader(raw_records))
           .processor(SchemaProcessor(coerce_schema, coerce=True))
           .writer(maestro.CollectionRecordWriter(coerced_sink))
           .build()).call()
for rec in coerced_sink:
    print(f"    id={rec['id']} (type={type(rec['id']).__name__}) "
          f"total={rec['total']} tag={rec.get('tag', 'standard')}")
print(f"    skipped: {report2.metrics.skipped_count}")

# ── 3g. ValidatedWork in a flow ───────────────────────────────────
print("\n  3g. ValidatedWork — validate WorkContext before/after a step:")
pre_schema  = Schema(order_id=field(str, Required()), amount=field(float, Range(min=0)))
post_schema = Schema(order_id=field(str), receipt_id=field(str, Required()))

def process_payment(ctx):
    ctx.put("receipt_id", f"REC-{ctx.get('order_id')}-001")

validated_step = ValidatedWork(
    work        = maestro.LambdaWork(process_payment, "process-payment"),
    pre_schema  = pre_schema,
    post_schema = post_schema,
)

flow3 = maestro.SequentialFlow.Builder().execute(validated_step).build()
r = maestro.WorkFlowEngine().run(flow3, maestro.WorkContext(order_id="ORD-42", amount=99.0))
print(f"    status: {r.status.value}")


# ════════════════════════════════════════════════════════════════════
#  4.  All three together — production-grade pipeline
# ════════════════════════════════════════════════════════════════════
print(); print(SEP); print("4. All three P1 features together"); print(SEP)
print("  Pattern: validate input → retry flaky service → observe everything\n")

mem2    = InMemoryObserver()
obs2    = MaestroObserver(observers=[mem2])
cb2     = CircuitBreaker(threshold=3, reset_after=30, name="payment-cb")

payment_schema = Schema(
    order_id = field(str,   Required(), NotEmpty()),
    amount   = field(float, Range(min=0.01, max=50_000.0), coerce=True),
    currency = field(str,   OneOf("USD", "EUR", "CLP"), required=False, default="USD"),
)

service_calls2 = [0]
def call_payment_api(ctx):
    service_calls2[0] += 1
    if service_calls2[0] < 2:
        raise TimeoutError("payment gateway timeout")
    ctx.put("payment_status", "approved")
    ctx.put("transaction_id", f"TXN-{ctx.get('order_id')}")

payment_step = ValidatedWork(
    work = retry(
        maestro.LambdaWork(call_payment_api, "payment-api"),
        max_attempts    = 3,
        backoff         = ConstantBackoff(0.0),
        on              = [TimeoutError],
        circuit_breaker = cb2,
        on_retry        = lambda a, d, e: print(f"    retry #{a}: {e}"),
    ),
    pre_schema  = payment_schema,
    post_schema = Schema(
        payment_status = field(str,   OneOf("approved", "declined")),
        transaction_id = field(str,   Required()),
    ),
)

flow4 = obs2.instrument_flow(
    maestro.aNewSequentialFlow()
    .named("payment-pipeline")
    .execute(payment_step)
    .then(maestro.LambdaWork(
        lambda c: print(f"    payment {c.get('payment_status')}: {c.get('transaction_id')}"),
        "notify"
    ))
    .build()
)

r4 = flow4.execute(maestro.WorkContext(order_id="ORD-100", amount="199.5", currency="USD"))
print(f"\n  Final status: {r4.status.value}")
print(f"  Service calls: {service_calls2[0]}  (1 retry = 2 calls)")
print(f"  Circuit breaker: {cb2.state.value}")
print(f"  Flow metrics: {len(mem2.events(module='flows'))} events recorded")

print(); print(SEP); print("All P1 feature examples completed."); print(SEP)
