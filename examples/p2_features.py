"""
examples/p2_features.py — Priority 2 features: events, graph, async.

Run: python examples/p2_features.py
"""
import sys, os, asyncio, threading, time, concurrent.futures
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import maestro
from maestro.events import (
    EventBus, AsyncEventBus, Topic, FSMEventBridge,
    RuleEventRouter, EventPublisherWork, EventSubscriberWork, BusRecordReader,
)
from maestro.graph import GraphBuilder, GraphReport
from maestro.async_ import (
    AsyncLambdaWork, AsyncNoOpWork, AsyncSequentialFlow,
    AsyncConditionalFlow, AsyncParallelFlow, AsyncRepeatFlow,
    AsyncWorkFlowEngine, AsyncIterableReader, AsyncCollectionWriter,
    AsyncJobBuilder, sync_to_async, async_to_sync,
)

SEP = "═" * 62


# ════════════════════════════════════════════════════════════════════
#  1.  maestro.events
# ════════════════════════════════════════════════════════════════════
print(SEP); print("1. maestro.events — reactive event bus"); print(SEP)

# ── 1a. Basic pub/sub ─────────────────────────────────────────────
print("  1a. Basic publish/subscribe:")
bus = EventBus()
orders = Topic[dict]("orders.created")
log1 = []

with orders.subscribe(bus, lambda m: log1.append(f"  subscriber A got: {m.payload}")):
    with orders.subscribe(bus, lambda m: log1.append(f"  subscriber B got: {m.payload}")):
        orders.publish(bus, {"id": 1, "total": 150.0})
        orders.publish(bus, {"id": 2, "total": 500.0})

for line in log1: print(line)
print(f"  stats: {bus.stats}")

# ── 1b. Wildcard subscription ─────────────────────────────────────
print("\n  1b. Wildcard (*) subscription:")
bus2   = EventBus()
caught = []
bus2.subscribe_fn("*", lambda m: caught.append(f"[{m.topic}] {m.payload}"))
bus2.publish("user.login",  "alice")
bus2.publish("order.placed", 42)
bus2.publish("payment.ok",  {"txn": "TXN-1"})
for c in caught: print(f"    {c}")

# ── 1c. wait_for — synchronous rendezvous ─────────────────────────
print("\n  1c. wait_for — receive one message from another thread:")
bus3 = EventBus()
def delayed_payment():
    time.sleep(0.05)
    bus3.publish("payment.confirmed", {"txn": "TXN-42", "amount": 99.0})
threading.Thread(target=delayed_payment, daemon=True).start()
msg = bus3.wait_for("payment.confirmed", timeout=2.0)
print(f"    received: {msg.payload}")

# ── 1d. FSMEventBridge — FSM → bus ────────────────────────────────
print("\n  1d. FSMEventBridge — FSM transitions published to bus:")
bus4 = EventBus()
fsm_log = []
bus4.subscribe_fn("order.fsm.transitioned", lambda m: fsm_log.append(
    f"  {m.payload['from']} --[{m.payload['event']}]--> {m.payload['to']}"))
bus4.subscribe_fn("order.fsm.entered.PAID", lambda m: print("  [hook] payment received!"))

pending, paid, shipped = maestro.State("PENDING"), maestro.State("PAID"), maestro.State("SHIPPED")
class Pay(maestro.Event): pass
class Ship(maestro.Event): pass

bridge = FSMEventBridge(bus4, topic_prefix="order.fsm")
fsm = (maestro.FiniteStateMachineBuilder(states={pending, paid, shipped}, initial_state=pending)
       .register_transition(maestro.TransitionBuilder().source_state(pending).event_type(Pay).target_state(paid).build())
       .register_transition(maestro.TransitionBuilder().source_state(paid).event_type(Ship).target_state(shipped).build())
       .register_listener(bridge)
       .build())

fsm.fire(Pay())
fsm.fire(Ship())
for line in fsm_log: print(line)

# ── 1e. RuleEventRouter ───────────────────────────────────────────
print("\n  1e. RuleEventRouter — rules decide which topic to route to:")
bus5 = EventBus()
premium_log = []; standard_log = []
bus5.subscribe_fn("orders.premium",  lambda m: premium_log.append(m.payload["id"]))
bus5.subscribe_fn("orders.standard", lambda m: standard_log.append(m.payload["id"]))

route_rule = (maestro.RuleBuilder()
              .name("vip-route")
              .when(lambda f: f.get("total", 0) > 500)
              .then(lambda f: f.put("route", "orders.premium"))
              .build())
router = RuleEventRouter(bus5, maestro.Rules(route_rule), default_topic="orders.standard")
bus5.subscribe("orders.all", router)

for order in [{"id": 1, "total": 750}, {"id": 2, "total": 100}, {"id": 3, "total": 1200}]:
    bus5.publish("orders.all", order)

print(f"  premium orders: {premium_log}")
print(f"  standard orders: {standard_log}")

# ── 1f. EventPublisherWork + EventSubscriberWork in a flow ─────────
print("\n  1f. EventPublisherWork + EventSubscriberWork in coordinated flows:")
bus6 = EventBus()
results = []

def producer_flow():
    flow = (maestro.aNewSequentialFlow()
            .named("producer")
            .execute(maestro.LambdaWork(lambda c: c.put("result", 42), "compute"))
            .then(EventPublisherWork(bus6, "results.ready",
                                    lambda c: {"value": c.get("result")}))
            .build())
    maestro.WorkFlowEngine().run(flow, maestro.WorkContext())

def consumer_flow():
    flow = (maestro.aNewSequentialFlow()
            .named("consumer")
            .execute(EventSubscriberWork(bus6, "results.ready", timeout=2.0,
                                         context_key="result"))
            .then(maestro.LambdaWork(
                lambda c: results.append(c.get("result", {}).get("value")), "process"))
            .build())
    maestro.WorkFlowEngine().run(flow, maestro.WorkContext())

t_prod = threading.Thread(target=lambda: (time.sleep(0.02), producer_flow()), daemon=True)
t_cons = threading.Thread(target=consumer_flow, daemon=True)
t_cons.start(); t_prod.start()
t_cons.join(timeout=3.0); t_prod.join(timeout=3.0)
print(f"  consumer received: {results}")

# ── 1g. AsyncEventBus ─────────────────────────────────────────────
print("\n  1g. AsyncEventBus — background-thread delivery:")
async_log = []
with AsyncEventBus() as async_bus:
    async_bus.subscribe_fn("events", lambda m: async_log.append(m.payload))
    for i in range(5):
        async_bus.publish("events", f"event-{i}")
    time.sleep(0.05)
print(f"  delivered {len(async_log)} messages async: {async_log}")


# ════════════════════════════════════════════════════════════════════
#  2.  maestro.graph — DAG workflow
# ════════════════════════════════════════════════════════════════════
print(); print(SEP); print("2. maestro.graph — DAG workflow execution"); print(SEP)

# ── 2a. Simple DAG ────────────────────────────────────────────────
print("  2a. DAG with parallel independent steps:")
timings = {}
def timed_work(name, sleep_s=0.0):
    def fn(ctx):
        t = time.monotonic()
        if sleep_s: time.sleep(sleep_s)
        timings[name] = time.monotonic() - t
        ctx.put(f"{name}_done", True)
        return maestro.DefaultWorkReport(maestro.WorkStatus.COMPLETED, ctx)
    return maestro.LambdaWork(fn, name=name)

flow2 = (GraphBuilder()
         .named("data-pipeline")
         .add("fetch-users",   timed_work("fetch-users",  0.04))
         .add("fetch-orders",  timed_work("fetch-orders", 0.03))
         .add("fetch-prefs",   timed_work("fetch-prefs",  0.02))
         .add("join",          timed_work("join"),          depends_on=["fetch-users", "fetch-orders"])
         .add("enrich",        timed_work("enrich"),        depends_on=["join", "fetch-prefs"])
         .add("export-csv",    timed_work("export-csv"),    depends_on=["enrich"])
         .add("export-json",   timed_work("export-json"),   depends_on=["enrich"])
         .add("notify",        timed_work("notify"),        depends_on=["export-csv", "export-json"])
         .build())

t0 = time.monotonic()
r2 = maestro.WorkFlowEngine().run(flow2, maestro.WorkContext())
total = time.monotonic() - t0

print(f"  Status: {r2.status.value}")
print(f"  Nodes: {len(r2.node_reports)} completed")
print(f"  Total wall time: {total*1000:.0f}ms (would be >{0.04+0.03+0.02:.0f}ms sequential)")
print(f"  Parallel wins: fetches ran concurrently, exports ran concurrently")

# ── 2b. Per-node report ───────────────────────────────────────────
print("\n  2b. Per-node reports:")
for name, nr in sorted(r2.node_reports.items(), key=lambda x: x[1].completion_order):
    print(f"    [{nr.completion_order}] {name:<20} {nr.status.value:<12} {nr.duration*1000:.1f}ms")

# ── 2c. DOT export ────────────────────────────────────────────────
print("\n  2c. DOT export (paste into graphviz):")
print(flow2.to_dot())

# ── 2d. fail_fast vs collect-all ─────────────────────────────────
print("\n  2d. fail_fast=False — continue running non-blocked nodes after failure:")
completed_nodes = []
flow_ff = (GraphBuilder()
           .named("partial")
           .fail_fast(False)
           .add("A", maestro.LambdaWork(lambda c: (completed_nodes.append("A"),
                maestro.DefaultWorkReport(maestro.WorkStatus.FAILED, c))[1], "A"))
           .add("B", maestro.LambdaWork(lambda c: (completed_nodes.append("B"),
                maestro.DefaultWorkReport(maestro.WorkStatus.COMPLETED, c))[1], "B"))
           .add("C", timed_work("C"), depends_on=["A"])  # blocked because A failed
           .build())
r_ff = maestro.WorkFlowEngine().run(flow_ff, maestro.WorkContext())
print(f"  status: {r_ff.status.value}  completed: {completed_nodes}  "
      f"succeeded: {r_ff.succeeded_nodes}  failed: {r_ff.failed_nodes}")


# ════════════════════════════════════════════════════════════════════
#  3.  maestro.async_ — AsyncIO native
# ════════════════════════════════════════════════════════════════════
print(); print(SEP); print("3. maestro.async_ — AsyncIO-native execution"); print(SEP)

async def p2_async_demo():
    # ── 3a. AsyncSequentialFlow ────────────────────────────────────
    print("  3a. AsyncSequentialFlow with real async I/O (simulated):")
    log3a = []

    async def fetch_user(ctx):
        await asyncio.sleep(0.01)  # simulate async HTTP
        ctx.put("user", {"id": 1, "name": "Alice"})
        log3a.append("fetched user")

    async def fetch_orders(ctx):
        await asyncio.sleep(0.01)
        ctx.put("orders", [{"id": 10}, {"id": 11}])
        log3a.append("fetched orders")

    async def compute(ctx):
        user   = ctx.get("user", {})
        orders = ctx.get("orders", [])
        ctx.put("summary", f"{user.get('name')} has {len(orders)} orders")
        log3a.append("computed summary")

    flow3a = (AsyncSequentialFlow.Builder()
              .named("user-pipeline")
              .execute(AsyncLambdaWork(fetch_user,   "fetch-user"))
              .then(AsyncLambdaWork(fetch_orders, "fetch-orders"))
              .then(AsyncLambdaWork(compute,      "compute"))
              .build())

    ctx3a = maestro.WorkContext()
    r3a   = await AsyncWorkFlowEngine().run(flow3a, ctx3a)
    print(f"    steps: {log3a}")
    print(f"    result: {ctx3a.get('summary')}")
    print(f"    status: {r3a.status.value}")

    # ── 3b. AsyncParallelFlow — true concurrency ───────────────────
    print("\n  3b. AsyncParallelFlow — concurrent async calls:")
    t0 = time.monotonic()
    timings3b = {}

    async def slow_io(name, delay):
        async def fn(ctx):
            await asyncio.sleep(delay)
            timings3b[name] = time.monotonic() - t0
        return AsyncLambdaWork(fn, name=name)

    par_flow = (AsyncParallelFlow.Builder()
                .execute(
                    await slow_io("fetch-A", 0.04),
                    await slow_io("fetch-B", 0.03),
                    await slow_io("fetch-C", 0.02),
                )
                .build())
    await AsyncWorkFlowEngine().run(par_flow, maestro.WorkContext())
    elapsed = time.monotonic() - t0
    print(f"    3 fetches (40ms+30ms+20ms) ran concurrently in {elapsed*1000:.0f}ms "
          f"(sequential would be 90ms+)")

    # ── 3c. AsyncConditionalFlow ───────────────────────────────────
    print("\n  3c. AsyncConditionalFlow:")
    log3c = []

    async def check_stock(ctx):
        await asyncio.sleep(0.005)
        in_stock = ctx.get("quantity", 0) > 0
        status   = maestro.WorkStatus.COMPLETED if in_stock else maestro.WorkStatus.FAILED
        return maestro.DefaultWorkReport(status, ctx)

    async def reserve(ctx):  log3c.append("reserved item")
    async def backorder(ctx): log3c.append("added to backorder")

    for qty in (5, 0):
        flow3c = (AsyncConditionalFlow.Builder()
                  .execute(AsyncLambdaWork(check_stock, "check"))
                  .when(maestro.WorkReportPredicate.COMPLETED)
                  .then(AsyncLambdaWork(reserve,   "reserve"))
                  .otherwise(AsyncLambdaWork(backorder, "backorder"))
                  .build())
        await AsyncWorkFlowEngine().run(flow3c, maestro.WorkContext(quantity=qty))

    print(f"    results: {log3c}")

    # ── 3d. AsyncRepeatFlow — retry with async backoff ─────────────
    print("\n  3d. AsyncRepeatFlow — retry until success:")
    attempts = [0]
    async def flaky_api(ctx):
        attempts[0] += 1
        await asyncio.sleep(0.005)
        if attempts[0] < 3:
            return maestro.DefaultWorkReport(maestro.WorkStatus.FAILED, ctx)
        ctx.put("api_result", "success")

    stop_on_success = maestro.WorkReportPredicate.COMPLETED
    flow3d = (AsyncRepeatFlow.Builder()
              .repeat(AsyncLambdaWork(flaky_api, "api"))
              .until(stop_on_success)
              .times(5)
              .build())
    ctx3d = maestro.WorkContext()
    await AsyncWorkFlowEngine().run(flow3d, ctx3d)
    print(f"    API succeeded on attempt {attempts[0]}, result={ctx3d.get('api_result')}")

    # ── 3e. AsyncJob ───────────────────────────────────────────────
    print("\n  3e. AsyncJob — async ETL pipeline:")
    sink3e = []

    async def async_transform(record):
        await asyncio.sleep(0.001)
        return str(record.payload).upper()

    class TransformMapper:
        def map_record(self, record):
            record.payload = record.payload.upper()
            return record

    job3e = (AsyncJobBuilder()
             .named("async-etl")
             .reader(AsyncIterableReader(["hello", "world", "maestro"]))
             .mapper(TransformMapper())
             .writer(AsyncCollectionWriter(sink3e))
             .batch_size(2)
             .build())
    report3e = await job3e.call()
    print(f"    transformed: {sink3e}")
    print(f"    written: {report3e.metrics.written_count}")

    # ── 3f. sync ↔ async adapters ──────────────────────────────────
    print("\n  3f. Adapters — mix sync and async freely:")
    log3f = []

    # Sync Work inside async flow (runs in thread pool)
    sync_step = maestro.LambdaWork(
        lambda ctx: log3f.append("sync-step-ran"), "sync")

    # Async Work in sync flow (via asyncio.run)
    async def async_fn(ctx): log3f.append("async-step-ran")
    async_step_for_sync = async_to_sync(AsyncLambdaWork(async_fn, "async"))

    # Async flow with sync step
    async_flow = (AsyncSequentialFlow.Builder()
                  .execute(sync_to_async(sync_step))   # sync in async flow
                  .then(AsyncLambdaWork(async_fn, "async2"))
                  .build())
    await AsyncWorkFlowEngine().run(async_flow, maestro.WorkContext())

    # Sync flow with async step
    maestro.WorkFlowEngine().run(
        maestro.SequentialFlow.Builder().execute(async_step_for_sync).build(),
        maestro.WorkContext()
    )
    print(f"    execution log: {log3f}")

asyncio.run(p2_async_demo())


# ════════════════════════════════════════════════════════════════════
#  4. All three P2 together: event-driven DAG with async steps
# ════════════════════════════════════════════════════════════════════
print(); print(SEP); print("4. All three P2 features together"); print(SEP)
print("  Pattern: async DAG steps publish events → FSM reacts → bus routes results\n")

async def combined_demo():
    """
    All three P2 features working together:
    - Async parallel flow (async_) fetches data concurrently
    - GraphFlow (graph) resolves step order with dependencies
    - EventBus (events) wires results to downstream consumers
    """
    bus7    = EventBus()
    results7 = []
    bus7.subscribe_fn("pipeline.completed", lambda m: results7.append(m.payload))

    # Step 1: async parallel fetches
    log = {}
    async def fetch_a(ctx):
        await asyncio.sleep(0.01)
        log["a"] = [1, 2, 3]

    async def fetch_b(ctx):
        await asyncio.sleep(0.01)
        log["b"] = [4, 5, 6]

    par = (AsyncParallelFlow.Builder()
           .execute(AsyncLambdaWork(fetch_a, "fetch-a"),
                    AsyncLambdaWork(fetch_b, "fetch-b"))
           .build())
    r_par = await AsyncWorkFlowEngine().run(par, maestro.WorkContext())
    print(f"  parallel fetch status: {r_par.status.value}")
    print(f"  fetched: a={log.get('a')} b={log.get('b')}")

    # Step 2: graph DAG — join → enrich → publish (all sync, running in GraphFlow threads)
    def join(ctx):      ctx.put("merged", log.get("a", []) + log.get("b", []))
    def enrich(ctx):    ctx.put("count",  len(ctx.get("merged", [])))
    def publish_fn(ctx):
        bus7.publish("pipeline.completed", {
            "count": ctx.get("count"), "source": "graph"
        })

    graph = (GraphBuilder()
             .named("dag-pipeline")
             .add("join",    maestro.LambdaWork(join, "join"))
             .add("enrich",  maestro.LambdaWork(enrich, "enrich"),   depends_on=["join"])
             .add("publish", maestro.LambdaWork(publish_fn, "pub"),  depends_on=["enrich"])
             .build())

    ctx_g = maestro.WorkContext()
    r_g   = maestro.WorkFlowEngine().run(graph, ctx_g)
    print(f"  graph status: {r_g.status.value}  merged count: {ctx_g.get('count')}")
    print(f"  bus received: {results7}")

asyncio.run(combined_demo())

print(); print(SEP); print("All P2 feature examples completed."); print(SEP)
