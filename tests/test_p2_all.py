"""tests/test_p2_all.py — Priority 2 feature tests (events, graph, async)"""
import sys, os, asyncio, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import maestro
from maestro.events import (
    Message, EventBus, AsyncEventBus, Topic, Subscription,
    EventPublisherWork, EventSubscriberWork, FSMEventBridge,
    RuleEventRouter, BusRecordReader,
)
from maestro.graph import (
    GraphBuilder, GraphFlow, GraphReport,
    CyclicDependencyError, UnknownDependencyError,
)
from maestro.async_ import (
    AsyncLambdaWork, AsyncNoOpWork, AsyncSequentialFlow,
    AsyncConditionalFlow, AsyncParallelFlow, AsyncRepeatFlow,
    AsyncWorkFlowEngine, AsyncIterableReader, AsyncCollectionWriter,
    AsyncJobBuilder, sync_to_async, async_to_sync,
)


# ══════════════════════════════════════════════════════════════════════
#  maestro.events
# ══════════════════════════════════════════════════════════════════════

class TestEventBus:
    def test_publish_subscribe(self):
        bus = EventBus()
        received = []
        bus.subscribe_fn("orders", lambda msg: received.append(msg.payload))
        bus.publish("orders", {"id": 1})
        assert received == [{"id": 1}]

    def test_multiple_subscribers(self):
        bus = EventBus()
        log = []
        bus.subscribe_fn("t", lambda m: log.append("A"))
        bus.subscribe_fn("t", lambda m: log.append("B"))
        bus.publish("t", None)
        assert sorted(log) == ["A", "B"]

    def test_wildcard_subscriber(self):
        bus = EventBus()
        log = []
        bus.subscribe_fn("*", lambda m: log.append(m.topic))
        bus.publish("x", 1)
        bus.publish("y", 2)
        assert "x" in log and "y" in log

    def test_cancel_subscription(self):
        bus = EventBus()
        log = []
        sub = bus.subscribe_fn("t", lambda m: log.append(1))
        bus.publish("t", None)
        sub.cancel()
        bus.publish("t", None)
        assert log == [1]

    def test_subscription_context_manager(self):
        bus = EventBus()
        log = []
        with bus.subscribe_fn("t", lambda m: log.append(1)):
            bus.publish("t", None)
        bus.publish("t", None)   # after __exit__, should not fire
        assert log == [1]

    def test_subscribe_once(self):
        bus = EventBus()
        log = []
        bus.subscribe_once("t", lambda m: log.append(m.payload))
        bus.publish("t", "first")
        bus.publish("t", "second")
        assert log == ["first"]

    def test_wait_for_returns_message(self):
        bus = EventBus()
        def delayed():
            time.sleep(0.02)
            bus.publish("ping", "pong")
        threading.Thread(target=delayed, daemon=True).start()
        msg = bus.wait_for("ping", timeout=1.0)
        assert msg is not None
        assert msg.payload == "pong"

    def test_wait_for_timeout_returns_none(self):
        bus = EventBus()
        msg = bus.wait_for("never", timeout=0.02)
        assert msg is None

    def test_stats(self):
        bus = EventBus()
        bus.subscribe_fn("t", lambda m: None)
        bus.publish("t", 1)
        stats = bus.stats
        assert stats["published"] == 1
        assert stats["delivered"] == 1
        assert "t" in stats["topics"]

    def test_error_in_subscriber_doesnt_break_others(self):
        bus = EventBus()
        log = []
        bus.subscribe_fn("t", lambda m: (_ for _ in ()).throw(RuntimeError("bad")))
        # rewrite cleanly
        def bad(m): raise RuntimeError("bad")
        def good(m): log.append("ok")
        bus2 = EventBus()
        bus2.subscribe_fn("t", bad)
        bus2.subscribe_fn("t", good)
        bus2.publish("t", None)
        assert log == ["ok"]

    def test_clear_topic(self):
        bus = EventBus()
        log = []
        bus.subscribe_fn("t", lambda m: log.append(1))
        bus.clear("t")
        bus.publish("t", None)
        assert log == []


class TestTopic:
    def test_typed_topic(self):
        bus    = EventBus()
        orders = Topic[dict]("orders.created")
        log = []
        orders.subscribe(bus, lambda m: log.append(m.payload["id"]))
        orders.publish(bus, {"id": 42})
        assert log == [42]

    def test_topic_wait_for(self):
        bus    = EventBus()
        pings  = Topic[str]("ping")
        def send():
            time.sleep(0.02)
            pings.publish(bus, "hello")
        threading.Thread(target=send, daemon=True).start()
        msg = pings.wait_for(bus, timeout=1.0)
        assert msg.payload == "hello"


class TestAsyncEventBus:
    def test_async_delivery(self):
        log = []
        with AsyncEventBus() as bus:
            bus.subscribe_fn("t", lambda m: log.append(m.payload))
            bus.publish("t", "hello")
            time.sleep(0.05)
        assert log == ["hello"]


class TestEventPublisherWork:
    def test_publishes_on_execute(self):
        bus = EventBus()
        log = []
        bus.subscribe_fn("order.done", lambda m: log.append(m.payload))

        work = EventPublisherWork(bus, "order.done", {"result": "ok"})
        report = work.execute(maestro.WorkContext())
        assert report.status == maestro.WorkStatus.COMPLETED
        assert log == [{"result": "ok"}]

    def test_dynamic_payload(self):
        bus = EventBus()
        log = []
        bus.subscribe_fn("result", lambda m: log.append(m.payload))

        work = EventPublisherWork(
            bus, "result",
            payload=lambda ctx: {"x": ctx.get("x")},
        )
        work.execute(maestro.WorkContext(x=99))
        assert log == [{"x": 99}]

    def test_inside_sequential_flow(self):
        bus = EventBus()
        log = []
        bus.subscribe_fn("step.done", lambda m: log.append("published"))

        flow = (maestro.SequentialFlow.Builder()
                .execute(maestro.LambdaWork(lambda c: c.put("x", 1), "set"))
                .then(EventPublisherWork(bus, "step.done", lambda c: c.get("x")))
                .build())
        maestro.WorkFlowEngine().run(flow, maestro.WorkContext())
        assert log == ["published"]


class TestEventSubscriberWork:
    def test_receives_message(self):
        bus = EventBus()
        def send():
            time.sleep(0.02)
            bus.publish("payment.ok", {"txn": "TXN-1"})
        threading.Thread(target=send, daemon=True).start()

        work   = EventSubscriberWork(bus, "payment.ok", timeout=1.0, context_key="payment")
        ctx    = maestro.WorkContext()
        report = work.execute(ctx)
        assert report.status == maestro.WorkStatus.COMPLETED
        assert ctx.get("payment") == {"txn": "TXN-1"}

    def test_timeout_returns_failed(self):
        bus  = EventBus()
        work = EventSubscriberWork(bus, "ghost.topic", timeout=0.02)
        r    = work.execute(maestro.WorkContext())
        assert r.status == maestro.WorkStatus.FAILED


class TestFSMEventBridge:
    def test_transition_published_to_bus(self):
        bus = EventBus()
        log = []
        bus.subscribe_fn("fsm.order.transitioned", lambda m: log.append(m.payload))

        locked, unlocked = maestro.State("locked"), maestro.State("unlocked")
        class Coin(maestro.Event): pass

        bridge = FSMEventBridge(bus, topic_prefix="fsm.order")
        fsm = (maestro.FiniteStateMachineBuilder(states={locked, unlocked}, initial_state=locked)
               .register_transition(
                   maestro.TransitionBuilder().source_state(locked).event_type(Coin).target_state(unlocked).build())
               .register_listener(bridge)
               .build())
        fsm.fire(Coin())

        assert len(log) == 1
        assert log[0]["to"] == "unlocked"
        assert log[0]["event"] == "Coin"


class TestRuleEventRouter:
    def test_routes_matching_message(self):
        bus = EventBus()
        premium_log = []
        standard_log = []
        bus.subscribe_fn("orders.premium",  lambda m: premium_log.append(m.payload))
        bus.subscribe_fn("orders.standard", lambda m: standard_log.append(m.payload))

        route_rule = (maestro.RuleBuilder()
                      .name("premium")
                      .when(lambda f: f.get("total", 0) > 500)
                      .then(lambda f: f.put("route", "orders.premium"))
                      .build())
        router = RuleEventRouter(bus, maestro.Rules(route_rule),
                                 default_topic="orders.standard")
        bus.subscribe("orders.all", router)

        bus.publish("orders.all", {"total": 750})
        bus.publish("orders.all", {"total": 100})

        assert premium_log  == [{"total": 750}]
        assert standard_log == [{"total": 100}]


class TestBusRecordReader:
    def test_reads_messages_as_records(self):
        bus    = EventBus()
        reader = BusRecordReader(bus, "events", max_records=2, timeout=1.0)
        reader.open()

        bus.publish("events", {"id": 1})
        bus.publish("events", {"id": 2})

        r1 = reader.read_record()
        r2 = reader.read_record()
        r3 = reader.read_record()  # max_records reached → None

        assert r1.payload == {"id": 1}
        assert r2.payload == {"id": 2}
        assert r3 is None
        reader.close()

    def test_in_batch_pipeline(self):
        bus  = EventBus()
        sink = []
        reader = BusRecordReader(bus, "data", max_records=3, timeout=1.0)

        def delayed_publish():
            time.sleep(0.05)
            for v in [10, 20, 30]:
                bus.publish("data", v)
        threading.Thread(target=delayed_publish, daemon=True).start()

        (maestro.JobBuilder()
         .named("bus-job")
         .reader(reader)
         .writer(maestro.CollectionRecordWriter(sink))
         .build()).call()

        assert sorted(sink) == [10, 20, 30]


# ══════════════════════════════════════════════════════════════════════
#  maestro.graph
# ══════════════════════════════════════════════════════════════════════

def _work(name, side_effect=None):
    def fn(ctx):
        if side_effect: side_effect(name)
        return maestro.DefaultWorkReport(maestro.WorkStatus.COMPLETED, ctx)
    return maestro.LambdaWork(fn, name=name)

def _fail_work(name):
    return maestro.LambdaWork(
        lambda ctx: maestro.DefaultWorkReport(maestro.WorkStatus.FAILED, ctx, error=Exception(name)),
        name=name,
    )


class TestGraphBuilder:
    def test_single_node(self):
        log = []
        flow = GraphBuilder().add("only", _work("only", log.append)).build()
        r = maestro.WorkFlowEngine().run(flow, maestro.WorkContext())
        assert r.status == maestro.WorkStatus.COMPLETED
        assert "only" in log

    def test_sequential_chain(self):
        log = []
        flow = (GraphBuilder()
                .add("A", _work("A", log.append))
                .add("B", _work("B", log.append), depends_on=["A"])
                .add("C", _work("C", log.append), depends_on=["B"])
                .build())
        r = maestro.WorkFlowEngine().run(flow, maestro.WorkContext())
        assert r.status == maestro.WorkStatus.COMPLETED
        assert log.index("A") < log.index("B") < log.index("C")

    def test_parallel_independent_nodes(self):
        import time
        timings = {}
        def timed_work(name):
            def fn(ctx):
                t = time.monotonic()
                time.sleep(0.05)
                timings[name] = t
                return maestro.DefaultWorkReport(maestro.WorkStatus.COMPLETED, ctx)
            return maestro.LambdaWork(fn, name=name)

        flow = (GraphBuilder()
                .add("A", timed_work("A"))
                .add("B", timed_work("B"))
                .build())
        t0 = time.monotonic()
        r  = maestro.WorkFlowEngine().run(flow, maestro.WorkContext())
        elapsed = time.monotonic() - t0
        # Both ran in parallel, so total < 0.09s (not 0.10s sequential)
        assert elapsed < 0.09
        assert r.status == maestro.WorkStatus.COMPLETED

    def test_diamond_dependency(self):
        """A → B, A → C, B+C → D"""
        log = []
        flow = (GraphBuilder()
                .add("A", _work("A", log.append))
                .add("B", _work("B", log.append), depends_on=["A"])
                .add("C", _work("C", log.append), depends_on=["A"])
                .add("D", _work("D", log.append), depends_on=["B", "C"])
                .build())
        r = maestro.WorkFlowEngine().run(flow, maestro.WorkContext())
        assert r.status == maestro.WorkStatus.COMPLETED
        assert log.index("A") < log.index("B")
        assert log.index("A") < log.index("C")
        assert log.index("B") < log.index("D")
        assert log.index("C") < log.index("D")

    def test_node_reports_populated(self):
        flow = (GraphBuilder()
                .add("A", _work("A"))
                .add("B", _work("B"), depends_on=["A"])
                .build())
        r = maestro.WorkFlowEngine().run(flow, maestro.WorkContext())
        assert isinstance(r, GraphReport)
        assert "A" in r.node_reports
        assert "B" in r.node_reports
        assert r.node_reports["A"].succeeded
        assert r.node_reports["B"].duration >= 0.0

    def test_fail_fast_stops_on_first_failure(self):
        log = []
        flow = (GraphBuilder()
                .fail_fast(True)
                .add("A", _fail_work("A"))
                .add("B", _work("B", log.append), depends_on=["A"])
                .build())
        r = maestro.WorkFlowEngine().run(flow, maestro.WorkContext())
        assert r.status == maestro.WorkStatus.FAILED
        assert "B" not in log

    def test_no_fail_fast_collects_all(self):
        log = []
        flow = (GraphBuilder()
                .fail_fast(False)
                .add("A", _fail_work("A"))
                .add("B", _work("B", log.append))
                .build())
        r = maestro.WorkFlowEngine().run(flow, maestro.WorkContext())
        assert r.status == maestro.WorkStatus.FAILED
        assert "B" in log  # B has no deps, still runs

    def test_cycle_detected_at_build_time(self):
        with pytest.raises(CyclicDependencyError):
            (GraphBuilder()
             .add("A", _work("A"), depends_on=["B"])
             .add("B", _work("B"), depends_on=["A"])
             .build())

    def test_unknown_dependency_raises(self):
        with pytest.raises(UnknownDependencyError):
            GraphBuilder().add("A", _work("A"), depends_on=["GHOST"]).build()

    def test_empty_graph_raises(self):
        with pytest.raises(ValueError):
            GraphBuilder().build()

    def test_context_shared_between_nodes(self):
        def write(ctx): ctx.put("written", True)
        def read(ctx):
            assert ctx.get("written") is True
            return maestro.DefaultWorkReport(maestro.WorkStatus.COMPLETED, ctx)
        flow = (GraphBuilder()
                .add("writer", maestro.LambdaWork(write, "writer"))
                .add("reader", maestro.LambdaWork(read, "reader"), depends_on=["writer"])
                .build())
        r = maestro.WorkFlowEngine().run(flow, maestro.WorkContext())
        assert r.status == maestro.WorkStatus.COMPLETED

    def test_to_dot_export(self):
        flow = (GraphBuilder()
                .add("A", _work("A"))
                .add("B", _work("B"), depends_on=["A"])
                .build())
        dot = flow.to_dot()
        assert '"A" -> "B"' in dot

    def test_succeeded_and_failed_nodes_properties(self):
        flow = (GraphBuilder()
                .fail_fast(False)
                .add("ok",   _work("ok"))
                .add("fail", _fail_work("fail"))
                .build())
        r = maestro.WorkFlowEngine().run(flow, maestro.WorkContext())
        assert "ok"   in r.succeeded_nodes
        assert "fail" in r.failed_nodes


# ══════════════════════════════════════════════════════════════════════
#  maestro.async_
# ══════════════════════════════════════════════════════════════════════

class TestAsyncSequentialFlow:
    def test_all_steps_run(self):
        log = []
        async def run():
            flow = (AsyncSequentialFlow.Builder()
                    .execute(AsyncLambdaWork(lambda ctx: log.append("a")))
                    .then(AsyncLambdaWork(lambda ctx: log.append("b")))
                    .build())
            return await AsyncWorkFlowEngine().run(flow, maestro.WorkContext())
        r = asyncio.run(run())
        assert r.status == maestro.WorkStatus.COMPLETED
        assert log == ["a", "b"]

    def test_stops_on_failure(self):
        log = []
        async def run():
            async def fail(ctx):
                return maestro.DefaultWorkReport(maestro.WorkStatus.FAILED, ctx)
            flow = (AsyncSequentialFlow.Builder()
                    .execute(AsyncLambdaWork(fail, "fail"))
                    .then(AsyncLambdaWork(lambda ctx: log.append("c"), "c"))
                    .build())
            return await AsyncWorkFlowEngine().run(flow, maestro.WorkContext())
        r = asyncio.run(run())
        assert r.status == maestro.WorkStatus.FAILED
        assert "c" not in log

    def test_context_threaded_through(self):
        async def run():
            flow = (AsyncSequentialFlow.Builder()
                    .execute(AsyncLambdaWork(lambda ctx: ctx.put("x", 1)))
                    .then(AsyncLambdaWork(lambda ctx: ctx.put("x", ctx.get("x") + 1)))
                    .build())
            ctx = maestro.WorkContext()
            await AsyncWorkFlowEngine().run(flow, ctx)
            return ctx
        ctx = asyncio.run(run())
        assert ctx.get("x") == 2


class TestAsyncConditionalFlow:
    def test_then_branch(self):
        log = []
        async def run():
            flow = (AsyncConditionalFlow.Builder()
                    .execute(AsyncNoOpWork())
                    .when(maestro.WorkReportPredicate.COMPLETED)
                    .then(AsyncLambdaWork(lambda ctx: log.append("then")))
                    .otherwise(AsyncLambdaWork(lambda ctx: log.append("else")))
                    .build())
            return await AsyncWorkFlowEngine().run(flow, maestro.WorkContext())
        asyncio.run(run())
        assert log == ["then"]

    def test_otherwise_branch(self):
        log = []
        async def run():
            async def fail(ctx):
                return maestro.DefaultWorkReport(maestro.WorkStatus.FAILED, ctx)
            flow = (AsyncConditionalFlow.Builder()
                    .execute(AsyncLambdaWork(fail, "fail"))
                    .when(maestro.WorkReportPredicate.COMPLETED)
                    .then(AsyncLambdaWork(lambda ctx: log.append("then")))
                    .otherwise(AsyncLambdaWork(lambda ctx: log.append("else")))
                    .build())
            return await AsyncWorkFlowEngine().run(flow, maestro.WorkContext())
        asyncio.run(run())
        assert log == ["else"]


class TestAsyncParallelFlow:
    def test_all_run_concurrently(self):
        log = []
        lock = asyncio.Lock()
        async def run():
            async def w(name):
                async def fn(ctx):
                    async with lock: log.append(name)
                return AsyncLambdaWork(fn, name)
            flow = (AsyncParallelFlow.Builder()
                    .execute(await w("A"), await w("B"), await w("C"))
                    .build())
            return await AsyncWorkFlowEngine().run(flow, maestro.WorkContext())
        asyncio.run(run())
        assert sorted(log) == ["A", "B", "C"]

    def test_status_completed_when_all_succeed(self):
        async def run():
            flow = AsyncParallelFlow.Builder().execute(AsyncNoOpWork(), AsyncNoOpWork()).build()
            return await AsyncWorkFlowEngine().run(flow, maestro.WorkContext())
        r = asyncio.run(run())
        assert r.status == maestro.WorkStatus.COMPLETED


class TestAsyncRepeatFlow:
    def test_fixed_times(self):
        log = []
        async def run():
            flow = (AsyncRepeatFlow.Builder()
                    .repeat(AsyncLambdaWork(lambda ctx: log.append(1)))
                    .times(4)
                    .build())
            return await AsyncWorkFlowEngine().run(flow, maestro.WorkContext())
        asyncio.run(run())
        assert len(log) == 4

    def test_until_predicate(self):
        async def run():
            ctx = maestro.WorkContext(n=0)
            async def inc(c): c.put("n", c.get("n") + 1)
            stop = maestro.WorkReportPredicate(lambda r: r.work_context.get("n", 0) >= 3)
            flow = (AsyncRepeatFlow.Builder()
                    .repeat(AsyncLambdaWork(inc, "inc"))
                    .until(stop)
                    .times(10)
                    .build())
            await AsyncWorkFlowEngine().run(flow, ctx)
            return ctx
        ctx = asyncio.run(run())
        assert ctx.get("n") == 3


class TestAsyncJob:
    def test_basic_pipeline(self):
        async def run():
            sink = []
            job  = (AsyncJobBuilder()
                    .named("async-test")
                    .reader(AsyncIterableReader([1, 2, 3]))
                    .writer(AsyncCollectionWriter(sink))
                    .build())
            report = await job.call()
            return sink, report
        sink, report = asyncio.run(run())
        assert sink == [1, 2, 3]
        assert report.status.value == "COMPLETED"
        assert report.metrics.written_count == 3

    def test_filter_in_async_job(self):
        async def run():
            sink = []
            job  = (AsyncJobBuilder()
                    .reader(AsyncIterableReader(range(1, 6)))
                    .filter(maestro.PredicateRecordFilter(lambda r: r.payload % 2 == 0))
                    .writer(AsyncCollectionWriter(sink))
                    .build())
            report = await job.call()
            return sink, report
        sink, report = asyncio.run(run())
        assert sink == [1, 3, 5]
        assert report.metrics.filtered_count == 2


class TestSyncAsyncAdapters:
    def test_sync_to_async(self):
        sync_work = maestro.LambdaWork(lambda ctx: ctx.put("done", True), "sync")
        async_work = sync_to_async(sync_work)
        async def run():
            ctx = maestro.WorkContext()
            r   = await async_work.execute(ctx)
            return ctx
        ctx = asyncio.run(run())
        assert ctx.get("done") is True

    def test_async_to_sync(self):
        async def async_fn(ctx):
            ctx.put("async_done", True)
        async_work = AsyncLambdaWork(async_fn, "async")
        sync_work  = async_to_sync(async_work)

        ctx = maestro.WorkContext()
        r   = sync_work.execute(ctx)
        assert ctx.get("async_done") is True

    def test_async_engine_run_sync(self):
        log = []
        flow = AsyncSequentialFlow.Builder().execute(
            AsyncLambdaWork(lambda ctx: log.append("ok"))
        ).build()
        r = AsyncWorkFlowEngine().run_sync(flow, maestro.WorkContext())
        assert r.status == maestro.WorkStatus.COMPLETED
        assert log == ["ok"]

    def test_mix_sync_and_async_in_flow(self):
        """Sync Work inside an AsyncSequentialFlow via sync_to_async."""
        log = []
        sync_work  = maestro.LambdaWork(lambda ctx: log.append("sync"), "sync")
        async_work = AsyncLambdaWork(lambda ctx: log.append("async"), "async")
        async def run():
            flow = (AsyncSequentialFlow.Builder()
                    .execute(sync_to_async(sync_work))
                    .then(async_work)
                    .build())
            return await AsyncWorkFlowEngine().run(flow, maestro.WorkContext())
        asyncio.run(run())
        assert log == ["sync", "async"]
