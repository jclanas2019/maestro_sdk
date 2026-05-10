"""
maestro.events — lightweight reactive event bus.

Connects the four Maestro modules through a shared message channel so
components can react to each other without being directly coupled.

    from maestro.events import EventBus, Topic, Message

    bus   = EventBus()
    topic = Topic[dict]("orders")

    topic.subscribe(bus, lambda msg: print(f"new order: {msg.payload}"))
    topic.publish(bus, {"id": 1, "total": 99.0})

Integration bridges
-------------------
* ``EventPublisherWork``  — publish a message when a flow step runs
* ``EventSubscriberWork`` — receive and process the next message in a flow
* ``FSMEventBridge``      — FSM transitions → bus messages
* ``RuleEventRouter``     — rules engine decides which topic to route to
* ``BatchEventReader``    — consume bus messages as batch records
"""
from __future__ import annotations

import logging
import queue
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ════════════════════════════════════════════════════════════════════════════
#  Message
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Message(Generic[T]):
    """An envelope wrapping any payload published on the bus."""
    topic:     str
    payload:   T
    source:    str  = ""
    id:        str  = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=lambda: __import__("time").monotonic())

    def __repr__(self) -> str:
        return f"Message(topic={self.topic!r}, id={self.id}, payload={self.payload!r})"


# ════════════════════════════════════════════════════════════════════════════
#  Subscriber
# ════════════════════════════════════════════════════════════════════════════

class Subscriber(ABC, Generic[T]):
    """Abstract message subscriber. Override ``on_message``."""
    @abstractmethod
    def on_message(self, message: Message[T]) -> None: ...


class FunctionSubscriber(Subscriber[T]):
    """Wraps a plain callable as a subscriber."""
    def __init__(self, fn: Callable[[Message[T]], None]) -> None:
        self._fn = fn
    def on_message(self, message: Message[T]) -> None:
        self._fn(message)


# ════════════════════════════════════════════════════════════════════════════
#  Subscription handle
# ════════════════════════════════════════════════════════════════════════════

class Subscription:
    """Handle returned by ``subscribe`` — call ``cancel()`` to unsubscribe."""
    def __init__(self, bus: "EventBus", topic: str, subscriber: Subscriber) -> None:
        self._bus        = bus
        self._topic      = topic
        self._subscriber = subscriber
        self._active     = True

    def cancel(self) -> None:
        if self._active:
            self._bus._remove(self._topic, self._subscriber)
            self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(self, *_) -> None:
        self.cancel()


# ════════════════════════════════════════════════════════════════════════════
#  EventBus
# ════════════════════════════════════════════════════════════════════════════

class EventBus:
    """
    Thread-safe synchronous publish/subscribe event bus.

    * Subscribers are called in the publisher's thread.
    * Multiple subscribers per topic are supported.
    * Wildcard subscriptions via ``"*"`` match every topic.

    Usage::

        bus = EventBus()
        sub = bus.subscribe("orders", lambda msg: print(msg.payload))
        bus.publish("orders", {"id": 1})
        sub.cancel()
    """

    def __init__(self, name: str = "maestro-bus") -> None:
        self._name        = name
        self._subscribers: dict[str, list[Subscriber]] = {}
        self._lock        = threading.RLock()
        self._published   = 0
        self._delivered   = 0

    def publish(self, topic: str, payload: Any, source: str = "") -> Message:
        """Publish *payload* to *topic*. Calls all matching subscribers synchronously."""
        msg = Message(topic=topic, payload=payload, source=source)
        with self._lock:
            self._published += 1
            targets = list(self._subscribers.get(topic, []))
            wildcards = list(self._subscribers.get("*", []))

        for subscriber in targets + wildcards:
            try:
                subscriber.on_message(msg)
            except Exception as exc:
                # Isolate subscriber failures: log and continue to next subscriber
                logger.error("Subscriber error on topic %r: %s", topic, exc)
            else:
                with self._lock: self._delivered += 1

        logger.debug("Published to %r: %s", topic, msg.id)
        return msg

    def subscribe(self, topic: str, subscriber: Subscriber) -> Subscription:
        """Subscribe *subscriber* to *topic*. Returns a :class:`Subscription`."""
        with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            self._subscribers[topic].append(subscriber)
        return Subscription(self, topic, subscriber)

    def subscribe_fn(self, topic: str,
                     fn: Callable[[Message], None]) -> Subscription:
        """Subscribe a plain callable to *topic*."""
        return self.subscribe(topic, FunctionSubscriber(fn))

    def subscribe_once(self, topic: str,
                       fn: Callable[[Message], None]) -> Subscription:
        """Subscribe a callable that fires exactly once then auto-cancels."""
        sub_ref: list = [None]

        def handler(msg: Message) -> None:
            fn(msg)
            if sub_ref[0]: sub_ref[0].cancel()

        sub = self.subscribe_fn(topic, handler)
        sub_ref[0] = sub
        return sub

    def wait_for(self, topic: str, timeout: Optional[float] = None) -> Optional[Message]:
        """
        Block until one message arrives on *topic* and return it.
        Returns ``None`` if *timeout* elapses.

        Useful in tests or sequential orchestration::

            msg = bus.wait_for("payment.confirmed", timeout=5.0)
        """
        received: queue.Queue = queue.Queue(maxsize=1)

        def handler(msg: Message) -> None:
            received.put_nowait(msg)

        with self.subscribe_fn(topic, handler):
            try:
                return received.get(timeout=timeout)
            except queue.Empty:
                return None

    def _remove(self, topic: str, subscriber: Subscriber) -> None:
        with self._lock:
            subs = self._subscribers.get(topic, [])
            try: subs.remove(subscriber)
            except ValueError: pass

    def clear(self, topic: Optional[str] = None) -> None:
        """Remove all subscribers for *topic* (or all topics if None)."""
        with self._lock:
            if topic:
                self._subscribers.pop(topic, None)
            else:
                self._subscribers.clear()

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "published": self._published,
                "delivered": self._delivered,
                "topics":    list(self._subscribers.keys()),
                "subscriber_count": sum(len(v) for v in self._subscribers.values()),
            }


# ════════════════════════════════════════════════════════════════════════════
#  AsyncEventBus — background-thread delivery
# ════════════════════════════════════════════════════════════════════════════

class AsyncEventBus(EventBus):
    """
    EventBus variant that delivers messages in a dedicated background thread,
    decoupling publishers from slow subscribers.

    Call ``start()`` before use and ``stop()`` when done::

        bus = AsyncEventBus()
        bus.start()
        ...
        bus.stop()

    Or use as a context manager::

        with AsyncEventBus() as bus:
            bus.publish("topic", payload)
    """

    def __init__(self, name: str = "maestro-async-bus",
                 queue_maxsize: int = 1_000) -> None:
        super().__init__(name)
        self._q      = queue.Queue(maxsize=queue_maxsize)
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> "AsyncEventBus":
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True,
                                          name=f"{self._name}-worker")
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._running = False
        self._q.put_nowait(None)   # sentinel
        if self._thread: self._thread.join(timeout=timeout)

    def publish(self, topic: str, payload: Any, source: str = "") -> Message:
        msg = Message(topic=topic, payload=payload, source=source)
        with self._lock: self._published += 1
        try: self._q.put_nowait(msg)
        except queue.Full:
            logger.warning("AsyncEventBus queue full — dropping message on %r", topic)
        return msg

    def _loop(self) -> None:
        while self._running:
            try:
                msg = self._q.get(timeout=0.1)
                if msg is None: break
                self._deliver(msg)
            except queue.Empty:
                continue

    def _deliver(self, msg: Message) -> None:
        with self._lock:
            targets   = list(self._subscribers.get(msg.topic, []))
            wildcards = list(self._subscribers.get("*", []))
        for sub in targets + wildcards:
            try:
                sub.on_message(msg)
                with self._lock: self._delivered += 1
            except Exception as exc:
                logger.error("AsyncEventBus subscriber error: %s", exc)

    def __enter__(self) -> "AsyncEventBus":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()


# ════════════════════════════════════════════════════════════════════════════
#  Topic — typed channel helper
# ════════════════════════════════════════════════════════════════════════════

class Topic(Generic[T]):
    """
    A named, typed channel on an :class:`EventBus`.

    Provides a cleaner API than raw ``bus.publish/subscribe`` by coupling
    the topic name with its payload type::

        orders = Topic[dict]("orders.created")

        sub = orders.subscribe(bus, lambda msg: process(msg.payload))
        orders.publish(bus, {"id": 1, "total": 99.0})
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def publish(self, bus: EventBus, payload: T, source: str = "") -> Message[T]:
        return bus.publish(self._name, payload, source=source)

    def subscribe(self, bus: EventBus,
                  fn: Callable[[Message[T]], None]) -> Subscription:
        return bus.subscribe_fn(self._name, fn)

    def subscribe_once(self, bus: EventBus,
                       fn: Callable[[Message[T]], None]) -> Subscription:
        return bus.subscribe_once(self._name, fn)

    def wait_for(self, bus: EventBus,
                 timeout: Optional[float] = None) -> Optional[Message[T]]:
        return bus.wait_for(self._name, timeout=timeout)

    def __repr__(self) -> str:
        return f"Topic({self._name!r})"


# ════════════════════════════════════════════════════════════════════════════
#  Flow integration
# ════════════════════════════════════════════════════════════════════════════

from maestro.flows._work import DefaultWorkReport, Work, WorkContext, WorkReport, WorkStatus


class EventPublisherWork(Work):
    """
    Publish a message to the bus when this work step executes.

    The payload can be a static value or a callable
    ``(WorkContext) → payload`` to build it dynamically.

    Example::

        bus = EventBus()
        step = EventPublisherWork(
            bus     = bus,
            topic   = "order.processed",
            payload = lambda ctx: {"order_id": ctx.get("order_id")},
            name    = "notify-processed",
        )
    """

    def __init__(self, bus: EventBus, topic: str, payload: Any,
                 source: str = "flow", name: str = "event-publisher") -> None:
        self._bus     = bus
        self._topic   = topic
        self._payload = payload
        self._source  = source
        self._name    = name

    def get_name(self) -> str:
        return self._name

    def execute(self, work_context: WorkContext) -> WorkReport:
        payload = self._payload(work_context) if callable(self._payload) else self._payload
        msg = self._bus.publish(self._topic, payload, source=self._source)
        work_context.put("last_published_message_id", msg.id)
        logger.debug("Published to %r (id=%s)", self._topic, msg.id)
        return DefaultWorkReport(WorkStatus.COMPLETED, work_context)


class EventSubscriberWork(Work):
    """
    Wait for a message on *topic* and process it in a flow step.

    The message payload is stored in the WorkContext under *context_key*.
    Returns FAILED if the timeout elapses with no message.

    Example::

        step = EventSubscriberWork(
            bus         = bus,
            topic       = "payment.confirmed",
            timeout     = 10.0,
            context_key = "payment",
        )
    """

    def __init__(self, bus: EventBus, topic: str,
                 timeout: Optional[float] = None,
                 context_key: str = "received_message",
                 name: str = "event-subscriber") -> None:
        self._bus         = bus
        self._topic       = topic
        self._timeout     = timeout
        self._context_key = context_key
        self._name        = name

    def get_name(self) -> str:
        return self._name

    def execute(self, work_context: WorkContext) -> WorkReport:
        msg = self._bus.wait_for(self._topic, timeout=self._timeout)
        if msg is None:
            err = TimeoutError(
                f"No message on topic {self._topic!r} within {self._timeout}s"
            )
            return DefaultWorkReport(WorkStatus.FAILED, work_context, error=err)
        work_context.put(self._context_key, msg.payload)
        work_context.put(f"{self._context_key}_id", msg.id)
        return DefaultWorkReport(WorkStatus.COMPLETED, work_context)


# ════════════════════════════════════════════════════════════════════════════
#  FSM integration — bridge FSM transitions → bus
# ════════════════════════════════════════════════════════════════════════════

from maestro.states._listener import TransitionListener


class FSMEventBridge(TransitionListener):
    """
    Bridges FSM transition events to the event bus.

    Each state transition publishes a :class:`Message` with the payload::

        {"transition": name, "from": source, "to": target, "event": EventClassName}

    Example::

        bridge = FSMEventBridge(bus, topic_prefix="fsm.order")
        fsm    = FiniteStateMachineBuilder(...).register_listener(bridge).build()
        # Every fsm.fire(X) publishes to "fsm.order.transitioned"
    """

    def __init__(self, bus: EventBus, topic_prefix: str = "fsm",
                 source: str = "fsm") -> None:
        self._bus    = bus
        self._prefix = topic_prefix
        self._source = source

    def on_transition_ended(self, event, transition, new_state) -> None:
        payload = {
            "transition": transition.name,
            "from":       transition.source_state.name,
            "to":         new_state.name,
            "event":      type(event).__name__,
        }
        self._bus.publish(f"{self._prefix}.transitioned", payload, source=self._source)
        self._bus.publish(f"{self._prefix}.entered.{new_state.name}", payload,
                          source=self._source)

    def on_no_transition(self, event, current_state) -> None:
        self._bus.publish(f"{self._prefix}.undefined_transition", {
            "state": current_state.name, "event": type(event).__name__,
        }, source=self._source)


# ════════════════════════════════════════════════════════════════════════════
#  Rules integration — rules engine routes events to topics
# ════════════════════════════════════════════════════════════════════════════

class RuleEventRouter(Subscriber):
    """
    Routes incoming messages to different topics based on a rules engine.

    Each rule can read message fields via Facts and call ``facts.put("route", "topic-name")``
    to declare the destination topic. If no rule fires, the message is forwarded to
    *default_topic* (if set) or discarded.

    Example::

        from maestro.rules import Rules, RuleBuilder, Facts

        premium_rule = (RuleBuilder()
                        .name("route-premium")
                        .when(lambda f: f.get("total", 0) > 1000)
                        .then(lambda f: f.put("route", "orders.premium"))
                        .build())

        router = RuleEventRouter(
            bus          = bus,
            rules        = Rules(premium_rule),
            default_topic = "orders.standard",
        )
        bus.subscribe("orders.all", router)
    """

    def __init__(self, bus: EventBus, rules, default_topic: Optional[str] = None,
                 engine=None) -> None:
        from maestro.rules import DefaultRulesEngine, Facts as RulesFacts
        self._bus   = bus
        self._rules = rules
        self._default = default_topic
        self._engine  = engine or DefaultRulesEngine()
        self._Facts   = RulesFacts

    def on_message(self, message: Message) -> None:
        payload = message.payload if isinstance(message.payload, dict) else {"value": message.payload}
        facts   = self._Facts(**payload)
        self._engine.fire(self._rules, facts)

        route = facts.get("route")
        if route:
            self._bus.publish(route, message.payload, source=message.source)
        elif self._default:
            self._bus.publish(self._default, message.payload, source=message.source)
        else:
            logger.debug("RuleEventRouter: no route for message %s — discarded", message.id)


# ════════════════════════════════════════════════════════════════════════════
#  Batch integration — consume bus messages as Records
# ════════════════════════════════════════════════════════════════════════════

import queue as _queue

from maestro.batch._record import Header, Record
from maestro.batch._reader import RecordReader


class BusRecordReader(RecordReader):
    """
    Reads messages from a bus topic as batch :class:`Record` objects.

    The reader buffers arriving messages in a queue and exposes them as
    records — bridging real-time events into a batch pipeline.

    Args:
        bus:         The event bus to subscribe to.
        topic:       Topic to consume.
        max_records: Stop after this many records (``None`` = run until ``stop()``).
        timeout:     Seconds to wait for the next message before returning ``None``.

    Example::

        reader = BusRecordReader(bus, topic="orders.created", max_records=100)
        job    = JobBuilder().reader(reader).processor(enrich_proc).build()
    """

    def __init__(self, bus: EventBus, topic: str,
                 max_records: Optional[int] = None,
                 timeout: float = 5.0) -> None:
        self._bus         = bus
        self._topic       = topic
        self._max         = max_records
        self._timeout     = timeout
        self._q:    _queue.Queue = _queue.Queue()
        self._sub:  Optional[Subscription] = None
        self._count = 0

    @property
    def source_name(self) -> str:
        return f"bus://{self._topic}"

    def open(self) -> None:
        self._count = 0
        self._sub = self._bus.subscribe_fn(
            self._topic,
            lambda msg: self._q.put_nowait(msg),
        )

    def read_record(self) -> Optional[Record]:
        if self._max is not None and self._count >= self._max:
            return None
        try:
            msg = self._q.get(timeout=self._timeout)
            self._count += 1
            return Record(Header(self._count, self.source_name), msg.payload)
        except _queue.Empty:
            return None

    def close(self) -> None:
        if self._sub: self._sub.cancel(); self._sub = None


__all__ = [
    "Message", "Subscriber", "FunctionSubscriber", "Subscription",
    "EventBus", "AsyncEventBus", "Topic",
    "EventPublisherWork", "EventSubscriberWork",
    "FSMEventBridge", "RuleEventRouter",
    "BusRecordReader",
]
