"""
tests/test_easy_states.py — comprehensive unit tests for the easy_states port.

Run:  python -m pytest tests/test_easy_states.py -v
"""
import sys, os


import pytest

from maestro.states import (
    State, Event, AbstractEvent,
    EventHandler, LambdaEventHandler,
    Transition, TransitionBuilder,
    FiniteStateMachine, FiniteStateMachineBuilder,
    TransitionListener,
    NoSuchTransitionException, FSMException,
)


# ─────────────────────────────────────────────────────────────────── #
#  Shared fixtures                                                    #
# ─────────────────────────────────────────────────────────────────── #

class CoinEvent(Event): pass
class PushEvent(Event): pass
class ResetEvent(Event): pass


def make_turnstile(ignore_undefined=False, listeners=None):
    locked   = State("locked")
    unlocked = State("unlocked")

    unlock_t = (
        TransitionBuilder().name("unlock")
        .source_state(locked).event_type(CoinEvent).target_state(unlocked).build()
    )
    push_locked_t = (
        TransitionBuilder().name("push-locked")
        .source_state(locked).event_type(PushEvent).target_state(locked).build()
    )
    lock_t = (
        TransitionBuilder().name("lock")
        .source_state(unlocked).event_type(PushEvent).target_state(locked).build()
    )
    coin_unlocked_t = (
        TransitionBuilder().name("coin-unlocked")
        .source_state(unlocked).event_type(CoinEvent).target_state(unlocked).build()
    )

    builder = (
        FiniteStateMachineBuilder(states={locked, unlocked}, initial_state=locked)
        .register_transition(unlock_t)
        .register_transition(push_locked_t)
        .register_transition(lock_t)
        .register_transition(coin_unlocked_t)
        .ignore_undefined_transitions(ignore_undefined)
    )
    for lst in (listeners or []):
        builder.register_listener(lst)
    return builder.build(), locked, unlocked


# ─────────────────────────────────────────────────────────────────── #
#  State                                                              #
# ─────────────────────────────────────────────────────────────────── #

class TestState:
    def test_name(self):
        s = State("active")
        assert s.name == "active"

    def test_equality_by_name(self):
        assert State("a") == State("a")
        assert State("a") != State("b")

    def test_hashable(self):
        s1 = State("x")
        s2 = State("x")
        assert {s1, s2} == {State("x")}

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            State("")
        with pytest.raises(ValueError):
            State("   ")

    def test_str(self):
        assert str(State("locked")) == "locked"


# ─────────────────────────────────────────────────────────────────── #
#  Event                                                              #
# ─────────────────────────────────────────────────────────────────── #

class TestEvent:
    def test_event_type_is_class(self):
        assert CoinEvent.event_type() is CoinEvent

    def test_timestamp(self):
        import datetime
        e = CoinEvent()
        assert isinstance(e.timestamp, datetime.datetime)

    def test_abstract_event_alias(self):
        assert AbstractEvent is Event

    def test_subclasses_are_distinct(self):
        assert type(CoinEvent()) is not type(PushEvent())


# ─────────────────────────────────────────────────────────────────── #
#  EventHandler                                                       #
# ─────────────────────────────────────────────────────────────────── #

class TestEventHandler:
    def test_lambda_handler(self):
        called = []
        h = LambdaEventHandler(lambda e: called.append(e))
        ev = CoinEvent()
        h.handle(ev)
        assert called == [ev]

    def test_subclass_handler(self):
        log = []

        class MyHandler(EventHandler):
            def handle(self, event):
                log.append("handled")

        MyHandler().handle(CoinEvent())
        assert log == ["handled"]


# ─────────────────────────────────────────────────────────────────── #
#  Transition / TransitionBuilder                                     #
# ─────────────────────────────────────────────────────────────────── #

class TestTransition:
    def test_build(self):
        s, t = State("s"), State("t")
        tr = (
            TransitionBuilder().name("test")
            .source_state(s).event_type(CoinEvent).target_state(t).build()
        )
        assert tr.name == "test"
        assert tr.source_state == s
        assert tr.event_type is CoinEvent
        assert tr.target_state == t
        assert tr.event_handler is None

    def test_with_handler(self):
        h = LambdaEventHandler(lambda e: None)
        s, t = State("s"), State("t")
        tr = (
            TransitionBuilder().source_state(s).event_type(CoinEvent)
            .event_handler(h).target_state(t).build()
        )
        assert tr.event_handler is h

    def test_missing_source_raises(self):
        with pytest.raises(ValueError):
            TransitionBuilder().event_type(CoinEvent).target_state(State("t")).build()

    def test_missing_event_raises(self):
        with pytest.raises(ValueError):
            TransitionBuilder().source_state(State("s")).target_state(State("t")).build()

    def test_missing_target_raises(self):
        with pytest.raises(ValueError):
            TransitionBuilder().source_state(State("s")).event_type(CoinEvent).build()

    def test_java_compat_names(self):
        s, t = State("s"), State("t")
        tr = (
            TransitionBuilder()
            .sourceState(s).eventType(CoinEvent).targetState(t).build()
        )
        assert tr.source_state == s

    def test_equality_by_source_and_event(self):
        s, t1, t2 = State("s"), State("t1"), State("t2")
        tr1 = TransitionBuilder().source_state(s).event_type(CoinEvent).target_state(t1).build()
        tr2 = TransitionBuilder().source_state(s).event_type(CoinEvent).target_state(t2).build()
        assert tr1 == tr2  # same (source, event_type) key

    def test_hashable(self):
        s, t = State("s"), State("t")
        tr = TransitionBuilder().source_state(s).event_type(CoinEvent).target_state(t).build()
        assert tr in {tr}


# ─────────────────────────────────────────────────────────────────── #
#  FiniteStateMachine                                                 #
# ─────────────────────────────────────────────────────────────────── #

class TestFiniteStateMachine:
    def test_initial_state(self):
        fsm, locked, _ = make_turnstile()
        assert fsm.current_state == locked

    def test_coin_unlocks(self):
        fsm, locked, unlocked = make_turnstile()
        new_state = fsm.fire(CoinEvent())
        assert new_state == unlocked
        assert fsm.current_state == unlocked

    def test_push_when_locked_stays_locked(self):
        fsm, locked, _ = make_turnstile()
        fsm.fire(PushEvent())
        assert fsm.current_state == locked

    def test_push_when_unlocked_locks(self):
        fsm, locked, unlocked = make_turnstile()
        fsm.fire(CoinEvent())   # unlock
        fsm.fire(PushEvent())   # lock
        assert fsm.current_state == locked

    def test_coin_when_unlocked_stays_unlocked(self):
        fsm, _, unlocked = make_turnstile()
        fsm.fire(CoinEvent())   # unlock
        fsm.fire(CoinEvent())   # extra coin
        assert fsm.current_state == unlocked

    def test_full_turnstile_sequence(self):
        fsm, locked, unlocked = make_turnstile()
        assert fsm.current_state == locked
        fsm.fire(PushEvent());  assert fsm.current_state == locked
        fsm.fire(CoinEvent());  assert fsm.current_state == unlocked
        fsm.fire(CoinEvent());  assert fsm.current_state == unlocked
        fsm.fire(PushEvent());  assert fsm.current_state == locked

    def test_no_transition_raises(self):
        fsm, _, _ = make_turnstile()
        with pytest.raises(NoSuchTransitionException):
            fsm.fire(ResetEvent())

    def test_ignore_undefined_no_raise(self):
        fsm, locked, _ = make_turnstile(ignore_undefined=True)
        state = fsm.fire(ResetEvent())
        assert state == locked

    def test_event_handler_is_called(self):
        called = []
        s = State("s")
        t = State("t")
        tr = (
            TransitionBuilder().source_state(s).event_type(CoinEvent)
            .event_handler(LambdaEventHandler(lambda e: called.append(type(e).__name__)))
            .target_state(t).build()
        )
        fsm = FiniteStateMachineBuilder(states={s, t}, initial_state=s).register_transition(tr).build()
        fsm.fire(CoinEvent())
        assert called == ["CoinEvent"]

    def test_initial_state_not_in_set_raises(self):
        with pytest.raises(FSMException):
            FiniteStateMachineBuilder(
                states={State("a")}, initial_state=State("b")
            ).build()

    def test_duplicate_transition_raises(self):
        s, t = State("s"), State("t")
        tr1 = TransitionBuilder().source_state(s).event_type(CoinEvent).target_state(t).build()
        tr2 = TransitionBuilder().source_state(s).event_type(CoinEvent).target_state(s).build()
        with pytest.raises(FSMException):
            FiniteStateMachineBuilder(states={s, t}, initial_state=s) \
                .register_transition(tr1).register_transition(tr2).build()

    def test_states_property(self):
        fsm, locked, unlocked = make_turnstile()
        assert locked in fsm.states
        assert unlocked in fsm.states

    def test_transitions_property(self):
        fsm, _, _ = make_turnstile()
        assert len(fsm.transitions) == 4

    def test_java_register_compat(self):
        s, t = State("s"), State("t")
        tr = TransitionBuilder().source_state(s).event_type(CoinEvent).target_state(t).build()
        fsm = (
            FiniteStateMachineBuilder(states={s, t}, initial_state=s)
            .registerTransition(tr)
            .build()
        )
        fsm.fire(CoinEvent())
        assert fsm.current_state == t


# ─────────────────────────────────────────────────────────────────── #
#  TransitionListener                                                 #
# ─────────────────────────────────────────────────────────────────── #

class TestTransitionListener:
    def test_on_transition_started_and_ended(self):
        events_started = []
        events_ended   = []

        class L(TransitionListener):
            def on_transition_started(self, event, transition):
                events_started.append(transition.name)
            def on_transition_ended(self, event, transition, new_state):
                events_ended.append(new_state.name)

        fsm, _, unlocked = make_turnstile(listeners=[L()])
        fsm.fire(CoinEvent())
        assert events_started == ["unlock"]
        assert events_ended   == ["unlocked"]

    def test_on_no_transition_called_when_ignoring(self):
        no_trans = []

        class L(TransitionListener):
            def on_no_transition(self, event, current_state):
                no_trans.append(current_state.name)

        fsm, locked, _ = make_turnstile(ignore_undefined=True, listeners=[L()])
        fsm.fire(ResetEvent())
        assert no_trans == ["locked"]

    def test_on_handler_error_called(self):
        errors = []

        class L(TransitionListener):
            def on_handler_error(self, event, transition, exc):
                errors.append(str(exc))

        s, t = State("s"), State("t")

        def boom(e): raise RuntimeError("handler exploded")

        tr = (
            TransitionBuilder().source_state(s).event_type(CoinEvent)
            .event_handler(LambdaEventHandler(boom)).target_state(t).build()
        )
        fsm = (
            FiniteStateMachineBuilder(states={s, t}, initial_state=s)
            .register_transition(tr)
            .register_listener(L())
            .build()
        )
        # Handler error should NOT prevent state change
        fsm.fire(CoinEvent())
        assert fsm.current_state == t
        assert errors == ["handler exploded"]

    def test_add_listener_at_runtime(self):
        log = []

        class L(TransitionListener):
            def on_transition_ended(self, e, t, s): log.append(s.name)

        fsm, _, _ = make_turnstile()
        fsm.add_listener(L())
        fsm.fire(CoinEvent())
        assert "unlocked" in log


# ─────────────────────────────────────────────────────────────────── #
#  DOT export                                                         #
# ─────────────────────────────────────────────────────────────────── #

class TestDotExport:
    def test_dot_contains_states(self):
        fsm, locked, unlocked = make_turnstile()
        dot = fsm.to_dot("test")
        assert "locked" in dot
        assert "unlocked" in dot

    def test_dot_contains_transitions(self):
        fsm, _, _ = make_turnstile()
        dot = fsm.to_dot()
        assert "CoinEvent" in dot
        assert "PushEvent" in dot

    def test_dot_header(self):
        fsm, _, _ = make_turnstile()
        dot = fsm.to_dot("my_fsm")
        assert 'digraph "my_fsm"' in dot


# ─────────────────────────────────────────────────────────────────── #
#  Traffic light — multi-state cycle                                  #
# ─────────────────────────────────────────────────────────────────── #

class TestTrafficLight:
    def test_cycle(self):
        red    = State("red")
        green  = State("green")
        yellow = State("yellow")

        class Tick(Event): pass

        fsm = (
            FiniteStateMachineBuilder(states={red, green, yellow}, initial_state=red)
            .register_transition(
                TransitionBuilder().source_state(red).event_type(Tick).target_state(green).build()
            )
            .register_transition(
                TransitionBuilder().source_state(green).event_type(Tick).target_state(yellow).build()
            )
            .register_transition(
                TransitionBuilder().source_state(yellow).event_type(Tick).target_state(red).build()
            )
            .build()
        )

        seq = [red, green, yellow, red, green, yellow]
        for expected in seq:
            assert fsm.current_state == expected
            fsm.fire(Tick())
        assert fsm.current_state == red


# ─────────────────────────────────────────────────────────────────── #
#  Order lifecycle                                                    #
# ─────────────────────────────────────────────────────────────────── #

class TestOrderLifecycle:
    def _build_fsm(self):
        pending   = State("PENDING")
        paid      = State("PAID")
        shipped   = State("SHIPPED")
        delivered = State("DELIVERED")
        cancelled = State("CANCELLED")

        class Pay(Event): pass
        class Ship(Event): pass
        class Deliver(Event): pass
        class Cancel(Event): pass

        fsm = (
            FiniteStateMachineBuilder(
                states={pending, paid, shipped, delivered, cancelled},
                initial_state=pending,
            )
            .register_transition(TransitionBuilder().source_state(pending).event_type(Pay).target_state(paid).build())
            .register_transition(TransitionBuilder().source_state(paid).event_type(Ship).target_state(shipped).build())
            .register_transition(TransitionBuilder().source_state(shipped).event_type(Deliver).target_state(delivered).build())
            .register_transition(TransitionBuilder().source_state(pending).event_type(Cancel).target_state(cancelled).build())
            .register_transition(TransitionBuilder().source_state(paid).event_type(Cancel).target_state(cancelled).build())
            .build()
        )
        return fsm, pending, paid, shipped, delivered, cancelled, Pay, Ship, Deliver, Cancel

    def test_happy_path(self):
        fsm, pending, paid, shipped, delivered, cancelled, Pay, Ship, Deliver, Cancel = self._build_fsm()
        fsm.fire(Pay());     assert fsm.current_state == paid
        fsm.fire(Ship());    assert fsm.current_state == shipped
        fsm.fire(Deliver()); assert fsm.current_state == delivered

    def test_cancel_from_pending(self):
        fsm, pending, paid, shipped, delivered, cancelled, Pay, Ship, Deliver, Cancel = self._build_fsm()
        fsm.fire(Cancel())
        assert fsm.current_state == cancelled

    def test_cancel_from_paid(self):
        fsm, pending, paid, shipped, delivered, cancelled, Pay, Ship, Deliver, Cancel = self._build_fsm()
        fsm.fire(Pay())
        fsm.fire(Cancel())
        assert fsm.current_state == cancelled

    def test_cannot_cancel_after_delivery(self):
        fsm, pending, paid, shipped, delivered, cancelled, Pay, Ship, Deliver, Cancel = self._build_fsm()
        fsm.fire(Pay()); fsm.fire(Ship()); fsm.fire(Deliver())
        with pytest.raises(NoSuchTransitionException):
            fsm.fire(Cancel())
