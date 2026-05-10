"""
Core abstractions: State, Event, EventHandler, and exceptions.

Erlang formula: State(S) × Event(E) → Actions(A), State(S')

State
-----
A named node in the automaton::

    locked   = State("locked")
    unlocked = State("unlocked")

Event
-----
A signal that may trigger a transition. Subclass :class:`Event` (or
:class:`AbstractEvent`) to create typed events::

    class CoinEvent(Event): pass
    class PushEvent(Event): pass

    # Fire with an instance:
    machine.fire(CoinEvent())

EventHandler
------------
Optional action executed when a transition fires.
Subclass :class:`EventHandler` or use a plain callable::

    class Unlock(EventHandler):
        def handle(self, event: Event) -> None:
            print("Unlocking turnstile")

    # Or a lambda:
    handler = LambdaEventHandler(lambda e: print("unlocked"))
"""
from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from typing import Any, Optional


# ═══════════════════════════════════════════════════════════════════════════ #
#  Exceptions                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

class NoSuchTransitionException(Exception):
    """
    Raised when :meth:`FiniteStateMachine.fire` is called with an event for
    which no transition is registered from the current state.
    """


class FSMException(Exception):
    """General FSM error (e.g. machine not initialised, duplicate transitions)."""


# ═══════════════════════════════════════════════════════════════════════════ #
#  State                                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

class State:
    """
    A named state of the finite state machine.

    States are compared by name (case-sensitive).

    Example::

        locked   = State("locked")
        unlocked = State("unlocked")
    """

    def __init__(self, name: str) -> None:
        if not name or not name.strip():
            raise ValueError("State name must not be empty.")
        self._name = name.strip()

    @property
    def name(self) -> str:
        return self._name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, State):
            return self._name == other._name
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._name)

    def __repr__(self) -> str:
        return f"State({self._name!r})"

    def __str__(self) -> str:
        return self._name


# ═══════════════════════════════════════════════════════════════════════════ #
#  Event                                                                      #
# ═══════════════════════════════════════════════════════════════════════════ #

class Event:
    """
    Base class for all FSM events.

    Subclass to create typed events used as transition triggers::

        class CoinEvent(Event): pass
        class PushEvent(Event): pass

    The FSM matches events by their **class** (``type(event)``), so each
    subclass acts as a distinct event type.
    """

    def __init__(self) -> None:
        self.timestamp: datetime.datetime = datetime.datetime.now(datetime.timezone.utc)

    @classmethod
    def event_type(cls) -> type:
        """Return the event's type (its own class)."""
        return cls

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# Alias to match Java's AbstractEvent name
AbstractEvent = Event


# ═══════════════════════════════════════════════════════════════════════════ #
#  EventHandler                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #

class EventHandler(ABC):
    """
    Abstract action executed when a transition fires.

    Implement :meth:`handle` to define the action::

        class Unlock(EventHandler):
            def handle(self, event: Event) -> None:
                print("Turnstile unlocked!")
    """

    @abstractmethod
    def handle(self, event: Event) -> None:
        """Execute the action associated with this handler."""


class LambdaEventHandler(EventHandler):
    """
    Wraps a plain callable as an :class:`EventHandler`.

    Example::

        handler = LambdaEventHandler(lambda e: print(f"Event: {e}"))
    """

    def __init__(self, fn: Any, name: str = "") -> None:
        self._fn = fn
        self._name = name or getattr(fn, "__name__", "lambda-handler")

    def handle(self, event: Event) -> None:
        self._fn(event)

    def __repr__(self) -> str:
        return f"LambdaEventHandler({self._name!r})"
