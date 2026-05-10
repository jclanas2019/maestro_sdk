"""
Transition — the fundamental relation of an FSM.

    State(S) × Event(E) → Actions(A), State(S')

Usage::

    unlock = (
        TransitionBuilder()
        .name("unlock")
        .source_state(locked)
        .event_type(CoinEvent)
        .event_handler(Unlock())
        .target_state(unlocked)
        .build()
    )
"""
from __future__ import annotations

import uuid
from typing import Optional, Type

from maestro.states._state import Event, EventHandler, State


class Transition:
    """
    An immutable transition edge in the automaton.

    Attributes:
        name:          Human-readable label (optional, auto-generated if omitted).
        source_state:  The state this transition departs from.
        event_type:    The *class* of event that triggers this transition.
        event_handler: Optional action executed when the transition fires.
        target_state:  The state reached after firing.
    """

    def __init__(
        self,
        name: str,
        source_state: State,
        event_type: Type[Event],
        target_state: State,
        event_handler: Optional[EventHandler] = None,
    ) -> None:
        self._name          = name
        self._source_state  = source_state
        self._event_type    = event_type
        self._target_state  = target_state
        self._event_handler = event_handler

    # ── properties ──────────────────────────────────────────────────── #

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_state(self) -> State:
        return self._source_state

    @property
    def event_type(self) -> Type[Event]:
        return self._event_type

    @property
    def target_state(self) -> State:
        return self._target_state

    @property
    def event_handler(self) -> Optional[EventHandler]:
        return self._event_handler

    # ── dunder ──────────────────────────────────────────────────────── #

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transition):
            return NotImplemented
        return (
            self._source_state == other._source_state
            and self._event_type is other._event_type
        )

    def __hash__(self) -> int:
        return hash((self._source_state, self._event_type))

    def __repr__(self) -> str:
        return (
            f"Transition({self._name!r}: "
            f"{self._source_state.name!r} --[{self._event_type.__name__}]--> "
            f"{self._target_state.name!r})"
        )


# ═══════════════════════════════════════════════════════════════════════════ #
#  TransitionBuilder                                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

class TransitionBuilder:
    """
    Fluent builder for :class:`Transition`.

    Example::

        t = (
            TransitionBuilder()
            .name("unlock")
            .source_state(locked)
            .event_type(CoinEvent)
            .event_handler(UnlockHandler())
            .target_state(unlocked)
            .build()
        )
    """

    def __init__(self) -> None:
        self._name: str                         = f"t-{uuid.uuid4().hex[:6]}"
        self._source_state: Optional[State]     = None
        self._event_type: Optional[Type[Event]] = None
        self._target_state: Optional[State]     = None
        self._event_handler: Optional[EventHandler] = None

    def name(self, name: str) -> "TransitionBuilder":
        self._name = name
        return self

    def source_state(self, state: State) -> "TransitionBuilder":
        self._source_state = state
        return self

    # Java API compat
    def sourceState(self, state: State) -> "TransitionBuilder":
        return self.source_state(state)

    def event_type(self, event_cls: Type[Event]) -> "TransitionBuilder":
        self._event_type = event_cls
        return self

    # Java API compat
    def eventType(self, event_cls: Type[Event]) -> "TransitionBuilder":
        return self.event_type(event_cls)

    def event_handler(self, handler: EventHandler) -> "TransitionBuilder":
        self._event_handler = handler
        return self

    # Java API compat
    def eventHandler(self, handler: EventHandler) -> "TransitionBuilder":
        return self.event_handler(handler)

    def target_state(self, state: State) -> "TransitionBuilder":
        self._target_state = state
        return self

    # Java API compat
    def targetState(self, state: State) -> "TransitionBuilder":
        return self.target_state(state)

    def build(self) -> Transition:
        if self._source_state is None:
            raise ValueError("Transition requires a source state (.source_state(…)).")
        if self._event_type is None:
            raise ValueError("Transition requires an event type (.event_type(…)).")
        if self._target_state is None:
            raise ValueError("Transition requires a target state (.target_state(…)).")
        return Transition(
            name=self._name,
            source_state=self._source_state,
            event_type=self._event_type,
            target_state=self._target_state,
            event_handler=self._event_handler,
        )
