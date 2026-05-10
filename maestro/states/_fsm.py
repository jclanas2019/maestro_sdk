"""
FiniteStateMachine — the core DFA engine.

Usage::

    fsm = (
        FiniteStateMachineBuilder(states={locked, unlocked}, initial_state=locked)
        .register_transition(unlock)
        .register_transition(push_locked)
        .register_transition(lock)
        .register_transition(coin_unlocked)
        .build()
    )

    fsm.fire(CoinEvent())   # locked  → unlocked
    fsm.fire(PushEvent())   # unlocked → locked
    print(fsm.current_state)
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Set, Type

from maestro.states._listener import TransitionListener
from maestro.states._state import Event, EventHandler, FSMException, LambdaEventHandler, NoSuchTransitionException, State
from maestro.states._transition import Transition, TransitionBuilder

logger = logging.getLogger(__name__)


class FiniteStateMachine:
    """
    A Deterministic Finite Automaton.

    The machine maintains exactly one *current state* at any time.
    Calling :meth:`fire` with an :class:`Event` looks up the registered
    :class:`Transition` for ``(current_state, type(event))``, runs the
    optional :class:`EventHandler`, and changes state.

    Args:
        states:                       The complete set of valid states.
        initial_state:                The state the machine starts in.
        transitions:                  All registered transitions.
        ignore_undefined_transitions: If ``True``, silently stay in the
                                      current state when no transition is
                                      found (default: ``False`` → raises
                                      :exc:`NoSuchTransitionException`).
        listeners:                    Zero or more :class:`TransitionListener`.
    """

    def __init__(
        self,
        states: set[State],
        initial_state: State,
        transitions: list[Transition],
        ignore_undefined_transitions: bool = False,
        listeners: Optional[list[TransitionListener]] = None,
    ) -> None:
        if initial_state not in states:
            raise FSMException(
                f"Initial state {initial_state!r} is not in the states set."
            )
        self._states     = set(states)
        self._state      = initial_state
        self._ignore     = ignore_undefined_transitions
        self._listeners  = listeners or []

        # Build lookup: (State, event_type) → Transition
        self._table: dict[tuple[State, type], Transition] = {}
        for t in transitions:
            key = (t.source_state, t.event_type)
            if key in self._table:
                raise FSMException(
                    f"Duplicate transition for state={t.source_state.name!r} "
                    f"event={t.event_type.__name__!r}. "
                    "Only one transition per (state, event) pair is allowed in a DFA."
                )
            self._table[key] = t

    # ── public API ──────────────────────────────────────────────────── #

    @property
    def current_state(self) -> State:
        """The state the machine is currently in."""
        return self._state

    @property
    def states(self) -> frozenset[State]:
        """All valid states registered with this machine."""
        return frozenset(self._states)

    @property
    def transitions(self) -> list[Transition]:
        """All registered transitions."""
        return list(self._table.values())

    def fire(self, event: Event) -> State:
        """
        Fire *event* against the current state.

        1. Look up the transition for ``(current_state, type(event))``.
        2. Call :meth:`EventHandler.handle` if a handler is registered.
        3. Change the current state to ``transition.target_state``.
        4. Return the new current state.

        Raises:
            NoSuchTransitionException: if no transition matches and
                ``ignore_undefined_transitions`` is ``False``.
        """
        event_cls = type(event)
        key       = (self._state, event_cls)
        transition = self._table.get(key)

        if transition is None:
            msg = (
                f"No transition from state={self._state.name!r} "
                f"on event={event_cls.__name__!r}."
            )
            for listener in self._listeners:
                listener.on_no_transition(event, self._state)
            if self._ignore:
                logger.debug("Ignoring undefined transition: %s", msg)
                return self._state
            raise NoSuchTransitionException(msg)

        logger.debug(
            "Firing transition '%s': %s --[%s]--> %s",
            transition.name,
            transition.source_state.name,
            event_cls.__name__,
            transition.target_state.name,
        )

        for listener in self._listeners:
            listener.on_transition_started(event, transition)

        if transition.event_handler is not None:
            try:
                transition.event_handler.handle(event)
            except Exception as exc:
                logger.error(
                    "EventHandler for transition '%s' raised: %s",
                    transition.name, exc,
                )
                for listener in self._listeners:
                    listener.on_handler_error(event, transition, exc)

        self._state = transition.target_state

        for listener in self._listeners:
            listener.on_transition_ended(event, transition, self._state)

        logger.debug("Current state is now: %s", self._state.name)
        return self._state

    def add_listener(self, listener: TransitionListener) -> None:
        """Register an additional :class:`TransitionListener` at runtime."""
        self._listeners.append(listener)

    # ── export ──────────────────────────────────────────────────────── #

    def to_dot(self, graph_name: str = "FSM") -> str:
        """
        Export the state machine as a **Graphviz DOT** string.

        Example::

            print(fsm.to_dot("turnstile"))
            # paste into https://dreampuf.github.io/GraphvizOnline/
        """
        lines = [f'digraph "{graph_name}" {{', '  rankdir=LR;']
        # Mark initial state with a double circle
        lines.append(f'  "{self._state.name}" [shape=doublecircle];')
        for t in self._table.values():
            label = t.event_type.__name__
            if t.name and not t.name.startswith("t-"):
                label = f"{t.name}\\n({label})"
            lines.append(
                f'  "{t.source_state.name}" -> "{t.target_state.name}" '
                f'[label="{label}"];'
            )
        lines.append("}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"FiniteStateMachine(state={self._state.name!r}, transitions={len(self._table)})"


# ═══════════════════════════════════════════════════════════════════════════ #
#  FiniteStateMachineBuilder                                                   #
# ═══════════════════════════════════════════════════════════════════════════ #

class FiniteStateMachineBuilder:
    """
    Fluent builder for :class:`FiniteStateMachine`.

    Example::

        fsm = (
            FiniteStateMachineBuilder(states={locked, unlocked}, initial_state=locked)
            .register_transition(unlock_t)
            .register_transition(push_locked_t)
            .register_transition(lock_t)
            .register_transition(coin_unlocked_t)
            .build()
        )
    """

    def __init__(
        self,
        states: Set[State],
        initial_state: State,
    ) -> None:
        self._states        = set(states)
        self._initial       = initial_state
        self._transitions:  list[Transition] = []
        self._ignore        = False
        self._listeners:    list[TransitionListener] = []

    # Java-compat alias
    def registerTransition(self, transition: Transition) -> "FiniteStateMachineBuilder":
        return self.register_transition(transition)

    def register_transition(self, transition: Transition) -> "FiniteStateMachineBuilder":
        """Register a transition. Returns self for chaining."""
        self._transitions.append(transition)
        return self

    def ignore_undefined_transitions(self, ignore: bool = True) -> "FiniteStateMachineBuilder":
        """
        When ``True``, firing an event with no registered transition silently
        keeps the machine in its current state (no exception raised).
        """
        self._ignore = ignore
        return self

    def register_listener(self, listener: TransitionListener) -> "FiniteStateMachineBuilder":
        self._listeners.append(listener)
        return self

    def build(self) -> FiniteStateMachine:
        return FiniteStateMachine(
            states=self._states,
            initial_state=self._initial,
            transitions=list(self._transitions),
            ignore_undefined_transitions=self._ignore,
            listeners=list(self._listeners),
        )
