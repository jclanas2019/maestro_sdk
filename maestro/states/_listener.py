"""
TransitionListener — observe FSM lifecycle events.

Override the methods you need::

    class AuditListener(TransitionListener):
        def on_transition_started(self, event, transition):
            print(f"Firing {transition.name} on {event}")

        def on_transition_ended(self, event, transition, new_state):
            print(f"Now in {new_state}")
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maestro.states._state import Event, State
    from .transition import Transition


class TransitionListener:
    """
    SPI for observing FSM transitions. Override the methods you need.
    All methods default to no-op.
    """

    def on_transition_started(
        self,
        event: "Event",
        transition: "Transition",
    ) -> None:
        """Called just before executing the event handler and changing state."""

    def on_transition_ended(
        self,
        event: "Event",
        transition: "Transition",
        new_state: "State",
    ) -> None:
        """Called after the state has changed."""

    def on_no_transition(
        self,
        event: "Event",
        current_state: "State",
    ) -> None:
        """
        Called when no transition is registered for the current state and
        the incoming event type (and ``ignore_undefined_transitions=True``).
        """

    def on_handler_error(
        self,
        event: "Event",
        transition: "Transition",
        exc: Exception,
    ) -> None:
        """Called when an event handler raises an exception."""
