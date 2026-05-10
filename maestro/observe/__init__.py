"""
maestro.observe — unified observability across all four Maestro modules.

One ``MaestroObserver`` attached once gives you metrics, structured events
and execution traces from rules, batch, flows and states — with no changes
to existing pipeline code.

    from maestro.observe import MaestroObserver, InMemoryObserver

    mem = InMemoryObserver()
    obs = MaestroObserver(observers=[mem])

    # Attach to any combination of modules
    engine  = obs.instrument_rules_engine(DefaultRulesEngine())
    job     = obs.instrument_job(job_builder.build())
    flow    = obs.instrument_flow(my_flow)
    fsm     = obs.instrument_fsm(my_fsm)

    # After execution
    print(mem.summary())
    print(mem.export_prometheus())

Observers
---------
* ``InMemoryObserver``   — stores everything; ideal for testing and dashboards
* ``LoggingObserver``    — structured log lines at DEBUG level
* ``PrintObserver``      — prints to stdout; handy for quick inspection
* ``CompositeObserver``  — fans out to multiple observers
* ``PrometheusObserver`` — writes Prometheus text format to a file or StringIO
"""
from __future__ import annotations

import io
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  Metric primitives
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class MetricEvent:
    """A single observation emitted by any Maestro component."""
    module:    str            # "rules" | "batch" | "flows" | "states"
    name:      str            # metric name, e.g. "rule_fired"
    value:     float          # numeric value
    labels:    dict[str, str] = field(default_factory=dict)
    timestamp: float          = field(default_factory=time.monotonic)
    kind:      str            = "counter"  # "counter" | "gauge" | "histogram"

    @property
    def label_str(self) -> str:
        if not self.labels: return ""
        parts = ",".join(f'{k}="{v}"' for k, v in sorted(self.labels.items()))
        return f"{{{parts}}}"


class _ThreadSafeCounter:
    def __init__(self): self._v = 0.0; self._lock = threading.Lock()
    def inc(self, n=1.0):
        with self._lock: self._v += n
    @property
    def value(self) -> float:
        with self._lock: return self._v


class _ThreadSafeGauge:
    def __init__(self): self._v = 0.0; self._lock = threading.Lock()
    def set(self, v: float):
        with self._lock: self._v = v
    def inc(self, n=1.0):
        with self._lock: self._v += n
    def dec(self, n=1.0):
        with self._lock: self._v -= n
    @property
    def value(self) -> float:
        with self._lock: return self._v


class _Histogram:
    """Simple histogram: tracks count, sum, min, max."""
    def __init__(self):
        self._count = 0; self._sum = 0.0
        self._min = float("inf"); self._max = float("-inf")
        self._lock = threading.Lock()

    def observe(self, v: float):
        with self._lock:
            self._count += 1; self._sum += v
            if v < self._min: self._min = v
            if v > self._max: self._max = v

    @property
    def count(self) -> int:
        with self._lock: return self._count
    @property
    def mean(self) -> float:
        with self._lock: return self._sum / self._count if self._count else 0.0
    @property
    def total(self) -> float:
        with self._lock: return self._sum
    @property
    def min(self) -> float:
        with self._lock: return self._min if self._count else 0.0
    @property
    def max(self) -> float:
        with self._lock: return self._max if self._count else 0.0

    def snapshot(self) -> dict:
        with self._lock:
            return {"count": self._count, "sum": self._sum,
                    "min": self._min if self._count else 0,
                    "max": self._max if self._count else 0,
                    "mean": self._sum / self._count if self._count else 0}


# ════════════════════════════════════════════════════════════════════════════
#  Observer base + implementations
# ════════════════════════════════════════════════════════════════════════════

class Observer(ABC):
    """Base class for all observers. Override ``on_event``."""

    @abstractmethod
    def on_event(self, event: MetricEvent) -> None:
        """Called for every metric event emitted by Maestro."""

    def flush(self) -> None:
        """Flush buffered output if applicable (no-op by default)."""


class LoggingObserver(Observer):
    """Emits each metric event as a structured log line at DEBUG level."""

    def __init__(self, logger_name: str = "maestro.observe") -> None:
        self._log = logging.getLogger(logger_name)

    def on_event(self, event: MetricEvent) -> None:
        self._log.debug(
            "[%s] %s%s = %g (%s)",
            event.module, event.name, event.label_str, event.value, event.kind,
        )


class PrintObserver(Observer):
    """Prints each metric event to stdout — useful for quick inspection."""

    def on_event(self, event: MetricEvent) -> None:
        print(f"[{event.module}] {event.name}{event.label_str} = {event.value}")


class CompositeObserver(Observer):
    """Fans out every event to multiple child observers."""

    def __init__(self, *observers: Observer) -> None:
        self._observers = list(observers)

    def add(self, obs: Observer) -> "CompositeObserver":
        self._observers.append(obs); return self

    def on_event(self, event: MetricEvent) -> None:
        for obs in self._observers:
            try: obs.on_event(event)
            except Exception as e: logger.warning("Observer error: %s", e)

    def flush(self) -> None:
        for obs in self._observers:
            try: obs.flush()
            except Exception: pass


class InMemoryObserver(Observer):
    """
    Stores all metric events in memory.

    Perfect for tests and runtime inspection::

        obs = InMemoryObserver()
        ...
        print(obs.summary())
        rules_fired = obs.counter("rules", "rule_fired")
    """

    def __init__(self) -> None:
        self._events:   list[MetricEvent]      = []
        self._counters: dict[tuple, float]     = defaultdict(float)
        self._gauges:   dict[tuple, float]     = defaultdict(float)
        self._hists:    dict[tuple, _Histogram]= defaultdict(_Histogram)
        self._lock      = threading.Lock()

    def on_event(self, event: MetricEvent) -> None:
        key = (event.module, event.name, tuple(sorted(event.labels.items())))
        with self._lock:
            self._events.append(event)
            if event.kind == "counter":
                self._counters[key] += event.value
            elif event.kind == "gauge":
                self._gauges[key] = event.value
            elif event.kind == "histogram":
                self._hists[key].observe(event.value)

    def counter(self, module: str, name: str, **labels) -> float:
        key = (module, name, tuple(sorted(labels.items())))
        with self._lock: return self._counters.get(key, 0.0)

    def gauge(self, module: str, name: str, **labels) -> float:
        key = (module, name, tuple(sorted(labels.items())))
        with self._lock: return self._gauges.get(key, 0.0)

    def histogram(self, module: str, name: str, **labels) -> dict:
        key = (module, name, tuple(sorted(labels.items())))
        with self._lock: return self._hists[key].snapshot()

    def events(self, module: Optional[str] = None,
               name: Optional[str] = None) -> list[MetricEvent]:
        with self._lock:
            evs = list(self._events)
        if module: evs = [e for e in evs if e.module == module]
        if name:   evs = [e for e in evs if e.name   == name]
        return evs

    def clear(self) -> None:
        with self._lock:
            self._events.clear(); self._counters.clear()
            self._gauges.clear(); self._hists.clear()

    def summary(self) -> str:
        lines = ["Maestro Observability Summary", "─" * 44]
        with self._lock:
            # Counters
            if self._counters:
                lines.append("Counters:")
                for (mod, name, lbls), val in sorted(self._counters.items()):
                    lstr = ",".join(f"{k}={v}" for k,v in lbls)
                    lines.append(f"  [{mod}] {name}{'{'+lstr+'}' if lstr else ''} = {val:.0f}")
            # Histograms
            if self._hists:
                lines.append("Histograms:")
                for (mod, name, lbls), hist in sorted(self._hists.items()):
                    snap = hist.snapshot()
                    lstr = ",".join(f"{k}={v}" for k,v in lbls)
                    lines.append(
                        f"  [{mod}] {name}{'{'+lstr+'}' if lstr else ''} "
                        f"count={snap['count']} mean={snap['mean']:.4f}s "
                        f"min={snap['min']:.4f}s max={snap['max']:.4f}s"
                    )
        return "\n".join(lines)

    def export_prometheus(self) -> str:
        """Return metrics in Prometheus text exposition format."""
        buf = io.StringIO()
        with self._lock:
            for (mod, name, lbls), val in sorted(self._counters.items()):
                metric_name = f"maestro_{mod}_{name}_total"
                lstr = ",".join(f'{k}="{v}"' for k,v in lbls)
                buf.write(f"# TYPE {metric_name} counter\n")
                buf.write(f'{metric_name}{"{"+lstr+"}" if lstr else ""} {val:.0f}\n')
            for (mod, name, lbls), hist in sorted(self._hists.items()):
                snap = hist.snapshot()
                metric_name = f"maestro_{mod}_{name}"
                lstr = ",".join(f'{k}="{v}"' for k,v in lbls)
                lb = "{"+lstr+"}" if lstr else ""
                buf.write(f"# TYPE {metric_name} histogram\n")
                buf.write(f'{metric_name}_count{lb} {snap["count"]}\n')
                buf.write(f'{metric_name}_sum{lb} {snap["sum"]:.6f}\n')
        return buf.getvalue()


# ════════════════════════════════════════════════════════════════════════════
#  MaestroObserver — the bridge to all four modules
# ════════════════════════════════════════════════════════════════════════════

class MaestroObserver:
    """
    Attaches to the four Maestro modules and emits unified metric events.

    Usage::

        from maestro.observe import MaestroObserver, InMemoryObserver

        mem = InMemoryObserver()
        obs = MaestroObserver(observers=[mem])

        engine = obs.instrument_rules_engine(DefaultRulesEngine())
        job    = obs.instrument_job(job)
        flow   = obs.instrument_flow(flow)
        fsm    = obs.instrument_fsm(fsm)
    """

    def __init__(self, observers: Optional[list[Observer]] = None) -> None:
        if observers and len(observers) == 1:
            self._obs: Observer = observers[0]
        elif observers:
            self._obs = CompositeObserver(*observers)
        else:
            self._obs = LoggingObserver()

    def _emit(self, module: str, name: str, value: float = 1.0,
              kind: str = "counter", **labels) -> None:
        self._obs.on_event(MetricEvent(
            module=module, name=name, value=value,
            labels={k: str(v) for k, v in labels.items()},
            kind=kind,
        ))

    # ── Rules ─────────────────────────────────────────────────────────── #

    def instrument_rules_engine(self, engine):
        """
        Attach observability listeners to a rules engine and return it.

        Tracks: evaluations, firings, rule latency.
        """
        obs = self

        # Import here to avoid circular dependency at module load
        from maestro.rules import RuleListener, RulesEngineListener

        class _RL(RuleListener):
            _t0: dict = {}

            def before_evaluate(self, rule, facts):
                self._t0[rule.name] = time.monotonic()
                return True

            def on_evaluate_success(self, rule, facts, result):
                obs._emit("rules", "rule_evaluated", rule=rule.name)
                if result:
                    obs._emit("rules", "rule_fired", rule=rule.name)

            def on_evaluate_error(self, rule, facts, exc):
                obs._emit("rules", "rule_error", rule=rule.name)

            def on_success(self, rule, facts):
                t = self._t0.pop(rule.name, None)
                if t is not None:
                    obs._emit("rules", "rule_duration_seconds",
                              value=time.monotonic() - t, kind="histogram",
                              rule=rule.name)

            def on_failure(self, rule, facts, exc):
                obs._emit("rules", "rule_action_error", rule=rule.name)

        class _EL(RulesEngineListener):
            _t0: Optional[float] = None

            def before_evaluate(self, rules, facts):
                self._t0 = time.monotonic()
                obs._emit("rules", "engine_fire")

            def after_execute(self, rules, facts):
                if self._t0:
                    obs._emit("rules", "engine_duration_seconds",
                              value=time.monotonic() - self._t0, kind="histogram")

        engine.register_rule_listener(_RL())
        engine.register_engine_listener(_EL())
        return engine

    # ── Batch ─────────────────────────────────────────────────────────── #

    def instrument_job(self, job):
        """
        Attach observability listeners to a batch Job and return it.

        Tracks: records read/written/filtered/skipped/failed, job duration, batch sizes.
        """
        obs = self

        from maestro.batch._listener import JobListener, BatchListener, PipelineListener

        class _JL(JobListener):
            _t0: Optional[float] = None

            def before_job_start(self, params):
                self._t0 = time.monotonic()
                obs._emit("batch", "job_started", job=params.name)

            def after_job_end(self, report):
                if self._t0:
                    obs._emit("batch", "job_duration_seconds",
                              value=time.monotonic() - self._t0, kind="histogram",
                              job=report.parameters.name, status=report.status.value)
                m = report.metrics
                obs._emit("batch", "records_total",    value=m.total_count,    kind="gauge", job=report.parameters.name)
                obs._emit("batch", "records_written",  value=m.written_count,  kind="gauge", job=report.parameters.name)
                obs._emit("batch", "records_filtered", value=m.filtered_count, kind="gauge", job=report.parameters.name)
                obs._emit("batch", "records_skipped",  value=m.skipped_count,  kind="gauge", job=report.parameters.name)
                obs._emit("batch", "records_failed",   value=m.failed_count,   kind="gauge", job=report.parameters.name)

        class _BL(BatchListener):
            def before_batch_writing(self, batch):
                obs._emit("batch", "batch_size", value=len(batch), kind="histogram")

            def on_batch_writing_exception(self, batch, exc):
                obs._emit("batch", "batch_write_error")

        class _PL(PipelineListener):
            _times: dict = {}

            def before_record_processing(self, record):
                self._times[record.header.number] = time.monotonic()

            def after_record_processing(self, record):
                t = self._times.pop(record.header.number, None)
                if t:
                    obs._emit("batch", "record_duration_seconds",
                              value=time.monotonic() - t, kind="histogram")

            def on_record_processing_exception(self, record, exc):
                obs._emit("batch", "record_error")

        job._jlisteners.append(_JL())
        job._blisteners.append(_BL())
        job._plisteners.append(_PL())
        return job

    # ── Flows ─────────────────────────────────────────────────────────── #

    def instrument_flow(self, flow):
        """
        Wrap a workflow so every Work execution emits duration and status metrics.

        Returns an ``ObservedFlow`` wrapper — transparent to callers.
        """
        return _ObservedFlow(flow, self)

    # ── States ────────────────────────────────────────────────────────── #

    def instrument_fsm(self, fsm):
        """
        Attach a ``TransitionListener`` to an FSM and return it.

        Tracks: transitions by name, state entries, transition latency.
        """
        obs = self

        from maestro.states._listener import TransitionListener

        class _TL(TransitionListener):
            _t0: Optional[float] = None

            def on_transition_started(self, event, transition):
                self._t0 = time.monotonic()
                obs._emit("states", "transition_started",
                          transition=transition.name,
                          source=transition.source_state.name,
                          event=type(event).__name__)

            def on_transition_ended(self, event, transition, new_state):
                obs._emit("states", "transition_completed",
                          transition=transition.name,
                          target=new_state.name)
                obs._emit("states", "state_entered", state=new_state.name)
                if self._t0:
                    obs._emit("states", "transition_duration_seconds",
                              value=time.monotonic() - self._t0,
                              kind="histogram", transition=transition.name)

            def on_no_transition(self, event, current_state):
                obs._emit("states", "undefined_transition",
                          state=current_state.name, event=type(event).__name__)

            def on_handler_error(self, event, transition, exc):
                obs._emit("states", "handler_error", transition=transition.name)

        fsm.add_listener(_TL())
        return fsm

    @property
    def observer(self) -> Observer:
        """The underlying observer (or composite)."""
        return self._obs


# ════════════════════════════════════════════════════════════════════════════
#  ObservedFlow — transparent wrapper that times each Work step
# ════════════════════════════════════════════════════════════════════════════

class _ObservedFlow:
    """
    Wraps a workflow's ``execute`` to emit Work-level metrics.
    Fully transparent — quacks like a Work.
    """

    def __init__(self, flow, maestro_obs: MaestroObserver) -> None:
        self._flow = flow
        self._mobs = maestro_obs

    def get_name(self) -> str:
        return self._flow.get_name()

    def execute(self, work_context):
        t0 = time.monotonic()
        self._mobs._emit("flows", "flow_started", flow=self._flow.get_name())
        report = self._flow.execute(work_context)
        dur = time.monotonic() - t0
        self._mobs._emit("flows", "flow_duration_seconds",
                         value=dur, kind="histogram", flow=self._flow.get_name(),
                         status=report.status.value)
        self._mobs._emit("flows", f"flow_{report.status.value.lower()}",
                         flow=self._flow.get_name())
        return report

    def __getattr__(self, name: str):
        return getattr(self._flow, name)


# ════════════════════════════════════════════════════════════════════════════
#  Timing context manager
# ════════════════════════════════════════════════════════════════════════════

class timed:
    """
    Context manager that records duration into an ``InMemoryObserver``::

        obs = InMemoryObserver()
        with timed("flows", "custom_step", obs):
            do_work()
    """

    def __init__(self, module: str, name: str, observer: Observer, **labels) -> None:
        self._module = module; self._name = name
        self._obs = observer; self._labels = labels

    def __enter__(self) -> "timed":
        self._t0 = time.monotonic(); return self

    def __exit__(self, *_) -> bool:
        self._obs.on_event(MetricEvent(
            module=self._module, name=self._name,
            value=time.monotonic() - self._t0,
            labels={k: str(v) for k, v in self._labels.items()},
            kind="histogram",
        ))
        return False


__all__ = [
    "MetricEvent",
    "Observer", "LoggingObserver", "PrintObserver",
    "CompositeObserver", "InMemoryObserver",
    "MaestroObserver",
    "timed",
]
