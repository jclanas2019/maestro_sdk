"""
maestro.rules — event-driven rules engine.

    State(S) × Event(E) → Actions(A), State(S')  (via easy-rules)

Quick start::

    from maestro.rules import Facts, Rules, RuleBuilder, DefaultRulesEngine, rule, condition, action

    @rule(name="weather", priority=1)
    class WeatherRule:
        @condition
        def it_rains(self, facts): return facts.get("rain", False)
        @action
        def umbrella(self, facts): print("Take an umbrella!")

    engine = DefaultRulesEngine()
    engine.fire(Rules(WeatherRule()), Facts(rain=True))
"""
from __future__ import annotations

import inspect
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger(__name__)


class MaestroSecurityWarning(UserWarning):
    """Emitted when a Maestro component is configured in an unsafe/insecure mode."""


# ── Sandboxed builtins for ExpressionRule ────────────────────────────────── #
# eval/exec with full __builtins__ allows __import__, open, and arbitrary code.
# SAFE_BUILTINS restricts expressions to a safe computational subset.
_SAFE_BUILTINS: dict = {
    "True": True, "False": False, "None": None,
    # Type constructors
    "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "tuple": tuple, "dict": dict, "set": set,
    "frozenset": frozenset, "bytes": bytes,
    # Math / comparison
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "pow": pow, "divmod": divmod,
    # Iterables
    "len": len, "any": any, "all": all,
    "sorted": sorted, "reversed": reversed,
    "enumerate": enumerate, "zip": zip, "range": range,
    "map": map, "filter": filter,
    # Type-checking
    "isinstance": isinstance, "issubclass": issubclass,
    "hasattr": hasattr, "getattr": getattr, "callable": callable,
    "type": type, "id": id,
    # String / repr
    "repr": repr, "str": str, "format": format,
    "chr": chr, "ord": ord, "hex": hex, "oct": oct, "bin": bin,
}  # print() intentionally excluded — side effects in sandbox

# ── Constants ────────────────────────────────────────────────────────────── #
DEFAULT_PRIORITY    = 0
DEFAULT_DESCRIPTION = "no description"
MAX_PRIORITY        = 2 ** 31 - 1


# ════════════════════════════════════════════════════════════════════════════
#  Facts
# ════════════════════════════════════════════════════════════════════════════

class Facts:
    """Typed key/value store passed to every rule condition and action."""

    def __init__(self, **kwargs: Any) -> None:
        self._data: dict[str, Any] = dict(kwargs)

    def put(self, name: str, value: Any) -> "Facts":
        self._data[name] = value; return self

    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)

    def remove(self, name: str) -> None: self._data.pop(name, None)
    def contains(self, name: str) -> bool: return name in self._data
    def as_map(self) -> dict[str, Any]: return dict(self._data)

    def __getitem__(self, k: str) -> Any: return self._data[k]
    def __setitem__(self, k: str, v: Any) -> None: self._data[k] = v
    def __contains__(self, k: str) -> bool: return k in self._data
    def __iter__(self) -> Iterator[str]: return iter(self._data)
    def __len__(self) -> int: return len(self._data)
    def __repr__(self) -> str: return f"Facts({self._data!r})"


# ════════════════════════════════════════════════════════════════════════════
#  Decorators
# ════════════════════════════════════════════════════════════════════════════

def condition(fn: Callable) -> Callable:
    """Mark a method as the rule condition."""
    fn._is_condition = True; return fn


def action(order_or_fn=0):
    """Mark a method as a rule action. Usage: ``@action`` or ``@action(order=N)``."""
    if callable(order_or_fn):
        order_or_fn._is_action = True; order_or_fn._action_order = 0
        return order_or_fn
    def decorator(fn: Callable) -> Callable:
        fn._is_action = True; fn._action_order = order_or_fn; return fn
    return decorator


def rule(name: str = "", description: str = DEFAULT_DESCRIPTION, priority: int = DEFAULT_PRIORITY):
    """Class decorator that registers a class as a Maestro rule."""
    def decorator(cls):
        cls._rule_name        = name or cls.__name__
        cls._rule_description = description
        cls._rule_priority    = priority
        cls._is_maestro_rule  = True
        return cls
    return decorator


# ════════════════════════════════════════════════════════════════════════════
#  Rule base
# ════════════════════════════════════════════════════════════════════════════

class Rule(ABC):
    @property
    def name(self) -> str: raise NotImplementedError
    @property
    def description(self) -> str: return DEFAULT_DESCRIPTION
    @property
    def priority(self) -> int: return DEFAULT_PRIORITY
    @abstractmethod
    def evaluate(self, facts: Facts) -> bool: ...
    @abstractmethod
    def execute(self, facts: Facts) -> None: ...
    def __lt__(self, other: "Rule") -> bool: return self.priority < other.priority
    def __eq__(self, o: object) -> bool:
        return self.name == o.name if isinstance(o, Rule) else NotImplemented
    def __hash__(self) -> int: return hash(self.name)
    def __repr__(self) -> str: return f"Rule({self.name!r}, priority={self.priority})"


class BasicRule(Rule):
    """Rule backed by plain callables. Created by ``RuleBuilder``."""
    def __init__(self, name: str, description: str = DEFAULT_DESCRIPTION,
                 priority: int = DEFAULT_PRIORITY,
                 condition_fn: Optional[Callable] = None,
                 action_fns: Optional[list] = None) -> None:
        self._name        = name
        self._description = description
        self._priority    = priority
        self._cond        = condition_fn or (lambda f: True)
        self._actions     = action_fns or []

    @property
    def name(self) -> str: return self._name
    @property
    def description(self) -> str: return self._description
    @property
    def priority(self) -> int: return self._priority

    def evaluate(self, facts: Facts) -> bool:
        try: return bool(self._cond(facts))
        except Exception as e: logger.exception("Condition error in %r: %s", self._name, e); return False

    def execute(self, facts: Facts) -> None:
        for fn in self._actions: fn(facts)


class AnnotatedRule(Rule):
    """Wraps a ``@rule``-decorated class instance."""
    def __init__(self, instance: Any) -> None:
        cls = type(instance)
        if not getattr(cls, "_is_maestro_rule", False):
            raise TypeError(f"{cls.__name__} must be decorated with @rule")
        self._instance    = instance
        self._name        = cls._rule_name
        self._description = cls._rule_description
        self._priority    = cls._rule_priority

        conditions = [m for _, m in inspect.getmembers(instance, predicate=inspect.ismethod)
                      if getattr(m, "_is_condition", False)]
        if len(conditions) != 1:
            raise TypeError(f"{cls.__name__}: exactly one @condition required, found {len(conditions)}")
        self._cond    = conditions[0]
        self._actions = sorted(
            [m for _, m in inspect.getmembers(instance, predicate=inspect.ismethod)
             if getattr(m, "_is_action", False)],
            key=lambda m: getattr(m, "_action_order", 0)
        )

    @property
    def name(self) -> str: return self._name
    @property
    def description(self) -> str: return self._description
    @property
    def priority(self) -> int: return self._priority

    def evaluate(self, facts: Facts) -> bool:
        sig = inspect.signature(self._cond)
        return bool(self._cond(facts) if sig.parameters else self._cond())

    def execute(self, facts: Facts) -> None:
        for m in self._actions:
            sig = inspect.signature(m)
            m(facts) if sig.parameters else m()


def wrap_rule(obj: Any) -> Rule:
    if isinstance(obj, Rule): return obj
    if getattr(type(obj), "_is_maestro_rule", False): return AnnotatedRule(obj)
    raise TypeError(f"Cannot convert {type(obj).__name__!r} to a Rule. Use @rule or subclass Rule.")


# ════════════════════════════════════════════════════════════════════════════
#  Rules collection
# ════════════════════════════════════════════════════════════════════════════

class Rules:
    """Sorted set of rules (ascending priority = fires first)."""
    def __init__(self, *rule_objs: Any) -> None:
        self._rules: list[Rule] = []
        if rule_objs: self.register(*rule_objs)

    def register(self, *rule_objs: Any) -> "Rules":
        existing = {r.name for r in self._rules}
        for obj in rule_objs:
            r = wrap_rule(obj)
            if r.name not in existing:
                self._rules.append(r); existing.add(r.name)
        self._rules.sort(key=lambda r: r.priority)
        return self

    def unregister(self, rule_or_name: Any) -> "Rules":
        name = rule_or_name if isinstance(rule_or_name, str) else wrap_rule(rule_or_name).name
        self._rules = [r for r in self._rules if r.name != name]
        return self

    def clear(self) -> "Rules": self._rules.clear(); return self
    def is_empty(self) -> bool: return not self._rules
    def __iter__(self) -> Iterator[Rule]: return iter(list(self._rules))
    def __len__(self) -> int: return len(self._rules)
    def __contains__(self, x: Any) -> bool:
        n = x if isinstance(x, str) else (x.name if isinstance(x, Rule) else None)
        return any(r.name == n for r in self._rules) if n else False
    def __repr__(self) -> str: return f"Rules({self._rules!r})"


# ════════════════════════════════════════════════════════════════════════════
#  RuleBuilder
# ════════════════════════════════════════════════════════════════════════════

class RuleBuilder:
    """Fluent builder for ``BasicRule``."""
    def __init__(self) -> None:
        self._name        = "rule"
        self._description = DEFAULT_DESCRIPTION
        self._priority    = DEFAULT_PRIORITY
        self._cond: Callable  = lambda f: True
        self._actions: list   = []

    def name(self, n: str)          -> "RuleBuilder": self._name = n;        return self
    def description(self, d: str)   -> "RuleBuilder": self._description = d; return self
    def priority(self, p: int)      -> "RuleBuilder": self._priority = p;    return self
    def when(self, cond: Callable)  -> "RuleBuilder": self._cond = cond;     return self
    def then(self, act: Callable)   -> "RuleBuilder": self._actions.append(act); return self

    def build(self) -> BasicRule:
        if not self._name: raise ValueError("Rule name must not be empty.")
        return BasicRule(self._name, self._description, self._priority,
                         self._cond, list(self._actions))


# ════════════════════════════════════════════════════════════════════════════
#  Listeners
# ════════════════════════════════════════════════════════════════════════════

class RuleListener:
    def before_evaluate(self, rule: Rule, facts: Facts) -> bool: return True
    def on_evaluate_success(self, rule: Rule, facts: Facts, result: bool) -> None: pass
    def on_evaluate_error(self, rule: Rule, facts: Facts, exc: Exception) -> None: pass
    def before_execute(self, rule: Rule, facts: Facts) -> None: pass
    def on_success(self, rule: Rule, facts: Facts) -> None: pass
    def on_failure(self, rule: Rule, facts: Facts, exc: Exception) -> None: pass


class RulesEngineListener:
    def before_evaluate(self, rules: Rules, facts: Facts) -> None: pass
    def after_execute(self, rules: Rules, facts: Facts) -> None: pass


# ════════════════════════════════════════════════════════════════════════════
#  Engine parameters
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class RulesEngineParameters:
    skip_on_first_applied_rule:      bool = False
    skip_on_first_failed_rule:       bool = False
    skip_on_first_non_triggered_rule: bool = False
    rule_priority_threshold:         int  = MAX_PRIORITY
    max_iterations:                  int  = 1_000


# ════════════════════════════════════════════════════════════════════════════
#  Engines
# ════════════════════════════════════════════════════════════════════════════

class _BaseEngine:
    def __init__(self, parameters=None, rule_listeners=None, engine_listeners=None):
        self._params    = parameters or RulesEngineParameters()
        self._rlisteners: list[RuleListener]        = rule_listeners   or []
        self._elisteners: list[RulesEngineListener] = engine_listeners or []

    def register_rule_listener(self, l: RuleListener) -> None:       self._rlisteners.append(l)
    def register_engine_listener(self, l: RulesEngineListener) -> None: self._elisteners.append(l)

    def _fire_loop(self, rules: Rules, facts: Facts) -> dict[Rule, bool]:
        result: dict[Rule, bool] = {}
        for rule in rules:
            if rule.priority > self._params.rule_priority_threshold: break
            if not all(l.before_evaluate(rule, facts) for l in self._rlisteners):
                result[rule] = False; continue
            try:
                triggered = rule.evaluate(facts)
                for l in self._rlisteners: l.on_evaluate_success(rule, facts, triggered)
            except Exception as e:
                for l in self._rlisteners: l.on_evaluate_error(rule, facts, e)
                result[rule] = False; continue
            if triggered:
                for l in self._rlisteners: l.before_execute(rule, facts)
                try:
                    rule.execute(facts)
                    for l in self._rlisteners: l.on_success(rule, facts)
                    result[rule] = True
                except Exception as e:
                    for l in self._rlisteners: l.on_failure(rule, facts, e)
                    result[rule] = False
                if self._params.skip_on_first_applied_rule: break
            else:
                result[rule] = False
                if self._params.skip_on_first_failed_rule: break
                if self._params.skip_on_first_non_triggered_rule: break
        return result


class DefaultRulesEngine(_BaseEngine):
    """Iterates rules once; fires every matching rule."""
    def fire(self, rules: Rules, facts: Facts) -> dict[Rule, bool]:
        if rules.is_empty(): return {}
        for l in self._elisteners: l.before_evaluate(rules, facts)
        result = self._fire_loop(rules, facts)
        for l in self._elisteners: l.after_execute(rules, facts)
        return result

    def check(self, rules: Rules, facts: Facts) -> dict[Rule, bool]:
        result: dict[Rule, bool] = {}
        for r in rules:
            if r.priority > self._params.rule_priority_threshold: break
            try: result[r] = r.evaluate(facts)
            except Exception: result[r] = False
        return result


class InferenceRulesEngine(_BaseEngine):
    """Forward-chaining: re-evaluates until no rule fires or max_iterations reached."""
    def fire(self, rules: Rules, facts: Facts) -> None:
        if rules.is_empty(): return
        for l in self._elisteners: l.before_evaluate(rules, facts)
        iterations = 0
        while iterations < self._params.max_iterations:
            result = self._fire_loop(rules, facts)
            iterations += 1
            if not any(result.values()): break
        for l in self._elisteners: l.after_execute(rules, facts)


class FirstApplicableRulesEngine(DefaultRulesEngine):
    """Fires only the highest-priority matching rule."""
    def __init__(self, parameters=None, rule_listeners=None, engine_listeners=None):
        params = parameters or RulesEngineParameters()
        params.skip_on_first_applied_rule = True
        super().__init__(params, rule_listeners, engine_listeners)


# ════════════════════════════════════════════════════════════════════════════
#  Composite rules
# ════════════════════════════════════════════════════════════════════════════

class CompositeRule(Rule):
    def __init__(self, name: str, description: str = "composite", priority: int = 0):
        self._name, self._description, self._priority = name, description, priority
        self._rules = Rules()
    @property
    def name(self) -> str: return self._name
    @property
    def description(self) -> str: return self._description
    @property
    def priority(self) -> int: return self._priority
    def add_rule(self, r: Rule) -> "CompositeRule":
        self._rules.register(r); return self


class UnitRuleGroup(CompositeRule):
    """All inner conditions must be True; all actions execute or none."""
    def evaluate(self, facts: Facts) -> bool: return all(r.evaluate(facts) for r in self._rules)
    def execute(self, facts: Facts) -> None:
        for r in self._rules: r.execute(facts)


class ConditionalRuleGroup(CompositeRule):
    """Lowest-priority rule acts as guard; rest fire only if guard fires."""
    def _sorted(self): return sorted(self._rules, key=lambda r: r.priority)
    def evaluate(self, facts: Facts) -> bool:
        return self._sorted()[0].evaluate(facts) if not self._rules.is_empty() else False
    def execute(self, facts: Facts) -> None:
        rules = self._sorted()
        rules[0].execute(facts)
        for r in rules[1:]:
            if r.evaluate(facts): r.execute(facts)


class ActivationRuleGroup(CompositeRule):
    """Fires only the first matching inner rule (mutually exclusive)."""
    def _sorted(self): return sorted(self._rules, key=lambda r: r.priority)
    def evaluate(self, facts: Facts) -> bool: return any(r.evaluate(facts) for r in self._sorted())
    def execute(self, facts: Facts) -> None:
        for r in self._sorted():
            if r.evaluate(facts): r.execute(facts); break


# ════════════════════════════════════════════════════════════════════════════
#  Expression rules
# ════════════════════════════════════════════════════════════════════════════

import textwrap as _tw


def _make_cond(expr: str, allow_unsafe: bool = False):
    compiled = compile(expr, "<condition>", "eval")
    if allow_unsafe:
        _globals: dict = {"__builtins__": __builtins__}
    else:
        # Empty __builtins__ blocks __import__, open, exec, eval, compile etc.
        # Allowed names are merged directly into the globals namespace.
        _globals = {"__builtins__": {}, **_SAFE_BUILTINS}
    def _cond(facts: Facts) -> bool:
        ns = {**_globals, **facts.as_map()}
        return bool(eval(compiled, ns))
    return _cond


def _make_action(stmt: str, allow_unsafe: bool = False):
    compiled = compile(_tw.dedent(stmt), "<action>", "exec")
    if allow_unsafe:
        _base: dict = {"__builtins__": __builtins__}
    else:
        _base = {"__builtins__": {}, **_SAFE_BUILTINS}
    def _act(facts: Facts) -> None:
        ns = {**_base, **facts.as_map()}
        try:
            exec(compiled, ns)
        except NameError as exc:
            if not allow_unsafe:
                # Blocked by sandbox (e.g. __import__ not defined) — log and skip.
                # Non-sandbox NameError in allow_unsafe=True mode should propagate.
                logger.warning("ExpressionRule sandbox blocked action: %s", exc)
                return
            raise  # allow_unsafe=True: let real NameError propagate
        # Sync back any new/modified fact keys (skip built-ins and dunders)
        for k, v in ns.items():
            if k not in _SAFE_BUILTINS and not k.startswith("__"):
                facts.put(k, v)
    return _act


class ExpressionRule(BasicRule):
    """
    Rule whose condition and actions are Python expression strings.

    Security
    --------
    By default (``allow_unsafe=False``) expressions are evaluated in a
    restricted sandbox that blocks ``__import__``, ``open``, ``exec``,
    and other dangerous builtins.  Only the safe computational subset
    defined in ``_SAFE_BUILTINS`` is available.

    Set ``allow_unsafe=True`` **only** when the expression source is fully
    trusted (e.g. written by your own team, not user-supplied).
    A :class:`SecurityWarning` is emitted every time an unsafe rule is built.

    Example::

        # Safe (default) — blocks __import__ and other dangerous calls
        rule = ExpressionRule("ok", "total > 100", ["discount = 0.10"])

        # Unsafe — only for fully-trusted, developer-authored expressions
        rule = ExpressionRule("admin", "os.path.exists(path)", [],
                              allow_unsafe=True)  # emits SecurityWarning
    """
    def __init__(self, name: str, condition: str, actions: list[str],
                 description: str = DEFAULT_DESCRIPTION, priority: int = DEFAULT_PRIORITY,
                 allow_unsafe: bool = False):
        if allow_unsafe:
            import warnings
            warnings.warn(
                f"ExpressionRule {name!r}: allow_unsafe=True grants full Python "
                "builtins inside eval/exec.  Only use with developer-authored, "
                "trusted expressions — never with user-supplied input.",
                MaestroSecurityWarning,
                stacklevel=2,
            )
        super().__init__(name, description, priority,
                         _make_cond(condition, allow_unsafe),
                         [_make_action(a, allow_unsafe) for a in actions])
        self._cond_expr    = condition
        self._action_exprs = actions
        self._allow_unsafe = allow_unsafe


class YamlRuleFactory:
    """Load ``ExpressionRule`` objects from YAML files or strings. Requires PyYAML."""


    def __init__(self, allow_unsafe: bool = False):
        try: import yaml  # noqa
        except ImportError as e: raise ImportError("pip install maestro-sdk[yaml]") from e
        self._allow_unsafe = allow_unsafe
        if allow_unsafe:
            import warnings
            warnings.warn(
                "YamlRuleFactory(allow_unsafe=True): YAML-defined rules will run "
                "with full Python builtins.  Only use with trusted YAML sources.",
                MaestroSecurityWarning,
                stacklevel=2,
            )

    def _build(self, d: dict) -> ExpressionRule:
        actions = d.get("actions") or []
        if isinstance(actions, str): actions = [actions]
        return ExpressionRule(d.get("name","rule"), d.get("condition","True"), actions,
                              d.get("description", DEFAULT_DESCRIPTION),
                              int(d.get("priority", DEFAULT_PRIORITY)),
                              allow_unsafe=self._allow_unsafe)

    def _load(self, path):
        import yaml
        from pathlib import Path
        return yaml.safe_load(Path(path).read_text("utf-8"))

    def from_file(self, path) -> "ExpressionRule":
        d = self._load(path)
        return self._build(d[0] if isinstance(d, list) else d)

    def from_files(self, path) -> list:
        d = self._load(path)
        return [self._build(x) for x in (d if isinstance(d, list) else [d])]

    def from_string(self, yaml_str: str) -> "ExpressionRule":
        import yaml; d = yaml.safe_load(yaml_str)
        return self._build(d[0] if isinstance(d, list) else d)

    def from_strings(self, yaml_str: str) -> list:
        import yaml; d = yaml.safe_load(yaml_str)
        return [self._build(x) for x in (d if isinstance(d, list) else [d])]


__all__ = [
    "Facts", "Rule", "BasicRule", "AnnotatedRule", "wrap_rule",
    "rule", "condition", "action", "Rules", "RuleBuilder",
    "RuleListener", "RulesEngineListener", "RulesEngineParameters",
    "DefaultRulesEngine", "InferenceRulesEngine", "FirstApplicableRulesEngine",
    "CompositeRule", "UnitRuleGroup", "ConditionalRuleGroup", "ActivationRuleGroup",
    "ExpressionRule", "YamlRuleFactory",
]
