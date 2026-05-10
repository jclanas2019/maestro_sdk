"""
tests/test_easy_rules.py — comprehensive unit tests for the Python easy-rules port.

Run with:  python -m pytest tests/ -v
"""
import sys, os


import pytest
from maestro.rules import (
    Facts, Rules, RuleBuilder, DefaultRulesEngine,
    InferenceRulesEngine, FirstApplicableRulesEngine,
    RulesEngineParameters, RuleListener,
    rule, condition, action,
    BasicRule, AnnotatedRule, wrap_rule,
    UnitRuleGroup, ConditionalRuleGroup, ActivationRuleGroup,
    ExpressionRule,
)


# ─────────────────────────────────────────────────────────────────── #
#  Helpers                                                            #
# ─────────────────────────────────────────────────────────────────── #

def _always_true_rule(name="always", priority=0):
    return (
        RuleBuilder().name(name).priority(priority)
        .when(lambda f: True).then(lambda f: None).build()
    )

def _always_false_rule(name="never", priority=0):
    return (
        RuleBuilder().name(name).priority(priority)
        .when(lambda f: False).then(lambda f: None).build()
    )


# ─────────────────────────────────────────────────────────────────── #
#  Facts                                                              #
# ─────────────────────────────────────────────────────────────────── #

class TestFacts:
    def test_put_get(self):
        f = Facts()
        f.put("x", 10)
        assert f.get("x") == 10

    def test_get_default(self):
        f = Facts()
        assert f.get("missing", 42) == 42

    def test_remove(self):
        f = Facts(a=1)
        f.remove("a")
        assert not f.contains("a")

    def test_contains(self):
        f = Facts(a=1)
        assert f.contains("a")
        assert not f.contains("b")

    def test_as_map(self):
        f = Facts(a=1, b=2)
        m = f.as_map()
        assert m == {"a": 1, "b": 2}

    def test_kwargs_init(self):
        f = Facts(rain=True, temp=22)
        assert f.get("rain") is True
        assert f.get("temp") == 22

    def test_setitem_getitem(self):
        f = Facts()
        f["x"] = 5
        assert f["x"] == 5

    def test_iter(self):
        f = Facts(a=1, b=2)
        assert set(f) == {"a", "b"}

    def test_len(self):
        f = Facts(a=1, b=2)
        assert len(f) == 2


# ─────────────────────────────────────────────────────────────────── #
#  Rules collection                                                   #
# ─────────────────────────────────────────────────────────────────── #

class TestRules:
    def test_register_and_len(self):
        rules = Rules()
        rules.register(_always_true_rule("r1"))
        rules.register(_always_true_rule("r2"))
        assert len(rules) == 2

    def test_duplicate_ignored(self):
        r = _always_true_rule("dup")
        rules = Rules(r, r)
        assert len(rules) == 1

    def test_unregister(self):
        r = _always_true_rule("r")
        rules = Rules(r)
        rules.unregister(r)
        assert "r" not in rules

    def test_unregister_by_name(self):
        r = _always_true_rule("r")
        rules = Rules(r)
        rules.unregister("r")
        assert "r" not in rules

    def test_priority_order(self):
        r_low = _always_true_rule("low", priority=10)
        r_high = _always_true_rule("high", priority=1)
        rules = Rules(r_low, r_high)
        ordered = list(rules)
        assert ordered[0].name == "high"
        assert ordered[1].name == "low"

    def test_is_empty(self):
        assert Rules().is_empty()
        assert not Rules(_always_true_rule()).is_empty()

    def test_clear(self):
        rules = Rules(_always_true_rule("r"))
        rules.clear()
        assert rules.is_empty()


# ─────────────────────────────────────────────────────────────────── #
#  RuleBuilder                                                        #
# ─────────────────────────────────────────────────────────────────── #

class TestRuleBuilder:
    def test_basic_build(self):
        r = RuleBuilder().name("r").when(lambda f: True).then(lambda f: None).build()
        assert r.name == "r"

    def test_condition_evaluated(self):
        r = RuleBuilder().name("r").when(lambda f: f.get("x") > 0).build()
        assert r.evaluate(Facts(x=5)) is True
        assert r.evaluate(Facts(x=-1)) is False

    def test_action_executed(self):
        store = []
        r = (
            RuleBuilder().name("r")
            .when(lambda f: True)
            .then(lambda f: store.append("fired"))
            .build()
        )
        r.execute(Facts())
        assert store == ["fired"]

    def test_multiple_actions(self):
        log = []
        r = (
            RuleBuilder().name("r")
            .when(lambda f: True)
            .then(lambda f: log.append(1))
            .then(lambda f: log.append(2))
            .build()
        )
        r.execute(Facts())
        assert log == [1, 2]

    def test_missing_name_raises(self):
        with pytest.raises(ValueError):
            RuleBuilder().name("").build()


# ─────────────────────────────────────────────────────────────────── #
#  Decorator-style rules                                              #
# ─────────────────────────────────────────────────────────────────── #

class TestAnnotatedRule:
    def test_basic(self):
        @rule(name="weather rule", priority=1)
        class WeatherRule:
            @condition
            def it_rains(self, facts):
                return facts.get("rain", False)

            @action
            def take_umbrella(self, facts):
                facts.put("action", "umbrella")

        r = wrap_rule(WeatherRule())
        assert r.name == "weather rule"
        assert r.priority == 1
        f = Facts(rain=True)
        assert r.evaluate(f) is True
        r.execute(f)
        assert f.get("action") == "umbrella"

    def test_no_condition_raises(self):
        @rule(name="bad")
        class BadRule:
            @action
            def do_something(self, facts):
                pass

        with pytest.raises(TypeError):
            AnnotatedRule(BadRule())

    def test_multiple_conditions_raises(self):
        @rule(name="bad")
        class BadRule2:
            @condition
            def cond1(self, facts):
                return True

            @condition
            def cond2(self, facts):
                return True

            @action
            def act(self, facts):
                pass

        with pytest.raises(TypeError):
            AnnotatedRule(BadRule2())


# ─────────────────────────────────────────────────────────────────── #
#  DefaultRulesEngine                                                 #
# ─────────────────────────────────────────────────────────────────── #

class TestDefaultRulesEngine:
    def test_fires_matching_rule(self):
        log = []
        r = RuleBuilder().name("r").when(lambda f: True).then(lambda f: log.append("x")).build()
        DefaultRulesEngine().fire(Rules(r), Facts())
        assert log == ["x"]

    def test_skips_non_matching_rule(self):
        log = []
        r = RuleBuilder().name("r").when(lambda f: False).then(lambda f: log.append("x")).build()
        DefaultRulesEngine().fire(Rules(r), Facts())
        assert log == []

    def test_fires_all_matching_rules(self):
        log = []
        r1 = RuleBuilder().name("r1").when(lambda f: True).then(lambda f: log.append(1)).build()
        r2 = RuleBuilder().name("r2").when(lambda f: True).then(lambda f: log.append(2)).build()
        DefaultRulesEngine().fire(Rules(r1, r2), Facts())
        assert sorted(log) == [1, 2]

    def test_result_map(self):
        r1 = RuleBuilder().name("yes").when(lambda f: True).then(lambda f: None).build()
        r2 = RuleBuilder().name("no").when(lambda f: False).then(lambda f: None).build()
        result = DefaultRulesEngine().fire(Rules(r1, r2), Facts())
        assert result[r1] is True
        assert result[r2] is False

    def test_check_dry_run(self):
        r = RuleBuilder().name("r").when(lambda f: f.get("x") > 0).build()
        result = DefaultRulesEngine().check(Rules(r), Facts(x=5))
        assert result[r] is True

    def test_priority_threshold(self):
        log = []
        r_below = RuleBuilder().name("below").priority(5).when(lambda f: True).then(lambda f: log.append("below")).build()
        r_above = RuleBuilder().name("above").priority(15).when(lambda f: True).then(lambda f: log.append("above")).build()
        params = RulesEngineParameters(rule_priority_threshold=10)
        DefaultRulesEngine(parameters=params).fire(Rules(r_below, r_above), Facts())
        assert "below" in log
        assert "above" not in log

    def test_skip_on_first_applied(self):
        log = []
        r1 = RuleBuilder().name("r1").priority(1).when(lambda f: True).then(lambda f: log.append(1)).build()
        r2 = RuleBuilder().name("r2").priority(2).when(lambda f: True).then(lambda f: log.append(2)).build()
        params = RulesEngineParameters(skip_on_first_applied_rule=True)
        DefaultRulesEngine(parameters=params).fire(Rules(r1, r2), Facts())
        assert log == [1]

    def test_skip_on_first_failed(self):
        log = []
        r1 = RuleBuilder().name("r1").priority(1).when(lambda f: False).then(lambda f: log.append(1)).build()
        r2 = RuleBuilder().name("r2").priority(2).when(lambda f: True).then(lambda f: log.append(2)).build()
        params = RulesEngineParameters(skip_on_first_failed_rule=True)
        DefaultRulesEngine(parameters=params).fire(Rules(r1, r2), Facts())
        assert log == []


# ─────────────────────────────────────────────────────────────────── #
#  FirstApplicableRulesEngine                                         #
# ─────────────────────────────────────────────────────────────────── #

class TestFirstApplicableRulesEngine:
    def test_fires_only_first(self):
        log = []
        r1 = RuleBuilder().name("r1").priority(1).when(lambda f: True).then(lambda f: log.append(1)).build()
        r2 = RuleBuilder().name("r2").priority(2).when(lambda f: True).then(lambda f: log.append(2)).build()
        FirstApplicableRulesEngine().fire(Rules(r1, r2), Facts())
        assert log == [1]


# ─────────────────────────────────────────────────────────────────── #
#  InferenceRulesEngine                                               #
# ─────────────────────────────────────────────────────────────────── #

class TestInferenceRulesEngine:
    def test_count_down(self):
        counter = {"n": 3}
        f = Facts(n=3)

        def decrement(facts):
            facts.put("n", facts.get("n") - 1)
            counter["n"] = facts.get("n")

        r = RuleBuilder().name("decrement").when(lambda f: f.get("n") > 0).then(decrement).build()
        InferenceRulesEngine().fire(Rules(r), f)
        assert f.get("n") == 0

    def test_max_iterations(self):
        f = Facts(n=1000)
        r = RuleBuilder().name("r").when(lambda f: f.get("n") > 0).then(lambda f: f.put("n", f.get("n") - 1)).build()
        params = RulesEngineParameters(max_iterations=5)
        InferenceRulesEngine(parameters=params).fire(Rules(r), f)
        # Should stop after 5 iterations, not reach 0
        assert f.get("n") == 995


# ─────────────────────────────────────────────────────────────────── #
#  Composite rules                                                    #
# ─────────────────────────────────────────────────────────────────── #

class TestUnitRuleGroup:
    def _r(self, name, cond):
        return RuleBuilder().name(name).when(cond).then(lambda f: None).build()

    def test_all_true(self):
        g = UnitRuleGroup("g")
        g.add_rule(self._r("a", lambda f: True))
        g.add_rule(self._r("b", lambda f: True))
        assert g.evaluate(Facts()) is True

    def test_one_false(self):
        g = UnitRuleGroup("g")
        g.add_rule(self._r("a", lambda f: True))
        g.add_rule(self._r("b", lambda f: False))
        assert g.evaluate(Facts()) is False

    def test_execute_all_actions(self):
        log = []
        r1 = RuleBuilder().name("r1").when(lambda f: True).then(lambda f: log.append(1)).build()
        r2 = RuleBuilder().name("r2").when(lambda f: True).then(lambda f: log.append(2)).build()
        g = UnitRuleGroup("g")
        g.add_rule(r1)
        g.add_rule(r2)
        g.execute(Facts())
        assert sorted(log) == [1, 2]


class TestActivationRuleGroup:
    def test_fires_first_match(self):
        log = []
        r1 = RuleBuilder().name("gold").priority(1).when(lambda f: f.get("tier") == "gold").then(lambda f: log.append("gold")).build()
        r2 = RuleBuilder().name("silver").priority(2).when(lambda f: True).then(lambda f: log.append("silver")).build()
        g = ActivationRuleGroup("discount")
        g.add_rule(r1)
        g.add_rule(r2)
        g.execute(Facts(tier="gold"))
        assert log == ["gold"]

    def test_falls_back_to_second(self):
        log = []
        r1 = RuleBuilder().name("gold").priority(1).when(lambda f: f.get("tier") == "gold").then(lambda f: log.append("gold")).build()
        r2 = RuleBuilder().name("silver").priority(2).when(lambda f: True).then(lambda f: log.append("silver")).build()
        g = ActivationRuleGroup("discount")
        g.add_rule(r1)
        g.add_rule(r2)
        g.execute(Facts(tier="bronze"))
        assert log == ["silver"]


class TestConditionalRuleGroup:
    def test_guard_blocks(self):
        log = []
        guard = RuleBuilder().name("guard").priority(0).when(lambda f: False).then(lambda f: log.append("guard")).build()
        child = RuleBuilder().name("child").priority(1).when(lambda f: True).then(lambda f: log.append("child")).build()
        g = ConditionalRuleGroup("g")
        g.add_rule(guard)
        g.add_rule(child)
        assert g.evaluate(Facts()) is False

    def test_guard_allows(self):
        log = []
        guard = RuleBuilder().name("guard").priority(0).when(lambda f: True).then(lambda f: log.append("guard")).build()
        child = RuleBuilder().name("child").priority(1).when(lambda f: True).then(lambda f: log.append("child")).build()
        g = ConditionalRuleGroup("g")
        g.add_rule(guard)
        g.add_rule(child)
        g.execute(Facts())
        assert "guard" in log
        assert "child" in log


# ─────────────────────────────────────────────────────────────────── #
#  ExpressionRule                                                     #
# ─────────────────────────────────────────────────────────────────── #

class TestExpressionRule:
    def test_condition_eval(self):
        r = ExpressionRule("r", condition="x > 0", actions=["pass"])
        assert r.evaluate(Facts(x=5)) is True
        assert r.evaluate(Facts(x=-1)) is False

    def test_action_exec(self):
        r = ExpressionRule("r", condition="True", actions=["result = x * 2"])
        f = Facts(x=5, result=0)
        r.execute(f)
        assert f.get("result") == 10

    def test_multi_actions(self):
        r = ExpressionRule("r", condition="True", actions=["a = 1", "b = 2"])
        f = Facts(a=0, b=0)
        r.execute(f)
        assert f.get("a") == 1
        assert f.get("b") == 2


# ─────────────────────────────────────────────────────────────────── #
#  RuleListener                                                       #
# ─────────────────────────────────────────────────────────────────── #

class TestRuleListener:
    def test_on_success_called(self):
        events = []

        class Listener(RuleListener):
            def on_success(self, rule, facts):
                events.append(("success", rule.name))

        r = RuleBuilder().name("r").when(lambda f: True).then(lambda f: None).build()
        engine = DefaultRulesEngine(rule_listeners=[Listener()])
        engine.fire(Rules(r), Facts())
        assert ("success", "r") in events

    def test_on_failure_called(self):
        events = []

        class Listener(RuleListener):
            def on_failure(self, rule, facts, exc):
                events.append(("failure", rule.name))

        r = RuleBuilder().name("r").when(lambda f: True).then(lambda f: (_ for _ in ()).throw(RuntimeError("oops"))).build()

        # rebuild with a simpler throwing action
        r2 = BasicRule(
            name="r2",
            condition_fn=lambda f: True,
            action_fns=[lambda f: (_ for _ in ()).throw(RuntimeError("oops"))],
        )
        # Use a different approach
        def throwing_action(f):
            raise RuntimeError("oops")

        r3 = BasicRule(name="r3", condition_fn=lambda f: True, action_fns=[throwing_action])
        engine = DefaultRulesEngine(rule_listeners=[Listener()])
        engine.fire(Rules(r3), Facts())
        assert ("failure", "r3") in events

    def test_veto_via_before_evaluate(self):
        log = []

        class VetoListener(RuleListener):
            def before_evaluate(self, rule, facts):
                return False  # veto everything

        r = RuleBuilder().name("r").when(lambda f: True).then(lambda f: log.append("x")).build()
        engine = DefaultRulesEngine(rule_listeners=[VetoListener()])
        engine.fire(Rules(r), Facts())
        assert log == []
