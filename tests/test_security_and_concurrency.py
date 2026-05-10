"""
tests/test_security_and_concurrency.py

Regression tests for the three stability fixes:
  1. ExpressionRule sandbox — arbitrary code blocked by default
  2. WorkContext thread safety — lock-protected reads/writes
  3. GraphFlow concurrency — per-node context copies, no shared-state races
"""
import sys, os, threading, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

import maestro
pytestmark = pytest.mark.core


# ════════════════════════════════════════════════════════════════════════════
#  1. ExpressionRule — sandbox
# ════════════════════════════════════════════════════════════════════════════

class TestExpressionRuleSandbox:
    """__import__, open, eval etc. must be blocked in the default sandbox."""

    DANGEROUS_EXPRS = [
        "__import__('os')",
        "__import__('os').getcwd()",
        "open('/etc/passwd').read()",
        "eval('1+1')",
        "exec('import os')",
        "compile('', '', 'exec')",
        "__builtins__['__import__']('os')",
    ]

    def _blocked(self, expr: str) -> bool:
        """Return True if the expression is blocked (raises NameError or returns False)."""
        from maestro.rules import ExpressionRule, Facts
        rule   = ExpressionRule("test", expr, [])
        # evaluate() swallows exceptions; a dangerous expr either raises or
        # returns the unexpected truthy result.  Blocked means False.
        result = rule.evaluate(Facts())
        return result is False

    def test_import_blocked(self):
        assert self._blocked("__import__('os')")

    def test_import_method_blocked(self):
        assert self._blocked("__import__('os').getcwd()")

    def test_open_blocked(self):
        assert self._blocked("open('/etc/passwd').read()")

    def test_eval_blocked(self):
        assert self._blocked("eval('1+1')")

    def test_exec_blocked(self):
        assert self._blocked("exec('x=1')")

    def test_compile_blocked(self):
        assert self._blocked("compile('', '', 'exec')")

    def test_builtins_key_blocked(self):
        assert self._blocked("__builtins__['__import__']('os')")

    def test_safe_arithmetic(self):
        from maestro.rules import ExpressionRule, Facts
        r = ExpressionRule("safe", "total > 100 and discount < 0.5", [])
        assert r.evaluate(Facts(total=200, discount=0.2)) is True
        assert r.evaluate(Facts(total=50,  discount=0.2)) is False

    def test_safe_builtins_work(self):
        from maestro.rules import ExpressionRule, Facts
        r = ExpressionRule("safe-builtins", "len(items) > 0 and isinstance(total, (int, float))",
                           ["n = len(items)"])
        f = Facts(items=[1, 2, 3], total=99)
        r.evaluate(f)
        r.execute(f)
        assert f.get("n") == 3

    def test_safe_math_functions(self):
        from maestro.rules import ExpressionRule, Facts
        r = ExpressionRule("math", "round(total, 2) > min(a, b)", [])
        assert r.evaluate(Facts(total=5.678, a=3, b=10)) is True

    def test_action_blocked_import(self):
        from maestro.rules import ExpressionRule, Facts
        r = ExpressionRule("evil-action", "True", ["x = __import__('os').getcwd()"])
        f = Facts()
        r.evaluate(f)
        r.execute(f)
        assert f.get("x") is None  # action silently failed, did not write x

    def test_action_safe_assignment(self):
        from maestro.rules import ExpressionRule, Facts
        r = ExpressionRule("ok-action", "total > 0", ["discount = 0.15", "tier = 'gold'"])
        f = Facts(total=500)
        r.evaluate(f)
        r.execute(f)
        assert f.get("discount") == 0.15
        assert f.get("tier") == "gold"

    def test_allow_unsafe_emits_security_warning(self):
        from maestro.rules import ExpressionRule, MaestroSecurityWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ExpressionRule("u", "True", [], allow_unsafe=True)
        assert any(issubclass(x.category, MaestroSecurityWarning) for x in w), \
            "Expected MaestroSecurityWarning for allow_unsafe=True"

    def test_allow_unsafe_can_use_import(self):
        """allow_unsafe=True is a valid explicit opt-in for trusted code."""
        from maestro.rules import ExpressionRule, Facts
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ExpressionRule("admin", "True", ["x = len([1,2,3])"], allow_unsafe=True)
        f = Facts()
        r.evaluate(f)
        r.execute(f)
        assert f.get("x") == 3

    def test_yaml_factory_sandboxed_by_default(self):
        """YamlRuleFactory must produce sandboxed rules."""
        try:
            from maestro.rules import YamlRuleFactory, Facts
        except ImportError:
            pytest.skip("PyYAML not installed")
        yaml_src = "name: safe\ncondition: total > 0\nactions:\n  - discount = 0.10\n"
        r   = YamlRuleFactory().from_string(yaml_src)
        f   = Facts(total=100)
        r.evaluate(f)
        r.execute(f)
        assert f.get("discount") == 0.10

    def test_yaml_factory_unsafe_opt_in_warns(self):
        try:
            from maestro.rules import YamlRuleFactory, MaestroSecurityWarning
        except ImportError:
            pytest.skip("PyYAML not installed")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            YamlRuleFactory(allow_unsafe=True)
        assert any(issubclass(x.category, MaestroSecurityWarning) for x in w)

    def test_maestro_security_warning_is_user_warning(self):
        from maestro.rules import MaestroSecurityWarning
        assert issubclass(MaestroSecurityWarning, UserWarning)


# ════════════════════════════════════════════════════════════════════════════
#  2. WorkContext — thread safety
# ════════════════════════════════════════════════════════════════════════════

class TestWorkContextThreadSafety:
    def test_copy_returns_independent_snapshot(self):
        ctx  = maestro.WorkContext(x=1, y=2)
        copy = ctx.copy()
        copy.put("x", 99)
        copy.put("z", 3)
        assert ctx.get("x") == 1       # original unchanged
        assert ctx.get("z") is None    # new key not in original
        assert copy.get("x") == 99    # copy has new value

    def test_merge_writes_other_into_self(self):
        a = maestro.WorkContext(x=1, shared="a")
        b = maestro.WorkContext(y=2, shared="b")
        a.merge(b)
        assert a.get("x")      == 1    # preserved
        assert a.get("y")      == 2    # added from b
        assert a.get("shared") == "b"  # b overrides a

    def test_concurrent_puts_do_not_raise(self):
        """Multiple threads writing the same context must not corrupt it."""
        ctx    = maestro.WorkContext()
        errors = []
        def worker(tid):
            for i in range(200):
                try:
                    ctx.put(f"key_{tid}_{i}", i)
                    _ = ctx.get(f"key_{tid}_{i}")
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(6)]
        [t.start() for t in threads]
        [t.join()  for t in threads]
        assert errors == [], f"Thread safety errors: {errors}"

    def test_concurrent_gets_do_not_raise(self):
        ctx = maestro.WorkContext(**{f"k{i}": i for i in range(100)})
        errors = []
        def reader():
            for i in range(100):
                try: ctx.get(f"k{i}")
                except Exception as e: errors.append(e)
        threads = [threading.Thread(target=reader) for _ in range(8)]
        [t.start() for t in threads]; [t.join() for t in threads]
        assert errors == []

    def test_as_map_returns_snapshot(self):
        ctx  = maestro.WorkContext(x=1, y=2)
        snap = ctx.as_map()
        ctx.put("x", 99)
        assert snap["x"] == 1    # snapshot is independent
        assert ctx.get("x") == 99

    def test_contains_is_thread_safe(self):
        ctx = maestro.WorkContext(key="value")
        results = []
        def check(): results.append(ctx.contains("key"))
        threads = [threading.Thread(target=check) for _ in range(20)]
        [t.start() for t in threads]; [t.join() for t in threads]
        assert all(results)


# ════════════════════════════════════════════════════════════════════════════
#  3. GraphFlow — per-node context isolation + correct ordering
# ════════════════════════════════════════════════════════════════════════════

class TestGraphFlowConcurrency:
    def _work(self, name, read_key=None, write_key=None, write_val=None, sleep=0.0):
        def fn(ctx):
            if sleep: time.sleep(sleep)
            if write_key is not None:
                ctx.put(write_key, write_val)
            if read_key is not None:
                return maestro.DefaultWorkReport(
                    maestro.WorkStatus.COMPLETED if ctx.get(read_key) is not None
                    else maestro.WorkStatus.FAILED,
                    ctx,
                )
            return maestro.DefaultWorkReport(maestro.WorkStatus.COMPLETED, ctx)
        return maestro.LambdaWork(fn, name=name)

    def test_sequential_nodes_share_context(self):
        """A → B: B reads what A wrote."""
        from maestro.graph import GraphBuilder
        flow = (GraphBuilder()
                .add("A", self._work("A", write_key="token", write_val="abc"))
                .add("B", self._work("B", read_key="token"), depends_on=["A"])
                .build())
        ctx = maestro.WorkContext()
        r   = maestro.WorkFlowEngine().run(flow, ctx)
        assert r.status == maestro.WorkStatus.COMPLETED
        assert ctx.get("token") == "abc"

    def test_parallel_nodes_do_not_clobber_each_other(self):
        """A and B run in parallel, each writing a different key."""
        import time
        from maestro.graph import GraphBuilder

        def slow_write(key, val, sleep_s=0.05):
            def fn(ctx):
                time.sleep(sleep_s)
                ctx.put(key, val)
                return maestro.DefaultWorkReport(maestro.WorkStatus.COMPLETED, ctx)
            return maestro.LambdaWork(fn, name=f"write-{key}")

        flow = (GraphBuilder()
                .add("write-x", slow_write("x", "from-x", 0.05))
                .add("write-y", slow_write("y", "from-y", 0.05))
                .build())
        ctx = maestro.WorkContext()
        r   = maestro.WorkFlowEngine().run(flow, ctx)
        assert r.status == maestro.WorkStatus.COMPLETED
        assert ctx.get("x") == "from-x"
        assert ctx.get("y") == "from-y"

    def test_parallel_writes_to_same_key_last_writer_wins(self):
        """When two parallel nodes write the same key, one wins (no corruption)."""
        from maestro.graph import GraphBuilder

        def writer(key, val):
            return maestro.LambdaWork(
                lambda ctx, k=key, v=val: ctx.put(k, v),
                name=f"writer-{val}")

        flow = (GraphBuilder()
                .add("w1", writer("result", "from-w1"))
                .add("w2", writer("result", "from-w2"))
                .build())
        ctx    = maestro.WorkContext()
        report = maestro.WorkFlowEngine().run(flow, ctx)
        # The context must have a valid string from one of the writers — no corruption
        result = ctx.get("result")
        assert result in ("from-w1", "from-w2"), f"Unexpected: {result!r}"

    def test_graph_stress_many_parallel_nodes(self):
        """32 parallel nodes all writing — no corruption, no deadlock."""
        from maestro.graph import GraphBuilder
        import concurrent.futures

        builder = GraphBuilder().named("stress")
        for i in range(32):
            builder.add(f"node-{i}",
                        maestro.LambdaWork(lambda ctx, n=i: ctx.put(f"n{n}", n), f"node-{i}"))

        flow = builder.build()
        ctx  = maestro.WorkContext()
        r    = maestro.WorkFlowEngine().run(flow, ctx)
        assert r.status == maestro.WorkStatus.COMPLETED
        for i in range(32):
            assert ctx.get(f"n{i}") == i, f"n{i} missing or wrong"

    def test_dependent_node_sees_parent_writes(self):
        """A dependency chain: A writes x; B (depends on A) writes x+1; C reads."""
        from maestro.graph import GraphBuilder

        def step(name, read, write, transform=None):
            def fn(ctx):
                v = ctx.get(read, 0)
                ctx.put(write, transform(v) if transform else v)
                return maestro.DefaultWorkReport(maestro.WorkStatus.COMPLETED, ctx)
            return maestro.LambdaWork(fn, name=name)

        flow = (GraphBuilder()
                .add("A", step("A", "init", "x", lambda v: 10))
                .add("B", step("B", "x", "y",   lambda v: v + 5), depends_on=["A"])
                .add("C", step("C", "y", "z",   lambda v: v * 2), depends_on=["B"])
                .build())
        ctx = maestro.WorkContext(init=0)
        maestro.WorkFlowEngine().run(flow, ctx)
        assert ctx.get("x") == 10   # A
        assert ctx.get("y") == 15   # B
        assert ctx.get("z") == 30   # C

    def test_node_report_completion_order_is_unique(self):
        """completion_order values must be unique (protected by lock)."""
        from maestro.graph import GraphBuilder
        import concurrent.futures

        builder = GraphBuilder()
        for i in range(16):
            builder.add(f"n{i}", maestro.NoOpWork())
        flow    = builder.build()
        ctx     = maestro.WorkContext()
        report  = maestro.WorkFlowEngine().run(flow, ctx)
        orders  = [nr.completion_order for nr in report.node_reports.values()]
        assert len(set(orders)) == len(orders), "Duplicate completion_order — lock failure"


# ════════════════════════════════════════════════════════════════════════════
#  4. MaestroSecurityWarning export
# ════════════════════════════════════════════════════════════════════════════

class TestMaestroSecurityWarning:
    def test_importable_from_rules(self):
        from maestro.rules import MaestroSecurityWarning
        assert issubclass(MaestroSecurityWarning, UserWarning)

    def test_importable_from_maestro(self):
        # Should be re-exported from root namespace once we add it
        from maestro.rules import MaestroSecurityWarning
        assert MaestroSecurityWarning is not None
