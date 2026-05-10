"""
tests/conftest.py — pytest configuration and skip helpers.

Test tiers
----------
core    No optional dependencies (rules, batch, flows, states, integration,
        retry, observe, validate, events, graph, async_, saga, schedule, config)
yaml    Requires PyYAML  (pip install maestro-sdk[yaml])
llm     Requires anthropic AND/OR openai  (pip install maestro-sdk[llm])
agents  Requires langgraph + langchain-core  (pip install maestro-sdk[agents])
slow    Timing-dependent tests (scheduler, sleep-based assertions)

Usage in test files
-------------------
Mark entire modules::

    pytestmark = pytest.mark.core          # all tests in the file

Mark individual tests::

    @pytest.mark.yaml
    def test_yaml_rule_factory(): ...

    @pytest.mark.agents
    def test_langgraph_agent(): ...

Running subsets::

    pytest -m core                         # only core tests
    pytest -m "core or yaml"               # core + yaml
    pytest -m "not slow"                   # skip slow tests
    pytest -m "not (agents or llm)"        # skip all optional-dep tests
"""
from __future__ import annotations

import importlib
import sys

import pytest


# ════════════════════════════════════════════════════════════════════════════
#  Availability helpers
# ════════════════════════════════════════════════════════════════════════════

def _available(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


HAS_YAML      = _available("yaml")
HAS_ANTHROPIC = _available("anthropic")
HAS_OPENAI    = _available("openai")
HAS_LANGGRAPH = _available("langgraph")
HAS_LANGCHAIN = _available("langchain_core")
HAS_LLM       = HAS_ANTHROPIC or HAS_OPENAI
HAS_AGENTS    = HAS_LANGGRAPH and HAS_LANGCHAIN


# ════════════════════════════════════════════════════════════════════════════
#  pytest marker registration
# ════════════════════════════════════════════════════════════════════════════

def pytest_configure(config):
    config.addinivalue_line("markers", "core: core SDK — no optional dependencies")
    config.addinivalue_line("markers", "yaml: requires PyYAML")
    config.addinivalue_line("markers", "llm: requires anthropic and/or openai")
    config.addinivalue_line("markers", "agents: requires langgraph + langchain-core")
    config.addinivalue_line("markers", "slow: timing-dependent (scheduler, sleep-based)")


# ════════════════════════════════════════════════════════════════════════════
#  Auto-skip on missing optional deps
# ════════════════════════════════════════════════════════════════════════════

def pytest_runtest_setup(item):
    markers = {m.name for m in item.iter_markers()}

    if "yaml" in markers and not HAS_YAML:
        pytest.skip("PyYAML not installed — pip install 'maestro-sdk[yaml]'")

    if "llm" in markers and not HAS_LLM:
        pytest.skip("No LLM provider — pip install 'maestro-sdk[llm]'")

    if "agents" in markers and not HAS_AGENTS:
        pytest.skip("LangGraph not installed — pip install 'maestro-sdk[agents]'")


# ════════════════════════════════════════════════════════════════════════════
#  Convenience skip decorators (for use in test files)
# ════════════════════════════════════════════════════════════════════════════

skip_no_yaml     = pytest.mark.skipif(not HAS_YAML,      reason="PyYAML not installed")
skip_no_anthropic = pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic not installed")
skip_no_openai   = pytest.mark.skipif(not HAS_OPENAI,    reason="openai not installed")
skip_no_llm      = pytest.mark.skipif(not HAS_LLM,       reason="no LLM provider installed")
skip_no_agents   = pytest.mark.skipif(not HAS_AGENTS,    reason="langgraph not installed")
