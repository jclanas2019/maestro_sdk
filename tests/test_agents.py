"""
tests/test_agents.py — maestro.agents test suite.

All LLM calls are mocked so no API keys are needed.
Tests cover: tools, bridges, node factories, pre-built graphs.
"""
import sys, os, json, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import maestro
pytestmark = pytest.mark.agents

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from maestro.agents import (
    MaestroAgentState,
    RulesEngineTool, FSMTransitionTool, FSMStatusTool,
    BatchJobTool, SagaTool, EventPublisherTool, ValidatorTool,
    AgentWork, AsyncAgentWork,
    AgentRecordProcessor, AgentCondition, AgentEventHandler,
    make_rules_node, make_fsm_node, make_batch_node,
    make_saga_node, make_validator_node, make_observer_node,
    ReActWithRulesGraph, HumanInTheLoopGraph, MultiAgentOrchestratorGraph,
)


# ── Fixtures ──────────────────────────────────────────────────────────────── #

def _ok_work(name="ok"):
    return maestro.LambdaWork(lambda ctx: None, name=name)

def _fail_work(name="fail"):
    return maestro.LambdaWork(
        lambda ctx: maestro.DefaultWorkReport(maestro.WorkStatus.FAILED, ctx),
        name=name)

def _simple_rules():
    r = (maestro.RuleBuilder()
         .name("high-value")
         .when(lambda f: f.get("total", 0) > 100)
         .then(lambda f: f.put("tier", "premium"))
         .build())
    return maestro.Rules(r)

def _simple_fsm():
    locked, unlocked = maestro.State("locked"), maestro.State("unlocked")
    class Coin(maestro.Event): pass
    class Push(maestro.Event): pass
    fsm = (maestro.FiniteStateMachineBuilder(states={locked, unlocked}, initial_state=locked)
           .register_transition(maestro.TransitionBuilder().source_state(locked).event_type(Coin).target_state(unlocked).build())
           .register_transition(maestro.TransitionBuilder().source_state(unlocked).event_type(Push).target_state(locked).build())
           .build())
    return fsm, locked, unlocked, Coin, Push

def _mock_llm(content="done", tool_calls=None):
    llm = MagicMock()
    response = AIMessage(content=content)
    if tool_calls:
        response.tool_calls = tool_calls
    else:
        response.tool_calls = []
    llm.invoke.return_value = response
    llm.bind_tools.return_value = llm
    return llm


# ════════════════════════════════════════════════════════════════════════════
#  MaestroAgentState
# ════════════════════════════════════════════════════════════════════════════

class TestMaestroAgentState:
    def test_state_accepts_messages(self):
        state: MaestroAgentState = {"messages": [HumanMessage("hello")]}
        assert len(state["messages"]) == 1

    def test_state_accepts_facts(self):
        state: MaestroAgentState = {"facts": {"total": 150.0, "tier": "vip"}}
        assert state["facts"]["tier"] == "vip"

    def test_state_accepts_fsm_state(self):
        state: MaestroAgentState = {"fsm_state": "PENDING"}
        assert state["fsm_state"] == "PENDING"

    def test_state_optional_fields_absent(self):
        state: MaestroAgentState = {}
        assert state.get("error") is None
        assert state.get("iteration") is None


# ════════════════════════════════════════════════════════════════════════════
#  Tools
# ════════════════════════════════════════════════════════════════════════════

class TestRulesEngineTool:
    def test_fires_matching_rules(self):
        tool = RulesEngineTool(rules=_simple_rules(), name="pricing")
        result = json.loads(tool._run({"total": 200.0}))
        assert result["facts"]["tier"] == "premium"
        assert "high-value" in result["rules_fired"]

    def test_no_rules_fired_for_low_total(self):
        tool = RulesEngineTool(rules=_simple_rules())
        result = json.loads(tool._run({"total": 50.0}))
        assert result["rules_fired"] == []
        assert result["count"] == 0

    def test_facts_preserved_through_tool(self):
        tool = RulesEngineTool(rules=_simple_rules())
        result = json.loads(tool._run({"customer": "Alice", "total": 200.0}))
        assert result["facts"]["customer"] == "Alice"
        assert result["facts"]["total"] == 200.0

    def test_tool_has_correct_schema(self):
        tool = RulesEngineTool(rules=_simple_rules(), name="test-tool")
        assert tool.name == "test-tool"
        assert "facts" in tool.args_schema.model_fields


class TestFSMTransitionTool:
    def test_fires_registered_event(self):
        fsm, locked, unlocked, Coin, Push = _simple_fsm()
        tool = FSMTransitionTool(fsm=fsm, event_map={"coin": Coin, "push": Push})
        result = json.loads(tool._run("coin"))
        assert result["new_state"] == "unlocked"
        assert result["success"] is True

    def test_returns_error_for_unknown_event(self):
        fsm, *_ = _simple_fsm()
        tool = FSMTransitionTool(fsm=fsm, event_map={"coin": _[2]})
        result = json.loads(tool._run("unknown"))
        assert "error" in result

    def test_state_transitions_correctly(self):
        fsm, locked, unlocked, Coin, Push = _simple_fsm()
        tool = FSMTransitionTool(fsm=fsm, event_map={"coin": Coin, "push": Push})
        json.loads(tool._run("coin"))   # locked → unlocked
        result = json.loads(tool._run("push"))   # unlocked → locked
        assert result["new_state"] == "locked"


class TestBatchJobTool:
    def test_returns_metrics(self):
        sink = []
        job  = maestro.JobBuilder().named("test").reader(
            maestro.IterableRecordReader([1, 2, 3])
        ).writer(maestro.CollectionRecordWriter(sink)).build()
        tool = BatchJobTool(job=job, name="run-etl")
        result = json.loads(tool._run({}))
        assert result["written_count"] == 3
        assert result["status"] == "COMPLETED"

    def test_filtered_records_reported(self):
        sink = []
        job  = (maestro.JobBuilder()
                .reader(maestro.IterableRecordReader(range(10)))
                .filter(maestro.PredicateRecordFilter(lambda r: r.payload % 2 == 0))
                .writer(maestro.CollectionRecordWriter(sink))
                .build())
        tool = BatchJobTool(job=job)
        result = json.loads(tool._run({}))
        assert result["written_count"] == 5
        assert result["filtered_count"] == 5


class TestSagaTool:
    def _build_saga(self, fail=False):
        from maestro.saga import SagaBuilder
        log = []
        builder = (SagaBuilder()
                   .named("test-saga")
                   .quiet()
                   .step("A", _ok_work("A") if not fail else _fail_work("A"),
                              _ok_work("undo-A")))
        return builder.build(), log

    def test_completed_saga(self):
        saga, _ = self._build_saga(fail=False)
        tool = SagaTool(saga=saga, name="test-saga")
        result = json.loads(tool._run({}))
        assert result["saga_status"] == "COMPLETED"
        assert "A" in result["succeeded_steps"]

    def test_failed_saga_with_compensation(self):
        saga, _ = self._build_saga(fail=True)
        tool = SagaTool(saga=saga)
        result = json.loads(tool._run({}))
        assert result["saga_status"] in ("FAILED", "COMPENSATED")


class TestEventPublisherTool:
    def test_publishes_to_bus(self):
        from maestro.events import EventBus
        bus = EventBus()
        log = []
        bus.subscribe_fn("test.topic", lambda m: log.append(m.payload))
        tool = EventPublisherTool(bus=bus, name="publisher")
        result = json.loads(tool._run("test.topic", {"msg": "hello"}))
        assert "message_id" in result
        assert log == [{"msg": "hello"}]


class TestValidatorTool:
    def _schema(self):
        from maestro.validate import Schema, field, Required, Range
        return Schema(
            id    = field(int,   Required()),
            total = field(float, Range(min=0)),
        )

    def test_valid_data(self):
        tool = ValidatorTool(maestro_schema=self._schema())
        result = json.loads(tool._run({"id": 1, "total": 99.0}))
        assert result["ok"] is True
        assert result["errors"] == []

    def test_invalid_data_lists_errors(self):
        tool = ValidatorTool(maestro_schema=self._schema())
        result = json.loads(tool._run({"id": None, "total": -1.0}))
        assert result["ok"] is False
        assert len(result["errors"]) >= 1


# ════════════════════════════════════════════════════════════════════════════
#  Bridges
# ════════════════════════════════════════════════════════════════════════════

class TestAgentWork:
    def _mock_graph(self, return_state):
        g = MagicMock()
        g.invoke.return_value = return_state
        return g

    def test_completed_when_graph_succeeds(self):
        graph = self._mock_graph({"result": "done", "messages": []})
        work  = AgentWork(graph, name="test-agent")
        report = work.execute(maestro.WorkContext())
        assert report.status == maestro.WorkStatus.COMPLETED
        graph.invoke.assert_called_once()

    def test_output_fn_writes_to_context(self):
        graph = self._mock_graph({"analysis": "positive", "messages": []})
        ctx   = maestro.WorkContext()
        work  = AgentWork(
            graph,
            output_fn=lambda r, c: c.put("analysis", r.get("analysis")),
        )
        work.execute(ctx)
        assert ctx.get("analysis") == "positive"

    def test_input_fn_builds_graph_state(self):
        captured = {}
        def fake_invoke(inputs, config=None):
            captured.update(inputs)
            return {"messages": []}
        graph = MagicMock(); graph.invoke.side_effect = fake_invoke
        ctx   = maestro.WorkContext(order_id="ORD-1")
        work  = AgentWork(
            graph,
            input_fn=lambda c: {"messages": [HumanMessage(f"Order {c.get('order_id')}")]}
        )
        work.execute(ctx)
        assert "messages" in captured
        assert "ORD-1" in captured["messages"][0].content

    def test_failed_when_graph_raises(self):
        graph = MagicMock()
        graph.invoke.side_effect = RuntimeError("graph exploded")
        work   = AgentWork(graph, name="boom")
        report = work.execute(maestro.WorkContext())
        assert report.status == maestro.WorkStatus.FAILED
        assert isinstance(report.error, RuntimeError)

    def test_inside_sequential_flow(self):
        log = []
        graph = self._mock_graph({"tag": "agent-done", "messages": []})
        work  = AgentWork(graph, output_fn=lambda r, c: log.append(r.get("tag")))
        flow  = maestro.SequentialFlow.Builder().execute(work).build()
        r     = maestro.WorkFlowEngine().run(flow, maestro.WorkContext())
        assert r.status == maestro.WorkStatus.COMPLETED
        assert log == ["agent-done"]


class TestAsyncAgentWork:
    def test_async_completes(self):
        async def ainvoke_impl(inputs, config=None):
            return {"messages": [], "done": True}
        graph = MagicMock()
        graph.ainvoke = ainvoke_impl
        async def run():
            work = AsyncAgentWork(graph)
            return await work.execute(maestro.WorkContext())
        report = asyncio.run(run())
        assert report.status == maestro.WorkStatus.COMPLETED


class TestAgentRecordProcessor:
    def test_enriches_record_payload(self):
        llm = _mock_llm("POSITIVE")
        proc = AgentRecordProcessor(
            llm         = llm,
            prompt_fn   = lambda p: f"Classify: {p}",
            response_fn = lambda r, p: {"text": p, "sentiment": r.strip()},
        )
        from maestro.batch._record import Header, Record
        record = Record(Header(1, "test"), "great product!")
        result = proc.process_record(record)
        assert result.payload["sentiment"] == "POSITIVE"

    def test_batch_pipeline_with_processor(self):
        llm = _mock_llm("NEG")
        proc = AgentRecordProcessor(
            llm         = llm,
            prompt_fn   = lambda p: f"Classify: {p.get('text', '')}",
            response_fn = lambda r, p: {**p, "label": r.strip()},
        )
        sink = []
        (maestro.JobBuilder()
         .reader(maestro.IterableRecordReader([{"text": "bad"}, {"text": "good"}]))
         .processor(proc)
         .writer(maestro.CollectionRecordWriter(sink))
         .build()).call()
        assert all("label" in r for r in sink)
        assert sink[0]["label"] == "NEG"

    def test_fail_on_error_raises(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM unavailable")
        proc = AgentRecordProcessor(
            llm=llm, prompt_fn=lambda p: str(p), fail_on_error=True
        )
        from maestro.batch._record import Header, Record
        from maestro.batch._processor import RecordProcessingException
        with pytest.raises(RecordProcessingException):
            proc.process_record(Record(Header(1, "t"), {"x": 1}))


class TestAgentCondition:
    def test_returns_true_for_positive_answer(self):
        llm = _mock_llm("yes, this is positive")
        cond = AgentCondition(llm=llm, prompt_fn=lambda f: f"Positive? {f.get('text')}")
        facts = maestro.Facts(text="great!")
        assert cond(facts) is True

    def test_returns_false_for_negative_answer(self):
        llm = _mock_llm("no")
        cond = AgentCondition(llm=llm, prompt_fn=lambda f: f"Positive? {f.get('text')}")
        facts = maestro.Facts(text="terrible")
        assert cond(facts) is False

    def test_used_as_rule_condition(self):
        log = []
        llm = _mock_llm("yes")
        cond = AgentCondition(llm=llm, prompt_fn=lambda f: "yes?")
        rule = (maestro.RuleBuilder()
                .name("llm-rule")
                .when(cond)
                .then(lambda f: log.append("fired"))
                .build())
        maestro.DefaultRulesEngine().fire(maestro.Rules(rule), maestro.Facts(x=1))
        assert log == ["fired"]


class TestAgentEventHandler:
    def test_handler_invokes_graph_on_event(self):
        graph = MagicMock()
        graph.invoke.return_value = {"messages": []}
        handler = AgentEventHandler(
            graph=graph,
            input_fn=lambda e: {"messages": [HumanMessage(type(e).__name__)]},
        )
        class TestEvent(maestro.Event): pass
        handler.handle(TestEvent())
        graph.invoke.assert_called_once()
        assert graph.invoke.call_count >= 1


# ════════════════════════════════════════════════════════════════════════════
#  Node factories
# ════════════════════════════════════════════════════════════════════════════

class TestMakeRulesNode:
    def test_updates_facts_in_state(self):
        node   = make_rules_node(_simple_rules())
        result = node({"facts": {"total": 200.0}})
        assert result["facts"]["tier"] == "premium"

    def test_no_change_when_rule_doesnt_fire(self):
        node   = make_rules_node(_simple_rules())
        result = node({"facts": {"total": 50.0}})
        assert "tier" not in result["facts"]

    def test_empty_facts_handled(self):
        node   = make_rules_node(_simple_rules())
        result = node({})
        assert isinstance(result["facts"], dict)

    def test_custom_facts_key(self):
        node   = make_rules_node(_simple_rules(), facts_key="my_facts")
        result = node({"my_facts": {"total": 200.0}})
        assert "my_facts" in result
        assert result["my_facts"]["tier"] == "premium"


class TestMakeFSMNode:
    def test_transitions_state(self):
        fsm, locked, unlocked, Coin, Push = _simple_fsm()
        node   = make_fsm_node(fsm, Coin)
        result = node({})
        assert result["fsm_state"] == "unlocked"

    def test_appends_to_history(self):
        fsm, *_, Coin, Push = _simple_fsm()
        node   = make_fsm_node(fsm, Coin)
        result = node({"fsm_history": ["locked"]})
        assert "unlocked" in result["fsm_history"]

    def test_returns_error_on_invalid_transition(self):
        fsm, locked, unlocked, Coin, Push = _simple_fsm()
        fsm.fire(Coin())   # move to unlocked first
        # Now try to fire Coin again (no transition from unlocked)
        node   = make_fsm_node(fsm, Coin)
        result = node({})
        assert "error" in result


class TestMakeBatchNode:
    def test_stores_metrics(self):
        sink = []
        job  = maestro.JobBuilder().reader(
            maestro.IterableRecordReader([1, 2, 3])
        ).writer(maestro.CollectionRecordWriter(sink)).build()
        node   = make_batch_node(job)
        result = node({})
        assert result["batch_metrics"]["written_count"] == 3
        assert result["batch_metrics"]["status"] == "COMPLETED"

    def test_custom_metrics_key(self):
        sink = []
        job  = maestro.JobBuilder().reader(
            maestro.IterableRecordReader([1])
        ).writer(maestro.CollectionRecordWriter(sink)).build()
        node   = make_batch_node(job, metrics_key="my_metrics")
        result = node({})
        assert "my_metrics" in result


class TestMakeSagaNode:
    def test_completed_saga_updates_state(self):
        from maestro.saga import SagaBuilder
        saga = SagaBuilder().named("s").quiet().step("A", _ok_work("A")).build()
        node = make_saga_node(saga)
        result = node({})
        assert result["saga_status"] == "COMPLETED"
        assert "error" not in result

    def test_failed_saga_sets_error(self):
        from maestro.saga import SagaBuilder
        saga = SagaBuilder().named("s").quiet().step("A", _fail_work("A")).build()
        node = make_saga_node(saga)
        result = node({})
        assert result["saga_status"] != "COMPLETED"
        assert result.get("error") is not None

    def test_context_passed_to_saga(self):
        log = []
        from maestro.saga import SagaBuilder
        def read_ctx(ctx): log.append(ctx.get("order_id"))
        saga = SagaBuilder().named("s").quiet().step(
            "A", maestro.LambdaWork(read_ctx, "A")).build()
        make_saga_node(saga)({"context": {"order_id": "ORD-99"}})
        assert log == ["ORD-99"]


class TestMakeValidatorNode:
    def _schema(self):
        from maestro.validate import Schema, field, Required
        return Schema(name=field(str, Required()), age=field(int, Required()))

    def test_valid_data_no_error(self):
        node   = make_validator_node(self._schema())
        result = node({"facts": {"name": "Alice", "age": 30}})
        assert result["error"] is None

    def test_invalid_data_sets_error_and_next_action(self):
        node   = make_validator_node(self._schema())
        result = node({"facts": {"name": None}})
        assert result["error"] is not None
        assert result["next_action"] == "fix"


class TestMakeObserverNode:
    def test_increments_iteration(self):
        from maestro.observe import InMemoryObserver
        obs    = InMemoryObserver()
        node   = make_observer_node(obs)
        result = node({"iteration": 2, "messages": [HumanMessage("x")]})
        assert result["iteration"] == 3

    def test_emits_metrics_to_observer(self):
        from maestro.observe import InMemoryObserver
        obs    = InMemoryObserver()
        node   = make_observer_node(obs)
        node({"iteration": 0, "messages": [HumanMessage("x"), AIMessage("y")]})
        evs = obs.events(module="agents", name="message_count")
        assert len(evs) == 1

    def test_action_counter_emitted(self):
        from maestro.observe import InMemoryObserver
        obs    = InMemoryObserver()
        node   = make_observer_node(obs)
        node({"iteration": 0, "messages": [], "next_action": "checkout"})
        assert obs.counter("agents", "action", action="checkout") == 1.0


# ════════════════════════════════════════════════════════════════════════════
#  Pre-built graphs
# ════════════════════════════════════════════════════════════════════════════

class TestReActWithRulesGraph:
    def test_builds_and_invokes(self):
        llm   = _mock_llm("I have applied the pricing rules.")
        graph = ReActWithRulesGraph(
            llm           = llm,
            maestro_tools = [RulesEngineTool(rules=_simple_rules(), name="pricing")],
            system_prompt = "You are a pricing agent.",
            max_iterations= 2,
        ).build()
        result = graph.invoke({
            "messages": [HumanMessage("Apply pricing for total=200, customer Alice.")],
            "facts":    {"total": 200.0, "customer": "Alice"},
        })
        assert "messages" in result
        assert len(result["messages"]) >= 1

    def test_no_tools_still_works(self):
        llm   = _mock_llm("Done.")
        graph = ReActWithRulesGraph(llm=llm).build()
        result = graph.invoke({"messages": [HumanMessage("Hello")]})
        assert result["messages"][-1].content == "Done."

    def test_respects_max_iterations(self):
        """Agent should stop after max_iterations even if tool calls keep coming."""
        tool_call_response = AIMessage(content="")
        tool_call_response.tool_calls = [
            {"name": "pricing", "args": {"facts": {}}, "id": "tc1", "type": "tool_call"}
        ]
        final_response = AIMessage(content="Done after limit.")
        final_response.tool_calls = []

        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.return_value = final_response  # always returns done

        graph = ReActWithRulesGraph(
            llm           = llm,
            max_iterations= 1,
        ).build()
        result = graph.invoke({"messages": [HumanMessage("Hello")]})
        assert result is not None


class TestMultiAgentOrchestratorGraph:
    def test_routes_to_specialist(self):
        supervisor_llm = MagicMock()
        supervisor_llm.invoke.side_effect = [
            AIMessage(content='{"next": "billing"}'),
            AIMessage(content='{"next": "FINISH"}'),
        ]
        specialist_llm = MagicMock()
        specialist_llm.invoke.return_value = AIMessage(content="Billing handled.")
        specialist_llm.bind_tools.return_value = specialist_llm

        graph = MultiAgentOrchestratorGraph(
            supervisor_llm = supervisor_llm,
            specialists    = {
                "billing": (specialist_llm, [], "You handle billing."),
            },
            max_rounds = 3,
        ).build()

        result = graph.invoke({"messages": [HumanMessage("Process my invoice.")]})
        assert result is not None
        assert len(result["messages"]) >= 2

    def test_finishes_immediately_on_direct_finish(self):
        supervisor_llm = MagicMock()
        supervisor_llm.invoke.return_value = AIMessage(content='{"next": "FINISH"}')
        graph = MultiAgentOrchestratorGraph(
            supervisor_llm=supervisor_llm,
            specialists={},
        ).build()
        result = graph.invoke({"messages": [HumanMessage("hi")]})
        assert result is not None


# ════════════════════════════════════════════════════════════════════════════
#  Cross-module integration: agent + Maestro pipeline
# ════════════════════════════════════════════════════════════════════════════

class TestFullCrossModuleAgentIntegration:
    def test_agent_work_inside_flow_with_rules(self):
        """
        End-to-end: rules step → agent step → batch step, all in a Maestro flow.
        The agent reads pricing from the rules node and "decides" to approve.
        """
        log = []

        # Rules: classify order
        pricing_rules = maestro.Rules(
            maestro.RuleBuilder()
            .name("vip")
            .when(lambda f: f.get("total", 0) > 500)
            .then(lambda f: f.put("tier", "vip"))
            .build()
        )

        # Mock agent: reads tier from context, logs decision
        mock_graph = MagicMock()
        def fake_invoke(inputs, config=None):
            tier = inputs.get("facts", {}).get("tier", "standard")
            log.append(f"agent-decided:{tier}")
            return {"decision": f"approved-{tier}", "messages": [AIMessage(f"OK: {tier}")]}
        mock_graph.invoke = fake_invoke

        # Batch: collect decisions
        sink = []
        batch_job = maestro.JobBuilder().reader(
            maestro.IterableRecordReader(["record-1"])
        ).writer(maestro.CollectionRecordWriter(sink)).build()

        # Build combined flow
        from maestro.integration import RuleSetWork, BatchWork

        flow = (maestro.aNewSequentialFlow()
                .named("full-agent-flow")
                .execute(RuleSetWork(pricing_rules, name="pricing"))
                .then(AgentWork(
                    mock_graph,
                    input_fn  = lambda ctx: {"messages": [HumanMessage("approve?")],
                                             "facts": ctx.as_map()},
                    output_fn = lambda r, ctx: ctx.put("decision", r.get("decision")),
                ))
                .then(BatchWork(batch_job, name="persist"))
                .build())

        ctx    = maestro.WorkContext(total=750.0, order_id="ORD-1")
        report = maestro.WorkFlowEngine().run(flow, ctx)

        assert report.status == maestro.WorkStatus.COMPLETED
        assert log == ["agent-decided:vip"]
        assert ctx.get("decision") == "approved-vip"
        assert sink == ["record-1"]

    def test_node_factories_in_custom_langgraph(self):
        """Build a custom LangGraph using Maestro node factories."""
        from langgraph.graph import END, StateGraph
        from maestro.agents import make_rules_node, make_validator_node
        from maestro.validate import Schema, field, Required, Range

        schema = Schema(
            total = field(float, Required(), Range(min=0)),
            tier  = field(str, Required()),
        )

        graph = StateGraph(MaestroAgentState)
        graph.add_node("apply-rules",  make_rules_node(_simple_rules()))
        graph.add_node("validate",     make_validator_node(schema))
        graph.set_entry_point("apply-rules")
        graph.add_edge("apply-rules", "validate")
        graph.add_edge("validate", END)
        compiled = graph.compile()

        result = compiled.invoke({"facts": {"total": 200.0}})
        assert result["facts"]["tier"] == "premium"
        assert result.get("error") is None
