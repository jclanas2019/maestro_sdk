"""
examples/agents_example.py — maestro.agents patterns.

Demonstrates four agentic integration patterns WITHOUT requiring a real
API key — all LLM calls use a FakeLLM that returns scripted responses.

To use a real LLM, replace FakeLLM with:
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model="claude-sonnet-4-5")

Patterns shown
--------------
1. Tools-only: Maestro components as LangChain tools in a custom graph
2. ReActWithRulesGraph: LLM reasoning + Maestro rules for decisions
3. AgentWork: LangGraph agent embedded inside a Maestro SequentialFlow
4. AgentRecordProcessor: LLM-powered batch record enrichment
5. Node factories: custom LangGraph graph with Maestro nodes
6. MultiAgentOrchestratorGraph: supervisor routing to specialists
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import maestro
from maestro.agents import (
    MaestroAgentState,
    RulesEngineTool, FSMTransitionTool, BatchJobTool, SagaTool,
    EventPublisherTool, ValidatorTool,
    AgentWork, AgentRecordProcessor, AgentCondition,
    make_rules_node, make_fsm_node, make_batch_node,
    make_saga_node, make_validator_node, make_observer_node,
    ReActWithRulesGraph, MultiAgentOrchestratorGraph,
)
from maestro.events import EventBus
from maestro.saga import SagaBuilder, SagaStatus
from maestro.observe import InMemoryObserver, MaestroObserver
from maestro.validate import Schema, field, Required, Range, OneOf
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

SEP = "═" * 62


# ════════════════════════════════════════════════════════════════════════════
#  FakeLLM — no API key needed
# ════════════════════════════════════════════════════════════════════════════

class FakeLLM:
    """Scripted LLM that cycles through a list of responses."""
    def __init__(self, responses):
        self._queue = list(responses)
        self._i     = 0

    def invoke(self, messages, **kwargs):
        resp = self._queue[self._i % len(self._queue)]
        self._i += 1
        msg = AIMessage(content=resp)
        msg.tool_calls = []
        return msg

    def bind_tools(self, tools):
        return _BoundFakeLLM(self, tools)

    def __call__(self, messages):
        return self.invoke(messages)


class _BoundFakeLLM(FakeLLM):
    def __init__(self, base, tools):
        self._queue = base._queue
        self._i     = base._i
        self._tools = {t.name: t for t in tools}

    def invoke(self, messages, **kwargs):
        resp = self._queue[self._i % len(self._queue)]
        self._i += 1
        # If response is a JSON tool call, emit a tool call message
        try:
            parsed = json.loads(resp)
            if "tool" in parsed:
                tool_name = parsed["tool"]
                tool_args = parsed.get("args", {})
                tc_msg = AIMessage(content="")
                tc_msg.tool_calls = [{
                    "name": tool_name,
                    "args": tool_args,
                    "id":   f"tc-{self._i}",
                    "type": "tool_call",
                }]
                return tc_msg
        except Exception:
            pass
        msg = AIMessage(content=resp)
        msg.tool_calls = []
        return msg


# ════════════════════════════════════════════════════════════════════════════
#  Shared setup
# ════════════════════════════════════════════════════════════════════════════

# Pricing rules
vip_rule = (maestro.RuleBuilder()
            .name("vip-discount").priority(1)
            .when(lambda f: f.get("tier") == "vip" and f.get("total", 0) > 100)
            .then(lambda f: (f.put("discount", 0.20), f.put("priority_shipping", True)))
            .build())
promo_rule = (maestro.RuleBuilder()
              .name("promo-code").priority(2)
              .when(lambda f: f.get("promo") == "SAVE25")
              .then(lambda f: f.put("discount", max(f.get("discount", 0), 0.25)))
              .build())
pricing_rules = maestro.Rules(vip_rule, promo_rule)

# Order schema
order_schema = Schema(
    order_id = field(str,   Required()),
    total    = field(float, Range(min=0.01), Required()),
    tier     = field(str,   OneOf("standard", "vip", "enterprise"), required=False),
)

# Order FSM
pending, paid, shipped = (maestro.State(s) for s in ("PENDING", "PAID", "SHIPPED"))
class PayEvent(maestro.Event):  pass
class ShipEvent(maestro.Event): pass
order_fsm = (
    maestro.FiniteStateMachineBuilder(states={pending, paid, shipped}, initial_state=pending)
    .register_transition(maestro.TransitionBuilder().source_state(pending).event_type(PayEvent).target_state(paid).build())
    .register_transition(maestro.TransitionBuilder().source_state(paid).event_type(ShipEvent).target_state(shipped).build())
    .build()
)

# Checkout saga
checkout_saga = (
    SagaBuilder().named("checkout").quiet()
    .step("reserve",
          maestro.LambdaWork(lambda ctx: ctx.put("reservation", f"RSV-{ctx.get('order_id')}"), "reserve"),
          maestro.LambdaWork(lambda ctx: print(f"    releasing {ctx.get('reservation')}"), "release"))
    .step("charge",
          maestro.LambdaWork(lambda ctx: ctx.put("txn_id", f"TXN-{ctx.get('order_id')}"), "charge"),
          maestro.LambdaWork(lambda ctx: print(f"    refunding {ctx.get('txn_id')}"), "refund"))
    .build()
)

# Event bus
bus = EventBus()
bus_log = []
bus.subscribe_fn("order.completed", lambda m: bus_log.append(f"[bus] completed: {m.payload}"))

# Observability
mem = InMemoryObserver()
obs = MaestroObserver(observers=[mem])


# ════════════════════════════════════════════════════════════════════════════
#  Pattern 1: Maestro tools in a custom ReAct-style graph
# ════════════════════════════════════════════════════════════════════════════
print(SEP); print("Pattern 1 — Maestro tools in a custom LangGraph"); print(SEP)

pricing_tool = RulesEngineTool(
    rules=pricing_rules, name="apply_pricing",
    description="Apply pricing and discount rules to an order. Returns updated facts."
)
saga_tool = SagaTool(
    saga=checkout_saga, name="checkout",
    description="Execute the checkout saga (reserve inventory + charge payment)."
)
fsm_tool = FSMTransitionTool(
    fsm=order_fsm,
    event_map={"pay": PayEvent, "ship": ShipEvent},
    name="update_order_state",
    description="Advance the order FSM by firing a named event."
)
validator_tool = ValidatorTool(
    maestro_schema=order_schema, name="validate_order",
    description="Validate an order dict. Returns ok=True or a list of errors."
)
event_tool = EventPublisherTool(
    bus=bus, name="notify",
    description="Publish a message to the event bus."
)

# Scripted LLM: validate → apply pricing → checkout → update state → notify
llm1 = _BoundFakeLLM(FakeLLM([]), tools=[pricing_tool, saga_tool, fsm_tool, validator_tool, event_tool])
llm1._queue = [
    json.dumps({"tool": "validate_order",     "args": {"data": {"order_id": "ORD-1", "total": 250.0, "tier": "vip"}}}),
    json.dumps({"tool": "apply_pricing",      "args": {"facts": {"order_id": "ORD-1", "total": 250.0, "tier": "vip"}}}),
    json.dumps({"tool": "checkout",           "args": {"context": {"order_id": "ORD-1", "total": 250.0}}}),
    json.dumps({"tool": "update_order_state", "args": {"event_name": "pay"}}),
    json.dumps({"tool": "notify",             "args": {"topic": "order.completed", "payload": {"order_id": "ORD-1", "status": "paid"}}}),
    "Order ORD-1 processed: validated, 20% VIP discount applied, payment captured, FSM → PAID.",
]

from langgraph.prebuilt import ToolNode
all_tools = [pricing_tool, saga_tool, fsm_tool, validator_tool, event_tool]

def agent_node_1(state):
    msgs = list(state.get("messages") or [])
    resp = llm1.invoke(msgs)
    return {"messages": [resp], "iteration": (state.get("iteration") or 0) + 1}

def should_continue_1(state):
    last = (state.get("messages") or [AIMessage("")])[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

graph1 = StateGraph(MaestroAgentState)
graph1.add_node("agent", agent_node_1)
graph1.add_node("tools", ToolNode(all_tools))
graph1.set_entry_point("agent")
graph1.add_conditional_edges("agent", should_continue_1, {"tools": "tools", END: END})
graph1.add_edge("tools", "agent")
compiled1 = graph1.compile()

result1 = compiled1.invoke({
    "messages": [HumanMessage("Process order ORD-1: total $250, VIP customer.")],
    "facts":    {"order_id": "ORD-1", "total": 250.0, "tier": "vip"},
})
print(f"  Messages: {len(result1['messages'])}")
print(f"  Final:    {result1['messages'][-1].content}")
print(f"  FSM state: {order_fsm.current_state}")
print(f"  Bus events: {bus_log}")


# ════════════════════════════════════════════════════════════════════════════
#  Pattern 2: ReActWithRulesGraph — structured + LLM reasoning
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("Pattern 2 — ReActWithRulesGraph"); print(SEP)

pricing_tool2 = RulesEngineTool(rules=pricing_rules, name="apply_pricing2")
llm2 = FakeLLM([
    json.dumps({"tool": "apply_pricing2", "args": {"facts": {"total": 150.0, "tier": "vip"}}}),
    "Pricing applied: 20% VIP discount. Final total $120.",
])

graph2 = ReActWithRulesGraph(
    llm           = llm2,
    maestro_tools = [pricing_tool2],
    system_prompt = "You are an order pricing agent. Apply pricing rules before confirming.",
    max_iterations= 5,
).build()

result2 = graph2.invoke({
    "messages": [HumanMessage("Apply pricing for VIP customer, total $150.")],
    "facts":    {"total": 150.0, "tier": "vip"},
})
print(f"  Agent response: {result2['messages'][-1].content}")
print(f"  Pricing rules applied. Discount would be 20% for VIP tier.")


# ════════════════════════════════════════════════════════════════════════════
#  Pattern 3: AgentWork — graph embedded in Maestro SequentialFlow
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("Pattern 3 — AgentWork inside SequentialFlow"); print(SEP)

# Mock agent: reads context, returns enriched decision
def fake_graph_invoke(inputs, config=None):
    facts  = inputs.get("facts", {})
    tier   = facts.get("tier", "standard")
    total  = facts.get("total", 0)
    discount = 0.20 if tier == "vip" else 0.10
    return {
        "messages":  [AIMessage(f"Decision: {tier} customer gets {discount:.0%} discount")],
        "decision":  f"approved-{tier}",
        "discount":  discount,
    }

from unittest.mock import MagicMock
mock_agent = MagicMock(); mock_agent.invoke = fake_graph_invoke

flow3 = (
    maestro.aNewSequentialFlow()
    .named("order-with-agent")
    .execute(maestro.LambdaWork(
        lambda ctx: ctx.put("validated", True), "validate"))
    .then(AgentWork(
        graph     = mock_agent,
        input_fn  = lambda ctx: {
            "messages": [HumanMessage(f"Process order {ctx.get('order_id')}")],
            "facts":    ctx.as_map(),
        },
        output_fn = lambda r, ctx: (
            ctx.put("decision", r.get("decision")),
            ctx.put("discount", r.get("discount")),
            ctx.put("agent_message", r["messages"][-1].content),
        ),
        name = "pricing-agent",
    ))
    .then(maestro.LambdaWork(
        lambda ctx: print(f"  discount={ctx.get('discount'):.0%}  decision={ctx.get('decision')}"),
        "log"))
    .build()
)

ctx3 = maestro.WorkContext(order_id="ORD-2", total=200.0, tier="vip")
r3   = maestro.WorkFlowEngine().run(flow3, ctx3)
print(f"  Flow status: {r3.status.value}")
print(f"  Agent: {ctx3.get('agent_message')}")


# ════════════════════════════════════════════════════════════════════════════
#  Pattern 4: AgentRecordProcessor — LLM enriches batch records
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("Pattern 4 — AgentRecordProcessor in batch pipeline"); print(SEP)

labels = ["POSITIVE", "NEGATIVE", "POSITIVE", "NEUTRAL", "NEGATIVE"]
label_idx = [0]

def fake_llm_call(messages):
    label = labels[label_idx[0] % len(labels)]
    label_idx[0] += 1
    msg = AIMessage(content=label)
    msg.tool_calls = []
    return msg

fake_llm = MagicMock(); fake_llm.invoke.side_effect = fake_llm_call

proc4 = AgentRecordProcessor(
    llm         = fake_llm,
    prompt_fn   = lambda p: f"Classify this review as POSITIVE/NEGATIVE/NEUTRAL: '{p['text']}'",
    response_fn = lambda r, p: {**p, "sentiment": r.strip()},
)

reviews = [
    {"id": 1, "text": "Amazing product, highly recommend!"},
    {"id": 2, "text": "Terrible quality, broke in one day."},
    {"id": 3, "text": "Best purchase I've ever made."},
    {"id": 4, "text": "It's okay, nothing special."},
    {"id": 5, "text": "Would not buy again."},
]
enriched_sink = []
report4 = (maestro.JobBuilder()
           .named("sentiment-analysis")
           .reader(maestro.IterableRecordReader(reviews))
           .processor(proc4)
           .writer(maestro.CollectionRecordWriter(enriched_sink))
           .build()).call()

print(f"  Processed: {report4.metrics.written_count} reviews")
for r in enriched_sink:
    print(f"    [{r['sentiment']:<8}] {r['text'][:45]}")


# ════════════════════════════════════════════════════════════════════════════
#  Pattern 5: Custom LangGraph with Maestro node factories
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("Pattern 5 — Custom LangGraph with Maestro node factories"); print(SEP)

# Reset FSM for a clean state
order_fsm2 = (
    maestro.FiniteStateMachineBuilder(states={pending, paid, shipped}, initial_state=pending)
    .register_transition(maestro.TransitionBuilder().source_state(pending).event_type(PayEvent).target_state(paid).build())
    .register_transition(maestro.TransitionBuilder().source_state(paid).event_type(ShipEvent).target_state(shipped).build())
    .build()
)

sink5 = []
batch_job5 = maestro.JobBuilder().reader(
    maestro.IterableRecordReader(["item-1", "item-2"])
).writer(maestro.CollectionRecordWriter(sink5)).build()

validate_schema5 = Schema(
    order_id = field(str, Required()),
    total    = field(float, Required()),
)

def decide_route(state):
    if state.get("error"):     return "error"
    if state.get("next_action") == "fix": return "error"
    return "process"

def error_node(state): return {"messages": [AIMessage(f"Error: {state.get('error')}")]}
def log_node(state):
    m = state.get("batch_metrics", {})
    print(f"    rules: {state.get('facts', {}).get('tier','?')}, "
          f"fsm: {state.get('fsm_state','?')}, "
          f"batch_written: {m.get('written_count','?')}")
    return {}

graph5 = StateGraph(MaestroAgentState)
graph5.add_node("validate", make_validator_node(validate_schema5, facts_key="facts"))
graph5.add_node("pricing",  make_rules_node(pricing_rules, facts_key="facts"))
graph5.add_node("pay",      make_fsm_node(order_fsm2, PayEvent))
graph5.add_node("batch",    make_batch_node(batch_job5))
graph5.add_node("observe",  make_observer_node(mem))
graph5.add_node("log",      log_node)
graph5.add_node("error",    error_node)

graph5.set_entry_point("validate")
graph5.add_conditional_edges("validate", decide_route, {"process": "pricing", "error": "error"})
graph5.add_edge("pricing", "pay")
graph5.add_edge("pay",     "batch")
graph5.add_edge("batch",   "observe")
graph5.add_edge("observe", "log")
graph5.add_edge("log",     END)
graph5.add_edge("error",   END)
compiled5 = graph5.compile()

print("  Valid order (ORD-3, VIP, $300):")
result5a = compiled5.invoke({
    "facts": {"order_id": "ORD-3", "total": 300.0, "tier": "vip"}
})

print()
print("  Invalid order (missing order_id):")
result5b = compiled5.invoke({
    "facts": {"total": 100.0}
})
print(f"    error: {result5b.get('error', 'none')}")

print(f"\n  Observability summary (agents + rules metrics):")
for line in mem.summary().split('\n')[:10]:
    if line.strip(): print(f"    {line.rstrip()}")


# ════════════════════════════════════════════════════════════════════════════
#  Pattern 6: MultiAgentOrchestratorGraph
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("Pattern 6 — MultiAgentOrchestratorGraph"); print(SEP)

calls6 = []
sup_responses = [
    '{"next": "billing"}',
    '{"next": "FINISH"}',
]
sup_idx = [0]

def fake_sup(messages, **kw):
    resp = AIMessage(content=sup_responses[sup_idx[0] % len(sup_responses)])
    resp.tool_calls = []
    sup_idx[0] += 1
    return resp

billing_responses = ["Pricing applied: 15% enterprise discount. Total $170."]
billing_idx = [0]
def fake_billing(messages, **kw):
    resp = AIMessage(content=billing_responses[billing_idx[0] % len(billing_responses)])
    resp.tool_calls = []
    billing_idx[0] += 1
    return resp

sup_llm6 = MagicMock(); sup_llm6.invoke.side_effect = fake_sup
billing_llm6 = MagicMock()
billing_llm6.invoke.side_effect = fake_billing
billing_llm6.bind_tools.return_value = billing_llm6

billing_tool6 = RulesEngineTool(rules=pricing_rules, name="pricing6",
                                description="Apply pricing rules.")

graph6 = MultiAgentOrchestratorGraph(
    supervisor_llm = sup_llm6,
    specialists = {
        "billing": (billing_llm6,
                    [billing_tool6],
                    "You are the billing specialist. Apply pricing rules."),
    },
    supervisor_prompt = (
        "Route requests to: 'billing' (pricing, discounts) or FINISH.\n"
        "Reply with JSON: {\"next\": \"billing\"} or {\"next\": \"FINISH\"}"
    ),
    max_rounds = 4,
).build()

result6 = graph6.invoke({
    "messages": [HumanMessage("Apply enterprise pricing for order $200.")],
    "facts":    {"total": 200.0, "tier": "enterprise"},
})
print(f"  Messages exchanged: {len(result6['messages'])}")
specialist_msgs = [m for m in result6["messages"] if hasattr(m, "content") and "discount" in m.content.lower()]
if specialist_msgs:
    print(f"  Specialist: {specialist_msgs[-1].content}")


# ════════════════════════════════════════════════════════════════════════════
#  Pattern 7: AgentCondition as a Maestro rule condition
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("Pattern 7 — AgentCondition as a rule condition"); print(SEP)

sentiment_calls = {"yes": 0, "no": 0}
def fake_sentiment(messages, **kw):
    text = messages[0].content if messages else ""
    is_positive = any(w in text.lower() for w in ("great", "love", "amazing"))
    answer = "yes" if is_positive else "no"
    sentiment_calls[answer] += 1
    msg = AIMessage(content=answer)
    msg.tool_calls = []
    return msg

sentiment_llm = MagicMock(); sentiment_llm.invoke.side_effect = fake_sentiment

sentiment_cond = AgentCondition(
    llm       = sentiment_llm,
    prompt_fn = lambda f: f"Is this review positive? Reply yes/no: '{f.get('text', '')}'",
)

highlight_rule = (maestro.RuleBuilder()
                  .name("highlight-positive")
                  .when(sentiment_cond)
                  .then(lambda f: f.put("action", "highlight"))
                  .build())

test_reviews = [
    maestro.Facts(text="This is an amazing product!", id=1),
    maestro.Facts(text="Broke after one use, terrible.", id=2),
    maestro.Facts(text="I love everything about it.", id=3),
]
rules7  = maestro.Rules(highlight_rule)
engine7 = maestro.DefaultRulesEngine()
for facts in test_reviews:
    engine7.fire(rules7, facts)
    action = facts.get("action", "none")
    print(f"  [{action:<12}] {facts.get('text')[:42]}")

print(f"\n  LLM calls: yes={sentiment_calls['yes']} no={sentiment_calls['no']}")

print(); print(SEP); print("All agentic pattern examples completed."); print(SEP)
print("""
To use with a real LLM:
  from langchain_anthropic import ChatAnthropic
  llm = ChatAnthropic(model="claude-sonnet-4-5")
  # Then pass llm= to any graph builder or AgentWork
""")
