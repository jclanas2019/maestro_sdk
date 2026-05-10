"""
examples/providers_example.py — Anthropic and OpenAI native provider adapters.

Demonstrates all provider features using mocked API calls.
To use real APIs, replace the mock client with the real one:

    Anthropic:
        llm = make_anthropic(model=AnthropicModels.HAIKU)
        # set ANTHROPIC_API_KEY env var

    OpenAI:
        llm = make_openai(model=OpenAIModels.GPT_4O_MINI)
        # set OPENAI_API_KEY env var

    With retry + observability:
        llm = make_anthropic().with_retry(max_attempts=3).with_observer(obs)

Patterns shown
--------------
1.  Message dataclass — building conversations
2.  AnthropicAdapter — text response
3.  AnthropicAdapter — tool calling loop
4.  OpenAIAdapter — text response
5.  OpenAIAdapter — tool calling loop
6.  with_retry() — rate limit resilience
7.  with_observer() — token usage metrics
8.  NativeReActAgent — full ReAct loop
9.  NativeAgentWork — agent inside a Maestro flow
10. NativeAgentBatchProcessor — LLM-powered ETL
11. Streaming — text chunk callbacks
12. Chained: Maestro flow → agent → FSM → event bus
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import maestro
from maestro.agents import (
    AnthropicAdapter, OpenAIAdapter, AnthropicModels, OpenAIModels,
    Message, LLMResponse, ToolCall, TokenUsage,
    RulesEngineTool, FSMTransitionTool, BatchJobTool, SagaTool, EventPublisherTool,
    NativeReActAgent, NativeAgentWork, NativeAgentBatchProcessor, AgentResult,
    make_anthropic, make_openai,
)
from maestro.agents._providers import _maestro_tool_to_anthropic, _maestro_tool_to_openai
from maestro.events import EventBus
from maestro.observe import InMemoryObserver, MaestroObserver
from maestro.saga import SagaBuilder, SagaStatus

SEP = "═" * 62


# ════════════════════════════════════════════════════════════════════════════
#  Mock clients — no API keys needed
# ════════════════════════════════════════════════════════════════════════════

from unittest.mock import MagicMock
from types import SimpleNamespace

def _anthropic_text_response(text, input_tokens=50, output_tokens=100):
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    block = MagicMock(); block.type = "text"; block.text = text
    resp.content = [block]
    return resp

def _anthropic_tool_response(tool_name, tool_id, arguments):
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.usage = MagicMock(input_tokens=40, output_tokens=30)
    tb = MagicMock(); tb.type = "tool_use"; tb.id = tool_id
    tb.name = tool_name; tb.input = arguments
    resp.content = [tb]
    return resp

def _openai_text_response(text, prompt_tokens=50, completion_tokens=100):
    resp = MagicMock()
    resp.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    choice = MagicMock(); choice.finish_reason = "stop"
    message = MagicMock(); message.content = text; message.tool_calls = None
    choice.message = message; resp.choices = [choice]
    return resp

def _openai_tool_response(tool_name, tool_id, arguments):
    resp = MagicMock()
    resp.usage = MagicMock(prompt_tokens=40, completion_tokens=30)
    choice = MagicMock(); choice.finish_reason = "tool_calls"
    fn  = SimpleNamespace(name=tool_name, arguments=json.dumps(arguments))
    otc = SimpleNamespace(id=tool_id, function=fn)
    msg = MagicMock(); msg.content = None; msg.tool_calls = [otc]
    choice.message = msg; resp.choices = [choice]
    return resp

def _make_anthropic_adapter(responses):
    """Anthropic adapter backed by a mock client."""
    adapter = AnthropicAdapter.__new__(AnthropicAdapter)
    import anthropic
    adapter._anthropic     = anthropic
    adapter._model         = AnthropicModels.HAIKU
    adapter._max_tokens    = 1024
    adapter._temperature   = 1.0
    adapter._timeout       = 30.0
    adapter._extra_headers = {}
    adapter._client        = MagicMock()
    adapter._client.messages.create.side_effect = responses
    return adapter

def _make_openai_adapter(responses):
    """OpenAI adapter backed by a mock client."""
    adapter = OpenAIAdapter.__new__(OpenAIAdapter)
    import openai
    adapter._openai          = openai
    adapter._model           = OpenAIModels.GPT_4O_MINI
    adapter._temperature     = 1.0
    adapter._max_tokens      = None
    adapter._timeout         = 30.0
    adapter._response_format = None
    adapter._client          = MagicMock()
    adapter._client.chat.completions.create.side_effect = responses
    return adapter


# ════════════════════════════════════════════════════════════════════════════
#  Shared Maestro setup
# ════════════════════════════════════════════════════════════════════════════

pricing_rules = maestro.Rules(
    maestro.RuleBuilder().name("vip")
    .when(lambda f: f.get("tier") == "vip")
    .then(lambda f: f.put("discount", 0.20))
    .build()
)
pricing_tool = RulesEngineTool(
    rules=pricing_rules, name="apply_pricing",
    description="Apply pricing and discount rules. Pass facts dict, get back updated facts."
)

pending, paid = maestro.State("PENDING"), maestro.State("PAID")
class PayEvent(maestro.Event): pass
order_fsm = (
    maestro.FiniteStateMachineBuilder(states={pending, paid}, initial_state=pending)
    .register_transition(maestro.TransitionBuilder().source_state(pending).event_type(PayEvent).target_state(paid).build())
    .build()
)
fsm_tool = FSMTransitionTool(
    fsm=order_fsm, event_map={"pay": PayEvent},
    name="update_order_state",
    description="Advance order FSM. Event: 'pay'."
)

bus = EventBus()
bus_log = []
bus.subscribe_fn("order.processed", lambda m: bus_log.append(m.payload))
event_tool = EventPublisherTool(bus=bus, name="publish_event",
                                description="Publish to event bus.")


# ════════════════════════════════════════════════════════════════════════════
#  1. Message dataclass
# ════════════════════════════════════════════════════════════════════════════
print(SEP); print("1. Message — building conversations"); print(SEP)

messages = [
    Message.system("You are an order processing agent."),
    Message.user("Apply VIP pricing for order ORD-1, total $200."),
    Message.assistant("I'll apply the pricing rules.", [
        ToolCall("tc1", "apply_pricing", {"facts": {"tier": "vip", "total": 200.0}})
    ]),
    Message.tool_result("tc1", '{"facts": {"tier": "vip", "total": 200.0, "discount": 0.2}}'),
    Message.assistant("Applied: 20% VIP discount. Final total: $160."),
]
for msg in messages:
    icon = {"system": "⚙", "user": "👤", "assistant": "🤖", "tool": "🔧"}.get(msg.role, "?")
    print(f"  {icon} [{msg.role}] {msg.content[:55]}{'...' if len(msg.content)>55 else ''}")
    if msg.tool_calls:
        for tc in msg.tool_calls: print(f"      → tool: {tc.name}({tc.arguments})")


# ════════════════════════════════════════════════════════════════════════════
#  2. AnthropicAdapter — simple text response
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("2. AnthropicAdapter — text response"); print(SEP)

llm_a = _make_anthropic_adapter([
    _anthropic_text_response("The capital of France is Paris.", input_tokens=12, output_tokens=8)
])
resp = llm_a.chat([Message.user("What is the capital of France?")])
print(f"  model:        {llm_a.model}")
print(f"  provider:     {llm_a.provider}")
print(f"  response:     {resp.content}")
print(f"  stop_reason:  {resp.stop_reason}")
print(f"  input tokens: {resp.usage.input_tokens}")
print(f"  output tokens:{resp.usage.output_tokens}")
print(f"  total tokens: {resp.usage.total_tokens}")


# ════════════════════════════════════════════════════════════════════════════
#  3. AnthropicAdapter — tool calling
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("3. AnthropicAdapter — tool calling"); print(SEP)

llm_a2 = _make_anthropic_adapter([
    _anthropic_tool_response("apply_pricing", "tu1", {"facts": {"tier": "vip", "total": 200.0}}),
    _anthropic_text_response("20% VIP discount applied. Final total: $160.")
])
agent_a = NativeReActAgent(
    llm=llm_a2, tools=[pricing_tool],
    system_prompt="You are a pricing agent. Use tools to apply discounts.",
    on_tool_call=lambda n, a: print(f"  → calling tool: {n}({list(a.keys())})"),
    on_tool_result=lambda n, r: print(f"  ← result: {r[:70]}"),
)
result_a = agent_a.run("Apply pricing for VIP customer, total $200.", context={"tier": "vip"})
print(f"\n  Final:       {result_a.final_response}")
print(f"  Tools used:  {result_a.tool_calls_made}")
print(f"  Iterations:  {result_a.iterations}")
print(f"  Total tokens:{result_a.usage.total_tokens}")


# ════════════════════════════════════════════════════════════════════════════
#  4. OpenAIAdapter — text response
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("4. OpenAIAdapter — text response"); print(SEP)

llm_o = _make_openai_adapter([
    _openai_text_response("Pi is approximately 3.14159.", prompt_tokens=10, completion_tokens=9)
])
resp_o = llm_o.chat([
    Message.system("Answer concisely."),
    Message.user("What is Pi?"),
])
print(f"  model:    {llm_o.model}")
print(f"  provider: {llm_o.provider}")
print(f"  response: {resp_o.content}")
print(f"  tokens:   {resp_o.usage.total_tokens}")


# ════════════════════════════════════════════════════════════════════════════
#  5. OpenAIAdapter — tool calling
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("5. OpenAIAdapter — tool calling loop"); print(SEP)

llm_o2 = _make_openai_adapter([
    _openai_tool_response("apply_pricing", "call_1", {"facts": {"tier": "vip", "total": 150.0}}),
    _openai_text_response("VIP pricing applied: 20% discount. Total: $120.")
])
agent_o = NativeReActAgent(
    llm=llm_o2, tools=[pricing_tool],
    system_prompt="You are a pricing agent.",
    on_tool_call=lambda n, a: print(f"  → tool: {n}"),
)
result_o = agent_o.run("Pricing for VIP, $150.")
print(f"\n  Response: {result_o.final_response}")
print(f"  Tools:    {result_o.tool_calls_made}")


# ════════════════════════════════════════════════════════════════════════════
#  6. with_retry() — rate limit resilience
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("6. with_retry() — resilience against transient errors"); print(SEP)

from maestro.agents._providers import _RetryingAdapter

call_count = [0]
base_llm = MagicMock()
type(base_llm).model    = property(lambda self: "test-model")
type(base_llm).provider = property(lambda self: "test")

def flaky_chat(messages, tools=None):
    call_count[0] += 1
    if call_count[0] < 3:
        raise ConnectionError(f"transient error #{call_count[0]}")
    return LLMResponse("Succeeded after retries.", usage=TokenUsage(10, 20))

base_llm.chat.side_effect = flaky_chat

retrying_llm = _RetryingAdapter(base_llm, max_attempts=5, on=[ConnectionError])
result_r = retrying_llm.chat([Message.user("Test retry.")])
print(f"  Attempts:  {call_count[0]} (2 failures + 1 success)")
print(f"  Response:  {result_r.content}")
print(f"  model:     {retrying_llm.model}")


# ════════════════════════════════════════════════════════════════════════════
#  7. with_observer() — token usage metrics
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("7. with_observer() — token usage metrics"); print(SEP)

from maestro.agents._providers import _ObservingAdapter

obs2 = InMemoryObserver()
base_llm2 = MagicMock()
type(base_llm2).model    = property(lambda self: "claude-sonnet-4-6")
type(base_llm2).provider = property(lambda self: "anthropic")
base_llm2.chat.return_value = LLMResponse(
    "Observed response.", usage=TokenUsage(input_tokens=42, output_tokens=88))

observed_llm = _ObservingAdapter(base_llm2, obs2)
observed_llm.chat([Message.user("Hello.")])
observed_llm.chat([Message.user("Hello again.")])

print(f"  LLM calls tracked:   {obs2.counter('agents', 'llm_call', model='claude-sonnet-4-6', provider='anthropic'):.0f}")
print(f"  Input tokens:        {obs2.gauge('agents', 'tokens_input', model='claude-sonnet-4-6', provider='anthropic'):.0f}")
print(f"  Output tokens:       {obs2.gauge('agents', 'tokens_output', model='claude-sonnet-4-6', provider='anthropic'):.0f}")
snap = obs2.histogram("agents", "llm_duration_seconds", model="claude-sonnet-4-6", provider="anthropic")
print(f"  Call durations:      {snap['count']} recorded, mean={snap['mean']*1000:.1f}ms")
print()
print(f"  Prometheus snippet:")
for line in obs2.export_prometheus().split('\n')[:6]:
    if line: print(f"    {line}")


# ════════════════════════════════════════════════════════════════════════════
#  8. NativeReActAgent — multi-tool loop
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("8. NativeReActAgent — multi-tool ReAct loop"); print(SEP)

llm_multi = _make_anthropic_adapter([
    _anthropic_tool_response("apply_pricing",    "tu1", {"facts": {"tier": "vip", "total": 300.0}}),
    _anthropic_tool_response("update_order_state","tu2", {"event_name": "pay"}),
    _anthropic_tool_response("publish_event",    "tu3",
                             {"topic": "order.processed", "payload": {"order_id": "ORD-7", "status": "paid"}}),
    _anthropic_text_response("Order ORD-7 processed: VIP discount, payment confirmed, event published."),
])

agent_multi = NativeReActAgent(
    llm           = llm_multi,
    tools         = [pricing_tool, fsm_tool, event_tool],
    system_prompt = ("You are an order processing agent. For each order:\n"
                     "1. Apply pricing rules\n2. Update order state to paid\n"
                     "3. Publish the completion event"),
    on_tool_call   = lambda n, a: print(f"  → {n}"),
    on_tool_result = lambda n, r: print(f"  ← {r[:60]}"),
)

result_multi = agent_multi.run(
    "Process order ORD-7: customer Alice, tier=vip, total=$300.",
    context={"order_id": "ORD-7", "customer": "Alice"}
)
print(f"\n  Final answer: {result_multi.final_response}")
print(f"  Tools called: {result_multi.tool_calls_made}")
print(f"  Iterations:   {result_multi.iterations}")
print(f"  FSM state:    {order_fsm.current_state}")
print(f"  Bus events:   {bus_log}")


# ════════════════════════════════════════════════════════════════════════════
#  9. NativeAgentWork — agent inside a Maestro SequentialFlow
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("9. NativeAgentWork — inside a Maestro SequentialFlow"); print(SEP)

llm_work = _make_openai_adapter([
    _openai_tool_response("apply_pricing", "c1", {"facts": {"tier": "vip", "total": 500.0}}),
    _openai_text_response("VIP order approved. Discount: 20%, final total: $400.")
])

order_agent_work = NativeAgentWork(
    llm           = llm_work,
    maestro_tools = [pricing_tool],
    system_prompt = "You are a pricing agent. Apply rules and confirm the order.",
    input_fn      = lambda ctx: {
        "query":   f"Process order {ctx.get('order_id')} for {ctx.get('customer')}",
        "context": {"tier": ctx.get("tier"), "total": ctx.get("total")},
    },
    output_fn = lambda r, ctx: (
        ctx.put("agent_answer",  r.final_response),
        ctx.put("tools_used",   r.tool_calls_made),
        ctx.put("agent_tokens", r.usage.total_tokens if r.usage else 0),
    ),
    name = "pricing-agent",
)

flow9 = (maestro.aNewSequentialFlow()
         .named("order-with-native-agent")
         .execute(maestro.LambdaWork(lambda c: c.put("validated", True), "validate"))
         .then(order_agent_work)
         .then(maestro.LambdaWork(
             lambda c: print(f"  Answer:  {c.get('agent_answer')}\n"
                             f"  Tools:   {c.get('tools_used')}\n"
                             f"  Tokens:  {c.get('agent_tokens')}"),
             "log"))
         .build())

ctx9 = maestro.WorkContext(order_id="ORD-9", customer="Bob", tier="vip", total=500.0)
r9   = maestro.WorkFlowEngine().run(flow9, ctx9)
print(f"  Flow status: {r9.status.value}")


# ════════════════════════════════════════════════════════════════════════════
#  10. NativeAgentBatchProcessor — LLM-powered ETL
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("10. NativeAgentBatchProcessor — batch ETL with LLM"); print(SEP)

sentiments = ["POSITIVE", "NEGATIVE", "POSITIVE", "NEUTRAL", "POSITIVE"]
s_idx = [0]

llm_batch = MagicMock()
type(llm_batch).model    = property(lambda self: "gpt-4o-mini")
type(llm_batch).provider = property(lambda self: "openai")
def fake_batch_chat(messages, tools=None):
    label = sentiments[s_idx[0] % len(sentiments)]
    s_idx[0] += 1
    return LLMResponse(label, usage=TokenUsage(8, 2))
llm_batch.chat.side_effect = fake_batch_chat

proc10 = NativeAgentBatchProcessor(
    llm           = llm_batch,
    system_prompt = "Classify the review as exactly one of: POSITIVE, NEGATIVE, NEUTRAL.",
    prompt_fn     = lambda p: f"Review: \"{p['text']}\"",
    response_fn   = lambda r, p: {**p, "sentiment": r.strip(), "llm_model": "gpt-4o-mini"},
)
reviews10 = [
    {"id": 1, "text": "Excellent product, exceeded expectations!"},
    {"id": 2, "text": "Broke after two days. Very disappointed."},
    {"id": 3, "text": "Solid build quality, great value for money."},
    {"id": 4, "text": "It's fine, does the job."},
    {"id": 5, "text": "Amazing — best purchase this year!"},
]
sink10 = []
report10 = (maestro.JobBuilder()
            .named("sentiment-analysis")
            .reader(maestro.IterableRecordReader(reviews10))
            .processor(proc10)
            .writer(maestro.CollectionRecordWriter(sink10))
            .build()).call()

print(f"  Processed: {report10.metrics.written_count} reviews")
for r in sink10:
    icon = {"POSITIVE": "✓", "NEGATIVE": "✗", "NEUTRAL": "~"}.get(r["sentiment"], "?")
    print(f"    {icon} [{r['sentiment']:<8}] {r['text'][:45]}")

total_tokens_10 = s_idx[0] * 10  # 8 input + 2 output per call
print(f"  Model: {sink10[0]['llm_model']}")
print(f"  ~{total_tokens_10} tokens used for {len(sink10)} records")


# ════════════════════════════════════════════════════════════════════════════
#  11. Streaming — text chunk callbacks
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("11. Streaming — text chunk callbacks"); print(SEP)

print("  Streaming response: ", end="", flush=True)
chunks = ["Once ", "upon ", "a ", "time ", "in ", "a ", "land ", "far ", "away..."]

llm_stream = _make_anthropic_adapter([
    _anthropic_text_response("Once upon a time in a land far away...")
])
# Simulate streaming by collecting chunks in on_text callback
streamed = []
def on_chunk(text): 
    streamed.append(text)
    print(text, end="", flush=True)

# Since we're mocked, the stream method falls back to chat
resp_stream = llm_stream.chat([Message.user("Tell me a story opening.")])
# Simulate chunk-by-chunk via callback
for chunk in chunks: on_chunk(chunk)
print()
print(f"\n  Chunks received:  {len(streamed)}")
print(f"  Content preview:  {resp_stream.content[:40]}")


# ════════════════════════════════════════════════════════════════════════════
#  12. Full pipeline: flow → native agent → FSM → event bus
# ════════════════════════════════════════════════════════════════════════════
print(); print(SEP); print("12. Full pipeline — flow + native agent + FSM + bus"); print(SEP)

order_fsm12 = (
    maestro.FiniteStateMachineBuilder(states={pending, paid}, initial_state=pending)
    .register_transition(maestro.TransitionBuilder().source_state(pending).event_type(PayEvent).target_state(paid).build())
    .build()
)
bus12  = EventBus()
bus12_log = []
bus12.subscribe_fn("order.complete", lambda m: bus12_log.append(m.payload))

fsm_tool12 = FSMTransitionTool(fsm=order_fsm12, event_map={"pay": PayEvent}, name="order_state")
event_tool12 = EventPublisherTool(bus=bus12, name="notify")

llm12 = _make_anthropic_adapter([
    _anthropic_tool_response("apply_pricing",  "t1", {"facts": {"tier": "vip", "total": 199.0}}),
    _anthropic_tool_response("order_state",     "t2", {"event_name": "pay"}),
    _anthropic_tool_response("notify",          "t3",
                             {"topic": "order.complete", "payload": {"order": "ORD-12", "ok": True}}),
    _anthropic_text_response("Order ORD-12: VIP discount applied, payment confirmed."),
])

work12 = NativeAgentWork(
    llm           = llm12,
    maestro_tools = [pricing_tool, fsm_tool12, event_tool12],
    system_prompt = "Process orders: price → pay → notify.",
    input_fn      = lambda c: {"query": f"Process {c.get('order_id')}", "context": c.as_map()},
    name          = "pipeline-agent",
)
flow12 = maestro.aNewSequentialFlow().named("full-pipeline").execute(work12).build()
maestro.WorkFlowEngine().run(flow12, maestro.WorkContext(
    order_id="ORD-12", tier="vip", total=199.0))

print(f"  FSM state:  {order_fsm12.current_state}")
print(f"  Bus events: {bus12_log}")
print(f"  Flow: completed")

print(); print(SEP); print("All provider examples completed."); print(SEP)
print("""
To use real APIs:
  from maestro.agents import make_anthropic, make_openai, AnthropicModels, OpenAIModels

  # Anthropic (set ANTHROPIC_API_KEY)
  llm = make_anthropic(model=AnthropicModels.HAIKU)
  llm = make_anthropic().with_retry().with_observer(InMemoryObserver())

  # OpenAI (set OPENAI_API_KEY)
  llm = make_openai(model=OpenAIModels.GPT_4O_MINI)

  # Any OpenAI-compatible endpoint (Ollama, Azure, Together AI…)
  llm = make_openai(base_url="http://localhost:11434/v1", model="llama3")
""")
