"""
tests/test_providers.py — AnthropicAdapter, OpenAIAdapter, NativeReActAgent tests.

All API calls are mocked. No real API key required.
"""
import sys, os, json, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

import maestro
from maestro.agents import (
    AnthropicAdapter, OpenAIAdapter, AnthropicModels, OpenAIModels,
    Message, LLMResponse, ToolCall, TokenUsage,
    RulesEngineTool, BatchJobTool, SagaTool,
    NativeReActAgent, NativeAgentWork, NativeAgentBatchProcessor, AgentResult,
    make_anthropic, make_openai,
)
from maestro.agents._providers import (
    _maestro_tool_to_anthropic, _maestro_tool_to_openai, _execute_tool,
)
if False: (
)
from maestro.agents._providers import _execute_tool


# ════════════════════════════════════════════════════════════════════════════
#  Test fixtures
# ════════════════════════════════════════════════════════════════════════════

def _pricing_rules():
    r = (maestro.RuleBuilder()
         .name("vip").when(lambda f: f.get("tier") == "vip")
         .then(lambda f: f.put("discount", 0.20))
         .build())
    return maestro.Rules(r)

def _pricing_tool():
    return RulesEngineTool(
        rules=_pricing_rules(), name="apply_pricing",
        description="Apply pricing rules."
    )

def _batch_tool():
    sink = []
    job = maestro.JobBuilder().reader(
        maestro.IterableRecordReader([1, 2])
    ).writer(maestro.CollectionRecordWriter(sink)).build()
    return BatchJobTool(job=job, name="run_etl")

def _make_anthropic_response(text="Done.", tool_calls=None, input_tokens=10, output_tokens=20):
    """Build a mock Anthropic API response."""
    response = MagicMock()
    response.stop_reason = "end_turn" if not tool_calls else "tool_use"
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    blocks = []
    if text:
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = text
        blocks.append(text_block)
    if tool_calls:
        for tc in tool_calls:
            tb = MagicMock()
            tb.type = "tool_use"
            tb.id   = tc.get("id", "tu_001")
            tb.name = tc["name"]
            tb.input = tc["arguments"]
            blocks.append(tb)
        response.stop_reason = "tool_use"
    response.content = blocks
    return response

def _make_openai_response(text="Done.", tool_calls=None, prompt_tokens=10, completion_tokens=20):
    """Build a mock OpenAI API response."""
    response = MagicMock()
    response.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    choice  = MagicMock()
    message = MagicMock()
    message.content    = text
    message.tool_calls = None
    choice.finish_reason = "stop"
    choice.message = message
    if tool_calls:
        oai_tcs = []
        for tc in tool_calls:
            from types import SimpleNamespace as _SN
            otc = _SN(
                id       = tc.get("id", "call_001"),
                function = _SN(name=tc["name"], arguments=json.dumps(tc["arguments"])),
            )
            oai_tcs.append(otc)
        message.tool_calls = oai_tcs
        choice.finish_reason = "tool_calls"
    response.choices = [choice]
    return response


# ════════════════════════════════════════════════════════════════════════════
#  Tool schema conversion
# ════════════════════════════════════════════════════════════════════════════

class TestToolSchemaConversion:
    def test_anthropic_schema(self):
        schema = _maestro_tool_to_anthropic(_pricing_tool())
        assert schema["name"] == "apply_pricing"
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"
        assert "facts" in schema["input_schema"]["properties"]
        assert "title" not in schema["input_schema"]   # cleaned up

    def test_openai_schema(self):
        schema = _maestro_tool_to_openai(_pricing_tool())
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "apply_pricing"
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"

    def test_execute_tool_calls_tool(self):
        tool = _pricing_tool()
        result = _execute_tool(tool, {"facts": {"tier": "vip", "total": 200.0}})
        parsed = json.loads(result)
        assert parsed["facts"]["discount"] == 0.20

    def test_execute_tool_handles_error(self):
        tool = MagicMock()
        tool._run.side_effect = RuntimeError("tool broken")
        result = _execute_tool(tool, {})
        assert "error" in json.loads(result)


# ════════════════════════════════════════════════════════════════════════════
#  Message dataclass
# ════════════════════════════════════════════════════════════════════════════

class TestMessage:
    def test_factory_methods(self):
        assert Message.user("hi").role == "user"
        assert Message.system("sys").role == "system"
        assert Message.assistant("ok").role == "assistant"
        assert Message.tool_result("id1", "result").role == "tool"

    def test_tool_result_has_id(self):
        msg = Message.tool_result("tc_001", "{'price': 10}")
        assert msg.tool_call_id == "tc_001"
        assert msg.content == "{'price': 10}"

    def test_assistant_with_tool_calls(self):
        tcs  = [ToolCall("tc1", "my_tool", {"x": 1})]
        msg  = Message.assistant("I'll call the tool.", tcs)
        assert msg.tool_calls == tcs


# ════════════════════════════════════════════════════════════════════════════
#  AnthropicAdapter (mocked)
# ════════════════════════════════════════════════════════════════════════════

class TestAnthropicAdapter:
    def _adapter(self, api_responses):
        adapter = AnthropicAdapter.__new__(AnthropicAdapter)
        import anthropic as _anthropic
        adapter._anthropic    = _anthropic
        adapter._model        = AnthropicModels.HAIKU
        adapter._max_tokens   = 1024
        adapter._temperature  = 1.0
        adapter._timeout      = 30.0
        adapter._extra_headers = {}
        adapter._client       = MagicMock()
        adapter._client.messages.create.side_effect = api_responses
        return adapter

    def test_simple_text_response(self):
        responses = [_make_anthropic_response("The answer is 42.")]
        adapter   = self._adapter(responses)
        result    = adapter.chat([Message.user("What is 6×7?")])
        assert result.content == "The answer is 42."
        assert result.has_tool_calls is False
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20

    def test_tool_use_response(self):
        responses = [_make_anthropic_response(
            text="",
            tool_calls=[{"id": "tu1", "name": "apply_pricing", "arguments": {"facts": {"tier": "vip"}}}],
        )]
        adapter = self._adapter(responses)
        result  = adapter.chat([Message.user("Apply pricing.")], tools=[_pricing_tool()])
        assert result.has_tool_calls is True
        assert result.tool_calls[0].name == "apply_pricing"
        assert result.tool_calls[0].arguments == {"facts": {"tier": "vip"}}

    def test_system_message_extracted(self):
        responses  = [_make_anthropic_response("OK")]
        adapter    = self._adapter(responses)
        messages   = [Message.system("You are helpful."), Message.user("Hi")]
        adapter.chat(messages)
        call_kwargs = adapter._client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful."
        assert all(m["role"] != "system" for m in call_kwargs["messages"])

    def test_tool_result_in_user_content(self):
        responses = [_make_anthropic_response("Done with tool result.")]
        adapter   = self._adapter(responses)
        messages  = [
            Message.user("Call the tool."),
            Message.assistant("", [ToolCall("tu1", "pricing", {})]),
            Message.tool_result("tu1", '{"price": 10}'),
        ]
        adapter.chat(messages)
        call_kwargs = adapter._client.messages.create.call_args[1]
        user_msgs = [m for m in call_kwargs["messages"] if m["role"] == "user"]
        # Tool result should be in a user message as a content block
        assert any(
            isinstance(m["content"], list) and m["content"][0]["type"] == "tool_result"
            for m in user_msgs
        )

    def test_stop_reason_parsed(self):
        responses = [_make_anthropic_response("Done")]
        adapter   = self._adapter(responses)
        result    = adapter.chat([Message.user("x")])
        assert result.stop_reason == "end_turn"

    def test_provider_and_model_properties(self):
        adapter = self._adapter([_make_anthropic_response("x")])
        assert adapter.provider == "anthropic"
        assert adapter.model    == AnthropicModels.HAIKU

    def test_async_chat(self):
        responses = [_make_anthropic_response("Async result")]
        adapter   = self._adapter(responses)
        result    = asyncio.run(adapter.achat([Message.user("test")]))
        assert result.content == "Async result"


# ════════════════════════════════════════════════════════════════════════════
#  OpenAIAdapter (mocked)
# ════════════════════════════════════════════════════════════════════════════

class TestOpenAIAdapter:
    def _adapter(self, api_responses):
        adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        import openai as _openai
        adapter._openai          = _openai
        adapter._model           = OpenAIModels.GPT_4O_MINI
        adapter._temperature     = 1.0
        adapter._max_tokens      = None
        adapter._timeout         = 30.0
        adapter._response_format = None
        adapter._client          = MagicMock()
        adapter._client.chat.completions.create.side_effect = api_responses
        return adapter

    def test_simple_text_response(self):
        responses = [_make_openai_response("The answer is 42.")]
        adapter   = self._adapter(responses)
        result    = adapter.chat([Message.user("What is 6×7?")])
        assert result.content == "The answer is 42."
        assert result.has_tool_calls is False
        assert result.usage.input_tokens  == 10
        assert result.usage.output_tokens == 20

    def test_tool_call_response(self):
        responses = [_make_openai_response(
            text=None,
            tool_calls=[{"id": "call_001", "name": "apply_pricing", "arguments": {"facts": {}}}],
        )]
        adapter = self._adapter(responses)
        result  = adapter.chat([Message.user("Apply pricing.")], tools=[_pricing_tool()])
        assert result.has_tool_calls is True
        assert result.tool_calls[0].name == "apply_pricing"

    def test_system_message_in_messages(self):
        responses  = [_make_openai_response("OK")]
        adapter    = self._adapter(responses)
        messages   = [Message.system("Be helpful."), Message.user("Hi")]
        adapter.chat(messages)
        call_msgs = adapter._client.chat.completions.create.call_args[1]["messages"]
        assert call_msgs[0]["role"] == "system"
        assert call_msgs[0]["content"] == "Be helpful."

    def test_tool_result_format(self):
        responses = [_make_openai_response("Result processed.")]
        adapter   = self._adapter(responses)
        messages  = [
            Message.user("Use tool."),
            Message.assistant("", [ToolCall("call_1", "tool", {})]),
            Message.tool_result("call_1", "result data"),
        ]
        adapter.chat(messages)
        call_msgs = adapter._client.chat.completions.create.call_args[1]["messages"]
        tool_msg  = next((m for m in call_msgs if m["role"] == "tool"), None)
        assert tool_msg is not None
        assert tool_msg["tool_call_id"] == "call_1"

    def test_finish_reason_mapped(self):
        responses = [_make_openai_response("Done")]
        adapter   = self._adapter(responses)
        result    = adapter.chat([Message.user("x")])
        assert result.stop_reason == "end_turn"

    def test_provider_and_model_properties(self):
        adapter = self._adapter([_make_openai_response("x")])
        assert adapter.provider == "openai"
        assert adapter.model    == OpenAIModels.GPT_4O_MINI


# ════════════════════════════════════════════════════════════════════════════
#  with_retry / with_observer wrappers
# ════════════════════════════════════════════════════════════════════════════

class TestAdapterWrappers:
    def _base_adapter(self, response):
        base = MagicMock(spec=["chat", "achat", "model", "provider", "with_retry",
                               "with_observer", "_RetryingAdapter", "_ObservingAdapter"])
        base.chat.return_value = response
        type(base).model    = PropertyMock(return_value="test-model")
        type(base).provider = PropertyMock(return_value="test-provider")
        return base

    def test_observing_adapter_emits_metrics(self):
        from maestro.observe import InMemoryObserver
        from maestro.agents._providers import _ObservingAdapter

        response = LLMResponse("Hello", usage=TokenUsage(10, 20))
        base     = self._base_adapter(response)
        obs      = InMemoryObserver()
        adapted  = _ObservingAdapter(base, obs)
        result   = adapted.chat([Message.user("test")])

        assert result.content == "Hello"
        assert obs.counter("agents", "llm_call", model="test-model", provider="test-provider") == 1
        assert obs.gauge("agents", "tokens_input",  model="test-model", provider="test-provider") == 10
        assert obs.gauge("agents", "tokens_output", model="test-model", provider="test-provider") == 20

    def test_retrying_adapter_delegates(self):
        from maestro.agents._providers import _RetryingAdapter
        response = LLMResponse("ok")
        base     = self._base_adapter(response)
        adapted  = _RetryingAdapter(base, max_attempts=3)
        result   = adapted.chat([Message.user("x")])
        assert result.content == "ok"

    def test_retrying_adapter_retries_on_error(self):
        from maestro.agents._providers import _RetryingAdapter
        base       = MagicMock()
        call_count = [0]
        def chat(*a, **kw):
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("transient")
            return LLMResponse("finally ok")
        type(base).model    = PropertyMock(return_value="m")
        type(base).provider = PropertyMock(return_value="p")
        base.chat.side_effect = chat
        adapted = _RetryingAdapter(base, max_attempts=5, on=[ConnectionError])
        result  = adapted.chat([Message.user("x")])
        assert result.content == "finally ok"
        assert call_count[0] == 3

    def test_chaining_with_retry_and_observer(self):
        """llm.with_retry().with_observer() — full chain works."""
        from maestro.agents._providers import _ObservingAdapter, _RetryingAdapter
        from maestro.observe import InMemoryObserver

        response = LLMResponse("chained", usage=TokenUsage(5, 10))
        base     = MagicMock()
        base.chat.return_value = response
        type(base).model    = PropertyMock(return_value="m")
        type(base).provider = PropertyMock(return_value="p")

        obs     = InMemoryObserver()
        retry   = _RetryingAdapter(base, max_attempts=2)
        adapted = _ObservingAdapter(retry, obs)
        result  = adapted.chat([Message.user("test")])
        assert result.content == "chained"
        assert obs.counter("agents", "llm_call", model="m", provider="p") == 1


# ════════════════════════════════════════════════════════════════════════════
#  NativeReActAgent
# ════════════════════════════════════════════════════════════════════════════

class TestNativeReActAgent:
    def _llm(self, responses: list[LLMResponse]):
        llm = MagicMock()
        llm.chat.side_effect = responses
        return llm

    def test_single_turn_no_tools(self):
        llm   = self._llm([LLMResponse("The sky is blue.", usage=TokenUsage(5, 10))])
        agent = NativeReActAgent(llm=llm)
        result = agent.run("What color is the sky?")
        assert result.final_response == "The sky is blue."
        assert result.iterations == 1
        assert result.tool_calls_made == []

    def test_tool_call_and_follow_up(self):
        tool_call = ToolCall("tc1", "apply_pricing", {"facts": {"tier": "vip"}})
        responses = [
            LLMResponse("", tool_calls=[tool_call], stop_reason="tool_use",
                        usage=TokenUsage(10, 5)),
            LLMResponse("Pricing applied: 20% discount.", usage=TokenUsage(15, 8)),
        ]
        llm   = self._llm(responses)
        tool  = _pricing_tool()
        agent = NativeReActAgent(llm=llm, tools=[tool])
        result = agent.run("Apply pricing for VIP customer.")
        assert result.final_response == "Pricing applied: 20% discount."
        assert "apply_pricing" in result.tool_calls_made
        assert result.iterations == 2

    def test_system_prompt_included(self):
        llm = self._llm([LLMResponse("OK", usage=TokenUsage(5, 5))])
        NativeReActAgent(llm=llm, system_prompt="You are an expert.").run("Hello")
        call_msgs = llm.chat.call_args[0][0]
        assert call_msgs[0].role == "system"
        assert "expert" in call_msgs[0].content

    def test_context_prepended_to_query(self):
        llm = self._llm([LLMResponse("OK", usage=TokenUsage(5, 5))])
        NativeReActAgent(llm=llm).run("Process this.", context={"order_id": "ORD-1"})
        call_msgs = llm.chat.call_args[0][0]
        user_msg  = next(m for m in call_msgs if m.role == "user")
        assert "ORD-1" in user_msg.content

    def test_max_iterations_safety(self):
        """Agent should stop after max_iterations even with infinite tool calls."""
        tool_call = ToolCall("tc1", "apply_pricing", {"facts": {}})
        infinite  = LLMResponse("", tool_calls=[tool_call], stop_reason="tool_use",
                                usage=TokenUsage(5, 5))
        llm       = self._llm([infinite] * 20)
        tool      = _pricing_tool()
        agent     = NativeReActAgent(llm=llm, tools=[tool], max_iterations=3)
        result    = agent.run("Run forever.")
        assert result.iterations == 3

    def test_token_usage_aggregated(self):
        tool_call = ToolCall("tc1", "apply_pricing", {"facts": {}})
        responses = [
            LLMResponse("", tool_calls=[tool_call], usage=TokenUsage(10, 5)),
            LLMResponse("Done.",                    usage=TokenUsage(15, 8)),
        ]
        llm    = self._llm(responses)
        agent  = NativeReActAgent(llm=llm, tools=[_pricing_tool()])
        result = agent.run("x")
        assert result.usage.input_tokens  == 25
        assert result.usage.output_tokens == 13

    def test_on_tool_call_callback(self):
        tool_call = ToolCall("tc1", "apply_pricing", {"facts": {}})
        log = []
        responses = [
            LLMResponse("", tool_calls=[tool_call], usage=TokenUsage(5, 5)),
            LLMResponse("Done.", usage=TokenUsage(5, 5)),
        ]
        llm    = self._llm(responses)
        agent  = NativeReActAgent(
            llm=llm, tools=[_pricing_tool()],
            on_tool_call=lambda name, args: log.append(("call", name)),
            on_tool_result=lambda name, res: log.append(("result", name)),
        )
        agent.run("x")
        assert ("call", "apply_pricing")   in log
        assert ("result", "apply_pricing") in log

    def test_unknown_tool_returns_error(self):
        """Agent gracefully handles a tool call for a non-registered tool."""
        bad_tc    = ToolCall("tc1", "nonexistent_tool", {})
        responses = [
            LLMResponse("", tool_calls=[bad_tc], usage=TokenUsage(5, 5)),
            LLMResponse("Tool not found.", usage=TokenUsage(5, 5)),
        ]
        llm    = self._llm(responses)
        agent  = NativeReActAgent(llm=llm, tools=[])
        result = agent.run("x")
        assert result.final_response == "Tool not found."

    def test_multiple_tools_in_one_turn(self):
        """Agent can call multiple tools in a single turn."""
        tcs = [
            ToolCall("tc1", "apply_pricing", {"facts": {"tier": "vip"}}),
            ToolCall("tc2", "run_etl", {}),
        ]
        responses = [
            LLMResponse("", tool_calls=tcs, usage=TokenUsage(10, 5)),
            LLMResponse("All tools done.", usage=TokenUsage(10, 8)),
        ]
        llm    = self._llm(responses)
        agent  = NativeReActAgent(llm=llm, tools=[_pricing_tool(), _batch_tool()])
        result = agent.run("Run everything.")
        assert "apply_pricing" in result.tool_calls_made
        assert "run_etl"       in result.tool_calls_made


# ════════════════════════════════════════════════════════════════════════════
#  NativeAgentWork
# ════════════════════════════════════════════════════════════════════════════

class TestNativeAgentWork:
    def _llm(self, final_text="Agent done."):
        llm = MagicMock()
        llm.chat.return_value = LLMResponse(final_text, usage=TokenUsage(10, 20))
        return llm

    def test_completed_when_agent_succeeds(self):
        work   = NativeAgentWork(llm=self._llm(), name="test-work")
        report = work.execute(maestro.WorkContext(query="Do something."))
        assert report.status == maestro.WorkStatus.COMPLETED

    def test_output_fn_writes_to_context(self):
        ctx  = maestro.WorkContext(query="x")
        work = NativeAgentWork(
            llm       = self._llm("The result is 42."),
            output_fn = lambda r, c: c.put("answer", r.final_response),
        )
        work.execute(ctx)
        assert ctx.get("answer") == "The result is 42."

    def test_default_output_fn_writes_standard_keys(self):
        ctx  = maestro.WorkContext(query="x")
        work = NativeAgentWork(llm=self._llm())
        work.execute(ctx)
        assert ctx.contains("agent_response")
        assert ctx.contains("agent_iterations")
        assert ctx.contains("agent_tokens")

    def test_custom_input_fn_string(self):
        captured = []
        llm = MagicMock()
        llm.chat.side_effect = lambda msgs, tools=None: (
            captured.append(msgs[-1].content) or LLMResponse("ok", usage=TokenUsage(5,5))
        )
        ctx  = maestro.WorkContext(order_id="ORD-99", total=200.0)
        work = NativeAgentWork(
            llm      = llm,
            input_fn = lambda c: f"Process order {c.get('order_id')} total={c.get('total')}",
        )
        work.execute(ctx)
        assert "ORD-99" in captured[-1]

    def test_custom_input_fn_dict(self):
        captured_context = []
        llm = MagicMock()
        def fake_chat(msgs, tools=None):
            captured_context.append(msgs)
            return LLMResponse("ok", usage=TokenUsage(5, 5))
        llm.chat.side_effect = fake_chat
        ctx  = maestro.WorkContext(x=1)
        work = NativeAgentWork(
            llm      = llm,
            input_fn = lambda c: {"query": "Do it", "context": {"x": c.get("x")}},
        )
        work.execute(ctx)
        user_msg = next(m for m in captured_context[-1] if m.role == "user")
        assert '"x"' in user_msg.content or "1" in user_msg.content

    def test_failed_when_llm_raises(self):
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("API down")
        work   = NativeAgentWork(llm=llm, name="failing")
        report = work.execute(maestro.WorkContext(query="x"))
        assert report.status == maestro.WorkStatus.FAILED
        assert isinstance(report.error, RuntimeError)

    def test_inside_sequential_flow(self):
        log  = []
        work = NativeAgentWork(
            llm       = self._llm("Flow step complete."),
            output_fn = lambda r, c: log.append(r.final_response),
            name      = "flow-agent",
        )
        flow = maestro.SequentialFlow.Builder().execute(work).build()
        r    = maestro.WorkFlowEngine().run(flow, maestro.WorkContext(query="Go!"))
        assert r.status == maestro.WorkStatus.COMPLETED
        assert log == ["Flow step complete."]

    def test_with_maestro_tools(self):
        """Tools are passed to the ReAct agent."""
        tool_call = ToolCall("tc1", "apply_pricing", {"facts": {"tier": "vip"}})
        llm = MagicMock()
        llm.chat.side_effect = [
            LLMResponse("", tool_calls=[tool_call], usage=TokenUsage(5, 5)),
            LLMResponse("Pricing applied.", usage=TokenUsage(5, 5)),
        ]
        work   = NativeAgentWork(
            llm           = llm,
            maestro_tools = [_pricing_tool()],
            input_fn      = lambda c: "Apply pricing.",
        )
        report = work.execute(maestro.WorkContext())
        assert report.status == maestro.WorkStatus.COMPLETED
        assert llm.chat.call_count == 2


# ════════════════════════════════════════════════════════════════════════════
#  NativeAgentBatchProcessor
# ════════════════════════════════════════════════════════════════════════════

class TestNativeAgentBatchProcessor:
    def _llm(self, response_text):
        llm = MagicMock()
        llm.chat.return_value = LLMResponse(response_text, usage=TokenUsage(5, 5))
        return llm

    def test_enriches_record(self):
        proc = NativeAgentBatchProcessor(
            llm       = self._llm("POSITIVE"),
            prompt_fn = lambda p: f"Classify: {p.get('text','')}",
            response_fn= lambda r, p: {**p, "sentiment": r.strip()},
        )
        from maestro.batch._record import Header, Record
        rec = Record(Header(1, "test"), {"text": "great!"})
        result = proc.process_record(rec)
        assert result.payload["sentiment"] == "POSITIVE"

    def test_system_prompt_included(self):
        captured = []
        llm = MagicMock()
        def fake_chat(msgs, tools=None):
            captured.extend(msgs)
            return LLMResponse("OK", usage=TokenUsage(5, 5))
        llm.chat.side_effect = fake_chat

        proc = NativeAgentBatchProcessor(
            llm           = llm,
            system_prompt = "You are a sentiment classifier.",
            prompt_fn     = lambda p: str(p),
        )
        from maestro.batch._record import Header, Record
        proc.process_record(Record(Header(1, "t"), "text"))
        assert any(m.role == "system" and "sentiment" in m.content for m in captured)

    def test_fail_on_error_raises(self):
        llm = MagicMock()
        llm.chat.side_effect = RuntimeError("API error")
        proc = NativeAgentBatchProcessor(
            llm=llm, prompt_fn=lambda p: str(p), fail_on_error=True
        )
        from maestro.batch._record import Header, Record
        from maestro.batch._processor import RecordProcessingException
        with pytest.raises(RecordProcessingException):
            proc.process_record(Record(Header(1, "t"), {"x": 1}))

    def test_in_batch_pipeline(self):
        reviews = [{"text": "great"}, {"text": "bad"}, {"text": "ok"}]
        labels  = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
        call_n  = [0]
        llm     = MagicMock()
        def fake_chat(msgs, tools=None):
            label = labels[call_n[0] % len(labels)]
            call_n[0] += 1
            return LLMResponse(label, usage=TokenUsage(5, 5))
        llm.chat.side_effect = fake_chat

        proc = NativeAgentBatchProcessor(
            llm=llm,
            prompt_fn=lambda p: f"Classify: {p['text']}",
            response_fn=lambda r, p: {**p, "label": r.strip()},
        )
        sink = []
        (maestro.JobBuilder()
         .reader(maestro.IterableRecordReader(reviews))
         .processor(proc)
         .writer(maestro.CollectionRecordWriter(sink))
         .build()).call()

        assert len(sink) == 3
        assert {r["label"] for r in sink} == {"POSITIVE", "NEGATIVE", "NEUTRAL"}


# ════════════════════════════════════════════════════════════════════════════
#  Factory functions
# ════════════════════════════════════════════════════════════════════════════

class TestFactoryFunctions:
    def test_make_anthropic_creates_adapter(self):
        with patch("anthropic.Anthropic"):
            adapter = make_anthropic(model=AnthropicModels.HAIKU)
            assert isinstance(adapter, AnthropicAdapter)
            assert adapter.model == AnthropicModels.HAIKU
            assert adapter.provider == "anthropic"

    def test_make_openai_creates_adapter(self):
        with patch("openai.OpenAI"):
            adapter = make_openai(model=OpenAIModels.GPT_4O_MINI)
            assert isinstance(adapter, OpenAIAdapter)
            assert adapter.model == OpenAIModels.GPT_4O_MINI
            assert adapter.provider == "openai"

    def test_make_openai_with_base_url(self):
        """Supports custom base_url for Ollama or Azure."""
        with patch("openai.OpenAI"):
            adapter = make_openai(base_url="http://localhost:11434/v1")
            assert isinstance(adapter, OpenAIAdapter)

    def test_model_constants(self):
        assert AnthropicModels.SONNET  == "claude-sonnet-4-6"
        assert AnthropicModels.HAIKU   == "claude-haiku-4-5-20251001"
        assert OpenAIModels.GPT_4O     == "gpt-4o"
        assert OpenAIModels.GPT_4O_MINI == "gpt-4o-mini"


# ════════════════════════════════════════════════════════════════════════════
#  AgentResult
# ════════════════════════════════════════════════════════════════════════════

class TestAgentResult:
    def test_succeeded_property(self):
        r = AgentResult(final_response="done")
        assert r.succeeded is True

    def test_failed_when_empty(self):
        r = AgentResult(final_response="")
        assert r.succeeded is False

    def test_str_repr(self):
        r = AgentResult(
            final_response="The answer is 42.",
            tool_calls_made=["apply_pricing"],
            iterations=2,
            usage=TokenUsage(10, 20),
        )
        text = str(r)
        assert "42" in text
        assert "apply_pricing" in text
        assert "30" in text   # total tokens
