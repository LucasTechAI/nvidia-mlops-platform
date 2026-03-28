"""Tests for the ReAct agent module."""

from src.agent.react_agent import SYSTEM_PROMPT, ReActAgent

# ---------------------------------------------------------------------------
# Tests — initialization
# ---------------------------------------------------------------------------


class TestReActAgentInit:
    def test_default_init(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        agent = ReActAgent()
        assert agent.llm_provider == "openai"
        assert agent.model_name == "gpt-4o-mini"
        assert len(agent.tools) == 4
        assert agent.temperature == 0.1
        assert agent.max_iterations == 8

    def test_custom_init(self):
        agent = ReActAgent(
            model_name="test-model",
            temperature=0.5,
            max_iterations=3,
            llm_provider="groq",
        )
        assert agent.model_name == "test-model"
        assert agent.temperature == 0.5
        assert agent.max_iterations == 3
        assert agent.llm_provider == "groq"

    def test_tool_descriptions_generated(self):
        agent = ReActAgent()
        assert "query_stock_data" in agent.tool_descriptions
        assert "predict_stock_prices" in agent.tool_descriptions


# ---------------------------------------------------------------------------
# Tests — _parse_action
# ---------------------------------------------------------------------------


class TestParseAction:
    def test_parse_valid_action(self):
        agent = ReActAgent()
        response = "Thought: I need to check stock data\nAction: query_stock_data\nAction Input: latest prices"
        action, action_input = agent._parse_action(response)
        assert action == "query_stock_data"
        assert action_input == "latest prices"

    def test_parse_no_action(self):
        agent = ReActAgent()
        response = "I think the answer is 42."
        action, action_input = agent._parse_action(response)
        assert action is None
        assert action_input is None

    def test_parse_action_without_input(self):
        agent = ReActAgent()
        response = "Action: get_model_metrics\n"
        action, action_input = agent._parse_action(response)
        assert action == "get_model_metrics"


# ---------------------------------------------------------------------------
# Tests — _parse_final_answer
# ---------------------------------------------------------------------------


class TestParseFinalAnswer:
    def test_parse_valid_final_answer(self):
        agent = ReActAgent()
        response = "Thought: I have the info.\nFinal Answer: NVIDIA closed at $500."
        answer = agent._parse_final_answer(response)
        assert answer == "NVIDIA closed at $500."

    def test_parse_no_final_answer(self):
        agent = ReActAgent()
        response = "Thought: I need more info.\nAction: query_stock_data\nAction Input: latest"
        answer = agent._parse_final_answer(response)
        assert answer is None

    def test_parse_multiline_final_answer(self):
        agent = ReActAgent()
        response = "Final Answer: Line 1\nLine 2\nLine 3"
        answer = agent._parse_final_answer(response)
        assert "Line 1" in answer
        assert "Line 3" in answer


# ---------------------------------------------------------------------------
# Tests — _execute_tool
# ---------------------------------------------------------------------------


class TestExecuteTool:
    def test_unknown_tool(self):
        agent = ReActAgent()
        result = agent._execute_tool("nonexistent_tool", "input")
        assert "Error" in result
        assert "Unknown tool" in result

    def test_tool_execution_error(self):
        agent = ReActAgent()
        agent.tools["bad_tool"] = lambda x: (_ for _ in ()).throw(ValueError("boom"))
        result = agent._execute_tool("bad_tool", "input")
        assert "Error" in result


# ---------------------------------------------------------------------------
# Tests — _call_llm
# ---------------------------------------------------------------------------


class TestCallLlm:
    def test_no_client_returns_fallback(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        agent = ReActAgent(llm_provider="openai")
        agent._client = None
        result = agent._call_llm([{"role": "user", "content": "test"}])
        assert "LLM not available" in result or "Final Answer" in result


# ---------------------------------------------------------------------------
# Tests — query
# ---------------------------------------------------------------------------


class TestQuery:
    def test_query_with_immediate_final_answer(self):
        agent = ReActAgent()
        # Mock LLM to immediately return a final answer
        agent._call_llm = lambda msgs: "Final Answer: The answer is 42."
        result = agent.query("What is the answer?")
        assert result["answer"] == "The answer is 42."
        assert result["iterations"] == 1
        assert result["tools_used"] == []

    def test_query_with_tool_use(self):
        agent = ReActAgent()
        call_count = 0

        def mock_llm(msgs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Thought: I need data\nAction: get_model_metrics\nAction Input: all"
            return "Final Answer: Model has RMSE of 5.0"

        agent._call_llm = mock_llm
        result = agent.query("What is model performance?")
        assert "get_model_metrics" in result["tools_used"]
        assert result["iterations"] == 2

    def test_query_max_iterations(self):
        agent = ReActAgent(max_iterations=2)
        # Never returns final answer
        agent._call_llm = lambda msgs: "Thought: Still thinking...\nAction: query_stock_data\nAction Input: test"
        result = agent.query("Tell me something")
        assert result["iterations"] <= 2


# ---------------------------------------------------------------------------
# Tests — SYSTEM_PROMPT
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_has_placeholder(self):
        assert "{tool_descriptions}" in SYSTEM_PROMPT

    def test_format_works(self):
        formatted = SYSTEM_PROMPT.format(tool_descriptions="- tool1: desc1")
        assert "tool1" in formatted
