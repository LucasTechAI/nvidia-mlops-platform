"""Agente ReAct com tools customizadas para o domínio financeiro.

Implements a ReAct (Reasoning + Acting) agent that uses custom tools
to answer questions about NVIDIA stock data, predictions, model metrics,
and financial analysis.

Referência: Yao et al. (2023) — ReAct: Synergizing Reasoning and Acting
            in Language Models. https://arxiv.org/abs/2210.03629
"""

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# System prompt for the ReAct agent
SYSTEM_PROMPT = """You are a specialized financial AI assistant for NVIDIA stock analysis.
You have access to the following tools to help answer questions:

{tool_descriptions}

When answering questions:
1. ALWAYS use tools to get real data — never make up numbers.
2. If the question is about stock prices, use query_stock_data.
3. If the question is about predictions/forecasts, use predict_stock_prices.
4. If the question is about model performance, use get_model_metrics.
5. If the question needs general context, use search_documents.
6. Include a risk disclaimer when giving predictions or investment advice.
7. Answer in the same language as the user's question (Portuguese or English).

Use the ReAct format:
Thought: [your reasoning about what to do]
Action: [tool_name]
Action Input: [input for the tool]
Observation: [tool output - you will see this after the action runs]
... (repeat as needed)
Thought: I now have enough information to answer.
Final Answer: [your comprehensive answer to the user]
"""


class ReActAgent:
    """ReAct agent with custom financial domain tools.

    This agent uses a reasoning-action loop to answer financial questions
    by invoking tools (stock data queries, LSTM predictions, model metrics,
    document search) and composing comprehensive responses.

    Attributes:
        tools: Dictionary of available tool functions.
        llm_provider: LLM provider ('openai', 'groq', or 'local').
        model_name: Name of the LLM model to use.
        temperature: Sampling temperature for LLM.
        max_iterations: Maximum reasoning-action loops.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_iterations: int = 8,
        llm_provider: Optional[str] = None,
    ):
        """Initialize the ReAct agent.

        Args:
            model_name: LLM model name. Defaults to env var or gpt-4o-mini.
            temperature: Sampling temperature (0.0 = deterministic).
            max_iterations: Max reasoning loops before forcing final answer.
            llm_provider: 'openai', 'groq', or 'local'. Auto-detected if None.
        """
        from src.agent.tools import TOOL_REGISTRY

        self.tools = {name: info["function"] for name, info in TOOL_REGISTRY.items()}
        self.tool_descriptions = "\n".join(
            f"- {info['name']}: {info['description']}" for info in TOOL_REGISTRY.values()
        )
        self.temperature = temperature
        self.max_iterations = max_iterations

        # Determine LLM provider and model
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "openai")
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-4o-mini")

        self._client = None
        logger.info(
            "ReActAgent initialized with provider=%s, model=%s, tools=%d",
            self.llm_provider,
            self.model_name,
            len(self.tools),
        )

    def _get_client(self):
        """Get or create the LLM client."""
        if self._client is not None:
            return self._client

        if self.llm_provider == "groq":
            try:
                from groq import Groq

                self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                return self._client
            except ImportError:
                logger.warning("groq not installed, falling back to openai")
                self.llm_provider = "openai"
            except Exception as e:
                logger.warning("Failed to create Groq client: %s", e)
                self.llm_provider = "openai"

        if self.llm_provider == "openai":
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                return self._client
            except ImportError:
                logger.warning("openai not installed, using tool-only mode")
                return None
            except Exception as e:
                logger.warning("Failed to create OpenAI client: %s", e)
                return None

        return None

    def _call_llm(self, messages: list[dict]) -> str:
        """Call the LLM with the given messages.

        Args:
            messages: Chat messages in OpenAI format.

        Returns:
            LLM response text.
        """
        client = self._get_client()
        if client is None:
            return "Final Answer: LLM not available. Please set OPENAI_API_KEY or GROQ_API_KEY."

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=2048,
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return f"Final Answer: Error calling LLM: {e}"

    def _parse_action(self, response: str) -> tuple[Optional[str], Optional[str]]:
        """Parse action and input from LLM response.

        Args:
            response: Raw LLM response text.

        Returns:
            Tuple of (action_name, action_input) or (None, None).
        """
        # Match "Action: tool_name" and "Action Input: input"
        action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", response)
        input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", response)

        if action_match:
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip() if input_match else ""
            return action, action_input

        return None, None

    def _parse_final_answer(self, response: str) -> Optional[str]:
        """Extract final answer from LLM response.

        Args:
            response: Raw LLM response text.

        Returns:
            Final answer string or None.
        """
        match = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _execute_tool(self, action: str, action_input: str) -> str:
        """Execute a tool and return the result.

        Args:
            action: Tool name.
            action_input: Input for the tool.

        Returns:
            Tool execution result.
        """
        if action not in self.tools:
            return f"Error: Unknown tool '{action}'. Available: {list(self.tools.keys())}"

        try:
            tool_fn = self.tools[action]
            result = tool_fn(action_input)
            return str(result)
        except Exception as e:
            logger.error("Tool execution failed: %s(%s) → %s", action, action_input, e)
            return f"Error executing {action}: {e}"

    def query(self, user_input: str) -> dict:
        """Process a user query through the ReAct loop.

        Runs the reasoning-action loop: the agent thinks about what to do,
        executes tools, observes results, and eventually provides a final answer.

        Args:
            user_input: User's question in natural language.

        Returns:
            Dict with 'answer', 'reasoning_trace', 'tools_used', 'iterations'.
        """
        system_message = SYSTEM_PROMPT.format(tool_descriptions=self.tool_descriptions)
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input},
        ]

        reasoning_trace: list[dict] = []
        tools_used: list[str] = []

        for iteration in range(self.max_iterations):
            # Get LLM response
            response = self._call_llm(messages)

            # Check for final answer
            final_answer = self._parse_final_answer(response)
            if final_answer:
                reasoning_trace.append({"step": "final_answer", "content": final_answer})
                return {
                    "answer": final_answer,
                    "reasoning_trace": reasoning_trace,
                    "tools_used": tools_used,
                    "iterations": iteration + 1,
                }

            # Parse action
            action, action_input = self._parse_action(response)
            if action:
                reasoning_trace.append(
                    {
                        "step": "action",
                        "thought": response.split("Action:")[0].strip(),
                        "action": action,
                        "action_input": action_input,
                    }
                )

                # Execute tool
                observation = self._execute_tool(action, action_input or "")
                tools_used.append(action)
                reasoning_trace.append({"step": "observation", "content": observation[:2000]})

                # Add to conversation
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Observation: {observation[:2000]}"})
            else:
                # No action found — treat as final answer
                reasoning_trace.append({"step": "direct_response", "content": response})
                return {
                    "answer": response,
                    "reasoning_trace": reasoning_trace,
                    "tools_used": tools_used,
                    "iterations": iteration + 1,
                }

        # Max iterations reached
        return {
            "answer": "I was unable to complete the analysis within the allowed iterations. "
            "Please try a more specific question.",
            "reasoning_trace": reasoning_trace,
            "tools_used": tools_used,
            "iterations": self.max_iterations,
        }

    def query_with_guardrails(self, user_input: str) -> dict:
        """Process a user query with input/output guardrails applied.

        Validates input, processes query, and sanitizes output.

        Args:
            user_input: User's question.

        Returns:
            Dict with 'answer', 'reasoning_trace', 'tools_used',
            'iterations', 'input_valid', 'output_sanitized'.
        """
        try:
            from src.security.guardrails import InputGuardrail, OutputGuardrail

            input_guard = InputGuardrail()
            is_valid, reason = input_guard.validate(user_input)

            if not is_valid:
                return {
                    "answer": reason,
                    "reasoning_trace": [{"step": "input_blocked", "reason": reason}],
                    "tools_used": [],
                    "iterations": 0,
                    "input_valid": False,
                    "output_sanitized": False,
                }

            result = self.query(user_input)

            output_guard = OutputGuardrail()
            result["answer"] = output_guard.sanitize(result["answer"])
            result["input_valid"] = True
            result["output_sanitized"] = True

            return result

        except ImportError:
            logger.warning("Guardrails not available, running without")
            result = self.query(user_input)
            result["input_valid"] = True
            result["output_sanitized"] = False
            return result


# Convenience function
def create_agent(
    model_name: Optional[str] = None,
    temperature: float = 0.1,
    llm_provider: Optional[str] = None,
) -> ReActAgent:
    """Create a configured ReAct agent instance.

    Args:
        model_name: LLM model name.
        temperature: Sampling temperature.
        llm_provider: LLM provider name.

    Returns:
        Configured ReActAgent.
    """
    return ReActAgent(
        model_name=model_name,
        temperature=temperature,
        llm_provider=llm_provider,
    )
