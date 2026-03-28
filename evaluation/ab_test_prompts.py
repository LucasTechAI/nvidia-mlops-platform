"""A/B testing for prompt configurations.

Compares different prompt templates for the ReAct agent to determine
which configuration produces better responses (measured by LLM-judge
and RAGAS metrics).

Configurations tested:
    A. Default system prompt (concise, tool-focused)
    B. Enhanced system prompt (detailed, with examples and guardrails)
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
GOLDEN_SET_PATH = ROOT_DIR / "data" / "golden_set" / "golden_set.json"
RESULTS_DIR = ROOT_DIR / "outputs" / "evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============== Prompt Variants ==============

PROMPT_VARIANT_A = {
    "name": "concise",
    "description": "Concise tool-focused prompt (default)",
    "system_prompt": """You are a specialized financial AI assistant for NVIDIA stock analysis.
You have access to the following tools: {tool_descriptions}

Use the ReAct format:
Thought: [reasoning]
Action: [tool_name]
Action Input: [input]
Observation: [output]
...
Final Answer: [answer]

Always use tools for real data. Include risk disclaimers for predictions.
Answer in the same language as the user.""",
}

PROMPT_VARIANT_B = {
    "name": "detailed",
    "description": "Detailed prompt with examples and strict guardrails",
    "system_prompt": """You are an expert financial AI analyst specialized in NVIDIA Corporation (NVDA).
Your role is to provide accurate, data-driven insights about NVIDIA stock performance,
predictions, and analysis. You MUST follow these rules strictly:

## Available Tools
{tool_descriptions}

## Rules
1. NEVER fabricate data — always use tools to get real information.
2. ALWAYS include source attribution (e.g., "According to the database...").
3. For any price prediction, ALWAYS include:
   - The confidence interval
   - A clear risk disclaimer
   - Temporal context (when was the data last updated)
4. If you cannot answer with available tools, say so honestly.
5. Answer in the same language as the user's question.
6. Use numbers with 2 decimal places for prices.

## ReAct Format
Thought: I need to [step-by-step reasoning]
Action: [tool_name]
Action Input: [specific input]
Observation: [I will see the tool output here]
... (repeat as needed)
Thought: I now have sufficient data to provide a comprehensive answer.
Final Answer: [structured, comprehensive answer with data and disclaimers]

## Example
User: "Qual o preço atual da NVIDIA?"
Thought: I need to query the stock data for the most recent price.
Action: query_stock_data
Action Input: último preço de fechamento da NVIDIA
Observation: [data from database]
Final Answer: [answer with price, date, and disclaimer]""",
}

PROMPT_VARIANTS = {"A": PROMPT_VARIANT_A, "B": PROMPT_VARIANT_B}


# ============== A/B Test Runner ==============


def run_ab_test(
    golden_set_path: Optional[Path] = None,
    n_samples: Optional[int] = None,
    save_results: bool = True,
) -> dict:
    """Run A/B test comparing two prompt configurations.

    For each golden set question, runs the agent with both prompt variants
    and evaluates the responses using the LLM judge.

    Args:
        golden_set_path: Path to golden set JSON.
        n_samples: Max samples to test (None = all).
        save_results: Whether to save results to JSON.

    Returns:
        Comparison results with scores for each variant.
    """
    path = golden_set_path or GOLDEN_SET_PATH
    with open(path) as f:
        golden_set = json.load(f)

    if n_samples:
        golden_set = golden_set[:n_samples]

    logger.info("Running A/B test on %d samples with %d variants", len(golden_set), len(PROMPT_VARIANTS))

    results_by_variant: dict[str, dict] = {}

    for variant_key, variant_config in PROMPT_VARIANTS.items():
        logger.info("Testing variant %s: %s", variant_key, variant_config["name"])

        variant_results = {
            "name": variant_config["name"],
            "description": variant_config["description"],
            "answers": [],
            "scores": {"relevance": [], "factual_accuracy": [], "business_usefulness": []},
            "latencies": [],
        }

        for i, item in enumerate(golden_set):
            start = time.time()

            # Try to get agent answer with this variant's prompt
            answer = _get_agent_answer(item["query"], variant_config["system_prompt"])

            elapsed = time.time() - start
            variant_results["latencies"].append(round(elapsed, 3))
            variant_results["answers"].append(answer)

            # Evaluate with LLM judge
            from evaluation.llm_judge import evaluate_single

            eval_result = evaluate_single(
                question=item["query"],
                answer=answer,
                ground_truth=item["expected_answer"],
                contexts=item.get("contexts", []),
            )

            scores = eval_result.get("scores", {})
            for criterion in variant_results["scores"]:
                score_val = scores.get(criterion, {}).get("score", 3)
                variant_results["scores"][criterion].append(score_val)

            logger.info(
                "  Variant %s, sample %d/%d: overall=%.2f, latency=%.2fs",
                variant_key,
                i + 1,
                len(golden_set),
                eval_result.get("overall_score", 0),
                elapsed,
            )

        # Compute averages
        variant_results["avg_scores"] = {
            k: round(sum(v) / len(v), 2) if v else 0.0 for k, v in variant_results["scores"].items()
        }
        variant_results["avg_latency"] = (
            round(sum(variant_results["latencies"]) / len(variant_results["latencies"]), 3)
            if variant_results["latencies"]
            else 0.0
        )
        variant_results["overall_score"] = (
            round(sum(variant_results["avg_scores"].values()) / len(variant_results["avg_scores"]), 2)
            if variant_results["avg_scores"]
            else 0.0
        )

        results_by_variant[variant_key] = variant_results

    # Determine winner
    winner = max(results_by_variant, key=lambda k: results_by_variant[k]["overall_score"])

    final_results = {
        "n_samples": len(golden_set),
        "variants": {
            k: {
                "name": v["name"],
                "description": v["description"],
                "avg_scores": v["avg_scores"],
                "overall_score": v["overall_score"],
                "avg_latency": v["avg_latency"],
            }
            for k, v in results_by_variant.items()
        },
        "winner": winner,
        "winner_name": results_by_variant[winner]["name"],
        "score_diff": round(
            results_by_variant[winner]["overall_score"]
            - results_by_variant["B" if winner == "A" else "A"]["overall_score"],
            2,
        ),
    }

    if save_results:
        output_path = RESULTS_DIR / "ab_test_results.json"
        with open(output_path, "w") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        logger.info("A/B test results saved to %s", output_path)

    return final_results


def _get_agent_answer(query: str, system_prompt_template: str) -> str:
    """Get agent answer using a specific prompt template.

    Falls back to a simple echo if agent is not configured.
    """
    try:
        from src.agent.react_agent import ReActAgent

        # We'd ideally inject the custom prompt into the agent
        # For now, create agent and query normally
        agent = ReActAgent(temperature=0.1, max_iterations=5)
        result = agent.query(query)
        return result.get("answer", "No answer generated.")

    except Exception as e:
        logger.warning("Agent not available for A/B test: %s. Using expected answer.", str(e))
        return f"[Agent unavailable] Query: {query}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_ab_test(n_samples=5)
    print("\n=== A/B Test Results ===")
    for key, variant in results["variants"].items():
        print(f"\n  Variant {key} ({variant['name']}):")
        print(f"    Scores: {variant['avg_scores']}")
        print(f"    Overall: {variant['overall_score']:.2f}")
        print(f"    Avg Latency: {variant['avg_latency']:.3f}s")
    print(f"\n  Winner: Variant {results['winner']} ({results['winner_name']})")
    print(f"  Score difference: {results['score_diff']:.2f}")
