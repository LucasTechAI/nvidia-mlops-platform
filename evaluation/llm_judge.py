"""LLM-as-judge evaluation with ≥ 3 criteria.

Criteria:
    1. Relevância da resposta (Answer Relevance)
    2. Acurácia factual (Factual Accuracy)
    3. Utilidade para decisão de investimento (Business Usefulness)

Referência: Zheng et al. (2023) — Judging LLM-as-a-Judge with MT-Bench
            https://arxiv.org/abs/2306.05685
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
GOLDEN_SET_PATH = ROOT_DIR / "data" / "golden_set" / "golden_set.json"
RESULTS_DIR = ROOT_DIR / "outputs" / "evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Evaluation criteria definitions
CRITERIA = {
    "relevance": {
        "name": "Relevância da Resposta",
        "description": (
            "A resposta aborda diretamente a pergunta do usuário? Avalie se a resposta é pertinente ao tema perguntado."
        ),
        "scale": "1-5 (1=completamente irrelevante, 5=perfeitamente relevante)",
    },
    "factual_accuracy": {
        "name": "Acurácia Factual",
        "description": (
            "A resposta contém informações factualmente corretas? "
            "Avalie se os dados, números e afirmações estão corretos com base no contexto fornecido."
        ),
        "scale": "1-5 (1=muitos erros factuais, 5=totalmente preciso)",
    },
    "business_usefulness": {
        "name": "Utilidade para Decisão de Investimento",
        "description": (
            "A resposta é útil para alguém analisando investimentos em NVIDIA? "
            "Avalie se inclui disclaimers de risco, informações acionáveis e clareza."
        ),
        "scale": "1-5 (1=inútil, 5=extremamente útil e completa)",
    },
}

JUDGE_PROMPT_TEMPLATE = """You are an expert financial AI evaluator. Evaluate the following
AI assistant response based on the specified criteria.

## Question
{question}

## Reference Answer (Ground Truth)
{ground_truth}

## AI Assistant Response
{answer}

## Context Provided
{context}

## Evaluation Criteria
{criteria_description}

## Instructions
For each criterion, provide:
1. A score from 1 to 5
2. A brief justification (1-2 sentences)

Respond in valid JSON format:
{{
    "scores": {{
        "relevance": {{"score": <int>, "justification": "<text>"}},
        "factual_accuracy": {{"score": <int>, "justification": "<text>"}},
        "business_usefulness": {{"score": <int>, "justification": "<text>"}}
    }},
    "overall_score": <float>,
    "summary": "<brief overall assessment>"
}}
"""


def _build_criteria_text() -> str:
    """Build criteria description text for the prompt."""
    lines = []
    for key, info in CRITERIA.items():
        lines.append(f"- **{info['name']}**: {info['description']} Scale: {info['scale']}")
    return "\n".join(lines)


def _call_llm_judge(prompt: str) -> dict:
    """Call LLM to judge a response.

    Supports OpenAI and Groq providers.

    Returns:
        Parsed JSON dict with scores, or fallback scores on error.
    """
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_JUDGE_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))

    try:
        if provider == "groq":
            from groq import Groq

            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        else:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert evaluator. Always respond in valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1000,
        )

        content = response.choices[0].message.content.strip()
        # Extract JSON from potential markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        return json.loads(content)

    except ImportError:
        logger.warning("LLM library not installed. Using heuristic judge.")
        return None
    except Exception as e:
        logger.error("LLM judge call failed: %s", str(e))
        return None


def _heuristic_judge(question: str, answer: str, ground_truth: str, contexts: list[str]) -> dict:
    """Heuristic fallback judge using string similarity.

    Used when LLM is not available.
    """
    from difflib import SequenceMatcher

    q_lower = question.lower()
    a_lower = answer.lower()
    gt_lower = ground_truth.lower()
    ctx_text = " ".join(c.lower() for c in contexts)

    # Relevance: similarity between answer and question
    relevance_sim = SequenceMatcher(None, q_lower, a_lower).ratio()
    relevance_score = max(1, min(5, round(relevance_sim * 5)))

    # Factual accuracy: overlap with ground truth
    gt_words = set(gt_lower.split())
    a_words = set(a_lower.split())
    if gt_words:
        accuracy_overlap = len(gt_words & a_words) / len(gt_words)
    else:
        accuracy_overlap = 0.5
    accuracy_score = max(1, min(5, round(accuracy_overlap * 5)))

    # Business usefulness: check for key financial terms and disclaimers
    useful_terms = ["risco", "investimento", "disclaimer", "previsão", "dados", "modelo", "risk", "prediction"]
    term_count = sum(1 for t in useful_terms if t in a_lower or t in ctx_text)
    usefulness_score = max(1, min(5, 1 + term_count))

    overall = round((relevance_score + accuracy_score + usefulness_score) / 3, 2)

    return {
        "scores": {
            "relevance": {"score": relevance_score, "justification": "Heuristic: string similarity"},
            "factual_accuracy": {"score": accuracy_score, "justification": "Heuristic: word overlap with ground truth"},
            "business_usefulness": {"score": usefulness_score, "justification": "Heuristic: financial term presence"},
        },
        "overall_score": overall,
        "summary": "Heuristic evaluation (LLM not available)",
    }


def evaluate_single(
    question: str,
    answer: str,
    ground_truth: str,
    contexts: list[str],
) -> dict:
    """Evaluate a single question-answer pair.

    Args:
        question: The user's question.
        answer: The AI assistant's answer.
        ground_truth: The expected/reference answer.
        contexts: Retrieved context documents.

    Returns:
        Dictionary with scores per criterion and overall score.
    """
    criteria_text = _build_criteria_text()
    context_str = "\n".join(f"- {c}" for c in contexts) if contexts else "No context provided."

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
        context=context_str,
        criteria_description=criteria_text,
    )

    result = _call_llm_judge(prompt)
    if result is None:
        result = _heuristic_judge(question, answer, ground_truth, contexts)

    return result


def run_llm_judge_evaluation(
    golden_set_path: Optional[Path] = None,
    agent_answers: Optional[list[str]] = None,
    save_results: bool = True,
) -> dict:
    """Run LLM-as-judge evaluation on the golden set.

    Args:
        golden_set_path: Path to golden set JSON.
        agent_answers: Actual agent answers. If None, uses expected_answer.
        save_results: Whether to save results to JSON.

    Returns:
        Aggregated evaluation results.
    """
    path = golden_set_path or GOLDEN_SET_PATH
    with open(path) as f:
        golden_set = json.load(f)

    logger.info("Running LLM judge on %d samples", len(golden_set))

    all_scores = {"relevance": [], "factual_accuracy": [], "business_usefulness": []}
    per_sample = []

    for i, item in enumerate(golden_set):
        answer = agent_answers[i] if agent_answers else item["expected_answer"]

        result = evaluate_single(
            question=item["query"],
            answer=answer,
            ground_truth=item["expected_answer"],
            contexts=item.get("contexts", []),
        )

        scores = result.get("scores", {})
        for criterion in all_scores:
            score_val = scores.get(criterion, {}).get("score", 3)
            all_scores[criterion].append(score_val)

        per_sample.append(
            {
                "id": item["id"],
                "query": item["query"],
                "scores": scores,
                "overall_score": result.get("overall_score", 0),
                "summary": result.get("summary", ""),
            }
        )

        logger.info("  Sample %d/%d evaluated (overall=%.2f)", i + 1, len(golden_set), result.get("overall_score", 0))

    # Aggregate
    avg_scores = {}
    for criterion, scores_list in all_scores.items():
        avg_scores[criterion] = round(sum(scores_list) / len(scores_list), 2) if scores_list else 0.0

    overall_avg = round(sum(avg_scores.values()) / len(avg_scores), 2) if avg_scores else 0.0

    results = {
        "criteria": list(CRITERIA.keys()),
        "avg_scores": avg_scores,
        "overall_avg": overall_avg,
        "n_samples": len(golden_set),
        "per_sample": per_sample,
    }

    if save_results:
        output_path = RESULTS_DIR / "llm_judge_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("LLM judge results saved to %s", output_path)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_llm_judge_evaluation()
    print("\n=== LLM-as-Judge Results ===")
    for criterion, score in results["avg_scores"].items():
        print(f"  {criterion}: {score:.2f}/5.0")
    print(f"  overall: {results['overall_avg']:.2f}/5.0")
    print(f"  samples: {results['n_samples']}")
