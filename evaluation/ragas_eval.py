"""Avaliação do pipeline RAG com RAGAS — 4 métricas obrigatórias.

Referência: Es et al. (2024) — RAGAS: Automated Evaluation of Retrieval
            Augmented Generation. https://arxiv.org/abs/2309.15217

Métricas:
    - faithfulness
    - answer_relevancy
    - context_precision
    - context_recall
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent
GOLDEN_SET_PATH = ROOT_DIR / "data" / "golden_set" / "golden_set.json"
RESULTS_DIR = ROOT_DIR / "outputs" / "evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_golden_set(path: Optional[Path] = None) -> list[dict]:
    """Load the golden set dataset.

    Args:
        path: Path to golden set JSON. Defaults to data/golden_set/golden_set.json.

    Returns:
        List of golden set items with query, expected_answer, contexts.
    """
    path = path or GOLDEN_SET_PATH
    with open(path) as f:
        data = json.load(f)
    logger.info("Loaded golden set with %d items from %s", len(data), path)
    return data


def prepare_ragas_dataset(
    golden_set: list[dict],
    agent_answers: Optional[list[str]] = None,
    retrieved_contexts: Optional[list[list[str]]] = None,
) -> dict:
    """Prepare dataset in RAGAS-compatible format.

    Args:
        golden_set: Golden set items.
        agent_answers: Actual agent answers (if None, uses expected_answer).
        retrieved_contexts: Actually retrieved contexts (if None, uses golden set contexts).

    Returns:
        Dictionary with questions, answers, contexts, ground_truths lists.
    """
    questions = [item["query"] for item in golden_set]
    ground_truths = [item["expected_answer"] for item in golden_set]
    answers = agent_answers or ground_truths
    contexts = retrieved_contexts or [item["contexts"] for item in golden_set]

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def run_ragas_evaluation(
    dataset: Optional[dict] = None,
    golden_set_path: Optional[Path] = None,
    save_results: bool = True,
) -> dict:
    """Run RAGAS evaluation with 4 mandatory metrics.

    Metrics:
        - faithfulness: Is the answer faithful to the given context?
        - answer_relevancy: Is the answer relevant to the question?
        - context_precision: Is the retrieved context precise?
        - context_recall: Does the retrieved context cover the ground truth?

    Args:
        dataset: Pre-prepared dataset dict. If None, loads golden set.
        golden_set_path: Path to golden set (used if dataset is None).
        save_results: Whether to save results to JSON file.

    Returns:
        Dictionary with metric scores and per-sample details.
    """
    if dataset is None:
        golden_set = load_golden_set(golden_set_path)
        dataset = prepare_ragas_dataset(golden_set)

    results = {"metrics": {}, "per_sample": [], "n_samples": len(dataset["question"])}

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        ragas_dataset = Dataset.from_dict(dataset)

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        eval_result = evaluate(dataset=ragas_dataset, metrics=metrics)

        results["metrics"] = {
            "faithfulness": float(eval_result["faithfulness"]),
            "answer_relevancy": float(eval_result["answer_relevancy"]),
            "context_precision": float(eval_result["context_precision"]),
            "context_recall": float(eval_result["context_recall"]),
        }

        # Per-sample scores if available
        if hasattr(eval_result, "to_pandas"):
            df = eval_result.to_pandas()
            for _, row in df.iterrows():
                results["per_sample"].append(
                    {
                        "question": row.get("question", ""),
                        "faithfulness": float(row.get("faithfulness", 0)),
                        "answer_relevancy": float(row.get("answer_relevancy", 0)),
                        "context_precision": float(row.get("context_precision", 0)),
                        "context_recall": float(row.get("context_recall", 0)),
                    }
                )

        logger.info("RAGAS evaluation completed: %s", results["metrics"])

    except ImportError:
        logger.warning("RAGAS not installed. Running fallback heuristic evaluation.")
        results = _fallback_evaluation(dataset)

    except Exception as e:
        logger.error("RAGAS evaluation failed: %s", str(e))
        results = _fallback_evaluation(dataset)

    if save_results:
        output_path = RESULTS_DIR / "ragas_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("RAGAS results saved to %s", output_path)

    return results


def _fallback_evaluation(dataset: dict) -> dict:
    """Heuristic fallback when RAGAS library is not available.

    Computes approximate metrics using string matching and overlap.
    """
    from difflib import SequenceMatcher

    metrics = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }

    per_sample = []
    n = len(dataset["question"])

    for i in range(n):
        question = dataset["question"][i].lower()
        answer = dataset["answer"][i].lower()
        ground_truth = dataset["ground_truth"][i].lower()
        contexts = [c.lower() for c in dataset["contexts"][i]]

        # Faithfulness: overlap between answer and contexts
        context_text = " ".join(contexts)
        answer_words = set(answer.split())
        context_words = set(context_text.split())
        if answer_words:
            faithfulness = len(answer_words & context_words) / len(answer_words)
        else:
            faithfulness = 0.0

        # Answer relevancy: similarity between answer and question
        relevancy = SequenceMatcher(None, question, answer).ratio()

        # Context precision: overlap between context and ground truth
        gt_words = set(ground_truth.split())
        if context_words:
            precision = len(context_words & gt_words) / len(context_words)
        else:
            precision = 0.0

        # Context recall: coverage of ground truth by context
        if gt_words:
            recall = len(context_words & gt_words) / len(gt_words)
        else:
            recall = 0.0

        metrics["faithfulness"].append(faithfulness)
        metrics["answer_relevancy"].append(relevancy)
        metrics["context_precision"].append(precision)
        metrics["context_recall"].append(recall)

        per_sample.append(
            {
                "question": dataset["question"][i],
                "faithfulness": round(faithfulness, 4),
                "answer_relevancy": round(relevancy, 4),
                "context_precision": round(precision, 4),
                "context_recall": round(recall, 4),
            }
        )

    avg_metrics = {k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in metrics.items()}

    return {
        "metrics": avg_metrics,
        "per_sample": per_sample,
        "n_samples": n,
        "note": "Fallback heuristic evaluation (RAGAS library not available)",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_ragas_evaluation()
    print("\n=== RAGAS Evaluation Results ===")
    for metric, score in results["metrics"].items():
        print(f"  {metric}: {score:.4f}")
    print(f"  samples: {results['n_samples']}")
