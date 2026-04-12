"""Avaliação do pipeline RAG com RAGAS — 4 métricas obrigatórias.

Referência: Es et al. (2024) — RAGAS: Automated Evaluation of Retrieval
            Augmented Generation. https://arxiv.org/abs/2309.15217

Métricas:
    - faithfulness
    - answer_relevancy
    - context_precision
    - context_recall
"""

import asyncio
import concurrent.futures
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
        import os
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        # Configure LLM backend — supports OpenRouter (OpenAI-compatible)
        provider = os.getenv("LLM_PROVIDER", "openai")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openrouter_base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        llm_model = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

        ragas_llm = None
        ragas_embeddings = None

        if provider == "openrouter" and openrouter_key:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper

            lc_llm = ChatOpenAI(
                model=llm_model,
                api_key=openrouter_key,
                base_url=openrouter_base,
                default_headers={
                    "HTTP-Referer": "https://nvidia-mlops-platform",
                    "X-Title": "NVIDIA MLOps RAGAS Eval",
                },
            )
            ragas_llm = LangchainLLMWrapper(lc_llm)

            # Embeddings: OpenRouter doesn't serve embeddings — fall back to
            # a lightweight local model via sentence-transformers if available,
            # otherwise use OpenAI embeddings with the same key/base.
            try:
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                except ImportError:
                    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
                lc_emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                ragas_embeddings = LangchainEmbeddingsWrapper(lc_emb)
                logger.info("Using HuggingFace embeddings (all-MiniLM-L6-v2) for RAGAS")
            except Exception:
                logger.warning("HuggingFace embeddings unavailable; answer_relevancy may be skipped")

            logger.info("RAGAS configured with OpenRouter (%s)", llm_model)

        ragas_dataset = Dataset.from_dict(dataset)

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        eval_kwargs: dict = {}
        if ragas_llm:
            eval_kwargs["llm"] = ragas_llm
        if ragas_embeddings:
            eval_kwargs["embeddings"] = ragas_embeddings

        # Run evaluate() in a dedicated thread with its own event loop.
        # FastAPI uses uvloop which does not support nested asyncio.run() calls;
        # a worker thread is isolated from the outer loop entirely.
        def _ragas_worker():
            worker_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(worker_loop)
            try:
                return evaluate(dataset=ragas_dataset, metrics=metrics, **eval_kwargs)
            finally:
                worker_loop.close()
                asyncio.set_event_loop(None)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
            eval_result = _pool.submit(_ragas_worker).result()

        def _to_float(v) -> float:
            """Handle both scalar and list results from different RAGAS versions."""
            if isinstance(v, list):
                valid = [x for x in v if x is not None]
                return float(sum(valid) / len(valid)) if valid else 0.0
            return float(v) if v is not None else 0.0

        results["metrics"] = {
            "faithfulness": _to_float(eval_result["faithfulness"]),
            "answer_relevancy": _to_float(eval_result["answer_relevancy"]),
            "context_precision": _to_float(eval_result["context_precision"]),
            "context_recall": _to_float(eval_result["context_recall"]),
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
        results = _fallback_evaluation(dataset, note="Fallback heuristic evaluation (ragas library not installed)")

    except Exception as e:
        error_str = str(e)
        if "api_key" in error_str.lower() or "openai" in error_str.lower() or "openrouter" in error_str.lower():
            note = "Fallback heuristic evaluation (LLM API key error — check OPENROUTER_API_KEY in .env)"
        else:
            note = f"Fallback heuristic evaluation (RAGAS error: {error_str[:120]})"
        logger.error("RAGAS evaluation failed: %s", error_str)
        results = _fallback_evaluation(dataset, note=note)

    if save_results:
        output_path = RESULTS_DIR / "ragas_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("RAGAS results saved to %s", output_path)

    return results


# Common stopwords to ignore when computing keyword overlap
_STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "of", "to", "and", "or", "for",
    "be", "are", "was", "were", "by", "at", "as", "do", "can", "its", "this",
    "that", "with", "from", "which", "how", "what", "why", "when", "does",
    "has", "have", "not", "but", "if", "so", "will", "their", "they",
}


def _keywords(text: str) -> set:
    """Return meaningful (non-stopword) lowercase tokens from text."""
    return {w for w in text.lower().split() if len(w) > 2 and w not in _STOPWORDS}


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets, returns 0.0 if both empty."""
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _fallback_evaluation(dataset: dict, note: str = "Fallback heuristic evaluation (RAGAS library not available)") -> dict:
    """Heuristic fallback when RAGAS library is not available.

    Uses keyword-based overlap rather than raw string similarity so that
    scores are proportional to actual semantic content overlap.

    Metrics (all in [0, 1]):
        faithfulness     — how much of the answer is supported by the contexts
        answer_relevancy — how many question keywords appear in the answer
        context_precision — how much of the context is relevant to ground truth
        context_recall   — how much of the ground truth is covered by contexts
    """
    metrics: dict = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }

    per_sample = []
    n = len(dataset["question"])

    for i in range(n):
        q_kw = _keywords(dataset["question"][i])
        a_kw = _keywords(dataset["answer"][i])
        gt_kw = _keywords(dataset["ground_truth"][i])
        ctx_kw = _keywords(" ".join(c for c in dataset["contexts"][i]))

        # Faithfulness: fraction of answer keywords that appear in contexts
        faithfulness = len(a_kw & ctx_kw) / len(a_kw) if a_kw else 0.0

        # Answer relevancy: fraction of question keywords present in the answer
        answer_relevancy = len(q_kw & a_kw) / len(q_kw) if q_kw else 0.0

        # Context precision: Jaccard between context keywords and ground-truth keywords
        precision = _jaccard(ctx_kw, gt_kw)

        # Context recall: fraction of ground-truth keywords covered by contexts
        context_recall = len(ctx_kw & gt_kw) / len(gt_kw) if gt_kw else 0.0

        # Clamp all to [0, 1]
        faithfulness = min(1.0, faithfulness)
        answer_relevancy = min(1.0, answer_relevancy)
        precision = min(1.0, precision)
        context_recall = min(1.0, context_recall)

        metrics["faithfulness"].append(faithfulness)
        metrics["answer_relevancy"].append(answer_relevancy)
        metrics["context_precision"].append(precision)
        metrics["context_recall"].append(context_recall)

        per_sample.append(
            {
                "question": dataset["question"][i],
                "faithfulness": round(faithfulness, 4),
                "answer_relevancy": round(answer_relevancy, 4),
                "context_precision": round(precision, 4),
                "context_recall": round(context_recall, 4),
            }
        )

    avg_metrics = {k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in metrics.items()}

    return {
        "metrics": avg_metrics,
        "per_sample": per_sample,
        "n_samples": n,
        "note": note,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_ragas_evaluation()
    print("\n=== RAGAS Evaluation Results ===")
    for metric, score in results["metrics"].items():
        print(f"  {metric}: {score:.4f}")
    print(f"  samples: {results['n_samples']}")
