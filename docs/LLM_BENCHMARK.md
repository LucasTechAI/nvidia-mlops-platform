# LLM Benchmark Report

## Overview

This document presents the benchmark and evaluation of the LLM agent
integrated into the NVIDIA MLOps platform, including quality metrics (RAGAS),
LLM-as-judge evaluation, and A/B prompt testing.

---

## 1. Agent Configuration

| Parameter | Value |
|-----------|-------|
| **Pattern** | ReAct (Reasoning + Acting) |
| **Provider** | OpenAI / Groq (configurable) |
| **Model** | gpt-4o-mini (default) |
| **Temperature** | 0.1 |
| **Max Iterations** | 8 |
| **Tools** | 4 (query_stock_data, predict_stock_prices, get_model_metrics, search_documents) |
| **RAG** | ChromaDB with 7 domain documents |

## 2. Golden Set

- **Total pairs**: 25
- **Languages**: Portuguese (16), English (9)
- **Covered domains**:
  - Price and data queries (5 pairs)
  - Predictions and model (6 pairs)
  - Architecture and technical features (5 pairs)
  - Security and monitoring (3 pairs)
  - General system usage (6 pairs)

## 3. RAGAS Evaluation (4 Metrics)

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Faithfulness** | Answer is faithful to the provided context | > 0.7 |
| **Answer Relevancy** | Answer is relevant to the question | > 0.7 |
| **Context Precision** | Retrieved context is precise | > 0.6 |
| **Context Recall** | Context covers the ground truth | > 0.6 |

### Execution

```bash
# Run RAGAS evaluation
python -m evaluation.ragas_eval
```

Results saved at: `outputs/evaluation/ragas_results.json`

### Interpretation
- **High Faithfulness**: Agent uses tool data and doesn't hallucinate
- **High Answer Relevancy**: Answers directly address the question
- **High Context Precision**: RAG retrieves relevant documents
- **High Context Recall**: Context covers ground truth information

## 4. LLM-as-Judge (3 Criteria)

### Evaluation Criteria

| Criterion | Description | Scale |
|-----------|-------------|-------|
| **Relevance** | Answer directly addresses the question | 1–5 |
| **Factual Accuracy** | Information is factually correct | 1–5 |
| **Business Usefulness** | Useful for investment analysis | 1–5 |

### Execution

```bash
# Run LLM-as-judge
python -m evaluation.llm_judge
```

Results saved at: `outputs/evaluation/llm_judge_results.json`

## 5. A/B Prompt Testing

### Tested Variants

| Variant | Name | Description |
|---------|------|-------------|
| **A** | Concise | Short prompt, tool-focused |
| **B** | Detailed | Detailed prompt, with examples and strict rules |

### Key Differences

| Aspect | Variant A | Variant B |
|--------|-----------|-----------|
| Length | ~300 tokens | ~600 tokens |
| Examples | None | 1 complete example |
| Rules | Implicit | 6 explicit rules |
| Number format | Unspecified | "2 decimal places" |
| Source attribution | Not required | Required |

### Execution

```bash
# Run A/B test (first 5 samples)
python -m evaluation.ab_test_prompts
```

Results saved at: `outputs/evaluation/ab_test_results.json`

## 6. Performance Metrics

### Expected Latency

| Operation | Latency (CPU) | Latency (GPU) |
|-----------|---------------|---------------|
| LSTM Prediction (30 days) | 0.5–2s | 0.1–0.5s |
| Agent query (1 tool) | 3–8s | 3–8s |
| Agent query (3+ tools) | 8–20s | 8–20s |
| RAG retrieval | < 0.5s | < 0.5s |
| Guardrail check | < 0.01s | < 0.01s |

### Estimated Token Usage

| Query Type | Prompt Tokens | Completion Tokens | Total |
|------------|---------------|-------------------|-------|
| Simple (1 tool) | ~500 | ~200 | ~700 |
| Medium (2 tools) | ~1000 | ~400 | ~1400 |
| Complex (3+ tools) | ~2000 | ~600 | ~2600 |

## 7. Known Limitations

1. **External API dependency**: Agent requires API key (OpenAI/Groq)
2. **LLM latency**: Dominates total agent response time
3. **Heuristic fallback**: When LLM is unavailable, evaluation uses simple heuristics
4. **Static golden set**: Expected answers may become outdated
5. **Language**: RAG knowledge docs in mixed English/Portuguese
6. **Offline evaluation**: RAGAS and LLM-judge are batch processes, not real-time

## 8. How to Reproduce

```bash
# 1. Configure environment variables
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...

# 2. Run all evaluations
python -m evaluation.ragas_eval
python -m evaluation.llm_judge
python -m evaluation.ab_test_prompts

# 3. Results in outputs/evaluation/
ls outputs/evaluation/
# ragas_results.json
# llm_judge_results.json
# ab_test_results.json
```

---

*Report generated in 2025. Metrics updated with each new agent or model version.*
