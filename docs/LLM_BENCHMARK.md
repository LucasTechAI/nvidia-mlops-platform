# LLM Benchmark Report

## Visão Geral

Este documento apresenta o benchmark e avaliação do agente LLM integrado
na plataforma NVIDIA MLOps, incluindo métricas de qualidade (RAGAS),
avaliação por LLM-as-judge, e testes A/B de prompts.

---

## 1. Configuração do Agente

| Parâmetro | Valor |
|-----------|-------|
| **Padrão** | ReAct (Reasoning + Acting) |
| **Provider** | OpenAI / Groq (configurável) |
| **Modelo** | gpt-4o-mini (default) |
| **Temperature** | 0.1 |
| **Max Iterations** | 8 |
| **Tools** | 4 (query_stock_data, predict_stock_prices, get_model_metrics, search_documents) |
| **RAG** | ChromaDB com 7 documentos de domínio |

## 2. Golden Set

- **Total de pares**: 25
- **Idiomas**: Português (16), Inglês (9)
- **Domínios cobertos**:
  - Consulta de preços e dados (5 pares)
  - Previsões e modelo (6 pares)
  - Arquitetura e features técnicas (5 pares)
  - Segurança e monitoramento (3 pares)
  - Uso geral do sistema (6 pares)

## 3. Avaliação RAGAS (4 Métricas)

### Métricas

| Métrica | Descrição | Target |
|---------|-----------|--------|
| **Faithfulness** | Resposta é fiel ao contexto fornecido | > 0.7 |
| **Answer Relevancy** | Resposta é relevante à pergunta | > 0.7 |
| **Context Precision** | Contexto recuperado é preciso | > 0.6 |
| **Context Recall** | Contexto cobre a ground truth | > 0.6 |

### Execução

```bash
# Executar avaliação RAGAS
python -m evaluation.ragas_eval
```

Resultados salvos em: `outputs/evaluation/ragas_results.json`

### Interpretação
- **Faithfulness alto**: O agente usa os dados das tools e não alucina
- **Answer Relevancy alto**: Respostas abordam diretamente a pergunta
- **Context Precision alto**: RAG recupera documentos relevantes
- **Context Recall alto**: Contexto cobre informações da ground truth

## 4. LLM-as-Judge (3 Critérios)

### Critérios de Avaliação

| Critério | Descrição | Escala |
|----------|-----------|--------|
| **Relevância** | A resposta aborda diretamente a pergunta | 1–5 |
| **Acurácia Factual** | Informações são factualmente corretas | 1–5 |
| **Utilidade para Negócio** | Útil para análise de investimentos | 1–5 |

### Execução

```bash
# Executar LLM-as-judge
python -m evaluation.llm_judge
```

Resultados salvos em: `outputs/evaluation/llm_judge_results.json`

## 5. Teste A/B de Prompts

### Variantes Testadas

| Variante | Nome | Descrição |
|----------|------|-----------|
| **A** | Concise | Prompt curto, focado em tools |
| **B** | Detailed | Prompt detalhado, com exemplos e regras estritas |

### Diferenças Principais

| Aspecto | Variante A | Variante B |
|---------|------------|------------|
| Comprimento | ~300 tokens | ~600 tokens |
| Exemplos | Nenhum | 1 exemplo completo |
| Regras | Implícitas | 6 regras explícitas |
| Formato numérico | Não especificado | "2 casas decimais" |
| Source attribution | Não requerido | Obrigatório |

### Execução

```bash
# Executar teste A/B (5 primeiras amostras)
python -m evaluation.ab_test_prompts
```

Resultados salvos em: `outputs/evaluation/ab_test_results.json`

## 6. Métricas de Performance

### Latência Esperada

| Operação | Latência (CPU) | Latência (GPU) |
|----------|---------------|----------------|
| Previsão LSTM (30 dias) | 0.5–2s | 0.1–0.5s |
| Query ao agente (1 tool) | 3–8s | 3–8s |
| Query ao agente (3+ tools) | 8–20s | 8–20s |
| RAG retrieval | < 0.5s | < 0.5s |
| Guardrail check | < 0.01s | < 0.01s |

### Token Usage Estimado

| Tipo de Query | Tokens Prompt | Tokens Completion | Total |
|---------------|---------------|-------------------|-------|
| Simples (1 tool) | ~500 | ~200 | ~700 |
| Média (2 tools) | ~1000 | ~400 | ~1400 |
| Complexa (3+ tools) | ~2000 | ~600 | ~2600 |

## 7. Limitações Conhecidas

1. **Dependência de API externa**: Agente requer API key (OpenAI/Groq)
2. **Latência do LLM**: Domina o tempo total de resposta do agente
3. **Fallback heurístico**: Quando LLM não está disponível, avaliação usa heurísticas simples
4. **Golden set estático**: Respostas esperadas podem ficar desatualizadas
5. **Idioma**: RAG knowledge docs em inglês/português misto
6. **Avaliação offline**: RAGAS e LLM-judge são processos batch, não real-time

## 8. Como Reproduzir

```bash
# 1. Configurar variáveis de ambiente
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...

# 2. Executar todas as avaliações
python -m evaluation.ragas_eval
python -m evaluation.llm_judge
python -m evaluation.ab_test_prompts

# 3. Resultados em outputs/evaluation/
ls outputs/evaluation/
# ragas_results.json
# llm_judge_results.json
# ab_test_results.json
```

---

*Relatório gerado em 2025. Métricas atualizadas a cada nova versão do agente ou modelo.*
