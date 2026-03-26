# OWASP Top 10 for LLM Applications — Mapeamento

## Referência

[OWASP Top 10 for Large Language Model Applications (2025)](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

---

## Mapeamento de Riscos e Mitigações

### LLM01: Prompt Injection

| Aspecto | Detalhe |
|---------|---------|
| **Risco** | Atacante manipula o prompt do agente para executar ações não autorizadas |
| **Nível** | 🟡 Médio |
| **Mitigação** | `InputGuardrail` com 16 padrões regex de prompt injection |
| **Implementação** | `src/security/guardrails.py` — `InputGuardrail._check_injection()` |
| **Padrões detectados** | "ignore previous instructions", "system:", "jailbreak", "DAN mode", "bypass safety", "reveal system prompt", entre outros |
| **Ação** | Input bloqueado antes de chegar ao LLM |

### LLM02: Insecure Output Handling

| Aspecto | Detalhe |
|---------|---------|
| **Risco** | Output do LLM contém conteúdo nocivo, PII ou instruções executáveis |
| **Nível** | 🟡 Médio |
| **Mitigação** | `OutputGuardrail` com PII detection, content filtering e disclaimers |
| **Implementação** | `src/security/guardrails.py` — `OutputGuardrail.validate()` |
| **Ações** | PII removido (Presidio), conteúdo nocivo bloqueado, disclaimers adicionados |

### LLM03: Training Data Poisoning

| Aspecto | Detalhe |
|---------|---------|
| **Risco** | Dados de treinamento comprometidos afetam as previsões |
| **Nível** | 🟢 Baixo |
| **Mitigação** | Dados de fonte confiável (Yahoo Finance), drift detection (PSI) |
| **Implementação** | `src/monitoring/drift.py`, `src/etl/preprocessing.py` |
| **Ação** | PSI > 0.2 aciona alerta de retreino |

### LLM04: Model Denial of Service

| Aspecto | Detalhe |
|---------|---------|
| **Risco** | Inputs extremamente longos ou complexos sobrecarregam o sistema |
| **Nível** | 🟡 Médio |
| **Mitigação** | Max input length (2000 chars), max iterations (8), timeouts |
| **Implementação** | `src/security/guardrails.py` — `MAX_INPUT_LENGTH`, `src/agent/react_agent.py` — `max_iterations` |
| **Ação** | Inputs grandes rejeitados, loops do agente limitados |

### LLM05: Supply Chain Vulnerabilities

| Aspecto | Detalhe |
|---------|---------|
| **Risco** | Dependências comprometidas (pip packages) |
| **Nível** | 🟡 Médio |
| **Mitigação** | `pip-audit` no CI, versões mínimas fixadas em requirements.txt |
| **Implementação** | `.github/workflows/ci.yml` — step pip-audit |
| **Ação** | CI falha se vulnerabilidades conhecidas são detectadas |

### LLM06: Sensitive Information Disclosure

| Aspecto | Detalhe |
|---------|---------|
| **Risco** | LLM expõe PII, chaves de API ou dados sensíveis |
| **Nível** | 🟡 Médio |
| **Mitigação** | PII detection + anonymization (Presidio), env vars para secrets |
| **Implementação** | `src/security/pii_detection.py`, `src/security/guardrails.py` |
| **Entidades** | CPF, email, telefone, cartão de crédito, IP |
| **Ação** | PII detectado é substituído por `<PII_REDACTED>` |

### LLM07: Insecure Plugin Design

| Aspecto | Detalhe |
|---------|---------|
| **Risco** | Tools do agente executam ações não autorizadas |
| **Nível** | 🟢 Baixo |
| **Mitigação** | Tools são read-only, restritas a dados financeiros da NVIDIA |
| **Implementação** | `src/agent/tools.py` — 4 tools controladas |
| **Tools** | query_stock_data (read), predict (inference), get_metrics (read), search_documents (read) |
| **Ação** | Nenhuma tool permite escrita ou acesso ao sistema de arquivos |

### LLM08: Excessive Agency

| Aspecto | Detalhe |
|---------|---------|
| **Risco** | Agente executa ações além do escopo permitido |
| **Nível** | 🟢 Baixo |
| **Mitigação** | Max iterations (8), topic validation, tools restritas |
| **Implementação** | `src/agent/react_agent.py`, `src/security/guardrails.py` |
| **Ação** | Agente limitado a consultas e previsões (sem ações externas) |

### LLM09: Overreliance

| Aspecto | Detalhe |
|---------|---------|
| **Risco** | Usuário confia cegamente nas previsões para investir |
| **Nível** | 🔴 Alto |
| **Mitigação** | Disclaimers obrigatórios, intervalos de confiança, Model Card |
| **Implementação** | `src/security/guardrails.py` — `OutputGuardrail._is_prediction_query()`, `docs/MODEL_CARD.md` |
| **Ação** | Disclaimer de risco adicionado automaticamente a respostas sobre previsões |

### LLM10: Model Theft

| Aspecto | Detalhe |
|---------|---------|
| **Risco** | Acesso não autorizado aos pesos do modelo |
| **Nível** | 🟢 Baixo |
| **Mitigação** | Modelo servido via API (pesos não expostos), Docker isolation |
| **Implementação** | `src/api/`, `Dockerfile.api` |
| **Ação** | API retorna apenas previsões, nunca pesos ou arquitetura |

---

## Resumo de Cobertura

| OWASP Risk | Nível | Status |
|------------|-------|--------|
| LLM01: Prompt Injection | 🟡 Médio | ✅ Mitigado |
| LLM02: Insecure Output | 🟡 Médio | ✅ Mitigado |
| LLM03: Data Poisoning | 🟢 Baixo | ✅ Mitigado |
| LLM04: Model DoS | 🟡 Médio | ✅ Mitigado |
| LLM05: Supply Chain | 🟡 Médio | ✅ Mitigado |
| LLM06: Info Disclosure | 🟡 Médio | ✅ Mitigado |
| LLM07: Insecure Plugin | 🟢 Baixo | ✅ Mitigado |
| LLM08: Excessive Agency | 🟢 Baixo | ✅ Mitigado |
| LLM09: Overreliance | 🔴 Alto | ✅ Mitigado |
| LLM10: Model Theft | 🟢 Baixo | ✅ Mitigado |

**Cobertura: 10/10 riscos mapeados e mitigados.**
