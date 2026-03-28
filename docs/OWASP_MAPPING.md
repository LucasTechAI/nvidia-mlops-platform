# OWASP Top 10 for LLM Applications — Mapping

## Reference

[OWASP Top 10 for Large Language Model Applications (2025)](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

---

## Risk Mapping and Mitigations

### LLM01: Prompt Injection

| Aspect | Detail |
|--------|--------|
| **Risk** | Attacker manipulates the agent's prompt to execute unauthorized actions |
| **Level** | 🟡 Medium |
| **Mitigation** | `InputGuardrail` with 16 regex patterns for prompt injection |
| **Implementation** | `src/security/guardrails.py` — `InputGuardrail._check_injection()` |
| **Detected patterns** | "ignore previous instructions", "system:", "jailbreak", "DAN mode", "bypass safety", "reveal system prompt", among others |
| **Action** | Input blocked before reaching the LLM |

### LLM02: Insecure Output Handling

| Aspect | Detail |
|--------|--------|
| **Risk** | LLM output contains harmful content, PII, or executable instructions |
| **Level** | 🟡 Medium |
| **Mitigation** | `OutputGuardrail` with PII detection, content filtering, and disclaimers |
| **Implementation** | `src/security/guardrails.py` — `OutputGuardrail.validate()` |
| **Actions** | PII removed (Presidio), harmful content blocked, disclaimers added |

### LLM03: Training Data Poisoning

| Aspect | Detail |
|--------|--------|
| **Risk** | Compromised training data affects predictions |
| **Level** | 🟢 Low |
| **Mitigation** | Data from trusted source (Yahoo Finance), drift detection (PSI) |
| **Implementation** | `src/monitoring/drift.py`, `src/etl/preprocessing.py` |
| **Action** | PSI > 0.2 triggers retraining alert |

### LLM04: Model Denial of Service

| Aspect | Detail |
|--------|--------|
| **Risk** | Extremely long or complex inputs overload the system |
| **Level** | 🟡 Medium |
| **Mitigation** | Max input length (2000 chars), max iterations (8), timeouts |
| **Implementation** | `src/security/guardrails.py` — `MAX_INPUT_LENGTH`, `src/agent/react_agent.py` — `max_iterations` |
| **Action** | Large inputs rejected, agent loops limited |

### LLM05: Supply Chain Vulnerabilities

| Aspect | Detail |
|--------|--------|
| **Risk** | Compromised dependencies (pip packages) |
| **Level** | 🟡 Medium |
| **Mitigation** | `pip-audit` in CI, minimum versions pinned in requirements.txt |
| **Implementation** | `.github/workflows/ci.yml` — pip-audit step |
| **Action** | CI fails if known vulnerabilities are detected |

### LLM06: Sensitive Information Disclosure

| Aspect | Detail |
|--------|--------|
| **Risk** | LLM exposes PII, API keys, or sensitive data |
| **Level** | 🟡 Medium |
| **Mitigation** | PII detection + anonymization (Presidio), env vars for secrets |
| **Implementation** | `src/security/pii_detection.py`, `src/security/guardrails.py` |
| **Entities** | CPF, email, phone, credit card, IP |
| **Action** | Detected PII replaced with `<PII_REDACTED>` |

### LLM07: Insecure Plugin Design

| Aspect | Detail |
|--------|--------|
| **Risk** | Agent tools execute unauthorized actions |
| **Level** | 🟢 Low |
| **Mitigation** | Tools are read-only, restricted to NVIDIA financial data |
| **Implementation** | `src/agent/tools.py` — 4 controlled tools |
| **Tools** | query_stock_data (read), predict (inference), get_metrics (read), search_documents (read) |
| **Action** | No tool allows writing or file system access |

### LLM08: Excessive Agency

| Aspect | Detail |
|--------|--------|
| **Risk** | Agent executes actions beyond the permitted scope |
| **Level** | 🟢 Low |
| **Mitigation** | Max iterations (8), topic validation, restricted tools |
| **Implementation** | `src/agent/react_agent.py`, `src/security/guardrails.py` |
| **Action** | Agent limited to queries and predictions (no external actions) |

### LLM09: Overreliance

| Aspect | Detail |
|--------|--------|
| **Risk** | User blindly trusts predictions for investment |
| **Level** | 🔴 High |
| **Mitigation** | Mandatory disclaimers, confidence intervals, Model Card |
| **Implementation** | `src/security/guardrails.py` — `OutputGuardrail._is_prediction_query()`, `docs/MODEL_CARD.md` |
| **Action** | Risk disclaimer automatically added to prediction-related responses |

### LLM10: Model Theft

| Aspect | Detail |
|--------|--------|
| **Risk** | Unauthorized access to model weights |
| **Level** | 🟢 Low |
| **Mitigation** | Model served via API (weights not exposed), Docker isolation |
| **Implementation** | `src/api/`, `Dockerfile.api` |
| **Action** | API returns only predictions, never weights or architecture |

---

## Coverage Summary

| OWASP Risk | Level | Status |
|------------|-------|--------|
| LLM01: Prompt Injection | 🟡 Medium | ✅ Mitigated |
| LLM02: Insecure Output | 🟡 Medium | ✅ Mitigated |
| LLM03: Data Poisoning | 🟢 Low | ✅ Mitigated |
| LLM04: Model DoS | 🟡 Medium | ✅ Mitigated |
| LLM05: Supply Chain | 🟡 Medium | ✅ Mitigated |
| LLM06: Info Disclosure | 🟡 Medium | ✅ Mitigated |
| LLM07: Insecure Plugin | 🟢 Low | ✅ Mitigated |
| LLM08: Excessive Agency | 🟢 Low | ✅ Mitigated |
| LLM09: Overreliance | 🔴 High | ✅ Mitigated |
| LLM10: Model Theft | 🟢 Low | ✅ Mitigated |

**Coverage: 10/10 risks mapped and mitigated.**
