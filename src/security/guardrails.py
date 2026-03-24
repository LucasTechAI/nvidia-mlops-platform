"""Input and output guardrails for the LLM agent.

Referência: OWASP Top 10 for LLM Applications (2025)
            https://owasp.org/www-project-top-10-for-large-language-model-applications/

InputGuardrail:
    - Prompt injection detection
    - Max input length
    - Topic validation

OutputGuardrail:
    - PII removal (Presidio)
    - Content filtering
"""
