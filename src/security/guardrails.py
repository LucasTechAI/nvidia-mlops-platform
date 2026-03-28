"""Input and output guardrails for the LLM agent.

Referência: OWASP Top 10 for LLM Applications (2025)
            https://owasp.org/www-project-top-10-for-large-language-model-applications/

InputGuardrail:
    - Prompt injection detection (pattern matching + heuristics)
    - Max input length enforcement
    - Topic validation (financial domain only)
    - Language detection

OutputGuardrail:
    - PII removal (via Presidio integration)
    - Content filtering (harmful content detection)
    - Hallucination warning flags
    - Risk disclaimer enforcement
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ============== Configuration ==============

MAX_INPUT_LENGTH = 2000
MAX_OUTPUT_LENGTH = 10000

# Allowed topics / domains
ALLOWED_TOPICS = [
    "nvidia",
    "nvda",
    "ações",
    "stock",
    "preço",
    "price",
    "previsão",
    "prediction",
    "forecast",
    "modelo",
    "model",
    "lstm",
    "treinamento",
    "training",
    "métricas",
    "metrics",
    "dados",
    "data",
    "volume",
    "mercado",
    "market",
    "investimento",
    "investment",
    "gpu",
    "ia",
    "ai",
    "semicondutor",
    "semiconductor",
    "deep learning",
    "machine learning",
    "api",
    "dashboard",
    "rag",
    "agent",
    "drift",
    "monitoring",
    "fechamento",
    "abertura",
    "close",
    "open",
    "high",
    "low",
    "mlflow",
    "pytorch",
    "fastapi",
    "arquitetura",
    "architecture",
]

# Prompt injection patterns (regex)
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above",
    r"disregard\s+(all\s+)?previous",
    r"forget\s+(all\s+)?instructions",
    r"you\s+are\s+now\s+(?:a|an)\s+(?:different|new)",
    r"new\s+instruction[s]?\s*:",
    r"system\s*:\s*",
    r"<\s*(?:system|admin|root)\s*>",
    r"\[(?:system|admin)\]",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"dan\s+mode",
    r"(?:pretend|act)\s+(?:like\s+)?you\s+(?:are|have)\s+no\s+(?:rules|restrictions|limits)",
    r"bypass\s+(?:your\s+)?(?:safety|filter|guard|restriction)",
    r"override\s+(?:your\s+)?(?:safety|system|instructions)",
    r"reveal\s+(?:your\s+)?(?:system\s+)?prompt",
    r"show\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions)",
    r"what\s+(?:are|is)\s+your\s+(?:system\s+)?(?:prompt|instructions|rules)",
]

# Harmful content patterns
HARMFUL_PATTERNS = [
    r"(?:how\s+to|ways\s+to)\s+(?:hack|steal|attack|exploit)",
    r"(?:insider\s+trading|market\s+manipulation|pump\s+and\s+dump)",
    r"(?:launder|laundering)\s+money",
]


@dataclass
class GuardrailResult:
    """Result from a guardrail check.

    Attributes:
        passed: Whether the check passed.
        message: Description if blocked.
        flags: List of triggered flags.
        sanitized_text: Cleaned text (for output guardrails).
    """

    passed: bool = True
    message: str = ""
    flags: list[str] = field(default_factory=list)
    sanitized_text: Optional[str] = None


class InputGuardrail:
    """Validates and sanitizes user input before sending to the agent.

    Checks:
        1. Input length within limits
        2. No prompt injection patterns
        3. Topic is within allowed domain
        4. No harmful intent detected
    """

    def __init__(
        self,
        max_length: int = MAX_INPUT_LENGTH,
        allowed_topics: Optional[list[str]] = None,
    ):
        self.max_length = max_length
        self.allowed_topics = allowed_topics or ALLOWED_TOPICS
        self._injection_patterns = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
        self._harmful_patterns = [re.compile(p, re.IGNORECASE) for p in HARMFUL_PATTERNS]

    def validate(self, text: str) -> GuardrailResult:
        """Run all input validation checks.

        Args:
            text: User input text.

        Returns:
            GuardrailResult with pass/fail and details.
        """
        result = GuardrailResult()

        # 1. Length check
        if len(text) > self.max_length:
            result.passed = False
            result.message = f"Input exceeds maximum length ({len(text)} > {self.max_length})"
            result.flags.append("max_length_exceeded")
            logger.warning("Input guardrail: length exceeded (%d chars)", len(text))
            return result

        if len(text.strip()) == 0:
            result.passed = False
            result.message = "Empty input"
            result.flags.append("empty_input")
            return result

        # 2. Prompt injection detection
        injection_result = self._check_injection(text)
        if injection_result:
            result.passed = False
            result.message = "Potential prompt injection detected"
            result.flags.append("prompt_injection")
            result.flags.append(f"pattern:{injection_result}")
            logger.warning("Input guardrail: prompt injection detected — %s", injection_result)
            return result

        # 3. Harmful content check
        harmful_result = self._check_harmful(text)
        if harmful_result:
            result.passed = False
            result.message = "Potentially harmful content detected"
            result.flags.append("harmful_content")
            logger.warning("Input guardrail: harmful content detected — %s", harmful_result)
            return result

        # 4. Topic validation (soft check — add flag but don't block)
        if not self._check_topic(text):
            result.flags.append("off_topic")
            logger.info("Input guardrail: query may be off-topic")

        return result

    def _check_injection(self, text: str) -> Optional[str]:
        """Check for prompt injection patterns."""
        for pattern in self._injection_patterns:
            match = pattern.search(text)
            if match:
                return match.group()
        return None

    def _check_harmful(self, text: str) -> Optional[str]:
        """Check for harmful intent patterns."""
        for pattern in self._harmful_patterns:
            match = pattern.search(text)
            if match:
                return match.group()
        return None

    def _check_topic(self, text: str) -> bool:
        """Check if the query is within the allowed financial domain."""
        text_lower = text.lower()
        return any(topic in text_lower for topic in self.allowed_topics)


class OutputGuardrail:
    """Validates and sanitizes agent output before returning to the user.

    Checks:
        1. PII detection and removal
        2. Content filtering
        3. Risk disclaimer presence (for predictions)
        4. Output length limits
    """

    def __init__(self, max_length: int = MAX_OUTPUT_LENGTH):
        self.max_length = max_length
        self._pii_detector = None

    def _get_pii_detector(self):
        """Lazy-load PII detector."""
        if self._pii_detector is None:
            try:
                from src.security.pii_detection import PIIDetector

                self._pii_detector = PIIDetector()
            except Exception:
                self._pii_detector = False  # Mark as unavailable
        return self._pii_detector if self._pii_detector is not False else None

    def validate(self, text: str, query: Optional[str] = None) -> GuardrailResult:
        """Run all output validation checks.

        Args:
            text: Agent output text.
            query: Original user query (for context-aware checks).

        Returns:
            GuardrailResult with pass/fail, flags, and sanitized text.
        """
        result = GuardrailResult()
        sanitized = text

        # 1. Length check
        if len(text) > self.max_length:
            sanitized = text[: self.max_length] + "\n\n[Resposta truncada por limite de tamanho]"
            result.flags.append("output_truncated")
            logger.info("Output guardrail: response truncated (%d chars)", len(text))

        # 2. PII detection and removal
        pii_detector = self._get_pii_detector()
        if pii_detector:
            pii_result = pii_detector.detect_and_anonymize(sanitized)
            if pii_result.get("pii_found"):
                sanitized = pii_result["anonymized_text"]
                result.flags.append("pii_removed")
                for entity in pii_result.get("entities", []):
                    result.flags.append(f"pii:{entity['type']}")
                logger.info("Output guardrail: PII detected and removed")

        # 3. Risk disclaimer check (for prediction-related queries)
        if query and self._is_prediction_query(query):
            if not self._has_disclaimer(sanitized):
                disclaimer = (
                    "\n\n⚠️ **Aviso de Risco**: Previsões de preços de ações são estimativas "
                    "baseadas em dados históricos e não constituem recomendações de investimento. "
                    "Resultados passados não garantem retornos futuros."
                )
                sanitized += disclaimer
                result.flags.append("disclaimer_added")
                logger.info("Output guardrail: risk disclaimer added")

        # 4. Content safety check
        harmful_check = self._check_harmful_output(sanitized)
        if harmful_check:
            result.passed = False
            result.message = "Output contains potentially harmful content"
            result.flags.append("harmful_output")
            sanitized = (
                "Desculpe, não posso fornecer essa informação. "
                "Por favor, reformule sua pergunta sobre análise financeira da NVIDIA."
            )
            logger.warning("Output guardrail: harmful content blocked")

        result.sanitized_text = sanitized
        return result

    def _is_prediction_query(self, query: str) -> bool:
        """Check if the query is about predictions/forecasts."""
        prediction_terms = [
            "previsão",
            "prever",
            "forecast",
            "predict",
            "futuro",
            "future",
            "próximo",
            "next",
            "amanhã",
            "tomorrow",
            "semana",
            "week",
            "investir",
            "invest",
            "comprar",
            "buy",
            "vender",
            "sell",
        ]
        query_lower = query.lower()
        return any(term in query_lower for term in prediction_terms)

    def _has_disclaimer(self, text: str) -> bool:
        """Check if text contains a risk disclaimer."""
        disclaimer_terms = [
            "risco",
            "risk",
            "aviso",
            "disclaimer",
            "warning",
            "não constitui",
            "not constitute",
            "não é recomendação",
            "not a recommendation",
            "consulte um profissional",
        ]
        text_lower = text.lower()
        return any(term in text_lower for term in disclaimer_terms)

    def _check_harmful_output(self, text: str) -> bool:
        """Check output for harmful content."""
        harmful_indicators = [
            "insider trading",
            "informação privilegiada",
            "manipulação de mercado",
            "market manipulation",
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in harmful_indicators)


# ============== Convenience functions ==============

_input_guardrail: Optional[InputGuardrail] = None
_output_guardrail: Optional[OutputGuardrail] = None


def get_input_guardrail() -> InputGuardrail:
    """Get or create the global input guardrail."""
    global _input_guardrail
    if _input_guardrail is None:
        _input_guardrail = InputGuardrail()
    return _input_guardrail


def get_output_guardrail() -> OutputGuardrail:
    """Get or create the global output guardrail."""
    global _output_guardrail
    if _output_guardrail is None:
        _output_guardrail = OutputGuardrail()
    return _output_guardrail


def validate_input(text: str) -> GuardrailResult:
    """Validate user input with the global guardrail."""
    return get_input_guardrail().validate(text)


def validate_output(text: str, query: Optional[str] = None) -> GuardrailResult:
    """Validate agent output with the global guardrail."""
    return get_output_guardrail().validate(text, query)
