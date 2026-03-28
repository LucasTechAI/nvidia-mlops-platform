"""PII detection and anonymization with Presidio.

Detects and anonymizes personally identifiable information in
user inputs and agent outputs.

Entities detected:
    - PERSON (names)
    - EMAIL_ADDRESS
    - PHONE_NUMBER
    - BR_CPF (Brazilian CPF)
    - CREDIT_CARD
    - IP_ADDRESS

Referência: Microsoft Presidio — Data protection and anonymization
            https://microsoft.github.io/presidio/
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class PIIDetector:
    """PII detection and anonymization.

    Uses Microsoft Presidio when available, falls back to
    regex-based detection for common PII patterns.

    Attributes:
        analyzer: Presidio AnalyzerEngine (or None).
        anonymizer: Presidio AnonymizerEngine (or None).
    """

    # Supported entity types
    ENTITY_TYPES = [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "BR_CPF",
        "CREDIT_CARD",
        "IP_ADDRESS",
    ]

    def __init__(self, language: str = "pt"):
        """Initialize PII detector.

        Args:
            language: Default language for analysis ('pt', 'en').
        """
        self.language = language
        self._analyzer = None
        self._anonymizer = None
        self._presidio_available = False

        self._init_presidio()

    def _init_presidio(self) -> None:
        """Try to initialize Presidio engines."""
        try:
            from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
            from presidio_anonymizer import AnonymizerEngine

            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()

            # Add Brazilian CPF recognizer
            cpf_pattern = Pattern(
                name="cpf_pattern",
                regex=r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b",
                score=0.85,
            )
            cpf_recognizer = PatternRecognizer(
                supported_entity="BR_CPF",
                patterns=[cpf_pattern],
                supported_language="pt",
            )
            self._analyzer.registry.add_recognizer(cpf_recognizer)

            self._presidio_available = True
            logger.info("Presidio PII detector initialized")

        except ImportError:
            logger.info("Presidio not installed. Using regex-based PII detection.")
        except Exception as e:
            logger.warning("Presidio init failed: %s. Using regex fallback.", str(e))

    def detect(self, text: str) -> list[dict]:
        """Detect PII entities in text.

        Args:
            text: Text to analyze.

        Returns:
            List of detected entities with type, start, end, score.
        """
        if self._presidio_available and self._analyzer:
            return self._detect_presidio(text)
        return self._detect_regex(text)

    def anonymize(self, text: str, entities: Optional[list[dict]] = None) -> str:
        """Anonymize detected PII in text.

        Args:
            text: Text to anonymize.
            entities: Pre-detected entities (if None, runs detection first).

        Returns:
            Anonymized text with PII replaced by placeholders.
        """
        if entities is None:
            entities = self.detect(text)

        if not entities:
            return text

        if self._presidio_available and self._anonymizer:
            return self._anonymize_presidio(text)

        return self._anonymize_regex(text, entities)

    def detect_and_anonymize(self, text: str) -> dict:
        """Detect PII and return anonymized text with metadata.

        Args:
            text: Text to process.

        Returns:
            Dictionary with anonymized_text, pii_found, entities.
        """
        entities = self.detect(text)

        if not entities:
            return {
                "anonymized_text": text,
                "pii_found": False,
                "entities": [],
            }

        anonymized = self.anonymize(text, entities)

        return {
            "anonymized_text": anonymized,
            "pii_found": True,
            "entities": entities,
            "n_entities": len(entities),
        }

    def _detect_presidio(self, text: str) -> list[dict]:
        """Detect PII using Presidio."""
        try:
            results = self._analyzer.analyze(
                text=text,
                entities=self.ENTITY_TYPES,
                language=self.language,
            )

            return [
                {
                    "type": r.entity_type,
                    "start": r.start,
                    "end": r.end,
                    "score": round(r.score, 2),
                    "text": text[r.start : r.end],
                }
                for r in results
            ]
        except Exception as e:
            logger.warning("Presidio detection failed: %s. Falling back to regex.", str(e))
            return self._detect_regex(text)

    def _anonymize_presidio(self, text: str) -> str:
        """Anonymize using Presidio."""
        try:
            from presidio_anonymizer.entities import OperatorConfig

            results = self._analyzer.analyze(
                text=text,
                entities=self.ENTITY_TYPES,
                language=self.language,
            )

            if not results:
                return text

            anonymized = self._anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<PII_REDACTED>"})},
            )

            return anonymized.text
        except Exception as e:
            logger.warning("Presidio anonymization failed: %s. Falling back to regex.", str(e))
            entities = self._detect_regex(text)
            return self._anonymize_regex(text, entities)

    def _detect_regex(self, text: str) -> list[dict]:
        """Fallback regex-based PII detection."""
        entities = []

        # Email
        for match in re.finditer(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text):
            entities.append(
                {
                    "type": "EMAIL_ADDRESS",
                    "start": match.start(),
                    "end": match.end(),
                    "score": 0.95,
                    "text": match.group(),
                }
            )

        # Brazilian CPF
        for match in re.finditer(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b", text):
            candidate = match.group().replace(".", "").replace("-", "")
            if len(candidate) == 11 and self._validate_cpf(candidate):
                entities.append(
                    {
                        "type": "BR_CPF",
                        "start": match.start(),
                        "end": match.end(),
                        "score": 0.85,
                        "text": match.group(),
                    }
                )

        # Phone numbers (Brazilian and international)
        for match in re.finditer(r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}[-.\s]?\d{4}\b", text):
            entities.append(
                {
                    "type": "PHONE_NUMBER",
                    "start": match.start(),
                    "end": match.end(),
                    "score": 0.75,
                    "text": match.group(),
                }
            )

        # Credit card numbers
        for match in re.finditer(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", text):
            entities.append(
                {
                    "type": "CREDIT_CARD",
                    "start": match.start(),
                    "end": match.end(),
                    "score": 0.80,
                    "text": match.group(),
                }
            )

        # IP addresses
        for match in re.finditer(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text):
            entities.append(
                {
                    "type": "IP_ADDRESS",
                    "start": match.start(),
                    "end": match.end(),
                    "score": 0.70,
                    "text": match.group(),
                }
            )

        return entities

    def _anonymize_regex(self, text: str, entities: list[dict]) -> str:
        """Fallback regex-based anonymization."""
        # Sort entities by start position (reverse) to replace from end
        sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)

        result = text
        for entity in sorted_entities:
            placeholder = f"<{entity['type']}>"
            result = result[: entity["start"]] + placeholder + result[entity["end"] :]

        return result

    @staticmethod
    def _validate_cpf(cpf: str) -> bool:
        """Validate Brazilian CPF number."""
        if len(cpf) != 11 or cpf == cpf[0] * 11:
            return False

        # First digit
        total = sum(int(cpf[i]) * (10 - i) for i in range(9))
        digit1 = 11 - (total % 11)
        digit1 = 0 if digit1 >= 10 else digit1

        if int(cpf[9]) != digit1:
            return False

        # Second digit
        total = sum(int(cpf[i]) * (11 - i) for i in range(10))
        digit2 = 11 - (total % 11)
        digit2 = 0 if digit2 >= 10 else digit2

        return int(cpf[10]) == digit2


# Convenience singleton
_detector: Optional[PIIDetector] = None


def get_pii_detector() -> PIIDetector:
    """Get or create the global PII detector."""
    global _detector
    if _detector is None:
        _detector = PIIDetector()
    return _detector


def detect_pii(text: str) -> list[dict]:
    """Detect PII in text using the global detector."""
    return get_pii_detector().detect(text)


def anonymize_pii(text: str) -> str:
    """Anonymize PII in text using the global detector."""
    return get_pii_detector().anonymize(text)
