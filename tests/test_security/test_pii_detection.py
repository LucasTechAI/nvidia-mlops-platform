"""Tests for the PII detection module."""

from src.security.pii_detection import PIIDetector, anonymize_pii, detect_pii


class TestPIIDetector:
    """Tests for PIIDetector regex fallback."""

    def setup_method(self):
        self.detector = PIIDetector()

    def test_detect_email(self):
        """Should detect email addresses."""
        entities = self.detector.detect("Meu email é user@example.com")
        types = [e["type"] for e in entities]
        assert "EMAIL_ADDRESS" in types

    def test_detect_cpf_formatted(self):
        """Should detect formatted CPF (valid)."""
        # Using a valid test CPF
        entities = self.detector.detect("CPF: 529.982.247-25")
        types = [e["type"] for e in entities]
        assert "BR_CPF" in types

    def test_reject_invalid_cpf(self):
        """Should not flag invalid CPF numbers."""
        # All same digits = invalid
        entities = self.detector.detect("Número: 111.111.111-11")
        types = [e["type"] for e in entities]
        assert "BR_CPF" not in types

    def test_detect_credit_card(self):
        """Should detect credit card numbers."""
        entities = self.detector.detect("Cartão: 4532-1234-5678-9012")
        types = [e["type"] for e in entities]
        assert "CREDIT_CARD" in types

    def test_detect_phone(self):
        """Should detect phone numbers."""
        entities = self.detector.detect("Tel: (11) 98765-4321")
        types = [e["type"] for e in entities]
        assert "PHONE_NUMBER" in types

    def test_detect_ip(self):
        """Should detect IP addresses."""
        entities = self.detector.detect("Servidor: 192.168.1.100")
        types = [e["type"] for e in entities]
        assert "IP_ADDRESS" in types

    def test_no_pii_in_clean_text(self):
        """Clean financial text should have no PII."""
        entities = self.detector.detect("O preço da NVIDIA fechou em $130.50 hoje.")
        # Should be empty or only very low-confidence matches
        high_confidence = [e for e in entities if e["score"] > 0.7]
        assert len(high_confidence) == 0

    def test_anonymize_email(self):
        """Should anonymize email addresses."""
        result = self.detector.anonymize("Contato: user@example.com")
        assert "user@example.com" not in result
        assert "<EMAIL_ADDRESS>" in result

    def test_anonymize_multiple_pii(self):
        """Should anonymize multiple PII entities."""
        text = "Email: a@b.com, IP: 192.168.1.1"
        result = self.detector.anonymize(text)
        assert "a@b.com" not in result
        assert "192.168.1.1" not in result

    def test_detect_and_anonymize(self):
        """Should return both detection results and anonymized text."""
        result = self.detector.detect_and_anonymize("Email: user@test.com")
        assert result["pii_found"] is True
        assert result["n_entities"] >= 1
        assert "user@test.com" not in result["anonymized_text"]

    def test_detect_and_anonymize_clean(self):
        """Clean text should return pii_found=False."""
        result = self.detector.detect_and_anonymize("Preço da NVIDIA hoje.")
        assert result["pii_found"] is False

    def test_cpf_validation_logic(self):
        """Test CPF validation algorithm."""
        # Valid CPFs
        assert PIIDetector._validate_cpf("52998224725") is True
        # Invalid: all same digits
        assert PIIDetector._validate_cpf("11111111111") is False
        # Invalid: wrong check digits
        assert PIIDetector._validate_cpf("12345678900") is False


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_detect_pii(self):
        entities = detect_pii("email: user@test.com")
        assert len(entities) > 0

    def test_anonymize_pii(self):
        result = anonymize_pii("email: user@test.com")
        assert "user@test.com" not in result
