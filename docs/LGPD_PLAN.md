# LGPD Compliance Plan

## General Data Protection Law (Lei 13.709/2018)

This document describes how the NVIDIA MLOps Platform complies with the
Brazilian General Data Protection Law (LGPD).

---

## 1. Personal Data Processed

### 1.1 Data Collected by the System
| Data | Classification | Source | Necessity |
|------|---------------|--------|-----------|
| NVIDIA stock data | Public data | Yahoo Finance | Core functionality |
| User queries | Personal data (potential) | User input | Agent functionality |
| API access logs | Personal data (IP) | HTTP requests | Monitoring/security |

### 1.2 Data NOT Collected
- ❌ User full name
- ❌ CPF, RG, or personal documents
- ❌ Personal banking or financial data
- ❌ Sensitive data (health, religion, biometrics)
- ❌ Data from minors

## 2. Legal Bases (Art. 7 of LGPD)

| Processing | Legal Basis | Justification |
|------------|------------|---------------|
| Market data | Art. 7, III — Public data | Public financial data from NASDAQ |
| Agent queries | Art. 7, I — Consent | User submits query voluntarily |
| Access logs | Art. 7, IX — Legitimate interest | System security and monitoring |

## 3. LGPD Principles Applied

### 3.1 Purpose (Art. 6, I)
- Market data: exclusively for NVIDIA price prediction
- Queries: processed by the agent and discarded (not permanently stored)

### 3.2 Adequacy (Art. 6, II)
- Only necessary data is collected
- Model trained exclusively with public financial data

### 3.3 Necessity (Art. 6, III)
- Data minimization: only Open, High, Low, Close, Volume
- Logs retained for a limited period (30 days)

### 3.4 Free Access (Art. 6, IV)
- API documented with Swagger UI
- Historical data accessible via /data endpoint

### 3.5 Data Quality (Art. 6, V)
- Data validated in the ETL pipeline
- Drift detection continuously monitors quality

### 3.6 Transparency (Art. 6, VI)
- Model Card documents the model and its limitations
- System Card documents the complete architecture
- API returns metadata about predictions

### 3.7 Security (Art. 6, VII)
- Guardrails protect against prompt injection
- PII detection (Presidio) anonymizes personal data
- OWASP LLM Top 10 applied
- Docker containers isolate components

### 3.8 Prevention (Art. 6, VIII)
- Continuous monitoring with Prometheus/Grafana
- Drift detection prevents model degradation
- Rate limiting on the API

### 3.9 Non-Discrimination (Art. 6, IX)
- Model does not use demographic data
- Predictions based solely on market data

### 3.10 Accountability (Art. 6, X)
- MLflow records all experiments (auditability)
- Telemetry logs track LLM usage
- Code and model versioning

## 4. Data Subject Rights (Art. 18)

| Right | Implementation |
|-------|---------------|
| **Confirmation** (Art. 18, I) | API returns whether data is being processed |
| **Access** (Art. 18, II) | /data endpoint provides access to the data |
| **Correction** (Art. 18, III) | Market data corrected via Yahoo Finance re-fetch |
| **Deletion** (Art. 18, VI) | Queries are not persisted by default |
| **Information** (Art. 18, VII) | Complete documentation available |
| **Revocation** (Art. 18, IX) | User can stop using the service |

## 5. PII Detection and Protection

### 5.1 Mechanism
- **Microsoft Presidio** for PII entity detection
- **Regex fallback** when Presidio is not available

### 5.2 Detected Entities
- Brazilian CPF (with digit validation)
- Email
- Phone number
- Credit card numbers
- IP addresses

### 5.3 Actions
- PII detected in inputs is flagged
- PII in outputs is automatically anonymized
- Logs of detected PII are recorded (without the data itself)

## 6. Technical Security Measures

| Measure | Implementation |
|---------|---------------|
| Encryption in transit | HTTPS via nginx |
| Isolation | Docker containers |
| Access control | CORS configuration |
| Attack detection | Prompt injection guardrails |
| Monitoring | Prometheus + Grafana |
| Auditability | MLflow + Langfuse telemetry |
| Anonymization | Presidio PII detection |

## 7. Data Protection Officer (DPO)

For data protection inquiries:
- **Responsible**: Define DPO before production deployment
- **Contact**: Define communication channel

## 8. Incident Response Plan (Art. 48)

1. **Detection**: Automatic monitoring via Prometheus/alerts
2. **Containment**: Deactivation of affected service
3. **Notification**: Report to ANPD within 72 hours (when applicable)
4. **Remediation**: Patch and re-deploy
5. **Documentation**: Record incident and actions taken

---

*Document updated in 2025. Review recommended every 6 months or when significant system changes occur.*
