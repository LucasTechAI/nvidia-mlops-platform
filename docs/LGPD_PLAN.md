# Plano de Adequação à LGPD

## Lei Geral de Proteção de Dados (Lei 13.709/2018)

Este documento descreve como o sistema NVIDIA MLOps Platform se adequa à
Lei Geral de Proteção de Dados Pessoais (LGPD) brasileira.

---

## 1. Dados Pessoais Tratados

### 1.1 Dados Coletados pelo Sistema
| Dado | Classificação | Fonte | Necessidade |
|------|--------------|-------|-------------|
| Dados de ações NVIDIA | Dados públicos | Yahoo Finance | Funcionalidade core |
| Queries do usuário | Dado pessoal (potencial) | Input do usuário | Funcionalidade do agente |
| Logs de acesso à API | Dado pessoal (IP) | Requisições HTTP | Monitoramento/segurança |

### 1.2 Dados NÃO Coletados
- ❌ Nome completo do usuário
- ❌ CPF, RG ou documentos pessoais
- ❌ Dados bancários ou financeiros pessoais
- ❌ Dados sensíveis (saúde, religião, biometria)
- ❌ Dados de menores de idade

## 2. Bases Legais (Art. 7º da LGPD)

| Tratamento | Base Legal | Justificativa |
|------------|-----------|---------------|
| Dados de mercado | Art. 7º, III — Dados públicos | Dados financeiros públicos da NASDAQ |
| Queries do agente | Art. 7º, I — Consentimento | Usuário envia query voluntariamente |
| Logs de acesso | Art. 7º, IX — Legítimo interesse | Segurança e monitoramento do sistema |

## 3. Princípios da LGPD Aplicados

### 3.1 Finalidade (Art. 6º, I)
- Dados de mercado: exclusivamente para previsão de preços da NVIDIA
- Queries: processadas pelo agente e descartadas (não armazenadas permanentemente)

### 3.2 Adequação (Art. 6º, II)
- Apenas dados necessários são coletados
- Modelo treinado apenas com dados financeiros públicos

### 3.3 Necessidade (Art. 6º, III)
- Minimização de dados: apenas Open, High, Low, Close, Volume
- Logs retidos por período limitado (30 dias)

### 3.4 Livre Acesso (Art. 6º, IV)
- API documentada com Swagger UI
- Dados históricos acessíveis via endpoint /data

### 3.5 Qualidade dos Dados (Art. 6º, V)
- Dados validados no pipeline ETL
- Drift detection monitora qualidade continuamente

### 3.6 Transparência (Art. 6º, VI)
- Model Card documenta o modelo e suas limitações
- System Card documenta a arquitetura completa
- API retorna metadados sobre previsões

### 3.7 Segurança (Art. 6º, VII)
- Guardrails protegem contra prompt injection
- PII detection (Presidio) anonimiza dados pessoais
- OWASP LLM Top 10 aplicado
- Docker containers isolam componentes

### 3.8 Prevenção (Art. 6º, VIII)
- Monitoramento contínuo com Prometheus/Grafana
- Drift detection previne degradação do modelo
- Rate limiting na API

### 3.9 Não Discriminação (Art. 6º, IX)
- Modelo não usa dados demográficos
- Predições baseadas apenas em dados de mercado

### 3.10 Responsabilização (Art. 6º, X)
- MLflow registra todos os experimentos (auditabilidade)
- Logs de telemetria rastreiam uso do LLM
- Versionamento de código e modelos

## 4. Direitos dos Titulares (Art. 18)

| Direito | Implementação |
|---------|---------------|
| **Confirmação** (Art. 18, I) | API retorna se dados estão sendo processados |
| **Acesso** (Art. 18, II) | Endpoint /data fornece acesso aos dados |
| **Correção** (Art. 18, III) | Dados de mercado corrigidos via re-fetch do Yahoo Finance |
| **Eliminação** (Art. 18, VI) | Queries não são persistidas por padrão |
| **Informação** (Art. 18, VII) | Documentação completa disponível |
| **Revogação** (Art. 18, IX) | Usuário pode parar de usar o serviço |

## 5. Detecção e Proteção de PII

### 5.1 Mecanismo
- **Microsoft Presidio** para detecção de entidades PII
- **Regex fallback** quando Presidio não está disponível

### 5.2 Entidades Detectadas
- CPF brasileiro (com validação de dígitos)
- E-mail
- Telefone
- Números de cartão de crédito
- Endereços IP

### 5.3 Ação
- PII detectado em inputs é sinalizado
- PII em outputs é automaticamente anonimizado
- Logs de PII detectado são registrados (sem o dado em si)

## 6. Medidas de Segurança Técnicas

| Medida | Implementação |
|--------|---------------|
| Criptografia em trânsito | HTTPS via nginx |
| Isolamento | Docker containers |
| Controle de acesso | CORS configuration |
| Detecção de ataques | Prompt injection guardrails |
| Monitoramento | Prometheus + Grafana |
| Auditabilidade | MLflow + Langfuse telemetry |
| Anonimização | Presidio PII detection |

## 7. Encarregado (DPO)

Para questões relacionadas à proteção de dados:
- **Responsável**: Definir DPO antes da produção
- **Contato**: Definir canal de comunicação

## 8. Plano de Incidentes (Art. 48)

1. **Detecção**: Monitoramento automático via Prometheus/alertas
2. **Contenção**: Desativação do serviço afetado
3. **Comunicação**: Notificação à ANPD em 72h (quando aplicável)
4. **Correção**: Patch e re-deploy
5. **Documentação**: Registro do incidente e ações tomadas

---

*Documento atualizado em 2025. Revisão recomendada a cada 6 meses ou quando houver mudanças significativas no sistema.*
