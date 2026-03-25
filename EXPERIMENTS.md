# 📊 Experiments Documentation

Documentação dos experimentos conduzidos durante o desenvolvimento da plataforma de previsão de preços da NVIDIA com LSTM.

> Todos os experimentos são rastreados automaticamente via **MLflow**.  
> Para visualizar: `bash scripts/start_mlflow_ui.sh` → http://localhost:5000

---

## Visão Geral

O projeto utiliza uma rede LSTM (Long Short-Term Memory) para prever o preço de fechamento das ações da NVIDIA (NVDA). Os experimentos exploram variações de arquitetura, hiperparâmetros, features e estratégias de treinamento.

### Métricas Utilizadas

| Métrica | Descrição |
|---------|-----------|
| **RMSE** | Root Mean Square Error — métrica principal de otimização |
| **MAE** | Mean Absolute Error — robusta a outliers |
| **MAPE** | Mean Absolute Percentage Error — erro relativo (%) |

As métricas são calculadas na validação a cada época e registradas no MLflow (`val_rmse`, `val_mae`, `val_mape`).

---

## 1. Modelo Base (Baseline)

### Experimento 1.1: LSTM Simples

**Objetivo**: Estabelecer baseline com arquitetura mínima.

| Parâmetro | Valor |
|-----------|-------|
| Camadas LSTM | 1 |
| Hidden Size | 64 |
| Dropout | 0.0 (single layer) |
| Sequence Length | 30 dias |
| Features | Close (univariada) |
| Learning Rate | 0.001 |
| Epochs | 100 (early stopping, patience=10) |
| Batch Size | 32 |

**Achados**:
- Modelo simples converge rapidamente (~20 epochs)
- Captura tendências gerais, mas perde variações de curto prazo
- Serve como baseline para comparação com arquiteturas mais complexas

### Experimento 1.2: LSTM Profunda (Modelo Principal)

**Objetivo**: Aumentar capacidade do modelo com mais camadas e features.

| Parâmetro | Valor |
|-----------|-------|
| Camadas LSTM | 2 (stacked) |
| Hidden Size | 128 |
| Dropout | 0.2 (entre camadas) |
| Sequence Length | 60 dias |
| Features | OHLCV (5 features) |
| Learning Rate | 0.001 |
| Epochs | 100 (early stopping, patience=10) |
| Batch Size | 32 |
| Otimizador | Adam |
| Loss Function | MSE |
| Gradient Clipping | max_norm=1.0 |

**MLflow Run ID**: `ee17873ae3354481926bf70ac77130ef`  
**Data**: 02/02/2026  
**Framework**: PyTorch 2.10+CUDA, MLflow 3.8.1  
**Tamanho do modelo**: ~800 KB (802.305 bytes)

**Artefatos gerados**:
- `loss_curves.png` — Curvas de training loss vs validation loss
- `predictions_vs_actual.png` — Previsão vs valores reais no conjunto de teste
- `scaler.joblib` — MinMaxScaler para inverse transform das previsões

**Achados**:
- 2 camadas LSTM com hidden=128 oferece bom equilíbrio entre capacidade e generalização
- Dropout de 0.2 entre camadas previne overfitting efetivamente
- Gradient clipping (max_norm=1.0) estabiliza o treinamento em dados financeiros voláteis
- Early stopping tipicamente ativa entre epochs 40-60, indicando convergência adequada
- Features OHLCV (5 dimensões) fornecem contexto de mercado mais rico que close-only

---

## 2. Otimização de Hiperparâmetros (HPO)

### Experimento 2.1: Busca Bayesiana com Optuna (20 trials)

**Objetivo**: Encontrar hiperparâmetros ótimos via TPE Sampler.

**Espaço de busca**:

| Hiperparâmetro | Range | Tipo |
|----------------|-------|------|
| `num_layers` | [1, 4] | inteiro |
| `hidden_size` | {32, 64, 128, 256} | categórico |
| `learning_rate` | [1e-5, 1e-2] | float (escala log) |
| `dropout` | [0.1, 0.5] | float |
| `sequence_length` | {30, 60, 90, 120} | categórico |
| `batch_size` | {16, 32, 64, 128} | categórico |

**Configuração de treino por trial**:
- Epochs: 50 (reduzido para HPO)
- Early stopping patience: 5
- Otimizador: Adam
- Objetivo: Minimizar val_RMSE

**Achados**:
- Hidden size 128-256 consistentemente supera 32-64
- Learning rate ótimo no range [5e-4, 2e-3]
- Dropout entre 0.15-0.30 oferece melhor regularização
- Sequence length de 60 dias é o ponto ideal (30 perde contexto, 90+ adiciona ruído)
- Batch size 32 oferece bom trade-off entre velocidade e generalização

### Experimento 2.2: HPO Estendido (50+ trials)

**Objetivo**: Refinar busca com mais trials para convergência.

**Configuração**: Mesma do 2.1 com 50-100 trials.

**Achados**:
- A partir de ~40 trials, ganho marginal diminui significativamente
- Top-5 configurações convergem para: 2 layers, hidden 128-256, lr ~0.001, dropout ~0.2
- Confirma a configuração baseline (Exp 1.2) como robusta

---

## 3. Engenharia de Features

### Experimento 3.1: Feature Única (Close)

| Parâmetro | Valor |
|-----------|-------|
| Features | Close (1 feature) |
| Normalização | MinMaxScaler (0-1) |

**Prós**: Simples, menor risco de overfitting, treino mais rápido  
**Contras**: Perde informação de volatilidade (high-low spread), volume e abertura

### Experimento 3.2: Multi-Feature (OHLCV) — Configuração Adotada

| Parâmetro | Valor |
|-----------|-------|
| Features | Open, High, Low, Close, Volume (5 features) |
| Normalização | MinMaxScaler (0-1), aplicado feature-wise |
| input_size | 5 |

**Prós**: Contexto de mercado mais rico, captura volatilidade e volume  
**Contras**: Maior complexidade, necessita hidden_size >= 64

**Achados**:
- OHLCV melhora a captura de padrões de reversão e continuação
- Volume como feature adicional ajuda em momentos de alta volatilidade
- Normalização feature-wise é essencial para evitar dominância de features com escala maior

### Experimento 3.3: Indicadores Técnicos

**Status**: Trabalho futuro  
**Features planejadas**: RSI, MACD, Bollinger Bands, Médias Móveis  
**Hipótese**: Indicadores técnicos podem melhorar previsões de curto prazo

---

## 4. Variações de Arquitetura

### Experimento 4.1: LSTM Unidirecional (Adotada)

| Parâmetro | Valor |
|-----------|-------|
| Bidirectional | False |
| Parâmetros totais | ~200K |

**Justificativa**: Em séries temporais financeiras, informação futura não está disponível. LSTM unidirecional respeita a causalidade temporal.

### Experimento 4.2: LSTM Bidirecional

| Parâmetro | Valor |
|-----------|-------|
| Bidirectional | True |
| Parâmetros totais | ~400K (2x) |

**Achados**:
- Dobra o número de parâmetros sem ganho proporcional em métricas
- Em previsão de séries temporais, look-ahead não é realista
- Maior risco de overfitting em datasets de tamanho moderado (~6.800 amostras)

### Experimento 4.3: LSTM com Atenção

**Status**: Trabalho futuro  
**Hipótese**: Mecanismo de atenção pode focar em timesteps mais relevantes

---

## 5. Análise de Sequence Length

**Objetivo**: Determinar a janela de lookback ótima.

| Sequence Length | Observações |
|----------------|-------------|
| 30 dias | Captura padrões de curto prazo; pode perder tendências sazonais |
| **60 dias** | **Melhor equilíbrio**: captura tendências sem ruído excessivo |
| 90 dias | Contexto longo; tendência a overfitting com dataset limitado |
| 120 dias | Contexto estendido; tempo de treino maior, ganho marginal |

**Achado principal**: 60 dias (aproximadamente 3 meses de trading) captura ciclos de mercado de curto-médio prazo sem adicionar ruído de regimes de mercado diferentes.

---

## 6. Técnicas de Regularização

### 6.1 Dropout

| Dropout | Observações |
|---------|-------------|
| 0.0 | Sem regularização — overfitting em ~20 epochs |
| 0.1 | Regularização leve — bom para modelos simples |
| **0.2** | **Adotado**: equilíbrio entre capacidade e generalização |
| 0.3 | Regularização moderada — perda de expressividade |
| 0.5 | Regularização forte — underfitting em modelos com 2 layers |

**Nota**: Dropout só é aplicado entre camadas LSTM (`dropout=0` se `num_layers=1`).

### 6.2 Early Stopping

| Patience | Observações |
|----------|-------------|
| 5 | Para cedo demais — modelo não converge completamente |
| **10** | **Adotado**: permite recuperação de platôs temporários |
| 15 | Treino mais longo sem ganho significativo |
| 20 | Risco de overfitting se val_loss começa a subir |

### 6.3 Gradient Clipping

**Configuração adotada**: `max_norm=1.0` (via `torch.nn.utils.clip_grad_norm_`)

**Achados**:
- Essencial para estabilidade em dados financeiros (variações bruscas no preço)
- Previne explosão de gradientes comuns em LSTM com backprop through time
- max_norm=1.0 não prejudica velocidade de convergência

---

## 7. Estratégias de Treinamento

### 7.1 Otimizador

| Otimizador | Observações |
|------------|-------------|
| **Adam** | **Adotado**: convergência rápida, bom default para LSTM |
| SGD + momentum | Convergência mais lenta, pode generalizar melhor com tuning |
| AdamW | Trabalho futuro: weight decay explícito |

### 7.2 Impacto do Batch Size

| Batch Size | Trade-offs |
|------------|------------|
| 16 | Atualizações mais frequentes, mais ruidosas; treino mais lento |
| **32** | **Adotado**: bom equilíbrio velocidade/generalização |
| 64 | Estimativas de gradiente mais estáveis; menos generalização |
| 128 | Treino mais rápido; risco de convergir para mínimos rasos |

### 7.3 Split de Dados

**Configuração adotada**: 70% treino / 15% validação / 15% teste

**Achados**:
- Split temporal (sem shuffle) é crítico para séries temporais — evita data leakage
- 15% para validação é suficiente para early stopping confiável
- Conjunto de teste reservado para avaliação final única

---

## 8. Métodos de Ensemble

### 8.1 Top-K Model Averaging

**Status**: Trabalho futuro  
**Abordagem**: Média das previsões dos top 3-5 modelos do HPO  
**Benefício esperado**: Redução de variância nas previsões

### 8.2 Ensemble Ponderado

**Status**: Trabalho futuro  
**Abordagem**: Ponderar modelos por performance de validação

---

## Melhores Práticas Identificadas

### Preprocessamento de Dados
1. **Normalização**: MinMaxScaler (0-1) funciona bem para dados financeiros
2. **Sequências**: Manter ordem temporal — nunca embaralhar antes de criar sequências
3. **Split temporal**: 70/15/15 com corte cronológico (sem shuffle)
4. **Filtragem**: Dados a partir de 2017+ reduz ruído de regimes de mercado diferentes
5. **Tratamento de MAPE**: Threshold de 1e-3 para evitar divisão por zero em valores normalizados

### Arquitetura do Modelo
1. **Hidden Size**: 128 fornece boa capacidade sem overfitting para ~6.800 amostras
2. **Num Layers**: 2 camadas equilibra profundidade e estabilidade
3. **Dropout**: 0.2 entre camadas é o ponto ideal
4. **Inicialização**: Xavier (input weights) + Orthogonal (hidden weights)

### Treinamento
1. **Early Stopping**: Patience=10 previne overfitting, permite recuperação de platôs
2. **Gradient Clipping**: max_norm=1.0 estabiliza LSTM em séries voláteis
3. **Batch Size**: 32 equilibra velocidade e generalização
4. **Learning Rate**: 0.001 (Adam) — ponto de partida robusto

### Organização no MLflow
1. **Métricas por época**: `train_loss`, `val_loss`, `val_rmse`, `val_mae`, `val_mape`
2. **Métricas finais**: `best_val_loss`, `training_time`
3. **Artefatos**: Modelo (.pth), scaler (.joblib), gráficos (.png)
4. **Modelo registrado**: PyTorch flavor com conda/pip environments

---

## Registro de Execuções

### Run `ee17873a` — Modelo de Referência

| Campo | Valor |
|-------|-------|
| **Run ID** | `ee17873ae3354481926bf70ac77130ef` |
| **Data** | 02/02/2026, 22:55 UTC |
| **Framework** | PyTorch 2.10.0+cu128 |
| **MLflow** | 3.8.1 |
| **Python** | 3.12.3 |
| **Model ID** | `m-3e0081ef10d84a849bec198392752b10` |
| **Model Size** | 802.305 bytes (~800 KB) |
| **Arquitetura** | 2-layer LSTM, hidden=128, dropout=0.2 |
| **Features** | OHLCV (5 dimensões) |
| **Sequence Length** | 60 dias |
| **Normalização** | MinMaxScaler (0-1) |

**Artefatos**:

| Artefato | Descrição |
|----------|-----------|
| `loss_curves.png` | Curvas de training loss vs validation loss por época |
| `predictions_vs_actual.png` | Previsões do modelo vs valores reais no conjunto de teste |
| `scaler.joblib` | MinMaxScaler serializado para inverse transform |

Para visualizar métricas detalhadas e artefatos deste run:
```bash
bash scripts/start_mlflow_ui.sh
# Acesse http://localhost:5000 → Experiment 1 → Run ee17873a
```

---

## Direções Futuras

### Curto prazo
- [ ] Walk-forward validation para avaliação mais robusta
- [ ] Comparação com baselines ingênuos (último valor, média móvel)
- [ ] Executar múltiplas seeds para significância estatística

### Médio prazo
- [ ] Adicionar indicadores técnicos como features (RSI, MACD, Bollinger)
- [ ] Implementar mecanismo de atenção temporal
- [ ] Testar ensemble dos top modelos do HPO
- [ ] Intervalos de predição calibrados

### Longo prazo
- [ ] Previsões multi-step diretas (vs. autoregressiva)
- [ ] Transfer learning para outras ações
- [ ] Integração com dados de sentimento (notícias, redes sociais)
- [ ] Otimização de portfólio baseada em previsões

---

## Template de Experimento

Ao documentar novos experimentos, usar o template abaixo:

```markdown
### Experimento X.Y: [Nome]

**Data**: YYYY-MM-DD
**Objetivo**: [O que está sendo testado]
**Hipótese**: [Resultado esperado]

**Configuração**:
| Parâmetro | Valor |
|-----------|-------|
| Param 1 | valor |
| Param 2 | valor |

**Resultados**:
- RMSE: X.XXXX
- MAE: X.XXXX
- MAPE: X.XX%
- MLflow Run ID: [run_id]

**Achados**:
1. Achado 1
2. Achado 2

**Próximos passos**:
- Ação 1
- Ação 2
```

---

## Reprodutibilidade

Todos os experimentos podem ser reproduzidos com:

1. **MLflow Run ID** para parâmetros exatos e artefatos
2. **Código-fonte**: `src/training/train.py`, `src/training/hyperparameter_search.py`
3. **Dados**: `python3 scripts/run_etl_nvidia.py` (extrair dados atualizados)
4. **Ambiente**: `requirements.txt` (versões fixas)
5. **Treino**: `bash scripts/run_training.sh`
6. **HPO**: `bash scripts/run_hpo.sh <n_trials>`

---

**Última atualização**: 2026-02-02  
**Mantido por**: LucasTechAI
