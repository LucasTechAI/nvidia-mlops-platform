/* ──────────────── Predictions ──────────────── */

export interface HistoricalDataPoint {
  Date: string;
  Close: number;
  Open?: number;
  High?: number;
  Low?: number;
  Volume?: number;
  [key: string]: string | number | undefined;
}

export interface ForecastPoint {
  date: string;
  predicted_price: number;
  lower_bound?: number;
  upper_bound?: number;
}

export interface ForecastResponse {
  predictions: ForecastPoint[];
  current_price: number;
  model_info: {
    model_type: string;
    features_used: string[];
  };
  metadata: {
    horizon_days: number;
    confidence_level: number;
    mc_samples: number;
    generated_at: string;
  };
}

/* ──────────────── Model Info ──────────────── */

export interface ModelConfig {
  input_size: number;
  hidden_size: number;
  num_layers: number;
  output_size: number;
  dropout?: number;
  bidirectional?: boolean;
  sequence_length?: number;
}

export interface ParameterInfo {
  layers: Record<string, { shape: number[]; count: number; dtype: string }>;
  total: number;
  trainable: number;
}

export interface ModelInfoResponse {
  model_config: ModelConfig;
  parameters: ParameterInfo;
  training_info: Record<string, unknown>;
  test_metrics: Record<string, number>;
  epoch: number;
  best_epoch: number;
  loss: number;
  best_loss: number | null;
  features: string[];
}

export interface TrainingHistory {
  [key: string]: number[];
}

/* ──────────────── Monitoring ──────────────── */

export interface DriftResult {
  feature?: string;
  psi_value?: number;
  drift_detected?: boolean;
  [key: string]: unknown;
}

export interface ChampionChallengerResult {
  champion?: Record<string, unknown>;
  challenger?: Record<string, unknown>;
  promoted?: boolean;
  promotion_reason?: string;
  timestamp?: string;
  [key: string]: unknown;
}

/* ──────────────── Evaluation ──────────────── */

export interface EvaluationResult {
  champion: Record<string, unknown>;
  challenger: Record<string, unknown>;
  promoted: boolean;
  promotion_reason: string;
  timestamp: string;
}

export interface ExplainabilityFeature {
  feature: string;
  importance: number;
}

export interface LLMEvaluation {
  golden_set: unknown[];
  evaluation_results: { filename: string; data: unknown }[];
}

/* ──────────────── Agent ──────────────── */

export interface AgentMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

export interface AgentResponse {
  answer: string;
  sources?: string[];
  reasoning_steps?: string[];
  execution_time?: number;
}

/* ──────────────── Health ──────────────── */

export interface HealthResponse {
  status: "healthy" | "degraded" | "unhealthy";
  model_loaded?: boolean;
  uptime?: number;
  checks?: Record<string, unknown>;
}
