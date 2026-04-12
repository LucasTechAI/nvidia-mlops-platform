/**
 * API client for the NVIDIA MLOps Platform backend.
 *
 * All calls go through Next.js rewrites → FastAPI at port 8000.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `API error: ${res.status}`);
  }

  return res.json();
}

/* ──────── Health ──────── */

export const api = {
  health: {
    check: () => request<Record<string, unknown>>("/health"),
  },

  /* ──────── Data ──────── */
  data: {
    historical: (limit?: number) =>
      request<Record<string, unknown>>(
        `/data${limit ? `?limit=${limit}` : ""}`
      ),
    recent: (days: number = 30) =>
      request<Record<string, unknown>>(`/data/latest?days=${days}`),
    stats: () => request<Record<string, unknown>>("/data/summary"),
    columns: () => request<{ columns: string[] }>("/data/columns"),
    live: () =>
      request<{ data: { date: string; close: number }[]; db_last_date: string; live_count: number }>(
        "/data/live"
      ),
  },

  /* ──────── Predictions ──────── */
  predict: {
    forecast: (params: {
      horizon?: number;
      with_uncertainty?: boolean;
      n_samples?: number;
      confidence_level?: number;
    }) =>
      request<Record<string, unknown>>("/predict", {
        method: "POST",
        body: JSON.stringify(params),
      }),
    backtest: (days: number = 60) =>
      request<{ backtest: { date: string; actual: number; predicted: number }[]; days: number }>(
        `/predict/backtest?days=${days}`
      ),
  },

  /* ──────── Model ──────── */
  model: {
    info: () => request<Record<string, unknown>>("/model/info"),
    trainingHistory: () =>
      request<Record<string, number[]>>("/model/training-history"),
    hpoResults: () => request<Record<string, unknown>>("/model/hpo-results"),
  },

  /* ──────── Monitoring ──────── */
  monitoring: {
    drift: () =>
      request<Record<string, unknown>>("/monitoring/drift", {
        method: "POST",
      }),
    allTriggers: () =>
      request<Record<string, unknown>>("/monitoring/drift/all-triggers", {
        method: "POST",
      }),
    championChallenger: () =>
      request<Record<string, unknown>>("/monitoring/champion-challenger"),
    runChampionChallenger: () =>
      request<Record<string, unknown>>(
        "/monitoring/champion-challenger/run",
        { method: "POST" }
      ),
    runsHistory: () =>
      request<Record<string, unknown>>("/monitoring/runs/history"),
  },

  /* ──────── Evaluation ──────── */
  evaluation: {
    results: () => request<Record<string, unknown>>("/evaluation/results"),
    explainability: () =>
      request<Record<string, unknown>>("/evaluation/explainability"),
    recomputeExplainability: () =>
      request<Record<string, unknown>>("/evaluation/explainability", {
        method: "POST",
      }),
    lime: () =>
      request<Record<string, unknown>>("/evaluation/lime"),
    recomputeLime: () =>
      request<Record<string, unknown>>("/evaluation/lime", {
        method: "POST",
      }),
    llmResults: () =>
      request<Record<string, unknown>>("/evaluation/llm-results"),
    runEvaluation: () =>
      request<Record<string, unknown>>("/evaluation/run", { method: "POST" }),
  },

  /* ──────── Agent ──────── */
  agent: {
    query: (params: {
      query: string;
      use_guardrails?: boolean;
      temperature?: number;
      max_iterations?: number;
    }) =>
      request<Record<string, unknown>>("/agent/query", {
        method: "POST",
        body: JSON.stringify(params),
      }),
    health: () => request<Record<string, unknown>>("/agent/health"),
  },

  /* ──────── Logs ──────── */
  logs: {
    stats: (since?: number) =>
      request<{
        total: number;
        by_level: Record<string, number>;
        by_source: Record<string, number>;
        error_rate: number;
        warning_rate: number;
        logs_per_minute: number;
      }>(`/logs/stats?since=${since ?? 120}`),
    timeline: (since?: number) =>
      request<{
        timeline: { time: string; INFO: number; WARNING: number; ERROR: number; DEBUG: number }[];
      }>(`/logs/timeline?since=${since ?? 120}`),
    entries: (params?: {
      level?: string;
      source?: string;
      search?: string;
      since?: number;
      limit?: number;
    }) => {
      const q = new URLSearchParams();
      if (params?.level) q.set("level", params.level);
      if (params?.source) q.set("source", params.source);
      if (params?.search) q.set("search", params.search);
      q.set("since", String(params?.since ?? 120));
      q.set("limit", String(params?.limit ?? 200));
      return request<{
        entries: {
          id: number;
          timestamp: string;
          level: string;
          source: string;
          module: string;
          message: string;
          extra: Record<string, unknown>;
        }[];
        count: number;
      }>(`/logs/entries?${q.toString()}`);
    },
    sources: () => request<{ sources: string[] }>("/logs/sources"),
    // Legacy file-based
    api: () => request<{ content: string }>("/logs/api"),
    training: () => request<{ content: string }>("/logs/training"),
    services: () => request<{ services: { name: string; logs: string }[] }>("/logs/services"),
    system: () => request<{ content: string }>("/logs/system"),
  },

  /* ──────── Business Metrics ──────── */
  businessMetrics: {
    snapshot: () =>
      request<{
        cumulative_pnl: number;
        roi_pct: number;
        sharpe_ratio: number;
        max_drawdown: number;
        win_rate: number;
        avg_error_pct: number;
        total_predictions: number;
        winning_predictions: number;
        daily_returns: number[];
      }>("/business-metrics/snapshot"),
    pnlHistory: (days?: number) =>
      request<{ history: { date: string; pnl: number; predicted: number; actual: number }[] }>(
        `/business-metrics/pnl-history?days=${days ?? 30}`
      ),
    dailySummaries: (days?: number) =>
      request<{ summaries: Record<string, unknown>[] }>(
        `/business-metrics/daily-summaries?days=${days ?? 30}`
      ),
  },

  /* ──────── SLA ──────── */
  sla: {
    report: (periodMinutes?: number) =>
      request<{
        uptime_pct: number;
        total_checks: number;
        successful_checks: number;
        avg_latency_ms: number;
        p50_latency_ms: number;
        p95_latency_ms: number;
        p99_latency_ms: number;
        error_rate_pct: number;
        total_requests: number;
        error_requests: number;
        sla_targets: Record<string, number>;
        violations: string[];
        overall_sla_met: boolean;
      }>(`/sla/report?period_minutes=${periodMinutes ?? 60}`),
    uptimeHistory: (days?: number) =>
      request<{ history: { date: string; uptime_pct: number; checks: number; requests: number; avg_latency_ms: number }[] }>(
        `/sla/uptime-history?days=${days ?? 7}`
      ),
  },

  /* ──────── Feature Store ──────── */
  featureStore: {
    list: () =>
      request<{ feature_sets: { name: string; latest_version: number; total_versions: number; last_updated: string }[] }>(
        "/feature-store/list"
      ),
    get: (name: string, version?: number) =>
      request<Record<string, unknown>>(
        `/feature-store/${name}${version ? `?version=${version}` : ""}`
      ),
    lineage: (name: string) =>
      request<{ lineage: Record<string, unknown>[] }>(`/feature-store/${name}/lineage`),
    preview: (name: string, rows?: number) =>
      request<{ columns: string[]; data: Record<string, unknown>[]; total_rows: number }>(
        `/feature-store/${name}/preview?rows=${rows ?? 10}`
      ),
  },

  /* ──────── Model Registry ──────── */
  modelRegistry: {
    list: () => request<{ models: Record<string, unknown>[] }>("/model-registry/models"),
    versions: (name: string) =>
      request<{ model_name: string; versions: { version: number; stage: string; created_at: string; metrics: Record<string, number> }[] }>(
        `/model-registry/${name}/versions`
      ),
    production: (name: string) => request<Record<string, unknown>>(`/model-registry/${name}/production`),
    promote: (name: string, version: number, stage?: string) =>
      request<Record<string, unknown>>(
        `/model-registry/${name}/promote/${version}?target_stage=${stage ?? "Production"}`,
        { method: "POST" }
      ),
    rollback: (name: string) =>
      request<Record<string, unknown>>(`/model-registry/${name}/rollback`, { method: "POST" }),
    history: (name: string) =>
      request<{ history: Record<string, unknown>[] }>(`/model-registry/${name}/history`),
  },

  /* ──────── Canary ──────── */
  canary: {
    deployments: (modelName?: string) =>
      request<{ deployments: Record<string, unknown>[] }>(
        `/canary/deployments${modelName ? `?model_name=${modelName}` : ""}`
      ),
    status: (id: string) => request<Record<string, unknown>>(`/canary/${id}/status`),
    start: (modelName: string, canaryVersion: number, baselineVersion: number) =>
      request<{ deployment_id: string; state: string; canary_weight: number }>(
        `/canary/start?model_name=${modelName}&canary_version=${canaryVersion}&baseline_version=${baselineVersion}`,
        { method: "POST" }
      ),
    evaluate: (id: string) =>
      request<{ action: string; status: Record<string, unknown> }>(`/canary/${id}/evaluate`, { method: "POST" }),
    rollbackHistory: () =>
      request<{ rollbacks: Record<string, unknown>[] }>("/canary/rollback-history"),
  },

  /* ──────── Cost Analysis ──────── */
  costAnalysis: {
    get: (days?: number) =>
      request<{
        period_days: number;
        grand_total: number;
        infra_total: number;
        llm_total: number;
        infra_pct: number;
        llm_pct: number;
        current_model: string;
        current_model_id: string;
        provider: string;
        total_input_tokens: number;
        total_output_tokens: number;
        total_requests: number;
        training_runs: number;
        infra_breakdown: { name: string; quantity: number; unit: string; unit_cost: number; total: number }[];
        llm_breakdown: { name: string; tokens: number; cost: number }[];
        daily_history: { date: string; infra: number; llm: number; total: number }[];
        model_comparison: { model: string; model_id: string; input_cost: number; output_cost: number; total_cost: number; is_current: boolean }[];
      }>(`/cost-analysis?days=${days ?? 30}`),
  },
};
