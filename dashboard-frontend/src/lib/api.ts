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
      request<Record<string, unknown>>("/evaluation/explainability", {
        method: "POST",
      }),
    llmResults: () =>
      request<Record<string, unknown>>("/evaluation/llm-results"),
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
};
