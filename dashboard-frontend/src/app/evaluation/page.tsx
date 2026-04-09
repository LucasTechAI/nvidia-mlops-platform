"use client";

import { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { RefreshCw } from "lucide-react";
import TabGroup from "@/components/tab-group";
import LoadingSpinner from "@/components/loading-spinner";
import { api } from "@/lib/api";

const TABS = [
  { id: "eval", label: "Evaluation Metrics", icon: "📊" },
  { id: "explain", label: "Explainability", icon: "🔍" },
  { id: "llm", label: "LLM Evaluation", icon: "🤖" },
];

export default function EvaluationPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold">📋 Evaluation</h2>
        <p className="mt-1 text-sm text-white/50">
          Model evaluation metrics, explainability analysis, and LLM benchmark results
        </p>
      </div>

      <TabGroup tabs={TABS}>
        {(activeTab) => {
          if (activeTab === "eval") return <EvalMetricsTab />;
          if (activeTab === "explain") return <ExplainabilityTab />;
          return <LLMEvalTab />;
        }}
      </TabGroup>
    </div>
  );
}

/* ──────────── Evaluation Metrics ──────────── */
function EvalMetricsTab() {
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadResults = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.evaluation.results();
      setData(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  };

  const champion = data?.champion as Record<string, unknown> | undefined;
  const challenger = data?.challenger as Record<string, unknown> | undefined;

  // Build comparison data for chart
  const metrics = ["rmse", "mae", "mape", "r2"];
  const comparisonData = metrics
    .filter((m) => champion?.[m] !== undefined || challenger?.[m] !== undefined)
    .map((m) => ({
      metric: m.toUpperCase(),
      Champion: Number(champion?.[m] ?? 0),
      Challenger: Number(challenger?.[m] ?? 0),
    }));

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Model Comparison</h3>
        <button
          onClick={loadResults}
          disabled={loading}
          className="flex items-center gap-2 rounded-lg bg-nvidia px-4 py-2 text-sm font-semibold text-black hover:bg-nvidia-dark disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Load Results
        </button>
      </div>

      {loading && <LoadingSpinner text="Loading evaluation results..." />}
      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-400">
          {error}
        </div>
      )}

      {data && !loading && (
        <>
          {/* Metric cards side by side */}
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            <div className="rounded-xl border border-nvidia/30 bg-surface-card p-6">
              <h4 className="mb-3 flex items-center gap-2 text-lg font-semibold text-nvidia">
                🏆 Champion
              </h4>
              {champion ? (
                <div className="space-y-2">
                  {Object.entries(champion).map(([k, v]) => (
                    <div key={k} className="flex justify-between text-sm">
                      <span className="text-white/50">{k}</span>
                      <span className="font-medium">
                        {typeof v === "number" ? v.toFixed(4) : String(v ?? "—")}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-white/40">No data</p>
              )}
            </div>
            <div className="rounded-xl border border-sky-400/30 bg-surface-card p-6">
              <h4 className="mb-3 flex items-center gap-2 text-lg font-semibold text-sky-400">
                ⚔️ Challenger
              </h4>
              {challenger ? (
                <div className="space-y-2">
                  {Object.entries(challenger).map(([k, v]) => (
                    <div key={k} className="flex justify-between text-sm">
                      <span className="text-white/50">{k}</span>
                      <span className="font-medium">
                        {typeof v === "number" ? v.toFixed(4) : String(v ?? "—")}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-white/40">No data</p>
              )}
            </div>
          </div>

          {/* Comparison chart */}
          {comparisonData.length > 0 && (
            <div className="rounded-xl border border-surface-border bg-surface-card p-6">
              <h4 className="mb-4 text-sm font-semibold text-white/70">
                Metrics Comparison
              </h4>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={comparisonData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="metric" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      background: "#1a1c24",
                      border: "1px solid rgba(118,185,0,0.3)",
                      borderRadius: 8,
                    }}
                  />
                  <Bar dataKey="Champion" fill="#76B900" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Challenger" fill="#4ECDC4" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Promotion status */}
          {data.promoted !== undefined && (
            <div
              className={`rounded-lg border p-4 ${
                data.promoted
                  ? "border-green-500/30 bg-green-500/10"
                  : "border-amber-500/30 bg-amber-500/10"
              }`}
            >
              {data.promoted !== undefined && (
                <p className={`font-semibold ${data.promoted ? "text-green-400" : "text-amber-400"}`}>
                  {data.promoted ? "✅ Challenger promoted!" : "⚠️ Champion retained"}
                </p>
              )}
              {typeof data.promotion_reason === "string" && data.promotion_reason && (
                <p className="mt-1 text-sm text-white/50">{data.promotion_reason}</p>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

/* ──────────── Explainability ──────────── */
function ExplainabilityTab() {
  const [features, setFeatures] = useState<{ feature: string; importance: number }[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.evaluation.explainability();
      const feats = (res as { features?: { feature: string; importance: number }[] }).features || [];
      setFeatures(feats);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed");
    } finally {
      setLoading(false);
    }
  };

  const chartData = features
    .sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance))
    .map((f) => ({
      feature: f.feature,
      importance: Number(f.importance.toFixed(4)),
    }));

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Permutation Importance</h3>
          <p className="text-sm text-white/40">
            Feature importance based on permutation analysis
          </p>
        </div>
        <button
          onClick={run}
          disabled={loading}
          className="flex items-center gap-2 rounded-lg bg-nvidia px-4 py-2 text-sm font-semibold text-black hover:bg-nvidia-dark disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Compute Importance
        </button>
      </div>

      {loading && <LoadingSpinner text="Computing feature importance..." />}
      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-400">
          {error}
        </div>
      )}

      {chartData.length > 0 && !loading && (
        <>
          <div className="rounded-xl border border-surface-border bg-surface-card p-6">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={120} />
                <Tooltip
                  contentStyle={{
                    background: "#1a1c24",
                    border: "1px solid rgba(118,185,0,0.3)",
                    borderRadius: 8,
                  }}
                />
                <Bar dataKey="importance" fill="#76B900" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="rounded-xl border border-surface-border bg-surface-card p-6">
            <h4 className="mb-3 text-sm font-semibold text-white/70">Feature Details</h4>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-surface-border text-left text-xs text-white/40">
                  <th className="pb-2">Feature</th>
                  <th className="pb-2">Importance</th>
                </tr>
              </thead>
              <tbody>
                {chartData.map((f) => (
                  <tr key={f.feature} className="border-b border-surface-border/50">
                    <td className="py-2 text-white/70">{f.feature}</td>
                    <td className="py-2 font-medium text-nvidia">{f.importance}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

/* ──────────── LLM Evaluation ──────────── */
function LLMEvalTab() {
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.evaluation.llmResults();
      setData(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed");
    } finally {
      setLoading(false);
    }
  };

  const goldenSet = (data?.golden_set as unknown[]) || [];
  const evalResults = (data?.evaluation_results as { filename: string; data: unknown }[]) || [];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">LLM Evaluation (RAGAS)</h3>
          <p className="text-sm text-white/40">
            Golden set evaluation and benchmark results
          </p>
        </div>
        <button
          onClick={load}
          disabled={loading}
          className="flex items-center gap-2 rounded-lg bg-nvidia px-4 py-2 text-sm font-semibold text-black hover:bg-nvidia-dark disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Load Results
        </button>
      </div>

      {loading && <LoadingSpinner text="Loading LLM evaluation..." />}
      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-400">
          {error}
        </div>
      )}

      {data && !loading && (
        <>
          {/* Golden Set */}
          <div className="rounded-xl border border-surface-border bg-surface-card p-6">
            <h4 className="mb-3 text-sm font-semibold text-white/70">
              Golden Set ({goldenSet.length} samples)
            </h4>
            <div className="max-h-60 space-y-2 overflow-auto">
              {goldenSet.map((item, i) => {
                const q = item as Record<string, unknown>;
                return (
                  <div key={i} className="rounded-lg bg-surface-hover p-3">
                    <p className="text-sm font-medium text-white/80">
                      {String(q.query || q.question || q.input || JSON.stringify(item))}
                    </p>
                    {typeof q.expected === "string" && q.expected && (
                      <p className="mt-1 text-xs text-white/40">
                        Expected: {q.expected}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Evaluation Results */}
          {evalResults.length > 0 && (
            <div className="rounded-xl border border-surface-border bg-surface-card p-6">
              <h4 className="mb-3 text-sm font-semibold text-white/70">
                Evaluation Results
              </h4>
              <div className="space-y-3">
                {evalResults.map((result, i) => (
                  <div key={i} className="rounded-lg bg-surface-hover p-4">
                    <p className="text-xs font-medium text-nvidia">{result.filename}</p>
                    <pre className="mt-2 max-h-40 overflow-auto text-xs text-white/60">
                      {JSON.stringify(result.data, null, 2)}
                    </pre>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
