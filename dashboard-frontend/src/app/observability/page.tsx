"use client";

import { useState } from "react";
import { ExternalLink, RefreshCw, CheckCircle, XCircle, AlertTriangle } from "lucide-react";
import TabGroup from "@/components/tab-group";
import LoadingSpinner from "@/components/loading-spinner";
import { api } from "@/lib/api";

const TABS = [
  { id: "drift", label: "Drift Detection", icon: "📉" },
  { id: "champion", label: "Champion-Challenger", icon: "🏆" },
  { id: "telemetry", label: "Telemetry", icon: "📡" },
];

export default function ObservabilityPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold">🔍 Observability</h2>
        <p className="mt-1 text-sm text-white/50">
          Monitor data drift, model performance, and system health
        </p>
      </div>

      <TabGroup tabs={TABS}>
        {(activeTab) => {
          if (activeTab === "drift") return <DriftTab />;
          if (activeTab === "champion") return <ChampionTab />;
          return <TelemetryTab />;
        }}
      </TabGroup>
    </div>
  );
}

/* ──────────── Drift Detection Tab ──────────── */
function DriftTab() {
  const [results, setResults] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runDrift = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.monitoring.drift();
      setResults(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Drift detection failed");
    } finally {
      setLoading(false);
    }
  };

  // Extract typed values from results for safe rendering
  const psiValues: Record<string, number> = results
    ? ((results.psi_values ?? {}) as Record<string, number>)
    : {};
  const featuresAnalyzed = results
    ? Number(results.features_analyzed ?? Object.keys(psiValues).length)
    : 0;
  const featuresDrifted = results
    ? Number(
        results.features_drifted ??
          Object.values(psiValues).filter((v) => v > 0.2).length
      )
    : 0;
  const driftDetected = results ? Boolean(results.drift_detected) : false;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">PSI-Based Drift Detection</h3>
          <p className="text-sm text-white/40">
            Detects distribution drift using Population Stability Index
          </p>
        </div>
        <button
          onClick={runDrift}
          disabled={loading}
          className="flex items-center gap-2 rounded-lg bg-nvidia px-4 py-2 text-sm font-semibold text-black hover:bg-nvidia-dark disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Run Drift Detection
        </button>
      </div>

      {loading && <LoadingSpinner text="Running drift detection..." />}

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-400">
          {error}
        </div>
      )}

      {results && !loading && (
        <div className="space-y-4">
          {/* Summary */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            <div className="rounded-lg border border-surface-border bg-surface-card p-4">
              <p className="text-xs text-white/40">Features Analyzed</p>
              <p className="text-2xl font-bold">{featuresAnalyzed}</p>
            </div>
            <div className="rounded-lg border border-surface-border bg-surface-card p-4">
              <p className="text-xs text-white/40">Drift Detected</p>
              <p className="text-2xl font-bold text-amber-400">
                {featuresDrifted}
              </p>
            </div>
            <div className="rounded-lg border border-surface-border bg-surface-card p-4">
              <p className="text-xs text-white/40">Status</p>
              <div className="mt-1 flex items-center gap-2">
                {driftDetected ? (
                  <>
                    <AlertTriangle className="h-5 w-5 text-amber-400" />
                    <span className="text-lg font-bold text-amber-400">
                      Drift Found
                    </span>
                  </>
                ) : (
                  <>
                    <CheckCircle className="h-5 w-5 text-green-400" />
                    <span className="text-lg font-bold text-green-400">
                      No Drift
                    </span>
                  </>
                )}
              </div>
            </div>
          </div>

          {/* PSI Values */}
          {Object.keys(psiValues).length > 0 && (
            <div className="rounded-xl border border-surface-border bg-surface-card p-6">
              <h4 className="mb-3 text-sm font-semibold text-white/70">
                PSI Values by Feature
              </h4>
              <div className="space-y-2">
                {Object.entries(psiValues).map(([feature, psi]) => {
                  const drifted = psi > 0.2;
                  const warning = psi > 0.1;
                  return (
                    <div
                      key={feature}
                      className="flex items-center justify-between rounded-lg bg-surface-hover px-4 py-2"
                    >
                      <span className="text-sm text-white/70">{feature}</span>
                      <div className="flex items-center gap-2">
                        <div className="h-2 w-32 overflow-hidden rounded-full bg-white/10">
                          <div
                            className="h-full rounded-full transition-all"
                            style={{
                              width: `${Math.min(psi / 0.5, 1) * 100}%`,
                              background: drifted
                                ? "#ef4444"
                                : warning
                                  ? "#fbbf24"
                                  : "#76B900",
                            }}
                          />
                        </div>
                        <span
                          className={`text-sm font-medium ${drifted ? "text-red-400" : warning ? "text-amber-400" : "text-green-400"}`}
                        >
                          {psi.toFixed(4)}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ──────────── Champion-Challenger Tab ──────────── */
function ChampionTab() {
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadResults = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.monitoring.championChallenger();
      setData(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  };

  const runPipeline = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await api.monitoring.runChampionChallenger();
      setData(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Pipeline failed");
    } finally {
      setRunning(false);
    }
  };

  const champion = data?.champion as Record<string, unknown> | undefined;
  const challenger = data?.challenger as Record<string, unknown> | undefined;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <button
          onClick={loadResults}
          disabled={loading}
          className="flex items-center gap-2 rounded-lg border border-surface-border bg-surface-hover px-4 py-2 text-sm text-white/70 hover:text-white disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Load Results
        </button>
        <button
          onClick={runPipeline}
          disabled={running}
          className="flex items-center gap-2 rounded-lg bg-nvidia px-4 py-2 text-sm font-semibold text-black hover:bg-nvidia-dark disabled:opacity-50"
        >
          {running ? "Running..." : "🚀 Run Pipeline"}
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-400">
          {error}
        </div>
      )}

      {loading && <LoadingSpinner text="Loading results..." />}

      {data && !loading && (
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          {/* Champion */}
          <div className="rounded-xl border border-nvidia/30 bg-surface-card p-6">
            <div className="mb-4 flex items-center gap-2">
              <span className="text-2xl">🏆</span>
              <h4 className="text-lg font-semibold text-nvidia">Champion</h4>
            </div>
            {champion ? (
              <div className="space-y-2">
                {Object.entries(champion).map(([k, v]) => (
                  <div key={k} className="flex justify-between text-sm">
                    <span className="text-white/50">{k}</span>
                    <span className="font-medium">
                      {typeof v === "number" ? v.toFixed(4) : String(v)}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-white/40">No champion data</p>
            )}
          </div>

          {/* Challenger */}
          <div className="rounded-xl border border-sky-400/30 bg-surface-card p-6">
            <div className="mb-4 flex items-center gap-2">
              <span className="text-2xl">⚔️</span>
              <h4 className="text-lg font-semibold text-sky-400">Challenger</h4>
            </div>
            {challenger ? (
              <div className="space-y-2">
                {Object.entries(challenger).map(([k, v]) => (
                  <div key={k} className="flex justify-between text-sm">
                    <span className="text-white/50">{k}</span>
                    <span className="font-medium">
                      {typeof v === "number" ? v.toFixed(4) : String(v)}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-white/40">No challenger data</p>
            )}
          </div>

          {/* Promotion Status */}
          {data.promoted !== undefined && (
            <div className="col-span-full rounded-xl border border-surface-border bg-surface-card p-6">
              <div className="flex items-center gap-3">
                {data.promoted ? (
                  <>
                    <CheckCircle className="h-6 w-6 text-green-400" />
                    <div>
                      <p className="font-semibold text-green-400">
                        Challenger Promoted! 🎉
                      </p>
                      <p className="text-sm text-white/50">
                        {typeof data.promotion_reason === "string"
                          ? data.promotion_reason
                          : "Challenger outperformed the champion model."}
                      </p>
                    </div>
                  </>
                ) : (
                  <>
                    <XCircle className="h-6 w-6 text-amber-400" />
                    <div>
                      <p className="font-semibold text-amber-400">
                        Champion Retained
                      </p>
                      <p className="text-sm text-white/50">
                        {typeof data.promotion_reason === "string"
                          ? data.promotion_reason
                          : "Current champion model remains the best."}
                      </p>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ──────────── Telemetry Tab ──────────── */
function TelemetryTab() {
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);

  const checkHealth = async () => {
    setLoading(true);
    try {
      const res = await api.health.check();
      setHealth(res);
    } catch {
      setHealth({ status: "unhealthy", error: "Cannot connect to API" });
    } finally {
      setLoading(false);
    }
  };

  const services = [
    {
      name: "Grafana",
      url: process.env.NEXT_PUBLIC_GRAFANA_URL || "http://localhost:3000",
      desc: "Metrics dashboards and alerting",
      icon: "📊",
    },
    {
      name: "MLflow",
      url: process.env.NEXT_PUBLIC_MLFLOW_URL || "http://localhost:5000",
      desc: "Experiment tracking and model registry",
      icon: "🔬",
    },
    {
      name: "Prometheus",
      url: process.env.NEXT_PUBLIC_PROMETHEUS_URL || "http://localhost:9090",
      desc: "Metrics collection and querying",
      icon: "🔥",
    },
  ];

  return (
    <div className="space-y-6">
      {/* Service Links */}
      <div>
        <h3 className="mb-3 text-lg font-semibold">🔗 External Services</h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          {services.map((svc) => (
            <a
              key={svc.name}
              href={svc.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group flex items-start gap-3 rounded-xl border border-surface-border bg-surface-card p-5 transition-all hover:border-nvidia/30"
            >
              <span className="text-2xl">{svc.icon}</span>
              <div>
                <div className="flex items-center gap-1.5">
                  <span className="font-semibold text-white group-hover:text-nvidia">
                    {svc.name}
                  </span>
                  <ExternalLink className="h-3.5 w-3.5 text-white/30 group-hover:text-nvidia" />
                </div>
                <p className="mt-0.5 text-xs text-white/40">{svc.desc}</p>
              </div>
            </a>
          ))}
        </div>
      </div>

      {/* API Health Check */}
      <div className="rounded-xl border border-surface-border bg-surface-card p-6">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-lg font-semibold">🏥 API Health Check</h3>
          <button
            onClick={checkHealth}
            disabled={loading}
            className="flex items-center gap-2 rounded-lg bg-nvidia px-4 py-2 text-sm font-semibold text-black hover:bg-nvidia-dark disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            Check Health
          </button>
        </div>

        {health && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              {health.status === "healthy" ? (
                <CheckCircle className="h-5 w-5 text-green-400" />
              ) : health.status === "degraded" ? (
                <AlertTriangle className="h-5 w-5 text-amber-400" />
              ) : (
                <XCircle className="h-5 w-5 text-red-400" />
              )}
              <span
                className={`text-lg font-bold ${
                  health.status === "healthy"
                    ? "text-green-400"
                    : health.status === "degraded"
                      ? "text-amber-400"
                      : "text-red-400"
                }`}
              >
                {String(health.status).toUpperCase()}
              </span>
            </div>
            {Object.entries(health)
              .filter(([k]) => k !== "status")
              .map(([k, v]) => (
                <div key={k} className="flex justify-between text-sm">
                  <span className="text-white/50">{k}</span>
                  <span className="font-medium">{String(v)}</span>
                </div>
              ))}
          </div>
        )}
      </div>
    </div>
  );
}
