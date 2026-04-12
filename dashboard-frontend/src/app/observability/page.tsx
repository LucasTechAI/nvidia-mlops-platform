"use client";

import { useState, useEffect } from "react";
import { ExternalLink, RefreshCw, CheckCircle, XCircle, AlertTriangle, Info, TrendingDown, Clock, BarChart3, Shield, Activity, History, SkipForward, Github, FileText } from "lucide-react";
import TabGroup from "@/components/tab-group";
import LoadingSpinner from "@/components/loading-spinner";
import { api } from "@/lib/api";

const TABS = [
  { id: "drift", label: "Drift Detection", icon: "📉" },
  { id: "champion", label: "Champion-Challenger", icon: "🏆" },
  { id: "history", label: "Model History", icon: "📜" },
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
          if (activeTab === "history") return <HistoryTab />;
          return <TelemetryTab />;
        }}
      </TabGroup>
    </div>
  );
}

/* ──────────── Drift Detection Tab ──────────── */
function DriftTab() {
  const [results, setResults] = useState<Record<string, unknown> | null>(null);
  const [allTriggersResults, setAllTriggersResults] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingAll, setLoadingAll] = useState(false);
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

  const runAllTriggers = async () => {
    setLoadingAll(true);
    setError(null);
    try {
      const res = await api.monitoring.allTriggers();
      setAllTriggersResults(res);
      // Also populate the PSI results from the nested data_drift trigger
      const trig = res.triggers as Record<string, Record<string, unknown>> | undefined;
      if (trig?.data_drift && trig.data_drift.status !== "error") {
        setResults(trig.data_drift);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Multi-trigger detection failed");
    } finally {
      setLoadingAll(false);
    }
  };

  // Extract typed values from results
  const psiValues: Record<string, number> = {};
  const featureDetails: Record<string, { psi: number; status: string; ref_mean: number; ref_std: number; cur_mean: number; cur_std: number }> = {};
  const featuresObj = (results?.features ?? {}) as Record<string, Record<string, unknown>>;
  for (const [fname, fdata] of Object.entries(featuresObj)) {
    const psi = Number(fdata.psi ?? 0);
    psiValues[fname] = psi;
    featureDetails[fname] = {
      psi,
      status: String(fdata.status ?? "ok"),
      ref_mean: Number(fdata.ref_mean ?? 0),
      ref_std: Number(fdata.ref_std ?? 0),
      cur_mean: Number(fdata.cur_mean ?? 0),
      cur_std: Number(fdata.cur_std ?? 0),
    };
  }
  // Also handle flat psi_values if backend uses that format
  if (Object.keys(psiValues).length === 0 && results?.psi_values) {
    for (const [k, v] of Object.entries(results.psi_values as Record<string, number>)) {
      psiValues[k] = v;
    }
  }

  const featuresAnalyzed = results
    ? Number(results.features_analyzed ?? Object.keys(psiValues).length)
    : 0;
  const featuresDrifted = results
    ? Number(results.drifted_features ?? results.features_drifted ?? Object.values(psiValues).filter((v) => v > 0.2).length)
    : 0;
  const driftDetected = results ? Boolean(results.drift_detected) : false;
  const overallStatus = results ? String(results.overall_status ?? "unknown") : "unknown";
  const avgPsi = results ? Number(results.avg_psi ?? 0) : 0;
  const nReference = results ? Number(results.n_reference ?? 0) : 0;
  const nCurrent = results ? Number(results.n_current ?? 0) : 0;
  const method = results ? String(results.method ?? "PSI") : "PSI";
  const timestamp = results ? String(results.timestamp ?? "") : "";
  const trainingCutoff = results ? String(results.training_cutoff_date ?? "") : "";
  const referenceStart = results ? String(results.reference_start ?? "") : "";
  const referenceEnd = results ? String(results.reference_end ?? "") : "";
  const currentStart = results ? String(results.current_start ?? "") : "";
  const currentEnd = results ? String(results.current_end ?? "") : "";
  const splitMethod = results ? String(results.split_method ?? "ratio") : "ratio";
  const analysisWindowDays = results ? Number(results.analysis_window_days ?? 30) : 30;

  // ── All-triggers extracted data ──
  const triggers = (allTriggersResults?.triggers ?? {}) as Record<string, Record<string, unknown>>;
  const staleness = triggers.staleness ?? null;
  const ciBreach = triggers.prediction_breach ?? null;
  const activeTriggers = (allTriggersResults?.active_triggers ?? []) as string[];
  const combinedRetrain = Boolean(allTriggersResults?.retrain_recommended);
  const combinedSummary = String(allTriggersResults?.summary ?? "");

  // Derive per-trigger live status for the cards
  const psiTriggerFired = activeTriggers.includes("data_drift_psi");
  const stalenessTriggerFired = activeTriggers.includes("model_staleness");
  const ciBreachTriggerFired = activeTriggers.includes("prediction_ci_breach");

  const psiLabel = (v: number) =>
    v > 0.2 ? "Retrain" : v > 0.1 ? "Warning" : "Stable";
  const psiColor = (v: number) =>
    v > 0.2 ? "text-red-400" : v > 0.1 ? "text-amber-400" : "text-green-400";
  const psiBg = (v: number) =>
    v > 0.2 ? "bg-red-500" : v > 0.1 ? "bg-amber-400" : "bg-green-500";
  const psiIcon = (v: number) =>
    v > 0.2 ? <XCircle className="inline h-4 w-4 text-red-400" /> : v > 0.1 ? <AlertTriangle className="inline h-4 w-4 text-amber-400" /> : <CheckCircle className="inline h-4 w-4 text-green-400" />;

  return (
    <div className="space-y-6">
      {/* ── Hero / Explanation ── */}
      <div className="rounded-xl border border-surface-border bg-gradient-to-br from-[#12131a] to-[#1a1c28] p-6">
        <div className="flex items-start justify-between">
          <div>
            <h3 className="flex items-center gap-2 text-lg font-bold">
              <Shield className="h-5 w-5 text-nvidia" />
              Multi-Trigger Retrain Detection
            </h3>
            <p className="mt-1 max-w-2xl text-sm text-white/50">
              Monitors model health through <strong className="text-nvidia">3 independent triggers</strong>.
              Any single trigger firing is sufficient to recommend retraining —
              defense-in-depth for production ML.
            </p>
          </div>
          <div className="flex flex-shrink-0 flex-col gap-2">
            <button
              onClick={runAllTriggers}
              disabled={loadingAll || loading}
              className="flex items-center gap-2 rounded-lg bg-nvidia px-5 py-2.5 text-sm font-semibold text-black shadow-lg shadow-nvidia/20 hover:bg-nvidia-dark disabled:opacity-50"
            >
              <Shield className={`h-4 w-4 ${loadingAll ? "animate-pulse" : ""}`} />
              Run All 3 Triggers
            </button>
            <button
              onClick={runDrift}
              disabled={loading || loadingAll}
              className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-5 py-2 text-xs font-medium text-white/60 hover:bg-white/10 hover:text-white disabled:opacity-50"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
              PSI Only
            </button>
          </div>
        </div>

        {/* PSI formula & thresholds */}
        <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-2">
          {/* Formula */}
          <div className="rounded-lg border border-white/5 bg-white/[0.02] p-4">
            <p className="mb-2 text-[10px] font-bold uppercase tracking-widest text-white/30">PSI Formula</p>
            <p className="font-mono text-sm text-white/70">
              PSI = Σ (P<sub>i</sub> − Q<sub>i</sub>) × ln(P<sub>i</sub> / Q<sub>i</sub>)
            </p>
            <p className="mt-2 text-[11px] text-white/40">
              Where <span className="text-white/60">P</span> = current distribution bins,{" "}
              <span className="text-white/60">Q</span> = reference (training) distribution bins.
              Measures divergence between the two across histogram bins.
            </p>
          </div>

          {/* Thresholds */}
          <div className="rounded-lg border border-white/5 bg-white/[0.02] p-4">
            <p className="mb-2 text-[10px] font-bold uppercase tracking-widest text-white/30">Decision Thresholds</p>
            <div className="space-y-2">
              <div className="flex items-center gap-3">
                <div className="h-3 w-3 rounded-full bg-green-500" />
                <div>
                  <span className="text-sm font-semibold text-green-400">PSI &lt; 0.1</span>
                  <span className="ml-2 text-xs text-white/40">— Stable, no significant drift</span>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="h-3 w-3 rounded-full bg-amber-400" />
                <div>
                  <span className="text-sm font-semibold text-amber-400">0.1 ≤ PSI &lt; 0.2</span>
                  <span className="ml-2 text-xs text-white/40">— Warning, moderate shift detected</span>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="h-3 w-3 rounded-full bg-red-500" />
                <div>
                  <span className="text-sm font-semibold text-red-400">PSI ≥ 0.2</span>
                  <span className="ml-2 text-xs text-white/40">— Critical drift → retrain recommended</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 3 Triggers overview — LIVE STATUS */}
        <div className="mt-4 rounded-lg border border-nvidia/20 bg-nvidia/5 p-3">
          <p className="mb-2 text-center text-[10px] font-bold uppercase tracking-widest text-nvidia">
            <Shield className="mr-1 inline h-3 w-3" />
            Multi-Trigger Retrain System — any 1 trigger = retrain recommendation
          </p>
          <div className="flex justify-center gap-3">
            {[
              {
                icon: "📊",
                name: "Data Drift (PSI)",
                desc: "PSI > 0.2",
                tested: results !== null || allTriggersResults !== null,
                fired: allTriggersResults ? psiTriggerFired : driftDetected,
              },
              {
                icon: "⏰",
                name: "Model Staleness",
                desc: "≥ 30 days",
                tested: allTriggersResults !== null,
                fired: stalenessTriggerFired,
              },
              {
                icon: "📉",
                name: "CI Breach",
                desc: "> 20% outside CI",
                tested: allTriggersResults !== null && ciBreach?.status !== "skipped",
                fired: ciBreachTriggerFired,
              },
            ].map((t) => (
              <div
                key={t.name}
                className={`flex items-center gap-2 rounded-lg border-2 px-3 py-2 transition-all ${
                  !t.tested
                    ? "border-white/10 bg-white/[0.02] opacity-50"
                    : t.fired
                      ? "border-red-500/40 bg-red-500/10"
                      : "border-green-500/40 bg-green-500/10"
                }`}
              >
                <span className="text-base">{t.icon}</span>
                <div>
                  <p className={`text-[10px] font-bold ${
                    !t.tested ? "text-white/40" : t.fired ? "text-red-400" : "text-green-400"
                  }`}>
                    {t.name}
                  </p>
                  <p className="text-[9px] text-white/30">
                    {!t.tested ? t.desc : t.fired ? <><XCircle className="mr-0.5 inline h-3 w-3 text-red-400" /> TRIGGERED</> : <><CheckCircle className="mr-0.5 inline h-3 w-3 text-green-400" /> Passed</>}
                  </p>
                </div>
              </div>
            ))}
          </div>
          {!allTriggersResults && (
            <p className="mt-2 text-center text-[9px] text-white/30">
              Click <strong className="text-nvidia">Run All 3 Triggers</strong> to test all triggers simultaneously, or <strong className="text-white/50">PSI Only</strong> for data drift alone.
            </p>
          )}
          {allTriggersResults && (
            <p className={`mt-2 text-center text-[10px] font-semibold ${combinedRetrain ? "text-red-400" : "text-green-400"}`}>
              {combinedRetrain ? <><AlertTriangle className="mr-1 inline h-3.5 w-3.5" /> </> : <><CheckCircle className="mr-1 inline h-3.5 w-3.5" /> </>}{combinedSummary}
            </p>
          )}
        </div>
      </div>

      {(loading || loadingAll) && (
        <LoadingSpinner text={loadingAll ? "Running all 3 retrain triggers (PSI + Staleness + CI Breach)..." : "Running PSI drift detection across all features..."} />
      )}

      {error && (
        <div className="flex items-center gap-3 rounded-lg border border-red-500/30 bg-red-500/10 p-4">
          <XCircle className="h-5 w-5 flex-shrink-0 text-red-400" />
          <div>
            <p className="text-sm font-semibold text-red-400">Drift detection failed</p>
            <p className="text-xs text-red-400/70">{error}</p>
          </div>
        </div>
      )}

      {results && !loading && (
        <div className="space-y-5">
          {/* ── Summary Cards ── */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <div className="rounded-xl border border-surface-border bg-surface-card p-4">
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-white/30" />
                <p className="text-[10px] font-bold uppercase tracking-wider text-white/40">Features Analyzed</p>
              </div>
              <p className="mt-2 text-3xl font-bold">{featuresAnalyzed}</p>
              <p className="text-[10px] text-white/30">input columns checked</p>
            </div>

            <div className="rounded-xl border border-surface-border bg-surface-card p-4">
              <div className="flex items-center gap-2">
                <TrendingDown className="h-4 w-4 text-amber-400/50" />
                <p className="text-[10px] font-bold uppercase tracking-wider text-white/40">Features Drifted</p>
              </div>
              <p className={`mt-2 text-3xl font-bold ${featuresDrifted > 0 ? "text-amber-400" : "text-green-400"}`}>
                {featuresDrifted}
              </p>
              <p className="text-[10px] text-white/30">{featuresDrifted > 0 ? "features with PSI > 0.1" : "all features stable"}</p>
            </div>

            <div className="rounded-xl border border-surface-border bg-surface-card p-4">
              <div className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-nvidia/50" />
                <p className="text-[10px] font-bold uppercase tracking-wider text-white/40">Avg PSI</p>
              </div>
              <p className={`mt-2 text-3xl font-bold ${psiColor(avgPsi)}`}>
                {avgPsi.toFixed(4)}
              </p>
              <p className="text-[10px] text-white/30">across all features</p>
            </div>

            <div className="rounded-xl border border-surface-border bg-surface-card p-4">
              <div className="flex items-center gap-2">
                {driftDetected ? (
                  <AlertTriangle className="h-4 w-4 text-amber-400/50" />
                ) : (
                  <CheckCircle className="h-4 w-4 text-green-400/50" />
                )}
                <p className="text-[10px] font-bold uppercase tracking-wider text-white/40">Status</p>
              </div>
              <div className="mt-2 flex items-center gap-2">
                {overallStatus === "retrain_recommended" ? (
                  <>
                    <XCircle className="h-5 w-5 text-red-400" />
                    <span className="text-lg font-bold text-red-400">Retrain</span>
                  </>
                ) : driftDetected ? (
                  <>
                    <AlertTriangle className="h-5 w-5 text-amber-400" />
                    <span className="text-lg font-bold text-amber-400">Warning</span>
                  </>
                ) : (
                  <>
                    <CheckCircle className="h-5 w-5 text-green-400" />
                    <span className="text-lg font-bold text-green-400">Stable</span>
                  </>
                )}
              </div>
              <p className="text-[10px] text-white/30">{overallStatus.replace(/_/g, " ")}</p>
            </div>
          </div>

          {/* ── Dataset Info Bar ── */}
          <div className="rounded-lg border border-white/5 bg-white/[0.02] px-5 py-3 space-y-2">
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex items-center gap-2 text-xs text-white/40">
                <Clock className="h-3.5 w-3.5" />
                <span>{timestamp ? new Date(timestamp).toLocaleString() : "—"}</span>
              </div>
              <div className="h-4 w-px bg-white/10" />
              <div className="text-xs text-white/40">
                Method: <span className="font-semibold text-white/60">{method}</span>
              </div>
              {trainingCutoff && (
                <>
                  <div className="h-4 w-px bg-white/10" />
                  <div className="text-xs text-white/40">
                    Training cutoff: <span className="font-semibold text-nvidia">{trainingCutoff}</span>
                  </div>
                </>
              )}
              <div className="h-4 w-px bg-white/10" />
              <div className="text-xs text-white/40">
                Split: <span className="font-semibold text-white/60">{splitMethod === "date" ? `📅 ±${analysisWindowDays}d around cutoff` : "📊 Last 60 rows (50/50)"}</span>
              </div>
            </div>

            {/* Date ranges */}
            <div className="flex flex-wrap items-center gap-4">
              <div className="text-xs text-white/40">
                📗 Reference <span className="text-white/25">(−{analysisWindowDays}d)</span>: <span className="font-semibold text-blue-400/70">{nReference.toLocaleString()} rows</span>
                {referenceStart && referenceEnd && (
                  <span className="ml-1 text-white/30">({referenceStart} → {referenceEnd})</span>
                )}
              </div>
              <div className="h-4 w-px bg-white/10" />
              <div className="text-xs text-white/40">
                📙 Current <span className="text-white/25">(+{analysisWindowDays}d)</span>: <span className="font-semibold text-purple-400/70">{nCurrent.toLocaleString()} rows</span>
                {currentStart && currentEnd && (
                  <span className="ml-1 text-white/30">({currentStart} → {currentEnd})</span>
                )}
              </div>
            </div>

            <p className="text-[10px] text-white/30">
              <Info className="mr-1 inline h-3 w-3 text-nvidia/40" />
              The model forecasts up to <strong className="text-nvidia/60">{analysisWindowDays} days</strong> ahead.
              PSI compares the <strong className="text-blue-400/50">{analysisWindowDays} days before</strong> training cutoff (last data the model saw) against
              the <strong className="text-purple-400/50">{analysisWindowDays} days after</strong> (production data it&apos;s predicting). This isolates real distribution shift from historical volatility.
            </p>
          </div>

          {/* ── Per-Feature PSI Detail ── */}
          {Object.keys(psiValues).length > 0 && (
            <div className="rounded-xl border border-surface-border bg-surface-card p-6">
              <h4 className="mb-1 text-sm font-bold text-white/80">
                PSI Values by Feature
              </h4>
              <p className="mb-4 text-[11px] text-white/40">
                Each feature is independently checked against the training distribution.
                Higher PSI = greater distribution shift.
              </p>

              <div className="space-y-3">
                {Object.entries(psiValues)
                  .sort(([, a], [, b]) => b - a)
                  .map(([feature, psi]) => {
                    const detail = featureDetails[feature];
                    return (
                      <div
                        key={feature}
                        className="rounded-lg border border-white/5 bg-white/[0.02] p-4"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <span className="text-base">{psiIcon(psi)}</span>
                            <div>
                              <p className="text-sm font-bold text-white/80">{feature}</p>
                              <p className={`text-[10px] font-semibold ${psiColor(psi)}`}>
                                {psiLabel(psi)}
                              </p>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className={`text-xl font-bold ${psiColor(psi)}`}>
                              {psi.toFixed(6)}
                            </p>
                            <p className="text-[10px] text-white/30">PSI score</p>
                          </div>
                        </div>

                        {/* PSI bar */}
                        <div className="mt-3">
                          <div className="relative h-3 w-full overflow-hidden rounded-full bg-white/5">
                            {/* Threshold markers */}
                            <div
                              className="absolute top-0 h-full w-px bg-amber-400/40"
                              style={{ left: `${(0.1 / 0.5) * 100}%` }}
                            />
                            <div
                              className="absolute top-0 h-full w-px bg-red-500/40"
                              style={{ left: `${(0.2 / 0.5) * 100}%` }}
                            />
                            {/* Value bar */}
                            <div
                              className={`h-full rounded-full transition-all duration-700 ${psiBg(psi)}`}
                              style={{ width: `${Math.min(psi / 0.5, 1) * 100}%` }}
                            />
                          </div>
                          <div className="mt-1 flex justify-between text-[9px] text-white/20">
                            <span>0</span>
                            <span className="text-amber-400/40">0.1</span>
                            <span className="text-red-500/40">0.2</span>
                            <span>0.5+</span>
                          </div>
                        </div>

                        {/* Distribution stats */}
                        {detail && detail.ref_mean !== 0 && (
                          <div className="mt-3 grid grid-cols-2 gap-3">
                            <div className="rounded-md bg-white/[0.03] p-2">
                              <p className="text-[9px] font-bold uppercase tracking-wider text-blue-400/70">
                                Reference (Training)
                              </p>
                              <div className="mt-1 flex gap-4 text-xs text-white/50">
                                <span>
                                  μ = <span className="font-mono font-semibold text-white/70">{detail.ref_mean.toFixed(2)}</span>
                                </span>
                                <span>
                                  σ = <span className="font-mono font-semibold text-white/70">{detail.ref_std.toFixed(2)}</span>
                                </span>
                              </div>
                            </div>
                            <div className="rounded-md bg-white/[0.03] p-2">
                              <p className="text-[9px] font-bold uppercase tracking-wider text-purple-400/70">
                                Current (Production)
                              </p>
                              <div className="mt-1 flex gap-4 text-xs text-white/50">
                                <span>
                                  μ = <span className="font-mono font-semibold text-white/70">{detail.cur_mean.toFixed(2)}</span>
                                </span>
                                <span>
                                  σ = <span className="font-mono font-semibold text-white/70">{detail.cur_std.toFixed(2)}</span>
                                </span>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
              </div>
            </div>
          )}

          {/* ── Trigger 2: Model Staleness Detail ── */}
          {staleness && (
            <div className={`rounded-xl border p-6 ${
              staleness.stale
                ? "border-red-500/30 bg-gradient-to-br from-red-500/5 to-red-900/5"
                : "border-green-500/30 bg-gradient-to-br from-green-500/5 to-green-900/5"
            }`}>
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-2xl">⏰</span>
                  <div>
                    <h4 className="text-sm font-bold text-white/80">
                      Trigger 2: Model Staleness
                    </h4>
                    <p className="text-[11px] text-white/40">
                      Checks if the model file is older than {Number(staleness.threshold_days)} days.
                      Financial markets evolve — periodic retraining captures recent patterns.
                    </p>
                  </div>
                </div>
                <div className={`rounded-lg px-3 py-1.5 text-xs font-bold ${
                  staleness.stale ? "bg-red-500/20 text-red-400" : "bg-green-500/20 text-green-400"
                }`}>
                  {staleness.stale ? <><XCircle className="mr-1 inline h-3.5 w-3.5" /> STALE</> : <><CheckCircle className="mr-1 inline h-3.5 w-3.5" /> FRESH</>}
                </div>
              </div>

              <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
                <div className="rounded-lg bg-white/[0.03] p-3">
                  <p className="text-[9px] font-bold uppercase tracking-wider text-white/30">Model Age</p>
                  <p className={`mt-1 text-2xl font-bold ${
                    staleness.stale ? "text-red-400" : "text-green-400"
                  }`}>
                    {Number(staleness.age_days).toFixed(1)}
                  </p>
                  <p className="text-[10px] text-white/30">days</p>
                </div>
                <div className="rounded-lg bg-white/[0.03] p-3">
                  <p className="text-[9px] font-bold uppercase tracking-wider text-white/30">Threshold</p>
                  <p className="mt-1 text-2xl font-bold text-white/60">{Number(staleness.threshold_days)}</p>
                  <p className="text-[10px] text-white/30">days max</p>
                </div>
                <div className="rounded-lg bg-white/[0.03] p-3">
                  <p className="text-[9px] font-bold uppercase tracking-wider text-white/30">Last Modified</p>
                  <p className="mt-1 text-sm font-semibold text-white/60">
                    {staleness.last_modified
                      ? new Date(String(staleness.last_modified)).toLocaleDateString()
                      : "N/A"}
                  </p>
                  <p className="text-[10px] text-white/30">model file date</p>
                </div>
                <div className="rounded-lg bg-white/[0.03] p-3">
                  <p className="text-[9px] font-bold uppercase tracking-wider text-white/30">Remaining</p>
                  <p className={`mt-1 text-2xl font-bold ${
                    Number(staleness.threshold_days) - Number(staleness.age_days) <= 5
                      ? "text-amber-400" : "text-green-400"
                  }`}>
                    {Math.max(0, Number(staleness.threshold_days) - Number(staleness.age_days)).toFixed(0)}
                  </p>
                  <p className="text-[10px] text-white/30">days until stale</p>
                </div>
              </div>

              {/* Staleness progress bar */}
              <div className="mt-3">
                <div className="relative h-3 w-full overflow-hidden rounded-full bg-white/5">
                  <div
                    className="absolute top-0 h-full w-px bg-red-500/50"
                    style={{ left: "100%" }}
                  />
                  <div
                    className={`h-full rounded-full transition-all duration-700 ${
                      staleness.stale ? "bg-red-500" : Number(staleness.age_days) / Number(staleness.threshold_days) > 0.7 ? "bg-amber-400" : "bg-green-500"
                    }`}
                    style={{ width: `${Math.min(Number(staleness.age_days) / Number(staleness.threshold_days), 1) * 100}%` }}
                  />
                </div>
                <div className="mt-1 flex justify-between text-[9px] text-white/20">
                  <span>0 days</span>
                  <span className="text-red-500/40">{Number(staleness.threshold_days)} days (threshold)</span>
                </div>
              </div>

              {staleness.reason != null && (
                <p className="mt-3 rounded-lg bg-white/[0.02] p-3 text-[11px] text-white/40">
                  <Info className="mr-1 inline h-3 w-3 text-white/20" />
                  {String(staleness.reason)}
                </p>
              )}
            </div>
          )}

          {/* ── Trigger 3: Prediction CI Breach Detail ── */}
          {ciBreach && (
            <div className={`rounded-xl border p-6 ${
              ciBreach.status === "skipped"
                ? "border-white/10 bg-white/[0.02]"
                : ciBreach.breach_detected
                  ? "border-red-500/30 bg-gradient-to-br from-red-500/5 to-red-900/5"
                  : "border-green-500/30 bg-gradient-to-br from-green-500/5 to-green-900/5"
            }`}>
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-2xl">📉</span>
                  <div>
                    <h4 className="text-sm font-bold text-white/80">
                      Trigger 3: Prediction CI Breach
                    </h4>
                    <p className="text-[11px] text-white/40">
                      Checks if actual stock prices are falling outside the model&apos;s
                      95% prediction confidence interval — signals <strong className="text-white/60">concept drift</strong>.
                    </p>
                  </div>
                </div>
                <div className={`rounded-lg px-3 py-1.5 text-xs font-bold ${
                  ciBreach.status === "skipped"
                    ? "bg-white/10 text-white/40"
                    : ciBreach.breach_detected
                      ? "bg-red-500/20 text-red-400"
                      : "bg-green-500/20 text-green-400"
                }`}>
                  {ciBreach.status === "skipped" ? <><SkipForward className="mr-1 inline h-3.5 w-3.5" /> SKIPPED</> : ciBreach.breach_detected ? <><XCircle className="mr-1 inline h-3.5 w-3.5 text-red-400" /> BREACH</> : <><CheckCircle className="mr-1 inline h-3.5 w-3.5 text-green-400" /> WITHIN CI</>}
                </div>
              </div>

              {ciBreach.status === "skipped" ? (
                <div className="mt-4 rounded-lg bg-white/[0.03] p-4">
                  <p className="text-xs text-white/40">
                    <Info className="mr-1 inline h-3.5 w-3.5 text-white/20" />
                    {String(ciBreach.reason ?? "No prediction/actual pairs available for CI breach analysis. Ensure the model is loaded and historical data exists.")}
                  </p>
                </div>
              ) : (
                <>
                  <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
                    <div className="rounded-lg bg-white/[0.03] p-3">
                      <p className="text-[9px] font-bold uppercase tracking-wider text-white/30">Breach Ratio</p>
                      <p className={`mt-1 text-2xl font-bold ${
                        ciBreach.breach_detected ? "text-red-400" : "text-green-400"
                      }`}>
                        {(Number(ciBreach.breach_ratio ?? 0) * 100).toFixed(1)}%
                      </p>
                      <p className="text-[10px] text-white/30">of actuals outside CI</p>
                    </div>
                    <div className="rounded-lg bg-white/[0.03] p-3">
                      <p className="text-[9px] font-bold uppercase tracking-wider text-white/30">Threshold</p>
                      <p className="mt-1 text-2xl font-bold text-white/60">
                        {(Number(ciBreach.breach_threshold ?? 0.2) * 100).toFixed(0)}%
                      </p>
                      <p className="text-[10px] text-white/30">max allowed breach</p>
                    </div>
                    <div className="rounded-lg bg-white/[0.03] p-3">
                      <p className="text-[9px] font-bold uppercase tracking-wider text-white/30">Breaches</p>
                      <p className={`mt-1 text-2xl font-bold ${
                        Number(ciBreach.n_breaches ?? 0) > 0 ? "text-amber-400" : "text-green-400"
                      }`}>
                        {Number(ciBreach.n_breaches ?? 0)}
                      </p>
                      <p className="text-[10px] text-white/30">of {Number(ciBreach.n_total ?? 0)} total</p>
                    </div>
                    <div className="rounded-lg bg-white/[0.03] p-3">
                      <p className="text-[9px] font-bold uppercase tracking-wider text-white/30">Residual σ</p>
                      <p className="mt-1 text-2xl font-bold text-white/60">
                        {Number(ciBreach.residual_std ?? 0).toFixed(2)}
                      </p>
                      <p className="text-[10px] text-white/30">prediction error std</p>
                    </div>
                  </div>

                  {/* CI breach progress bar */}
                  <div className="mt-3">
                    <div className="relative h-3 w-full overflow-hidden rounded-full bg-white/5">
                      <div
                        className="absolute top-0 h-full w-px bg-red-500/50"
                        style={{ left: `${Number(ciBreach.breach_threshold ?? 0.2) * 100}%` }}
                      />
                      <div
                        className={`h-full rounded-full transition-all duration-700 ${
                          ciBreach.breach_detected ? "bg-red-500" : Number(ciBreach.breach_ratio ?? 0) / Number(ciBreach.breach_threshold ?? 0.2) > 0.7 ? "bg-amber-400" : "bg-green-500"
                        }`}
                        style={{ width: `${Math.min(Number(ciBreach.breach_ratio ?? 0), 1) * 100}%` }}
                      />
                    </div>
                    <div className="mt-1 flex justify-between text-[9px] text-white/20">
                      <span>0%</span>
                      <span className="text-red-500/40">{(Number(ciBreach.breach_threshold ?? 0.2) * 100).toFixed(0)}% threshold</span>
                      <span>100%</span>
                    </div>
                  </div>

                  {/* Methodology */}
                  <div className="mt-3 rounded-lg bg-white/[0.02] p-3">
                    <p className="text-[10px] text-white/40">
                      <Info className="mr-1 inline h-3 w-3 text-white/20" />
                      CI built from residual standard error (σ = {Number(ciBreach.residual_std ?? 0).toFixed(2)}).
                      {" "}CI = predicted ± z<sub>α/2</sub> × σ at{" "}
                      {(Number(ciBreach.confidence_level ?? 0.95) * 100).toFixed(0)}% confidence.
                      Mean residual (bias): {Number(ciBreach.mean_residual ?? 0).toFixed(2)}.
                    </p>
                  </div>
                </>
              )}
            </div>
          )}

          {/* ── Interpretation Guide ── */}
          <div className="rounded-xl border border-surface-border bg-surface-card p-6">
            <h4 className="mb-3 flex items-center gap-2 text-sm font-bold text-white/80">
              <Info className="h-4 w-4 text-nvidia" />
              How to Interpret These Results
            </h4>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              <div className="rounded-lg border border-green-500/20 bg-green-500/5 p-4">
                <div className="mb-2 flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-green-400" />
                  <p className="text-xs font-bold text-green-400">All Features Stable</p>
                </div>
                <p className="text-[11px] text-white/40">
                  The production data closely matches the training distribution.
                  Your model should perform as expected. No action needed —
                  continue monitoring.
                </p>
              </div>
              <div className="rounded-lg border border-amber-400/20 bg-amber-400/5 p-4">
                <div className="mb-2 flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-amber-400" />
                  <p className="text-xs font-bold text-amber-400">Warning (PSI 0.1–0.2)</p>
                </div>
                <p className="text-[11px] text-white/40">
                  Moderate distribution shift detected. This could be natural market
                  volatility or early signs of drift. Monitor closely and compare
                  model accuracy on recent predictions. Consider running
                  Champion-Challenger.
                </p>
              </div>
              <div className="rounded-lg border border-red-500/20 bg-red-500/5 p-4">
                <div className="mb-2 flex items-center gap-2">
                  <XCircle className="h-4 w-4 text-red-400" />
                  <p className="text-xs font-bold text-red-400">Retrain (PSI ≥ 0.2)</p>
                </div>
                <p className="text-[11px] text-white/40">
                  Significant distribution change. The model was trained on
                  data that no longer represents current market conditions.
                  Retraining with recent data via the <strong>Champion-Challenger
                  pipeline</strong> (with Optuna HPO) is strongly recommended.
                </p>
              </div>
            </div>
          </div>

          {/* ── What to Do Next ── */}
          <div className="rounded-xl border border-nvidia/20 bg-nvidia/5 p-5">
            <h4 className="mb-2 text-sm font-bold text-nvidia">📋 Recommended Next Steps</h4>
            <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
              {[
                {
                  step: "1. Verify with Metrics",
                  desc: "Check the Metrics page for recent RMSE/MAE trends to confirm if prediction accuracy is degrading.",
                  done: false,
                },
                {
                  step: "2. Run Champion-Challenger",
                  desc: "Go to the Champion-Challenger tab to train an Optuna-optimized challenger with fresh data.",
                  done: false,
                },
                {
                  step: "3. Review Predictions",
                  desc: "Compare recent predictions vs actuals on the Predictions page to spot systematic errors.",
                  done: false,
                },
                {
                  step: "4. Check Model Staleness",
                  desc: "The multi-trigger system also checks model age (≥30 days) and prediction CI breach (>20% outside 95% CI).",
                  done: false,
                },
              ].map((s) => (
                <div key={s.step} className="flex gap-3 rounded-lg border border-white/5 bg-white/[0.02] p-3">
                  <span className="mt-0.5 text-sm text-nvidia">▸</span>
                  <div>
                    <p className="text-xs font-bold text-white/70">{s.step}</p>
                    <p className="text-[10px] text-white/40">{s.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
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

  // ── Map the actual API structure ──
  const comparison = data?.comparison as Record<string, unknown> | undefined;
  const champion = comparison?.champion as Record<string, unknown> | undefined;
  const challenger = comparison?.challenger as Record<string, unknown> | undefined;
  const promoted = Boolean(data?.promoted ?? comparison?.promote);
  const reason = String(comparison?.reason ?? "");
  const rmseDeltaPct = Number(comparison?.rmse_delta_pct ?? 0);
  const timestamp = String(data?.timestamp ?? "");
  const driftDetected = Boolean(data?.drift_detected);
  const retrained = Boolean(data?.retrained);
  const trainingResult = data?.training_result as Record<string, unknown> | undefined;

  // Metric labels & formatting
  const metricLabel: Record<string, string> = {
    rmse: "RMSE",
    mae: "MAE",
    mape: "MAPE",
    r2: "R²",
    directional_accuracy: "Directional Acc",
    timestamp: "Evaluated at",
  };
  const metricKeys = ["rmse", "mae", "mape", "r2", "directional_accuracy"];

  const fmtVal = (k: string, v: unknown) => {
    if (k === "timestamp") return String(v).slice(0, 19);
    if (typeof v === "number") return v < 0.01 ? v.toExponential(3) : v.toFixed(4);
    return String(v);
  };

  const betterSide = (k: string, champV: number, challV: number) => {
    if (k === "r2" || k === "directional_accuracy") return challV > champV ? "challenger" : champV > challV ? "champion" : "tie";
    return challV < champV ? "challenger" : champV < challV ? "champion" : "tie";
  };

  return (
    <div className="space-y-5">
      {/* Header explanation */}
      <div className="rounded-lg border border-white/5 bg-white/[0.02] px-5 py-3">
        <p className="text-xs text-white/50">
          <span className="mr-1 text-nvidia font-semibold">Champion-Challenger</span>
          automatically retrains a new model (challenger) when drift is detected, compares it head-to-head against the
          current production model (champion), and promotes the winner. This ensures the deployed model is always the best available.
        </p>
      </div>

      {/* Action buttons */}
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
          {running ? "Running…" : "🚀 Run Pipeline"}
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-400">
          {error}
        </div>
      )}

      {loading && <LoadingSpinner text="Loading results…" />}

      {data && !loading && (
        <div className="space-y-5">
          {/* ── Pipeline metadata bar ── */}
          <div className="flex flex-wrap items-center gap-4 rounded-lg border border-white/5 bg-white/[0.02] px-5 py-3">
            <div className="flex items-center gap-2 text-xs text-white/40">
              <Clock className="h-3.5 w-3.5" />
              <span>{timestamp ? new Date(timestamp).toLocaleString() : "—"}</span>
            </div>
            <div className="h-4 w-px bg-white/10" />
            <div className="text-xs text-white/40">
              Drift: <span className={`font-semibold ${driftDetected ? "text-red-400" : "text-green-400"}`}>
                {driftDetected ? "🔴 Detected" : "🟢 None"}
              </span>
            </div>
            <div className="h-4 w-px bg-white/10" />
            <div className="text-xs text-white/40">
              Retrained: <span className={`font-semibold ${retrained ? "text-nvidia" : "text-white/60"}`}>
                {retrained ? "✅ Yes" : "— No"}
              </span>
            </div>
            {typeof trainingResult?.run_id === "string" && trainingResult.run_id.length > 0 && (
              <>
                <div className="h-4 w-px bg-white/10" />
                <div className="text-xs text-white/40">
                  MLflow run: <span className="font-mono text-white/50">{(trainingResult.run_id as string).slice(0, 12)}…</span>
                </div>
              </>
            )}
          </div>

          {/* ── Head-to-head comparison table ── */}
          {champion && challenger && (
            <div className="overflow-hidden rounded-xl border border-surface-border bg-surface-card">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/5 bg-white/[0.02]">
                    <th className="px-5 py-3 text-left text-xs font-medium uppercase tracking-wider text-white/40">Metric</th>
                    <th className="px-5 py-3 text-right text-xs font-medium uppercase tracking-wider text-nvidia">🏆 Champion</th>
                    <th className="px-5 py-3 text-right text-xs font-medium uppercase tracking-wider text-sky-400">⚔️ Challenger</th>
                    <th className="px-5 py-3 text-center text-xs font-medium uppercase tracking-wider text-white/40">Winner</th>
                  </tr>
                </thead>
                <tbody>
                  {metricKeys.map((k) => {
                    const cv = Number(champion[k] ?? 0);
                    const av = Number(challenger[k] ?? 0);
                    const winner = betterSide(k, cv, av);
                    return (
                      <tr key={k} className="border-b border-white/5 last:border-b-0">
                        <td className="px-5 py-2.5 text-white/60 font-medium">{metricLabel[k] ?? k}</td>
                        <td className={`px-5 py-2.5 text-right font-mono ${winner === "champion" ? "text-nvidia font-semibold" : "text-white/50"}`}>
                          {fmtVal(k, cv)}
                        </td>
                        <td className={`px-5 py-2.5 text-right font-mono ${winner === "challenger" ? "text-sky-400 font-semibold" : "text-white/50"}`}>
                          {fmtVal(k, av)}
                        </td>
                        <td className="px-5 py-2.5 text-center">
                          {winner === "challenger" ? <span className="text-sky-400">⚔️</span>
                            : winner === "champion" ? <span className="text-nvidia">🏆</span>
                            : <span className="text-white/30">—</span>}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* ── Champion & Challenger detail cards (side by side) ── */}
          <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
            {/* Champion card */}
            <div className="rounded-xl border border-nvidia/30 bg-surface-card p-5">
              <div className="mb-3 flex items-center gap-2">
                <span className="text-2xl">🏆</span>
                <h4 className="text-lg font-semibold text-nvidia">Champion</h4>
              </div>
              {champion ? (
                <div className="space-y-1.5">
                  {Object.entries(champion).map(([k, v]) => (
                    <div key={k} className="flex justify-between text-sm">
                      <span className="text-white/40">{metricLabel[k] ?? k}</span>
                      <span className="font-mono text-white/70">{fmtVal(k, v)}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-white/40">No champion data — click <strong>Load Results</strong> or <strong>Run Pipeline</strong></p>
              )}
            </div>

            {/* Challenger card */}
            <div className="rounded-xl border border-sky-400/30 bg-surface-card p-5">
              <div className="mb-3 flex items-center gap-2">
                <span className="text-2xl">⚔️</span>
                <h4 className="text-lg font-semibold text-sky-400">Challenger</h4>
              </div>
              {challenger ? (
                <div className="space-y-1.5">
                  {Object.entries(challenger).map(([k, v]) => (
                    <div key={k} className="flex justify-between text-sm">
                      <span className="text-white/40">{metricLabel[k] ?? k}</span>
                      <span className="font-mono text-white/70">{fmtVal(k, v)}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-white/40">No challenger data — click <strong>Load Results</strong> or <strong>Run Pipeline</strong></p>
              )}
            </div>
          </div>

          {/* ── Promotion verdict ── */}
          <div className={`rounded-xl border p-5 ${promoted ? "border-green-500/30 bg-green-500/5" : "border-amber-500/30 bg-amber-500/5"}`}>
            <div className="flex items-start gap-3">
              {promoted ? (
                <CheckCircle className="mt-0.5 h-6 w-6 shrink-0 text-green-400" />
              ) : (
                <XCircle className="mt-0.5 h-6 w-6 shrink-0 text-amber-400" />
              )}
              <div className="space-y-1">
                <p className={`font-semibold ${promoted ? "text-green-400" : "text-amber-400"}`}>
                  {promoted ? "Challenger Promoted! 🎉" : "Champion Retained"}
                </p>
                <p className="text-sm text-white/50">
                  {reason || (promoted ? "Challenger outperformed the champion model." : "Current champion remains the best.")}
                </p>
                {rmseDeltaPct !== 0 && (
                  <div className="mt-2 inline-flex items-center gap-2 rounded-full bg-white/5 px-3 py-1">
                    <TrendingDown className="h-3.5 w-3.5 text-green-400" />
                    <span className="text-xs font-semibold text-green-400">
                      RMSE improved {Math.abs(rmseDeltaPct * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ──────────── Model History Tab ──────────── */
interface RunRecord {
  run_id: string;
  run_name: string;
  experiment: string;
  status: string;
  start_time: string | null;
  end_time: string | null;
  duration_s: number | null;
  metrics: Record<string, number>;
  params: Record<string, string>;
  source: string;
}

function HistoryTab() {
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<string>("start_time");
  const [sortAsc, setSortAsc] = useState(false);

  const loadHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.monitoring.runsHistory();
      setRuns((res.runs ?? []) as RunRecord[]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  };

  // Auto-load on mount
  useEffect(() => {
    loadHistory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSort = (key: string) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(key === "run_name" || key === "experiment");
    }
  };

  const sorted = [...runs].sort((a, b) => {
    let va: string | number = "";
    let vb: string | number = "";
    if (sortKey === "start_time") { va = a.start_time ?? ""; vb = b.start_time ?? ""; }
    else if (sortKey === "run_name") { va = a.run_name; vb = b.run_name; }
    else if (sortKey === "experiment") { va = a.experiment; vb = b.experiment; }
    else if (sortKey === "status") { va = a.status; vb = b.status; }
    else if (sortKey === "duration_s") { va = a.duration_s ?? 0; vb = b.duration_s ?? 0; }
    else { va = a.metrics[sortKey] ?? 0; vb = b.metrics[sortKey] ?? 0; }
    if (va < vb) return sortAsc ? -1 : 1;
    if (va > vb) return sortAsc ? 1 : -1;
    return 0;
  });

  const fmtMetric = (v: number) => {
    if (v === 0) return "—";
    if (Math.abs(v) < 0.01) return v.toExponential(2);
    if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 1 });
    return v.toFixed(4);
  };

  const fmtDate = (iso: string | null) => {
    if (!iso) return "—";
    return new Date(iso).toLocaleDateString("pt-BR", { day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit", minute: "2-digit" });
  };

  const fmtDuration = (s: number | null) => {
    if (s === null || s === undefined) return "—";
    if (s < 60) return `${s.toFixed(0)}s`;
    return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  };

  const statusBadge = (st: string) => {
    const colors: Record<string, string> = {
      FINISHED: "text-green-400 bg-green-500/10 border-green-500/20",
      RUNNING: "text-blue-400 bg-blue-500/10 border-blue-500/20",
      FAILED: "text-red-400 bg-red-500/10 border-red-500/20",
    };
    return colors[st] ?? "text-white/40 bg-white/5 border-white/10";
  };

  // Key metrics to show in the table columns
  const keyMetrics = ["best_val_loss", "test_rmse", "test_r2_score", "test_mape"];
  const keyMetricLabels: Record<string, string> = {
    best_val_loss: "Best Val Loss",
    test_rmse: "Test RMSE",
    test_r2_score: "Test R²",
    test_mape: "Test MAPE %",
  };

  const SortHeader = ({ label, k }: { label: string; k: string }) => (
    <th
      className="cursor-pointer select-none px-3 py-2.5 text-left text-xs font-medium uppercase tracking-wider text-white/40 hover:text-white/70 transition-colors"
      onClick={() => handleSort(k)}
    >
      {label} {sortKey === k ? (sortAsc ? "↑" : "↓") : ""}
    </th>
  );

  // Find best value per metric for highlighting
  const bestValues: Record<string, number> = {};
  for (const mk of keyMetrics) {
    const vals = runs.map((r) => r.metrics[mk]).filter((v) => v !== undefined && v !== 0);
    if (vals.length > 0) {
      bestValues[mk] = mk === "test_r2_score" ? Math.max(...vals) : Math.min(...vals);
    }
  }

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="rounded-lg border border-white/5 bg-white/[0.02] px-5 py-3">
        <p className="text-xs text-white/50">
          <span className="mr-1 text-nvidia font-semibold">Model History</span>
          — every training run logged to MLflow, with hyperparameters and evaluation metrics.
          The <span className="text-nvidia">best value</span> in each metric column is highlighted.
        </p>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={loadHistory}
          disabled={loading}
          className="flex items-center gap-2 rounded-lg bg-nvidia px-4 py-2 text-sm font-semibold text-black hover:bg-nvidia-dark disabled:opacity-50"
        >
          <History className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          {loading ? "Loading…" : "Load Run History"}
        </button>
        {runs.length > 0 && (
          <span className="text-xs text-white/40">{runs.length} runs found</span>
        )}
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-400">{error}</div>
      )}

      {loading && <LoadingSpinner text="Loading MLflow runs…" />}

      {runs.length > 0 && !loading && (
        <div className="overflow-x-auto rounded-xl border border-surface-border bg-surface-card">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/5 bg-white/[0.02]">
                <SortHeader label="Date" k="start_time" />
                <SortHeader label="Experiment" k="experiment" />
                <SortHeader label="Run" k="run_name" />
                <th className="px-3 py-2.5 text-left text-xs font-medium uppercase tracking-wider text-white/40">Status</th>
                <SortHeader label="Duration" k="duration_s" />
                {keyMetrics.map((mk) => (
                  <SortHeader key={mk} label={keyMetricLabels[mk] ?? mk} k={mk} />
                ))}
                <th className="px-3 py-2.5 w-8" />
              </tr>
            </thead>
            <tbody>
              {sorted.map((run, idx) => {
                const isExpanded = expandedId === run.run_id;
                return (
                  <tr key={`${run.run_id}-${idx}`} className="border-b border-white/5 last:border-b-0 hover:bg-white/[0.02] transition-colors">
                    <td className="px-3 py-2 text-white/50 whitespace-nowrap text-xs">{fmtDate(run.start_time)}</td>
                    <td className="px-3 py-2">
                      <span className="rounded bg-white/5 px-2 py-0.5 text-xs text-white/60">{run.experiment}</span>
                    </td>
                    <td className="px-3 py-2 font-medium text-white/70 whitespace-nowrap">{run.run_name}</td>
                    <td className="px-3 py-2">
                      <span className={`inline-block rounded-full border px-2 py-0.5 text-[10px] font-semibold ${statusBadge(run.status)}`}>
                        {run.status}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-white/50 text-xs font-mono">{fmtDuration(run.duration_s)}</td>
                    {keyMetrics.map((mk) => {
                      const v = run.metrics[mk];
                      const isBest = v !== undefined && v !== 0 && v === bestValues[mk];
                      return (
                        <td key={mk} className={`px-3 py-2 font-mono text-xs ${isBest ? "text-nvidia font-bold" : "text-white/50"}`}>
                          {v !== undefined ? fmtMetric(v) : "—"}
                        </td>
                      );
                    })}
                    <td className="px-3 py-2">
                      <button
                        onClick={() => setExpandedId(isExpanded ? null : run.run_id)}
                        className="text-white/30 hover:text-white/70 text-xs transition-colors"
                        title="Show details"
                      >
                        {isExpanded ? "▲" : "▼"}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {/* Expanded detail panel */}
          {expandedId && (() => {
            const run = runs.find((r) => r.run_id === expandedId);
            if (!run) return null;

            const allMetricKeys = Object.keys(run.metrics).sort();
            const paramKeys = Object.keys(run.params).sort();

            return (
              <div className="border-t border-white/5 bg-white/[0.01] px-5 py-4">
                <div className="mb-3 flex items-center gap-3">
                  <span className="font-mono text-xs text-white/30">ID: {run.run_id}</span>
                  <span className="text-xs text-white/30">|</span>
                  <span className="text-xs text-white/30">Source: {run.source}</span>
                </div>
                <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                  {/* Metrics */}
                  <div>
                    <h5 className="mb-2 text-xs font-semibold uppercase text-white/40">📊 All Metrics</h5>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                      {allMetricKeys.map((k) => (
                        <div key={k} className="flex justify-between text-xs">
                          <span className="text-white/40">{k}</span>
                          <span className="font-mono text-white/60">{fmtMetric(run.metrics[k])}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  {/* Params */}
                  <div>
                    <h5 className="mb-2 text-xs font-semibold uppercase text-white/40">⚙️ Hyperparameters</h5>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                      {paramKeys.map((k) => (
                        <div key={k} className="flex justify-between text-xs">
                          <span className="text-white/40">{k}</span>
                          <span className="font-mono text-white/60">{run.params[k]}</span>
                        </div>
                      ))}
                      {paramKeys.length === 0 && (
                        <span className="text-xs text-white/30">No hyperparameters logged</span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
}

/* ──────────── Telemetry Tab ──────────── */

interface ServiceLink {
  label: string;
  href: string;
  icon: "doc" | "github" | "external";
}

interface ServiceStatus {
  name: string;
  url: string;
  healthUrl: string;
  desc: string;
  details: string;
  icon: string;
  port: number;
  links?: ServiceLink[];
  status: "checking" | "online" | "offline" | "idle";
  latencyMs: number | null;
}

const SERVICE_DEFS: Omit<ServiceStatus, "status" | "latencyMs">[] = [
  {
    name: "FastAPI",
    url: "http://localhost:8000",
    healthUrl: "http://localhost:8000/health",
    desc: "REST API — predictions, evaluation, monitoring",
    details: "The core backend. Exposes all endpoints: stock price predictions (LSTM model), RAGAS & LLM-Judge evaluation, Champion-Challenger pipeline, drift detection, LIME/Permutation explainability, and the RAG agent. Swagger docs at /docs.",
    icon: "⚡",
    port: 8000,
    links: [
      { label: "Swagger", href: "http://localhost:8000/docs", icon: "doc" },
      { label: "ReDoc", href: "http://localhost:8000/redoc", icon: "doc" },
      { label: "GitHub", href: "https://github.com/LucasTechAI/nvidia-mlops-platform", icon: "github" },
    ],
  },
  {
    name: "Next.js",
    url: "http://localhost:3001",
    healthUrl: "http://localhost:3001",
    desc: "Dashboard frontend (this app)",
    details: "The React/Next.js frontend you're looking at right now. Server-side rendered with App Router. Consumes all FastAPI endpoints and renders interactive charts, tables, and evaluation results.",
    icon: "🖥️",
    port: 3001,
    links: [
      { label: "GitHub", href: "https://github.com/LucasTechAI/nvidia-mlops-platform", icon: "github" },
    ],
  },
  {
    name: "MLflow",
    url: "http://localhost:5000",
    healthUrl: "http://localhost:5000",
    desc: "Experiment tracking & model registry",
    details: "Tracks every training run: hyperparameters, metrics (RMSE, MAE, R²), model artifacts, and governance tags (git SHA, author). The Champion-Challenger pipeline logs Optuna HPO trials and promotion decisions here.",
    icon: "🔬",
    port: 5000,
    links: [
      { label: "Docs", href: "https://mlflow.org/docs/latest/index.html", icon: "doc" },
      { label: "GitHub", href: "https://github.com/LucasTechAI/nvidia-mlops-platform", icon: "github" },
    ],
  },
  {
    name: "Prometheus",
    url: "http://localhost:9090",
    healthUrl: "http://localhost:9090/-/ready",
    desc: "Metrics collection & alerting rules",
    details: "Scrapes /metrics from FastAPI every 15s. Collects request latency (p50/p95/p99), throughput, error rates, prediction counts, and drift scores. Powers Grafana dashboards and can trigger alerts on SLA breaches.",
    icon: "🔥",
    port: 9090,
    links: [
      { label: "Targets", href: "http://localhost:9090/targets", icon: "external" },
      { label: "GitHub", href: "https://github.com/LucasTechAI/nvidia-mlops-platform", icon: "github" },
    ],
  },
  {
    name: "Grafana",
    url: "http://localhost:3000",
    healthUrl: "http://localhost:3000/api/health",
    desc: "Dashboards & visualization (admin/admin)",
    details: "Pre-configured with provisioned dashboards for API latency, throughput, error rate, and model drift over time. Connects to Prometheus as data source. Login: admin/admin. Ideal for real-time production monitoring.",
    icon: "📊",
    port: 3000,
    links: [
      { label: "Dashboards", href: "http://localhost:3000/dashboards", icon: "external" },
      { label: "GitHub", href: "https://github.com/LucasTechAI/nvidia-mlops-platform", icon: "github" },
    ],
  },
  {
    name: "Optuna Dashboard",
    url: "http://localhost:8080",
    healthUrl: "http://localhost:8080",
    desc: "Hyperparameter optimization studies",
    details: "Visualizes all Optuna HPO studies from Champion-Challenger runs. Shows trial history, parameter importance, optimization convergence (contour/parallel coordinate plots), and best hyperparameter combinations.",
    icon: "🎯",
    port: 8080,
    links: [
      { label: "GitHub", href: "https://github.com/LucasTechAI/nvidia-mlops-platform", icon: "github" },
    ],
  },
];

function TelemetryTab() {
  const [services, setServices] = useState<ServiceStatus[]>(
    SERVICE_DEFS.map((s) => ({ ...s, status: "idle", latencyMs: null }))
  );
  const [checking, setChecking] = useState(false);
  const [lastCheck, setLastCheck] = useState<string | null>(null);
  const [expandedInfo, setExpandedInfo] = useState<string | null>(null);

  // FastAPI detailed health
  const [apiHealth, setApiHealth] = useState<Record<string, unknown> | null>(null);

  // Auto-check on mount
  useEffect(() => {
    checkAllServices();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const checkAllServices = async () => {
    setChecking(true);

    // Mark all as "checking"
    setServices((prev) => prev.map((s) => ({ ...s, status: "checking", latencyMs: null })));

    const results: ServiceStatus[] = [];

    for (const def of SERVICE_DEFS) {
      const start = performance.now();
      let status: ServiceStatus["status"] = "offline";
      let latencyMs: number | null = null;

      try {
        if (def.name === "FastAPI") {
          // Use our API client for FastAPI (avoids CORS)
          const res = await api.health.check();
          latencyMs = Math.round(performance.now() - start);
          status = res?.status === "healthy" || res?.status === "degraded" ? "online" : "offline";
          setApiHealth(res);
        } else if (def.name === "Next.js") {
          // We're already on Next.js, so it's online
          latencyMs = 0;
          status = "online";
        } else {
          // External services — fetch with timeout
          const controller = new AbortController();
          const timeout = setTimeout(() => controller.abort(), 5000);
          try {
            const r = await fetch(def.healthUrl, {
              mode: "no-cors",
              signal: controller.signal,
            });
            clearTimeout(timeout);
            latencyMs = Math.round(performance.now() - start);
            // no-cors returns opaque response (status 0), but if it didn't throw → service is reachable
            status = r.status === 0 || (r.status >= 200 && r.status < 500) ? "online" : "offline";
          } catch {
            clearTimeout(timeout);
            latencyMs = null;
            status = "offline";
          }
        }
      } catch {
        status = "offline";
        latencyMs = null;
      }

      results.push({ ...def, status, latencyMs });
    }

    setServices(results);
    setLastCheck(new Date().toLocaleTimeString());
    setChecking(false);
  };

  const onlineCount = services.filter((s) => s.status === "online").length;
  const totalCount = services.length;
  const allOnline = onlineCount === totalCount && services[0].status !== "idle";
  const anyOffline = services.some((s) => s.status === "offline");

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold">📡 Service Health Monitor</h3>
          <p className="text-xs text-white/40">
            Check if all {totalCount} platform services are running correctly
          </p>
        </div>
        <div className="flex items-center gap-3">
          {lastCheck && (
            <span className="text-xs text-white/30">
              Last check: {lastCheck}
            </span>
          )}
          <button
            onClick={checkAllServices}
            disabled={checking}
            className="flex items-center gap-2 rounded-lg bg-nvidia px-4 py-2 text-sm font-semibold text-black hover:bg-nvidia-dark disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${checking ? "animate-spin" : ""}`} />
            {checking ? "Checking…" : "Check All Services"}
          </button>
        </div>
      </div>

      {/* Summary bar */}
      {services[0].status !== "idle" && (
        <div className={`flex items-center gap-3 rounded-lg border px-5 py-3 ${
          allOnline
            ? "border-green-500/30 bg-green-500/5"
            : anyOffline
              ? "border-red-500/30 bg-red-500/5"
              : "border-amber-500/30 bg-amber-500/5"
        }`}>
          {allOnline ? (
            <CheckCircle className="h-5 w-5 text-green-400" />
          ) : anyOffline ? (
            <XCircle className="h-5 w-5 text-red-400" />
          ) : (
            <AlertTriangle className="h-5 w-5 text-amber-400" />
          )}
          <span className={`text-sm font-semibold ${
            allOnline ? "text-green-400" : anyOffline ? "text-red-400" : "text-amber-400"
          }`}>
            {allOnline
              ? `All ${totalCount} services online ✅`
              : `${onlineCount}/${totalCount} services online`}
          </span>
        </div>
      )}

      {/* Service cards grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {services.map((svc) => {
          const statusColor =
            svc.status === "online"
              ? "border-green-500/30"
              : svc.status === "offline"
                ? "border-red-500/30"
                : svc.status === "checking"
                  ? "border-amber-500/30"
                  : "border-surface-border";

          const statusDot =
            svc.status === "online"
              ? "bg-green-400"
              : svc.status === "offline"
                ? "bg-red-400"
                : svc.status === "checking"
                  ? "bg-amber-400 animate-pulse"
                  : "bg-white/20";

          const statusLabel =
            svc.status === "online"
              ? "Online"
              : svc.status === "offline"
                ? "Offline"
                : svc.status === "checking"
                  ? "Checking…"
                  : "Not checked";

          return (
            <div
              key={svc.name}
              className={`rounded-xl border bg-surface-card p-5 transition-all ${statusColor}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-2xl">{svc.icon}</span>
                  <div>
                    <div className="flex items-center gap-1.5">
                      <h4 className="font-semibold text-white">{svc.name}</h4>
                      <button
                        onClick={() => setExpandedInfo(expandedInfo === svc.name ? null : svc.name)}
                        className={`rounded-full p-0.5 transition-colors ${
                          expandedInfo === svc.name
                            ? "text-nvidia"
                            : "text-white/25 hover:text-white/60"
                        }`}
                        title={`About ${svc.name}`}
                      >
                        <Info className="h-3.5 w-3.5" />
                      </button>
                    </div>
                    <p className="text-xs text-white/40">{svc.desc}</p>
                  </div>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className={`h-2.5 w-2.5 rounded-full ${statusDot}`} />
                  <span className={`text-xs font-medium ${
                    svc.status === "online"
                      ? "text-green-400"
                      : svc.status === "offline"
                        ? "text-red-400"
                        : "text-white/40"
                  }`}>
                    {statusLabel}
                  </span>
                </div>
              </div>

              {/* Expandable info panel */}
              {expandedInfo === svc.name && (
                <div className="mt-3 rounded-lg border border-nvidia/20 bg-nvidia/5 px-4 py-3">
                  <p className="text-xs leading-relaxed text-white/60">{svc.details}</p>
                </div>
              )}

              <div className="mt-3 flex items-center justify-between border-t border-white/5 pt-3">
                <div className="flex items-center gap-4 text-xs text-white/40">
                  <span>Port <span className="font-mono text-white/60">{svc.port}</span></span>
                  {svc.latencyMs !== null && (
                    <span>
                      Latency{" "}
                      <span className={`font-mono ${
                        svc.latencyMs < 100
                          ? "text-green-400"
                          : svc.latencyMs < 500
                            ? "text-amber-400"
                            : "text-red-400"
                      }`}>
                        {svc.latencyMs}ms
                      </span>
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  {(svc.links ?? []).map((lnk) => (
                    <a
                      key={lnk.label}
                      href={lnk.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1 rounded-md border border-white/5 bg-white/[0.03] px-2 py-1 text-[11px] text-white/40 hover:border-nvidia/30 hover:text-nvidia transition-colors"
                    >
                      {lnk.icon === "github" ? <Github className="h-3 w-3" /> : lnk.icon === "doc" ? <FileText className="h-3 w-3" /> : <ExternalLink className="h-3 w-3" />}
                      {lnk.label}
                    </a>
                  ))}
                  <a
                    href={svc.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1 text-xs text-white/30 hover:text-nvidia"
                  >
                    Open <ExternalLink className="h-3 w-3" />
                  </a>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* FastAPI detailed health */}
      {apiHealth && (
        <div className="rounded-xl border border-surface-border bg-surface-card p-6">
          <h3 className="mb-4 text-lg font-semibold">🏥 FastAPI Detailed Health</h3>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {Object.entries(apiHealth)
              .filter(([k]) => k !== "status")
              .map(([k, v]) => {
                const isBoolean = typeof v === "boolean" || v === "true" || v === "false";
                const boolVal = String(v) === "true";
                return (
                  <div
                    key={k}
                    className="flex items-center justify-between rounded-lg border border-white/5 bg-white/[0.02] px-4 py-2.5"
                  >
                    <span className="text-sm text-white/50">{k.replace(/_/g, " ")}</span>
                    {isBoolean ? (
                      <span className={`text-sm font-semibold ${boolVal ? "text-green-400" : "text-red-400"}`}>
                        {boolVal ? "✅ Yes" : "❌ No"}
                      </span>
                    ) : (
                      <span className="text-sm font-mono text-white/70">{String(v)}</span>
                    )}
                  </div>
                );
              })}
          </div>
        </div>
      )}
    </div>
  );
}
