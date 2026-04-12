"use client";

import { useState, useEffect, useCallback } from "react";
import {
  RefreshCw,
  Shield,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Database,
  GitBranch,
  Layers,
  Zap,
  Clock,
  BarChart3,
  Settings,
  FileText,
  Wrench,
  Brain,
  Bot,
  Rocket,
  Plus,
  ArrowUp,
  SkipForward,
  DollarSign,
  Cpu,
  MessageSquare,
  Server,
} from "lucide-react";
import {
  BarChart,
  Bar,
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { api } from "@/lib/api";

/* ── Helpers ────────────────────────────────────────────────── */
function StatCard({
  title,
  value,
  subtitle,
  icon: Icon,
  color = "text-nvidia",
  trend,
}: {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ElementType;
  color?: string;
  trend?: "up" | "down" | "neutral";
}) {
  return (
    <div className="rounded-xl border border-surface-border bg-surface-card p-4">
      <div className="flex items-center justify-between">
        <p className="text-xs text-white/50">{title}</p>
        <Icon className={`h-4 w-4 ${color}`} />
      </div>
      <p className={`mt-1 text-2xl font-bold ${color}`}>{value}</p>
      {subtitle && (
        <p className="mt-0.5 text-xs text-white/40">
          {trend === "up" && "▲ "}
          {trend === "down" && "▼ "}
          {subtitle}
        </p>
      )}
    </div>
  );
}

const TABS = [
  { id: "sla", label: "SLA & Uptime", icon: Shield },
  { id: "business", label: "Business Metrics", icon: TrendingUp },
  { id: "registry", label: "Model Registry", icon: Layers },
  { id: "features", label: "Feature Store", icon: Database },
  { id: "canary", label: "Canary Deploy", icon: Zap },
  { id: "cost", label: "Cost Analysis", icon: DollarSign },
] as const;

type TabId = (typeof TABS)[number]["id"];

const COLORS = ["#76b900", "#0ea5e9", "#f59e0b", "#ef4444", "#a855f7", "#14b8a6"];

export default function MLOpsPage() {
  const [activeTab, setActiveTab] = useState<TabId>("sla");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  /* ── SLA state ── */
  const [slaReport, setSlaReport] = useState<Record<string, unknown> | null>(null);
  const [uptimeHistory, setUptimeHistory] = useState<Record<string, unknown>[]>([]);

  /* ── Business state ── */
  const [bizSnapshot, setBizSnapshot] = useState<Record<string, unknown> | null>(null);
  const [pnlHistory, setPnlHistory] = useState<Record<string, unknown>[]>([]);

  /* ── Registry state ── */
  const [registryModels, setRegistryModels] = useState<Record<string, unknown>[]>([]);
  const [modelVersions, setModelVersions] = useState<{ version: number; stage: string; created_at: string; metrics: Record<string, number> }[]>([]);
  const [transitionHistory, setTransitionHistory] = useState<Record<string, unknown>[]>([]);

  /* ── Feature Store state ── */
  const [featureSets, setFeatureSets] = useState<{ name: string; latest_version: number; total_versions: number; last_updated: string }[]>([]);

  /* ── Canary state ── */
  const [deployments, setDeployments] = useState<Record<string, unknown>[]>([]);
  const [rollbacks, setRollbacks] = useState<Record<string, unknown>[]>([]);

  /* ── Cost state ── */
  const [costData, setCostData] = useState<Record<string, unknown> | null>(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      if (activeTab === "sla") {
        const [rep, hist] = await Promise.all([api.sla.report(1440), api.sla.uptimeHistory(7)]);
        setSlaReport(rep);
        setUptimeHistory(hist.history);
      } else if (activeTab === "business") {
        const [snap, pnl] = await Promise.all([api.businessMetrics.snapshot(), api.businessMetrics.pnlHistory(60)]);
        setBizSnapshot(snap);
        setPnlHistory(pnl.history);
      } else if (activeTab === "registry") {
        const models = await api.modelRegistry.list();
        setRegistryModels(models.models);
        if (models.models.length > 0) {
          const name = (models.models[0] as Record<string, unknown>).name as string;
          const [vers, hist] = await Promise.all([api.modelRegistry.versions(name), api.modelRegistry.history(name)]);
          setModelVersions(vers.versions);
          setTransitionHistory(hist.history);
        }
      } else if (activeTab === "features") {
        const fs = await api.featureStore.list();
        setFeatureSets(fs.feature_sets);
      } else if (activeTab === "canary") {
        const [deps, rb] = await Promise.all([api.canary.deployments(), api.canary.rollbackHistory()]);
        setDeployments(deps.deployments);
        setRollbacks(rb.rollbacks);
      } else if (activeTab === "cost") {
        const data = await api.costAnalysis.get(30);
        setCostData(data);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load data");
    } finally {
      setLoading(false);
    }
  }, [activeTab]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-bold text-white"><Settings className="h-6 w-6 text-nvidia" /> MLOps & Advanced Monitoring</h1>
          <p className="text-sm text-white/50">SLA, business metrics, registry, feature store & canary deploy</p>
        </div>
        <button
          onClick={loadData}
          disabled={loading}
          className="flex items-center gap-2 rounded-lg border border-nvidia/30 bg-nvidia/10 px-4 py-2 text-sm text-nvidia transition hover:bg-nvidia/20 disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 rounded-xl border border-surface-border bg-surface-card p-1">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex flex-1 items-center justify-center gap-2 rounded-lg px-3 py-2.5 text-xs font-medium transition ${
                activeTab === tab.id
                  ? "bg-nvidia/20 text-nvidia"
                  : "text-white/50 hover:bg-surface-hover hover:text-white"
              }`}
            >
              <Icon className="h-3.5 w-3.5" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-400">{error}</div>
      )}

      {/* ══════════════ SLA TAB ══════════════ */}
      {activeTab === "sla" && slaReport && (
        <div className="space-y-6">
          {/* SLA Status Banner */}
          <div
            className={`flex items-center gap-3 rounded-xl border p-4 ${
              (slaReport as Record<string, unknown>).overall_sla_met
                ? "border-green-500/30 bg-green-500/10"
                : "border-red-500/30 bg-red-500/10"
            }`}
          >
            {(slaReport as Record<string, unknown>).overall_sla_met ? (
              <CheckCircle className="h-6 w-6 text-green-400" />
            ) : (
              <XCircle className="h-6 w-6 text-red-400" />
            )}
            <div>
              <p className="font-semibold text-white">
                {(slaReport as Record<string, unknown>).overall_sla_met ? "All SLAs met" : "SLA violated"}
              </p>
              {((slaReport as Record<string, unknown>).violations as string[])?.length > 0 && (
                <p className="text-sm text-red-300">
                  {((slaReport as Record<string, unknown>).violations as string[]).join(" | ")}
                </p>
              )}
            </div>
          </div>

          {/* SLA Cards */}
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatCard title="Uptime" value={`${slaReport.uptime_pct}%`} subtitle={`Target: ${(slaReport.sla_targets as Record<string, number>)?.uptime_pct ?? 99.5}%`} icon={Shield} color={Number(slaReport.uptime_pct) >= 99.5 ? "text-green-400" : "text-red-400"} />
            <StatCard title="p95 Latency" value={`${slaReport.p95_latency_ms}ms`} subtitle={`Target: <${(slaReport.sla_targets as Record<string, number>)?.latency_p95_ms ?? 500}ms`} icon={Clock} color={Number(slaReport.p95_latency_ms) <= 500 ? "text-green-400" : "text-yellow-400"} />
            <StatCard title="Error Rate" value={`${slaReport.error_rate_pct}%`} subtitle={`Target: <${(slaReport.sla_targets as Record<string, number>)?.error_rate_pct ?? 1}%`} icon={AlertTriangle} color={Number(slaReport.error_rate_pct) <= 1 ? "text-green-400" : "text-red-400"} />
            <StatCard title="Requests (24h)" value={String(slaReport.total_requests)} subtitle={`${slaReport.error_requests} errors`} icon={BarChart3} />
          </div>

          {/* Uptime History Chart */}
          <div className="rounded-xl border border-surface-border bg-surface-card p-5">
            <h3 className="mb-4 text-sm font-semibold text-white">Uptime & Latency (7 days)</h3>
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={uptimeHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="date" tick={{ fill: "#888", fontSize: 11 }} tickFormatter={(v) => v.slice(5)} />
                <YAxis yAxisId="left" domain={[95, 100]} tick={{ fill: "#888", fontSize: 11 }} unit="%" />
                <YAxis yAxisId="right" orientation="right" tick={{ fill: "#888", fontSize: 11 }} unit="ms" />
                <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid #333", borderRadius: 8 }} />
                <Legend />
                <Area yAxisId="left" type="monotone" dataKey="uptime_pct" name="Uptime %" stroke="#76b900" fill="#76b900" fillOpacity={0.15} />
                <Line yAxisId="right" type="monotone" dataKey="avg_latency_ms" name="Avg Latency ms" stroke="#0ea5e9" dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Latency Breakdown */}
          <div className="grid grid-cols-3 gap-4">
            <div className="rounded-xl border border-surface-border bg-surface-card p-4 text-center">
              <p className="text-xs text-white/50">p50 Latency</p>
              <p className="mt-1 text-xl font-bold text-white">{String(slaReport.p50_latency_ms)} ms</p>
            </div>
            <div className="rounded-xl border border-surface-border bg-surface-card p-4 text-center">
              <p className="text-xs text-white/50">p95 Latency</p>
              <p className="mt-1 text-xl font-bold text-yellow-400">{String(slaReport.p95_latency_ms)} ms</p>
            </div>
            <div className="rounded-xl border border-surface-border bg-surface-card p-4 text-center">
              <p className="text-xs text-white/50">p99 Latency</p>
              <p className="mt-1 text-xl font-bold text-orange-400">{String(slaReport.p99_latency_ms)} ms</p>
            </div>
          </div>
        </div>
      )}

      {/* ══════════════ BUSINESS METRICS TAB ══════════════ */}
      {activeTab === "business" && bizSnapshot && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatCard title="Cumulative P&L" value={`$${Number(bizSnapshot.cumulative_pnl).toFixed(2)}`} icon={TrendingUp} color={Number(bizSnapshot.cumulative_pnl) >= 0 ? "text-green-400" : "text-red-400"} trend={Number(bizSnapshot.cumulative_pnl) >= 0 ? "up" : "down"} />
            <StatCard title="ROI" value={`${Number(bizSnapshot.roi_pct).toFixed(1)}%`} icon={BarChart3} color={Number(bizSnapshot.roi_pct) >= 0 ? "text-green-400" : "text-red-400"} />
            <StatCard title="Sharpe Ratio" value={Number(bizSnapshot.sharpe_ratio).toFixed(2)} subtitle={Number(bizSnapshot.sharpe_ratio) >= 2 ? "Excellent" : Number(bizSnapshot.sharpe_ratio) >= 1 ? "Good" : "Low"} icon={Shield} color={Number(bizSnapshot.sharpe_ratio) >= 1 ? "text-green-400" : "text-yellow-400"} />
            <StatCard title="Win Rate" value={`${Number(bizSnapshot.win_rate).toFixed(1)}%`} subtitle={`${bizSnapshot.winning_predictions}/${bizSnapshot.total_predictions} predictions`} icon={CheckCircle} color={Number(bizSnapshot.win_rate) >= 50 ? "text-green-400" : "text-red-400"} />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <StatCard title="Max Drawdown" value={`${Number(bizSnapshot.max_drawdown).toFixed(2)}%`} icon={AlertTriangle} color="text-red-400" />
            <StatCard title="Avg Error" value={`${Number(bizSnapshot.avg_error_pct).toFixed(2)}%`} icon={XCircle} color="text-yellow-400" />
          </div>

          {/* P&L Chart */}
          <div className="rounded-xl border border-surface-border bg-surface-card p-5">
            <h3 className="mb-4 text-sm font-semibold text-white">P&L Evolution (60 days)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={pnlHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="date" tick={{ fill: "#888", fontSize: 11 }} tickFormatter={(v) => v.slice(5)} />
                <YAxis tick={{ fill: "#888", fontSize: 11 }} />
                <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid #333", borderRadius: 8 }} />
                <Legend />
                <Area type="monotone" dataKey="cumulative_pnl" name="P&L ($)" stroke="#76b900" fill="#76b900" fillOpacity={0.2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ══════════════ MODEL REGISTRY TAB ══════════════ */}
      {activeTab === "registry" && (
        <div className="space-y-6">
          {/* Models List */}
          {registryModels.length === 0 ? (
            <div className="rounded-xl border border-surface-border bg-surface-card p-8 text-center text-white/40">
              No models registered
            </div>
          ) : (
            <>
              <div className="rounded-xl border border-surface-border bg-surface-card p-5">
                <h3 className="mb-4 text-sm font-semibold text-white">Model Versions</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead>
                      <tr className="border-b border-surface-border text-xs text-white/40">
                        <th className="pb-2 pr-4">Version</th>
                        <th className="pb-2 pr-4">Stage</th>
                        <th className="pb-2 pr-4">RMSE</th>
                        <th className="pb-2 pr-4">MAE</th>
                        <th className="pb-2 pr-4">R²</th>
                        <th className="pb-2 pr-4">Dir. Accuracy</th>
                        <th className="pb-2">Created</th>
                      </tr>
                    </thead>
                    <tbody>
                      {modelVersions.map((v) => (
                        <tr key={v.version} className="border-b border-surface-border/50">
                          <td className="py-2 pr-4 font-mono font-bold text-white">v{v.version}</td>
                          <td className="py-2 pr-4">
                            <span
                              className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                                v.stage === "Production"
                                  ? "bg-green-500/20 text-green-400"
                                  : v.stage === "Staging"
                                  ? "bg-yellow-500/20 text-yellow-400"
                                  : v.stage === "Archived"
                                  ? "bg-white/10 text-white/40"
                                  : "bg-blue-500/20 text-blue-400"
                              }`}
                            >
                              {v.stage}
                            </span>
                          </td>
                          <td className="py-2 pr-4 text-white/70">{v.metrics?.rmse?.toFixed(3) ?? "—"}</td>
                          <td className="py-2 pr-4 text-white/70">{v.metrics?.mae?.toFixed(3) ?? "—"}</td>
                          <td className="py-2 pr-4 text-white/70">{v.metrics?.r2?.toFixed(3) ?? "—"}</td>
                          <td className="py-2 pr-4 text-white/70">{v.metrics?.directional_accuracy ? `${(v.metrics.directional_accuracy * 100).toFixed(0)}%` : "—"}</td>
                          <td className="py-2 text-white/40">{v.created_at?.slice(0, 10) ?? "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* RMSE Comparison */}
              {modelVersions.length > 0 && (
                <div className="rounded-xl border border-surface-border bg-surface-card p-5">
                  <h3 className="mb-4 text-sm font-semibold text-white">Metrics Comparison by Version</h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={modelVersions.map((v) => ({ name: `v${v.version} (${v.stage})`, rmse: v.metrics?.rmse, mae: v.metrics?.mae, r2: v.metrics?.r2 }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                      <XAxis dataKey="name" tick={{ fill: "#888", fontSize: 11 }} />
                      <YAxis tick={{ fill: "#888", fontSize: 11 }} />
                      <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid #333", borderRadius: 8 }} />
                      <Legend />
                      <Bar dataKey="rmse" name="RMSE" fill="#ef4444" />
                      <Bar dataKey="mae" name="MAE" fill="#f59e0b" />
                      <Bar dataKey="r2" name="R²" fill="#76b900" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Transition History */}
              {transitionHistory.length > 0 && (
                <div className="rounded-xl border border-surface-border bg-surface-card p-5">
                  <h3 className="mb-4 text-sm font-semibold text-white">Transition History</h3>
                  <div className="space-y-2">
                    {transitionHistory.slice(0, 10).map((t, i) => (
                      <div key={i} className="flex items-center gap-3 rounded-lg border border-surface-border/50 bg-surface-hover p-3 text-sm">
                        <GitBranch className="h-4 w-4 text-nvidia" />
                        <span className="font-mono text-white/70">v{String(t.version)}</span>
                        <span className="text-white/30">→</span>
                        <span className={`rounded-full px-2 py-0.5 text-xs ${
                          t.to_stage === "Production" ? "bg-green-500/20 text-green-400" :
                          t.to_stage === "Staging" ? "bg-yellow-500/20 text-yellow-400" :
                          t.to_stage === "Archived" ? "bg-white/10 text-white/40" :
                          "bg-blue-500/20 text-blue-400"
                        }`}>{String(t.to_stage)}</span>
                        <span className="flex-1 text-white/40">{String(t.reason)}</span>
                        <span className="text-xs text-white/30">{String(t.timestamp).slice(0, 16)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* ══════════════ FEATURE STORE TAB ══════════════ */}
      {activeTab === "features" && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            {featureSets.map((fs) => (
              <div key={fs.name} className="rounded-xl border border-surface-border bg-surface-card p-5">
                <div className="flex items-start justify-between">
                  <div>
                    <h4 className="font-semibold text-white">{fs.name}</h4>
                    <p className="text-xs text-white/40">
                      v{fs.latest_version} · {fs.total_versions} version(s)
                    </p>
                  </div>
                  <Database className="h-5 w-5 text-nvidia" />
                </div>
                <div className="mt-3 flex items-center gap-4 text-xs text-white/50">
                  <span>Updated: {fs.last_updated?.slice(0, 10)}</span>
                </div>
              </div>
            ))}
          </div>

          {featureSets.length === 0 && (
            <div className="rounded-xl border border-surface-border bg-surface-card p-8 text-center text-white/40">
              No feature sets registered
            </div>
          )}

          {/* Feature lineage diagram placeholder */}
          <div className="rounded-xl border border-surface-border bg-surface-card p-5">
            <h3 className="mb-3 text-sm font-semibold text-white">Feature Pipeline</h3>
            <div className="flex items-center justify-center gap-3 py-6">
              <div className="flex items-center gap-1.5 rounded-lg bg-blue-500/20 px-4 py-2 text-sm text-blue-400"><FileText className="h-3.5 w-3.5" /> CSV Raw Data</div>
              <span className="text-white/30">→</span>
              <div className="flex items-center gap-1.5 rounded-lg bg-yellow-500/20 px-4 py-2 text-sm text-yellow-400"><Wrench className="h-3.5 w-3.5" /> Transform</div>
              <span className="text-white/30">→</span>
              <div className="flex items-center gap-1.5 rounded-lg bg-green-500/20 px-4 py-2 text-sm text-green-400"><BarChart3 className="h-3.5 w-3.5" /> Technical Indicators</div>
              <span className="text-white/30">→</span>
              <div className="flex items-center gap-1.5 rounded-lg bg-purple-500/20 px-4 py-2 text-sm text-purple-400"><Brain className="h-3.5 w-3.5" /> Lag Features</div>
              <span className="text-white/30">→</span>
              <div className="flex items-center gap-1.5 rounded-lg bg-nvidia/20 px-4 py-2 text-sm text-nvidia"><Bot className="h-3.5 w-3.5" /> LSTM Model</div>
            </div>
          </div>
        </div>
      )}

      {/* ══════════════ CANARY DEPLOY TAB ══════════════ */}
      {activeTab === "canary" && (
        <div className="space-y-6">
          {/* Deployments */}
          <div className="rounded-xl border border-surface-border bg-surface-card p-5">
            <h3 className="mb-4 text-sm font-semibold text-white">Canary Deployments</h3>
            {deployments.length === 0 ? (
              <p className="text-center text-white/40">No deployments recorded</p>
            ) : (
              <div className="space-y-3">
                {deployments.map((d, i) => (
                  <div key={i} className="rounded-lg border border-surface-border/50 bg-surface-hover p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <Zap className={`h-5 w-5 ${
                          d.state === "promoted" ? "text-green-400" :
                          d.state === "rolled_back" ? "text-red-400" :
                          d.state === "canary" ? "text-yellow-400" :
                          "text-white/40"
                        }`} />
                        <div>
                          <p className="text-sm font-medium text-white">{String(d.model_name)}</p>
                          <p className="text-xs text-white/40">
                            v{String(d.canary_version)} (canary) vs v{String(d.baseline_version)} (baseline)
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className={`rounded-full px-3 py-1 text-xs font-medium ${
                          d.state === "promoted" ? "bg-green-500/20 text-green-400" :
                          d.state === "rolled_back" ? "bg-red-500/20 text-red-400" :
                          d.state === "canary" ? "bg-yellow-500/20 text-yellow-400" :
                          "bg-white/10 text-white/40"
                        }`}>
                          {String(d.state).replace("_", " ").toUpperCase()}
                        </span>
                        <span className="text-xs text-white/30">{String(d.started_at).slice(0, 16)}</span>
                      </div>
                    </div>
                    {/* Weight bar */}
                    <div className="mt-3">
                      <div className="flex items-center justify-between text-xs text-white/40">
                        <span>Canary Weight</span>
                        <span>{Number(d.canary_weight).toFixed(0)}%</span>
                      </div>
                      <div className="mt-1 h-2 overflow-hidden rounded-full bg-surface-border">
                        <div
                          className={`h-full rounded-full transition-all ${
                            d.state === "promoted" ? "bg-green-500" :
                            d.state === "rolled_back" ? "bg-red-500" : "bg-nvidia"
                          }`}
                          style={{ width: `${Number(d.canary_weight)}%` }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Rollback Log */}
          {rollbacks.length > 0 && (
            <div className="rounded-xl border border-red-500/20 bg-surface-card p-5">
              <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-red-400">
                <AlertTriangle className="h-4 w-4" /> Rollback History
              </h3>
              <div className="space-y-2">
                {rollbacks.map((r, i) => (
                  <div key={i} className="flex items-center gap-3 rounded-lg bg-red-500/5 p-3 text-sm">
                    <XCircle className="h-4 w-4 text-red-400" />
                    <span className="text-white/70">v{String(r.canary_version)} → v{String(r.rolled_back_to)}</span>
                    <span className="flex-1 text-white/40">{String(r.reason)}</span>
                    <span className="text-xs text-white/30">{String(r.timestamp).slice(0, 16)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Canary Deploy Flow */}
          <div className="rounded-xl border border-surface-border bg-surface-card p-5">
            <h3 className="mb-3 text-sm font-semibold text-white">Canary Deploy Flow</h3>
            <div className="flex items-center justify-center gap-2 py-4 text-xs">
              <div className="flex items-center gap-1.5 rounded-lg bg-blue-500/20 px-3 py-2 text-blue-400"><Plus className="h-3.5 w-3.5" /> New Version</div>
              <span className="text-white/20">→</span>
              <div className="flex items-center gap-1.5 rounded-lg bg-yellow-500/20 px-3 py-2 text-yellow-400"><Zap className="h-3.5 w-3.5" /> Canary 5%</div>
              <span className="text-white/20">→</span>
              <div className="flex items-center gap-1.5 rounded-lg bg-orange-500/20 px-3 py-2 text-orange-400"><ArrowUp className="h-3.5 w-3.5" /> Ramp Up</div>
              <span className="text-white/20">→</span>
              <div className="flex items-center gap-1.5 rounded-lg bg-nvidia/20 px-3 py-2 text-nvidia"><CheckCircle className="h-3.5 w-3.5" /> Health Check</div>
              <span className="text-white/20">→</span>
              <div className="flex items-center gap-1.5 rounded-lg bg-green-500/20 px-3 py-2 text-green-400"><Rocket className="h-3.5 w-3.5" /> Promote 100%</div>
            </div>
            <div className="mt-2 flex items-center justify-center gap-2 text-xs text-red-400">
              <span className="text-white/20">⤷</span>
              <div className="flex items-center gap-1.5 rounded-lg bg-red-500/20 px-3 py-2"><AlertTriangle className="h-3.5 w-3.5" /> Error &gt; 2% → Auto Rollback</div>
            </div>
          </div>
        </div>
      )}

      {/* ══════════════ COST ANALYSIS TAB ══════════════ */}
      {activeTab === "cost" && costData && (() => {
        const data = costData as Record<string, unknown>;
        const infraBreakdown = (data.infra_breakdown ?? []) as { name: string; quantity: number; unit: string; unit_cost: number; total: number }[];
        const llmBreakdown = (data.llm_breakdown ?? []) as { name: string; tokens: number; cost: number }[];
        const dailyHistory = (data.daily_history ?? []) as { date: string; infra: number; llm: number; total: number }[];
        const modelComparison = (data.model_comparison ?? []) as { model: string; model_id: string; input_cost: number; output_cost: number; total_cost: number; is_current: boolean }[];

        const fmtTokens = (n: number) => {
          if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
          if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
          return String(n);
        };

        return (
          <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
              <StatCard title="Total Estimated Cost" value={`$${Number(data.grand_total).toFixed(2)}`} subtitle={`${data.period_days}-day period`} icon={DollarSign} color="text-nvidia" />
              <StatCard title="Infrastructure" value={`$${Number(data.infra_total).toFixed(2)}`} subtitle={`${Number(data.infra_pct).toFixed(0)}% of total`} icon={Server} color="text-blue-400" />
              <StatCard title="LLM Costs" value={`$${Number(data.llm_total).toFixed(4)}`} subtitle={`${Number(data.llm_pct).toFixed(1)}% of total`} icon={MessageSquare} color="text-purple-400" />
              <StatCard title="Total Tokens" value={fmtTokens(Number(data.total_input_tokens) + Number(data.total_output_tokens))} subtitle={`${fmtTokens(Number(data.total_input_tokens))} in / ${fmtTokens(Number(data.total_output_tokens))} out`} icon={Cpu} color="text-amber-400" />
              <StatCard title="Training Runs" value={String(data.training_runs)} subtitle="GPU sessions" icon={Zap} color="text-green-400" />
            </div>

            {/* Current Model Badge */}
            <div className="flex items-center gap-3 rounded-xl border border-nvidia/20 bg-nvidia/5 p-4">
              <MessageSquare className="h-5 w-5 text-nvidia" />
              <div>
                <p className="text-sm font-semibold text-white">Active LLM: <span className="text-nvidia">{String(data.current_model)}</span></p>
                <p className="text-xs text-white/40">Provider: {String(data.provider)} · Model ID: <span className="font-mono text-white/50">{String(data.current_model_id)}</span></p>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              {/* Infra Breakdown */}
              <div className="rounded-xl border border-surface-border bg-surface-card p-5">
                <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-white">
                  <Server className="h-4 w-4 text-blue-400" /> Infrastructure Breakdown
                </h3>
                <div className="space-y-2">
                  {infraBreakdown.map((item) => (
                    <div key={item.name} className="flex items-center justify-between rounded-lg border border-white/5 bg-white/[0.02] px-4 py-2.5">
                      <div>
                        <p className="text-sm text-white/70">{item.name}</p>
                        <p className="text-xs text-white/30">{item.quantity} {item.unit} × ${item.unit_cost}/{item.unit.replace(/s$/, "")}</p>
                      </div>
                      <span className="font-mono text-sm font-semibold text-blue-400">${item.total.toFixed(2)}</span>
                    </div>
                  ))}
                  <div className="flex items-center justify-between border-t border-white/10 pt-2 mt-2">
                    <span className="text-sm font-semibold text-white">Total Infrastructure</span>
                    <span className="font-mono text-sm font-bold text-blue-400">${Number(data.infra_total).toFixed(2)}</span>
                  </div>
                </div>
              </div>

              {/* LLM Breakdown */}
              <div className="rounded-xl border border-surface-border bg-surface-card p-5">
                <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-white">
                  <MessageSquare className="h-4 w-4 text-purple-400" /> LLM Token Breakdown
                </h3>
                <div className="space-y-2">
                  {llmBreakdown.map((item) => (
                    <div key={item.name} className="flex items-center justify-between rounded-lg border border-white/5 bg-white/[0.02] px-4 py-2.5">
                      <div>
                        <p className="text-sm text-white/70">{item.name}</p>
                        <p className="text-xs text-white/30">{fmtTokens(item.tokens)} tokens</p>
                      </div>
                      <span className="font-mono text-sm font-semibold text-purple-400">${item.cost.toFixed(4)}</span>
                    </div>
                  ))}
                  <div className="flex items-center justify-between border-t border-white/10 pt-2 mt-2">
                    <span className="text-sm font-semibold text-white">Total LLM</span>
                    <span className="font-mono text-sm font-bold text-purple-400">${Number(data.llm_total).toFixed(4)}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Daily Cost Chart */}
            <div className="rounded-xl border border-surface-border bg-surface-card p-5">
              <h3 className="mb-4 text-sm font-semibold text-white">Daily Cost Trend ({String(data.period_days)} days)</h3>
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={dailyHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis dataKey="date" tick={{ fill: "#888", fontSize: 11 }} tickFormatter={(v) => v.slice(5)} />
                  <YAxis tick={{ fill: "#888", fontSize: 11 }} unit="$" />
                  <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid #333", borderRadius: 8 }} formatter={(value: number) => [`$${value.toFixed(2)}`, undefined]} />
                  <Legend />
                  <Area type="monotone" dataKey="infra" name="Infrastructure" stroke="#0ea5e9" fill="#0ea5e9" fillOpacity={0.15} stackId="1" />
                  <Area type="monotone" dataKey="llm" name="LLM" stroke="#a855f7" fill="#a855f7" fillOpacity={0.15} stackId="1" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Model Cost Comparison */}
            <div className="rounded-xl border border-surface-border bg-surface-card p-5">
              <h3 className="mb-4 text-sm font-semibold text-white">💡 LLM Cost Comparison (same workload)</h3>
              <p className="mb-3 text-xs text-white/40">What would it cost with a different model for the same {fmtTokens(Number(data.total_input_tokens) + Number(data.total_output_tokens))} tokens?</p>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead>
                    <tr className="border-b border-surface-border text-xs text-white/40">
                      <th className="pb-2 pr-4">Model</th>
                      <th className="pb-2 pr-4 text-right">Input Cost</th>
                      <th className="pb-2 pr-4 text-right">Output Cost</th>
                      <th className="pb-2 text-right">Total</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelComparison.map((m) => (
                      <tr key={m.model_id} className={`border-b border-surface-border/50 ${m.is_current ? "bg-nvidia/5" : ""}`}>
                        <td className="py-2.5 pr-4">
                          <span className="text-white/80">{m.model}</span>
                          {m.is_current && <span className="ml-2 rounded-full bg-nvidia/20 px-2 py-0.5 text-[10px] font-semibold text-nvidia">CURRENT</span>}
                        </td>
                        <td className="py-2.5 pr-4 text-right font-mono text-white/50">${m.input_cost.toFixed(4)}</td>
                        <td className="py-2.5 pr-4 text-right font-mono text-white/50">${m.output_cost.toFixed(4)}</td>
                        <td className={`py-2.5 text-right font-mono font-semibold ${
                          m.is_current ? "text-nvidia" : m.total_cost <= Number(data.llm_total) ? "text-green-400" : "text-red-400"
                        }`}>${m.total_cost.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Cost Split Pie */}
            <div className="rounded-xl border border-surface-border bg-surface-card p-5">
              <h3 className="mb-4 text-sm font-semibold text-white">Cost Distribution</h3>
              <div className="flex items-center justify-center">
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={[
                        { name: "Infrastructure", value: Number(data.infra_total) },
                        { name: "LLM", value: Number(data.llm_total) },
                      ]}
                      cx="50%" cy="50%"
                      outerRadius={90}
                      innerRadius={50}
                      paddingAngle={5}
                      dataKey="value"
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    >
                      <Cell fill="#0ea5e9" />
                      <Cell fill="#a855f7" />
                    </Pie>
                    <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid #333", borderRadius: 8 }} formatter={(v: number) => `$${v.toFixed(4)}`} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        );
      })()}

      {/* Loading placeholder */}
      {loading && !slaReport && !bizSnapshot && registryModels.length === 0 && featureSets.length === 0 && deployments.length === 0 && !costData && (
        <div className="flex items-center justify-center py-20">
          <RefreshCw className="h-8 w-8 animate-spin text-nvidia" />
        </div>
      )}
    </div>
  );
}
