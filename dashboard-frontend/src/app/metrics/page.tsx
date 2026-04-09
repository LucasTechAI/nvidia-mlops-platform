"use client";

import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Legend,
} from "recharts";
import { Award, Zap, Target, BarChart3, TrendingDown } from "lucide-react";
import StatCard from "@/components/stat-card";
import LoadingSpinner from "@/components/loading-spinner";
import { api } from "@/lib/api";

export default function MetricsPage() {
  const [modelInfo, setModelInfo] = useState<Record<string, unknown> | null>(null);
  const [history, setHistory] = useState<Record<string, number[]> | null>(null);
  const [hpo, setHpo] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const [infoRes, histRes] = await Promise.allSettled([
          api.model.info(),
          api.model.trainingHistory(),
        ]);

        if (infoRes.status === "fulfilled") setModelInfo(infoRes.value);
        if (histRes.status === "fulfilled") setHistory(histRes.value as Record<string, number[]>);

        // HPO is optional
        try {
          const hpoRes = await api.model.hpoResults();
          setHpo(hpoRes);
        } catch {
          /* HPO data not available */
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load metrics");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) return <LoadingSpinner text="Loading model metrics..." />;
  if (error)
    return (
      <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-red-400">
        {error}
      </div>
    );

  const config = (modelInfo?.model_config ?? {}) as Record<string, unknown>;
  const testMetrics = (modelInfo?.test_metrics ?? {}) as Record<string, number>;
  const trainingInfo = (modelInfo?.training_info ?? {}) as Record<string, unknown>;

  // Build training curves data
  const curveData = history
    ? (history.train_loss || []).map((_: number, i: number) => ({
        epoch: i + 1,
        train_loss: history.train_loss?.[i],
        val_loss: history.val_loss?.[i],
        train_rmse: history.train_rmse?.[i],
        val_rmse: history.val_rmse?.[i],
        train_mae: history.train_mae?.[i],
        val_mae: history.val_mae?.[i],
        train_r2: history.train_r2?.[i],
        val_r2: history.val_r2?.[i],
      }))
    : [];

  // HPO radar data
  const hpoParams = (hpo as Record<string, unknown>)?.best_params as Record<string, number> | undefined;
  const radarData = hpoParams
    ? Object.entries(hpoParams).map(([key, val]) => ({
        param: key.replace(/_/g, " "),
        value: typeof val === "number" ? val : 0,
        fullMark: 1,
      }))
    : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-semibold">📈 Model Metrics</h2>
        <p className="mt-1 text-sm text-white/50">
          Training performance, test metrics, and hyperparameter optimization results
        </p>
      </div>

      {/* Training Overview */}
      <div>
        <p className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-white/30">
          Training Overview
        </p>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            label="Best Epoch"
            value={String(modelInfo?.best_epoch ?? modelInfo?.epoch ?? "—")}
            icon={<Award className="h-5 w-5 text-nvidia" />}
          />
          <StatCard
            label="Best Val Loss"
            value={
              modelInfo?.best_loss != null
                ? (modelInfo.best_loss as number).toFixed(6)
                : (modelInfo?.loss as number)?.toFixed(6) ?? "—"
            }
            icon={<TrendingDown className="h-5 w-5 text-sky-400" />}
            accentColor="#38bdf8"
          />
          <StatCard
            label="Total Epochs"
            value={String(modelInfo?.epoch ?? "—")}
            icon={<Zap className="h-5 w-5 text-amber-400" />}
            accentColor="#fbbf24"
          />
          <StatCard
            label="Early Stopping"
            value={
              trainingInfo.early_stopping_patience
                ? `Patience ${trainingInfo.early_stopping_patience}`
                : "Enabled"
            }
            icon={<Target className="h-5 w-5 text-purple-400" />}
            accentColor="#a78bfa"
          />
        </div>
      </div>

      {/* Test Metrics */}
      {Object.keys(testMetrics).length > 0 && (
        <div>
          <p className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-white/30">
            Test Performance
          </p>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
            {Object.entries(testMetrics).map(([key, val]) => (
              <div
                key={key}
                className="rounded-lg border border-surface-border bg-surface-card p-4"
              >
                <p className="text-xs font-medium uppercase text-white/40">
                  {key.replace(/_/g, " ")}
                </p>
                <p className="mt-1 text-xl font-bold">
                  {typeof val === "number" ? val.toFixed(4) : String(val)}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Training Curves */}
      {curveData.length > 0 && (
        <div>
          <p className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-white/30">
            Training Curves
          </p>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* Loss */}
            <div className="rounded-xl border border-surface-border bg-surface-card p-6">
              <h4 className="mb-4 text-sm font-semibold text-white/80">
                Loss Curve
              </h4>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={curveData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{
                      background: "#1a1c24",
                      border: "1px solid rgba(118,185,0,0.3)",
                      borderRadius: 8,
                    }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="train_loss" stroke="#76B900" strokeWidth={2} dot={false} name="Train" />
                  <Line type="monotone" dataKey="val_loss" stroke="#4ECDC4" strokeWidth={2} dot={false} name="Validation" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* RMSE */}
            <div className="rounded-xl border border-surface-border bg-surface-card p-6">
              <h4 className="mb-4 text-sm font-semibold text-white/80">
                RMSE Curve
              </h4>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={curveData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{
                      background: "#1a1c24",
                      border: "1px solid rgba(118,185,0,0.3)",
                      borderRadius: 8,
                    }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="train_rmse" stroke="#76B900" strokeWidth={2} dot={false} name="Train" />
                  <Line type="monotone" dataKey="val_rmse" stroke="#4ECDC4" strokeWidth={2} dot={false} name="Validation" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* MAE */}
            <div className="rounded-xl border border-surface-border bg-surface-card p-6">
              <h4 className="mb-4 text-sm font-semibold text-white/80">
                MAE Curve
              </h4>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={curveData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip
                    contentStyle={{
                      background: "#1a1c24",
                      border: "1px solid rgba(118,185,0,0.3)",
                      borderRadius: 8,
                    }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="train_mae" stroke="#76B900" strokeWidth={2} dot={false} name="Train" />
                  <Line type="monotone" dataKey="val_mae" stroke="#4ECDC4" strokeWidth={2} dot={false} name="Validation" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* R² */}
            <div className="rounded-xl border border-surface-border bg-surface-card p-6">
              <h4 className="mb-4 text-sm font-semibold text-white/80">
                R² Score Curve
              </h4>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={curveData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} domain={[0, 1]} />
                  <Tooltip
                    contentStyle={{
                      background: "#1a1c24",
                      border: "1px solid rgba(118,185,0,0.3)",
                      borderRadius: 8,
                    }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="train_r2" stroke="#76B900" strokeWidth={2} dot={false} name="Train" />
                  <Line type="monotone" dataKey="val_r2" stroke="#4ECDC4" strokeWidth={2} dot={false} name="Validation" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* HPO Radar Chart */}
      {radarData.length > 0 && (
        <div className="rounded-xl border border-surface-border bg-surface-card p-6">
          <h3 className="mb-4 text-lg font-semibold">
            🎯 Hyperparameter Optimization
          </h3>
          <ResponsiveContainer width="100%" height={350}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="rgba(255,255,255,0.1)" />
              <PolarAngleAxis dataKey="param" tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 11 }} />
              <PolarRadiusAxis tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 10 }} />
              <Radar name="Best Params" dataKey="value" stroke="#76B900" fill="#76B900" fillOpacity={0.3} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Metrics Interpretation */}
      <div className="rounded-xl border border-surface-border bg-surface-card p-6">
        <h3 className="mb-4 text-lg font-semibold">📖 Metrics Interpretation</h3>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {[
            { name: "RMSE", desc: "Root Mean Squared Error — penalizes large errors more heavily. Lower is better." },
            { name: "MAE", desc: "Mean Absolute Error — average magnitude of errors. More robust to outliers than RMSE." },
            { name: "MAPE", desc: "Mean Absolute Percentage Error — scale-independent metric. Values < 10% indicate excellent predictions." },
            { name: "R²", desc: "Coefficient of Determination — proportion of variance explained by the model. 1.0 is perfect." },
            { name: "Directional Accuracy", desc: "Percentage of correctly predicted price movement direction (up/down)." },
            { name: "Sharpe Ratio", desc: "Risk-adjusted return metric. Higher values indicate better risk-adjusted performance." },
          ].map((m) => (
            <div key={m.name} className="rounded-lg bg-surface-hover p-4">
              <p className="text-sm font-semibold text-nvidia">{m.name}</p>
              <p className="mt-1 text-xs text-white/50">{m.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
