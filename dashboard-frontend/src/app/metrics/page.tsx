"use client";

import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Legend,
  ComposedChart,
  Bar,
  Area,
  Cell,
} from "recharts";
import { Award, Zap, Target, BarChart3, TrendingDown, GitCompareArrows, Info, LineChart as LineChartIcon } from "lucide-react";
import StatCard from "@/components/stat-card";
import LoadingSpinner from "@/components/loading-spinner";
import { api } from "@/lib/api";

function MetricCard({ label, value, tooltip }: { label: string; value: string; tooltip?: string }) {
  const [showTip, setShowTip] = useState(false);
  return (
    <div className="rounded-lg border border-surface-border bg-surface-card p-4">
      <div className="flex items-center gap-1.5">
        <p className="text-xs font-medium uppercase text-white/40">{label}</p>
        {tooltip && (
          <div
            className="relative"
            onMouseEnter={() => setShowTip(true)}
            onMouseLeave={() => setShowTip(false)}
          >
            <Info className="h-3 w-3 cursor-help text-white/25 transition-colors hover:text-white/60" />
            {showTip && (
              <div className="absolute bottom-full left-1/2 z-50 mb-2 w-56 -translate-x-1/2 rounded-lg border border-surface-border bg-[#1a1c24] px-3 py-2 text-[11px] font-normal normal-case tracking-normal text-white/70 shadow-xl">
                {tooltip}
                <div className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-[#1a1c24]" />
              </div>
            )}
          </div>
        )}
      </div>
      <p className="mt-1 text-xl font-bold">{value}</p>
    </div>
  );
}

function ChartHeader({ title, tooltip }: { title: string; tooltip: string }) {
  const [showTip, setShowTip] = useState(false);
  return (
    <div className="mb-4 flex items-center gap-2">
      <h4 className="text-sm font-semibold text-white/80">{title}</h4>
      <div
        className="relative"
        onMouseEnter={() => setShowTip(true)}
        onMouseLeave={() => setShowTip(false)}
      >
        <Info className="h-3.5 w-3.5 cursor-help text-white/25 transition-colors hover:text-white/60" />
        {showTip && (
          <div className="absolute bottom-full left-1/2 z-50 mb-2 w-64 -translate-x-1/2 rounded-lg border border-surface-border bg-[#1a1c24] px-3 py-2 text-[11px] font-normal normal-case tracking-normal text-white/70 shadow-xl">
            {tooltip}
            <div className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-[#1a1c24]" />
          </div>
        )}
      </div>
    </div>
  );
}

interface BacktestPoint {
  date: string;
  actual: number;
  predicted: number;
}

export default function MetricsPage() {
  const [modelInfo, setModelInfo] = useState<Record<string, unknown> | null>(null);
  const [history, setHistory] = useState<Record<string, number[]> | null>(null);
  const [hpo, setHpo] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [backtestData, setBacktestData] = useState<BacktestPoint[]>([]);
  const [backtestLoading, setBacktestLoading] = useState(false);

  // Load backtest data
  useEffect(() => {
    async function loadBacktest() {
      setBacktestLoading(true);
      try {
        const res = await api.predict.backtest(504);
        setBacktestData(res.backtest || []);
      } catch (err) {
        console.error("Failed to load backtest data:", err);
      } finally {
        setBacktestLoading(false);
      }
    }
    loadBacktest();
  }, []);

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
  // Test metrics — prefer per-epoch arrays; fall back to scalar for legacy checkpoints
  const testLossArr = history ? (Array.isArray(history.test_loss) ? history.test_loss : null) : null;
  const testRmseArr = history ? (Array.isArray(history.test_rmse) ? history.test_rmse : null) : null;
  const testMaeArr = history ? (Array.isArray(history.test_mae) ? history.test_mae : null) : null;
  const testR2Arr = history ? (Array.isArray(history.test_r2) ? history.test_r2 : null) : null;
  const hasPerEpochTest = testLossArr !== null && testLossArr.length > 1;

  // Scalar fallback (shown as flat ReferenceLine when per-epoch data is unavailable)
  const testLoss = hasPerEpochTest ? NaN : (history ? Number(history.test_loss ?? NaN) : NaN);
  const testRmse = hasPerEpochTest ? NaN : (history ? Number(history.test_rmse ?? NaN) : NaN);
  const testMae = hasPerEpochTest ? NaN : (history ? Number(history.test_mae ?? NaN) : NaN);
  const testR2 = hasPerEpochTest ? NaN : (history ? Number(history.test_r2 ?? NaN) : NaN);
  const testN = history ? Number(history.test_n_samples ?? 0) : 0;

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
        ...(testLossArr ? { test_loss: testLossArr[i] } : {}),
        ...(testRmseArr ? { test_rmse: testRmseArr[i] } : {}),
        ...(testMaeArr ? { test_mae: testMaeArr[i] } : {}),
        ...(testR2Arr ? { test_r2: testR2Arr[i] } : {}),
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
        <h2 className="flex items-center gap-2 text-2xl font-semibold"><LineChartIcon className="h-6 w-6 text-nvidia" /> Model Metrics</h2>
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
            tooltip="Epoch where the model achieved the lowest validation error. Training may have continued past this point, but this was the best checkpoint."
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
            tooltip="Lowest loss value achieved on the validation set. The lower this value, the better the model generalizes to unseen data."
          />
          <StatCard
            label="Total Epochs"
            value={String(modelInfo?.epoch ?? "—")}
            icon={<Zap className="h-5 w-5 text-amber-400" />}
            accentColor="#fbbf24"
            tooltip="Total number of epochs executed during training. One epoch = one complete pass through all training data."
          />
          <StatCard
            label="Early Stopping"
            value={
              trainingInfo["Early Stopped"] === true
                ? "Triggered"
                : trainingInfo.early_stopping_patience
                  ? `Patience ${trainingInfo.early_stopping_patience}`
                  : "Enabled"
            }
            icon={<Target className="h-5 w-5 text-purple-400" />}
            accentColor="#a78bfa"
            tooltip="Technique that stops training when validation error stops improving, preventing overfitting. 'Triggered' = stopped before reaching the maximum number of epochs."
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
            {Object.entries(testMetrics).map(([key, val]) => {
              const tooltips: Record<string, string> = {
                rmse: "Root Mean Squared Error — penalizes large errors more severely. Lower is better. Measured in dollars ($).",
                mae: "Mean Absolute Error — average magnitude of errors. More robust to outliers than RMSE. Measured in dollars ($).",
                mape: "Mean Absolute Percentage Error — scale-independent metric. Values below 10% indicate excellent accuracy.",
                r2_score: "Coefficient of Determination (R²) — proportion of variance explained by the model. 1.0 = perfect, 0 = no predictive power.",
                correlation: "Pearson correlation between actual and predicted values. The closer to 1.0, the more aligned the predictions.",
                directional_accuracy: "Directional Accuracy — percentage of times the model correctly predicted the direction (up/down). Above 50% = better than random chance.",
                sharpe_ratio: "Sharpe Ratio — risk-adjusted return. Values above 1.0 indicate a good return/risk ratio in the predictions.",
                max_drawdown: "Maximum Drawdown — largest percentage drop from peak to trough. The lower, the more stable the model-based strategy.",
              };
              return (
                <MetricCard
                  key={key}
                  label={key.replace(/_/g, " ")}
                  value={typeof val === "number" ? val.toFixed(4) : String(val)}
                  tooltip={tooltips[key]}
                />
              );
            })}
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
              <ChartHeader title="Loss Curve" tooltip="Loss (MSE) curve per epoch. Shows how well the model is learning. Train and validation lines should converge — if validation rises while train drops, overfitting is occurring." />
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
                  {hasPerEpochTest && <Line type="monotone" dataKey="test_loss" stroke="#f97316" strokeWidth={2} dot={false} name="Test" />}
                  {!hasPerEpochTest && !isNaN(testLoss) && (
                    <ReferenceLine y={testLoss} stroke="#f97316" strokeDasharray="6 3" strokeWidth={1.5} label={{ value: `Test ${testLoss.toFixed(4)}`, fill: "#f97316", fontSize: 10, position: "right" }} />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* RMSE */}
            {(history?.train_rmse || history?.val_rmse) && (
            <div className="rounded-xl border border-surface-border bg-surface-card p-6">
              <ChartHeader title="RMSE Curve" tooltip="Root Mean Squared Error per epoch. Measures average deviation of predictions from actual values (in the same unit as the data). Lower means more accurate predictions." />
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
                  {history?.train_rmse && <Line type="monotone" dataKey="train_rmse" stroke="#76B900" strokeWidth={2} dot={false} name="Train" />}
                  {history?.val_rmse && <Line type="monotone" dataKey="val_rmse" stroke="#4ECDC4" strokeWidth={2} dot={false} name="Validation" />}
                  {hasPerEpochTest && <Line type="monotone" dataKey="test_rmse" stroke="#f97316" strokeWidth={2} dot={false} name="Test" />}
                  {!hasPerEpochTest && !isNaN(testRmse) && (
                    <ReferenceLine y={testRmse} stroke="#f97316" strokeDasharray="6 3" strokeWidth={1.5} label={{ value: `Test ${testRmse.toFixed(4)}`, fill: "#f97316", fontSize: 10, position: "right" }} />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
            )}

            {/* MAE */}
            {(history?.train_mae || history?.val_mae) && (
            <div className="rounded-xl border border-surface-border bg-surface-card p-6">
              <ChartHeader title="MAE Curve" tooltip="Mean Absolute Error per epoch. Indicates the average magnitude of errors regardless of direction. Less sensitive to outliers than RMSE. Lower is better." />
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
                  {history?.train_mae && <Line type="monotone" dataKey="train_mae" stroke="#76B900" strokeWidth={2} dot={false} name="Train" />}
                  {history?.val_mae && <Line type="monotone" dataKey="val_mae" stroke="#4ECDC4" strokeWidth={2} dot={false} name="Validation" />}
                  {hasPerEpochTest && <Line type="monotone" dataKey="test_mae" stroke="#f97316" strokeWidth={2} dot={false} name="Test" />}
                  {!hasPerEpochTest && !isNaN(testMae) && (
                    <ReferenceLine y={testMae} stroke="#f97316" strokeDasharray="6 3" strokeWidth={1.5} label={{ value: `Test ${testMae.toFixed(4)}`, fill: "#f97316", fontSize: 10, position: "right" }} />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
            )}

            {/* R² */}
            {(history?.train_r2 || history?.val_r2) && (
            <div className="rounded-xl border border-surface-border bg-surface-card p-6">
              <ChartHeader title="R² Score Curve" tooltip="Coefficient of determination per epoch. Ranges from 0 to 1 — values close to 1 indicate the model explains the data variance well. Ideal: both curves rising and converging." />
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
                  {history?.train_r2 && <Line type="monotone" dataKey="train_r2" stroke="#76B900" strokeWidth={2} dot={false} name="Train" />}
                  {history?.val_r2 && <Line type="monotone" dataKey="val_r2" stroke="#4ECDC4" strokeWidth={2} dot={false} name="Validation" />}
                  {hasPerEpochTest && <Line type="monotone" dataKey="test_r2" stroke="#f97316" strokeWidth={2} dot={false} name="Test" />}
                  {!hasPerEpochTest && !isNaN(testR2) && (
                    <ReferenceLine y={testR2} stroke="#f97316" strokeDasharray="6 3" strokeWidth={1.5} label={{ value: `Test ${testR2.toFixed(4)}`, fill: "#f97316", fontSize: 10, position: "right" }} />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
            )}
          </div>

          {/* Test baseline legend */}
          {hasPerEpochTest && (
            <div className="rounded-lg border border-orange-500/20 bg-orange-500/5 px-5 py-3">
              <div className="flex flex-wrap items-center gap-5">
                <div className="flex items-center gap-2">
                  <div className="h-0.5 w-6 bg-orange-500" />
                  <span className="text-xs font-semibold text-orange-400">Test Set (per-epoch)</span>
                  <span className="text-[10px] text-white/30">(last 15% of data, evaluated each epoch)</span>
                </div>
                <div className="flex flex-wrap gap-4 text-xs text-white/50">
                  <span>Final Loss: <strong className="text-orange-400">{testLossArr![testLossArr!.length - 1]?.toFixed(6)}</strong></span>
                  {testRmseArr && <span>Final RMSE: <strong className="text-orange-400">{testRmseArr[testRmseArr.length - 1]?.toFixed(6)}</strong></span>}
                  {testMaeArr && <span>Final MAE: <strong className="text-orange-400">{testMaeArr[testMaeArr.length - 1]?.toFixed(6)}</strong></span>}
                  {testR2Arr && <span>Final R²: <strong className="text-orange-400">{testR2Arr[testR2Arr.length - 1]?.toFixed(4)}</strong></span>}
                </div>
              </div>
            </div>
          )}
          {!hasPerEpochTest && !isNaN(testLoss) && (
            <div className="rounded-lg border border-orange-500/20 bg-orange-500/5 px-5 py-3">
              <div className="flex flex-wrap items-center gap-5">
                <div className="flex items-center gap-2">
                  <div className="h-px w-6 border-t-2 border-dashed border-orange-500" />
                  <span className="text-xs font-semibold text-orange-400">Test Set Baseline</span>
                  <span className="text-[10px] text-white/30">({testN} samples, last 15% of data)</span>
                </div>
                <div className="flex flex-wrap gap-4 text-xs text-white/50">
                  <span>Loss: <strong className="text-orange-400">{testLoss.toFixed(6)}</strong></span>
                  {!isNaN(testRmse) && <span>RMSE: <strong className="text-orange-400">{testRmse.toFixed(6)}</strong></span>}
                  {!isNaN(testMae) && <span>MAE: <strong className="text-orange-400">{testMae.toFixed(6)}</strong></span>}
                  {!isNaN(testR2) && <span>R²: <strong className="text-orange-400">{testR2.toFixed(4)}</strong></span>}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* HPO Radar Chart */}
      {radarData.length > 0 && (
        <div className="rounded-xl border border-surface-border bg-surface-card p-6">
          <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold">
            <Target className="h-5 w-5 text-nvidia" /> Hyperparameter Optimization
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

      {/* Actual vs Predicted (Backtest) */}
      <div className="rounded-xl border border-surface-border bg-surface-card p-6">
        <div className="mb-4 flex items-center gap-3">
          <GitCompareArrows className="h-5 w-5 text-nvidia" />
          <div>
            <h3 className="text-lg font-semibold">Actual vs Predicted (Backtest)</h3>
            <p className="text-xs text-white/40">
              Model predictions over the entire known historical period — direct comparison between actual and predicted prices
            </p>
          </div>
        </div>

        {backtestLoading ? (
          <LoadingSpinner text="Loading backtest data..." />
        ) : backtestData.length > 0 ? (
          <div>
            <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={backtestData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 10 }}
                    tickLine={false}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis
                    tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 11 }}
                    tickLine={false}
                    domain={["auto", "auto"]}
                    tickFormatter={(v) => `$${v}`}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#1a1c24",
                      border: "1px solid rgba(118,185,0,0.3)",
                      borderRadius: 8,
                      color: "#fff",
                    }}
                    formatter={(value: number, name: string) => [
                      `$${value?.toFixed(2)}`,
                      name === "actual" ? "Actual" : "Predicted",
                    ]}
                  />
                  <Legend
                    verticalAlign="top"
                    align="right"
                    iconType="line"
                    wrapperStyle={{ paddingBottom: 8, fontSize: 12, color: "rgba(255,255,255,0.6)" }}
                  />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    stroke="#4ECDC4"
                    strokeWidth={2}
                    dot={false}
                    name="Actual"
                  />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    stroke="#76B900"
                    strokeWidth={2}
                    strokeDasharray="6 3"
                    dot={false}
                    name="Predicted"
                  />
                </ComposedChart>
              </ResponsiveContainer>

            {/* Residual (Error) Chart */}
            <div className="mt-6">
              <ChartHeader title="Residual Error (Actual − Predicted)" tooltip="Difference between the actual and predicted value per day. Green bars indicate the model underestimated (actual > predicted), red bars indicate overestimation. Ideally residuals stay close to zero." />
              <ResponsiveContainer width="100%" height={250}>
                <ComposedChart data={backtestData.map((d) => ({ ...d, error: +(d.actual - d.predicted).toFixed(2) }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 10 }}
                    tickLine={false}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis
                    tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 11 }}
                    tickLine={false}
                    tickFormatter={(v) => `$${v}`}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "#1a1c24",
                      border: "1px solid rgba(118,185,0,0.3)",
                      borderRadius: 8,
                      color: "#fff",
                    }}
                    formatter={(value: number) => [`$${value?.toFixed(2)}`, "Error"]}
                  />
                  <ReferenceLine y={0} stroke="rgba(255,255,255,0.3)" strokeDasharray="4 4" />
                  <Bar
                    dataKey="error"
                    name="Error"
                    radius={[2, 2, 0, 0]}
                  >
                    {backtestData.map((d, idx) => (
                      <Cell key={idx} fill={d.actual - d.predicted >= 0 ? "#4ECDC4" : "#ef4444"} opacity={0.8} />
                    ))}
                  </Bar>
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-12 text-white/30">
            <GitCompareArrows className="mb-3 h-10 w-10" />
            <p>No backtest data available</p>
          </div>
        )}

        {/* Backtest metrics summary */}
        {backtestData.length > 0 && (() => {
          const errors = backtestData.map((d) => d.actual - d.predicted);
          const mae = errors.reduce((s, e) => s + Math.abs(e), 0) / errors.length;
          const rmse = Math.sqrt(errors.reduce((s, e) => s + e * e, 0) / errors.length);
          const mean = backtestData.reduce((s, d) => s + d.actual, 0) / backtestData.length;
          const ssTot = backtestData.reduce((s, d) => s + (d.actual - mean) ** 2, 0);
          const ssRes = errors.reduce((s, e) => s + e * e, 0);
          const r2 = 1 - ssRes / ssTot;
          const mape = errors.reduce((s, e, i) => s + Math.abs(e) / backtestData[i].actual, 0) / errors.length * 100;
          return (
            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                { label: "MAE", value: `$${mae.toFixed(2)}` },
                { label: "RMSE", value: `$${rmse.toFixed(2)}` },
                { label: "R²", value: r2.toFixed(4) },
                { label: "MAPE", value: `${mape.toFixed(2)}%` },
              ].map((m) => (
                <div key={m.label} className="rounded-lg bg-surface-hover px-4 py-3 text-center">
                  <p className="text-xs text-white/40">{m.label}</p>
                  <p className="text-sm font-semibold text-nvidia">{m.value}</p>
                </div>
              ))}
            </div>
          );
        })()}
      </div>
    </div>
  );
}
