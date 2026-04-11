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
  ComposedChart,
  Bar,
  Area,
  ReferenceLine,
  Cell,
} from "recharts";
import { Award, Zap, Target, BarChart3, TrendingDown, GitCompareArrows, Info } from "lucide-react";
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
  // Test metrics (single values — shown as flat lines across all epochs)
  const testLoss = history ? Number(history.test_loss ?? NaN) : NaN;
  const testRmse = history ? Number(history.test_rmse ?? NaN) : NaN;
  const testMae = history ? Number(history.test_mae ?? NaN) : NaN;
  const testR2 = history ? Number(history.test_r2 ?? NaN) : NaN;
  const testN = history ? Number(history.test_n_samples ?? 0) : 0;

  const curveData = history
    ? (history.train_loss || []).map((_: number, i: number) => ({
        epoch: i + 1,
        train_loss: history.train_loss?.[i],
        val_loss: history.val_loss?.[i],
        test_loss: isNaN(testLoss) ? undefined : testLoss,
        train_rmse: history.train_rmse?.[i],
        val_rmse: history.val_rmse?.[i],
        test_rmse: isNaN(testRmse) ? undefined : testRmse,
        train_mae: history.train_mae?.[i],
        val_mae: history.val_mae?.[i],
        test_mae: isNaN(testMae) ? undefined : testMae,
        train_r2: history.train_r2?.[i],
        val_r2: history.val_r2?.[i],
        test_r2: isNaN(testR2) ? undefined : testR2,
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
            tooltip="Época em que o modelo obteve o menor erro de validação. O treinamento pode ter continuado após essa época, mas este foi o melhor ponto."
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
            tooltip="Menor valor de perda (loss) alcançado no conjunto de validação. Quanto menor, melhor o modelo generaliza para dados não vistos."
          />
          <StatCard
            label="Total Epochs"
            value={String(modelInfo?.epoch ?? "—")}
            icon={<Zap className="h-5 w-5 text-amber-400" />}
            accentColor="#fbbf24"
            tooltip="Número total de épocas executadas durante o treinamento. Uma época = uma passagem completa por todos os dados de treino."
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
            tooltip="Técnica que interrompe o treinamento quando o erro de validação para de melhorar, evitando overfitting. 'Triggered' = parou antes do máximo de épocas."
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
                rmse: "Raiz do Erro Quadrático Médio — penaliza erros grandes mais severamente. Quanto menor, melhor. Está em dólares ($).",
                mae: "Erro Absoluto Médio — média da magnitude dos erros. Mais robusto a outliers que o RMSE. Está em dólares ($).",
                mape: "Erro Percentual Absoluto Médio — métrica independente de escala. Valores < 10% indicam excelente precisão.",
                r2_score: "Coeficiente de Determinação (R²) — proporção da variância explicada pelo modelo. 1.0 = perfeito, 0 = sem poder preditivo.",
                correlation: "Correlação de Pearson entre valores reais e preditos. Quanto mais próximo de 1.0, mais alinhadas estão as previsões.",
                directional_accuracy: "Acurácia Direcional — percentual de vezes que o modelo acertou a direção (subiu/desceu). Acima de 50% = melhor que o acaso.",
                sharpe_ratio: "Índice de Sharpe — retorno ajustado ao risco. Valores > 1.0 indicam boa relação retorno/risco nas previsões.",
                max_drawdown: "Drawdown Máximo — maior queda percentual do pico ao vale. Quanto menor, mais estável é a estratégia baseada no modelo.",
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
              <ChartHeader title="Loss Curve" tooltip="Curva de perda (MSE) por época. Mostra o quão bem o modelo está aprendendo. As linhas de treino e validação devem convergir — se a validação subir enquanto o treino cai, há overfitting." />
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
                  {!isNaN(testLoss) && (
                    <Line type="monotone" dataKey="test_loss" stroke="#f97316" strokeWidth={1.5} strokeDasharray="6 3" dot={false} name="Test" />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* RMSE */}
            {(history?.train_rmse || history?.val_rmse) && (
            <div className="rounded-xl border border-surface-border bg-surface-card p-6">
              <ChartHeader title="RMSE Curve" tooltip="Raiz do Erro Quadrático Médio por época. Mede o desvio médio das previsões em relação aos valores reais (mesma unidade dos dados). Quanto menor, mais preciso o modelo." />
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
                  {!isNaN(testRmse) && (
                    <Line type="monotone" dataKey="test_rmse" stroke="#f97316" strokeWidth={1.5} strokeDasharray="6 3" dot={false} name="Test" />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
            )}

            {/* MAE */}
            {(history?.train_mae || history?.val_mae) && (
            <div className="rounded-xl border border-surface-border bg-surface-card p-6">
              <ChartHeader title="MAE Curve" tooltip="Erro Absoluto Médio por época. Indica a magnitude média dos erros sem considerar a direção. Menos sensível a outliers que o RMSE. Quanto menor, melhor." />
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
                  {!isNaN(testMae) && (
                    <Line type="monotone" dataKey="test_mae" stroke="#f97316" strokeWidth={1.5} strokeDasharray="6 3" dot={false} name="Test" />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
            )}

            {/* R² */}
            {(history?.train_r2 || history?.val_r2) && (
            <div className="rounded-xl border border-surface-border bg-surface-card p-6">
              <ChartHeader title="R² Score Curve" tooltip="Coeficiente de determinação por época. Varia de 0 a 1 — valores próximos de 1 indicam que o modelo explica bem a variância dos dados. Ideal: ambas as curvas subindo e convergindo." />
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
                  {!isNaN(testR2) && (
                    <Line type="monotone" dataKey="test_r2" stroke="#f97316" strokeWidth={1.5} strokeDasharray="6 3" dot={false} name="Test" />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
            )}
          </div>

          {/* Test baseline legend */}
          {!isNaN(testLoss) && (
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

      {/* Actual vs Predicted (Backtest) */}
      <div className="rounded-xl border border-surface-border bg-surface-card p-6">
        <div className="mb-4 flex items-center gap-3">
          <GitCompareArrows className="h-5 w-5 text-nvidia" />
          <div>
            <h3 className="text-lg font-semibold">Actual vs Predicted (Backtest)</h3>
            <p className="text-xs text-white/40">
              Previsões do modelo sobre todo o período histórico conhecido — comparação direta entre preço real e previsto
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
              <ChartHeader title="Erro Residual (Actual − Predicted)" tooltip="Diferença entre o valor real e o previsto por dia. Barras verdes indicam que o modelo subestimou (real > previsto), vermelhas indicam superestimação. Idealmente os resíduos ficam próximos de zero." />
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
                    formatter={(value: number) => [`$${value?.toFixed(2)}`, "Erro"]}
                  />
                  <ReferenceLine y={0} stroke="rgba(255,255,255,0.3)" strokeDasharray="4 4" />
                  <Bar
                    dataKey="error"
                    name="Erro"
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
