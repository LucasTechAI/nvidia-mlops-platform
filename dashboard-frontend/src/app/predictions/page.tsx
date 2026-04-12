"use client";

import { useCallback, useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  ComposedChart,
  Bar,
  BarChart,
  Legend,
  ReferenceLine,
} from "recharts";
import { Download, TrendingUp, TrendingDown, DollarSign, Target, BarChart3, Rocket } from "lucide-react";
import StatCard from "@/components/stat-card";
import LoadingSpinner from "@/components/loading-spinner";
import { api } from "@/lib/api";

interface PredictionPoint {
  date: string;
  predicted_price: number;
  lower_bound?: number;
  upper_bound?: number;
}

interface HistoricalPoint {
  date: string;
  close: number;
  [key: string]: unknown;
}

export default function PredictionsPage() {
  const [horizon, setHorizon] = useState(30);
  const [contextDays, setContextDays] = useState(90);
  const [showConfidence, setShowConfidence] = useState(true);
  const [loading, setLoading] = useState(false);
  const [historicalData, setHistoricalData] = useState<HistoricalPoint[]>([]);
  const [predictions, setPredictions] = useState<PredictionPoint[]>([]);
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const [liveData, setLiveData] = useState<{ date: string; close: number }[]>([]);
  const [dbLastDate, setDbLastDate] = useState<string | null>(null);

  // Load live data (newer than DB) from Yahoo Finance
  useEffect(() => {
    async function loadLive() {
      try {
        const res = await api.data.live();
        setLiveData(res.data || []);
        setDbLastDate(res.db_last_date || null);
        // Update current price to the most recent live close if available
        if (res.data && res.data.length > 0) {
          setCurrentPrice(res.data[res.data.length - 1].close);
        }
      } catch (err) {
        console.error("Failed to load live data:", err);
      }
    }
    loadLive();
  }, []);


  // Load historical data
  useEffect(() => {
    async function loadData() {
      try {
        const res = await api.data.recent(contextDays);
        const data = (res as { data?: HistoricalPoint[] }).data || [];
        setHistoricalData(data);
        if (data.length > 0 && liveData.length === 0) {
          setCurrentPrice(data[data.length - 1].close);
        }
      } catch (err) {
        console.error("Failed to load historical data:", err);
      }
    }
    loadData();
  }, [contextDays]);

  // Generate predictions
  const generateForecast = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.predict.forecast({
        horizon: horizon,
        with_uncertainty: showConfidence,
        n_samples: 100,
        confidence_level: 0.95,
      });
      const forecastRes = res as {
        predictions?: PredictionPoint[];
        last_known_price?: number;
      };
      setPredictions(forecastRes.predictions || []);
      if (forecastRes.last_known_price && liveData.length === 0) {
        setCurrentPrice(forecastRes.last_known_price);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Forecast failed");
    } finally {
      setLoading(false);
    }
  }, [horizon, showConfidence]);

  // Auto-generate forecast on page load
  useEffect(() => {
    generateForecast();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Build chart data combining historical + predictions + live
  // Normalize date to YYYY-MM-DD for consistent matching
  const normalizeDate = (d: string) => d.slice(0, 10);

  // Create a map of live data by date for quick lookup
  const liveByDate = new Map(liveData.map((d) => [normalizeDate(d.date), d.close]));
  const predByDate = new Map(predictions.map((p) => [normalizeDate(p.date), p]));
  const histByDate = new Set(historicalData.map((h) => normalizeDate(h.date)));

  const chartData = [
    ...historicalData.map((d, i) => ({
      date: normalizeDate(d.date),
      historical: d.close,
      // Bridge: last historical point also gets "predicted" so the line connects
      ...(i === historicalData.length - 1 && predictions.length > 0
        ? { predicted: d.close }
        : {}),
    })),
    ...predictions.map((p) => {
      const dateKey = normalizeDate(p.date);
      const liveClose = liveByDate.get(dateKey);
      return {
        date: dateKey,
        predicted: p.predicted_price,
        lower: p.lower_bound,
        upper: p.upper_bound,
        confidenceBand: p.lower_bound != null && p.upper_bound != null
          ? [p.lower_bound, p.upper_bound]
          : undefined,
        ...(liveClose != null ? { real: liveClose } : {}),
      };
    }),
    // Also include live days that go beyond prediction horizon
    ...liveData
      .filter((d) => {
        const dk = normalizeDate(d.date);
        return !predByDate.has(dk) && !histByDate.has(dk);
      })
      .map((d) => ({
        date: normalizeDate(d.date),
        real: d.close,
      })),
  ];

  // Daily changes from predictions
  const dailyChanges = predictions.map((p, i) => {
    const prev = i === 0 ? currentPrice : predictions[i - 1].predicted_price;
    const change = p.predicted_price - prev;
    const d = new Date(p.date);
    const label = d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    return {
      date: label,
      change: Number(change.toFixed(2)),
      fill: change >= 0 ? "#76B900" : "#ef4444",
    };
  });

  // Summary stats
  const lastPrediction = predictions[predictions.length - 1];
  const forecastChange = lastPrediction
    ? ((lastPrediction.predicted_price - currentPrice) / currentPrice) * 100
    : 0;
  const forecastLow = predictions.length
    ? Math.min(...predictions.map((p) => p.lower_bound ?? p.predicted_price))
    : 0;
  const forecastHigh = predictions.length
    ? Math.max(...predictions.map((p) => p.upper_bound ?? p.predicted_price))
    : 0;

  // CSV download
  const downloadCSV = () => {
    if (!predictions.length) return;
    const header = "Date,Predicted Price,Lower Bound,Upper Bound\n";
    const rows = predictions
      .map(
        (p) =>
          `${p.date},${p.predicted_price.toFixed(2)},${(p.lower_bound ?? 0).toFixed(2)},${(p.upper_bound ?? 0).toFixed(2)}`
      )
      .join("\n");
    const blob = new Blob([header + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `nvidia_forecast_${horizon}d.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="flex items-center gap-2 text-2xl font-semibold"><BarChart3 className="h-6 w-6 text-nvidia" /> Stock Predictions</h2>
        <p className="mt-1 text-sm text-white/50">
          LSTM-based forecasts with Monte Carlo Dropout uncertainty estimation
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-end gap-4 rounded-xl border border-surface-border bg-surface-card p-5">
        <div>
          <label className="mb-1 block text-xs font-medium text-white/50">
            Forecast Horizon
          </label>
          <div className="flex gap-2">
            {[7, 30, 60, 90].map((d) => (
              <button
                key={d}
                onClick={() => setHorizon(d)}
                className={`rounded-lg px-4 py-2 text-sm font-medium transition-all ${
                  horizon === d
                    ? "bg-nvidia text-black"
                    : "bg-surface-hover text-white/60 hover:text-white"
                }`}
              >
                {d}d
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="mb-1 block text-xs font-medium text-white/50">
            Historical Context (days)
          </label>
          <input
            type="range"
            min={30}
            max={365}
            value={contextDays}
            onChange={(e) => setContextDays(Number(e.target.value))}
            className="w-48 accent-nvidia"
          />
          <span className="ml-2 text-sm text-white/60">{contextDays}</span>
        </div>

        <label className="flex items-center gap-2 text-sm text-white/60">
          <input
            type="checkbox"
            checked={showConfidence}
            onChange={(e) => setShowConfidence(e.target.checked)}
            className="accent-nvidia"
          />
          Confidence Intervals
        </label>

        <button
          onClick={generateForecast}
          disabled={loading}
          className="ml-auto rounded-lg bg-nvidia px-6 py-2.5 text-sm font-semibold text-black transition-all hover:bg-nvidia-dark disabled:opacity-50"
        >
          {loading ? "Generating..." : <><Rocket className="inline h-4 w-4" /> Generate Forecast</>}
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Stat Cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Current Price"
          value={`$${currentPrice.toFixed(2)}`}
          icon={<DollarSign className="h-5 w-5 text-nvidia" />}
        />
        <StatCard
          label="Predicted Price"
          value={
            lastPrediction ? `$${lastPrediction.predicted_price.toFixed(2)}` : "—"
          }
          delta={
            lastPrediction ? `${forecastChange >= 0 ? "+" : ""}${forecastChange.toFixed(1)}%` : undefined
          }
          deltaType={forecastChange >= 0 ? "positive" : "negative"}
          icon={
            forecastChange >= 0 ? (
              <TrendingUp className="h-5 w-5 text-green-400" />
            ) : (
              <TrendingDown className="h-5 w-5 text-red-400" />
            )
          }
          accentColor={forecastChange >= 0 ? "#4ade80" : "#ef4444"}
        />
        <StatCard
          label="Forecast Low"
          value={forecastLow ? `$${forecastLow.toFixed(2)}` : "—"}
          icon={<TrendingDown className="h-5 w-5 text-sky-400" />}
          accentColor="#38bdf8"
        />
        <StatCard
          label="Forecast High"
          value={forecastHigh ? `$${forecastHigh.toFixed(2)}` : "—"}
          icon={<Target className="h-5 w-5 text-amber-400" />}
          accentColor="#fbbf24"
        />
      </div>

      {/* Forecast Chart */}
      <div className="rounded-xl border border-surface-border bg-surface-card p-6">
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h3 className="text-lg font-semibold">Price Forecast</h3>
            {liveData.length > 0 && (
              <span className="rounded-full bg-amber-500/15 px-2.5 py-0.5 text-[10px] font-semibold text-amber-400">
                {liveData.length} live {liveData.length === 1 ? "day" : "days"} (after {dbLastDate})
              </span>
            )}
          </div>
          {predictions.length > 0 && (
            <button
              onClick={downloadCSV}
              className="flex items-center gap-2 rounded-lg bg-surface-hover px-3 py-1.5 text-xs text-white/60 hover:text-white"
            >
              <Download className="h-3.5 w-3.5" /> Export CSV
            </button>
          )}
        </div>

        {loading ? (
          <LoadingSpinner text="Generating forecast..." />
        ) : chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={420}>
            <ComposedChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis
                dataKey="date"
                tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 11 }}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 11 }}
                tickLine={false}
                domain={[0, "auto"]}
                tickFormatter={(v) => `$${v}`}
              />
              <Tooltip
                contentStyle={{
                  background: "#1a1c24",
                  border: "1px solid rgba(118,185,0,0.3)",
                  borderRadius: 8,
                  color: "#fff",
                }}
                formatter={(value: unknown, name: string) => {
                  if (name === "confidenceBand" && Array.isArray(value)) {
                    return [`$${Number(value[0]).toFixed(2)} – $${Number(value[1]).toFixed(2)}`, "95% Confidence"];
                  }
                  const num = Number(value);
                  const label =
                    name === "historical" ? "Historical"
                    : name === "predicted" ? "Predicted"
                    : name === "real" ? "Real (Live)"
                    : name;
                  return [`$${num.toFixed(2)}`, label];
                }}
              />
              <Legend
                verticalAlign="top"
                align="right"
                iconType="line"
                wrapperStyle={{ paddingBottom: 12, fontSize: 12, color: "rgba(255,255,255,0.6)" }}
                payload={[
                  { value: "Historical", type: "line", color: "#4ECDC4" },
                  { value: "Predicted", type: "line", color: "#76B900" },
                  ...(liveData.length > 0 ? [{ value: "Real (Live)", type: "line" as const, color: "#FBBF24" }] : []),
                  ...(showConfidence ? [{ value: "95% Confidence", type: "rect" as const, color: "rgba(118,185,0,0.35)" }] : []),
                ]}
              />

              {/* Confidence band as a single area range */}
              {showConfidence && (
                <Area
                  type="monotone"
                  dataKey="confidenceBand"
                  stroke="rgba(118,185,0,0.4)"
                  strokeWidth={1}
                  fill="rgba(118,185,0,0.15)"
                  name="confidenceBand"
                  legendType="none"
                />
              )}

              {/* Historical line */}
              <Line
                type="monotone"
                dataKey="historical"
                stroke="#4ECDC4"
                strokeWidth={2}
                dot={false}
                name="Historical"
              />

              {/* Prediction line */}
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="#76B900"
                strokeWidth={2}
                strokeDasharray="6 3"
                dot={false}
                name="Predicted"
              />

              {/* Real (Live) line — actual market data more recent than DB */}
              {liveData.length > 0 && (
                <Line
                  type="monotone"
                  dataKey="real"
                  stroke="#FBBF24"
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: "#FBBF24", stroke: "#FBBF24" }}
                  name="Real (Live)"
                  connectNulls
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex flex-col items-center justify-center py-16 text-white/30">
            <TrendingUp className="mb-3 h-12 w-12" />
            <p>Click &quot;Generate Forecast&quot; to see predictions</p>
          </div>
        )}
      </div>

      {/* Predictions Table + Daily Changes */}
      {predictions.length > 0 && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Table */}
          <div className="rounded-xl border border-surface-border bg-surface-card p-6">
            <h3 className="mb-4 text-lg font-semibold">Predictions Table</h3>
            <div className="max-h-80 overflow-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-surface-card">
                  <tr className="border-b border-surface-border text-left text-xs text-white/40">
                    <th className="pb-2">Date</th>
                    <th className="pb-2">Price</th>
                    <th className="pb-2">Low</th>
                    <th className="pb-2">High</th>
                  </tr>
                </thead>
                <tbody>
                  {predictions.map((p, i) => (
                    <tr
                      key={i}
                      className="border-b border-surface-border/50 hover:bg-surface-hover"
                    >
                      <td className="py-2 text-white/70">{p.date}</td>
                      <td className="py-2 font-medium">
                        ${p.predicted_price.toFixed(2)}
                      </td>
                      <td className="py-2 text-white/50">
                        ${(p.lower_bound ?? 0).toFixed(2)}
                      </td>
                      <td className="py-2 text-white/50">
                        ${(p.upper_bound ?? 0).toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Daily Changes Chart */}
          <div className="rounded-xl border border-surface-border bg-surface-card p-6">
            <h3 className="mb-4 text-lg font-semibold">Daily Changes</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={dailyChanges}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(255,255,255,0.05)"
                />
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
                  formatter={(value: number) => [`$${value.toFixed(2)}`, "Change"]}
                />
                <ReferenceLine y={0} stroke="rgba(255,255,255,0.2)" />
                <Bar dataKey="change" radius={[4, 4, 0, 0]}>
                  {dailyChanges.map((entry, i) => (
                    <rect key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

    </div>
  );
}
