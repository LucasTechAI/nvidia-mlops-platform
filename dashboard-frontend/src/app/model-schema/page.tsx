"use client";

import { useEffect, useState } from "react";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { Cpu, Layers, Hash, Box } from "lucide-react";
import StatCard from "@/components/stat-card";
import LoadingSpinner from "@/components/loading-spinner";
import { api } from "@/lib/api";

const COLORS = ["#76B900", "#4ECDC4", "#45B7D1", "#FF6B35"];

export default function ModelSchemaPage() {
  const [modelInfo, setModelInfo] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.model
      .info()
      .then((data) => setModelInfo(data))
      .catch((err) => setError(err instanceof Error ? err.message : "Failed"))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner text="Loading model architecture..." />;
  if (error || !modelInfo)
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <span className="text-6xl">🤖</span>
        <h3 className="mt-4 text-xl font-semibold text-amber-400">No Model Found</h3>
        <p className="mt-2 text-sm text-white/50">
          {error || "Please train the model first to view architecture details."}
        </p>
      </div>
    );

  const config = modelInfo.model_config as Record<string, unknown>;
  const params = modelInfo.parameters as {
    layers: Record<string, { shape: number[]; count: number; dtype: string }>;
    total: number;
    trainable: number;
  };
  const trainingInfo = modelInfo.training_info as Record<string, unknown>;

  const inputSize = Number(config.input_size ?? 1);
  const hiddenSize = Number(config.hidden_size ?? 128);
  const numLayers = Number(config.num_layers ?? 2);
  const outputSize = Number(config.output_size ?? 1);
  const dropout = Number(config.dropout ?? 0.2);
  const bidirectional = Boolean(config.bidirectional);
  const seqLength = Number(config.sequence_length ?? 60);

  // Parameter distribution
  const layerGroups: Record<string, number> = {
    "LSTM Weights (ih)": 0,
    "LSTM Weights (hh)": 0,
    "LSTM Biases": 0,
    "Dense Layer": 0,
  };
  if (params?.layers) {
    for (const [name, info] of Object.entries(params.layers)) {
      if (name.includes("weight_ih")) layerGroups["LSTM Weights (ih)"] += info.count;
      else if (name.includes("weight_hh")) layerGroups["LSTM Weights (hh)"] += info.count;
      else if (name.includes("bias") && name.toLowerCase().includes("lstm"))
        layerGroups["LSTM Biases"] += info.count;
      else layerGroups["Dense Layer"] += info.count;
    }
  }

  const pieData = Object.entries(layerGroups)
    .filter(([, v]) => v > 0)
    .map(([name, value]) => ({ name, value }));

  const sizeMB = params ? (params.total * 4) / (1024 * 1024) : 0;

  // LSTM layer params count
  const lstmParams = Object.entries(params?.layers ?? {})
    .filter(([k]) => k.toLowerCase().includes("lstm"))
    .reduce((sum, [, v]) => sum + v.count, 0);
  const fcParams = Object.entries(params?.layers ?? {})
    .filter(([k]) => k.includes("fc") || k.includes("linear"))
    .reduce((sum, [, v]) => sum + v.count, 0);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-semibold">🧠 Model Architecture</h2>
        <p className="mt-1 text-sm text-white/50">
          Explore the LSTM neural network architecture, layer configuration, and parameter distribution.
        </p>
      </div>

      {/* Model Overview */}
      <div>
        <p className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-white/30">
          📋 Model Overview
        </p>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="gradient-card rounded-xl border border-nvidia/20 p-5">
            <h4 className="text-sm font-semibold text-nvidia">🔷 Architecture Type</h4>
            <p className="mt-1 font-semibold text-white">LSTM (Long Short-Term Memory)</p>
            <p className="mt-1 text-sm text-white/50">
              A recurrent neural network capable of learning long-term dependencies in sequential data.
            </p>
          </div>
          <div className="gradient-card rounded-xl border border-blue-400/20 p-5">
            <h4 className="text-sm font-semibold text-blue-400">⚡ Key Characteristics</h4>
            <div className="mt-2 space-y-2">
              {[
                "Sequential Processing — Processes data one step at a time",
                "Memory Cells — Maintains information over long sequences",
                "Gating Mechanism — Input, Forget, Output gates",
                `Bidirectional — ${bidirectional ? "Enabled" : "Not enabled"}`,
              ].map((item, i) => (
                <div key={i} className="flex items-center gap-2 text-sm text-white/70">
                  <span className="text-nvidia">▸</span>
                  <span>{item}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Architecture Configuration */}
      <div>
        <p className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-white/30">
          ⚙️ Architecture Configuration
        </p>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatCard label="Input Size" value={inputSize} subtitle="Number of input features" icon={<Box className="h-5 w-5 text-nvidia" />} />
          <StatCard label="Hidden Size" value={hiddenSize} subtitle="LSTM hidden state dim" icon={<Layers className="h-5 w-5 text-sky-400" />} accentColor="#38bdf8" />
          <StatCard label="Num Layers" value={numLayers} subtitle="Stacked LSTM layers" icon={<Hash className="h-5 w-5 text-amber-400" />} accentColor="#fbbf24" />
          <StatCard label="Output Size" value={outputSize} subtitle="Prediction dimension" icon={<Cpu className="h-5 w-5 text-purple-400" />} accentColor="#a78bfa" />
        </div>
        <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="rounded-lg border border-surface-border bg-surface-card p-4">
            <p className="text-xs text-white/40">Dropout</p>
            <p className="text-xl font-bold">{(dropout * 100).toFixed(0)}%</p>
          </div>
          <div className="rounded-lg border border-surface-border bg-surface-card p-4">
            <p className="text-xs text-white/40">Bidirectional</p>
            <p className="text-xl font-bold">{bidirectional ? "Yes ✅" : "No ❌"}</p>
          </div>
          <div className="rounded-lg border border-surface-border bg-surface-card p-4">
            <p className="text-xs text-white/40">Sequence Length</p>
            <p className="text-xl font-bold">{seqLength}</p>
          </div>
          <div className="rounded-lg border border-surface-border bg-surface-card p-4">
            <p className="text-xs text-white/40">Directions</p>
            <p className="text-xl font-bold">{bidirectional ? 2 : 1}</p>
          </div>
        </div>
      </div>

      {/* Model Tree */}
      <div>
        <p className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-white/30">
          🌳 Model Tree Structure
        </p>
        <div className="gradient-card rounded-xl border border-nvidia/30 p-6 font-mono text-sm">
          <div className="font-semibold text-nvidia">📦 NvidiaLSTM</div>
          <div className="ml-5 mt-2 border-l-2 border-nvidia/30 pl-5 space-y-3">
            {/* Input */}
            <div className="flex items-center gap-2">
              <span className="text-[#4ECDC4]">📥</span>
              <span className="font-semibold text-[#4ECDC4]">Input Layer</span>
              <span className="rounded bg-[#4ECDC4]/20 px-2 py-0.5 text-xs text-[#4ECDC4]">
                shape: (batch, {seqLength}, {inputSize})
              </span>
            </div>

            {/* LSTM */}
            <div>
              <div className="flex items-center gap-2">
                <span className="text-nvidia">🧠</span>
                <span className="font-semibold text-nvidia">LSTM</span>
                <span className="rounded bg-nvidia/20 px-2 py-0.5 text-xs text-nvidia">
                  {lstmParams.toLocaleString()} params
                </span>
              </div>
              <div className="ml-7 mt-1 border-l-2 border-nvidia/20 pl-4 text-xs text-white/60 space-y-0.5">
                <div>├─ <span className="text-nvidia-light">input_size:</span> {inputSize}</div>
                <div>├─ <span className="text-nvidia-light">hidden_size:</span> {hiddenSize}</div>
                <div>├─ <span className="text-nvidia-light">num_layers:</span> {numLayers}</div>
                <div>├─ <span className="text-nvidia-light">bidirectional:</span> {String(bidirectional)}</div>
                <div>├─ <span className="text-nvidia-light">batch_first:</span> True</div>
                <div>└─ <span className="text-nvidia-light">dropout:</span> {dropout}</div>
              </div>
              <div className="ml-7 mt-2 space-y-0.5">
                {Array.from({ length: numLayers }, (_, i) => (
                  <div key={i} className="flex items-center gap-1.5 text-xs text-white/50">
                    <span>{i === numLayers - 1 ? "└─" : "├─"}</span>
                    <span className={i % 2 === 0 ? "text-nvidia" : "text-nvidia-light"}>
                      Layer {i}
                    </span>
                    <span className="text-white/30">
                      ({i === 0 ? inputSize : hiddenSize} → {hiddenSize})
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Dropout */}
            <div className="flex items-center gap-2">
              <span className="text-[#45B7D1]">💧</span>
              <span className="font-semibold text-[#45B7D1]">Dropout</span>
              <span className="rounded bg-[#45B7D1]/20 px-2 py-0.5 text-xs text-[#45B7D1]">
                p={dropout}
              </span>
            </div>

            {/* FC */}
            <div>
              <div className="flex items-center gap-2">
                <span className="text-[#FF6B35]">🔗</span>
                <span className="font-semibold text-[#FF6B35]">Linear (FC)</span>
                <span className="rounded bg-[#FF6B35]/20 px-2 py-0.5 text-xs text-[#FF6B35]">
                  {fcParams.toLocaleString()} params
                </span>
              </div>
              <div className="ml-7 mt-1 border-l-2 border-[#FF6B35]/20 pl-4 text-xs text-white/60 space-y-0.5">
                <div>├─ <span className="text-[#FF8C5A]">in_features:</span> {hiddenSize}</div>
                <div>└─ <span className="text-[#FF8C5A]">out_features:</span> {outputSize}</div>
              </div>
            </div>

            {/* Output */}
            <div className="flex items-center gap-2">
              <span className="text-[#96CEB4]">📤</span>
              <span className="font-semibold text-[#96CEB4]">Output</span>
              <span className="rounded bg-[#96CEB4]/20 px-2 py-0.5 text-xs text-[#96CEB4]">
                shape: (batch, {outputSize})
              </span>
            </div>
          </div>

          {/* Summary */}
          <div className="mt-4 flex items-center justify-between border-t border-white/10 pt-3">
            <span className="text-xs text-white/40">
              Total Parameters:{" "}
              <span className="font-semibold text-nvidia">{params?.total.toLocaleString()}</span>
            </span>
            <span className="rounded-md border border-nvidia/30 bg-nvidia/10 px-3 py-1 text-xs text-nvidia">
              PyTorch LSTM
            </span>
          </div>
        </div>
      </div>

      {/* Parameter Analysis */}
      <div>
        <p className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-white/30">
          📊 Parameter Analysis
        </p>
        <div className="grid grid-cols-3 gap-4">
          <div className="rounded-lg border border-surface-border bg-surface-card p-4">
            <p className="text-xs text-white/40">Total Parameters</p>
            <p className="text-2xl font-bold">{params?.total.toLocaleString()}</p>
          </div>
          <div className="rounded-lg border border-surface-border bg-surface-card p-4">
            <p className="text-xs text-white/40">Trainable Parameters</p>
            <p className="text-2xl font-bold">{params?.trainable.toLocaleString()}</p>
          </div>
          <div className="rounded-lg border border-surface-border bg-surface-card p-4">
            <p className="text-xs text-white/40">Model Size</p>
            <p className="text-2xl font-bold">{sizeMB.toFixed(2)} MB</p>
          </div>
        </div>

        {/* Pie chart */}
        {pieData.length > 0 && (
          <div className="mt-4 rounded-xl border border-surface-border bg-surface-card p-6">
            <h4 className="mb-2 text-sm font-semibold text-white/80">
              Parameter Distribution by Layer Type
            </h4>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={4}
                  dataKey="value"
                  label={({ name, percent }) =>
                    `${name} ${(percent * 100).toFixed(0)}%`
                  }
                >
                  {pieData.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: "#1a1c24",
                    border: "1px solid rgba(118,185,0,0.3)",
                    borderRadius: 8,
                  }}
                  formatter={(value: number) => [value.toLocaleString(), "Parameters"]}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Layer Detail Table */}
      {params?.layers && (
        <div className="rounded-xl border border-surface-border bg-surface-card p-6">
          <h3 className="mb-4 text-lg font-semibold">📋 Detailed Layer Information</h3>
          <div className="overflow-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-surface-border text-left text-xs text-white/40">
                  <th className="pb-2">Layer</th>
                  <th className="pb-2">Shape</th>
                  <th className="pb-2">Parameters</th>
                  <th className="pb-2">Type</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(params.layers).map(([name, info]) => (
                  <tr key={name} className="border-b border-surface-border/50 hover:bg-surface-hover">
                    <td className="py-2 font-mono text-xs text-nvidia">{name}</td>
                    <td className="py-2 text-white/60">{JSON.stringify(info.shape)}</td>
                    <td className="py-2 font-medium">{info.count.toLocaleString()}</td>
                    <td className="py-2 text-white/50">{info.dtype}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Data Flow */}
      <div className="rounded-xl border border-surface-border bg-surface-card p-6">
        <h3 className="mb-4 text-lg font-semibold">🔄 Data Flow</h3>
        <div className="space-y-4">
          {[
            { step: "1. Input Preparation", desc: `Historical prices are normalized and shaped into sequences of ${seqLength} time steps.`, shape: `(batch, ${seqLength}, ${inputSize})` },
            { step: "2. LSTM Processing", desc: `${numLayers} stacked LSTM layers process the sequence, learning temporal patterns.`, shape: `(batch, ${seqLength}, ${hiddenSize})` },
            { step: "3. Final Hidden State", desc: "The last hidden state from the final LSTM layer is extracted.", shape: `(batch, ${hiddenSize})` },
            { step: "4. Dropout", desc: `Dropout with rate ${dropout} is applied for regularization.`, shape: `(batch, ${hiddenSize})` },
            { step: "5. Dense Layer", desc: "Fully connected layer maps hidden state to prediction.", shape: `(batch, ${outputSize})` },
            { step: "6. Inverse Transform", desc: "Prediction is converted back to original price scale.", shape: "Predicted Price ($)" },
          ].map((s) => (
            <div key={s.step} className="flex items-start justify-between gap-4 rounded-lg bg-surface-hover p-4">
              <div>
                <p className="font-semibold text-white">{s.step}</p>
                <p className="mt-0.5 text-sm text-white/50">{s.desc}</p>
              </div>
              <code className="flex-shrink-0 rounded bg-surface-card px-3 py-1 text-xs text-nvidia">
                {s.shape}
              </code>
            </div>
          ))}
        </div>
      </div>

      {/* Training Configuration */}
      {Object.keys(trainingInfo).length > 0 && (
        <div className="rounded-xl border border-surface-border bg-surface-card p-6">
          <h3 className="mb-4 text-lg font-semibold">🎯 Training Configuration</h3>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            <div>
              <h4 className="mb-2 text-sm font-semibold text-white/60">Optimizer &amp; Learning</h4>
              <div className="space-y-2">
                {[
                  ["Optimizer", trainingInfo.optimizer ?? "Adam"],
                  ["Learning Rate", trainingInfo.learning_rate ?? 0.001],
                  ["Weight Decay", trainingInfo.weight_decay ?? 1e-5],
                  ["Batch Size", trainingInfo.batch_size ?? 32],
                ].map(([k, v]) => (
                  <div key={String(k)} className="flex justify-between text-sm">
                    <span className="text-white/50">{String(k)}</span>
                    <code className="text-nvidia">{typeof v === "number" && v < 0.01 ? Number(v).toExponential(2) : String(v)}</code>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h4 className="mb-2 text-sm font-semibold text-white/60">Regularization &amp; Stopping</h4>
              <div className="space-y-2">
                {[
                  ["Dropout Rate", dropout],
                  ["Early Stopping Patience", trainingInfo.early_stopping_patience ?? 10],
                  ["Gradient Clipping", trainingInfo.gradient_clip_value ?? 1.0],
                  ["LR Scheduler", trainingInfo.use_scheduler ?? true],
                ].map(([k, v]) => (
                  <div key={String(k)} className="flex justify-between text-sm">
                    <span className="text-white/50">{String(k)}</span>
                    <code className="text-nvidia">
                      {typeof v === "boolean" ? (v ? "Yes" : "No") : String(v)}
                    </code>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
