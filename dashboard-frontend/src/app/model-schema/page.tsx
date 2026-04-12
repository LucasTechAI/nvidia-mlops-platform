"use client";

import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { Cpu, Layers, Hash, Box, Info, Trophy, SlidersHorizontal, Brain, Target, ClipboardList, Settings, Zap, Diamond, BarChart3, Package, RefreshCw, Award, ArrowDown, ArrowUp, Link2, Droplets } from "lucide-react";
import StatCard from "@/components/stat-card";
import LoadingSpinner from "@/components/loading-spinner";
import { api } from "@/lib/api";

const COLORS = ["#76B900", "#4ECDC4", "#45B7D1", "#FF6B35"];

function SectionHeader({ title, tooltip }: { title: string; tooltip: string }) {
  const [showTip, setShowTip] = useState(false);
  return (
    <div className="mb-4 flex items-center gap-2">
      <h3 className="text-lg font-semibold">{title}</h3>
      <div
        className="relative"
        onMouseEnter={() => setShowTip(true)}
        onMouseLeave={() => setShowTip(false)}
      >
        <Info className="h-3.5 w-3.5 cursor-help text-white/25 transition-colors hover:text-white/60" />
        {showTip && (
          <div className="absolute bottom-full left-1/2 z-50 mb-2 w-72 -translate-x-1/2 rounded-lg border border-surface-border bg-[#1a1c24] px-3 py-2 text-[11px] font-normal normal-case tracking-normal text-white/70 shadow-xl">
            {tooltip}
            <div className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-[#1a1c24]" />
          </div>
        )}
      </div>
    </div>
  );
}

export default function ModelSchemaPage() {
  const [modelInfo, setModelInfo] = useState<Record<string, unknown> | null>(null);
  const [hpo, setHpo] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const data = await api.model.info();
        setModelInfo(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed");
      } finally {
        setLoading(false);
      }
      // HPO is optional
      try {
        const hpoRes = await api.model.hpoResults();
        setHpo(hpoRes);
      } catch {
        /* HPO data not available */
      }
    };
    fetchAll();
  }, []);

  if (loading) return <LoadingSpinner text="Loading model architecture..." />;
  if (error || !modelInfo)
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <Brain className="h-16 w-16 text-nvidia/40" />
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
        <h2 className="flex items-center gap-2 text-2xl font-semibold"><Brain className="h-6 w-6 text-nvidia" /> Model Architecture</h2>
        <p className="mt-1 text-sm text-white/50">
          Explore the LSTM neural network architecture, layer configuration, and parameter distribution.
        </p>
      </div>

      {/* Model Purpose */}
      <div className="rounded-xl border border-nvidia/20 bg-gradient-to-br from-nvidia/5 via-surface-card to-surface-card p-6">
        <div className="flex items-start gap-4">
          <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-xl bg-nvidia/10">
            <Target className="h-6 w-6 text-nvidia" />
          </div>
          <div>
            <h3 className="mb-2 text-lg font-semibold">Model Purpose</h3>
            <p className="text-sm leading-relaxed text-white/60">
              This model uses an{" "}
              <span className="font-medium text-nvidia">LSTM (Long Short-Term Memory)</span> neural network to
              forecast the <span className="font-medium text-nvidia">Close price</span> of{" "}
              <span className="font-medium text-nvidia">NVIDIA (NVDA)</span> stock over the
              next 30 days. LSTMs are especially effective for time series due to their ability to
              memorize long-term dependencies — it analyzes {seqLength} days of history across 5 input features
              (Open, High, Low, Close and Volume) to identify trend patterns, seasonality and volatility,
              outputting a single Close price prediction with confidence intervals via Monte Carlo Dropout.
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              {[
                { label: "Type", value: "Recurrent LSTM" },
                { label: "Asset", value: "NVDA" },
                { label: "Target", value: "Close Price" },
                { label: "Horizon", value: "30 days" },
                { label: "Window", value: `${seqLength} days` },
                { label: "Input", value: `OHLCV (${inputSize} features)` },
              ].map((tag) => (
                <span
                  key={tag.label}
                  className="rounded-md border border-nvidia/20 bg-nvidia/5 px-2.5 py-1 text-xs text-white/60"
                >
                  <span className="font-medium text-nvidia">{tag.label}:</span> {tag.value}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Model Overview */}
      <div>
        <p className="mb-3 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-widest text-white/30">
          <ClipboardList className="h-3.5 w-3.5" /> Model Overview
        </p>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="gradient-card rounded-xl border border-nvidia/20 p-5">
            <div className="flex items-center gap-2">
              <h4 className="text-sm font-semibold text-nvidia"><Diamond className="mr-1 inline h-3.5 w-3.5" /> Architecture Type</h4>
              <div className="group relative">
                <Info className="h-3 w-3 cursor-help text-white/25 transition-colors hover:text-white/60" />
                <div className="invisible absolute bottom-full left-1/2 z-50 mb-2 w-64 -translate-x-1/2 rounded-lg border border-surface-border bg-[#1a1c24] px-3 py-2 text-[11px] font-normal text-white/70 opacity-0 shadow-xl transition-all group-hover:visible group-hover:opacity-100">
                  LSTM é uma rede neural recorrente projetada para aprender padrões em séries temporais. É ideal para previsão de preços de ações por capturar dependências de longo prazo.
                  <div className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-[#1a1c24]" />
                </div>
              </div>
            </div>
            <p className="mt-1 font-semibold text-white">LSTM (Long Short-Term Memory)</p>
            <p className="mt-1 text-sm text-white/50">
              A recurrent neural network capable of learning long-term dependencies in sequential data.
            </p>
          </div>
          <div className="gradient-card rounded-xl border border-blue-400/20 p-5">
            <div className="flex items-center gap-2">
              <h4 className="text-sm font-semibold text-blue-400"><Zap className="mr-1 inline h-3.5 w-3.5" /> Key Characteristics</h4>
              <div className="group relative">
                <Info className="h-3 w-3 cursor-help text-white/25 transition-colors hover:text-white/60" />
                <div className="invisible absolute bottom-full left-1/2 z-50 mb-2 w-64 -translate-x-1/2 rounded-lg border border-surface-border bg-[#1a1c24] px-3 py-2 text-[11px] font-normal text-white/70 opacity-0 shadow-xl transition-all group-hover:visible group-hover:opacity-100">
                  Características principais da LSTM: processamento sequencial para dados ordenados no tempo, células de memória para reter informações, e gates para controlar o fluxo de dados.
                  <div className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-[#1a1c24]" />
                </div>
              </div>
            </div>
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
        <p className="mb-3 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-widest text-white/30">
          <Settings className="h-3.5 w-3.5" /> Architecture Configuration
        </p>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatCard label="Input Size" value={inputSize} subtitle="Number of input features" icon={<Box className="h-5 w-5 text-nvidia" />} tooltip="Número de features de entrada (Open, High, Low, Close, Volume). Cada timestep tem essas 5 dimensões." />
          <StatCard label="Hidden Size" value={hiddenSize} subtitle="LSTM hidden state dim" icon={<Layers className="h-5 w-5 text-sky-400" />} accentColor="#38bdf8" tooltip="Dimensão do estado oculto da LSTM. Valores maiores permitem capturar padrões mais complexos, mas aumentam o risco de overfitting." />
          <StatCard label="Num Layers" value={numLayers} subtitle="Stacked LSTM layers" icon={<Hash className="h-5 w-5 text-amber-400" />} accentColor="#fbbf24" tooltip="Número de camadas LSTM empilhadas. Mais camadas permitem abstrações hierárquicas, mas tornam o treinamento mais lento." />
          <StatCard label="Output Size" value={`${outputSize} → 1`} subtitle="OHLCV out → Close only" icon={<Cpu className="h-5 w-5 text-purple-400" />} accentColor="#a78bfa" tooltip="The FC layer outputs 5 values (OHLCV), but only the Close price (index 3) is extracted for the final prediction." />
        </div>
        <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="rounded-lg border border-surface-border bg-surface-card p-4">
            <div className="flex items-center gap-1.5">
              <p className="text-xs text-white/40">Dropout</p>
              <div className="group relative">
                <Info className="h-3 w-3 cursor-help text-white/25 transition-colors hover:text-white/60" />
                <div className="invisible absolute bottom-full left-1/2 z-50 mb-2 w-56 -translate-x-1/2 rounded-lg border border-surface-border bg-[#1a1c24] px-3 py-2 text-[11px] font-normal text-white/70 opacity-0 shadow-xl transition-all group-hover:visible group-hover:opacity-100">
                  Taxa de dropout aplicada entre as camadas LSTM e antes da camada densa. Desativa aleatoriamente neurônios durante o treino para evitar overfitting.
                  <div className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-[#1a1c24]" />
                </div>
              </div>
            </div>
            <p className="text-xl font-bold">{(dropout * 100).toFixed(0)}%</p>
          </div>
          <div className="rounded-lg border border-surface-border bg-surface-card p-4">
            <div className="flex items-center gap-1.5">
              <p className="text-xs text-white/40">Bidirectional</p>
              <div className="group relative">
                <Info className="h-3 w-3 cursor-help text-white/25 transition-colors hover:text-white/60" />
                <div className="invisible absolute bottom-full left-1/2 z-50 mb-2 w-56 -translate-x-1/2 rounded-lg border border-surface-border bg-[#1a1c24] px-3 py-2 text-[11px] font-normal text-white/70 opacity-0 shadow-xl transition-all group-hover:visible group-hover:opacity-100">
                  Se ativado, a LSTM processa a sequência nos dois sentidos (passado→futuro e futuro→passado), dobrando o hidden size efetivo.
                  <div className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-[#1a1c24]" />
                </div>
              </div>
            </div>
            <p className="text-xl font-bold">{bidirectional ? "Yes" : "No"}</p>
          </div>
          <div className="rounded-lg border border-surface-border bg-surface-card p-4">
            <div className="flex items-center gap-1.5">
              <p className="text-xs text-white/40">Sequence Length</p>
              <div className="group relative">
                <Info className="h-3 w-3 cursor-help text-white/25 transition-colors hover:text-white/60" />
                <div className="invisible absolute bottom-full left-1/2 z-50 mb-2 w-56 -translate-x-1/2 rounded-lg border border-surface-border bg-[#1a1c24] px-3 py-2 text-[11px] font-normal text-white/70 opacity-0 shadow-xl transition-all group-hover:visible group-hover:opacity-100">
                  Número de dias passados que o modelo analisa para fazer uma previsão. 60 dias = ~3 meses de histórico por previsão.
                  <div className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-[#1a1c24]" />
                </div>
              </div>
            </div>
            <p className="text-xl font-bold">{seqLength}</p>
          </div>
          <div className="rounded-lg border border-surface-border bg-surface-card p-4">
            <div className="flex items-center gap-1.5">
              <p className="text-xs text-white/40">Directions</p>
              <div className="group relative">
                <Info className="h-3 w-3 cursor-help text-white/25 transition-colors hover:text-white/60" />
                <div className="invisible absolute bottom-full left-1/2 z-50 mb-2 w-56 -translate-x-1/2 rounded-lg border border-surface-border bg-[#1a1c24] px-3 py-2 text-[11px] font-normal text-white/70 opacity-0 shadow-xl transition-all group-hover:visible group-hover:opacity-100">
                  Número de direções de processamento. 1 = unidirecional (só olha o passado), 2 = bidirecional.
                  <div className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-[#1a1c24]" />
                </div>
              </div>
            </div>
            <p className="text-xl font-bold">{bidirectional ? 2 : 1}</p>
          </div>
        </div>
      </div>

      {/* Model Tree */}
      <div>
        <div className="mb-3 flex items-center gap-2">
          <p className="text-[10px] font-semibold uppercase tracking-widest text-white/30">🌳 Model Tree Structure</p>
          <div className="group relative">
            <Info className="h-3 w-3 cursor-help text-white/25 transition-colors hover:text-white/60" />
            <div className="invisible absolute bottom-full left-1/2 z-50 mb-2 w-64 -translate-x-1/2 rounded-lg border border-surface-border bg-[#1a1c24] px-3 py-2 text-[11px] font-normal text-white/70 opacity-0 shadow-xl transition-all group-hover:visible group-hover:opacity-100">
              Visualização hierárquica do modelo mostrando cada camada, seus parâmetros e a forma dos tensores em cada etapa.
              <div className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-[#1a1c24]" />
            </div>
          </div>
        </div>
        <div className="gradient-card rounded-xl border border-nvidia/30 p-6 font-mono text-sm">
          <div className="flex items-center gap-1.5 font-semibold text-nvidia"><Package className="h-4 w-4" /> NvidiaLSTM</div>
          <div className="ml-5 mt-2 border-l-2 border-nvidia/30 pl-5 space-y-3">
            {/* Input */}
            <div className="flex items-center gap-2">
              <span className="text-[#4ECDC4]"><ArrowDown className="inline h-4 w-4" /></span>
              <span className="font-semibold text-[#4ECDC4]">Input Layer</span>
              <span className="rounded bg-[#4ECDC4]/20 px-2 py-0.5 text-xs text-[#4ECDC4]">
                shape: (batch, {seqLength}, {inputSize})
              </span>
            </div>

            {/* LSTM */}
            <div>
              <div className="flex items-center gap-2">
                <span className="text-nvidia"><Brain className="inline h-4 w-4" /></span>
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
              <span className="text-[#45B7D1]"><Droplets className="inline h-4 w-4" /></span>
              <span className="font-semibold text-[#45B7D1]">Dropout</span>
              <span className="rounded bg-[#45B7D1]/20 px-2 py-0.5 text-xs text-[#45B7D1]">
                p={dropout}
              </span>
            </div>

            {/* FC */}
            <div>
              <div className="flex items-center gap-2">
                <span className="text-[#FF6B35]"><Link2 className="inline h-4 w-4" /></span>
                <span className="font-semibold text-[#FF6B35]">Linear (FC)</span>
                <span className="rounded bg-[#FF6B35]/20 px-2 py-0.5 text-xs text-[#FF6B35]">
                  {fcParams.toLocaleString()} params
                </span>
              </div>
              <div className="ml-7 mt-1 border-l-2 border-[#FF6B35]/20 pl-4 text-xs text-white/60 space-y-0.5">
                <div>├─ <span className="text-[#FF8C5A]">in_features:</span> {hiddenSize}</div>
                <div>└─ <span className="text-[#FF8C5A]">out_features:</span> {outputSize} <span className="text-white/30">(OHLCV)</span></div>
              </div>
            </div>

            {/* Close Extraction */}
            <div>
              <div className="flex items-center gap-2">
                <span className="text-[#F59E0B]"><Target className="inline h-4 w-4" /></span>
                <span className="font-semibold text-[#F59E0B]">Close Extraction</span>
                <span className="rounded bg-[#F59E0B]/20 px-2 py-0.5 text-xs text-[#F59E0B]">
                  index [3]
                </span>
              </div>
              <div className="ml-7 mt-1 border-l-2 border-[#F59E0B]/20 pl-4 text-xs text-white/50">
                Output has {outputSize} features (O,H,L,C,V) → only <span className="font-medium text-[#F59E0B]">Close</span> is used
              </div>
            </div>

            {/* Output */}
            <div className="flex items-center gap-2">
              <span className="text-[#96CEB4]"><ArrowUp className="inline h-4 w-4" /></span>
              <span className="font-semibold text-[#96CEB4]">Output</span>
              <span className="rounded bg-[#96CEB4]/20 px-2 py-0.5 text-xs text-[#96CEB4]">
                shape: (batch, 1) — Close Price
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
        <div className="mb-3 flex items-center gap-2">
          <p className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-widest text-white/30"><BarChart3 className="h-3.5 w-3.5" /> Parameter Analysis</p>
          <div className="group relative">
            <Info className="h-3 w-3 cursor-help text-white/25 transition-colors hover:text-white/60" />
            <div className="invisible absolute bottom-full left-1/2 z-50 mb-2 w-64 -translate-x-1/2 rounded-lg border border-surface-border bg-[#1a1c24] px-3 py-2 text-[11px] font-normal text-white/70 opacity-0 shadow-xl transition-all group-hover:visible group-hover:opacity-100">
              Análise quantitativa dos parâmetros do modelo. Cada parâmetro é um peso aprendido durante o treinamento. Mais parâmetros = maior capacidade, mas mais risco de overfitting.
              <div className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-[#1a1c24]" />
            </div>
          </div>
        </div>
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

        {/* Bar chart */}
        {pieData.length > 0 && (
          <div className="mt-4 rounded-xl border border-surface-border bg-surface-card p-6">
            <h4 className="mb-2 text-sm font-semibold text-white/80">
              Parameter Distribution by Layer Type
            </h4>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={pieData} layout="vertical" margin={{ left: 20, right: 30, top: 5, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                <XAxis
                  type="number"
                  tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 11 }}
                  tickLine={false}
                  tickFormatter={(v: number) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : String(v)}
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 11 }}
                  tickLine={false}
                  width={130}
                />
                <Tooltip
                  contentStyle={{
                    background: "#1a1c24",
                    border: "1px solid rgba(118,185,0,0.3)",
                    borderRadius: 8,
                  }}
                  formatter={(value: number) => [value.toLocaleString(), "Parameters"]}
                />
                <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={28}>
                  {pieData.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Layer Detail Table */}
      {params?.layers && (
        <div className="rounded-xl border border-surface-border bg-surface-card p-6">
          <SectionHeader title="Detailed Layer Information" tooltip="Tabela com cada camada do modelo, sua forma (shape), número de parâmetros e tipo de dado. Permite entender onde estão concentrados os pesos do modelo." />
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
          <SectionHeader title="Data Flow" tooltip="Fluxo completo dos dados desde a entrada até a previsão final. Mostra como os dados são transformados em cada etapa do modelo." />
        <div className="space-y-4">
          {[
            { step: "1. Input Preparation", desc: `Historical prices are normalized and shaped into sequences of ${seqLength} time steps.`, shape: `(batch, ${seqLength}, ${inputSize})` },
            { step: "2. LSTM Processing", desc: `${numLayers} stacked LSTM layers process the sequence, learning temporal patterns.`, shape: `(batch, ${seqLength}, ${hiddenSize})` },
            { step: "3. Final Hidden State", desc: "The last hidden state from the final LSTM layer is extracted.", shape: `(batch, ${hiddenSize})` },
            { step: "4. Dropout", desc: `Dropout with rate ${dropout} is applied for regularization.`, shape: `(batch, ${hiddenSize})` },
            { step: "5. Dense Layer", desc: `Fully connected layer maps hidden state to ${outputSize} OHLCV outputs.`, shape: `(batch, ${outputSize})` },
            { step: "6. Close Extraction", desc: "Only the Close price (index 3) is extracted from the 5-feature output.", shape: "(batch, 1)" },
            { step: "7. Inverse Transform", desc: "Close prediction is converted back to original price scale.", shape: "Predicted Close ($)" },
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
          <SectionHeader title="Training Configuration" tooltip="Hiperparâmetros usados no treinamento do modelo. Estes valores controlam a velocidade de aprendizado, regularização e critérios de parada." />
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

      {/* Best Parameters */}
      {(() => {
        const hpoParams = (hpo as Record<string, unknown>)?.best_params as Record<string, number> | undefined;
        const hasHpo = hpoParams && Object.keys(hpoParams).length > 0;

        // Fallback: use model config + training info as best params
        const bestParams: Record<string, number | string> = hasHpo
          ? hpoParams
          : {
              hidden_size: hiddenSize,
              num_layers: numLayers,
              dropout: dropout,
              learning_rate: Number(trainingInfo.learning_rate ?? 0.001),
              batch_size: Number(trainingInfo.batch_size ?? 32),
              sequence_length: seqLength,
              best_epoch: Number(trainingInfo["Best Epoch"] ?? trainingInfo.best_epoch ?? "-"),
              total_epochs: Number(trainingInfo["Total Epochs"] ?? trainingInfo.total_epochs ?? "-"),
            };

        const labelMap: Record<string, { label: string; icon: string; format: (v: number) => string }> = {
          hidden_size: { label: "Hidden Size", icon: "▸", format: (v) => String(v) },
          num_layers: { label: "Num Layers", icon: "▸", format: (v) => String(v) },
          learning_rate: { label: "Learning Rate", icon: "▸", format: (v) => v < 0.01 ? v.toExponential(2) : String(v) },
          dropout: { label: "Dropout", icon: "▸", format: (v) => `${(v * 100).toFixed(0)}%` },
          batch_size: { label: "Batch Size", icon: "▸", format: (v) => String(v) },
          weight_decay: { label: "Weight Decay", icon: "▸", format: (v) => v < 0.01 ? v.toExponential(2) : String(v) },
          sequence_length: { label: "Sequence Length", icon: "▸", format: (v) => String(v) },
          epochs: { label: "Epochs", icon: "▸", format: (v) => String(v) },
          best_epoch: { label: "Best Epoch", icon: "▸", format: (v) => String(v) },
          total_epochs: { label: "Total Epochs", icon: "▸", format: (v) => String(v) },
        };

        return (
          <div className="rounded-xl border border-amber-400/30 bg-gradient-to-br from-amber-400/5 to-transparent p-6">
            <SectionHeader
              title="Best Parameters"
              tooltip={hasHpo
                ? "Best hyperparameters found by automatic optimization (HPO — Hyperparameter Optimization). Selected via Bayesian search to maximize performance."
                : "Parameters used to train the current model. This configuration produced the best result (lowest validation loss)."
              }
            />
            <div className="mb-4 flex items-center gap-2 rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2">
              <Trophy className="h-4 w-4 text-amber-400" />
              <span className="text-xs text-amber-300/80">
                {hasHpo
                  ? "Automatically optimized via Optuna — selected for best performance"
                  : "Model configuration with best performance (lowest validation loss)"}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {Object.entries(bestParams)
                .filter(([, val]) => val !== "-" && val !== 0 && !isNaN(Number(val)))
                .map(([key, val]) => {
                  const meta = labelMap[key] ?? { label: key.replace(/_/g, " "), icon: "▸", format: (v: number) => String(v) };
                  return (
                    <div
                      key={key}
                      className="group relative rounded-lg border border-surface-border bg-surface-card p-4 transition-colors hover:border-amber-400/30"
                    >
                      <div className="flex items-center gap-1.5">
                        <span className="text-sm">{meta.icon}</span>
                        <p className="text-xs text-white/40">{meta.label}</p>
                      </div>
                      <p className="mt-1 text-xl font-bold text-amber-400">
                        {typeof val === "number" ? meta.format(val) : String(val)}
                      </p>
                    </div>
                  );
                })}
            </div>
          </div>
        );
      })()}
    </div>
  );
}
