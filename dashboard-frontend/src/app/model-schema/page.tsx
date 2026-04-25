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
import { PageHeader } from "@/components/page-header";

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
    <div className="mx-auto max-w-7xl space-y-6">
      {/* Header */}
      <PageHeader
        label="ML · Arquitetura"
        title="Arquitetura do"
        gradient="Modelo"
        subtitle="Explore a rede neural LSTM — camadas, configuração e distribuição de parâmetros."
        icon={Brain}
      />

      {/* Model Purpose */}
      <div className="rounded-xl border border-nvidia/20 bg-gradient-to-br from-nvidia/5 via-surface-card to-surface-card p-6">
        <div className="flex items-start gap-4">
          <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-xl bg-nvidia/10">
            <Target className="h-6 w-6 text-nvidia" />
          </div>
          <div>
            <h3 className="mb-2 text-lg font-semibold">Propósito do Modelo</h3>
            <p className="text-sm leading-relaxed text-white/60">
              Este modelo utiliza uma rede neural{" "}
              <span className="font-medium text-nvidia">LSTM (Long Short-Term Memory)</span> para prever o{" "}
              <span className="font-medium text-nvidia">Close price</span> da ação da{" "}
              <span className="font-medium text-nvidia">NVIDIA (NVDA)</span> pelos próximos 30 dias.
              LSTMs são especialmente eficazes em séries temporais por capturar dependências de longo prazo
              — o modelo analisa {seqLength} dias de histórico em 5 features de entrada
              (Open, High, Low, Close e Volume) para identificar padrões de tendência, sazonalidade e volatilidade,
              gerando uma previsão de Close price com intervalos de confiança via Monte Carlo Dropout.
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              {[
                { label: "Tipo", value: "Recurrent LSTM" },
                { label: "Ativo", value: "NVDA" },
                { label: "Alvo", value: "Close Price" },
                { label: "Horizonte", value: "30 dias" },
                { label: "Janela", value: `${seqLength} dias` },
                { label: "Entrada", value: `OHLCV (${inputSize} features)` },
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
          <ClipboardList className="h-3.5 w-3.5" /> Visão Geral do Modelo
        </p>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="gradient-card rounded-xl border border-nvidia/20 p-5">
            <div className="flex items-center gap-2">
              <h4 className="text-sm font-semibold text-nvidia"><Diamond className="mr-1 inline h-3.5 w-3.5" /> Tipo de Arquitetura</h4>
              <div className="group relative">
                <Info className="h-3 w-3 cursor-help text-white/25 transition-colors hover:text-white/60" />
                <div className="invisible absolute bottom-full left-1/2 z-50 mb-2 w-64 -translate-x-1/2 rounded-lg border border-surface-border bg-[#1a1c24] px-3 py-2 text-[11px] font-normal text-white/70 opacity-0 shadow-xl transition-all group-hover:visible group-hover:opacity-100">
                  LSTM é uma rede neural recorrente projetada para aprender padrões em séries temporais. Ideal para previsão de preços de ações por capturar dependências de longo prazo.
                  <div className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-[#1a1c24]" />
                </div>
              </div>
            </div>
            <p className="mt-1 font-semibold text-white">LSTM (Long Short-Term Memory)</p>
            <p className="mt-1 text-sm text-white/50">
              Rede neural recorrente capaz de aprender dependências de longo prazo em dados sequenciais.
            </p>
          </div>
          <div className="gradient-card rounded-xl border border-blue-400/20 p-5">
            <div className="flex items-center gap-2">
              <h4 className="text-sm font-semibold text-blue-400"><Zap className="mr-1 inline h-3.5 w-3.5" /> Características Principais</h4>
              <div className="group relative">
                <Info className="h-3 w-3 cursor-help text-white/25 transition-colors hover:text-white/60" />
                <div className="invisible absolute bottom-full left-1/2 z-50 mb-2 w-64 -translate-x-1/2 rounded-lg border border-surface-border bg-[#1a1c24] px-3 py-2 text-[11px] font-normal text-white/70 opacity-0 shadow-xl transition-all group-hover:visible group-hover:opacity-100">
                  Características principais da LSTM: processamento sequencial para dados ordenados no tempo, Memory Cells para reter informações e Gates para controlar o fluxo de dados.
                  <div className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-[#1a1c24]" />
                </div>
              </div>
            </div>
            <div className="mt-2 space-y-2">
              {[
                "Sequential Processing — Processa os dados um passo de cada vez",
                "Memory Cells — Mantém informações ao longo de longas sequências",
                "Gating Mechanism — Gates Input, Forget e Output",
                `Bidirectional — ${bidirectional ? "Ativado" : "Não ativado"}`,
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
          <Settings className="h-3.5 w-3.5" /> Configuração da Arquitetura
        </p>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatCard label="Input Size" value={inputSize} subtitle="Número de features de entrada" icon={<Box className="h-5 w-5 text-nvidia" />} tooltip="Número de features de entrada (Open, High, Low, Close, Volume). Cada timestep possui essas 5 dimensões." />
          <StatCard label="Hidden Size" value={hiddenSize} subtitle="Dimensão do hidden state LSTM" icon={<Layers className="h-5 w-5 text-sky-400" />} accentColor="#38bdf8" tooltip="Dimensão do hidden state da LSTM. Valores maiores capturam padrões mais complexos, mas aumentam o risco de overfitting." />
          <StatCard label="Num Layers" value={numLayers} subtitle="Camadas LSTM empilhadas" icon={<Hash className="h-5 w-5 text-amber-400" />} accentColor="#fbbf24" tooltip="Número de camadas LSTM empilhadas. Mais camadas permitem abstrações hierárquicas, mas tornam o treinamento mais lento." />
          <StatCard label="Output Size" value={`${outputSize} → 1`} subtitle="OHLCV out → Close only" icon={<Cpu className="h-5 w-5 text-purple-400" />} accentColor="#a78bfa" tooltip="A camada FC gera 5 valores (OHLCV), mas apenas o Close price (índice 3) é extraído para a previsão final." />
        </div>
        <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="rounded-lg border border-surface-border bg-surface-card p-4">
            <div className="flex items-center gap-1.5">
              <p className="text-xs text-white/40">Dropout</p>
              <div className="group relative">
                <Info className="h-3 w-3 cursor-help text-white/25 transition-colors hover:text-white/60" />
                <div className="invisible absolute bottom-full left-1/2 z-50 mb-2 w-56 -translate-x-1/2 rounded-lg border border-surface-border bg-[#1a1c24] px-3 py-2 text-[11px] font-normal text-white/70 opacity-0 shadow-xl transition-all group-hover:visible group-hover:opacity-100">
                  Taxa de dropout aplicada entre as camadas LSTM e antes da camada Dense. Desativa neurônios aleatoriamente durante o treinamento para evitar overfitting.
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
                  Quando ativado, a LSTM processa a sequência nos dois sentidos (passado→futuro e futuro→passado), dobrando o hidden size efetivo.
                  <div className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-[#1a1c24]" />
                </div>
              </div>
            </div>
            <p className="text-xl font-bold">{bidirectional ? "Sim" : "Não"}</p>
          </div>
          <div className="rounded-lg border border-surface-border bg-surface-card p-4">
            <div className="flex items-center gap-1.5">
              <p className="text-xs text-white/40">Sequence Length</p>
              <div className="group relative">
                <Info className="h-3 w-3 cursor-help text-white/25 transition-colors hover:text-white/60" />
                <div className="invisible absolute bottom-full left-1/2 z-50 mb-2 w-56 -translate-x-1/2 rounded-lg border border-surface-border bg-[#1a1c24] px-3 py-2 text-[11px] font-normal text-white/70 opacity-0 shadow-xl transition-all group-hover:visible group-hover:opacity-100">
                  Número de dias anteriores que o modelo analisa para fazer uma previsão. 60 dias ≈ 3 meses de histórico por inferência.
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
                  Número de direções de processamento. 1 = unidirecional (apenas o passado), 2 = bidirecional.
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
          <p className="text-[10px] font-semibold uppercase tracking-widest text-white/30">🌳 Estrutura do Modelo</p>
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
          <p className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-widest text-white/30"><BarChart3 className="h-3.5 w-3.5" /> Análise de Parâmetros</p>
          <div className="group relative">
            <Info className="h-3 w-3 cursor-help text-white/25 transition-colors hover:text-white/60" />
            <div className="invisible absolute bottom-full left-1/2 z-50 mb-2 w-64 -translate-x-1/2 rounded-lg border border-surface-border bg-[#1a1c24] px-3 py-2 text-[11px] font-normal text-white/70 opacity-0 shadow-xl transition-all group-hover:visible group-hover:opacity-100">
              Análise quantitativa dos parâmetros do modelo. Cada parâmetro é um peso aprendido durante o treinamento. Mais parâmetros = maior capacidade, mas maior risco de overfitting.
              <div className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-[#1a1c24]" />
            </div>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-4">
          <div className="rounded-lg border border-surface-border bg-surface-card p-4">
            <p className="text-xs text-white/40">Total de Parâmetros</p>
            <p className="text-2xl font-bold">{params?.total.toLocaleString()}</p>
          </div>
          <div className="rounded-lg border border-surface-border bg-surface-card p-4">
            <p className="text-xs text-white/40">Parâmetros Treináveis</p>
            <p className="text-2xl font-bold">{params?.trainable.toLocaleString()}</p>
          </div>
          <div className="rounded-lg border border-surface-border bg-surface-card p-4">
            <p className="text-xs text-white/40">Tamanho do Modelo</p>
            <p className="text-2xl font-bold">{sizeMB.toFixed(2)} MB</p>
          </div>
        </div>

        {/* Bar chart */}
        {pieData.length > 0 && (
          <div className="mt-4 rounded-xl border border-surface-border bg-surface-card p-6">
            <h4 className="mb-2 text-sm font-semibold text-white/80">
              Distribuição de Parâmetros por Tipo de Camada
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
                  formatter={(value: number) => [value.toLocaleString(), "Parâmetros"]}
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
          <SectionHeader title="Informações Detalhadas por Camada" tooltip="Tabela com cada camada do modelo, seu shape, número de parâmetros e tipo de dado. Permite identificar onde os pesos estão concentrados." />
          <div className="overflow-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-surface-border text-left text-xs text-white/40">
                <th className="pb-2">Camada</th>
                  <th className="pb-2">Shape</th>
                  <th className="pb-2">Parâmetros</th>
                  <th className="pb-2">Tipo</th>
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
          <SectionHeader title="Fluxo de Dados" tooltip="Fluxo completo dos dados desde a entrada até a previsão final. Mostra como os dados são transformados em cada etapa do modelo." />
        <div className="space-y-4">
          {[
            { step: "1. Preparação dos Dados", desc: `Preços históricos são normalizados e organizados em sequências de ${seqLength} timesteps.`, shape: `(batch, ${seqLength}, ${inputSize})` },
            { step: "2. Processamento LSTM", desc: `${numLayers} camadas LSTM empilhadas processam a sequência, aprendendo padrões temporais.`, shape: `(batch, ${seqLength}, ${hiddenSize})` },
            { step: "3. Hidden State Final", desc: "O último hidden state da camada LSTM final é extraído.", shape: `(batch, ${hiddenSize})` },
            { step: "4. Dropout", desc: `Dropout com taxa ${dropout} é aplicado para regularização.`, shape: `(batch, ${hiddenSize})` },
            { step: "5. Camada Dense", desc: `Camada totalmente conectada mapeia o hidden state para ${outputSize} saídas OHLCV.`, shape: `(batch, ${outputSize})` },
            { step: "6. Extração do Close", desc: "Apenas o Close price (índice 3) é extraído das 5 saídas.", shape: "(batch, 1)" },
            { step: "7. Transformação Inversa", desc: "A previsão de Close é convertida de volta para a escala de preço original.", shape: "Close Previsto ($)" },
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
          <SectionHeader title="Configuração de Treinamento" tooltip="Hiperparâmetros usados no treinamento do modelo. Esses valores controlam a taxa de aprendizado, regularização e critérios de parada." />
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            <div>
              <h4 className="mb-2 text-sm font-semibold text-white/60">Optimizer &amp; Aprendizado</h4>
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
              <h4 className="mb-2 text-sm font-semibold text-white/60">Regularização &amp; Parada</h4>
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
                      {typeof v === "boolean" ? (v ? "Sim" : "Não") : String(v)}
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
              title="Melhores Parâmetros"
              tooltip={hasHpo
                ? "Melhores hiperparâmetros encontrados por otimização automática (HPO — Hyperparameter Optimization). Selecionados via Bayesian search para maximizar a performance."
                : "Parâmetros usados para treinar o modelo atual. Essa configuração produziu o melhor resultado (menor validation loss)."
              }
            />
            <div className="mb-4 flex items-center gap-2 rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2">
              <Trophy className="h-4 w-4 text-amber-400" />
              <span className="text-xs text-amber-300/80">
                {hasHpo
                  ? "Otimizado automaticamente via Optuna — selecionado pela melhor performance"
                  : "Configuração do modelo com melhor desempenho (menor validation loss)"}
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
