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
  HardDrive,
  Globe,
  Flame,
  MemoryStick,
  Network,
  Activity,
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
import { PageHeader } from "@/components/page-header";
import { InfoTooltip } from "@/components/info-tooltip";

/* ── Helpers ────────────────────────────────────────────────── */
function StatCard({
  title,
  value,
  subtitle,
  icon: Icon,
  color = "text-nvidia",
  trend,
  info,
}: {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ElementType;
  color?: string;
  trend?: "up" | "down" | "neutral";
  info?: string;
}) {
  return (
    <div className="rounded-xl border border-surface-border bg-surface-card p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <p className="text-xs text-white/50">{title}</p>
          {info && <InfoTooltip text={info} />}
        </div>
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
  { id: "business", label: "Métricas de Negócio", icon: TrendingUp },
  { id: "registry", label: "Registro de Modelos", icon: Layers },
  { id: "features", label: "Feature Store", icon: Database },
  { id: "canary", label: "Canary Deploy", icon: Zap },
  { id: "cost", label: "Análise de Custos", icon: DollarSign },
  { id: "hardware", label: "Hardware Setup", icon: HardDrive },
] as const;

type TabId = (typeof TABS)[number]["id"];

const COLORS = ["#76b900", "#0ea5e9", "#f59e0b", "#ef4444", "#a855f7", "#14b8a6"];

/* ══════════════════════════════════════════════════════════════
   HARDWARE SETUP TAB — estimativa de recursos por componente
══════════════════════════════════════════════════════════════ */

type SpecRow = { label: string; min: string; rec: string; prod: string };

function SpecTable({ rows }: { rows: SpecRow[] }) {
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="border-b border-surface-border text-left text-white/30">
          <th className="pb-1.5 font-medium">Recurso</th>
          <th className="pb-1.5 font-medium text-amber-400">Mínimo</th>
          <th className="pb-1.5 font-medium text-nvidia">Recomendado</th>
          <th className="pb-1.5 font-medium text-sky-400">Produção</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.label} className="border-b border-surface-border/40">
            <td className="py-1.5 text-white/50">{r.label}</td>
            <td className="py-1.5 text-amber-300">{r.min}</td>
            <td className="py-1.5 font-semibold text-nvidia">{r.rec}</td>
            <td className="py-1.5 text-sky-300">{r.prod}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function HwCard({
  icon: Icon,
  iconColor,
  title,
  badge,
  badgeColor,
  note,
  rows,
  infoText,
  cost,
}: {
  icon: React.ElementType;
  iconColor: string;
  title: string;
  badge: string;
  badgeColor: string;
  note: string;
  rows: SpecRow[];
  infoText: string;
  cost: { min: string; rec: string; prod: string; unit?: string };
}) {
  return (
    <div className="rounded-xl border border-surface-border bg-surface-card p-5 space-y-4">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <Icon className={`h-5 w-5 ${iconColor}`} />
          <h3 className="font-semibold text-white">{title}</h3>
          <InfoTooltip text={infoText} />
        </div>
        <span className={`rounded-full border px-2.5 py-0.5 text-[10px] font-semibold ${badgeColor}`}>
          {badge}
        </span>
      </div>
      <SpecTable rows={rows} />
      <p className="text-[11px] text-white/35 leading-relaxed">{note}</p>
      {/* Cost footer */}
      <div className="rounded-lg border border-green-500/20 bg-green-500/5 px-3 py-2">
        <div className="flex items-center gap-1.5 mb-1.5">
          <DollarSign className="h-3 w-3 text-green-400" />
          <span className="text-[10px] font-semibold text-green-400 uppercase tracking-wide">
            Custo estimado{cost.unit ? ` (${cost.unit})` : " / mês"}
          </span>
        </div>
        <div className="flex gap-3 text-[11px]">
          <span className="text-white/40">Mín <span className="text-amber-300 font-semibold">{cost.min}</span></span>
          <span className="text-white/20">·</span>
          <span className="text-white/40">Rec <span className="text-green-300 font-bold">{cost.rec}</span></span>
          <span className="text-white/20">·</span>
          <span className="text-white/40">Prod <span className="text-sky-300 font-semibold">{cost.prod}</span></span>
        </div>
      </div>
    </div>
  );
}

function HardwareSetupTab() {
  return (
    <div className="space-y-6">
      {/* Summary banner */}
      <div className="flex flex-wrap items-center gap-4 rounded-xl border border-nvidia/30 bg-nvidia/5 p-4">
        <HardDrive className="h-6 w-6 text-nvidia shrink-0" />
        <div className="flex-1 min-w-0">
          <p className="font-semibold text-white">Estimativa de Infraestrutura — Stack E2E</p>
          <p className="text-xs text-white/40 mt-0.5">
            Baseada no perfil atual: LSTM 3-camadas · RAG + ChromaDB · FastAPI · Next.js · MLflow
          </p>
        </div>
        <div className="flex flex-wrap gap-3 text-xs">
          {[
            { label: "vCPU total", value: "8–16", color: "text-nvidia" },
            { label: "RAM total", value: "16–32 GB", color: "text-sky-400" },
            { label: "Storage", value: "60–120 GB", color: "text-amber-400" },
            { label: "GPU (opcional)", value: "T4 / A10", color: "text-purple-400" },
          ].map((s) => (
            <div key={s.label} className="rounded-lg border border-surface-border bg-surface-hover px-3 py-1.5 text-center">
              <p className="text-white/40">{s.label}</p>
              <p className={`font-bold ${s.color}`}>{s.value}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Legenda */}
      <div className="flex items-center gap-5 text-[11px]">
        <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-amber-400" />Mínimo (dev / demo)</span>
        <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-nvidia" />Recomendado (staging)</span>
        <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-sky-400" />Produção (alta disponibilidade)</span>
      </div>

      {/* Cards — linha 1 */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        <HwCard
          icon={Globe}
          iconColor="text-sky-400"
          title="Web — Next.js Frontend"
          badge="CPU"
          badgeColor="border-sky-400/30 text-sky-400"
          note="Aplicação estática servida via Nginx. Bundle inicial ~87 KB JS. Sem SSR — não exige CPU significativa em runtime."
          infoText="Next.js exportado como estático. Servido por Nginx container. Recursos dominados pelo número de conexões simultâneas."
          cost={{ min: "~$5", rec: "~$10", prod: "~$20" }}
          rows={[
            { label: "vCPU", min: "0.5", rec: "1", prod: "2" },
            { label: "RAM", min: "256 MB", rec: "512 MB", prod: "1 GB" },
            { label: "Storage", min: "1 GB", rec: "2 GB", prod: "5 GB" },
            { label: "Rede", min: "10 Mbps", rec: "100 Mbps", prod: "1 Gbps" },
          ]}
        />

        <HwCard
          icon={Server}
          iconColor="text-nvidia"
          title="FastAPI — Backend"
          badge="CPU"
          badgeColor="border-nvidia/30 text-nvidia"
          note="Uvicorn async. Inclui endpoints de SLA, Feature Store, Canary, RAG queries e chamadas ao ChromaDB. Pico de ~50 req/s durante previsões em lote."
          infoText="Uvicorn workers assíncronos. Cada query RAG pode bloquear por ~200 ms (embedding + ChromaDB lookup). Escalar com múltiplos workers."
          cost={{ min: "~$35", rec: "~$75", prod: "~$150" }}
          rows={[
            { label: "vCPU", min: "2", rec: "4", prod: "8" },
            { label: "RAM", min: "2 GB", rec: "4 GB", prod: "8 GB" },
            { label: "Storage", min: "5 GB", rec: "10 GB", prod: "20 GB" },
            { label: "Rede", min: "100 Mbps", rec: "1 Gbps", prod: "10 Gbps" },
          ]}
        />

        <HwCard
          icon={Brain}
          iconColor="text-purple-400"
          title="LSTM — Inferência"
          badge="CPU / GPU"
          badgeColor="border-purple-400/30 text-purple-400"
          note="Modelo LSTM 3 camadas, ~1.2 M parâmetros. Latência ~12 ms (CPU) / ~2 ms (GPU). Janela de entrada: 60 timesteps × 9 features."
          infoText="Inferência via PyTorch CPU é viável para até ~200 req/s. GPU reduz latência 6× e permite batching. Recomendado T4 em produção."
          cost={{ min: "~$35", rec: "~$75", prod: "~$380" }}
          rows={[
            { label: "vCPU", min: "2", rec: "4", prod: "4" },
            { label: "RAM", min: "2 GB", rec: "4 GB", prod: "8 GB" },
            { label: "GPU VRAM", min: "—", rec: "—", prod: "16 GB (T4)" },
            { label: "Latência", min: "~25 ms", rec: "~12 ms", prod: "~2 ms" },
          ]}
        />
      </div>

      {/* Cards — linha 2 */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        <HwCard
          icon={Flame}
          iconColor="text-orange-400"
          title="LSTM — Treinamento"
          badge="CPU / GPU"
          badgeColor="border-orange-400/30 text-orange-400"
          note="Treinamento com 50 épocas, batch 32, Adam optimizer. Dataset PETR4 (~5 anos, ~1.200 amostras). HPO (Optuna) executa até 20 trials."
          infoText="Com CPU: ~45 min por run completo. Com GPU A100: ~3 min. HPO com 20 trials multiplica o tempo. Idealmente em nó dedicado de treinamento."
          cost={{ min: "~$0.50", rec: "~$1.50", prod: "~$8", unit: "por run" }}
          rows={[
            { label: "vCPU", min: "4", rec: "8", prod: "16" },
            { label: "RAM", min: "8 GB", rec: "16 GB", prod: "32 GB" },
            { label: "GPU VRAM", min: "—", rec: "T4 16 GB", prod: "A100 40 GB" },
            { label: "Tempo/run", min: "~45 min", rec: "~10 min", prod: "~3 min" },
          ]}
        />

        <HwCard
          icon={Database}
          iconColor="text-teal-400"
          title="ChromaDB — Vector Store"
          badge="CPU + SSD"
          badgeColor="border-teal-400/30 text-teal-400"
          note="Armazena embeddings de documentos financeiros para o pipeline RAG. Corpus atual: ~120 documentos, ~3.000 chunks, dimensão 1536 (OpenAI ada-002)."
          infoText="ChromaDB persiste em disco. Busca vetorial ~5 ms para corpus atual. Escalar com réplicas ou migrar para Qdrant/Weaviate em produção."
          cost={{ min: "~$15", rec: "~$30", prod: "~$60" }}
          rows={[
            { label: "vCPU", min: "1", rec: "2", prod: "4" },
            { label: "RAM", min: "2 GB", rec: "4 GB", prod: "8 GB" },
            { label: "Storage (SSD)", min: "5 GB", rec: "20 GB", prod: "100 GB" },
            { label: "Latência busca", min: "~20 ms", rec: "~8 ms", prod: "~3 ms" },
          ]}
        />

        <HwCard
          icon={HardDrive}
          iconColor="text-amber-400"
          title="SQLite + MLflow"
          badge="CPU + Disco"
          badgeColor="border-amber-400/30 text-amber-400"
          note="SQLite: logs estruturados, métricas SLA, Feature Store, dados de negócio. MLflow: tracking de experimentos, artefatos de modelos (checkpoints .pt)."
          infoText="SQLite não suporta writes concorrentes — adequado para demo e staging. Em produção, migrar para PostgreSQL. MLflow pode usar S3 para artefatos."
          cost={{ min: "~$8", rec: "~$15", prod: "~$30" }}
          rows={[
            { label: "vCPU", min: "0.5", rec: "1", prod: "2" },
            { label: "RAM", min: "256 MB", rec: "1 GB", prod: "4 GB" },
            { label: "Storage", min: "5 GB", rec: "20 GB", prod: "50 GB" },
            { label: "IOPS", min: "100", rec: "500", prod: "3.000+" },
          ]}
        />
      </div>

      {/* Tabela comparativa completa */}
      <div className="rounded-xl border border-surface-border bg-surface-card p-6 overflow-x-auto">
        <h3 className="mb-4 flex items-center gap-2 font-semibold text-white">
          <Activity className="h-4 w-4 text-nvidia" />
          Resumo — Stack Completa
          <InfoTooltip text="Soma de todos os componentes. Produção assume alta disponibilidade com 2 réplicas da API e inferência em GPU dedicada." />
        </h3>
        <table className="w-full text-sm min-w-[600px]">
          <thead>
            <tr className="border-b border-surface-border text-left text-xs text-white/30">
              <th className="pb-2 font-medium">Componente</th>
              <th className="pb-2 font-medium">Tipo</th>
              <th className="pb-2 font-medium text-amber-400">vCPU (rec)</th>
              <th className="pb-2 font-medium text-sky-400">RAM (rec)</th>
              <th className="pb-2 font-medium text-purple-400">GPU</th>
              <th className="pb-2 font-medium text-teal-400">Storage</th>
              <th className="pb-2 font-medium text-green-400">Custo (rec)</th>
            </tr>
          </thead>
          <tbody>
            {[
              { name: "Next.js",          type: "Web",       cpu: "1",  ram: "512 MB", gpu: "—",          storage: "2 GB",  costRec: "~$10/mo"  },
              { name: "FastAPI",          type: "Backend",   cpu: "4",  ram: "4 GB",   gpu: "—",          storage: "10 GB", costRec: "~$75/mo"  },
              { name: "LSTM Inferência",  type: "ML",        cpu: "4",  ram: "4 GB",   gpu: "T4 opcional", storage: "2 GB",  costRec: "~$75/mo"  },
              { name: "LSTM Treinamento", type: "ML (batch)",cpu: "8",  ram: "16 GB",  gpu: "T4 / A100",  storage: "5 GB",  costRec: "~$1.50/run"},
              { name: "ChromaDB",         type: "Vector DB", cpu: "2",  ram: "4 GB",   gpu: "—",          storage: "20 GB", costRec: "~$30/mo"  },
              { name: "SQLite",           type: "Storage",   cpu: "1",  ram: "1 GB",   gpu: "—",          storage: "20 GB", costRec: "~$8/mo"   },
              { name: "MLflow",           type: "Tracking",  cpu: "1",  ram: "1 GB",   gpu: "—",          storage: "20 GB", costRec: "~$7/mo"   },
            ].map((row, i) => (
              <tr key={row.name} className={`border-b border-surface-border/40 ${i % 2 === 0 ? "" : "bg-surface-hover/30"}`}>
                <td className="py-2 font-medium text-white">{row.name}</td>
                <td className="py-2 text-white/40">{row.type}</td>
                <td className="py-2 text-amber-300">{row.cpu}</td>
                <td className="py-2 text-sky-300">{row.ram}</td>
                <td className="py-2 text-purple-300">{row.gpu}</td>
                <td className="py-2 text-teal-300">{row.storage}</td>
                <td className="py-2 text-green-300 font-semibold">{row.costRec}</td>
              </tr>
            ))}
            <tr className="border-t-2 border-nvidia/40 bg-nvidia/5">
              <td className="py-2.5 font-bold text-nvidia">TOTAL</td>
              <td className="py-2.5 text-white/40">—</td>
              <td className="py-2.5 font-bold text-nvidia">21 vCPU</td>
              <td className="py-2.5 font-bold text-nvidia">30.5 GB</td>
              <td className="py-2.5 font-bold text-nvidia">T4 + A100</td>
              <td className="py-2.5 font-bold text-nvidia">79 GB</td>
              <td className="py-2.5 font-bold text-green-400">~$205/mo <span className="text-white/30 font-normal text-[10px]">+ treino</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

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
      } else if (activeTab === "hardware") {
        // static tab — no API call needed
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
    <div className="mx-auto max-w-7xl space-y-6">
      {/* Header */}
      <PageHeader
        label="Operações · Produção"
        title="MLOps &"
        gradient="Monitoramento Avançado"
        subtitle="SLA, métricas de negócio, registro de modelos, feature store e canary deploy."
        icon={Settings}
      />
      <div className="flex justify-end">
        <button
          onClick={loadData}
          disabled={loading}
          className="flex items-center gap-2 rounded-lg border border-nvidia/30 bg-nvidia/10 px-4 py-2 text-sm text-nvidia transition hover:bg-nvidia/20 disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Atualizar
        </button>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 rounded-xl border border-surface-border bg-surface-card p-1">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{ fontFamily: "'Outfit', sans-serif" }}
            className={`flex flex-1 items-center justify-center rounded-lg px-3 py-2.5 text-sm font-light tracking-wide transition ${
              activeTab === tab.id
                ? "bg-nvidia/20 text-nvidia"
                : "text-white hover:bg-surface-hover hover:text-white"
            }`}
          >
            {tab.label}
          </button>
        ))}
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
                {(slaReport as Record<string, unknown>).overall_sla_met ? "Todos os SLAs cumpridos" : "SLA violado"}
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
            <StatCard title="Uptime" value={`${slaReport.uptime_pct}%`} subtitle={`Meta: ${(slaReport.sla_targets as Record<string, number>)?.uptime_pct ?? 99.5}%`} icon={Shield} color={Number(slaReport.uptime_pct) >= 99.5 ? "text-green-400" : "text-red-400"} info="Percentual de tempo em que o serviço esteve disponível nas últimas 24 horas." />
            <StatCard title="p95 Latency" value={`${slaReport.p95_latency_ms}ms`} subtitle={`Meta: <${(slaReport.sla_targets as Record<string, number>)?.latency_p95_ms ?? 500}ms`} icon={Clock} color={Number(slaReport.p95_latency_ms) <= 500 ? "text-green-400" : "text-yellow-400"} info="Latência no percentil 95 — 95% das requisições foram respondidas abaixo deste valor." />
            <StatCard title="Taxa de Erros" value={`${slaReport.error_rate_pct}%`} subtitle={`Meta: <${(slaReport.sla_targets as Record<string, number>)?.error_rate_pct ?? 1}%`} icon={AlertTriangle} color={Number(slaReport.error_rate_pct) <= 1 ? "text-green-400" : "text-red-400"} info="Percentual de requisições que resultaram em erro (HTTP 5xx) nas últimas 24 horas." />
            <StatCard title="Requisições (24h)" value={String(slaReport.total_requests)} subtitle={`${slaReport.error_requests} erros`} icon={BarChart3} info="Total de requisições recebidas pelo serviço nas últimas 24 horas." />
          </div>

          {/* Uptime History Chart */}
          <div className="rounded-xl border border-surface-border bg-surface-card p-5">
            <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-white">
              Uptime & Latência (7 dias)
              <InfoTooltip text="Histórico diário de disponibilidade (%) e latência média (ms) do serviço nos últimos 7 dias." />
            </h3>
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
              <div className="flex items-center justify-center gap-1.5">
                <p className="text-xs text-white/50">p50 Latency</p>
                <InfoTooltip text="Mediana da latência — metade das requisições foram mais rápidas que este valor." />
              </div>
              <p className="mt-1 text-xl font-bold text-white">{String(slaReport.p50_latency_ms)} ms</p>
            </div>
            <div className="rounded-xl border border-surface-border bg-surface-card p-4 text-center">
              <div className="flex items-center justify-center gap-1.5">
                <p className="text-xs text-white/50">p95 Latency</p>
                <InfoTooltip text="95% das requisições foram respondidas abaixo deste valor. Indica a experiência da maioria dos usuários." />
              </div>
              <p className="mt-1 text-xl font-bold text-yellow-400">{String(slaReport.p95_latency_ms)} ms</p>
            </div>
            <div className="rounded-xl border border-surface-border bg-surface-card p-4 text-center">
              <div className="flex items-center justify-center gap-1.5">
                <p className="text-xs text-white/50">p99 Latency</p>
                <InfoTooltip text="99% das requisições foram respondidas abaixo deste valor. Identifica os piores casos de latência." />
              </div>
              <p className="mt-1 text-xl font-bold text-orange-400">{String(slaReport.p99_latency_ms)} ms</p>
            </div>
          </div>
        </div>
      )}

      {/* ══════════════ BUSINESS METRICS TAB ══════════════ */}
      {activeTab === "business" && bizSnapshot && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatCard title="P&L Acumulado" value={`$${Number(bizSnapshot.cumulative_pnl).toFixed(2)}`} icon={TrendingUp} color={Number(bizSnapshot.cumulative_pnl) >= 0 ? "text-green-400" : "text-red-400"} trend={Number(bizSnapshot.cumulative_pnl) >= 0 ? "up" : "down"} info="Lucro e Prejuízo acumulado gerado pelas predições do modelo ao longo do período." />
            <StatCard title="ROI" value={`${Number(bizSnapshot.roi_pct).toFixed(1)}%`} icon={BarChart3} color={Number(bizSnapshot.roi_pct) >= 0 ? "text-green-400" : "text-red-400"} info="Retorno sobre Investimento das operações guiadas pelas predições do modelo." />
            <StatCard title="Sharpe Ratio" value={Number(bizSnapshot.sharpe_ratio).toFixed(2)} subtitle={Number(bizSnapshot.sharpe_ratio) >= 2 ? "Excelente" : Number(bizSnapshot.sharpe_ratio) >= 1 ? "Bom" : "Baixo"} icon={Shield} color={Number(bizSnapshot.sharpe_ratio) >= 1 ? "text-green-400" : "text-yellow-400"} info="Retorno ajustado ao risco. Valores >1 indicam boa relação risco-retorno; >2 é excelente." />
            <StatCard title="Taxa de Acerto" value={`${Number(bizSnapshot.win_rate).toFixed(1)}%`} subtitle={`${bizSnapshot.winning_predictions}/${bizSnapshot.total_predictions} predições`} icon={CheckCircle} color={Number(bizSnapshot.win_rate) >= 50 ? "text-green-400" : "text-red-400"} info="Percentual de predições direcionais corretas (acima/abaixo do preço atual)." />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <StatCard title="Max Drawdown" value={`${Number(bizSnapshot.max_drawdown).toFixed(2)}%`} icon={AlertTriangle} color="text-red-400" info="Maior queda percentual acumulada observada no período — mede o pior cenário de perda." />
            <StatCard title="Erro Médio" value={`${Number(bizSnapshot.avg_error_pct).toFixed(2)}%`} icon={XCircle} color="text-yellow-400" info="Erro percentual médio das predições de preço em relação ao valor real." />
          </div>

          {/* P&L Chart */}
          <div className="rounded-xl border border-surface-border bg-surface-card p-5">
            <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-white">
              Evolução do P&L (60 dias)
              <InfoTooltip text="Curva de Lucro e Prejuízo acumulado ao longo dos últimos 60 dias de operação guiada pelo modelo." />
            </h3>
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
              Nenhum modelo registrado
            </div>
          ) : (
            <>
              <div className="rounded-xl border border-surface-border bg-surface-card p-5">
                <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-white">
                  Versões do Modelo
                  <InfoTooltip text="Versões registradas no MLflow com métricas de avaliação e estágio atual (Production, Staging, Archived)." />
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead>
                      <tr className="border-b border-surface-border text-xs text-white/40">
                        <th className="pb-2 pr-4">Versão</th>
                        <th className="pb-2 pr-4">Estágio</th>
                        <th className="pb-2 pr-4">RMSE</th>
                        <th className="pb-2 pr-4">MAE</th>
                        <th className="pb-2 pr-4">R²</th>
                        <th className="pb-2 pr-4">Acurácia Direcional</th>
                        <th className="pb-2">Criado</th>
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
                  <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-white">
                    Comparação de Métricas por Versão
                    <InfoTooltip text="Comparação visual de RMSE, MAE e R² entre versões do modelo. Menor RMSE/MAE e maior R² indicam melhor desempenho." />
                  </h3>
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
                  <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-white">
                    Histórico de Transições
                    <InfoTooltip text="Registro de todas as promoções e rebaixamentos de versão de modelo entre estágios no MLflow." />
                  </h3>
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
                    <div className="flex items-center gap-1.5">
                      <h4 className="font-semibold text-white">{fs.name}</h4>
                      <InfoTooltip text="Conjunto de features versionado e armazenado no Feature Store. Garante reprodutibilidade do treinamento." />
                    </div>
                    <p className="text-xs text-white/40">
                      v{fs.latest_version} · {fs.total_versions} versão(ões)
                    </p>
                  </div>
                  <Database className="h-5 w-5 text-nvidia" />
                </div>
                <div className="mt-3 flex items-center gap-4 text-xs text-white/50">
                  <span>Atualizado: {fs.last_updated?.slice(0, 10)}</span>
                </div>
              </div>
            ))}
          </div>

          {featureSets.length === 0 && (
            <div className="rounded-xl border border-surface-border bg-surface-card p-8 text-center text-white/40">
              Nenhum feature set registrado
            </div>
          )}

          {/* Feature lineage diagram placeholder */}
          <div className="rounded-xl border border-surface-border bg-surface-card p-5">
            <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-white">
              Pipeline de Features
              <InfoTooltip text="Fluxo de transformação dos dados brutos CSV até as features finais consumidas pelo modelo LSTM." />
            </h3>
            <div className="flex items-center justify-center gap-3 py-6">
              <div className="flex items-center gap-1.5 rounded-lg bg-blue-500/20 px-4 py-2 text-sm text-blue-400"><FileText className="h-3.5 w-3.5" /> CSV Dados Brutos</div>
              <span className="text-white/30">→</span>
              <div className="flex items-center gap-1.5 rounded-lg bg-yellow-500/20 px-4 py-2 text-sm text-yellow-400"><Wrench className="h-3.5 w-3.5" /> Transformação</div>
              <span className="text-white/30">→</span>
              <div className="flex items-center gap-1.5 rounded-lg bg-green-500/20 px-4 py-2 text-sm text-green-400"><BarChart3 className="h-3.5 w-3.5" /> Indicadores Técnicos</div>
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
            <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-white">
              Canary Deployments
              <InfoTooltip text="Histórico de deployments canary — o tráfego é roteado gradualmente para a nova versão antes da promoção total." />
            </h3>
            {deployments.length === 0 ? (
              <p className="text-center text-white/40">Nenhum deploy registrado</p>
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
                        <span>Peso Canary</span>
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
                <AlertTriangle className="h-4 w-4" /> Histórico de Rollbacks
                <InfoTooltip text="Rollbacks automáticos acionados quando a taxa de erros do canary ultrapassou o limite configurado (2%)." />
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
            <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-white">
              Fluxo de Canary Deploy
              <InfoTooltip text="Processo de promoção gradual: a nova versão começa com 5% do tráfego e é incrementada até 100% se os health checks passarem." />
            </h3>
            <div className="flex items-center justify-center gap-2 py-4 text-xs">
              <div className="flex items-center gap-1.5 rounded-lg bg-blue-500/20 px-3 py-2 text-blue-400"><Plus className="h-3.5 w-3.5" /> Nova Versão</div>
              <span className="text-white/20">→</span>
              <div className="flex items-center gap-1.5 rounded-lg bg-yellow-500/20 px-3 py-2 text-yellow-400"><Zap className="h-3.5 w-3.5" /> Canary 5%</div>
              <span className="text-white/20">→</span>
              <div className="flex items-center gap-1.5 rounded-lg bg-orange-500/20 px-3 py-2 text-orange-400"><ArrowUp className="h-3.5 w-3.5" /> Incremento</div>
              <span className="text-white/20">→</span>
              <div className="flex items-center gap-1.5 rounded-lg bg-nvidia/20 px-3 py-2 text-nvidia"><CheckCircle className="h-3.5 w-3.5" /> Health Check</div>
              <span className="text-white/20">→</span>
              <div className="flex items-center gap-1.5 rounded-lg bg-green-500/20 px-3 py-2 text-green-400"><Rocket className="h-3.5 w-3.5" /> Promover 100%</div>
            </div>
            <div className="mt-2 flex items-center justify-center gap-2 text-xs text-red-400">
              <span className="text-white/20">⤷</span>
              <div className="flex items-center gap-1.5 rounded-lg bg-red-500/20 px-3 py-2"><AlertTriangle className="h-3.5 w-3.5" /> Erro &gt; 2% → Rollback Automático</div>
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
              <StatCard title="Custo Total Estimado" value={`$${Number(data.grand_total).toFixed(2)}`} subtitle={`Período de ${data.period_days} dias`} icon={DollarSign} color="text-nvidia" info="Soma de todos os custos estimados no período: infraestrutura + LLM." />
              <StatCard title="Infraestrutura" value={`$${Number(data.infra_total).toFixed(2)}`} subtitle={`${Number(data.infra_pct).toFixed(0)}% do total`} icon={Server} color="text-blue-400" info="Custo estimado de GPU, armazenamento e rede para treino e inferência." />
              <StatCard title="Custos LLM" value={`$${Number(data.llm_total).toFixed(4)}`} subtitle={`${Number(data.llm_pct).toFixed(1)}% do total`} icon={MessageSquare} color="text-purple-400" info="Custo de tokens consumidos pela API do LLM para geração de análises e explicações." />
              <StatCard title="Total de Tokens" value={fmtTokens(Number(data.total_input_tokens) + Number(data.total_output_tokens))} subtitle={`${fmtTokens(Number(data.total_input_tokens))} in / ${fmtTokens(Number(data.total_output_tokens))} out`} icon={Cpu} color="text-amber-400" info="Total de tokens processados pelo LLM no período, separados por tokens de entrada e saída." />
              <StatCard title="Execuções de Treino" value={String(data.training_runs)} subtitle="sessões GPU" icon={Zap} color="text-green-400" info="Número de sessões de treinamento executadas na GPU no período analisado." />
            </div>

            {/* Current Model Badge */}
            <div className="flex items-center gap-3 rounded-xl border border-nvidia/20 bg-nvidia/5 p-4">
              <MessageSquare className="h-5 w-5 text-nvidia" />
              <div>
                <p className="text-sm font-semibold text-white">LLM Ativo: <span className="text-nvidia">{String(data.current_model)}</span></p>
                <p className="text-xs text-white/40">Provedor: {String(data.provider)} · ID do Modelo: <span className="font-mono text-white/50">{String(data.current_model_id)}</span></p>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              {/* Infra Breakdown */}
              <div className="rounded-xl border border-surface-border bg-surface-card p-5">
                <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-white">
                  <Server className="h-4 w-4 text-blue-400" /> Detalhamento de Infraestrutura
                  <InfoTooltip text="Breakdown detalhado dos custos de infraestrutura: GPU, armazenamento e transferência de dados." />
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
                    <span className="text-sm font-semibold text-white">Total Infraestrutura</span>
                    <span className="font-mono text-sm font-bold text-blue-400">${Number(data.infra_total).toFixed(2)}</span>
                  </div>
                </div>
              </div>

              {/* LLM Breakdown */}
              <div className="rounded-xl border border-surface-border bg-surface-card p-5">
                <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-white">
                  <MessageSquare className="h-4 w-4 text-purple-400" /> Detalhamento de Tokens LLM
                  <InfoTooltip text="Consumo de tokens por componente: RAG, análise de predições, geração de relatórios e outros usos do LLM." />
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
              <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-white">
                Tendência de Custo Diário ({String(data.period_days)} dias)
                <InfoTooltip text="Evolução diária dos custos separados por infraestrutura e LLM, permitindo identificar picos e tendências." />
              </h3>
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={dailyHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis dataKey="date" tick={{ fill: "#888", fontSize: 11 }} tickFormatter={(v) => v.slice(5)} />
                  <YAxis tick={{ fill: "#888", fontSize: 11 }} unit="$" />
                  <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid #333", borderRadius: 8 }} formatter={(value: number) => [`$${value.toFixed(2)}`, undefined]} />
                  <Legend />
                  <Area type="monotone" dataKey="infra" name="Infraestrutura" stroke="#0ea5e9" fill="#0ea5e9" fillOpacity={0.15} stackId="1" />
                  <Area type="monotone" dataKey="llm" name="LLM" stroke="#a855f7" fill="#a855f7" fillOpacity={0.15} stackId="1" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Model Cost Comparison */}
            <div className="rounded-xl border border-surface-border bg-surface-card p-5">
              <h3 className="mb-4 flex items-center gap-2 text-sm font-semibold text-white">
                Comparação de Custo LLM (mesma carga)
                <InfoTooltip text="Simulação do custo total se os mesmos tokens fossem processados por outros modelos LLM disponíveis no mercado." />
              </h3>
              <p className="mb-3 text-xs text-white/40">Quanto custaria com um modelo diferente para os mesmos {fmtTokens(Number(data.total_input_tokens) + Number(data.total_output_tokens))} tokens?</p>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead>
                    <tr className="border-b border-surface-border text-xs text-white/40">
                      <th className="pb-2 pr-4">Modelo</th>
                      <th className="pb-2 pr-4 text-right">Custo Input</th>
                      <th className="pb-2 pr-4 text-right">Custo Output</th>
                      <th className="pb-2 text-right">Total</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelComparison.map((m) => (
                      <tr key={m.model_id} className={`border-b border-surface-border/50 ${m.is_current ? "bg-nvidia/5" : ""}`}>
                        <td className="py-2.5 pr-4">
                          <span className="text-white/80">{m.model}</span>
                          {m.is_current && <span className="ml-2 rounded-full bg-nvidia/20 px-2 py-0.5 text-[10px] font-semibold text-nvidia">ATUAL</span>}
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

          </div>
        );
      })()}

      {/* ══════════════ HARDWARE SETUP TAB ══════════════ */}
      {activeTab === "hardware" && <HardwareSetupTab />}

      {/* Loading placeholder */}
      {loading && !slaReport && !bizSnapshot && registryModels.length === 0 && featureSets.length === 0 && deployments.length === 0 && !costData && activeTab !== "hardware" && (
        <div className="flex items-center justify-center py-20">
          <RefreshCw className="h-8 w-8 animate-spin text-nvidia" />
        </div>
      )}
    </div>
  );
}
