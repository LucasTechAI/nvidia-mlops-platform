"use client";

import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { RefreshCw, ClipboardList, Trophy, Swords, Play, BarChart2, Microscope, Bot } from "lucide-react";
import LoadingSpinner from "@/components/loading-spinner";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/page-header";
import { InfoTooltip } from "@/components/info-tooltip";

const TABS = [
  { id: "eval", label: "Métricas de Avaliação", icon: BarChart2 },
  { id: "explain", label: "Explicabilidade", icon: Microscope },
  { id: "llm", label: "Avaliação do LLM", icon: Bot },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function EvaluationPage() {
  const [activeTab, setActiveTab] = useState<TabId>("eval");

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <PageHeader
        label="Avaliação · Qualidade"
        title="Avaliação do"
        gradient="Modelo"
        subtitle="Métricas de avaliação, análise de explicabilidade e resultados de benchmark LLM."
        icon={ClipboardList}
      />

      <div className="flex justify-center">
        <div className="flex gap-1 rounded-xl border border-surface-border bg-surface-card p-1">
          {TABS.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-xs font-medium transition ${
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
      </div>

      <div>
        {activeTab === "eval" && <EvalMetricsTab />}
        {activeTab === "explain" && <ExplainabilityTab />}
        {activeTab === "llm" && <LLMEvalTab />}
      </div>
    </div>
  );
}

/* ──────────── Evaluation Metrics ──────────── */
function EvalMetricsTab() {
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadResults = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.evaluation.results();
      setData(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadResults(); }, []);

  const champion = data?.champion as Record<string, unknown> | undefined;
  const challenger = data?.challenger as Record<string, unknown> | undefined;

  // Build comparison data for chart
  const metrics = ["rmse", "mae", "mape", "r2"];
  const comparisonData = metrics
    .filter((m) => champion?.[m] !== undefined || challenger?.[m] !== undefined)
    .map((m) => ({
      metric: m.toUpperCase(),
      Champion: Number(champion?.[m] ?? 0),
      Challenger: Number(challenger?.[m] ?? 0),
    }));

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Comparação de Modelos</h3>
        <button
          onClick={loadResults}
          disabled={loading}
          className="flex items-center gap-2 rounded-lg bg-nvidia px-4 py-2 text-sm font-semibold text-black hover:bg-nvidia-dark disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Carregar Resultados
        </button>
      </div>

      {loading && <LoadingSpinner text="Carregando resultados da avaliação..." />}
      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-400">
          {error}
        </div>
      )}

      {data && !loading && (
        <>
          {/* Metric cards side by side */}
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            <div className="rounded-xl border border-nvidia/30 bg-surface-card p-6">
              <h4 className="mb-3 flex items-center gap-2 text-lg font-semibold text-nvidia">
                <Trophy className="h-5 w-5" /> Champion
                <InfoTooltip text="Modelo atualmente em produção com melhor desempenho histórico. Métricas calculadas no conjunto de teste (15% dos dados)." />
              </h4>
              {champion ? (
                <div className="space-y-2">
                  {Object.entries(champion).map(([k, v]) => (
                    <div key={k} className="flex justify-between text-sm">
                      <span className="text-white/50">{k}</span>
                      <span className="font-medium">
                        {typeof v === "number" ? v.toFixed(4) : String(v ?? "—")}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-white/40">Sem dados</p>
              )}
            </div>
            <div className="rounded-xl border border-sky-400/30 bg-surface-card p-6">
              <h4 className="mb-3 flex items-center gap-2 text-lg font-semibold text-sky-400">
                <Swords className="h-5 w-5" /> Challenger
                <InfoTooltip text="Modelo candidato treinado recentemente. Será promovido a Champion automaticamente se superar o RMSE do modelo atual." />
              </h4>
              {challenger ? (
                <div className="space-y-2">
                  {Object.entries(challenger).map(([k, v]) => (
                    <div key={k} className="flex justify-between text-sm">
                      <span className="text-white/50">{k}</span>
                      <span className="font-medium">
                        {typeof v === "number" ? v.toFixed(4) : String(v ?? "—")}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-white/40">Sem dados</p>
              )}
            </div>
          </div>

          {/* Comparison chart */}
          {comparisonData.length > 0 && (
            <div className="rounded-xl border border-surface-border bg-surface-card p-6">
              <h4 className="mb-4 flex items-center gap-2 text-sm font-semibold text-white/70">
                Comparação de Métricas
                <InfoTooltip text="Comparação visual de RMSE, MAE, MAPE e R² entre Champion e Challenger. Menor RMSE/MAE e maior R² indica melhor desempenho." />
              </h4>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={comparisonData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="metric" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      background: "#1a1c24",
                      border: "1px solid rgba(118,185,0,0.3)",
                      borderRadius: 8,
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="Champion" fill="#76B900" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Challenger" fill="#4ECDC4" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Promotion status */}
          {data.promoted !== undefined && (
            <div
              className={`rounded-lg border p-4 ${
                data.promoted
                  ? "border-green-500/30 bg-green-500/10"
                  : "border-amber-500/30 bg-amber-500/10"
              }`}
            >
              {data.promoted !== undefined && (
                <p className={`flex items-center gap-2 font-semibold ${data.promoted ? "text-green-400" : "text-amber-400"}`}>
                  {data.promoted ? "✅ Challenger promovido!" : "⚠️ Champion mantido"}
                  <InfoTooltip text="Resultado automático: o Challenger é promovido a Champion se seu RMSE for menor que o do Champion atual." />
                </p>
              )}
              {typeof data.promotion_reason === "string" && data.promotion_reason && (
                <p className="mt-1 text-sm text-white/50">{data.promotion_reason}</p>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

/* ──────────── Explainability ──────────── */
function ExplainabilityTab() {
  const [subTab, setSubTab] = useState<"perm" | "lime">("perm");

  return (
    <div className="space-y-4">
      {/* Sub-tab switcher */}
      <div className="flex gap-1 rounded-xl border border-surface-border bg-surface-card p-1 w-fit">
        <button
          onClick={() => setSubTab("perm")}
          className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
            subTab === "perm" ? "bg-nvidia text-black" : "text-white/50 hover:text-white"
          }`}
        >
          Permutation Importance
        </button>
        <button
          onClick={() => setSubTab("lime")}
          className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
            subTab === "lime" ? "bg-[#22C55E] text-black" : "text-white/50 hover:text-white"
          }`}
        >
          LIME
        </button>
      </div>

      {subTab === "perm" ? <PermutationSection /> : <LimeSection />}
    </div>
  );
}

/* ──────────── Permutation Importance ──────────── */
function PermutationSection() {
  const [features, setFeatures] = useState<{ feature: string; importance: number }[]>([]);
  const [loading, setLoading] = useState(false);
  const [computing, setComputing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadCached = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.evaluation.explainability();
      const feats = (res as { features?: { feature: string; importance: number }[] }).features || [];
      setFeatures(feats);
    } catch {
      setFeatures([]);
    } finally {
      setLoading(false);
    }
  };

  const recompute = async () => {
    setComputing(true);
    setError(null);
    try {
      const res = await api.evaluation.recomputeExplainability();
      const feats = (res as { features?: { feature: string; importance: number }[] }).features || [];
      setFeatures(feats);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed");
    } finally {
      setComputing(false);
    }
  };

  useEffect(() => { loadCached(); }, []);

  const chartData = features
    .sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance))
    .map((f) => ({ feature: f.feature, importance: Number(f.importance.toFixed(4)) }));

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Permutation Importance</h3>
          <p className="text-sm text-white/40">Importância de features baseada em análise de permutação</p>
        </div>
        <button
          onClick={recompute}
          disabled={computing}
          className="flex items-center gap-2 rounded-lg bg-nvidia px-4 py-2 text-sm font-semibold text-black hover:bg-nvidia-dark disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${computing ? "animate-spin" : ""}`} />
          {computing ? "Calculando..." : "Calcular Importância"}
        </button>
      </div>

      {(loading || computing) && <LoadingSpinner text={computing ? "Calculando importância das features (~30s)..." : "Carregando..."} />}
      {error && <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-400">{error}</div>}

      {chartData.length > 0 && !loading && (
        <>
          <div className="rounded-xl border border-surface-border bg-surface-card p-6">
            <div className="mb-3 flex items-center gap-2">
              <span className="text-sm font-semibold text-white/70">Importância por Permutação</span>
              <InfoTooltip text="Mede quanto o erro do modelo aumenta ao embaralhar cada feature aleatoriamente. Maior queda na performance = feature mais importante." />
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={120} />
                <Tooltip contentStyle={{ background: "#1a1c24", border: "1px solid rgba(118,185,0,0.3)", borderRadius: 8 }} />
                <Bar dataKey="importance" fill="#76B900" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="rounded-xl border border-surface-border bg-surface-card p-6">
            <h4 className="mb-3 flex items-center gap-2 text-sm font-semibold text-white/70">
              Detalhes das Features
              <InfoTooltip text="Ranking detalhado de importância das features por Permutation Importance. Features no topo têm maior impacto na precisão do modelo." />
            </h4>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-surface-border text-left text-xs text-white/40">
                  <th className="pb-2">Feature</th>
                  <th className="pb-2">Importância</th>
                </tr>
              </thead>
              <tbody>
                {chartData.map((f) => (
                  <tr key={f.feature} className="border-b border-surface-border/50">
                    <td className="py-2 text-white/70">{f.feature}</td>
                    <td className="py-2 font-medium text-nvidia">{f.importance}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

/* ──────────── LIME Section ──────────── */
function LimeSection() {
  const [features, setFeatures] = useState<{ feature: string; importance: number; std: number }[]>([]);
  const [meta, setMeta] = useState<{ n_explained: number; global_ranking: string[] } | null>(null);
  const [loading, setLoading] = useState(false);
  const [computing, setComputing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadCached = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.evaluation.lime();
      const feats = (res as { features?: { feature: string; importance: number; std: number }[] }).features || [];
      setFeatures(feats);
      setMeta({
        n_explained: (res as { n_explained?: number }).n_explained ?? 0,
        global_ranking: (res as { global_ranking?: string[] }).global_ranking ?? [],
      });
    } catch {
      setFeatures([]);
    } finally {
      setLoading(false);
    }
  };

  const recompute = async () => {
    setComputing(true);
    setError(null);
    try {
      const res = await api.evaluation.recomputeLime();
      const feats = (res as { features?: { feature: string; importance: number; std: number }[] }).features || [];
      setFeatures(feats);
      setMeta({
        n_explained: (res as { n_explained?: number }).n_explained ?? 0,
        global_ranking: (res as { global_ranking?: string[] }).global_ranking ?? [],
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed");
    } finally {
      setComputing(false);
    }
  };

  useEffect(() => { loadCached(); }, []);

  const chartData = features
    .sort((a, b) => b.importance - a.importance)
    .map((f) => ({
      feature: f.feature,
      importance: Number(f.importance.toFixed(4)),
      std: Number(f.std.toFixed(4)),
    }));

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">LIME — Local Explanations</h3>
          <p className="text-sm text-white/40">
            Peso médio |weight| em {meta?.n_explained ?? "N"} amostras explicadas (Ribeiro et al. 2016)
          </p>
        </div>
        <button
          onClick={recompute}
          disabled={computing}
          className="flex items-center gap-2 rounded-lg bg-[#22C55E] px-4 py-2 text-sm font-semibold text-black hover:bg-[#16A34A] disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${computing ? "animate-spin" : ""}`} />
          {computing ? "Calculando..." : "Calcular LIME"}
        </button>
      </div>

      <div className="rounded-lg border border-[#22C55E]/20 bg-[#22C55E]/5 p-3 text-xs text-white/50">
        <span className="font-semibold text-[#22C55E]">Como funciona:</span> O LIME aproxima a LSTM localmente com um modelo linear ao redor de cada previsão. Cada feature recebe um peso com sinal; o gráfico exibe o peso absoluto médio entre as amostras — indicando relevância global sob perspectiva local.
      </div>

      {(loading || computing) && <LoadingSpinner text={computing ? "Calculando explicações LIME (~60s)..." : "Carregando..."} />}
      {error && <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-400">{error}</div>}

      {chartData.length > 0 && !loading && (
        <>
          <div className="rounded-xl border border-surface-border bg-surface-card p-6">
            <div className="mb-3 flex items-center gap-2">
              <span className="text-sm font-semibold text-white/70">Importância por LIME</span>
              <InfoTooltip text="Peso absoluto médio de cada feature em múltiplas amostras. O LIME aproxima o LSTM localmente com um modelo linear — features com maior peso médio têm maior influência local." />
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={120} />
                <Tooltip contentStyle={{ background: "#1a1c24", border: "1px solid rgba(34,197,94,0.3)", borderRadius: 8 }} />
                <Bar dataKey="importance" fill="#22C55E" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="rounded-xl border border-surface-border bg-surface-card p-6">
            <h4 className="mb-3 flex items-center gap-2 text-sm font-semibold text-white/70">
              Detalhes das Features
              <InfoTooltip text="Importância média e desvio padrão de cada feature calculados pelo LIME em múltiplas amostras. Desvio alto indica comportamento instável para aquela feature." />
            </h4>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-surface-border text-left text-xs text-white/40">
                  <th className="pb-2">Feature</th>
                  <th className="pb-2">Peso Médio |Peso|</th>
                  <th className="pb-2">Desvio Padrão</th>
                </tr>
              </thead>
              <tbody>
                {chartData.map((f) => (
                  <tr key={f.feature} className="border-b border-surface-border/50">
                    <td className="py-2 text-white/70">{f.feature}</td>
                    <td className="py-2 font-medium text-[#22C55E]">{f.importance}</td>
                    <td className="py-2 text-white/40">±{f.std}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {meta?.global_ranking && meta.global_ranking.length > 0 && (
              <p className="mt-3 text-xs text-white/30">Ranking: {meta.global_ranking.join(" › ")}</p>
            )}
          </div>
        </>
      )}
    </div>
  );
}


/* ──────────── Score bar sub-component ──────────── */
function ScoreBar({ label, value, max }: { label: string; value: number; max: number }) {
  const pct = Math.min(100, (value / max) * 100);
  const ratio = value / max;
  const color = ratio >= 0.8 ? "#22C55E" : ratio >= 0.5 ? "#F59E0B" : "#EF4444";
  const display = max === 1 ? value.toFixed(3) : `${value.toFixed(2)} / ${max}`;
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="capitalize text-white/60">{label.replace(/_/g, " ")}</span>
        <span className="font-semibold" style={{ color }}>{display}</span>
      </div>
      <div className="h-2 rounded-full bg-white/10">
        <div className="h-2 rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

/* ──────────── LLM Evaluation ──────────── */
type GoldenItem = { id: number; query: string; expected_answer: string; contexts: string[] };
type RagasResult = { metrics: Record<string, number>; n_samples: number; note?: string };
type JudgeResult = { avg_scores: Record<string, number>; overall_avg: number; n_samples: number };

function LLMEvalTab() {
  const [goldenSet, setGoldenSet] = useState<GoldenItem[]>([]);
  const [ragas, setRagas] = useState<RagasResult | null>(null);
  const [judge, setJudge] = useState<JudgeResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.evaluation.llmResults();
      const r = res as { golden_set?: GoldenItem[]; ragas?: RagasResult; llm_judge?: JudgeResult };
      setGoldenSet(r.golden_set || []);
      setRagas(r.ragas || null);
      setJudge(r.llm_judge || null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed");
    } finally {
      setLoading(false);
    }
  };

  const runEval = async () => {
    setRunning(true);
    setError(null);
    try {
      await api.evaluation.runEvaluation();
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha na avaliação");
    } finally {
      setRunning(false);
    }
  };

  useEffect(() => { load(); }, []);

  const hasResults = ragas || judge;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Avaliação do LLM (RAGAS)</h3>
          <p className="text-sm text-white/40">Avaliação do golden set e resultados de benchmark</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={load}
            disabled={loading || running}
            className="flex items-center gap-2 rounded-lg border border-surface-border px-4 py-2 text-sm font-semibold text-white/70 hover:bg-surface-hover disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            Carregar Resultados
          </button>
          <button
            onClick={runEval}
            disabled={running || loading}
            className="flex items-center gap-2 rounded-lg bg-nvidia px-4 py-2 text-sm font-semibold text-black hover:bg-nvidia-dark disabled:opacity-50"
          >
            <Play className={`h-4 w-4 ${running ? "animate-pulse" : ""}`} />
            {running ? "Executando..." : "Executar Avaliação"}
          </button>
        </div>
      </div>

      {(loading || running) && (
        <LoadingSpinner text={running ? "Executando RAGAS + LLM-Judge no golden set (~60s)..." : "Carregando..."} />
      )}
      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-400">{error}</div>
      )}

      {/* Empty state */}
      {!hasResults && !loading && !running && (
        <div className="rounded-xl border border-surface-border bg-surface-card p-10 text-center">
          <p className="text-white/40">Nenhum resultado de avaliação ainda.</p>
          <p className="mt-1 text-xs text-white/30">
            Clique em <span className="font-semibold text-nvidia">Executar Avaliação</span> para calcular os scores RAGAS + LLM-Judge no golden set de {goldenSet.length || 25} amostras.
          </p>
        </div>
      )}

      {/* RAGAS metrics */}
      {ragas && !loading && (
        <div className="rounded-xl border border-surface-border bg-surface-card p-6">
          <div className="mb-4 flex items-center gap-3">
            <h4 className="text-sm font-semibold text-white/70">Métricas RAGAS</h4>
            <span className="text-xs text-white/30">{ragas.n_samples} amostras · escala 0 → 1</span>
            <InfoTooltip text="4 métricas automáticas do pipeline RAG: faithfulness (fidelidade ao contexto), answer relevancy (relevância da resposta), context precision e context recall. Escala 0–1." />
          </div>
          {ragas.note && (
            <div className="mb-3 rounded-lg border border-amber-500/20 bg-amber-500/10 px-3 py-2 text-xs text-amber-400">
              <p>⚠️ {ragas.note}</p>
              {ragas.note.toLowerCase().includes("openai") && (
                <p className="mt-1 text-amber-300/60">
                  Defina <code className="rounded bg-black/30 px-1">OPENAI_API_KEY</code> no seu ambiente e reinicie a API para habilitar o scoring RAGAS baseado em LLM.
                </p>
              )}
            </div>
          )}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {Object.entries(ragas.metrics).map(([k, v]) => (
              <ScoreBar key={k} label={k} value={v} max={1} />
            ))}
          </div>
        </div>
      )}

      {/* LLM-Judge */}
      {judge && !loading && (
        <div className="rounded-xl border border-surface-border bg-surface-card p-6">
          <div className="mb-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <h4 className="text-sm font-semibold text-white/70">LLM-as-Judge</h4>
              <span className="text-xs text-white/30">{judge.n_samples} amostras · escala 1 → 5</span>
              <InfoTooltip text="Um LLM avalia as respostas do agente em 3 critérios: Relevância da Resposta, Acurácia Factual e Utilidade para Investimento. Escala 1 (ruim) a 5 (excelente)." />
            </div>
            <span className="rounded-full border border-nvidia/30 bg-nvidia/10 px-3 py-0.5 text-xs font-semibold text-nvidia">
              Geral: {judge.overall_avg} / 5.0
            </span>
          </div>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            {Object.entries(judge.avg_scores).map(([k, v]) => (
              <ScoreBar key={k} label={k} value={v} max={5} />
            ))}
          </div>
        </div>
      )}

      {/* Golden Set */}
      {goldenSet.length > 0 && !loading && (
        <div className="rounded-xl border border-surface-border bg-surface-card p-6">
          <h4 className="mb-3 flex items-center gap-2 text-sm font-semibold text-white/70">
            Conjunto de Avaliação — Golden Set ({goldenSet.length} amostras)
            <InfoTooltip text="25 pares pergunta-resposta criados manualmente, usados como gabarito para avaliar automaticamente a qualidade das respostas do agente RAG." />
          </h4>
          <div className="space-y-2 pr-1">
            {goldenSet.map((item) => (
              <div key={item.id} className="rounded-lg bg-surface-hover p-3">
                <p className="mb-1 text-xs font-semibold text-white/30">Amostra #{item.id}</p>
                <p className="text-[11px] font-semibold uppercase tracking-wide text-white/30">Pergunta</p>
                <p className="text-sm font-medium text-white/80">{item.query}</p>
                <p className="mt-2 text-[11px] font-semibold uppercase tracking-wide text-white/30">Resposta Esperada</p>
                <p className="mt-0.5 line-clamp-2 text-xs text-white/40">{item.expected_answer}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
