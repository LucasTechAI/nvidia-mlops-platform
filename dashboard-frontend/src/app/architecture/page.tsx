"use client";

import { useState } from "react";
import { Info, GitBranch } from "lucide-react";
import { PageHeader } from "@/components/page-header";

/* ─── tiny tooltip helper ──────────────────────────────────────────── */
function Tip({ text }: { text: string }) {
  const [show, setShow] = useState(false);
  return (
    <div
      className="relative inline-block"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <Info className="h-3 w-3 cursor-help text-white/20 transition-colors hover:text-white/50" />
      {show && (
        <div className="absolute bottom-full left-1/2 z-50 mb-2 w-56 -translate-x-1/2 rounded-lg border border-surface-border bg-[#1a1c24] px-3 py-2 text-[11px] font-normal text-white/70 shadow-xl">
          {text}
          <div className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-[#1a1c24]" />
        </div>
      )}
    </div>
  );
}

/* ─── box node (draw.io style) ─────────────────────────────────────── */
interface BoxProps {
  title: string;
  subtitle?: string;
  icon: string;
  color: string;         // border / accent
  bg?: string;           // background tint
  tech?: string[];
  tip?: string;
  className?: string;
}

function Box({ title, subtitle, icon, color, bg, tech, tip, className = "" }: BoxProps) {
  return (
    <div
      className={`relative rounded-xl border-2 px-4 py-3 shadow-lg transition-transform hover:scale-[1.03] ${className}`}
      style={{ borderColor: color, background: bg ?? "rgba(26,28,36,0.95)" }}
    >
      <div className="flex items-center gap-2">
        <span className="text-lg">{icon}</span>
        <span className="text-sm font-bold" style={{ color }}>{title}</span>
        {tip && <Tip text={tip} />}
      </div>
      {subtitle && <p className="mt-0.5 text-[11px] text-white/50">{subtitle}</p>}
      {tech && tech.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {tech.map((t) => (
            <span
              key={t}
              className="rounded-md px-1.5 py-0.5 text-[10px] font-medium"
              style={{ background: `${color}22`, color }}
            >
              {t}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

/* ─── arrow components (draw.io connectors) ────────────────────────── */
function ArrowDown({ label, color = "#76B900" }: { label?: string; color?: string }) {
  return (
    <div className="flex flex-col items-center py-1">
      <div className="h-6 w-0.5" style={{ background: color }} />
      {label && (
        <span className="rounded bg-[#1a1c24] px-2 py-0.5 text-[10px] font-medium" style={{ color }}>
          {label}
        </span>
      )}
      <svg width="12" height="8" className="-mt-0.5">
        <polygon points="0,0 12,0 6,8" fill={color} />
      </svg>
    </div>
  );
}

function ArrowRight({ label, color = "#76B900" }: { label?: string; color?: string }) {
  return (
    <div className="flex items-center gap-0">
      <div className="h-0.5 w-8" style={{ background: color }} />
      {label && (
        <span className="rounded bg-[#1a1c24] px-1.5 py-0.5 text-[10px] font-medium" style={{ color }}>
          {label}
        </span>
      )}
      <svg width="8" height="12">
        <polygon points="0,0 8,6 0,12" fill={color} />
      </svg>
    </div>
  );
}

/* ─── section divider with lane label ──────────────────────────────── */
function Lane({ label, color }: { label: string; color: string }) {
  return (
    <div className="flex items-center gap-3 py-2">
      <span
        className="rounded-md px-2.5 py-1 text-[10px] font-bold uppercase tracking-widest"
        style={{ background: `${color}15`, color, border: `1px solid ${color}30` }}
      >
        {label}
      </span>
      <div className="h-px flex-1" style={{ background: `${color}25` }} />
    </div>
  );
}

/* ─── sub-diagram canvas wrapper ───────────────────────────────── */
function SubCanvas({ title, subtitle, children }: { title: string; subtitle: string; children: React.ReactNode }) {
  return (
    <div className="overflow-x-auto rounded-2xl border border-surface-border bg-[#12131a] p-6">
      <div className="mb-1 flex items-center gap-3">
        <h3 className="text-base font-semibold">{title}</h3>
        <div className="h-px flex-1 bg-white/5" />
      </div>
      <p className="mb-5 text-xs text-white/40">{subtitle}</p>
      <div className="relative">
        <div
          className="pointer-events-none absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage: "radial-gradient(circle, #fff 1px, transparent 1px)",
            backgroundSize: "24px 24px",
          }}
        />
        <div className="relative space-y-2">{children}</div>
      </div>
    </div>
  );
}

/* ─── checklist row ────────────────────────────────────────────── */
function Check({ label, detail, done = true }: { label: string; detail: string; done?: boolean }) {
  return (
    <div className="flex items-start gap-2 rounded-lg border border-surface-border bg-surface-hover px-3 py-2">
      <span className="mt-0.5 text-sm">{done ? "✅" : "⬜"}</span>
      <div>
        <p className="text-xs font-semibold text-white/80">{label}</p>
        <p className="text-[10px] text-white/40">{detail}</p>
      </div>
    </div>
  );
}

/* ─── tab definitions ──────────────────────────────────────────── */
const TABS = [
  { id: "general",       label: "General" },
  { id: "webapp",        label: "Web Application" },
  { id: "agent",         label: "LLM Agent & RAG" },
  { id: "evaluation",    label: "Evaluation" },
  { id: "monitoring",    label: "Monitoring & Drift" },
  { id: "security",      label: "Security & Gov." },
  { id: "champion",      label: "Champion-Challenger" },
  { id: "testing",       label: "Testing & CI/CD" },
  { id: "explainability", label: "Explainability" },
  { id: "checklist",     label: "Requirements" },
] as const;

type TabId = (typeof TABS)[number]["id"];

/* ━━━━━━━━━━━━━━━━ PAGE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
export default function ArchitecturePage() {
  const [activeTab, setActiveTab] = useState<TabId>("general");

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <PageHeader
        label="Arquitetura · Plataforma"
        title="Arquitetura do"
        gradient="Projeto"
        subtitle="Visão ponta a ponta da plataforma MLOps — dos dados brutos à predição final."
        icon={GitBranch}
      />

      {/* Tab Bar — 2 rows of 5 */}
      <div className="sticky top-0 z-30 -mx-1 rounded-xl border border-surface-border bg-[#12131a]/95 px-1 py-1.5 backdrop-blur-md">
        {[TABS.slice(0, 5), TABS.slice(5)].map((row, rowIdx) => (
          <div key={rowIdx} className="flex gap-1">
            {row.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 rounded-lg px-3 py-2 text-xs font-medium transition-all ${
                  activeTab === tab.id
                    ? "bg-nvidia/20 text-nvidia shadow-sm shadow-nvidia/10"
                    : "text-white/40 hover:bg-white/5 hover:text-white/70"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        ))}
      </div>

      {/* ════════════ GENERAL TAB ════════════ */}
      {activeTab === "general" && (<>
      {/* Canvas (draw.io style) */}
      <div className="relative overflow-x-auto rounded-2xl border border-surface-border bg-[#12131a] p-8">
        {/* Grid dots background */}
        <div
          className="pointer-events-none absolute inset-0 opacity-[0.04]"
          style={{
            backgroundImage: "radial-gradient(circle, #fff 1px, transparent 1px)",
            backgroundSize: "24px 24px",
          }}
        />

        <div className="relative space-y-2">
          {/* ── LANE 1: Data Sources ─────────────────────────────── */}
          <Lane label="Data Sources" color="#4ECDC4" />
          <div className="flex items-center justify-center gap-4">
            <Box
              icon="📡"
              title="Yahoo Finance API"
              subtitle="NVDA real-time market data"
              color="#4ECDC4"
              tech={["yfinance", "REST API"]}
              tip="API do Yahoo Finance usada para coletar dados OHLCV diários das ações da NVIDIA desde 2017."
            />
            <ArrowRight label="OHLCV" color="#4ECDC4" />
            <Box
              icon="⚙️"
              title="ETL Pipeline"
              subtitle="Extraction, validation & loading"
              color="#4ECDC4"
              tech={["Python", "pandas", "yfinance"]}
              tip="Pipeline de ETL que extrai dados do Yahoo Finance, valida qualidade e carrega no banco de dados SQLite."
            />
            <ArrowRight label="Insert" color="#4ECDC4" />
            <Box
              icon="🗄️"
              title="SQLite Database"
              subtitle="6,846 historical records"
              color="#4ECDC4"
              tech={["SQLite", "6846 rows"]}
              tip="Banco de dados local armazenando todo o histórico de preços e volumes da NVDA (até 2026-04-10)."
            />
          </div>

          {/* Arrow down to processing */}
          <div className="flex justify-center">
            <ArrowDown label="Raw Data" color="#45B7D1" />
          </div>

          {/* ── LANE 2: Data Processing ──────────────────────────── */}
          <Lane label="Data Processing" color="#45B7D1" />
          <div className="flex items-center justify-center gap-4">
            <Box
              icon="🔄"
              title="Preprocessing"
              subtitle="MinMaxScaler normalization"
              color="#45B7D1"
              tech={["scikit-learn", "MinMaxScaler"]}
              tip="Normaliza os dados para o intervalo [0,1] usando MinMaxScaler, essencial para convergência da LSTM."
            />
            <ArrowRight label="Normalized" color="#45B7D1" />
            <Box
              icon="📊"
              title="Sequence Generator"
              subtitle="60-day sliding windows"
              color="#45B7D1"
              tech={["numpy", "sliding window"]}
              tip="Cria sequências de 60 timesteps (3 meses) como entrada para a LSTM. Cada sequência tem 5 features (OHLCV)."
            />
            <ArrowRight label="Tensors" color="#45B7D1" />
            <Box
              icon="📦"
              title="DataLoader"
              subtitle="Training batches"
              color="#45B7D1"
              tech={["PyTorch", "batch=32"]}
              tip="Organiza os dados em batches de 32, com shuffle para treinamento e sem shuffle para validação/teste."
            />
          </div>

          {/* Arrow down to training */}
          <div className="flex justify-center">
            <ArrowDown label="Train / Val / Test Split" color="#76B900" />
          </div>

          {/* ── LANE 3: Training & Optimization ──────────────────── */}
          <Lane label="Training & Optimization" color="#76B900" />
          <div className="flex items-center justify-center gap-6">
            <div className="flex flex-col items-center gap-2">
              <Box
                icon="🧠"
                title="LSTM Model"
                subtitle="2 layers × 128 hidden units"
                color="#76B900"
                bg="rgba(118,185,0,0.05)"
                tech={["PyTorch", "LSTM", "2 layers"]}
                tip="Rede LSTM com 2 camadas empilhadas, 128 unidades ocultas, dropout 0.2 e camada densa de saída."
                className="min-w-[200px]"
              />
              <div className="flex gap-8">
                <div className="flex flex-col items-center">
                  <div className="h-4 w-0.5 bg-nvidia/40" />
                  <svg width="12" height="8"><polygon points="0,0 12,0 6,8" fill="#76B900" /></svg>
                </div>
              </div>
              <Box
                icon="📉"
                title="Training Loop"
                subtitle="Adam + Early Stopping + Grad Clip"
                color="#76B900"
                tech={["Adam", "MSELoss", "early stop"]}
                tip="Loop de treinamento com otimizador Adam, MSE Loss, gradient clipping (max=1.0) e early stopping (patience=10)."
              />
            </div>

            <div className="flex flex-col items-center gap-3">
              <div className="h-0.5 w-10 bg-amber-400/40" />
              <span className="rounded bg-[#1a1c24] px-2 py-0.5 text-[10px] font-medium text-amber-400">HPO</span>
              <div className="h-0.5 w-10 bg-amber-400/40" />
            </div>

            <Box
              icon="🔬"
              title="Optuna HPO"
              subtitle="Bayesian search (50+ trials)"
              color="#EF5B5B"
              tech={["Optuna", "Bayesian", "50 trials"]}
              tip="Hyperparameter Optimization com Optuna usando busca Bayesiana. Otimiza hidden_size, num_layers, learning_rate e dropout."
            />
          </div>

          {/* Arrow down to tracking */}
          <div className="flex justify-center">
            <ArrowDown label="Best Model + Metrics" color="#0194E2" />
          </div>

          {/* ── LANE 4: Experiment Tracking ───────────────────────── */}
          <Lane label="Experiment Tracking & Versioning" color="#0194E2" />
          <div className="flex items-center justify-center gap-4">
            <Box
              icon="📋"
              title="MLflow Tracking"
              subtitle="Metrics, parameters & artifacts"
              color="#0194E2"
              tech={["MLflow", "port 5000"]}
              tip="Registra todas as métricas (loss, RMSE, MAE, R²), parâmetros e artefatos de cada experimento."
            />
            <ArrowRight label="Log" color="#0194E2" />
            <Box
              icon="💾"
              title="Model Registry"
              subtitle="Versioned .pth checkpoint"
              color="#0194E2"
              tech={["PyTorch", ".pth", "DVC"]}
              tip="Modelo salvo como checkpoint PyTorch com state_dict, config, training_info e métricas. Versionado com DVC."
            />
            <ArrowRight label="Load" color="#0194E2" />
            <Box
              icon="🔐"
              title="DVC + Git"
              subtitle="Data & model versioning"
              color="#945DD6"
              tech={["DVC", "Git", "GitHub"]}
              tip="DVC versiona datasets e modelos grandes. Git versiona código. Reprodutibilidade total do pipeline."
            />
          </div>

          {/* Arrow down to serving */}
          <div className="flex justify-center">
            <ArrowDown label="best_model.pth" color="#FF6B35" />
          </div>

          {/* ── LANE 5: Serving & API ────────────────────────────── */}
          <Lane label="Serving & API" color="#FF6B35" />
          <div className="flex items-center justify-center gap-4">
            <Box
              icon="🚀"
              title="FastAPI"
              subtitle="REST API — port 8000"
              color="#FF6B35"
              bg="rgba(255,107,53,0.04)"
              tech={["FastAPI", "Uvicorn", "port 8000"]}
              tip="API REST com FastAPI servindo endpoints para predição, info do modelo, métricas e HPO results."
              className="min-w-[180px]"
            />
            <ArrowRight label="JSON" color="#FF6B35" />
            <div className="space-y-2">
              <div className="flex gap-3">
                <div className="rounded-lg border border-white/10 bg-[#1a1c24] px-3 py-1.5 text-[10px] text-white/50">
                  <span className="font-mono text-[#FF6B35]">GET</span> /model/info
                </div>
                <div className="rounded-lg border border-white/10 bg-[#1a1c24] px-3 py-1.5 text-[10px] text-white/50">
                  <span className="font-mono text-[#FF6B35]">GET</span> /predict
                </div>
              </div>
              <div className="flex gap-3">
                <div className="rounded-lg border border-white/10 bg-[#1a1c24] px-3 py-1.5 text-[10px] text-white/50">
                  <span className="font-mono text-[#FF6B35]">GET</span> /model/training-history
                </div>
                <div className="rounded-lg border border-white/10 bg-[#1a1c24] px-3 py-1.5 text-[10px] text-white/50">
                  <span className="font-mono text-[#FF6B35]">GET</span> /model/hpo-results
                </div>
              </div>
            </div>
          </div>

          {/* Arrow down to frontend */}
          <div className="flex justify-center">
            <ArrowDown label="REST Calls" color="#ffffff" />
          </div>

          {/* ── LANE 6: Frontend & Monitoring ────────────────────── */}
          <Lane label="Frontend & Monitoring" color="#ffffff" />
          <div className="flex items-center justify-center gap-4">
            <Box
              icon="🖥️"
              title="Next.js Dashboard"
              subtitle="Interactive UI — port 3001"
              color="#ffffff"
              bg="rgba(255,255,255,0.03)"
              tech={["Next.js 14", "React", "Recharts", "Tailwind"]}
              tip="Dashboard interativo com 8 páginas: Home, Predictions, Architecture, Metrics, Observability, Evaluation, Agent e esta Architecture."
            />
            <ArrowRight label="Visualize" color="#ffffff" />
            <Box
              icon="📊"
              title="Dashboard Pages"
              subtitle="Full pipeline visualization"
              color="#ffffff"
              tech={["Predictions", "MLOps", "Metrics", "Evaluation", "Logs"]}
              tip="12 páginas: Home, Predictions, Metrics, Model Schema, Observability, Evaluation, Agent, Architecture, MLOps, Logs, Next Steps e Landing."
            />
          </div>

          {/* Arrow down to infra */}
          <div className="flex justify-center">
            <ArrowDown label="Deploy" color="#2496ED" />
          </div>

          {/* ── LANE 7: Infrastructure ───────────────────────────── */}
          <Lane label="Infrastructure & CI/CD" color="#2496ED" />
          <div className="flex items-center justify-center gap-4">
            <Box
              icon="🐳"
              title="Docker Compose"
              subtitle="Multi-service orchestration"
              color="#2496ED"
              tech={["Docker", "Compose", "multi-stage"]}
              tip="Docker Compose orquestra todos os serviços: MLflow, Training, HPO, Prediction, ETL, API e Dashboard."
            />
            <ArrowRight label="CI/CD" color="#2496ED" />
            <Box
              icon="⚡"
              title="GitHub Actions"
              subtitle="Automated CI/CD pipeline"
              color="#2496ED"
              tech={["CI/CD", "pytest", "lint", "build"]}
              tip="GitHub Actions executa testes automatizados, linting, build e deploy a cada push."
            />
            <ArrowRight label="Security" color="#2496ED" />
            <Box
              icon="🛡️"
              title="Security & LGPD"
              subtitle="OWASP + compliance"
              color="#2496ED"
              tech={["OWASP", "LGPD", "Red Team"]}
              tip="Boas práticas de segurança baseadas em OWASP, conformidade com LGPD e relatório de Red Team para IA."
            />
          </div>
        </div>

        {/* Watermark */}
        <div className="mt-6 flex items-center justify-between border-t border-white/5 pt-4">
          <span className="text-[10px] text-white/15">TradeOps Platform — E2E Architecture Diagram</span>
          <div className="flex items-center gap-2">
            <span className="rounded border border-white/10 px-2 py-0.5 text-[9px] text-white/20">draw.io style</span>
            <span className="text-[10px] text-white/15">v1.0</span>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="rounded-xl border border-surface-border bg-surface-card p-5">
        <h3 className="mb-3 text-sm font-semibold text-white/60">Color Legend</h3>
        <div className="flex flex-wrap gap-4">
          {[
            { color: "#4ECDC4", label: "Data Sources & ETL" },
            { color: "#45B7D1", label: "Data Processing" },
            { color: "#76B900", label: "Training & Model" },
            { color: "#EF5B5B", label: "HPO (Optuna)" },
            { color: "#0194E2", label: "Tracking & Versioning" },
            { color: "#FF6B35", label: "Serving & API" },
            { color: "#ffffff", label: "Frontend & Dashboard" },
            { color: "#2496ED", label: "Infrastructure & CI/CD" },
            { color: "#945DD6", label: "Data Version Control" },
          ].map((item) => (
            <div key={item.label} className="flex items-center gap-2">
              <div className="h-3 w-3 rounded-sm" style={{ background: item.color }} />
              <span className="text-xs text-white/50">{item.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Tech Summary */}
      <div className="rounded-xl border border-surface-border bg-surface-card p-5">
        <h3 className="mb-3 text-sm font-semibold text-white/60">Full Tech Stack</h3>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            { category: "Deep Learning", items: "PyTorch, LSTM, NumPy" },
            { category: "MLOps", items: "MLflow Tracing, Optuna, DVC" },
            { category: "Backend", items: "FastAPI, Uvicorn, SQLite" },
            { category: "Frontend", items: "Next.js 14, React, Recharts" },
            { category: "Data", items: "pandas, yfinance, scikit-learn" },
            { category: "Infra", items: "Docker, Compose, GitHub Actions" },
            { category: "Security", items: "OWASP, LGPD, Red Team" },
            { category: "Quality", items: "pytest, ruff, mypy, coverage" },
          ].map((s) => (
            <div key={s.category} className="rounded-lg border border-surface-border bg-surface-hover p-3">
              <p className="text-[10px] font-bold uppercase tracking-wider text-nvidia">{s.category}</p>
              <p className="mt-1 text-xs text-white/50">{s.items}</p>
            </div>
          ))}
        </div>
      </div>
      </>)}

      {/* ════════════ WEB APP TAB ════════════ */}
      {activeTab === "webapp" && (<>
      {/* ── Frontend Sub-Canvas ── */}
      <SubCanvas
        title="🖥️ Frontend — Next.js Dashboard"
        subtitle="12-page interactive dashboard built with Next.js 14 App Router, React 18, Recharts and Tailwind CSS — port 3001."
      >
        <Lane label="App Router — Pages" color="#ffffff" />
        <div className="flex flex-wrap items-center justify-center gap-2">
          {[
            { page: "/", name: "Landing", icon: "🚀" },
            { page: "/home", name: "Home", icon: "🏠" },
            { page: "/predictions", name: "Predictions", icon: "🔮" },
            { page: "/metrics", name: "Metrics", icon: "📊" },
            { page: "/model-schema", name: "Model Schema", icon: "🧬" },
            { page: "/observability", name: "Observability", icon: "📡" },
            { page: "/evaluation", name: "Evaluation", icon: "🔬" },
            { page: "/agent", name: "Agent Chat", icon: "🤖" },
            { page: "/architecture", name: "Architecture", icon: "🏗️" },
            { page: "/mlops", name: "MLOps", icon: "⚙️" },
            { page: "/logs", name: "Logs", icon: "📜" },
            { page: "/next-steps", name: "Next Steps", icon: "🚩" },
          ].map((p) => (
            <div key={p.page} className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2">
              <span className="text-sm">{p.icon}</span>
              <div>
                <p className="text-[10px] font-bold text-white/80">{p.name}</p>
                <p className="font-mono text-[9px] text-white/30">{p.page}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="flex justify-center"><ArrowDown label="Renders with" color="#0194E2" /></div>

        <Lane label="Shared Components" color="#0194E2" />
        <div className="flex items-center justify-center gap-4">
          <Box icon="📁" title="Sidebar" subtitle="Navigation layout" color="#0194E2" tech={["sidebar.tsx"]} tip="Persistent navigation sidebar with links to all 9 pages, highlighting the active route." />
          <Box icon="📈" title="Stat Card" subtitle="KPI display" color="#0194E2" tech={["stat-card.tsx"]} tip="Reusable card component for displaying key performance indicators with trend arrows." />
          <Box icon="📑" title="Tab Group" subtitle="Sub-tab navigation" color="#0194E2" tech={["tab-group.tsx"]} tip="Reusable tabbed interface component used across pages (e.g. Architecture 10 tabs)." />
          <Box icon="⏳" title="Loading Spinner" subtitle="Async loading state" color="#0194E2" tech={["loading-spinner.tsx"]} tip="Loading indicator shown during data fetching from the backend API." />
        </div>

        <div className="flex justify-center"><ArrowDown label="Data Layer" color="#F59E0B" /></div>

        <Lane label="API Client & Types" color="#F59E0B" />
        <div className="flex items-center justify-center gap-4">
          <Box icon="🔌" title="API Client" subtitle="lib/api.ts" color="#F59E0B" bg="rgba(245,158,11,0.05)" tech={["fetch", "typed responses"]} tip="Centralized API client with typed methods: getHealth(), getData(), getPredict(), getModelInfo(), getMonitoring(), getEvaluation(), askAgent(), etc." className="min-w-[200px]" />
          <ArrowRight label="Types" color="#F59E0B" />
          <Box icon="📐" title="TypeScript Interfaces" subtitle="types/index.ts" color="#F59E0B" tech={["StockData", "Prediction", "ModelInfo", "Agent"]} tip="Shared interfaces: StockData, Prediction, ModelInfo, TrainingHistory, MonitoringData, EvaluationResult, AgentResponse, HPOResult, and more." className="min-w-[200px]" />
          <ArrowRight label="Proxy" color="#F59E0B" />
          <Box icon="🔄" title="Next.js Rewrite" subtitle="/api/:path* → :8000" color="#F59E0B" bg="rgba(245,158,11,0.05)" tech={["next.config.mjs"]} tip="Next.js rewrites proxy all /api/* requests from the browser to FastAPI on port 8000, avoiding CORS in production." className="min-w-[180px]" />
        </div>

        {/* Tech stack badges */}
        <div className="flex justify-center">
          <div className="flex flex-wrap justify-center gap-2 rounded-lg border border-white/10 bg-white/[0.02] px-4 py-2">
            {[
              { name: "Next.js 14", color: "#ffffff" },
              { name: "React 18", color: "#61DAFB" },
              { name: "TypeScript 5", color: "#3178C6" },
              { name: "Tailwind CSS", color: "#06B6D4" },
              { name: "Recharts", color: "#FF7300" },
              { name: "Lucide Icons", color: "#F56565" },
              { name: "App Router", color: "#ffffff" },
            ].map((s) => (
              <span key={s.name} className="rounded-md px-2 py-0.5 text-[10px] font-medium" style={{ background: `${s.color}15`, color: s.color }}>
                {s.name}
              </span>
            ))}
          </div>
        </div>
      </SubCanvas>

      {/* ── Backend Sub-Canvas ── */}
      <SubCanvas
        title="⚡ Backend — FastAPI REST API"
        subtitle="High-performance async API with 10 route modules, Prometheus metrics, CORS middleware and nginx reverse proxy — port 8000."
      >
        <Lane label="Request Flow" color="#FF6B35" />
        <div className="flex items-center justify-center gap-3">
          <Box icon="🌐" title="Browser" subtitle="Next.js :3001" color="#ffffff" bg="rgba(255,255,255,0.03)" tech={["fetch"]} tip="User's browser running the Next.js dashboard on port 3001." />
          <ArrowRight label="/api/*" color="#ffffff" />
          <Box icon="🔀" title="Nginx" subtitle="Reverse proxy :80" color="#2496ED" bg="rgba(36,150,237,0.05)" tech={["rate limit", "gzip", "headers"]} tip="Nginx fronts the API: rate limiting (10 req/s per IP, burst 20), max 10 concurrent connections, gzip for JSON ≥1KB, and security headers (X-Frame-Options, X-Content-Type-Options, X-XSS-Protection)." className="min-w-[180px]" />
          <ArrowRight label="upstream" color="#2496ED" />
          <Box icon="🚀" title="FastAPI" subtitle="Uvicorn ASGI :8000" color="#FF6B35" bg="rgba(255,107,53,0.06)" tech={["async", "OpenAPI", "Swagger"]} tip="FastAPI application served by Uvicorn ASGI server on port 8000. Auto-generated OpenAPI docs at /docs." className="min-w-[180px]" />
        </div>

        <div className="flex justify-center"><ArrowDown label="Middleware Stack" color="#EF4444" /></div>

        <Lane label="Middleware" color="#EF4444" />
        <div className="flex items-center justify-center gap-4">
          <Box icon="🌍" title="CORS" subtitle="Allow all origins" color="#EF4444" tech={["CORSMiddleware"]} tip="Allows all origins, methods and headers for development. In production, restrict to dashboard domain." />
          <ArrowRight color="#EF4444" />
          <Box icon="📊" title="Prometheus" subtitle="HTTP metrics" color="#F97316" bg="rgba(249,115,22,0.05)" tech={["histogram", "counter", "gauge"]} tip="Custom middleware tracks: request duration histogram (by method, path, status), active requests gauge, and total request counter. Exposed at /metrics." className="min-w-[180px]" />
          <ArrowRight color="#F97316" />
          <Box icon="🛡️" title="Nginx Headers" subtitle="Security layer" color="#2496ED" tech={["XSS", "MIME", "frame"]} tip="Nginx injects X-Frame-Options: SAMEORIGIN, X-Content-Type-Options: nosniff, X-XSS-Protection: 1; mode=block on all responses." />
        </div>

        <div className="flex justify-center"><ArrowDown label="Route" color="#76B900" /></div>

        <Lane label="API Routers (10 modules)" color="#76B900" />
        <div className="flex flex-wrap items-center justify-center gap-2">
          {[
            { name: "/predict", desc: "LSTM forecast", icon: "🔮", color: "#4ECDC4" },
            { name: "/model", desc: "Info, history, HPO", icon: "🧠", color: "#76B900" },
            { name: "/data", desc: "Stock OHLCV", icon: "📈", color: "#45B7D1" },
            { name: "/monitoring", desc: "Drift & health", icon: "📡", color: "#F97316" },
            { name: "/evaluation", desc: "RAGAS & Judge", icon: "📊", color: "#EC4899" },
            { name: "/agent", desc: "Chat / ask", icon: "🤖", color: "#A855F7" },
            { name: "/train", desc: "Training pipeline", icon: "🏋️", color: "#EF5B5B" },
            { name: "/explainability", desc: "Feature importance", icon: "🔬", color: "#945DD6" },
            { name: "/logs", desc: "Request logs", icon: "📜", color: "#06B6D4" },
            { name: "/mlops", desc: "SLA, registry, costs", icon: "⚙️", color: "#F59E0B" },
          ].map((r) => (
            <div key={r.name} className="rounded-lg border-2 px-3 py-2 text-center" style={{ borderColor: `${r.color}40`, background: `${r.color}08` }}>
              <span className="text-sm">{r.icon}</span>
              <p className="text-[10px] font-bold" style={{ color: r.color }}>{r.name}</p>
              <p className="text-[9px] text-white/40">{r.desc}</p>
            </div>
          ))}
        </div>

        <div className="flex justify-center"><ArrowDown label="Business Logic" color="#0194E2" /></div>

        <Lane label="Backend Services" color="#0194E2" />
        <div className="flex items-center justify-center gap-4">
          <Box icon="🧠" title="LSTM Model" subtitle="PyTorch inference" color="#76B900" tech={["best_model.pth"]} tip="Loads the trained LSTM model for real-time predictions." />
          <Box icon="🗄️" title="SQLite" subtitle="Stock database" color="#4ECDC4" tech={["6700+ rows"]} tip="Local SQLite database with NVIDIA historical stock data." />
          <Box icon="📋" title="MLflow" subtitle="Experiment store :5000" color="#0194E2" tech={["tracking", "registry"]} tip="MLflow tracking server on port 5000 storing all experiments, metrics, and model artifacts." />
          <Box icon="🔥" title="Prometheus" subtitle="Metrics scrape :9090" color="#F97316" tech={["15s interval"]} tip="Prometheus scrapes /metrics every 15 seconds for monitoring dashboards." />
          <Box icon="📈" title="Grafana" subtitle="Dashboards :3000" color="#F97316" bg="rgba(249,115,22,0.05)" tech={["auto-provision"]} tip="Grafana on port 3000 with auto-provisioned dashboards for system health, latency, and model metrics." />
        </div>

        {/* Connection diagram: Frontend ↔ Backend */}
        <div className="mt-2 flex justify-center">
          <div className="rounded-xl border border-nvidia/20 bg-nvidia/5 p-4">
            <p className="mb-3 text-center text-[10px] font-bold uppercase tracking-widest text-nvidia">Full Request Lifecycle</p>
            <div className="flex items-center justify-center gap-2 text-[10px]">
              <span className="rounded border border-white/20 bg-white/5 px-2 py-1 font-bold text-white/70">Browser :3001</span>
              <span className="text-white/30">→</span>
              <span className="rounded border border-white/20 bg-white/5 px-2 py-1 font-mono text-white/50">/api/*</span>
              <span className="text-white/30">→</span>
              <span className="rounded border border-[#2496ED]/30 bg-[#2496ED]/10 px-2 py-1 font-bold text-[#2496ED]">Nginx :80</span>
              <span className="text-white/30">→</span>
              <span className="rounded border border-[#FF6B35]/30 bg-[#FF6B35]/10 px-2 py-1 font-bold text-[#FF6B35]">FastAPI :8000</span>
              <span className="text-white/30">→</span>
              <span className="rounded border border-[#76B900]/30 bg-[#76B900]/10 px-2 py-1 font-bold text-[#76B900]">Service Layer</span>
              <span className="text-white/30">→</span>
              <span className="rounded border border-[#F59E0B]/30 bg-[#F59E0B]/10 px-2 py-1 font-bold text-[#F59E0B]">JSON Response</span>
            </div>
          </div>
        </div>
      </SubCanvas>
      </>)}

      {/* ════════════ AGENT TAB ════════════ */}
      {activeTab === "agent" && (
      <SubCanvas
        title="🤖 LLM Agent & RAG Pipeline"
        subtitle="ReAct agent with 4 custom financial tools and ChromaDB-powered Retrieval-Augmented Generation. LLM: google/gemini-2.0-flash-001 via OpenRouter (configurável por env var)."
      >
        <Lane label="Agent Layer" color="#A855F7" />
        <div className="flex items-center justify-center gap-4">
          <Box icon="💬" title="User Query" subtitle="Natural language question" color="#A855F7" tip="Usuário faz perguntas em linguagem natural sobre ações da NVIDIA, previsões e métricas." />
          <ArrowRight label="Parse" color="#A855F7" />
          <Box icon="🧠" title="ReAct Agent" subtitle="Thought → Action → Observation" color="#A855F7" bg="rgba(168,85,247,0.05)" tech={["ReAct", "LLM"]} tip="Agente que segue o padrão ReAct (Yao et al. 2023): raciocina, escolhe uma tool, observa resultado e repete até resposta final." className="min-w-[200px]" />
          <ArrowRight label="Select" color="#A855F7" />
          <div className="flex flex-col gap-2">
            <div className="flex gap-2">
              <Box icon="📈" title="Stock Data" subtitle="Historical OHLCV" color="#F59E0B" tech={["SQLite"]} tip="Consulta dados históricos de preços e volumes da NVDA no banco SQLite." />
              <Box icon="🔮" title="LSTM Predict" subtitle="5-day forecast" color="#F59E0B" tech={["PyTorch"]} tip="Executa a LSTM para gerar previsões de 5 dias dos preços da NVDA." />
            </div>
            <div className="flex gap-2">
              <Box icon="📊" title="MLflow Metrics" subtitle="Experiment data" color="#F59E0B" tech={["MLflow"]} tip="Consulta métricas e parâmetros dos experimentos registrados no MLflow." />
              <Box icon="🔍" title="RAG Search" subtitle="Doc retrieval" color="#F59E0B" tech={["ChromaDB"]} tip="Busca documentos relevantes na base de conhecimento usando RAG." />
            </div>
          </div>
        </div>

        <div className="flex justify-center"><ArrowDown label="Context" color="#10B981" /></div>

        <Lane label="RAG Layer" color="#10B981" />
        <div className="flex items-center justify-center gap-4">
          <Box icon="📚" title="Knowledge Base" subtitle="Docs & data" color="#10B981" tech={["Markdown"]} tip="Base de conhecimento com documentação do projeto, informações sobre NVIDIA e mercado financeiro." />
          <ArrowRight label="Embed" color="#10B981" />
          <Box icon="🔢" title="Embeddings" subtitle="all-MiniLM-L6-v2" color="#10B981" tech={["Sentence Transformers"]} tip="Modelo de embeddings que converte texto em vetores densos de 384 dimensões." />
          <ArrowRight label="Index" color="#10B981" />
          <Box icon="💎" title="ChromaDB" subtitle="Vector store" color="#10B981" tech={["cosine similarity"]} tip="Banco de vetores ChromaDB que armazena embeddings e permite busca por similaridade." />
          <ArrowRight label="Top-K" color="#10B981" />
          <Box icon="✨" title="Generator" subtitle="Augmented response" color="#10B981" bg="rgba(16,185,129,0.05)" tech={["LLM", "RAG"]} tip="Gera resposta final combinando contexto recuperado com a capacidade do LLM." />
        </div>
      </SubCanvas>
      )}

      {/* ════════════ EVALUATION TAB ════════════ */}
      {activeTab === "evaluation" && (
      <SubCanvas
        title="📊 Evaluation Pipeline"
        subtitle="RAGAS metrics, LLM-as-Judge and A/B prompt testing for RAG quality assurance."
      >
        <Lane label="Golden Set Evaluation" color="#EC4899" />
        <div className="flex items-center justify-center gap-4">
          <Box icon="🏅" title="Golden Set" subtitle="≥ 20 curated Q&A pairs" color="#EC4899" tech={["JSON"]} tip="Conjunto dourado com pelo menos 20 pares (query, expected_answer, contexts) curados manualmente." />
          <ArrowRight label="Evaluate" color="#EC4899" />
          <Box icon="🔬" title="RAG Pipeline" subtitle="Generate answers" color="#EC4899" tech={["agent + RAG"]} tip="Pipeline completo gera respostas para cada query do golden set usando o agente com RAG." />
          <ArrowRight label="Score" color="#EC4899" />
          <Box icon="📏" title="RAGAS (4 Metrics)" subtitle="Automated evaluation" color="#EC4899" bg="rgba(236,72,153,0.05)" tech={["faithfulness", "relevancy", "precision", "recall"]} tip="RAGAS calcula 4 métricas: faithfulness, answer_relevancy, context_precision e context_recall." className="min-w-[200px]" />
        </div>

        <div className="flex justify-center"><ArrowDown label="Quality Gate" color="#8B5CF6" /></div>

        <Lane label="LLM-as-Judge" color="#8B5CF6" />
        <div className="flex items-center justify-center gap-4">
          <Box icon="⚖️" title="LLM Judge" subtitle="OpenRouter LLM evaluates responses" color="#8B5CF6" tech={["Gemini 2.0 Flash", "gpt-4o-mini", "3 criteria"]} tip="LLM avalia cada resposta com 3 critérios: relevância (1-5), acurácia factual (1-5) e utilidade para investimento (1-5). Usa o modelo configurado em LLM_JUDGE_MODEL (padrão: google/gemini-2.0-flash-001 via OpenRouter)." />
          <ArrowRight color="#8B5CF6" />
          <div className="flex gap-3">
            {[
              { name: "Relevance", score: "1-5" },
              { name: "Accuracy", score: "1-5" },
              { name: "Business", score: "1-5" },
            ].map((c) => (
              <div key={c.name} className="rounded-xl border-2 border-[#8B5CF6]/30 bg-[#8B5CF6]/5 px-4 py-2 text-center">
                <p className="text-[10px] font-bold text-[#8B5CF6]">{c.name}</p>
                <p className="text-lg font-bold text-white/80">{c.score}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="flex justify-center"><ArrowDown label="Compare" color="#F59E0B" /></div>

        <Lane label="A/B Prompt Testing" color="#F59E0B" />
        <div className="flex items-center justify-center gap-6">
          <Box icon="🅰️" title="Prompt A" subtitle="Concise prompt" color="#F59E0B" tech={["baseline"]} tip="Prompt de sistema conciso e direto, usado como baseline." />
          <span className="text-sm font-bold text-[#F59E0B]">vs</span>
          <Box icon="🅱️" title="Prompt B" subtitle="Enhanced prompt" color="#F59E0B" tech={["challenger"]} tip="Prompt aprimorado com instruções mais detalhadas e contexto adicional." />
          <ArrowRight label="Winner" color="#F59E0B" />
          <Box icon="🏆" title="Statistical Comparison" subtitle="RAGAS + Judge scores" color="#F59E0B" bg="rgba(245,158,11,0.05)" tech={["comparison"]} tip="Compara estatisticamente métricas RAGAS e scores do LLM-judge para selecionar o melhor prompt." />
        </div>
      </SubCanvas>
      )}

      {/* ════════════ MONITORING TAB ════════════ */}
      {activeTab === "monitoring" && (
      <SubCanvas
        title="📡 Multi-Trigger Monitoring & Retrain System"
        subtitle="3 independent retrain triggers: PSI data drift, model staleness (30 days), and prediction CI breach (concept drift)."
      >
        {/* ── Banner: 3 Triggers ── */}
        <div className="mb-2 rounded-xl border border-nvidia/20 bg-nvidia/5 p-3">
          <p className="mb-2 text-center text-xs font-bold text-nvidia">⚡ Any single trigger firing is sufficient to recommend retraining (defense-in-depth)</p>
          <div className="flex justify-center gap-3">
            {[
              { icon: "📊", name: "Data Drift", desc: "PSI > 0.2", color: "#EF4444" },
              { icon: "⏰", name: "Staleness", desc: "≥ 30 days", color: "#F59E0B" },
              { icon: "📉", name: "CI Breach", desc: "> 20% outside CI", color: "#8B5CF6" },
            ].map((t) => (
              <div key={t.name} className="flex items-center gap-2 rounded-lg border-2 px-3 py-2" style={{ borderColor: `${t.color}40`, background: `${t.color}08` }}>
                <span className="text-lg">{t.icon}</span>
                <div>
                  <p className="text-[10px] font-bold" style={{ color: t.color }}>{t.name}</p>
                  <p className="text-[9px] text-white/40">{t.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <Lane label="Trigger 1 — Data Drift (PSI)" color="#EF4444" />
        <div className="flex items-center justify-center gap-4">
          <Box icon="📊" title="Production Data" subtitle="Live prediction inputs" color="#EF4444" tech={["daily data"]} tip="Dados de produção coletados para comparar com a distribuição de treinamento." />
          <ArrowRight label="Compare" color="#EF4444" />
          <Box icon="📐" title="PSI Calculator" subtitle="Population Stability Index" color="#EF4444" bg="rgba(239,68,68,0.05)" tech={["Evidently", "PSI"]} tip="PSI = Σ (P_actual - P_expected) × ln(P_actual / P_expected). Measures how much input distributions shifted." className="min-w-[180px]" />
          <ArrowRight color="#EF4444" />
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2 rounded-lg border-2 border-yellow-500/30 bg-yellow-500/5 px-3 py-1.5">
              <span className="text-sm">⚠️</span>
              <div><p className="text-[10px] font-bold text-yellow-500">PSI &gt; 0.1 — Warning</p><p className="text-[9px] text-white/40">Monitor closely</p></div>
            </div>
            <div className="flex items-center gap-2 rounded-lg border-2 border-red-500/30 bg-red-500/5 px-3 py-1.5">
              <span className="text-sm">🔴</span>
              <div><p className="text-[10px] font-bold text-red-500">PSI &gt; 0.2 — Retrain</p><p className="text-[9px] text-white/40">→ Champion-Challenger</p></div>
            </div>
          </div>
        </div>

        <div className="flex justify-center"><ArrowDown label="OR" color="#F59E0B" /></div>

        <Lane label="Trigger 2 — Model Staleness (≥ 30 days)" color="#F59E0B" />
        <div className="flex items-center justify-center gap-4">
          <Box icon="🗓️" title="Model Checkpoint" subtitle="best_model.pth mtime" color="#F59E0B" tech={["file system"]} tip="Checks the filesystem modification timestamp of the model checkpoint file." />
          <ArrowRight label="Age check" color="#F59E0B" />
          <Box icon="⏰" title="Staleness Detector" subtitle="now() - last_modified" color="#F59E0B" bg="rgba(245,158,11,0.05)" tech={["timedelta"]} tip="Computes how many days since the model was last trained. Even without measurable drift, markets evolve and a stale model may silently degrade (Sculley et al. 2015)." className="min-w-[200px]" />
          <ArrowRight color="#F59E0B" />
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2 rounded-lg border-2 border-green-500/30 bg-green-500/5 px-3 py-1.5">
              <span className="text-sm">✅</span>
              <div><p className="text-[10px] font-bold text-green-500">&lt; 30 days — Fresh</p><p className="text-[9px] text-white/40">No action needed</p></div>
            </div>
            <div className="flex items-center gap-2 rounded-lg border-2 border-orange-500/30 bg-orange-500/5 px-3 py-1.5">
              <span className="text-sm">⏰</span>
              <div><p className="text-[10px] font-bold text-orange-500">≥ 30 days — Stale</p><p className="text-[9px] text-white/40">→ Champion-Challenger</p></div>
            </div>
          </div>
        </div>

        <div className="flex justify-center"><ArrowDown label="OR" color="#8B5CF6" /></div>

        <Lane label="Trigger 3 — Prediction CI Breach (Concept Drift)" color="#8B5CF6" />
        <div className="flex items-center justify-center gap-4">
          <Box icon="🔮" title="Predictions" subtitle="Model forecasts" color="#8B5CF6" tech={["LSTM output"]} tip="Model predictions from the LSTM for recent dates." />
          <ArrowRight label="vs" color="#8B5CF6" />
          <Box icon="📈" title="Actual Values" subtitle="Observed Close prices" color="#8B5CF6" tech={["market data"]} tip="Real observed Close prices from the market." />
          <ArrowRight label="CI check" color="#8B5CF6" />
          <Box icon="📉" title="CI Breach Detector" subtitle="residual_std × z_0.975" color="#8B5CF6" bg="rgba(139,92,246,0.05)" tech={["95% CI", "z-score"]} tip="Computes 95% CI from residual std. If >20% of actuals fall outside, the model's learned patterns no longer match reality (Gama et al. 2014 — Concept Drift)." className="min-w-[200px]" />
          <ArrowRight color="#8B5CF6" />
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2 rounded-lg border-2 border-green-500/30 bg-green-500/5 px-3 py-1.5">
              <span className="text-sm">✅</span>
              <div><p className="text-[10px] font-bold text-green-500">≤ 20% outside CI</p><p className="text-[9px] text-white/40">Model fits well</p></div>
            </div>
            <div className="flex items-center gap-2 rounded-lg border-2 border-purple-500/30 bg-purple-500/5 px-3 py-1.5">
              <span className="text-sm">💥</span>
              <div><p className="text-[10px] font-bold text-purple-400">&gt; 20% outside CI</p><p className="text-[9px] text-white/40">→ Champion-Challenger</p></div>
            </div>
          </div>
        </div>

        <div className="flex justify-center"><ArrowDown label="Metrics Export" color="#F97316" /></div>

        <Lane label="Operational Monitoring" color="#F97316" />
        <div className="flex items-center justify-center gap-4">
          <Box icon="🚀" title="FastAPI Metrics" subtitle="Custom counters & histograms" color="#F97316" tech={["latency", "errors", "count"]} tip="FastAPI expõe métricas: histograma de latência, contador de requests, gauge ativo e contador de erros." />
          <ArrowRight label="/metrics" color="#F97316" />
          <Box icon="🔥" title="Prometheus" subtitle="Time series — port 9090" color="#F97316" tech={["15s scrape"]} tip="Prometheus coleta métricas do FastAPI a cada 15s e armazena como séries temporais." />
          <ArrowRight label="Query" color="#F97316" />
          <Box icon="📈" title="Grafana" subtitle="Dashboard — port 3000" color="#F97316" bg="rgba(249,115,22,0.05)" tech={["auto-provision"]} tip="Dashboard Grafana com painéis de System Health, latência, métricas de modelo e drift." />
        </div>

        <div className="flex justify-center"><ArrowDown label="LLM Traces" color="#06B6D4" /></div>

        <Lane label="LLM Telemetry" color="#06B6D4" />
        <div className="flex items-center justify-center gap-4">
          <Box icon="🤖" title="LLM Calls" subtitle="Agent & RAG interactions" color="#06B6D4" tech={["traced calls"]} tip="Todas as chamadas ao LLM feitas pelo agente e RAG são instrumentadas." />
          <ArrowRight label="Trace" color="#06B6D4" />
          <Box icon="🔭" title="MLflow LLM Tracing" subtitle="Native OpenAI-compatible traces" color="#06B6D4" tech={["MLflow", "mlruns/mlflow.db"]} tip="MLflow instrumenta automaticamente chamadas LLM via mlflow.openai.autolog(). Traces hierárquicos com latência, tokens e custo são armazenados em mlruns/mlflow.db e consultados pelo /cost-analysis." />
          <ArrowRight label="Analyze" color="#06B6D4" />
          <div className="flex gap-2">
            {["Faithfulness", "Relevancy", "Token Usage", "Latency"].map((m) => (
              <div key={m} className="rounded-lg border border-[#06B6D4]/30 bg-[#06B6D4]/5 px-3 py-2 text-center">
                <p className="text-[9px] font-medium text-[#06B6D4]">{m}</p>
              </div>
            ))}
          </div>
        </div>
      </SubCanvas>
      )}

      {/* ════════════ SECURITY TAB ════════════ */}
      {activeTab === "security" && (
      <SubCanvas
        title="🔒 Security & Governance"
        subtitle="OWASP-aligned guardrails, PII detection, Red Team testing, LGPD compliance and documentation."
      >
        <Lane label="Input Protection" color="#EF4444" />
        <div className="flex items-center justify-center gap-4">
          <Box icon="💬" title="User Input" subtitle="Raw text query" color="#EF4444" tip="Texto bruto enviado pelo usuário que precisa ser validado antes de processar." />
          <ArrowRight label="Validate" color="#EF4444" />
          <Box icon="🛡️" title="Input Guardrail" subtitle="Multi-layer validation" color="#EF4444" bg="rgba(239,68,68,0.05)" tech={["injection detect", "topic filter", "max length"]} tip="Detecta prompt injection (6+ padrões regex), valida tópico financeiro e limita tamanho a 4096 chars." className="min-w-[200px]" />
          <ArrowRight label="Clean" color="#EF4444" />
          <Box icon="✅" title="Sanitized Input" subtitle="Safe to process" color="#10B981" tech={["validated"]} tip="Input limpo e validado, pronto para ser processado pelo LLM/modelo." />
        </div>

        <div className="flex justify-center"><ArrowDown label="LLM Processing" color="#8B5CF6" /></div>

        <Lane label="Output Protection" color="#F97316" />
        <div className="flex items-center justify-center gap-4">
          <Box icon="🤖" title="LLM Output" subtitle="Raw response" color="#F97316" tip="Resposta gerada pelo LLM que precisa ser sanitizada." />
          <ArrowRight label="Sanitize" color="#F97316" />
          <Box icon="🔍" title="PII Detection" subtitle="Presidio analyzer" color="#F97316" tech={["CPF", "email", "phone", "name"]} tip="Detecta PII usando Presidio: PERSON, EMAIL, PHONE, BR_CPF, CREDIT_CARD, IP. Regex fallback quando indisponível." />
          <ArrowRight label="Filter" color="#F97316" />
          <Box icon="📝" title="Output Guardrail" subtitle="Content filter + disclaimer" color="#F97316" bg="rgba(249,115,22,0.05)" tech={["anonymize", "disclaimer"]} tip="Anonimiza PII detectado, filtra conteúdo e adiciona disclaimer de risco para decisões financeiras." />
        </div>

        <div className="flex justify-center"><ArrowDown label="Governance" color="#0194E2" /></div>

        <Lane label="Documentation & Compliance" color="#0194E2" />
        <div className="flex items-center justify-center gap-4">
          <Box icon="📋" title="OWASP Mapping" subtitle="≥ 5 threats mapped" color="#0194E2" tech={["Top 10 LLM"]} tip="Mapeamento de pelo menos 5 ameaças do OWASP Top 10 for LLM Applications com mitigações implementadas." />
          <ArrowRight color="#0194E2" />
          <Box icon="🎯" title="Red Team Report" subtitle="≥ 5 adversarial scenarios" color="#0194E2" tech={["adversarial"]} tip="Relatório com pelo menos 5 cenários adversariais testados e documentados." />
          <ArrowRight color="#0194E2" />
          <Box icon="📜" title="LGPD Plan" subtitle="Data protection compliance" color="#0194E2" tech={["Lei 13.709"]} tip="Plano de conformidade com a Lei Geral de Proteção de Dados (LGPD)." />
          <ArrowRight color="#0194E2" />
          <div className="flex flex-col gap-2">
            <Box icon="📄" title="Model Card" subtitle="Architecture + performance" color="#945DD6" tech={["Mitchell et al."]} tip="Model Card documentando arquitetura, dados, métricas, limitações e vieses do modelo." />
            <Box icon="📑" title="System Card" subtitle="Full system documentation" color="#945DD6" tech={["end-to-end"]} tip="System Card com documentação completa do sistema: propósito, riscos, fairness e explicabilidade." />
          </div>
        </div>
      </SubCanvas>
      )}

      {/* ════════════ CHAMPION TAB ════════════ */}
      {activeTab === "champion" && (
      <SubCanvas
        title="🏆 Champion-Challenger Pipeline"
        subtitle="Multi-trigger retraining with Optuna HPO Bayesian search, holdout comparison and promotion gate."
      >
        <Lane label="Retrain Triggers (any 1 of 3)" color="#EF4444" />
        <div className="flex items-center justify-center gap-3">
          <Box icon="📊" title="Data Drift" subtitle="PSI > 0.2" color="#EF4444" bg="rgba(239,68,68,0.05)" tech={["Evidently"]} tip="When PSI exceeds 0.2 on any monitored feature, input distributions have shifted significantly." />
          <span className="text-xs font-bold text-white/20">OR</span>
          <Box icon="⏰" title="Staleness" subtitle="≥ 30 days old" color="#F59E0B" bg="rgba(245,158,11,0.05)" tech={["timedelta"]} tip="If the model hasn't been retrained in 30+ days, it may miss recent market regime changes." />
          <span className="text-xs font-bold text-white/20">OR</span>
          <Box icon="📉" title="CI Breach" subtitle="> 20% outside 95% CI" color="#8B5CF6" bg="rgba(139,92,246,0.05)" tech={["concept drift"]} tip="If more than 20% of actual values fall outside the model's 95% confidence interval, the learned patterns no longer match reality." />
        </div>

        <div className="flex justify-center"><ArrowDown label="Trigger fired" color="#EF5B5B" /></div>

        <Lane label="Optuna HPO — Challenger Training" color="#EF5B5B" />
        <div className="flex items-center justify-center gap-4">
          <Box icon="📂" title="New Data" subtitle="Latest market data" color="#EF5B5B" tech={["SQLite"]} tip="Data more recent than the champion's training set, capturing current market dynamics." />
          <ArrowRight label="Feed" color="#EF5B5B" />
          <Box icon="🎯" title="Optuna TPE" subtitle="Bayesian HPO search" color="#EF5B5B" bg="rgba(239,91,91,0.08)" tech={["TPE sampler", "20 trials"]} tip="Optuna's Tree-structured Parzen Estimator runs 20 trials exploring: hidden_size ∈ {32,64,128,256}, num_layers ∈ [1,4], lr ∈ [1e-5,1e-2], dropout ∈ [0.1,0.5], batch_size ∈ {16,32,64,128}." className="min-w-[200px]" />
          <ArrowRight label="Best params" color="#EF5B5B" />
          <Box icon="⚔️" title="Challenger LSTM" subtitle="Full training with best HP" color="#EF5B5B" bg="rgba(239,91,91,0.05)" tech={["PyTorch", "early stop"]} tip="Final challenger is trained with Optuna's best hyperparameters using full epochs and early stopping. This gives it a fair chance to beat the champion even when data dynamics have changed." className="min-w-[200px]" />
        </div>

        {/* Optuna search space badges */}
        <div className="flex justify-center">
          <div className="flex flex-wrap justify-center gap-2 rounded-lg border border-[#EF5B5B]/20 bg-[#EF5B5B]/5 px-4 py-2">
            {[
              { param: "hidden_size", range: "{32, 64, 128, 256}" },
              { param: "num_layers", range: "[1, 4]" },
              { param: "learning_rate", range: "[1e-5, 1e-2] log" },
              { param: "dropout", range: "[0.1, 0.5]" },
              { param: "batch_size", range: "{16, 32, 64, 128}" },
            ].map((s) => (
              <div key={s.param} className="rounded bg-[#1a1c24] px-2 py-1 text-center">
                <p className="text-[9px] font-bold text-[#EF5B5B]">{s.param}</p>
                <p className="text-[8px] font-mono text-white/40">{s.range}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="flex justify-center"><ArrowDown label="Evaluate Both" color="#76B900" /></div>

        <Lane label="Holdout Set Comparison" color="#76B900" />
        <div className="flex items-center justify-center gap-6">
          <Box icon="👑" title="Champion" subtitle="Current best_model.pth" color="#0194E2" bg="rgba(1,148,226,0.05)" tech={["production", "fixed HP"]} tip="Current production model (best_model.pth) with its original fixed hyperparameters." className="min-w-[180px]" />
          <div className="flex flex-col items-center gap-1">
            <span className="text-xs font-bold text-white/40">vs</span>
            <div className="rounded-lg border border-nvidia/30 bg-nvidia/5 px-4 py-2 text-center">
              <p className="text-[10px] font-bold text-nvidia">Holdout Set</p>
              <p className="text-[9px] text-white/40">RMSE comparison</p>
            </div>
          </div>
          <Box icon="⚔️" title="Challenger" subtitle="Optuna-optimized LSTM" color="#EF5B5B" bg="rgba(239,91,91,0.05)" tech={["Optuna HPO", "new data"]} tip="Challenger trained with Bayesian-optimized hyperparameters on latest data. Has advantage of both HPO and fresh data." className="min-w-[180px]" />
        </div>

        <div className="flex justify-center"><ArrowDown label="Decision" color="#F59E0B" /></div>

        <Lane label="Promotion Gate" color="#F59E0B" />
        <div className="flex items-center justify-center gap-6">
          <div className="flex items-center gap-2 rounded-xl border-2 border-red-500/30 bg-red-500/5 px-5 py-3 text-center">
            <span className="text-lg">❌</span>
            <div><p className="text-xs font-bold text-red-400">RMSE &lt; 0.5% better</p><p className="text-[10px] text-white/40">Keep Champion</p></div>
          </div>
          <div className="text-xs font-bold text-white/20">OR</div>
          <div className="flex items-center gap-2 rounded-xl border-2 border-green-500/30 bg-green-500/5 px-5 py-3 text-center">
            <span className="text-lg">✅</span>
            <div><p className="text-xs font-bold text-green-400">RMSE ≥ 0.5% better</p><p className="text-[10px] text-white/40">Promote Challenger → New Champion</p></div>
          </div>
        </div>
      </SubCanvas>
      )}

      {/* ════════════ TESTING TAB ════════════ */}
      {activeTab === "testing" && (
      <SubCanvas
        title="🧪 Testing & CI/CD Quality Gates"
        subtitle="12 test modules (46 files, 577+ tests), pytest with coverage, linting, type checking and automated CI/CD."
      >
        <Lane label="Test Suites" color="#2496ED" />
        <div className="flex flex-wrap items-center justify-center gap-2">
          {[
            { name: "test_agent", count: 4, color: "#A855F7" },
            { name: "test_api", count: 14, color: "#FF6B35" },
            { name: "test_training", count: 3, color: "#76B900" },
            { name: "test_monitoring", count: 6, color: "#F97316" },
            { name: "test_security", count: 2, color: "#EF4444" },
            { name: "test_models", count: 2, color: "#76B900" },
            { name: "test_prediction", count: 2, color: "#4ECDC4" },
            { name: "test_explainability", count: 1, color: "#945DD6" },
            { name: "test_data", count: 3, color: "#45B7D1" },
            { name: "test_etl", count: 5, color: "#4ECDC4" },
            { name: "test_config", count: 1, color: "#0194E2" },
            { name: "test_utils", count: 1, color: "#06B6D4" },
          ].map((t) => (
            <div key={t.name} className="rounded-lg border px-3 py-1.5 text-center" style={{ borderColor: `${t.color}40`, background: `${t.color}08` }}>
              <p className="text-[10px] font-bold" style={{ color: t.color }}>{t.name}/</p>
              <p className="text-[9px] text-white/40">{t.count} file{t.count > 1 ? "s" : ""}</p>
            </div>
          ))}
        </div>

        <div className="flex justify-center"><ArrowDown label="pytest runner" color="#2496ED" /></div>

        <Lane label="Quality Gates (CI)" color="#2496ED" />
        <div className="flex items-center justify-center gap-3">
          <Box icon="✨" title="ruff lint" subtitle="Code style" color="#2496ED" tech={["ruff check"]} tip="Linter ruff verifica estilo de código e padrões Python." />
          <ArrowRight color="#2496ED" />
          <Box icon="🔤" title="mypy" subtitle="Type checking" color="#2496ED" tech={["type hints"]} tip="Verificação estática de tipos com mypy --ignore-missing-imports." />
          <ArrowRight color="#2496ED" />
          <Box icon="🔒" title="bandit" subtitle="Security scan" color="#2496ED" tech={["SAST"]} tip="Bandit escaneia código Python buscando vulnerabilidades de segurança." />
          <ArrowRight color="#2496ED" />
          <Box icon="📦" title="pip-audit" subtitle="Dependency audit" color="#2496ED" tech={["CVE check"]} tip="pip-audit verifica dependências contra bases de vulnerabilidades conhecidas (CVE)." />
          <ArrowRight color="#2496ED" />
          <Box icon="📊" title="Coverage" subtitle="≥ 60% threshold" color="#76B900" bg="rgba(118,185,0,0.05)" tech={["--cov-fail-under=60"]} tip="Cobertura mínima de 60% é obrigatória. Pipeline falha se abaixo do threshold." />
        </div>

        <div className="flex justify-center"><ArrowDown label="All Pass" color="#10B981" /></div>

        <Lane label="Build & Deploy" color="#10B981" />
        <div className="flex items-center justify-center gap-4">
          <Box icon="⚙️" title="GitHub Actions" subtitle="Automated CI/CD" color="#10B981" tech={["on push", "on PR"]} tip="GitHub Actions executa pipeline completo automaticamente a cada push ou PR em src/, tests/ ou evaluation/." />
          <ArrowRight label="Build" color="#10B981" />
          <Box icon="🐳" title="Docker Build" subtitle="Multi-stage images" color="#10B981" tech={["Dockerfile", "API"]} tip="Build de imagens Docker multi-stage para aplicação principal e API." />
          <ArrowRight label="Health Check" color="#10B981" />
          <Box icon="✅" title="Deploy Ready" subtitle="All gates passed" color="#10B981" bg="rgba(16,185,129,0.05)" tech={["compose up"]} tip="Docker Compose sobe todos os serviços e executa health check no endpoint /health." />
        </div>
      </SubCanvas>
      )}

      {/* ════════════ EXPLAINABILITY TAB ════════════ */}
      {activeTab === "explainability" && (<>
      {/* Overview banner */}
      <div className="rounded-xl border border-[#945DD6]/20 bg-[#945DD6]/5 p-4">
        <p className="mb-2 text-center text-xs font-bold text-[#945DD6]">🔬 Two complementary explainability methods — Global + Local</p>
        <div className="flex justify-center gap-6">
          <div className="flex items-center gap-2 rounded-lg border-2 border-[#945DD6]/30 bg-[#945DD6]/8 px-4 py-2">
            <span className="text-lg">🔀</span>
            <div>
              <p className="text-[10px] font-bold text-[#945DD6]">Permutation Importance</p>
              <p className="text-[9px] text-white/40">Global: which features matter most overall?</p>
            </div>
          </div>
          <div className="flex items-center gap-2 rounded-lg border-2 border-[#22C55E]/30 bg-[#22C55E]/8 px-4 py-2">
            <span className="text-lg">🍋</span>
            <div>
              <p className="text-[10px] font-bold text-[#22C55E]">LIME</p>
              <p className="text-[9px] text-white/40">Local: why did the model predict <em>this</em> value?</p>
            </div>
          </div>
        </div>
      </div>

      {/* ── Method 1: Permutation Importance (global) ── */}
      <SubCanvas
        title="🔀 Permutation Importance — Global Explanation"
        subtitle="Model-agnostic global method: shuffle each feature across all samples and measure RMSE increase (Breiman 2001)."
      >
        <Lane label="Global Pipeline" color="#945DD6" />
        <div className="flex items-center justify-center gap-4">
          <Box icon="🧠" title="Trained LSTM" subtitle="best_model.pth" color="#945DD6" tech={["PyTorch"]} tip="Trained LSTM model loaded for feature importance analysis." />
          <ArrowRight label="For each feature" color="#945DD6" />
          <Box icon="🔀" title="Feature Permutation" subtitle="Shuffle across all samples" color="#945DD6" bg="rgba(148,93,214,0.05)" tech={["10 repeats", "model-agnostic"]} tip="For each feature (Open, High, Low, Close, Volume), shuffles its values across all samples and sequence positions, then measures the increase in RMSE. 10 repeats for statistical stability." className="min-w-[200px]" />
          <ArrowRight label="ΔRMSE" color="#945DD6" />
          <Box icon="📊" title="Importance Ranking" subtitle="Higher ΔRMSE = more important" color="#945DD6" bg="rgba(148,93,214,0.05)" tech={["bar chart", "JSON"]} tip="Features ranked by mean RMSE increase. A large increase means the model relies heavily on that feature. Results saved to outputs/explainability/permutation_importance.json." className="min-w-[180px]" />
        </div>

        <div className="flex justify-center"><ArrowDown label="Global Ranking" color="#945DD6" /></div>

        <div className="flex justify-center">
          <div className="flex gap-3">
            {["Close", "High", "Low", "Open", "Volume"].map((f, i) => (
              <div key={f} className="rounded-lg border-2 px-4 py-2.5 text-center" style={{ borderColor: i === 0 ? "#945DD6" : "rgba(148,93,214,0.3)", background: i === 0 ? "rgba(148,93,214,0.12)" : "rgba(148,93,214,0.04)" }}>
                <p className="text-[10px] font-bold text-[#945DD6]">#{i + 1}</p>
                <p className="text-xs font-semibold text-white/70">{f}</p>
                <p className="text-[9px] text-white/30">{i === 0 ? "highest impact" : ""}</p>
              </div>
            ))}
          </div>
        </div>
      </SubCanvas>

      {/* ── Method 2: LIME (local) ── */}
      <SubCanvas
        title="🍋 LIME — Local Explanation"
        subtitle="Local Interpretable Model-agnostic Explanations: approximates the LSTM with a linear model around each prediction (Ribeiro et al. 2016)."
      >
        <Lane label="Single-Sample Pipeline" color="#22C55E" />
        <div className="flex items-center justify-center gap-3">
          <Box icon="📌" title="Target Sample" subtitle="1 sequence to explain" color="#22C55E" tech={["last timestep"]} tip="Selects a single prediction to explain. LIME focuses on the last timestep of the 60-day sequence, which is the most recent information the model sees." />
          <ArrowRight label="Perturb" color="#22C55E" />
          <Box icon="🎲" title="LIME Perturbation" subtitle="500 neighborhood samples" color="#22C55E" bg="rgba(34,197,94,0.05)" tech={["LimeTabularExplainer", "regression"]} tip="LIME generates ~500 perturbed versions of the input by varying each feature around its original value, then queries the LSTM on each perturbation." className="min-w-[200px]" />
          <ArrowRight label="Fit" color="#22C55E" />
          <Box icon="📐" title="Local Linear Model" subtitle="Weighted linear regression" color="#22C55E" bg="rgba(34,197,94,0.05)" tech={["interpretable", "local R²"]} tip="Fits a weighted linear regression on the perturbed samples, where weights decrease with distance from the original point. Produces per-feature contribution weights and a local R² score." className="min-w-[200px]" />
        </div>

        <div className="flex justify-center"><ArrowDown label="Per-feature weights" color="#22C55E" /></div>

        <Lane label="Local Explanation Output" color="#22C55E" />
        <div className="flex items-center justify-center gap-4">
          <div className="flex gap-2">
            {[
              { name: "Close", w: "+0.08", color: "#22C55E" },
              { name: "Open", w: "+0.05", color: "#22C55E" },
              { name: "Low", w: "+0.04", color: "#22C55E" },
              { name: "High", w: "-0.03", color: "#EF4444" },
              { name: "Volume", w: "-0.01", color: "#EF4444" },
            ].map((f) => (
              <div key={f.name} className="rounded-lg border-2 px-3 py-2 text-center" style={{ borderColor: `${f.color}40`, background: `${f.color}08` }}>
                <p className="text-[10px] font-bold" style={{ color: f.color }}>{f.w}</p>
                <p className="text-xs font-semibold text-white/70">{f.name}</p>
                <p className="text-[9px] text-white/30">{parseFloat(f.w) > 0 ? "pushes ↑" : "pushes ↓"}</p>
              </div>
            ))}
          </div>
          <div className="flex flex-col items-center gap-1 rounded-xl border border-[#22C55E]/30 bg-[#22C55E]/5 px-4 py-3 text-center">
            <p className="text-[9px] font-bold uppercase tracking-wider text-[#22C55E]">Local R²</p>
            <p className="text-xl font-bold text-white/80">0.85</p>
            <p className="text-[9px] text-white/30">linear approximation fit</p>
          </div>
        </div>

        <div className="flex justify-center"><ArrowDown label="Aggregate N samples" color="#0194E2" /></div>

        <Lane label="Batch LIME → Global View" color="#0194E2" />
        <div className="flex items-center justify-center gap-4">
          <Box icon="📦" title="Batch Explain" subtitle="20+ random samples" color="#0194E2" tech={["explain_batch_with_lime"]} tip="Explains 20+ randomly selected samples and collects their local feature weights." />
          <ArrowRight label="Mean |w|" color="#0194E2" />
          <Box icon="📊" title="Aggregated Ranking" subtitle="mean |LIME weight| per feature" color="#0194E2" bg="rgba(1,148,226,0.05)" tech={["global ranking", "std"]} tip="Averaging absolute LIME weights across many samples yields a global importance ranking comparable to permutation importance, plus per-sample detail." className="min-w-[200px]" />
          <ArrowRight label="Log" color="#0194E2" />
          <Box icon="📋" title="MLflow" subtitle="Artifacts & metrics" color="#0194E2" tech={["JSON", "plots", "tags"]} tip="LIME results (JSON, bar charts, local R² scores) are logged as MLflow artifacts with tag explainability_lime=true." />
        </div>

        {/* LIME vs Permutation comparison */}
        <div className="mt-2 flex justify-center">
          <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4">
            <p className="mb-3 text-center text-[10px] font-bold uppercase tracking-widest text-white/40">Permutation vs LIME — complementary views</p>
            <div className="grid grid-cols-2 gap-4">
              <div className="rounded-lg border border-[#945DD6]/20 bg-[#945DD6]/5 p-3">
                <p className="text-[10px] font-bold text-[#945DD6]">🔀 Permutation Importance</p>
                <ul className="mt-1 space-y-0.5 text-[9px] text-white/40">
                  <li>• <strong>Global</strong> — ranks all features</li>
                  <li>• Measures RMSE increase when shuffled</li>
                  <li>• Answers: <em>"What matters most overall?"</em></li>
                  <li>• Breiman 2001 · Molnar 2022</li>
                </ul>
              </div>
              <div className="rounded-lg border border-[#22C55E]/20 bg-[#22C55E]/5 p-3">
                <p className="text-[10px] font-bold text-[#22C55E]">🍋 LIME</p>
                <ul className="mt-1 space-y-0.5 text-[9px] text-white/40">
                  <li>• <strong>Local</strong> — explains 1 prediction</li>
                  <li>• Fits linear model in neighborhood</li>
                  <li>• Answers: <em>"Why this specific value?"</em></li>
                  <li>• Ribeiro, Singh & Guestrin 2016</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </SubCanvas>
      </>)}

      {/* ════════════ CHECKLIST TAB ════════════ */}
      {activeTab === "checklist" && (
      <div className="rounded-xl border border-nvidia/20 bg-surface-card p-5">
        <h3 className="mb-1 text-sm font-semibold">📋 Datathon Phase 5 — Requirements Coverage</h3>
        <p className="mb-4 text-[11px] text-white/40">
          Mapping of all Datathon evaluation criteria to implemented components.
        </p>

        {/* Etapa 1 */}
        <p className="mb-2 mt-4 text-xs font-bold text-[#4ECDC4]">STAGE 1 — Data + Baseline (Phases 01-02) · 10%</p>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <Check label="EDA Documented" detail="notebooks/EDA.ipynb — exploratory analysis with insights" />
          <Check label="Baseline Trained + MLflow" detail="src/training/train.py — LSTM with full MLflow tracking" />
          <Check label="Versioned Pipeline (DVC + Docker)" detail="dvc.yaml + docker-compose.yml — reproducible pipeline" />
          <Check label="Business Metrics Mapped" detail="RMSE, MAE, R², MAPE → stock prediction accuracy" />
          <Check label="pyproject.toml Dependencies" detail="pyproject.toml with all managed dependencies" />
        </div>

        {/* Etapa 2 */}
        <p className="mb-2 mt-4 text-xs font-bold text-[#A855F7]">STAGE 2 — LLM + Agent (Phases 03-05) · 15%</p>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <Check label="LLM Served via API" detail="src/api/ — FastAPI with LLM endpoints on port 8000" />
          <Check label="ReAct Agent with ≥ 3 Tools" detail="src/agent/ — 4 tools: stock data, LSTM predict, MLflow, RAG" />
          <Check label="RAG Pipeline" detail="src/agent/rag_pipeline.py — ChromaDB + Sentence Transformers" />
          <Check label="CI/CD Pipeline" detail=".github/workflows/ci.yml — lint → test → build → deploy" />
          <Check label="LLM Benchmark (≥ 3 configs)" detail="docs/LLM_BENCHMARK.md — documented configurations" />
        </div>

        {/* Etapa 3 */}
        <p className="mb-2 mt-4 text-xs font-bold text-[#EC4899]">STAGE 3 — Evaluation + Observability (Phases 03-05) · 20%</p>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <Check label="Golden Set ≥ 20 Pairs" detail="data/golden_set/golden_set.json — curated Q&A pairs" />
          <Check label="RAGAS 4 Metrics" detail="evaluation/ragas_eval.py — faithfulness, relevancy, precision, recall" />
          <Check label="LLM-as-Judge ≥ 3 Criteria" detail="evaluation/llm_judge.py — relevance, accuracy, business utility" />
          <Check label="Telemetry Dashboard" detail="Prometheus:9090 + Grafana:3000 + MLflow LLM Tracing (traces em mlruns/mlflow.db)" />
          <Check label="Drift Detection Documented" detail="src/monitoring/drift.py — Evidently PSI with thresholds" />
        </div>

        {/* Etapa 4 */}
        <p className="mb-2 mt-4 text-xs font-bold text-[#0194E2]">STAGE 4 — Security + Governance (Phases 04-05) · 15%</p>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <Check label="OWASP Mapping (≥ 5 Threats)" detail="docs/OWASP_MAPPING.md — LLM-specific threat mitigations" />
          <Check label="Input + Output Guardrails" detail="src/security/guardrails.py — injection, PII, content filter" />
          <Check label="Red Team (≥ 5 Scenarios)" detail="docs/RED_TEAM_REPORT.md — adversarial test scenarios" />
          <Check label="LGPD Compliance Plan" detail="docs/LGPD_PLAN.md — data protection compliance" />
          <Check label="Explainability + Fairness" detail="src/explainability/ — Permutation Importance (global) + LIME (local)" />
          <Check label="System Card + Model Card" detail="docs/SYSTEM_CARD.md + docs/MODEL_CARD.md" />
        </div>

        {/* Additional */}
        <p className="mb-2 mt-4 text-xs font-bold text-[#76B900]">ADDITIONAL — PyTorch + MLflow · 5% | Documentation · 5%</p>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <Check label="PyTorch LSTM Model" detail="src/models/ — 2-layer LSTM with 128 hidden units" />
          <Check label="MLflow Experiment Tracking" detail="Metrics, params, artifacts, model registry" />
          <Check label="Champion-Challenger" detail="src/training/champion_challenger.py — automated promotion" />
          <Check label="Test Coverage ≥ 60%" detail="45+ test files across 12 modules, pytest --cov-fail-under=60" />
          <Check label="Type Hints + Docstrings" detail="mypy type checking + structured logging throughout" />
          <Check label="Makefile Shortcuts" detail="make train, make test, make serve — documented workflows" />
        </div>
      </div>
      )}
    </div>
  );
}
