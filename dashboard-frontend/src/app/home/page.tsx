"use client";

import {
  Brain,
  Database,
  BarChart3,
  Cpu,
  GitBranch,
  Shield,
  Github,
  Mail,
  Layers,
  ArrowRight,
  Zap,
  LineChart,
  Bot,
  Search,
} from "lucide-react";
import Link from "next/link";

const TECH_STACK = [
  { name: "PyTorch", desc: "Deep Learning framework", color: "#EE4C2C" },
  { name: "LSTM", desc: "Recurrent Neural Network", color: "#76B900" },
  { name: "MLflow", desc: "Experiment Tracking", color: "#0194E2" },
  { name: "Optuna", desc: "Hyperparameter Optimization", color: "#EF5B5B" },
  { name: "FastAPI", desc: "REST API", color: "#009688" },
  { name: "Next.js", desc: "Dashboard Frontend", color: "#ffffff" },
  { name: "Docker", desc: "Containerization", color: "#2496ED" },
  { name: "DVC", desc: "Data Version Control", color: "#945DD6" },
];

const PIPELINE_STEPS = [
  {
    icon: Database,
    title: "ETL Pipeline",
    desc: "Extração de dados via Yahoo Finance com carregamento em SQLite. Dados desde 2017 com 6700+ registros.",
  },
  {
    icon: Cpu,
    title: "Treinamento LSTM",
    desc: "Modelo LSTM de 2 camadas com 128 unidades ocultas, dropout 0.2, early stopping e gradient clipping.",
  },
  {
    icon: Zap,
    title: "Otimização HPO",
    desc: "Busca Bayesiana de hiperparâmetros com Optuna (50+ trials) para encontrar a melhor configuração.",
  },
  {
    icon: BarChart3,
    title: "Predição",
    desc: "Forecast iterativo de 30 dias com Monte Carlo Dropout para intervalos de confiança.",
  },
  {
    icon: LineChart,
    title: "Monitoramento",
    desc: "Tracking completo com MLflow, métricas em tempo real e versionamento de modelos.",
  },
  {
    icon: Shield,
    title: "Segurança & Deploy",
    desc: "Pipeline CI/CD com GitHub Actions, Docker Compose e boas práticas de segurança OWASP.",
  },
];

const QUICK_LINKS = [
  { href: "/predictions", label: "Stock Predictions", icon: BarChart3, emoji: "📊", desc: "Veja previsões do modelo LSTM" },
  { href: "/metrics", label: "Model Metrics", icon: LineChart, emoji: "📈", desc: "Métricas e performance do modelo" },
  { href: "/model-schema", label: "Model Architecture", icon: Brain, emoji: "🧠", desc: "Arquitetura da rede neural" },
  { href: "/evaluation", label: "Evaluation", icon: Search, emoji: "📋", desc: "Avaliação e benchmarks" },
  { href: "/agent", label: "AI Agent", icon: Bot, emoji: "🤖", desc: "Assistente IA para análises" },
];

export default function HomePage() {
  return (
    <div className="space-y-8">
      {/* Hero */}
      <div className="relative overflow-hidden rounded-2xl border border-nvidia/30 bg-gradient-to-br from-nvidia/10 via-surface-card to-surface-card p-8">
        <div className="absolute -right-16 -top-16 h-64 w-64 rounded-full bg-nvidia/5 blur-3xl" />
        <div className="absolute -bottom-20 -left-20 h-48 w-48 rounded-full bg-nvidia/5 blur-3xl" />
        <div className="relative">
          <div className="mb-2 flex items-center gap-2">
            <span className="rounded-full bg-nvidia/20 px-3 py-1 text-xs font-semibold text-nvidia">
              FIAP Post-Tech MLET — Tech Challenge Phase 4 / Phase 5
            </span>
          </div>
          <h1 className="mb-3 text-4xl font-bold">
            <span className="text-nvidia">NVIDIA</span> MLOps Platform
          </h1>
          <p className="max-w-2xl text-lg text-white/60">
            Plataforma end-to-end de MLOps para predição do preço das ações da NVIDIA (NVDA) 
            utilizando <span className="text-nvidia font-medium">LSTM</span> com tracking de experimentos via{" "}
            <span className="font-medium text-sky-400">MLflow</span>, otimização de hiperparâmetros com{" "}
            <span className="font-medium text-red-400">Optuna</span>, API REST com{" "}
            <span className="font-medium text-teal-400">FastAPI</span> e dashboard interativo com{" "}
            <span className="font-medium text-white">Next.js</span>.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link
              href="/predictions"
              className="flex items-center gap-2 rounded-lg bg-nvidia px-5 py-2.5 text-sm font-semibold text-black transition-all hover:bg-nvidia-dark"
            >
              <BarChart3 className="h-4 w-4" />
              Ver Previsões
              <ArrowRight className="h-4 w-4" />
            </Link>
            <a
              href="https://github.com/LucasTechAI/nvidia-mlops-platform"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 rounded-lg border border-surface-border bg-surface-hover px-5 py-2.5 text-sm font-medium text-white/70 transition-all hover:text-white"
            >
              <Github className="h-4 w-4" />
              GitHub Repository
            </a>
          </div>
        </div>
      </div>

      {/* Model Purpose */}
      <div className="rounded-xl border border-nvidia/20 bg-gradient-to-br from-nvidia/5 via-surface-card to-surface-card p-6">
        <div className="flex items-start gap-4">
          <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-xl bg-nvidia/10 text-2xl">
            🎯
          </div>
          <div>
            <h2 className="mb-2 text-lg font-semibold">Model Purpose</h2>
            <p className="text-sm leading-relaxed text-white/60">
              This model uses an <span className="font-medium text-nvidia">LSTM (Long Short-Term Memory)</span> neural network to forecast 
              the <span className="font-medium text-nvidia">Close price</span> of <span className="font-medium text-nvidia">NVIDIA (NVDA)</span> stock over the next 30 days. 
              It takes 5 input features (Open, High, Low, Close, Volume) from 60 days of history 
              to learn temporal patterns and market trends, generating predictions with confidence intervals 
              via Monte Carlo Dropout to aid in investment scenario analysis.
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              {[
                { label: "Asset", value: "NVDA" },
                { label: "Target", value: "Close Price" },
                { label: "Horizon", value: "30 days" },
                { label: "Input", value: "OHLCV (5 features)" },
                { label: "History", value: "Since 2017" },
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

      {/* About Author */}
      <div className="rounded-xl border border-surface-border bg-surface-card p-6">
        <h2 className="mb-4 text-xl font-semibold">👤 Sobre o Autor</h2>
        <div className="flex flex-col gap-6 sm:flex-row sm:items-start">
          <div className="flex h-20 w-20 flex-shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-nvidia/30 to-nvidia/10 text-4xl">
            👨‍💻
          </div>
          <div className="space-y-3">
            <div>
              <h3 className="text-lg font-semibold text-nvidia">Lucas Mendes</h3>
              <p className="text-sm text-white/40">LucasTechAI</p>
            </div>
            <p className="text-sm leading-relaxed text-white/60">
              Desenvolvedor e entusiasta de Machine Learning, cursando a Pós-Graduação em{" "}
              <span className="font-medium text-white/80">Machine Learning Engineering (MLET)</span> na{" "}
              <span className="font-medium text-red-400">FIAP</span>. Este projeto é o Tech Challenge das Fases 4 e 5, 
              demonstrando competências em Deep Learning, MLOps e engenharia de software aplicada a dados financeiros.
            </p>
            <div className="flex flex-wrap gap-3">
              <a
                href="https://github.com/LucasTechAI"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 rounded-lg bg-surface-hover px-3 py-1.5 text-xs text-white/60 transition-all hover:text-white"
              >
                <Github className="h-3.5 w-3.5" />
                @LucasTechAI
              </a>
              <a
                href="mailto:lucas.mendestech@gmail.com"
                className="flex items-center gap-2 rounded-lg bg-surface-hover px-3 py-1.5 text-xs text-white/60 transition-all hover:text-white"
              >
                <Mail className="h-3.5 w-3.5" />
                lucas.mendestech@gmail.com
              </a>
            </div>
          </div>
        </div>
      </div>

      {/* Pipeline Steps */}
      <div>
        <h2 className="mb-4 text-xl font-semibold">⚙️ Pipeline MLOps</h2>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {PIPELINE_STEPS.map((step, i) => {
            const Icon = step.icon;
            return (
              <div
                key={i}
                className="group rounded-xl border border-surface-border bg-surface-card p-5 transition-all hover:border-nvidia/30"
              >
                <div className="mb-3 flex items-center gap-3">
                  <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-nvidia/10">
                    <Icon className="h-5 w-5 text-nvidia" />
                  </div>
                  <div className="flex h-6 w-6 items-center justify-center rounded-full bg-surface-hover text-xs font-bold text-white/40">
                    {i + 1}
                  </div>
                </div>
                <h3 className="mb-1 text-sm font-semibold">{step.title}</h3>
                <p className="text-xs leading-relaxed text-white/50">{step.desc}</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Tech Stack */}
      <div>
        <h2 className="mb-4 text-xl font-semibold">🛠️ Tech Stack</h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {TECH_STACK.map((tech) => (
            <div
              key={tech.name}
              className="group rounded-xl border border-surface-border bg-surface-card p-4 transition-all hover:border-nvidia/30"
            >
              <div
                className="mb-1 h-1 w-8 rounded-full"
                style={{ backgroundColor: tech.color }}
              />
              <h3 className="text-sm font-semibold">{tech.name}</h3>
              <p className="text-xs text-white/40">{tech.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Links */}
      <div>
        <h2 className="mb-4 text-xl font-semibold">🚀 Navegação Rápida</h2>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {QUICK_LINKS.map((link) => {
            const Icon = link.icon;
            return (
              <Link
                key={link.href}
                href={link.href}
                className="group flex items-center gap-4 rounded-xl border border-surface-border bg-surface-card p-4 transition-all hover:border-nvidia/30 hover:bg-surface-hover"
              >
                <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-nvidia/10 text-lg">
                  {link.emoji}
                </div>
                <div className="flex-1">
                  <h3 className="text-sm font-semibold group-hover:text-nvidia">
                    {link.label}
                  </h3>
                  <p className="text-xs text-white/40">{link.desc}</p>
                </div>
                <ArrowRight className="h-4 w-4 text-white/20 transition-all group-hover:text-nvidia" />
              </Link>
            );
          })}
        </div>
      </div>

      {/* Architecture Diagram */}
      <div className="rounded-xl border border-surface-border bg-surface-card p-6">
        <h2 className="mb-4 text-xl font-semibold">🏗️ Arquitetura</h2>
        <div className="overflow-x-auto">
          <pre className="text-xs leading-relaxed text-white/50">
{`┌─────────────────────────────────────────────────────────────────────┐
│                        NVIDIA MLOps Platform                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │ Yahoo    │───▶│ ETL Pipeline │───▶│ SQLite DB    │              │
│  │ Finance  │    │ (yfinance)   │    │ (6700+ rows) │              │
│  └──────────┘    └──────────────┘    └──────┬───────┘              │
│                                             │                       │
│                  ┌──────────────────────────┼───────────┐           │
│                  │       Data Pipeline      │           │           │
│                  │                          ▼           │           │
│                  │  ┌──────────────┐  ┌──────────┐     │           │
│                  │  │ Preprocessing│  │ Sequence │     │           │
│                  │  │ MinMaxScaler │─▶│ Generator│     │           │
│                  │  └──────────────┘  └────┬─────┘     │           │
│                  └──────────────────────────────────────┘           │
│                                             │                       │
│   ┌─────────────┐  ┌─────────────┐  ┌──────┴───────┐              │
│   │  FastAPI    │  │  Next.js    │  │  LSTM Model  │              │
│   │  REST API   │  │  Dashboard  │  │  PyTorch     │              │
│   │  :8000      │  │  :3001      │  │  2 Layers    │              │
│   └─────────────┘  └─────────────┘  └──────────────┘              │
│                                                                     │
│   ┌─────────────┐  ┌─────────────┐  ┌──────────────┐              │
│   │  MLflow     │  │  Optuna     │  │  Docker      │              │
│   │  Tracking   │  │  HPO        │  │  Compose     │              │
│   │  :5000      │  │  50+ trials │  │  Multi-svc   │              │
│   └─────────────┘  └─────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘`}
          </pre>
        </div>
      </div>

      {/* Footer */}
      <div className="pb-4 text-center text-xs text-white/20">
        NVIDIA MLOps Platform • MIT License • FIAP Post-Tech MLET 2026
      </div>
    </div>
  );
}
