"use client";

import { useState } from "react";
import {
  CheckCircle2,
  Circle,
  Rocket,
  Brain,
  TrendingUp,
  FileText,
  Settings,
  Database,
  Target,
  BarChart3,
  Shield,
  Clock,
  ClipboardList,
  GitBranch,
  DollarSign,
  Briefcase,
  MessageCircle,
  Mail,
} from "lucide-react";

interface NextStep {
  id: string;
  title: string;
  description: string;
  category: "model" | "production" | "features" | "data" | "research";
  priority: "high" | "medium" | "low";
  status: "planned" | "in-progress" | "completed";
  icon: React.ElementType;
  details: string[];
}

const NEXT_STEPS: NextStep[] = [
  {
    id: "test-models",
    title: "Test Alternative Models",
    description: "Compare LSTM with other time-series models",
    category: "model",
    priority: "high",
    status: "planned",
    icon: Brain,
    details: [
      "▸ ARIMA/SARIMA — Traditional statistical models",
      "▸ Prophet (Meta) — For seasonality and trends",
      "▸ Transformer — Attention mechanism for time series",
      "▸ GRU — Lightweight alternative to LSTM",
      "▸ XGBoost/LightGBM — Ensemble models",
      "▸ TCN (Temporal Convolutional Networks)",
      "▸ N-BEATS — Neural basis expansion",
    ],
  },
  {
    id: "production-deploy",
    title: "Production Deployment",
    description: "Full deploy with CI/CD and monitoring",
    category: "production",
    priority: "high",
    status: "planned",
    icon: Rocket,
    details: [
      "▸ Kubernetes deployment with auto-scaling",
      "▸ CI/CD pipeline (GitHub Actions / GitLab CI)",
      "▸ Blue-Green deployment strategy",
      "▸ Load balancer & reverse proxy (Nginx)",
      "▸ SSL/TLS certificates (Let's Encrypt)",
      "▸ Database replication & automated backup",
      "▸ Aggregated logging (ELK Stack / Loki)",
      "▸ Alerting & on-call (PagerDuty / Opsgenie)",
    ],
  },
  {
    id: "multi-target",
    title: "Multi-Target Predictions",
    description: "Expand predictions beyond closing price",
    category: "features",
    priority: "high",
    status: "planned",
    icon: TrendingUp,
    details: [
      "▸ Predict High (daily maximum)",
      "▸ Predict Low (daily minimum)",
      "▸ Predict Volume (trading volume)",
      "▸ Predict intraday volatility",
      "▸ Multi-task learning (multiple outputs)",
      "▸ Confidence intervals for each prediction",
      "▸ Predict multiple horizons (1d, 7d, 30d)",
    ],
  },
  {
    id: "news-integration",
    title: "News & Research Papers Integration",
    description: "Sentiment analysis and insight extraction",
    category: "research",
    priority: "medium",
    status: "planned",
    icon: FileText,
    details: [
      "▸ Web scraping of financial news (Reuters, Bloomberg)",
      "▸ News API (NewsAPI, Finnhub, Alpha Vantage)",
      "▸ Sentiment analysis with NLP (BERT, FinBERT)",
      "▸ Entity extraction (companies, products, events)",
      "▸ Scientific papers (arXiv, Google Scholar API)",
      "▸ Correlation between sentiment and price movements",
      "▸ RAG with news & research papers knowledge base",
      "▸ Automated alerts for relevant news",
    ],
  },
  {
    id: "access-control",
    title: "Access Control & Authentication",
    description: "Google Auth login, roles and per-user permissions",
    category: "production",
    priority: "high",
    status: "planned",
    icon: Shield,
    details: [
      "▸ Google OAuth 2.0 login (NextAuth.js / Firebase Auth)",
      "▸ Enterprise SSO (Single Sign-On) via SAML/OIDC",
      "▸ Roles: Admin, Analyst, Viewer, API-only",
      "▸ Granular permissions per page and resource",
      "▸ Admin sees everything: logs, model, settings, agent",
      "▸ Analyst sees predictions, data, explainability",
      "▸ Viewer sees dashboard and predictions only",
      "▸ API keys with scopes and rate limits per user",
      "▸ Audit trail: who accessed what and when",
      "▸ Session management with JWT + refresh tokens",
      "▸ Optional 2FA (Two-Factor Authentication)",
      "▸ Admin page for managing users and roles",
    ],
  },
  {
    id: "microservices",
    title: "Microservices Architecture",
    description: "Break the monolith into independent, scalable services",
    category: "production",
    priority: "high",
    status: "planned",
    icon: GitBranch,
    details: [
      "▸ Split into separate repos: api-gateway, model-service, evaluation-service, agent-service, dashboard",
      "▸ API Gateway (Kong / Traefik) for routing, rate limiting, and auth",
      "▸ Model Service: training, inference, registry (gRPC for low latency)",
      "▸ Evaluation Service: RAGAS, LLM-Judge, golden set management",
      "▸ Agent Service: ReAct agent, RAG, ChromaDB vector store",
      "▸ ETL Service: data ingestion, feature engineering, scheduling",
      "▸ Monitoring Service: Prometheus, Grafana, SLA, drift detection",
      "▸ Dashboard: standalone Next.js app consuming all service APIs",
      "▸ Shared contracts via OpenAPI specs + Protobuf schemas",
      "▸ Independent CI/CD pipelines per service (GitHub Actions)",
      "▸ Docker Compose for local dev, Kubernetes (Helm charts) for prod",
      "▸ Service mesh (Istio / Linkerd) for observability and mTLS",
      "▸ Event-driven communication via Redis Streams or Kafka",
      "▸ Independent scaling: GPU nodes for model, CPU for API/dashboard",
    ],
  },
  {
    id: "multi-asset",
    title: "Multi-Asset Big Tech Coverage",
    description: "Expand beyond NVIDIA to cover all major tech stocks",
    category: "features",
    priority: "high",
    status: "planned",
    icon: TrendingUp,
    details: [
      "▸ Add MSFT (Microsoft), GOOGL (Alphabet), META (Meta), AAPL (Apple), AMZN (Amazon)",
      "▸ TSLA (Tesla), AMD, INTC (Intel), TSM (TSMC), AVGO (Broadcom)",
      "▸ Asset selector in dashboard — switch between stocks or view all",
      "▸ Unified ETL pipeline: Yahoo Finance / Alpha Vantage / Polygon.io for all tickers",
      "▸ One LSTM model per asset with shared architecture, independent weights",
      "▸ Cross-asset correlation matrix (heatmap) for portfolio analysis",
      "▸ Sector comparison: GPU (NVDA vs AMD) / Cloud (MSFT vs GOOGL vs AMZN)",
      "▸ Portfolio simulation: allocate weights, backtest, Sharpe & drawdown",
      "▸ Relative strength index (RSI) and momentum across all assets",
      "▸ Agent RAG updated with multi-company knowledge base",
      "▸ Combined P&L tracking for multi-asset portfolio",
      "▸ Alerts: cross-asset divergence, unusual volume, earnings calendar",
      "▸ Benchmark comparison vs S&P 500 (SPY) and Nasdaq (QQQ)",
    ],
  },
  {
    id: "cost-control",
    title: "Infrastructure Cost Control",
    description: "Monitor, optimize and reduce cloud and compute costs",
    category: "production",
    priority: "high",
    status: "planned",
    icon: DollarSign,
    details: [
      "▸ Real-time cost dashboard: GPU hours, storage, network, API calls",
      "▸ Budget alerts: configurable thresholds per service (Slack/email)",
      "▸ GPU utilization tracking — idle time detection & auto-shutdown",
      "▸ Spot instance strategy: fallback to on-demand with cost comparison",
      "▸ Model inference cost-per-prediction tracking and optimization",
      "▸ LLM token usage analytics: cost per query, daily/weekly trends",
      "▸ Storage lifecycle policies: auto-archive old checkpoints & logs",
      "▸ Right-sizing recommendations: CPU/RAM/GPU based on actual usage",
      "▸ Multi-cloud cost comparison (AWS vs GCP vs Azure vs on-prem)",
      "▸ FinOps reports: monthly cost breakdown by team, project, model",
      "▸ Cost attribution per pipeline stage (ETL, training, inference, monitoring)",
      "▸ Automated resource scaling based on prediction demand patterns",
    ],
  },
  {
    id: "productization",
    title: "Productization & Business Model",
    description: "Transform the platform into a commercial SaaS product",
    category: "production",
    priority: "medium",
    status: "planned",
    icon: Briefcase,
    details: [
      "▸ Multi-tenant architecture: isolated environments per customer",
      "▸ Subscription tiers: Free (1 stock), Pro (10 stocks), Enterprise (unlimited)",
      "▸ User authentication & authorization (OAuth2, SSO, RBAC)",
      "▸ API key management: rate limiting, usage quotas, billing per call",
      "▸ White-label dashboard: custom branding, logos, color themes",
      "▸ SLA guarantees: 99.9% uptime, <200ms prediction latency",
      "▸ Self-service onboarding: guided setup wizard for new customers",
      "▸ Usage analytics & billing dashboard for customers",
      "▸ Marketplace integration: AWS Marketplace, GCP Marketplace",
      "▸ Revenue tracking: MRR, churn, LTV, CAC metrics",
      "▸ Compliance certifications: SOC2, ISO 27001, LGPD/GDPR",
      "▸ Documentation portal: API docs, tutorials, SDK (Python/JS)",
    ],
  },
  {
    id: "integrations",
    title: "Messaging & Notification Integrations",
    description: "Deliver alerts and predictions via Email, WhatsApp and Telegram",
    category: "features",
    priority: "high",
    status: "planned",
    icon: MessageCircle,
    details: [
      "▸ Telegram Bot: daily predictions, alerts, portfolio summary commands",
      "▸ WhatsApp Business API: automated prediction delivery to subscribers",
      "▸ Email reports: scheduled daily/weekly PDF with charts and insights",
      "▸ Custom alert rules: price threshold, drift detected, model retrained",
      "▸ Multi-channel routing: user picks preferred channel per alert type",
      "▸ Interactive Telegram commands: /predict NVDA, /portfolio, /status",
      "▸ WhatsApp quick-reply buttons for buy/sell/hold confirmations",
      "▸ Email digest: weekly performance summary with P&L and accuracy",
      "▸ Slack integration for team-based alerts and incident notifications",
      "▸ Push notifications: mobile PWA support with service workers",
      "▸ Webhook support: generic HTTP callbacks for custom integrations",
      "▸ Notification preferences dashboard: frequency, channels, quiet hours",
    ],
  },
];

const CATEGORIES = [
  { id: "all", label: "All", icon: Target },
  { id: "model", label: "Models", icon: Brain },
  { id: "production", label: "Production", icon: Rocket },
  { id: "features", label: "Features", icon: Settings },
  { id: "data", label: "Data", icon: Database },
  { id: "research", label: "Research", icon: FileText },
];

const PRIORITY_COLORS = {
  high: "border-red-500/30 bg-red-500/10 text-red-400",
  medium: "border-yellow-500/30 bg-yellow-500/10 text-yellow-400",
  low: "border-blue-500/30 bg-blue-500/10 text-blue-400",
};

const STATUS_COLORS = {
  planned: "text-white/40",
  "in-progress": "text-nvidia",
  completed: "text-green-400",
};

export default function NextStepsPage() {
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const PRIORITY_ORDER: Record<string, number> = { high: 0, medium: 1, low: 2 };

  const filteredSteps = (
    selectedCategory === "all"
      ? NEXT_STEPS
      : NEXT_STEPS.filter((step) => step.category === selectedCategory)
  ).sort((a, b) => PRIORITY_ORDER[a.priority] - PRIORITY_ORDER[b.priority]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="flex items-center gap-2 text-2xl font-semibold"><Rocket className="h-6 w-6 text-nvidia" /> Next Steps</h2>
        <p className="mt-1 text-sm text-white/50">
          Roadmap, planned improvements and upcoming features
        </p>
      </div>

      {/* Category Filter */}
      <div className="flex gap-2 overflow-x-auto pb-2">
        {CATEGORIES.map((cat) => (
          <button
            key={cat.id}
            onClick={() => setSelectedCategory(cat.id)}
            className={`flex items-center gap-2 whitespace-nowrap rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
              selectedCategory === cat.id
                ? "bg-nvidia text-black"
                : "bg-surface-hover text-white/70 hover:bg-surface-border"
            }`}
          >
            <cat.icon className="h-4 w-4" />
            {cat.label}
          </button>
        ))}
      </div>

      {/* Next Steps List */}
      <div className="grid gap-4 md:grid-cols-2">
        {filteredSteps.map((step) => (
          <StepCard
            key={step.id}
            step={step}
            isExpanded={expandedId === step.id}
            onToggle={() => setExpandedId(expandedId === step.id ? null : step.id)}
          />
        ))}
      </div>

      {filteredSteps.length === 0 && (
        <div className="rounded-xl border border-surface-border bg-surface-card p-12 text-center">
          <p className="text-white/40">No items found in this category</p>
        </div>
      )}
    </div>
  );
}

/* ──────────── Step Card ──────────── */
interface StepCardProps {
  step: NextStep;
  isExpanded: boolean;
  onToggle: () => void;
}

function StepCard({ step, isExpanded, onToggle }: StepCardProps) {
  const Icon = step.icon;
  const StatusIcon = step.status === "completed" ? CheckCircle2 : Circle;

  return (
    <div
      className="cursor-pointer rounded-xl border border-surface-border bg-surface-card transition-all hover:border-nvidia/30"
      onClick={onToggle}
    >
      <div className="p-6">
        {/* Header */}
        <div className="flex items-start gap-4">
          <div className="rounded-lg bg-nvidia/20 p-3">
            <Icon className="h-6 w-6 text-nvidia" />
          </div>
          <div className="flex-1">
            <div className="flex items-start justify-between gap-2">
              <h3 className="text-lg font-semibold">{step.title}</h3>
              <StatusIcon className={`h-5 w-5 ${STATUS_COLORS[step.status]}`} />
            </div>
            <p className="mt-1 text-sm text-white/50">{step.description}</p>

            {/* Tags */}
            <div className="mt-3 flex flex-wrap gap-2">
              <span
                className={`rounded-full px-2.5 py-1 text-xs font-medium ${
                  PRIORITY_COLORS[step.priority]
                }`}
              >
                {step.priority === "high" && <><span className="inline-block h-2 w-2 rounded-full bg-red-400" /> High</>}
                {step.priority === "medium" && <><span className="inline-block h-2 w-2 rounded-full bg-yellow-400" /> Medium</>}
                {step.priority === "low" && <><span className="inline-block h-2 w-2 rounded-full bg-blue-400" /> Low</>}
              </span>
              <span className="rounded-full border border-surface-border bg-surface-hover px-2.5 py-1 text-xs text-white/70">
                {step.status === "completed" && <><CheckCircle2 className="inline h-3.5 w-3.5" /> Completed</>}
                {step.status === "in-progress" && <><Clock className="inline h-3.5 w-3.5" /> In Progress</>}
                {step.status === "planned" && <><ClipboardList className="inline h-3.5 w-3.5" /> Planned</>}
              </span>
            </div>
          </div>
        </div>

        {/* Expanded Details */}
        {isExpanded && (
          <div className="mt-4 space-y-2 border-t border-surface-border pt-4">
            {step.details.map((detail, idx) => (
              <div key={idx} className="flex items-start gap-2 text-sm text-white/70">
                <span className="mt-0.5">{detail}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Expand Indicator */}
      <div className="border-t border-surface-border px-6 py-3 text-center text-xs text-white/40">
        {isExpanded ? "▲ Click to collapse" : "▼ Click to expand details"}
      </div>
    </div>
  );
}
