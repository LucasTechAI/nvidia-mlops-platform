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
import { PageHeader } from "@/components/page-header";

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
    title: "Testar Modelos Alternativos",
    description: "Comparar LSTM com outros modelos de séries temporais",
    category: "model",
    priority: "high",
    status: "planned",
    icon: Brain,
    details: [
      "▸ ARIMA/SARIMA — Modelos estatísticos tradicionais",
      "▸ Prophet (Meta) — Para sazonalidade e tendências",
      "▸ Transformer — Mecanismo de atenção para séries temporais",
      "▸ GRU — Alternativa leve ao LSTM",
      "▸ XGBoost/LightGBM — Modelos ensemble",
      "▸ TCN (Temporal Convolutional Networks)",
      "▸ N-BEATS — Expansão de base neural",
    ],
  },
  {
    id: "production-deploy",
    title: "Deploy em Produção",
    description: "Deploy completo com CI/CD e monitoramento",
    category: "production",
    priority: "high",
    status: "planned",
    icon: Rocket,
    details: [
      "▸ Deploy com Kubernetes e auto-scaling",
      "▸ Pipeline CI/CD (GitHub Actions / GitLab CI)",
      "▸ Estratégia de deploy Blue-Green",
      "▸ Load balancer e reverse proxy (Nginx)",
      "▸ Certificados SSL/TLS (Let's Encrypt)",
      "▸ Replicação de banco de dados e backup automatizado",
      "▸ Logging centralizado (ELK Stack / Loki)",
      "▸ Alertas e on-call (PagerDuty / Opsgenie)",
    ],
  },
  {
    id: "multi-target",
    title: "Predições Multi-Alvo",
    description: "Expandir predições além do preço de fechamento",
    category: "features",
    priority: "high",
    status: "planned",
    icon: TrendingUp,
    details: [
      "▸ Prever Máxima (máximo diário)",
      "▸ Prever Mínima (mínimo diário)",
      "▸ Prever Volume (volume de negociações)",
      "▸ Prever volatilidade intradiária",
      "▸ Multi-task learning (múltiplas saídas)",
      "▸ Intervalos de confiança para cada predição",
      "▸ Prever múltiplos horizontes (1d, 7d, 30d)",
    ],
  },
  {
    id: "news-integration",
    title: "Integração com Notícias e Papers",
    description: "Análise de sentimento e extração de insights",
    category: "research",
    priority: "medium",
    status: "planned",
    icon: FileText,
    details: [
      "▸ Web scraping de notícias financeiras (Reuters, Bloomberg)",
      "▸ News API (NewsAPI, Finnhub, Alpha Vantage)",
      "▸ Análise de sentimento com NLP (BERT, FinBERT)",
      "▸ Extração de entidades (empresas, produtos, eventos)",
      "▸ Papers científicos (arXiv, Google Scholar API)",
      "▸ Correlação entre sentimento e movimentos de preço",
      "▸ RAG com base de conhecimento de notícias e papers",
      "▸ Alertas automáticos para notícias relevantes",
    ],
  },
  {
    id: "access-control",
    title: "Controle de Acesso e Autenticação",
    description: "Login Google Auth, papéis e permissões por usuário",
    category: "production",
    priority: "high",
    status: "planned",
    icon: Shield,
    details: [
      "▸ Login Google OAuth 2.0 (NextAuth.js / Firebase Auth)",
      "▸ SSO empresarial (Single Sign-On) via SAML/OIDC",
      "▸ Papéis: Admin, Analista, Visualizador, Somente API",
      "▸ Permissões granulares por página e recurso",
      "▸ Admin vê tudo: logs, modelo, configurações, agente",
      "▸ Analista vê predições, dados, explicabilidade",
      "▸ Visualizador vê apenas dashboard e predições",
      "▸ Chaves de API com escopos e rate limits por usuário",
      "▸ Trilha de auditoria: quem acessou o quê e quando",
      "▸ Gerenciamento de sessão com JWT + refresh tokens",
      "▸ 2FA opcional (Autenticação de Dois Fatores)",
      "▸ Página de admin para gerenciar usuários e papéis",
    ],
  },
  {
    id: "microservices",
    title: "Arquitetura de Microsserviços",
    description: "Dividir o monólito em serviços independentes e escaláveis",
    category: "production",
    priority: "high",
    status: "planned",
    icon: GitBranch,
    details: [
      "▸ Dividir em repositórios: api-gateway, model-service, evaluation-service, agent-service, dashboard",
      "▸ API Gateway (Kong / Traefik) para roteamento, rate limiting e autenticação",
      "▸ Model Service: treinamento, inferência, registro (gRPC para baixa latência)",
      "▸ Evaluation Service: RAGAS, LLM-Judge, gestão do golden set",
      "▸ Agent Service: agente ReAct, RAG, ChromaDB vector store",
      "▸ ETL Service: ingestão de dados, feature engineering, agendamento",
      "▸ Monitoring Service: Prometheus, Grafana, SLA, detecção de drift",
      "▸ Dashboard: app Next.js standalone consumindo todas as APIs dos serviços",
      "▸ Contratos compartilhados via OpenAPI specs + Protobuf schemas",
      "▸ Pipelines CI/CD independentes por serviço (GitHub Actions)",
      "▸ Docker Compose para dev local, Kubernetes (Helm charts) para produção",
      "▸ Service mesh (Istio / Linkerd) para observabilidade e mTLS",
      "▸ Comunicação orientada a eventos via Redis Streams ou Kafka",
      "▸ Escalamento independente: nós GPU para modelo, CPU para API/dashboard",
    ],
  },
  {
    id: "multi-asset",
    title: "Cobertura Multi-Ativo Big Tech",
    description: "Expandir além da NVIDIA para cobrir as principais ações de tecnologia",
    category: "features",
    priority: "high",
    status: "planned",
    icon: TrendingUp,
    details: [
      "▸ Add MSFT (Microsoft), GOOGL (Alphabet), META (Meta), AAPL (Apple), AMZN (Amazon)",
      "▸ TSLA (Tesla), AMD, INTC (Intel), TSM (TSMC), AVGO (Broadcom)",
      "▸ Seletor de ativos no dashboard — alternar entre ações ou ver todas",
      "▸ Pipeline ETL unificado: Yahoo Finance / Alpha Vantage / Polygon.io para todos os tickers",
      "▸ Um modelo LSTM por ativo com arquitetura compartilhada e pesos independentes",
      "▸ Matriz de correlação entre ativos (heatmap) para análise de portfólio",
      "▸ Comparação setorial: GPU (NVDA vs AMD) / Cloud (MSFT vs GOOGL vs AMZN)",
      "▸ Simulação de portfólio: alocar pesos, backtest, Sharpe & drawdown",
      "▸ Índice de força relativa (RSI) e momentum em todos os ativos",
      "▸ RAG do agente atualizado com base de conhecimento multi-empresa",
      "▸ Rastreamento de P&L combinado para portfólio multi-ativo",
      "▸ Alertas: divergência entre ativos, volume incomum, calendário de resultados",
      "▸ Comparação com benchmark S&P 500 (SPY) e Nasdaq (QQQ)",
    ],
  },
  {
    id: "cost-control",
    title: "Controle de Custos de Infraestrutura",
    description: "Monitorar, otimizar e reduzir custos de cloud e computação",
    category: "production",
    priority: "high",
    status: "planned",
    icon: DollarSign,
    details: [
      "▸ Dashboard de custos em tempo real: horas GPU, armazenamento, rede, chamadas API",
      "▸ Alertas de orçamento: limites configuráveis por serviço (Slack/email)",
      "▸ Rastreamento de uso de GPU — detecção de tempo ocioso e desligamento automático",
      "▸ Estratégia de spot instances: fallback para on-demand com comparação de custos",
      "▸ Rastreamento de custo por predição na inferência e otimização",
      "▸ Analytics de uso de tokens LLM: custo por query, tendências diárias/semanais",
      "▸ Políticas de ciclo de vida de storage: auto-arquivar checkpoints e logs antigos",
      "▸ Recomendações de right-sizing: CPU/RAM/GPU baseado no uso real",
      "▸ Comparação de custos multi-cloud (AWS vs GCP vs Azure vs on-prem)",
      "▸ Relatórios FinOps: detalhamento mensal por equipe, projeto, modelo",
      "▸ Atribuição de custo por estágio do pipeline (ETL, treinamento, inferência, monitoramento)",
      "▸ Escalamento automatizado de recursos baseado em padrões de demanda de predição",
    ],
  },
  {
    id: "productization",
    title: "Produtização & Modelo de Negócio",
    description: "Transformar a plataforma em um produto SaaS comercial",
    category: "production",
    priority: "medium",
    status: "planned",
    icon: Briefcase,
    details: [
      "▸ Arquitetura multi-tenant: ambientes isolados por cliente",
      "▸ Planos de assinatura: Free (1 ação), Pro (10 ações), Enterprise (ilimitado)",
      "▸ Autenticação e autorização de usuários (OAuth2, SSO, RBAC)",
      "▸ Gestão de chaves API: rate limiting, cotas de uso, cobrança por chamada",
      "▸ Dashboard white-label: branding personalizado, logos, temas de cores",
      "▸ Garantias de SLA: 99.9% uptime, <200ms de latência de predição",
      "▸ Onboarding self-service: assistente guiado de configuração para novos clientes",
      "▸ Analytics de uso e dashboard de faturamento para clientes",
      "▸ Integração com marketplace: AWS Marketplace, GCP Marketplace",
      "▸ Rastreamento de receita: métricas MRR, churn, LTV, CAC",
      "▸ Certificações de conformidade: SOC2, ISO 27001, LGPD/GDPR",
      "▸ Portal de documentação: docs de API, tutoriais, SDK (Python/JS)",
    ],
  },
  {
    id: "integrations",
    title: "Integrações de Mensagens e Notificações",
    description: "Entregar alertas e predições via Email, WhatsApp e Telegram",
    category: "features",
    priority: "high",
    status: "planned",
    icon: MessageCircle,
    details: [
      "▸ Telegram Bot: predições diárias, alertas, comandos de resumo de portfólio",
      "▸ WhatsApp Business API: entrega automática de predições para assinantes",
      "▸ Relatórios por email: PDF diário/semanal agendado com gráficos e insights",
      "▸ Regras de alerta personalizadas: limite de preço, drift detectado, modelo retreinado",
      "▸ Roteamento multi-canal: usuário escolhe canal preferido por tipo de alerta",
      "▸ Comandos interativos no Telegram: /predict NVDA, /portfolio, /status",
      "▸ Botões de resposta rápida no WhatsApp para confirmações de compra/venda/manutenção",
      "▸ Digest de email: resumo semanal de desempenho com P&L e acurácia",
      "▸ Integração com Slack para alertas de equipe e notificações de incidentes",
      "▸ Push notifications: suporte PWA mobile com service workers",
      "▸ Suporte a webhook: callbacks HTTP genéricos para integrações customizadas",
      "▸ Dashboard de preferências de notificação: frequência, canais, horário silencioso",
    ],
  },
];

const CATEGORIES = [
  { id: "all", label: "Todos", icon: Target },
  { id: "model", label: "Modelos", icon: Brain },
  { id: "production", label: "Produção", icon: Rocket },
  { id: "features", label: "Funcionalidades", icon: Settings },
  { id: "data", label: "Dados", icon: Database },
  { id: "research", label: "Pesquisa", icon: FileText },
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
    <div className="relative mx-auto max-w-7xl space-y-6">
      {/* Header */}
      <PageHeader
        label="Evolução · Roadmap"
        title="Próximos"
        gradient="Passos"
        subtitle="Roadmap, melhorias planejadas e próximas funcionalidades."
        icon={Rocket}
      />

      {/* Category Filter */}
      <div className="flex justify-center">
        <div className="flex gap-1 rounded-xl border border-surface-border bg-surface-card p-1">
          {CATEGORIES.map((cat) => {
            const Icon = cat.icon;
            return (
              <button
                key={cat.id}
                onClick={() => setSelectedCategory(cat.id)}
                className={`flex items-center justify-center gap-2 rounded-lg px-3 py-2.5 text-xs font-medium transition ${
                  selectedCategory === cat.id
                    ? "bg-nvidia/20 text-nvidia"
                    : "text-white/50 hover:bg-surface-hover hover:text-white"
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                {cat.label}
              </button>
            );
          })}
        </div>
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
          <p className="text-white/40">Nenhum item encontrado nesta categoria</p>
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
                {step.priority === "high" && <><span className="inline-block h-2 w-2 rounded-full bg-red-400" /> Alta</>}
                {step.priority === "medium" && <><span className="inline-block h-2 w-2 rounded-full bg-yellow-400" /> Média</>}
                {step.priority === "low" && <><span className="inline-block h-2 w-2 rounded-full bg-blue-400" /> Baixa</>}
              </span>
              <span className="rounded-full border border-surface-border bg-surface-hover px-2.5 py-1 text-xs text-white/70">
                {step.status === "completed" && <><CheckCircle2 className="inline h-3.5 w-3.5" /> Concluído</>}
                {step.status === "in-progress" && <><Clock className="inline h-3.5 w-3.5" /> Em Andamento</>}
                {step.status === "planned" && <><ClipboardList className="inline h-3.5 w-3.5" /> Planejado</>}
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
        {isExpanded ? "▲ Clique para recolher" : "▼ Clique para expandir detalhes"}
      </div>
    </div>
  );
}
