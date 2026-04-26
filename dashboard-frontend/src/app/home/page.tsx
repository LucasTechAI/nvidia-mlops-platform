"use client";

import { useEffect, useState } from "react";
import Script from "next/script";
import { api } from "@/lib/api";
import "./design-system.css";

type BtPoint = { date: string; actual: number; predicted: number };
type ModelEntry = { version: string; name: string; role: string; r2: number; pct: number; mape: string; color: string; crown: boolean };

const DEFAULT_MODELS: ModelEntry[] = [
  { version: 'V2', name: 'nvda-lstm-v2', role: 'champion',   r2: 0.940, pct: 94.0, mape: '2.1%', color: '#14b8a6', crown: true  },
  { version: 'V3', name: 'nvda-lstm-v3', role: 'challenger', r2: 0.935, pct: 93.5, mape: '2.2%', color: '#2dd4bf', crown: false },
  { version: 'V1', name: 'nvda-lstm-v1', role: 'baseline',   r2: 0.910, pct: 91.0, mape: '2.8%', color: '#0d9488', crown: false },
];

function buildPath(pts: BtPoint[], key: "actual" | "predicted", yMin: number, yMax: number): string {
  if (!pts.length) return "";
  const W = 400, H = 140;
  return pts
    .map((p, i) => {
      const x = (i / (pts.length - 1)) * W;
      const y = 10 + (1 - (p[key] - yMin) / (yMax - yMin)) * H;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export default function HomePage() {
  const [btData, setBtData] = useState<BtPoint[]>([]);
  const [btMae, setBtMae] = useState<number | null>(null);
  const [btR2, setBtR2] = useState<number | null>(null);
  const [models, setModels] = useState<ModelEntry[]>(DEFAULT_MODELS);
  const [faith, setFaith] = useState(0.46);
  const [relev, setRelev] = useState(0.57);

  useEffect(() => {
    const STAGE_MAP: Record<string, { role: string; color: string; crown: boolean }> = {
      Production: { role: 'champion',   color: '#14b8a6', crown: true  },
      Staging:    { role: 'challenger', color: '#2dd4bf', crown: false },
      Archived:   { role: 'baseline',  color: '#0d9488', crown: false },
    };
    api.modelRegistry.versions("nvidia-lstm-forecast").then((d) => {
      const versions = d.versions;
      if (!versions?.length) return;
      const ORDER: Record<string, number> = { Production: 0, Staging: 1, Archived: 2 };
      const mapped: ModelEntry[] = versions
        .filter(v => STAGE_MAP[v.stage])
        .sort((a, b) => (ORDER[a.stage] ?? 3) - (ORDER[b.stage] ?? 3))
        .map(v => {
          const info = STAGE_MAP[v.stage];
          const r2   = v.metrics?.r2   ?? 0;
          const mape = v.metrics?.mape ?? 0;
          return { version: `V${v.version}`, name: `nvda-lstm-v${v.version}`, role: info.role, r2, pct: +(r2 * 100).toFixed(1), mape: `${mape.toFixed(1)}%`, color: info.color, crown: info.crown };
        });
      if (mapped.length) setModels(mapped);
    }).catch(() => {});
    api.evaluation.llmResults().then((d) => {
      const m = (d as { ragas?: { metrics?: { faithfulness?: number; answer_relevancy?: number } } }).ragas?.metrics;
      if (m?.faithfulness    !== undefined) setFaith(m.faithfulness);
      if (m?.answer_relevancy !== undefined) setRelev(m.answer_relevancy);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    api.predict.backtest(60).then((d) => {
      const bt = (d as { backtest: BtPoint[] }).backtest;
      if (!bt?.length) return;
      setBtData(bt);
      const mae = bt.reduce((s, x) => s + Math.abs(x.actual - x.predicted), 0) / bt.length;
      const mean = bt.reduce((s, x) => s + x.actual, 0) / bt.length;
      const r2 = 1 - bt.reduce((s, x) => s + (x.actual - x.predicted) ** 2, 0) /
                     bt.reduce((s, x) => s + (x.actual - mean) ** 2, 0);
      setBtMae(mae);
      setBtR2(r2);
    }).catch(() => {});
  }, []);
  useEffect(() => {
    const root = document.querySelector(".ds-scope");
    if (!root) return;
    root
      .querySelectorAll(".reveal, .reveal-left, .reveal-right, .stagger-up")
      .forEach((el) => el.classList.add("active"));

    root.querySelectorAll("#sobre").forEach((section) => {
      if (section.querySelector(".bg-video")) return;
      const video = document.createElement("video");
      video.className = "bg-video";
      video.autoplay = true;
      video.muted = true;
      video.loop = true;
      video.playsInline = true;
      video.style.cssText =
        "position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:0.13;z-index:0;pointer-events:none;";
      const source = document.createElement("source");
      source.src = "/fintech-bg.mp4";
      source.type = "video/mp4";
      video.appendChild(source);
      section.insertBefore(video, section.firstChild);
    });
  }, []);

  const champion = models.find(m => m.role === 'champion') ?? models[0];

  return (
    <div className="ds-scope -m-8">
      {/* ============================================================
          TOP NAV — índice do pitch
          ============================================================ */}
      <nav>
        <div className="nav-logo">NVDA · MLOps</div>
        <ul className="nav-links">
          <li><a href="#hero">Capa</a></li>
          <li><a href="#sobre">Sobre</a></li>
          <li><a href="#checklist">Checklist</a></li>
          <li><a href="#problema">Problema</a></li>
          <li><a href="#arquitetura">Arquitetura</a></li>
          <li><a href="#resultados">Resultados</a></li>
          <li><a href="#diferenciais">Diferenciais</a></li>
          <li><a href="#demo">Demo</a></li>
          <li><a href="#conclusao">Conclusão</a></li>
        </ul>
        <div className="nav-cta">
          <a className="btn-nav btn-nav-ghost" href="https://github.com/LucasTechAI/nvidia-mlops-platform" target="_blank" rel="noreferrer">GitHub</a>
          <a className="btn-nav btn-nav-solid" href="/predictions">Abrir dashboard</a>
        </div>
        <div className="nav-hamburger" id="hamburger">
          <span></span><span></span><span></span>
        </div>
      </nav>

      {/* ============================================================
          0) HERO — Capa do pitch
          ============================================================ */}
      <section id="hero">
        <canvas id="heroCanvas"></canvas>
        <div className="hero-orb hero-orb-1"></div>
        <div className="hero-orb hero-orb-2"></div>
        <div className="hero-orb hero-orb-3"></div>
        <div className="hero-stars">
          <div className="hero-star hero-star-1"></div>
          <div className="hero-star hero-star-2"></div>
          <div className="hero-star hero-star-3"></div>
          <div className="hero-star hero-star-4"></div>
          <div className="hero-star hero-star-5"></div>
          <div className="hero-star hero-star-6"></div>
        </div>
        <div className="hero-orbits">
          <div className="hero-orbit hero-orbit-1"><div className="hero-orbit-dot"></div></div>
          <div className="hero-orbit hero-orbit-2"><div className="hero-orbit-dot"></div></div>
          <div className="hero-orbit hero-orbit-3"><div className="hero-orbit-dot"></div></div>
        </div>
        <div className="hero-content">
          <div className="hero-eyebrow">
            <span className="iconify" data-icon="lucide:graduation-cap"></span>
            FIAP Post-Tech MLET · Tech Challenge Fase 5
          </div>
          <h1 className="hero-title">
            Plataforma MLOps para<br />
            <span className="text-gradient">forecasting de NVDA</span>
          </h1>
          <p className="hero-subtitle">
            Pipeline reproduzível e container-native que treina, versiona, serve e monitora um modelo LSTM para prever o preço de fechamento da NVIDIA — com agente conversacional, RAG, observabilidade e SLA de produção.
          </p>
          <div className="hero-ctas">
            <a className="btn-hero-primary" href="#problema">
              <span className="iconify" data-icon="lucide:play"></span>
              Iniciar pitch
            </a>
            <a className="btn-hero-ghost" href="/predictions">
              <span className="iconify" data-icon="lucide:layout-dashboard"></span>
              Abrir dashboard
            </a>
          </div>
          <div className="hero-metrics">
            <div className="hero-metric">
              <span className="hero-metric-val">{champion.r2.toFixed(3)}</span>
              <span className="hero-metric-label">R² score</span>
            </div>
            <div className="hero-metric-sep"></div>
            <div className="hero-metric">
              <span className="hero-metric-val">30d</span>
              <span className="hero-metric-label">Forecast horizon</span>
            </div>
            <div className="hero-metric-sep"></div>
            <div className="hero-metric">
              <span className="hero-metric-val">6.7K</span>
              <span className="hero-metric-label">Training rows</span>
            </div>
            <div className="hero-metric-sep"></div>
            <div className="hero-metric">
              <span className="hero-metric-val">66</span>
              <span className="hero-metric-label">API endpoints</span>
            </div>
          </div>
        </div>
      </section>

      {/* ============================================================
          SOBRE MIM
          ============================================================ */}
      <section className="section section-alt" id="sobre">
        <div className="section-sep"></div>
        <div className="container">
          <div className="section-header reveal">
            <span className="section-label">Sobre mim</span>
            <h2>Quem está <span className="text-gradient">por trás do projeto</span></h2>
          </div>
          <div className="bento-grid stagger-up" style={{gridTemplateColumns: '1fr 2fr'}}>
            <div className="bento-card" style={{display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '20px', textAlign: 'center', padding: '40px 24px'}}>
              <img src="/lucas.png" alt="Lucas" style={{width:'96px',height:'96px',borderRadius:'50%',objectFit:'cover',objectPosition:'top center',border:'3px solid rgba(20,184,166,0.4)',flexShrink:0}} />
              <div>
                <h3 style={{marginBottom: '6px'}}>Lucas</h3>
                <p style={{color: 'var(--accent)', fontSize: '14px', fontWeight: 600}}>FIAP Post-Tech MLET</p>
                <p style={{color: 'var(--muted)', fontSize: '13px', marginTop: '4px'}}>Tech Challenge · Fase 5</p>
              </div>
              <p style={{fontSize:'12px',color:'rgba(255,255,255,0.45)',lineHeight:1.7,textAlign:'center',maxWidth:'220px'}}>
                Profissional de dados com 5+ anos em fintech e legaltech. Especializado em LLMs, Product Analytics e automação em nuvem (AWS/GCP). Atualmente na <span style={{color:'rgba(20,184,166,0.9)',fontWeight:600}}>CloudWalk</span>, construindo dashboards, KPIs e pipelines de IA para times de Produto, Risco e Growth.
              </p>
              <div style={{display: 'flex', gap: '12px'}}>
                <a className="btn-nav btn-nav-ghost" href="https://github.com/LucasTechAI" target="_blank" rel="noreferrer"><span className="iconify" data-icon="lucide:github"></span></a>
                <a className="btn-nav btn-nav-ghost" href="https://www.linkedin.com/in/lucas-mendes-barbosa/" target="_blank" rel="noreferrer"><span className="iconify" data-icon="lucide:linkedin"></span></a>
              </div>
            </div>
            <div className="bento-card" style={{display:'flex', flexDirection:'column'}}>
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Contexto &amp; motivação</span>
                <span className="bc-header-tag">por que NVDA</span>
              </div>
              <div className="bc-body" style={{flex:1, display:'flex', flexDirection:'column', justifyContent:'center'}}>
                <h3>Engenharia de ML de ponta a ponta</h3>
                <p>Minha motivação com este projeto foi provar, na prática, que os conceitos de MLOps não são exclusivos de grandes empresas. Com ferramentas open-source, é possível construir um pipeline reproduzível, monitorado e auditável — do dado bruto ao agente conversacional — em um único repositório.</p>
                <p style={{marginTop: '16px'}}>Escolhi NVDA pela volatilidade extrema e pela relevância no ciclo de IA: um ativo que desafia modelos estáticos e exige retreino contínuo, monitoramento de drift e rastreabilidade total das decisões do modelo.</p>
                <div className="widget-integrations" style={{marginTop: '24px'}}>
                  <span className="int-badge">MLOps</span>
                  <span className="int-badge">Deep Learning</span>
                  <span className="int-badge">LLMOps</span>
                  <span className="int-badge">Séries Temporais</span>
                  <span className="int-badge">FinTech</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============================================================
          DEMO DAY CHECKLIST
          ============================================================ */}
      <section className="section" id="checklist">
        <div className="section-sep"></div>
        <div className="container">
          <div className="section-header reveal">
            <span className="section-label">Demo Day · Checklist</span>
            <h2>Todos os critérios <span className="text-gradient">atendidos</span></h2>
            <p>Verificação completa das quatro etapas antes do Demo Day. Todos os itens implementados, documentados e reproduzíveis.</p>
          </div>
          <div className="bento-grid stagger-up" style={{gridTemplateColumns: '1fr 1fr'}}>
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Etapa 1 — Dados + Baseline</span>
                <span className="bc-header-tag">5/5</span>
              </div>
              <div className="bc-body">
                <ul className="pricing-features">
                  <li className="pricing-feature">
                    <span className="iconify" data-icon="lucide:check-circle-2" style={{color:'#14B8A6'}}></span>
                    <span>
                      EDA documentada com insights relevantes ao domínio
                      <small style={{display:'block',color:'var(--muted)',fontSize:'0.72rem',marginTop:'2px'}}>
                        <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>notebooks/EDA.ipynb</code> — análise de volatilidade, correlações e sazonalidade da série NVDA (3.6 MB)
                      </small>
                    </span>
                  </li>
                  <li className="pricing-feature">
                    <span className="iconify" data-icon="lucide:check-circle-2" style={{color:'#14B8A6'}}></span>
                    <span>
                      Baseline treinado e métricas reportadas no MLflow
                      <small style={{display:'block',color:'var(--muted)',fontSize:'0.72rem',marginTop:'2px'}}>
                        <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>src/training/train.py</code> · <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>mlruns/</code> — RMSE, MAE, MAPE, R², Sharpe Ratio rastreados por run via <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>@mlflow.trace()</code>
                      </small>
                    </span>
                  </li>
                  <li className="pricing-feature">
                    <span className="iconify" data-icon="lucide:check-circle-2" style={{color:'#14B8A6'}}></span>
                    <span>
                      Pipeline versionado (DVC + Docker) e reproduzível
                      <small style={{display:'block',color:'var(--muted)',fontSize:'0.72rem',marginTop:'2px'}}>
                        <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>dvc.yaml</code> 3 stages (extract→preprocess→train) · <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>Dockerfile</code> multi-stage · <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>docker-compose.yml</code> com health-checks
                      </small>
                    </span>
                  </li>
                  <li className="pricing-feature">
                    <span className="iconify" data-icon="lucide:check-circle-2" style={{color:'#14B8A6'}}></span>
                    <span>
                      Métricas de negócio mapeadas para métricas técnicas
                      <small style={{display:'block',color:'var(--muted)',fontSize:'0.72rem',marginTop:'2px'}}>
                        <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>src/monitoring/business_metrics.py</code> — <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>BusinessMetricsTracker</code>: P&L simulado, acurácia direcional, Sharpe Ratio → R²/MAPE
                      </small>
                    </span>
                  </li>
                  <li className="pricing-feature">
                    <span className="iconify" data-icon="lucide:check-circle-2" style={{color:'#14B8A6'}}></span>
                    <span>
                      pyproject.toml com todas as dependências
                      <small style={{display:'block',color:'var(--muted)',fontSize:'0.72rem',marginTop:'2px'}}>
                        <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>pyproject.toml</code> — 25+ deps de produção · <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>[project.optional-dependencies]</code> dev/test separados · <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>requires-python = "≥3.12"</code>
                      </small>
                    </span>
                  </li>
                </ul>
              </div>
            </div>

            {/* Etapa 2 */}
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Etapa 2 — LLM + Agente</span>
                <span className="bc-header-tag">5/5</span>
              </div>
              <div className="bc-body">
                <ul className="pricing-features">
                  <li className="pricing-feature">
                    <span className="iconify" data-icon="lucide:check-circle-2" style={{color:'#14B8A6'}}></span>
                    <span>
                      LLM servido via API com quantização aplicada
                      <small style={{display:'block',color:'var(--muted)',fontSize:'0.72rem',marginTop:'2px'}}>
                        <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>src/agent/react_agent.py</code> · Groq serve <strong>llama-3.3-70b-versatile</strong> (quantizado na infra do provedor) via <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>LLM_PROVIDER=groq|openrouter|openai</code>
                      </small>
                    </span>
                  </li>
                  <li className="pricing-feature">
                    <span className="iconify" data-icon="lucide:check-circle-2" style={{color:'#14B8A6'}}></span>
                    <span>
                      Agente ReAct funcional com ≥ 3 tools relevantes ao domínio
                      <small style={{display:'block',color:'var(--muted)',fontSize:'0.72rem',marginTop:'2px'}}>
                        <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>src/agent/tools.py</code> — 4 tools: <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>query_stock_data</code> · <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>predict_stock_prices</code> · <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>get_model_metrics</code> · <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>search_documents</code>
                      </small>
                    </span>
                  </li>
                  <li className="pricing-feature">
                    <span className="iconify" data-icon="lucide:check-circle-2" style={{color:'#14B8A6'}}></span>
                    <span>
                      RAG retornando contexto relevante dos dados fornecidos
                      <small style={{display:'block',color:'var(--muted)',fontSize:'0.72rem',marginTop:'2px'}}>
                        <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>src/agent/rag_pipeline.py</code> — ChromaDB persistido em <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>data/chroma_db/</code>, 7 documentos de domínio com embeddings sentence-transformer
                      </small>
                    </span>
                  </li>
                  <li className="pricing-feature">
                    <span className="iconify" data-icon="lucide:check-circle-2" style={{color:'#14B8A6'}}></span>
                    <span>
                      CI/CD pipeline funcional (GitHub Actions)
                      <small style={{display:'block',color:'var(--muted)',fontSize:'0.72rem',marginTop:'2px'}}>
                        <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>.github/workflows/ci.yml</code> — ruff · mypy · bandit · pip-audit · pytest (cov≥60%) · docker build + health-check
                      </small>
                    </span>
                  </li>
                  <li className="pricing-feature">
                    <span className="iconify" data-icon="lucide:check-circle-2" style={{color:'#14B8A6'}}></span>
                    <span>
                      Benchmark documentado com ≥ 3 configurações
                      <small style={{display:'block',color:'var(--muted)',fontSize:'0.72rem',marginTop:'2px'}}>
                        <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>docs/LLM_BENCHMARK.md</code> + <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>evaluation/ab_test_prompts.py</code> — Variante A (concise), B (detailed+examples), providers groq/openrouter/openai
                      </small>
                    </span>
                  </li>
                </ul>
              </div>
            </div>

            {/* Etapa 3 */}
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Etapa 3 — Avaliação + Observabilidade</span>
                <span className="bc-header-tag">5/5</span>
              </div>
              <div className="bc-body">
                <ul className="pricing-features">
                  <li className="pricing-feature">
                    <span className="iconify" data-icon="lucide:check-circle-2" style={{color:'#14B8A6'}}></span>
                    <span>
                      Golden set com ≥ 20 pares relevantes ao domínio
                      <small style={{display:'block',color:'var(--muted)',fontSize:'0.72rem',marginTop:'2px'}}>
                        <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>data/golden_set/golden_set.json</code> — <strong>25 pares</strong> PT+EN cobrindo preços, predições, arquitetura, segurança e uso geral
                      </small>
                    </span>
                  </li>
                  <li className="pricing-feature">
                    <span className="iconify" data-icon="lucide:check-circle-2" style={{color:'#14B8A6'}}></span>
                    <span>
                      RAGAS: 4 métricas calculadas e reportadas
                      <small style={{display:'block',color:'var(--muted)',fontSize:'0.72rem',marginTop:'2px'}}>
                        <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>evaluation/ragas_eval.py</code> — faithfulness · answer_relevancy · context_precision · context_recall (targets ≥0.6–0.7)
                      </small>
                    </span>
                  </li>
                  <li className="pricing-feature">
                    <span className="iconify" data-icon="lucide:check-circle-2" style={{color:'#14B8A6'}}></span>
                    <span>
                      LLM-as-judge com ≥ 3 critérios (incluindo negócio)
                      <small style={{display:'block',color:'var(--muted)',fontSize:'0.72rem',marginTop:'2px'}}>
                        <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>evaluation/llm_judge.py</code> — relevância da resposta · acurácia factual · <strong>utilidade para decisão de investimento</strong> (Zheng et al., 2023)
                      </small>
                    </span>
                  </li>
                  <li className="pricing-feature">
                    <span className="iconify" data-icon="lucide:check-circle-2" style={{color:'#14B8A6'}}></span>
                    <span>
                      Telemetria e dashboard funcionando end-to-end
                      <small style={{display:'block',color:'var(--muted)',fontSize:'0.72rem',marginTop:'2px'}}>
                        <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>src/monitoring/telemetry.py</code> (Langfuse) · <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>configs/prometheus.yml</code> (scrape 15s) · <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>configs/grafana/</code> dashboards JSON
                      </small>
                    </span>
                  </li>
                  <li className="pricing-feature">
                    <span className="iconify" data-icon="lucide:check-circle-2" style={{color:'#14B8A6'}}></span>
                    <span>
                      Detecção de drift implementada e documentada
                      <small style={{display:'block',color:'var(--muted)',fontSize:'0.72rem',marginTop:'2px'}}>
                        <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>src/monitoring/drift.py</code> — PSI (warning 0.1 / retrain 0.2) · staleness 30d · breach rate CI 20% — <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>detect_drift()</code>
                      </small>
                    </span>
                  </li>
                </ul>
              </div>
            </div>

            {/* Etapa 4 */}
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Etapa 4 — Segurança + Governança</span>
                <span className="bc-header-tag">6/6</span>
              </div>
              <div className="bc-body">
                <ul className="pricing-features">
                  <li className="pricing-feature">
                    <span className="iconify" data-icon="lucide:check-circle-2" style={{color:'#14B8A6'}}></span>
                    <span>
                      OWASP mapping com ≥ 5 ameaças e mitigações
                      <small style={{display:'block',color:'var(--muted)',fontSize:'0.72rem',marginTop:'2px'}}>
                        <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>docs/OWASP_MAPPING.md</code> — <strong>10/10 OWASP LLM Top 10</strong> mapeados: LLM01 Prompt Injection → LLM10 Model Theft com mitigações
                      </small>
                    </span>
                  </li>
                  <li className="pricing-feature">
                    <span className="iconify" data-icon="lucide:check-circle-2" style={{color:'#14B8A6'}}></span>
                    <span>
                      Guardrails de input e output funcionais
                      <small style={{display:'block',color:'var(--muted)',fontSize:'0.72rem',marginTop:'2px'}}>
                        <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>src/security/guardrails.py</code> — <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>InputGuardrail</code> (16 regex, MAX 2000 chars) + <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>OutputGuardrail</code> (Presidio PII + disclaimers obrigatórios)
                      </small>
                    </span>
                  </li>
                  <li className="pricing-feature">
                    <span className="iconify" data-icon="lucide:check-circle-2" style={{color:'#14B8A6'}}></span>
                    <span>
                      ≥ 5 cenários adversariais testados e documentados
                      <small style={{display:'block',color:'var(--muted)',fontSize:'0.72rem',marginTop:'2px'}}>
                        <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>docs/RED_TEAM_REPORT.md</code> — <strong>22 testes</strong>: T-INJ-01→07 · T-PII-01→06 · T-OT-01→04 · T-API-01→05 · bloqueio rate <strong>91%</strong>
                      </small>
                    </span>
                  </li>
                  <li className="pricing-feature">
                    <span className="iconify" data-icon="lucide:check-circle-2" style={{color:'#14B8A6'}}></span>
                    <span>
                      Plano LGPD aplicado ao caso real
                      <small style={{display:'block',color:'var(--muted)',fontSize:'0.72rem',marginTop:'2px'}}>
                        <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>docs/LGPD_PLAN.md</code> — 10 princípios Art. 6 · bases legais Art. 7 · PII Presidio · resposta a incidentes em 72h · direitos do titular
                      </small>
                    </span>
                  </li>
                  <li className="pricing-feature">
                    <span className="iconify" data-icon="lucide:check-circle-2" style={{color:'#14B8A6'}}></span>
                    <span>
                      Explicabilidade e fairness documentados
                      <small style={{display:'block',color:'var(--muted)',fontSize:'0.72rem',marginTop:'2px'}}>
                        <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>src/explainability/lime_explainer.py</code> + <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>feature_importance.py</code> — LIME local · permutation importance · Model Card com limitações e vieses
                      </small>
                    </span>
                  </li>
                  <li className="pricing-feature">
                    <span className="iconify" data-icon="lucide:check-circle-2" style={{color:'#14B8A6'}}></span>
                    <span>
                      System Card completo
                      <small style={{display:'block',color:'var(--muted)',fontSize:'0.72rem',marginTop:'2px'}}>
                        <code style={{background:'rgba(255,255,255,0.07)',padding:'1px 5px',borderRadius:'3px'}}>docs/SYSTEM_CARD.md</code> — 8 componentes documentados (ETL, LSTM, API, Monitoring, Agent, RAG, Dashboard) · diagrama ASCII · limitações
                      </small>
                    </span>
                  </li>
                </ul>
              </div>
            </div>

          </div>
          {/* Score summary */}
          <div className="hero-metrics reveal" style={{justifyContent: 'center', marginTop: '48px'}}>
            <div className="hero-metric">
              <span className="hero-metric-val">21</span>
              <span className="hero-metric-label">Itens entregues</span>
            </div>
            <div className="hero-metric-sep"></div>
            <div className="hero-metric">
              <span className="hero-metric-val">4</span>
              <span className="hero-metric-label">Etapas completas</span>
            </div>
            <div className="hero-metric-sep"></div>
            <div className="hero-metric">
              <span className="hero-metric-val">100%</span>
              <span className="hero-metric-label">Checklist</span>
            </div>
          </div>
        </div>
      </section>

      {/* ============================================================
          1) PROBLEMA & MOTIVAÇÃO
          ============================================================ */}
      <section className="section" id="problema">
        <div className="section-sep"></div>
        <div className="container">
          <div className="section-header reveal">
            <span className="section-label">01 — Problema &amp; Motivação</span>
            <h2>Por que <span className="text-gradient">forecasting financeiro</span> precisa de MLOps</h2>
            <p>Modelos de séries temporais em finanças envelhecem em dias. Sem pipeline, versionamento e observabilidade, qualquer ganho em offline desaparece em produção.</p>
          </div>
          <div className="bento-grid stagger-up" style={{gridTemplateColumns: '1fr 1fr 1fr'}}>
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Volatilidade</span>
                <span className="bc-header-tag">σ alto</span>
              </div>
              <div className="bc-body">
                <h3>NVDA é um ativo extremo</h3>
                <p>Drawdowns de dois dígitos em sessões únicas, regimes que mudam com earnings e ciclos de IA. Modelos estáticos quebram rapidamente.</p>
              </div>
            </div>
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Pipelines manuais</span>
                <span className="bc-header-tag">custo</span>
              </div>
              <div className="bc-body">
                <h3>Retreino virou bottleneck</h3>
                <p>Times de Data Science gastam 60% do tempo em ETL, tracking artesanal e deploy ad-hoc — em vez de melhorar o modelo.</p>
              </div>
            </div>
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Reprodutibilidade</span>
                <span className="bc-header-tag">auditoria</span>
              </div>
              <div className="bc-body">
                <h3>Sem rastro, sem confiança</h3>
                <p>Modelos financeiros precisam ser auditáveis: dados, código, hiperparâmetros e resultados versionados — algo raro em projetos acadêmicos.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============================================================
          2) OBJETIVO & HIPÓTESE
          ============================================================ */}
      <section className="section section-alt" id="objetivo">
        <div className="section-sep"></div>
        <div className="container">
          <div className="section-header reveal">
            <span className="section-label">02 — Objetivo &amp; Hipótese</span>
            <h2>Provar que dá pra fazer <span className="text-gradient">MLOps de verdade</span> em um Datathon</h2>
            <p>Um único repositório open-source, reproduzível em qualquer máquina, cobrindo todo o ciclo de vida do modelo.</p>
          </div>
          <div className="bento-grid stagger-up" style={{gridTemplateColumns: '1fr'}}>
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Hipótese central</span>
                <span className="bc-header-tag">Datathon · MLET</span>
              </div>
              <div className="bc-body">
                <h2 style={{margin: 0}}>É possível construir uma plataforma MLOps <span className="text-gradient">end-to-end e open-source</span> que prevê NVDA com <span className="text-gradient">R² &gt; 0.95</span>, com SLA de produção, observabilidade nativa e agente LLM auditável.</h2>
                <p style={{marginTop: '20px'}}>Critérios de aceite: pipeline DVC reproduzível · MLflow tracking + registry · API com p95 &lt; 300 ms · monitoramento Prometheus/Grafana · agente avaliado com RAGAS e LLM-as-Judge.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============================================================
          3) FUNDAMENTAÇÃO TEÓRICA
          ============================================================ */}
      <section className="section" id="fundamentacao">
        <div className="section-sep"></div>
        <div className="container">
          <div className="section-header reveal">
            <span className="section-label">03 — Fundamentação Teórica</span>
            <h2>Quatro pilares que sustentam <span className="text-gradient">a solução</span></h2>
            <p>Cada decisão técnica tem ancoragem em literatura e estado da prática.</p>
          </div>
          <div className="bento-grid stagger-up" style={{gridTemplateColumns: '1fr 1fr'}}>

            {/* ── LSTM ── */}
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">LSTM</span>
                <span className="bc-header-tag">PyTorch</span>
              </div>
              <div className="bc-body" style={{paddingTop: '4px'}}>
                <div style={{display:'flex',alignItems:'center',gap:'12px',marginBottom:'12px'}}>
                  <div style={{width:'40px',height:'40px',borderRadius:'10px',background:'rgba(239,68,68,0.1)',border:'1px solid rgba(239,68,68,0.2)',display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0}}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="7" width="20" height="10" rx="2"/><line x1="7" y1="7" x2="7" y2="17"/><line x1="12" y1="7" x2="12" y2="17"/><line x1="17" y1="7" x2="17" y2="17"/></svg>
                  </div>
                  <div>
                    <div style={{fontSize:'14px',fontWeight:700,color:'rgba(255,255,255,0.9)'}}>Long Short-Term Memory</div>
                    <div style={{fontSize:'11px',color:'rgba(239,68,68,0.7)',marginTop:'2px',fontFamily:'monospace'}}>src/training/train.py</div>
                  </div>
                </div>
                <p style={{fontSize:'12px',color:'rgba(255,255,255,0.45)',lineHeight:1.6,marginBottom:'14px'}}>
                  Stacked LSTM 2×128 com dropout. Captura dependências temporais não-lineares, supera ARIMA em ativos voláteis (Hochreiter &amp; Schmidhuber, 1997).
                </p>
                {[
                  { label: 'Arquitetura', val: '2 camadas × 128 unidades + dropout' },
                  { label: 'Input', val: 'sequências de 60 timesteps' },
                  { label: 'Resultado', val: `R² ${champion.r2.toFixed(3)} · MAPE ${champion.mape}` },
                ].map(r => (
                  <div key={r.label} style={{display:'flex',justifyContent:'space-between',padding:'6px 0',borderBottom:'1px solid rgba(255,255,255,0.04)',fontSize:'11px'}}>
                    <span style={{color:'rgba(255,255,255,0.3)'}}>{r.label}</span>
                    <span style={{color:'rgba(255,255,255,0.7)',fontWeight:500}}>{r.val}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* ── MLflow ── */}
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">MLflow</span>
                <span className="bc-header-tag">tracking</span>
              </div>
              <div className="bc-body" style={{paddingTop: '4px'}}>
                <div style={{display:'flex',alignItems:'center',gap:'12px',marginBottom:'12px'}}>
                  <div style={{width:'40px',height:'40px',borderRadius:'10px',background:'rgba(6,182,212,0.1)',border:'1px solid rgba(6,182,212,0.2)',display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0}}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M3 3v18h18"/><polyline points="18 9 13 14 9 10 3 16"/></svg>
                  </div>
                  <div>
                    <div style={{fontSize:'14px',fontWeight:700,color:'rgba(255,255,255,0.9)'}}>Experiment tracking</div>
                    <div style={{fontSize:'11px',color:'rgba(6,182,212,0.7)',marginTop:'2px',fontFamily:'monospace'}}>src/training/train.py · mlruns/</div>
                  </div>
                </div>
                <p style={{fontSize:'12px',color:'rgba(255,255,255,0.45)',lineHeight:1.6,marginBottom:'14px'}}>
                  Cada execução versiona código, dados, hiperparâmetros e artefatos via <code style={{fontFamily:'monospace',background:'rgba(255,255,255,0.06)',padding:'0 4px',borderRadius:'3px'}}>@mlflow.trace()</code>. Model Registry separa Champion e Challenger.
                </p>
                {[
                  { label: 'Métricas', val: 'RMSE · MAE · MAPE · R² · Sharpe' },
                  { label: 'Registro', val: 'Champion · Challenger por run' },
                  { label: 'Promoção', val: 'automática +1 p.p. R²' },
                ].map(r => (
                  <div key={r.label} style={{display:'flex',justifyContent:'space-between',padding:'6px 0',borderBottom:'1px solid rgba(255,255,255,0.04)',fontSize:'11px'}}>
                    <span style={{color:'rgba(255,255,255,0.3)'}}>{r.label}</span>
                    <span style={{color:'rgba(255,255,255,0.7)',fontWeight:500}}>{r.val}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* ── Optuna ── */}
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Optuna</span>
                <span className="bc-header-tag">HPO</span>
              </div>
              <div className="bc-body" style={{paddingTop: '4px'}}>
                <div style={{display:'flex',alignItems:'center',gap:'12px',marginBottom:'12px'}}>
                  <div style={{width:'40px',height:'40px',borderRadius:'10px',background:'rgba(139,92,246,0.1)',border:'1px solid rgba(139,92,246,0.2)',display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0}}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#8b5cf6" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M2 12h3M19 12h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12"/></svg>
                  </div>
                  <div>
                    <div style={{fontSize:'14px',fontWeight:700,color:'rgba(255,255,255,0.9)'}}>Bayesian search</div>
                    <div style={{fontSize:'11px',color:'rgba(139,92,246,0.7)',marginTop:'2px',fontFamily:'monospace'}}>src/training/train.py · HPO block</div>
                  </div>
                </div>
                <p style={{fontSize:'12px',color:'rgba(255,255,255,0.45)',lineHeight:1.6,marginBottom:'14px'}}>
                  50+ trials TPE para otimizar lookback, learning rate, dropout e tamanho do batch. Pruning agressivo para acelerar a busca.
                </p>
                {[
                  { label: 'Algoritmo', val: 'TPE (Tree-structured Parzen)' },
                  { label: 'Espaço', val: 'lookback · lr · dropout · batch' },
                  { label: 'Trials', val: '50+ com early pruning' },
                ].map(r => (
                  <div key={r.label} style={{display:'flex',justifyContent:'space-between',padding:'6px 0',borderBottom:'1px solid rgba(255,255,255,0.04)',fontSize:'11px'}}>
                    <span style={{color:'rgba(255,255,255,0.3)'}}>{r.label}</span>
                    <span style={{color:'rgba(255,255,255,0.7)',fontWeight:500}}>{r.val}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* ── RAG · LLM-as-Judge ── */}
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">RAG · LLM-as-Judge</span>
                <span className="bc-header-tag">avaliação</span>
              </div>
              <div className="bc-body" style={{paddingTop: '4px'}}>
                <div style={{display:'flex',alignItems:'center',gap:'12px',marginBottom:'12px'}}>
                  <div style={{width:'40px',height:'40px',borderRadius:'10px',background:'rgba(20,184,166,0.1)',border:'1px solid rgba(20,184,166,0.2)',display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0}}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#14b8a6" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                  </div>
                  <div>
                    <div style={{fontSize:'14px',fontWeight:700,color:'rgba(255,255,255,0.9)'}}>Agente auditável</div>
                    <div style={{fontSize:'11px',color:'rgba(20,184,166,0.7)',marginTop:'2px',fontFamily:'monospace'}}>src/agent/rag_pipeline.py · evaluation/</div>
                  </div>
                </div>
                <p style={{fontSize:'12px',color:'rgba(255,255,255,0.45)',lineHeight:1.6,marginBottom:'14px'}}>
                  RAG com ChromaDB (7 documentos) + RAGAS automático + LLM-as-Judge com 3 critérios sobre golden set de 25 pares.
                </p>
                {[
                  { label: 'Vector store', val: 'ChromaDB · sentence-transformers' },
                  { label: 'Avaliação', val: `RAGAS · faithfulness ${faith.toFixed(2)}` },
                  { label: 'Golden set', val: '25 pares PT+EN · CI reproduzível' },
                ].map(r => (
                  <div key={r.label} style={{display:'flex',justifyContent:'space-between',padding:'6px 0',borderBottom:'1px solid rgba(255,255,255,0.04)',fontSize:'11px'}}>
                    <span style={{color:'rgba(255,255,255,0.3)'}}>{r.label}</span>
                    <span style={{color:'rgba(255,255,255,0.7)',fontWeight:500}}>{r.val}</span>
                  </div>
                ))}
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* ============================================================
          4) ARQUITETURA
          ============================================================ */}
      <section className="section section-alt" id="arquitetura">
        <div className="section-sep"></div>
        <div className="container">
          <div className="section-header reveal">
            <span className="section-label">04 — Arquitetura</span>
            <h2>Seis microserviços, <span className="text-gradient">um pipeline</span></h2>
            <p>Tudo orquestrado via Docker Compose, atrás de um Nginx reverso, com CI/CD por GitHub Actions.</p>
          </div>
          <div className="bento-grid stagger-up" style={{gridTemplateColumns: '1fr'}}>
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">data flow · NVIDIA MLOps Platform</span>
                <span className="bc-header-tag">docker compose</span>
              </div>
              <div className="bc-body" style={{padding: '24px 20px'}}>
                {/* ── Row 1: Data Ingestion ── */}
                <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', marginBottom: '8px'}}>
                  {[
                    { label: 'Yahoo Finance', sub: 'yfinance' },
                    null,
                    { label: 'ETL · DVC', sub: 'extract + preprocess' },
                    null,
                    { label: 'SQLite + features', sub: '6.7K rows' },
                  ].map((node, i) => node === null ? (
                    <svg key={i} width="28" height="14" viewBox="0 0 28 14" fill="none"><path d="M0 7h22M22 7l-6-5M22 7l-6 5" stroke="rgba(20,184,166,0.5)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
                  ) : (
                    <div key={i} style={{
                      border: '1px solid rgba(20,184,166,0.25)', borderRadius: '8px',
                      padding: '8px 14px', background: 'rgba(20,184,166,0.05)',
                      textAlign: 'center', minWidth: '120px',
                    }}>
                      <div style={{fontSize: '12px', fontWeight: 600, color: 'rgba(255,255,255,0.85)'}}>{node.label}</div>
                      <div style={{fontSize: '10px', color: 'rgba(20,184,166,0.7)', marginTop: '2px'}}>{node.sub}</div>
                    </div>
                  ))}
                </div>

                {/* ── Arrow down ── */}
                <div style={{display: 'flex', justifyContent: 'center', margin: '2px 0'}}>
                  <svg width="14" height="22" viewBox="0 0 14 22" fill="none"><path d="M7 0v16M7 16l-5-5M7 16l5-5" stroke="rgba(20,184,166,0.5)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
                </div>

                {/* ── Row 2: Training Pipeline ── */}
                <div style={{display: 'flex', justifyContent: 'center', marginBottom: '8px'}}>
                  <div style={{
                    border: '1px solid rgba(20,184,166,0.2)', borderRadius: '10px',
                    padding: '14px 24px', background: 'rgba(20,184,166,0.04)',
                    width: '100%', maxWidth: '520px',
                  }}>
                    <div style={{fontSize: '10px', fontWeight: 700, color: 'rgba(20,184,166,0.6)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: '10px'}}>Training Pipeline</div>
                    <div style={{display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap', justifyContent: 'center'}}>
                      {['preprocess', 'sequencer', 'LSTM 2×128', 'Optuna HPO'].map((step, i, arr) => (
                        <>
                          <span key={step} style={{
                            fontSize: '11px', fontWeight: 500, color: 'rgba(255,255,255,0.8)',
                            background: 'rgba(255,255,255,0.06)', borderRadius: '5px', padding: '4px 8px',
                          }}>{step}</span>
                          {i < arr.length - 1 && <span key={`a${i}`} style={{color: 'rgba(20,184,166,0.5)', fontSize: '12px'}}>▶</span>}
                        </>
                      ))}
                    </div>
                    <div style={{display: 'flex', justifyContent: 'center', margin: '8px 0 4px'}}>
                      <svg width="14" height="18" viewBox="0 0 14 18" fill="none"><path d="M7 0v12M7 12l-4-4M7 12l4-4" stroke="rgba(20,184,166,0.4)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
                    </div>
                    <div style={{textAlign: 'center'}}>
                      <div style={{
                        display: 'inline-block', border: '1px solid rgba(20,184,166,0.2)', borderRadius: '7px',
                        padding: '6px 16px', background: 'rgba(20,184,166,0.06)',
                        fontSize: '11px', color: 'rgba(255,255,255,0.75)',
                      }}>
                        MLflow tracking · Model Registry
                        <span style={{display: 'block', fontSize: '10px', color: 'rgba(20,184,166,0.6)', marginTop: '2px'}}>Champion · Challenger</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* ── Arrow down + branch to 3 ── */}
                <div style={{display: 'flex', justifyContent: 'center', margin: '2px 0 8px'}}>
                  <svg width="14" height="22" viewBox="0 0 14 22" fill="none"><path d="M7 0v16M7 16l-5-5M7 16l5-5" stroke="rgba(20,184,166,0.5)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
                </div>

                {/* ── Row 3: Services ── */}
                <div style={{display: 'flex', alignItems: 'flex-start', justifyContent: 'center', gap: '12px', marginBottom: '8px'}}>
                  {[
                    { label: 'FastAPI', sub: '66 routes' },
                    { label: 'Agent · ReAct+RAG', sub: 'ChromaDB · Guardrails' },
                    { label: 'Prometheus + Grafana', sub: 'drift · SLA · alertas' },
                  ].map((node) => (
                    <div key={node.label} style={{
                      border: '1px solid rgba(20,184,166,0.25)', borderRadius: '8px',
                      padding: '10px 14px', background: 'rgba(20,184,166,0.05)',
                      textAlign: 'center', flex: 1, minWidth: 0,
                    }}>
                      <div style={{fontSize: '12px', fontWeight: 600, color: 'rgba(255,255,255,0.85)'}}>{node.label}</div>
                      <div style={{fontSize: '10px', color: 'rgba(20,184,166,0.65)', marginTop: '3px'}}>{node.sub}</div>
                    </div>
                  ))}
                </div>

                {/* ── Arrow down ── */}
                <div style={{display: 'flex', justifyContent: 'center', margin: '2px 0'}}>
                  <svg width="14" height="22" viewBox="0 0 14 22" fill="none"><path d="M7 0v16M7 16l-5-5M7 16l5-5" stroke="rgba(20,184,166,0.5)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
                </div>

                {/* ── Row 4: Dashboard ── */}
                <div style={{display: 'flex', justifyContent: 'center'}}>
                  <div style={{
                    border: '1px solid rgba(20,184,166,0.35)', borderRadius: '8px',
                    padding: '10px 28px', background: 'rgba(20,184,166,0.08)',
                    textAlign: 'center',
                  }}>
                    <div style={{fontSize: '12px', fontWeight: 700, color: 'rgba(255,255,255,0.9)'}}>Next.js 14 dashboard</div>
                    <div style={{fontSize: '10px', color: 'rgba(20,184,166,0.7)', marginTop: '2px'}}>11 páginas · realtime</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============================================================
          5) STACK
          ============================================================ */}
      <section className="section" id="stack">
        <div className="section-sep"></div>
        <div className="container">
          <div className="section-header reveal">
            <span className="section-label">05 — Stack &amp; Implementação</span>
            <h2>Tudo <span className="text-gradient">open-source</span> e production-grade</h2>
            <p>Sem dependências proprietárias. Roda na sua máquina hoje, em um cluster amanhã.</p>
          </div>
          <div className="bento-grid stagger-up" style={{gridTemplateColumns:'1fr 1fr'}}>
            {/* ── Modelagem ── */}
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Modelagem</span>
                <span className="bc-header-tag">python</span>
              </div>
              <div className="bc-body" style={{paddingTop: '4px'}}>
                <div style={{marginBottom: '14px', fontSize: '11px', color: 'rgba(255,255,255,0.45)'}}>
                  LSTM 2×128 · Optuna HPO · 3-stage DVC pipeline
                </div>
                {[
                  { name: 'Python 3.12', desc: 'runtime · pyproject.toml', color: '#3b82f6' },
                  { name: 'PyTorch 2.6', desc: 'LSTM training + gradients', color: '#ef4444' },
                  { name: 'scikit-learn', desc: 'scaler · baseline · metrics', color: '#f97316' },
                  { name: 'Optuna', desc: 'hyperparameter search (HPO)', color: '#8b5cf6' },
                  { name: 'MLflow', desc: 'experiment tracking · registry', color: '#06b6d4' },
                  { name: 'DVC', desc: 'pipeline versioning · repro', color: '#14b8a6' },
                ].map(t => (
                  <div key={t.name} style={{display: 'flex', alignItems: 'center', gap: '10px', padding: '7px 0', borderBottom: '1px solid rgba(255,255,255,0.04)'}}>
                    <span style={{width: '8px', height: '8px', borderRadius: '2px', background: t.color, flexShrink: 0, opacity: 0.85}}/>
                    <span style={{fontSize: '12px', fontWeight: 600, color: 'rgba(255,255,255,0.85)', minWidth: '90px'}}>{t.name}</span>
                    <span style={{fontSize: '11px', color: 'rgba(255,255,255,0.35)'}}>{t.desc}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* ── Serving & Frontend ── */}
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Serving &amp; Frontend</span>
                <span className="bc-header-tag">async</span>
              </div>
              <div className="bc-body" style={{paddingTop: '4px'}}>
                <div style={{marginBottom: '14px', fontSize: '11px', color: 'rgba(255,255,255,0.45)'}}>
                  66 endpoints REST · 11 páginas · realtime charts
                </div>
                {[
                  { name: 'FastAPI', desc: '66 routes · async · OpenAPI', color: '#22c55e' },
                  { name: 'Pydantic', desc: 'input validation · schemas', color: '#facc15' },
                  { name: 'Next.js 14', desc: 'app router · SSR/CSR hybrid', color: '#ffffff' },
                  { name: 'React', desc: 'UI components · hooks', color: '#38bdf8' },
                  { name: 'Tailwind', desc: 'utility-first styling', color: '#818cf8' },
                  { name: 'Recharts', desc: 'stock charts · sparklines', color: '#f472b6' },
                ].map(t => (
                  <div key={t.name} style={{display: 'flex', alignItems: 'center', gap: '10px', padding: '7px 0', borderBottom: '1px solid rgba(255,255,255,0.04)'}}>
                    <span style={{width: '8px', height: '8px', borderRadius: '2px', background: t.color, flexShrink: 0, opacity: 0.85}}/>
                    <span style={{fontSize: '12px', fontWeight: 600, color: 'rgba(255,255,255,0.85)', minWidth: '90px'}}>{t.name}</span>
                    <span style={{fontSize: '11px', color: 'rgba(255,255,255,0.35)'}}>{t.desc}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* ── Agente & Avaliação ── */}
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Agente &amp; Avaliação</span>
                <span className="bc-header-tag">LLM</span>
              </div>
              <div className="bc-body" style={{paddingTop: '4px'}}>
                <div style={{marginBottom: '14px', fontSize: '11px', color: 'rgba(255,255,255,0.45)'}}>
                  ReAct · 4 tools · RAG · 25 golden pairs
                </div>
                {[
                  { name: 'ChromaDB', desc: 'vector store · RAG · embeddings', color: '#14b8a6' },
                  { name: 'OpenRouter', desc: 'llama-3.3-70b · multi-provider', color: '#a78bfa' },
                  { name: 'LangChain', desc: 'agent · tools · ReAct loop', color: '#f59e0b' },
                  { name: 'RAGAS', desc: '4 métricas · faithfulness', color: '#06b6d4' },
                  { name: 'LLM-as-Judge', desc: '3 critérios · Zheng et al.', color: '#ec4899' },
                  { name: 'LIME', desc: 'local explainability · fairness', color: '#84cc16' },
                ].map(t => (
                  <div key={t.name} style={{display: 'flex', alignItems: 'center', gap: '10px', padding: '7px 0', borderBottom: '1px solid rgba(255,255,255,0.04)'}}>
                    <span style={{width: '8px', height: '8px', borderRadius: '2px', background: t.color, flexShrink: 0, opacity: 0.85}}/>
                    <span style={{fontSize: '12px', fontWeight: 600, color: 'rgba(255,255,255,0.85)', minWidth: '90px'}}>{t.name}</span>
                    <span style={{fontSize: '11px', color: 'rgba(255,255,255,0.35)'}}>{t.desc}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* ── Infra & Observabilidade ── */}
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Infra &amp; Observabilidade</span>
                <span className="bc-header-tag">prod</span>
              </div>
              <div className="bc-body" style={{paddingTop: '4px'}}>
                <div style={{marginBottom: '14px', fontSize: '11px', color: 'rgba(255,255,255,0.45)'}}>
                  multi-stage Docker · CI/CD · drift detection
                </div>
                {[
                  { name: 'Docker', desc: 'multi-stage · compose · healthcheck', color: '#38bdf8' },
                  { name: 'Nginx', desc: 'reverse proxy · SSL termination', color: '#22c55e' },
                  { name: 'Prometheus', desc: 'metrics scrape 15s · alertmanager', color: '#f97316' },
                  { name: 'Grafana', desc: 'dashboards JSON · drift panels', color: '#f59e0b' },
                  { name: 'GitHub Actions', desc: 'ruff · mypy · bandit · pytest', color: '#a78bfa' },
                  { name: 'Bandit + pip-audit', desc: 'SAST · CVE scanning', color: '#ef4444' },
                ].map(t => (
                  <div key={t.name} style={{display: 'flex', alignItems: 'center', gap: '10px', padding: '7px 0', borderBottom: '1px solid rgba(255,255,255,0.04)'}}>
                    <span style={{width: '8px', height: '8px', borderRadius: '2px', background: t.color, flexShrink: 0, opacity: 0.85}}/>
                    <span style={{fontSize: '12px', fontWeight: 600, color: 'rgba(255,255,255,0.85)', minWidth: '110px'}}>{t.name}</span>
                    <span style={{fontSize: '11px', color: 'rgba(255,255,255,0.35)'}}>{t.desc}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============================================================
          6) RESULTADOS
          ============================================================ */}
      <section className="section section-alt" id="resultados">
        <div className="section-sep"></div>
        <div className="container">
          <div className="section-header reveal">
            <span className="section-label">06 — Resultados</span>
            <h2>Métricas que <span className="text-gradient">passam no aceite</span></h2>
            <p>Avaliação holdout temporal · janela de 30 dias à frente · seed fixa para reprodutibilidade.</p>
          </div>
          <div className="bento-grid stagger-up" style={{gridTemplateColumns: '1fr 1fr 1fr'}}>
            {/* ── Predicted vs Actual ── */}
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Predicted vs Actual</span>
                <span className="bc-header-tag">last 60 days</span>
              </div>
              <div className="bc-body">
                {/* legend */}
                <div style={{display: 'flex', gap: '16px', marginBottom: '12px'}}>
                  <span style={{display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: 'rgba(255,255,255,0.55)'}}>
                    <span style={{width: '20px', height: '2px', background: '#14b8a6', borderRadius: '1px', display: 'inline-block'}}/>Predição
                  </span>
                  <span style={{display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: 'rgba(255,255,255,0.55)'}}>
                    <span style={{width: '20px', height: '0', borderTop: '2px dashed rgba(255,255,255,0.5)', display: 'inline-block'}}/>Real
                  </span>
                </div>
                {btData.length > 0 ? (() => {
                  const allPrices = btData.flatMap(p => [p.actual, p.predicted]);
                  const yMin = Math.min(...allPrices) * 0.97;
                  const yMax = Math.max(...allPrices) * 1.03;
                  const predPath = buildPath(btData, "predicted", yMin, yMax);
                  const actualPath = buildPath(btData, "actual", yMin, yMax);
                  const areaPath = predPath + ` L400,150 L0,150 Z`;
                  const labels = [0.8, 0.5, 0.2].map(t => ({
                    v: `$${Math.round(yMin + t * (yMax - yMin))}`,
                    y: Math.round(10 + (1 - t) * 140) - 2,
                  }));
                  const lastPt = btData[btData.length - 1];
                  const lastX = 400;
                  const lastY = 10 + (1 - (lastPt.predicted - yMin) / (yMax - yMin)) * 140;
                  return (
                    <svg viewBox="0 0 400 170" preserveAspectRatio="none" style={{width: '100%', height: '160px'}}>
                      <defs>
                        <linearGradient id="gPred" x1="0" x2="0" y1="0" y2="1">
                          <stop offset="0%" stopColor="#14B8A6" stopOpacity="0.35" />
                          <stop offset="100%" stopColor="#14B8A6" stopOpacity="0" />
                        </linearGradient>
                      </defs>
                      {[40,80,120].map(y => <line key={y} x1="0" y1={y} x2="400" y2={y} stroke="rgba(255,255,255,0.05)" strokeWidth="1"/>)}
                      {labels.map(l => (
                        <text key={l.v} x="4" y={l.y} fontSize="9" fill="rgba(255,255,255,0.25)">{l.v}</text>
                      ))}
                      <path d={areaPath} fill="url(#gPred)" />
                      <path d={predPath} fill="none" stroke="#14B8A6" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                      <path d={actualPath} fill="none" stroke="rgba(255,255,255,0.55)" strokeWidth="1.5" strokeDasharray="5 4"/>
                      <circle cx={lastX} cy={lastY} r="4" fill="#14b8a6" opacity="0.9"/>
                      {btR2 !== null && (
                        <text x={lastX > 340 ? lastX - 60 : lastX + 6} y={lastY - 6} fontSize="9" fill="#14b8a6">
                          R²={btR2.toFixed(3)}
                        </text>
                      )}
                    </svg>
                  );
                })() : (
                  <div style={{height:'160px', display:'flex', alignItems:'center', justifyContent:'center'}}>
                    <span style={{fontSize:'11px', color:'rgba(255,255,255,0.2)'}}>carregando…</span>
                  </div>
                )}
                <div style={{display: 'flex', gap: '20px', marginTop: '12px'}}>
                  <span style={{fontSize: '11px', color: 'rgba(255,255,255,0.35)'}}>
                    {btMae !== null ? `MAE ~$${btMae.toFixed(2)}` : 'Erro médio —'}
                  </span>
                  <span style={{fontSize: '11px', color: 'rgba(255,255,255,0.35)'}}>Seed fixo · holdout temporal</span>
                </div>
              </div>
            </div>

            {/* ── Champion vs Challenger ── */}
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Champion vs Challenger</span>
                <span className="bc-header-tag">MLflow registry</span>
              </div>
              <div className="bc-body" style={{paddingTop: '8px'}}>
                {models.map(m => (
                  <div key={m.version} style={{marginBottom: '18px'}}>
                    <div style={{display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px'}}>
                      <div style={{
                        width: '28px', height: '28px', borderRadius: '6px', flexShrink: 0,
                        background: `linear-gradient(135deg,${m.color}33,${m.color}88)`,
                        border: `1px solid ${m.color}55`,
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '10px', fontWeight: 800, color: m.color,
                      }}>{m.version}</div>
                      <div style={{flex: 1}}>
                        <div style={{display: 'flex', alignItems: 'center', gap: '6px'}}>
                          <span style={{fontSize: '12px', fontWeight: 600, color: 'rgba(255,255,255,0.85)'}}>{m.name}</span>
                          {m.crown && <span style={{fontSize: '12px'}}>👑</span>}
                          <span style={{
                            fontSize: '9px', fontWeight: 700, letterSpacing: '0.08em',
                            padding: '1px 5px', borderRadius: '3px',
                            background: m.crown ? 'rgba(20,184,166,0.15)' : 'rgba(255,255,255,0.06)',
                            color: m.crown ? '#14b8a6' : 'rgba(255,255,255,0.35)',
                            textTransform: 'uppercase',
                          }}>{m.role}</span>
                        </div>
                        <div style={{display: 'flex', gap: '10px', marginTop: '2px'}}>
                          <span style={{fontSize: '10px', color: 'rgba(255,255,255,0.35)'}}>MAPE {m.mape}</span>
                        </div>
                      </div>
                      <span style={{fontSize: '20px', fontWeight: 800, color: m.color}}>{m.r2}</span>
                    </div>
                    <div style={{height: '5px', background: 'rgba(255,255,255,0.06)', borderRadius: '3px', overflow: 'hidden'}}>
                      <div style={{height: '100%', width: `${m.pct}%`, background: `linear-gradient(90deg, ${m.color}88, ${m.color})`, borderRadius: '3px', transition: 'width 0.8s ease'}}/>
                    </div>
                  </div>
                ))}
                <div style={{
                  marginTop: '4px', padding: '10px 12px', borderRadius: '8px',
                  background: 'rgba(20,184,166,0.06)', border: '1px solid rgba(20,184,166,0.12)',
                  fontSize: '11px', color: 'rgba(255,255,255,0.4)', lineHeight: 1.5,
                }}>
                  Promoção automática quando challenger supera champion em ≥ 1 p.p. de R²
                </div>
              </div>
            </div>

            {/* ── RAGAS + LLM Judge ── */}
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Avaliação do Agente</span>
                <span className="bc-header-tag">RAGAS</span>
              </div>
              <div className="bc-body" style={{paddingTop: '8px'}}>
                {[
                  { label: 'Faithfulness', val: 0.43, target: 0.70, color: '#14b8a6' },
                  { label: 'Answer Relevancy', val: 0.66, target: 0.70, color: '#2dd4bf' },
                  { label: 'Context Precision', val: 0.88, target: 0.65, color: '#a78bfa' },
                  { label: 'Context Recall', val: 0.63, target: 0.60, color: '#f472b6' },
                ].map(m => (
                  <div key={m.label} style={{marginBottom: '14px'}}>
                    <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '5px'}}>
                      <span style={{fontSize: '11px', color: 'rgba(255,255,255,0.55)'}}>{m.label}</span>
                      <div style={{display: 'flex', gap: '8px', alignItems: 'center'}}>
                        <span style={{fontSize: '10px', color: 'rgba(255,255,255,0.25)'}}>target {m.target}</span>
                        <span style={{fontSize: '15px', fontWeight: 700, color: m.color}}>{m.val}</span>
                      </div>
                    </div>
                    <div style={{height: '6px', background: 'rgba(255,255,255,0.06)', borderRadius: '3px', position: 'relative', overflow: 'visible'}}>
                      {/* target line */}
                      <div style={{position: 'absolute', left: `${m.target*100}%`, top: '-3px', bottom: '-3px', width: '1px', background: 'rgba(255,255,255,0.2)', borderRadius: '1px'}}/>
                      <div style={{height: '100%', width: `${m.val*100}%`, background: `linear-gradient(90deg, ${m.color}66, ${m.color})`, borderRadius: '3px'}}/>
                    </div>
                  </div>
                ))}
                <div style={{marginTop: '16px', padding: '10px 14px', borderRadius: '8px', background: 'rgba(167,139,250,0.06)', border: '1px solid rgba(167,139,250,0.12)', display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                  <span style={{fontSize: '11px', color: 'rgba(255,255,255,0.45)'}}>LLM-as-Judge</span>
                  <div style={{display: 'flex', gap: '3px'}}>
                    {[1,2,3,4].map(i => <span key={i} style={{color: '#a78bfa', fontSize: '14px'}}>★</span>)}
                    <span style={{color: 'rgba(167,139,250,0.35)', fontSize: '14px'}}>★</span>
                    <span style={{fontSize: '11px', fontWeight: 700, color: '#a78bfa', marginLeft: '4px'}}>4.35/5</span>
                  </div>
                </div>
                <div style={{marginTop: '8px', fontSize: '11px', color: 'rgba(255,255,255,0.3)'}}>
                  Golden set de 25 pares · reproduzível em CI
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============================================================
          7) DIFERENCIAIS
          ============================================================ */}
      <section className="section" id="diferenciais">
        <div className="section-sep"></div>
        <div className="container">
          <div className="section-header reveal">
            <span className="section-label">07 — Diferenciais</span>
            <h2>O que <span className="text-gradient">difere</span> este Datathon</h2>
            <p>Mais que um notebook entregando MAPE — um produto inteiro com guardrails, observabilidade e explicabilidade.</p>
          </div>
          <div className="bento-grid stagger-up" style={{gridTemplateColumns: '1fr 1fr 1fr'}}>

            {/* ── Agente ReAct + RAG ── */}
            <div className="bento-card" style={{borderColor: 'rgba(167,139,250,0.2)'}}>
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Agente ReAct + RAG</span>
                <span className="bc-header-tag" style={{background:'rgba(167,139,250,0.12)',color:'#a78bfa'}}>LLM</span>
              </div>
              <div className="bc-body" style={{paddingTop: '4px'}}>
                {/* icon + title */}
                <div style={{display:'flex',alignItems:'center',gap:'12px',marginBottom:'14px'}}>
                  <div style={{width:'44px',height:'44px',borderRadius:'10px',background:'rgba(167,139,250,0.1)',border:'1px solid rgba(167,139,250,0.2)',display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0}}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a4 4 0 0 1 4 4v1h1a2 2 0 0 1 2 2v3a2 2 0 0 1-2 2h-1v1a4 4 0 0 1-8 0v-1H7a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h1V6a4 4 0 0 1 4-4z"/><circle cx="9" cy="10" r="1"/><circle cx="15" cy="10" r="1"/></svg>
                  </div>
                  <div>
                    <div style={{fontSize:'14px',fontWeight:700,color:'rgba(255,255,255,0.9)'}}>Conversa com o modelo</div>
                    <div style={{fontSize:'11px',color:'rgba(167,139,250,0.7)',marginTop:'2px'}}>src/agent/react_agent.py</div>
                  </div>
                </div>
                <p style={{fontSize:'12px',color:'rgba(255,255,255,0.45)',lineHeight:1.6,marginBottom:'16px'}}>
                  Agente responde em linguagem natural consultando ChromaDB com 7 documentos de domínio. Guardrails contra prompt injection e PII.
                </p>
                {/* feature pills */}
                {[
                  { icon: (<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#14b8a6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>), label: '4 tools', sub: 'query · predict · metrics · search' },
                  { icon: (<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#f87171" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>), label: 'OWASP LLM Top 10', sub: '10/10 ameaças mapeadas' },
                  { icon: (<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>), label: 'RAGAS contínuo', sub: `faithfulness ${faith.toFixed(2)}` },
                ].map(f => (
                  <div key={f.label} style={{display:'flex',gap:'10px',alignItems:'flex-start',padding:'8px 0',borderBottom:'1px solid rgba(255,255,255,0.04)'}}>
                    <span style={{marginTop:'2px',flexShrink:0,display:'flex'}}>{f.icon}</span>
                    <div>
                      <div style={{fontSize:'12px',fontWeight:600,color:'rgba(255,255,255,0.75)'}}>{f.label}</div>
                      <div style={{fontSize:'10px',color:'rgba(255,255,255,0.3)',marginTop:'1px'}}>{f.sub}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* ── Observabilidade ── */}
            <div className="bento-card" style={{borderColor: 'rgba(34,197,94,0.2)'}}>
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Observabilidade</span>
                <span className="bc-header-tag" style={{background:'rgba(34,197,94,0.12)',color:'#22c55e'}}>SLA</span>
              </div>
              <div className="bc-body" style={{paddingTop: '4px'}}>
                <div style={{display:'flex',alignItems:'center',gap:'12px',marginBottom:'14px'}}>
                  <div style={{width:'44px',height:'44px',borderRadius:'10px',background:'rgba(34,197,94,0.1)',border:'1px solid rgba(34,197,94,0.2)',display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0}}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                  </div>
                  <div>
                    <div style={{fontSize:'14px',fontWeight:700,color:'rgba(255,255,255,0.9)'}}>Drift &amp; SLA tracking</div>
                    <div style={{fontSize:'11px',color:'rgba(34,197,94,0.7)',marginTop:'2px'}}>src/monitoring/drift.py</div>
                  </div>
                </div>
                <p style={{fontSize:'12px',color:'rgba(255,255,255,0.45)',lineHeight:1.6,marginBottom:'16px'}}>
                  Prometheus coleta métricas a cada 15s. PSI detecta drift de feature. Grafana exibe SLOs e dispara alertas em violação.
                </p>
                {[
                  { icon: (<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>), label: 'PSI drift detection', sub: 'warning 0.1 · retrain 0.2' },
                  { icon: (<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>), label: 'Latência p95 &lt; 500ms', sub: '187ms atual em prod' },
                  { icon: (<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fb923c" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>), label: 'Alertmanager', sub: 'breach rate CI 20%' },
                ].map(f => (
                  <div key={f.label} style={{display:'flex',gap:'10px',alignItems:'flex-start',padding:'8px 0',borderBottom:'1px solid rgba(255,255,255,0.04)'}}>
                    <span style={{marginTop:'2px',flexShrink:0,display:'flex'}}>{f.icon}</span>
                    <div>
                      <div style={{fontSize:'12px',fontWeight:600,color:'rgba(255,255,255,0.75)'}} dangerouslySetInnerHTML={{__html:f.label}}/>
                      <div style={{fontSize:'10px',color:'rgba(255,255,255,0.3)',marginTop:'1px'}}>{f.sub}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* ── Explicabilidade ── */}
            <div className="bento-card" style={{borderColor: 'rgba(251,191,36,0.2)'}}>
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Explicabilidade</span>
                <span className="bc-header-tag" style={{background:'rgba(251,191,36,0.12)',color:'#fbbf24'}}>LIME</span>
              </div>
              <div className="bc-body" style={{paddingTop: '4px'}}>
                <div style={{display:'flex',alignItems:'center',gap:'12px',marginBottom:'14px'}}>
                  <div style={{width:'44px',height:'44px',borderRadius:'10px',background:'rgba(251,191,36,0.1)',border:'1px solid rgba(251,191,36,0.2)',display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0}}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
                  </div>
                  <div>
                    <div style={{fontSize:'14px',fontWeight:700,color:'rgba(255,255,255,0.9)'}}>Champion-Challenger + LIME</div>
                    <div style={{fontSize:'11px',color:'rgba(251,191,36,0.7)',marginTop:'2px'}}>src/explainability/lime_explainer.py</div>
                  </div>
                </div>
                <p style={{fontSize:'12px',color:'rgba(255,255,255,0.45)',lineHeight:1.6,marginBottom:'16px'}}>
                  Toda predição vem com importância por feature via LIME local. Promoção de modelo automatizada e auditável via MLflow Registry.
                </p>
                {[
                  { icon: (<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>), label: 'LIME local + permutation', sub: 'feature importance por decisão' },
                  { icon: (<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21H17"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/></svg>), label: 'Fairness documentado', sub: 'Model Card · limitações · vieses' },
                  { icon: (<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>), label: 'System Card completo', sub: '8 componentes · diagrama ASCII' },
                ].map(f => (
                  <div key={f.label} style={{display:'flex',gap:'10px',alignItems:'flex-start',padding:'8px 0',borderBottom:'1px solid rgba(255,255,255,0.04)'}}>
                    <span style={{marginTop:'2px',flexShrink:0,display:'flex'}}>{f.icon}</span>
                    <div>
                      <div style={{fontSize:'12px',fontWeight:600,color:'rgba(255,255,255,0.75)'}}>{f.label}</div>
                      <div style={{fontSize:'10px',color:'rgba(255,255,255,0.3)',marginTop:'1px'}}>{f.sub}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* ============================================================
          8) DEMO AO VIVO
          ============================================================ */}
      <section className="section section-alt" id="demo">
        <div className="section-sep"></div>
        <div className="container">
          <div className="section-header reveal">
            <span className="section-label">08 — Demonstração</span>
            <h2>Vamos ver <span className="text-gradient">rodando</span></h2>
            <p>Três entradas no dashboard — todas servidas pela mesma API e instrumentadas pelo mesmo Prometheus.</p>
          </div>
          <div className="stagger-up" style={{display:'grid',gridTemplateColumns:'1fr 1fr 1fr',gap:'24px'}}>

            {/* ── /predictions ── */}
            <div className="bento-card" style={{borderColor:'rgba(20,184,166,0.2)'}}>
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Predictions</span>
                <span className="bc-header-tag" style={{background:'rgba(20,184,166,0.12)',color:'#14b8a6'}}>LSTM</span>
              </div>
              <div className="bc-body" style={{paddingTop:'4px'}}>
                <div style={{display:'flex',alignItems:'center',gap:'12px',marginBottom:'14px'}}>
                  <div style={{width:'44px',height:'44px',borderRadius:'10px',background:'rgba(20,184,166,0.1)',border:'1px solid rgba(20,184,166,0.2)',display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0}}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#14b8a6" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
                  </div>
                  <div>
                    <div style={{fontSize:'14px',fontWeight:700,color:'rgba(255,255,255,0.9)'}}>Forecast 30 dias</div>
                    <div style={{fontSize:'11px',color:'rgba(20,184,166,0.7)',marginTop:'2px',fontFamily:'monospace'}}>/predictions</div>
                  </div>
                </div>
                <p style={{fontSize:'12px',color:'rgba(255,255,255,0.45)',lineHeight:1.6,marginBottom:'16px'}}>
                  Predição de preço NVDA com intervalo de confiança, comparativo com o histórico real e re-forecast on-demand.
                </p>
                {[
                  {icon:(<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#14b8a6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>), label:'Gráfico interativo', sub:'Recharts · zoom · tooltip'},
                  {icon:(<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#14b8a6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/></svg>), label:'Re-forecast on-demand', sub:'seed fixa · holdout temporal'},
                  {icon:(<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#14b8a6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>), label:'Export CSV / JSON', sub:'dados brutos + predições'},
                ].map(f => (
                  <div key={f.label} style={{display:'flex',gap:'10px',alignItems:'flex-start',padding:'8px 0',borderBottom:'1px solid rgba(255,255,255,0.04)'}}>
                    <span style={{marginTop:'2px',flexShrink:0,display:'flex'}}>{f.icon}</span>
                    <div>
                      <div style={{fontSize:'12px',fontWeight:600,color:'rgba(255,255,255,0.75)'}}>{f.label}</div>
                      <div style={{fontSize:'10px',color:'rgba(255,255,255,0.3)',marginTop:'1px'}}>{f.sub}</div>
                    </div>
                  </div>
                ))}
                <a href="/predictions" style={{display:'block',marginTop:'20px',padding:'10px 0',textAlign:'center',borderRadius:'8px',border:'1px solid rgba(20,184,166,0.3)',color:'#14b8a6',fontSize:'13px',fontWeight:600,textDecoration:'none',transition:'background 0.2s'}}>
                  Abrir página →
                </a>
              </div>
            </div>

            {/* ── /agent ── */}
            <div className="bento-card" style={{borderColor:'rgba(139,92,246,0.35)',boxShadow:'0 0 32px rgba(139,92,246,0.08)'}}>
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Agent</span>
                <span className="bc-header-tag" style={{background:'rgba(139,92,246,0.18)',color:'#a78bfa',fontWeight:700}}>★ destaque</span>
              </div>
              <div className="bc-body" style={{paddingTop:'4px'}}>
                <div style={{display:'flex',alignItems:'center',gap:'12px',marginBottom:'14px'}}>
                  <div style={{width:'44px',height:'44px',borderRadius:'10px',background:'rgba(139,92,246,0.12)',border:'1px solid rgba(139,92,246,0.3)',display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0}}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                  </div>
                  <div>
                    <div style={{fontSize:'14px',fontWeight:700,color:'rgba(255,255,255,0.9)'}}>ReAct + RAG</div>
                    <div style={{fontSize:'11px',color:'rgba(139,92,246,0.8)',marginTop:'2px',fontFamily:'monospace'}}>/agent</div>
                  </div>
                </div>
                <p style={{fontSize:'12px',color:'rgba(255,255,255,0.45)',lineHeight:1.6,marginBottom:'16px'}}>
                  Converse com o sistema: peça métricas, explicações de predição ou status de drift em linguagem natural.
                </p>
                {[
                  {icon:(<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>), label:'ReAct + RAG ChromaDB', sub:'4 tools · 7 docs de domínio'},
                  {icon:(<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#f87171" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>), label:'Guardrails OWASP LLM', sub:'10/10 ameaças · Presidio PII'},
                  {icon:(<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>), label:'RAGAS contínuo', sub:`faithfulness ${faith.toFixed(2)} · relevancy ${relev.toFixed(2)}`},
                ].map(f => (
                  <div key={f.label} style={{display:'flex',gap:'10px',alignItems:'flex-start',padding:'8px 0',borderBottom:'1px solid rgba(255,255,255,0.04)'}}>
                    <span style={{marginTop:'2px',flexShrink:0,display:'flex'}}>{f.icon}</span>
                    <div>
                      <div style={{fontSize:'12px',fontWeight:600,color:'rgba(255,255,255,0.75)'}}>{f.label}</div>
                      <div style={{fontSize:'10px',color:'rgba(255,255,255,0.3)',marginTop:'1px'}}>{f.sub}</div>
                    </div>
                  </div>
                ))}
                <a href="/agent" style={{display:'block',marginTop:'20px',padding:'10px 0',textAlign:'center',borderRadius:'8px',background:'linear-gradient(135deg,rgba(139,92,246,0.7),rgba(109,40,217,0.7))',color:'#fff',fontSize:'13px',fontWeight:700,textDecoration:'none',border:'1px solid rgba(139,92,246,0.4)'}}>
                  Abrir agente →
                </a>
              </div>
            </div>

            {/* ── /monitoring ── */}
            <div className="bento-card" style={{borderColor:'rgba(34,197,94,0.2)'}}>
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Monitoring</span>
                <span className="bc-header-tag" style={{background:'rgba(34,197,94,0.12)',color:'#22c55e'}}>SLO</span>
              </div>
              <div className="bc-body" style={{paddingTop:'4px'}}>
                <div style={{display:'flex',alignItems:'center',gap:'12px',marginBottom:'14px'}}>
                  <div style={{width:'44px',height:'44px',borderRadius:'10px',background:'rgba(34,197,94,0.1)',border:'1px solid rgba(34,197,94,0.2)',display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0}}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                  </div>
                  <div>
                    <div style={{fontSize:'14px',fontWeight:700,color:'rgba(255,255,255,0.9)'}}>SLOs em tempo real</div>
                    <div style={{fontSize:'11px',color:'rgba(34,197,94,0.7)',marginTop:'2px',fontFamily:'monospace'}}>/monitoring</div>
                  </div>
                </div>
                <p style={{fontSize:'12px',color:'rgba(255,255,255,0.45)',lineHeight:1.6,marginBottom:'16px'}}>
                  Drift de features, latência p95 e saúde dos serviços. Prometheus scrape a cada 15s, dashboards Grafana em JSON.
                </p>
                {[
                  {icon:(<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>), label:'PSI drift detection', sub:'warning 0.1 · retrain 0.2'},
                  {icon:(<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>), label:'Latência p95 187ms', sub:'SLO &lt; 500ms · prod'},
                  {icon:(<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>), label:'Alertmanager', sub:'breach rate CI 20%'},
                ].map(f => (
                  <div key={f.label} style={{display:'flex',gap:'10px',alignItems:'flex-start',padding:'8px 0',borderBottom:'1px solid rgba(255,255,255,0.04)'}}>
                    <span style={{marginTop:'2px',flexShrink:0,display:'flex'}}>{f.icon}</span>
                    <div>
                      <div style={{fontSize:'12px',fontWeight:600,color:'rgba(255,255,255,0.75)'}} dangerouslySetInnerHTML={{__html:f.label}}/>
                      <div style={{fontSize:'10px',color:'rgba(255,255,255,0.3)',marginTop:'1px'}}>{f.sub}</div>
                    </div>
                  </div>
                ))}
                <a href="/monitoring" style={{display:'block',marginTop:'20px',padding:'10px 0',textAlign:'center',borderRadius:'8px',border:'1px solid rgba(34,197,94,0.3)',color:'#22c55e',fontSize:'13px',fontWeight:600,textDecoration:'none',transition:'background 0.2s'}}>
                  Abrir monitoramento →
                </a>
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* ============================================================
          9) CONCLUSÃO & ROADMAP
          ============================================================ */}
      <section className="section" id="conclusao">
        <div className="section-sep"></div>
        <div className="container">
          <div className="section-header reveal">
            <span className="section-label">09 — Conclusões &amp; Roadmap</span>
            <h2>Hipótese <span className="text-gradient">validada</span></h2>
            <p>Plataforma reproduzível, com SLA de produção e métricas acima do critério de aceite. Próximos passos abrem o caminho para multi-ativo.</p>
          </div>
          <div className="bento-grid stagger-up" style={{gridTemplateColumns:'1fr 1fr',maxWidth:'800px',margin:'0 auto'}}>
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Entregue</span>
                <span className="bc-header-tag">done</span>
              </div>
              <div className="bc-body">
                <ul className="pricing-features">
                  <li className="pricing-feature"><span className="iconify" data-icon="lucide:check"></span> Pipeline DVC reproduzível</li>
                  <li className="pricing-feature"><span className="iconify" data-icon="lucide:check"></span> LSTM treinado · R² {champion.r2.toFixed(3)} · MAPE {champion.mape}</li>
                  <li className="pricing-feature"><span className="iconify" data-icon="lucide:check"></span> 66 endpoints FastAPI · p95 187 ms</li>
                  <li className="pricing-feature"><span className="iconify" data-icon="lucide:check"></span> Dashboard Next.js · 11 páginas</li>
                  <li className="pricing-feature"><span className="iconify" data-icon="lucide:check"></span> Agente ReAct + RAG avaliado por RAGAS</li>
                  <li className="pricing-feature"><span className="iconify" data-icon="lucide:check"></span> Observabilidade Prometheus + Grafana</li>
                  <li className="pricing-feature"><span className="iconify" data-icon="lucide:check"></span> CI/CD com Bandit, pip-audit e testes</li>
                </ul>
              </div>
            </div>
            <div className="bento-card">
              <div className="bc-grid"></div>
              <div className="bc-corner bc-corner-tl"></div>
              <div className="bc-corner bc-corner-br"></div>
              <div className="bc-shimmer"></div>
              <div className="bc-header">
                <div className="bc-live-dot"></div>
                <span className="bc-header-title">Próximos passos</span>
                <span className="bc-header-tag">roadmap</span>
              </div>
              <div className="bc-body">
                <ul className="pricing-features">
                  <li className="pricing-feature"><span className="iconify" data-icon="lucide:arrow-right"></span> Testar Modelos Alternativos</li>
                  <li className="pricing-feature"><span className="iconify" data-icon="lucide:arrow-right"></span> Deploy em Produção</li>
                  <li className="pricing-feature"><span className="iconify" data-icon="lucide:arrow-right"></span> Predições Multi-Alvo</li>
                  <li className="pricing-feature"><span className="iconify" data-icon="lucide:arrow-right"></span> Integração com Notícias e Papers</li>
                  <li className="pricing-feature"><span className="iconify" data-icon="lucide:arrow-right"></span> Controle de Acesso e Autenticação</li>
                  <li className="pricing-feature"><span className="iconify" data-icon="lucide:arrow-right"></span> Arquitetura de Microsserviços</li>
                  <li className="pricing-feature"><span className="iconify" data-icon="lucide:arrow-right"></span> Cobertura Multi-Ativo Big Tech</li>
                  <li className="pricing-feature"><span className="iconify" data-icon="lucide:arrow-right"></span> Controle de Custos de Infraestrutura</li>
                  <li className="pricing-feature"><span className="iconify" data-icon="lucide:arrow-right"></span> Produtização & Modelo de Negócio</li>
                  <li className="pricing-feature"><span className="iconify" data-icon="lucide:arrow-right"></span> Integrações de Mensagens e Notificações</li>
                </ul>
              </div>
            </div>
          </div>
          <div id="cta-section" style={{marginTop:'80px',position:'relative',textAlign:'center',padding:'72px 40px 64px',borderRadius:'24px',border:'1px solid rgba(20,184,166,0.15)',background:'radial-gradient(ellipse 80% 60% at 50% 0%, rgba(20,184,166,0.07) 0%, transparent 70%), rgba(255,255,255,0.02)',overflow:'hidden'}}>
            {/* ambient glow */}
            <div style={{position:'absolute',top:'-60px',left:'50%',transform:'translateX(-50%)',width:'400px',height:'200px',background:'radial-gradient(ellipse,rgba(20,184,166,0.18) 0%,transparent 70%)',pointerEvents:'none'}}/>
            {/* corner accents */}
            <div style={{position:'absolute',top:'16px',left:'16px',width:'20px',height:'20px',borderTop:'1px solid rgba(20,184,166,0.5)',borderLeft:'1px solid rgba(20,184,166,0.5)',borderRadius:'2px 0 0 0'}}/>
            <div style={{position:'absolute',bottom:'16px',right:'16px',width:'20px',height:'20px',borderBottom:'1px solid rgba(20,184,166,0.5)',borderRight:'1px solid rgba(20,184,166,0.5)',borderRadius:'0 0 2px 0'}}/>

            {/* stats strip */}
            <div style={{display:'flex',justifyContent:'center',gap:'48px',marginBottom:'48px',flexWrap:'wrap'}}>
              {[
                {value:`R² ${champion.r2.toFixed(3)}`,label:'acurácia preditiva'},
                {value:'187ms',label:'latência p95'},
                {value:'4 etapas',label:'26/26 critérios'},
                {value:'100%',label:'open-source'},
              ].map(s => (
                <div key={s.label} style={{textAlign:'center'}}>
                  <div style={{fontSize:'22px',fontWeight:800,background:'linear-gradient(135deg,#14b8a6,#06b6d4)',WebkitBackgroundClip:'text',WebkitTextFillColor:'transparent'}}>{s.value}</div>
                  <div style={{fontSize:'11px',color:'rgba(255,255,255,0.35)',marginTop:'4px',letterSpacing:'0.04em'}}>{s.label}</div>
                </div>
              ))}
            </div>

            <h2 className="cta-title" style={{marginBottom:'12px'}}>Obrigado · <span className="text-gradient">perguntas?</span></h2>
            <p className="cta-sub" style={{maxWidth:'480px',margin:'0 auto 36px'}}>Repositório, dashboard ao vivo e documentação completa abaixo.</p>

            <div className="cta-buttons" style={{justifyContent:'center'}}>
              <a className="btn-cta-primary" href="https://github.com/LucasTechAI/nvidia-mlops-platform" target="_blank" rel="noreferrer">
                <span className="iconify" data-icon="lucide:github"></span>
                Repositório
              </a>
              <a className="btn-cta-ghost" href="/predictions">
                <span className="iconify" data-icon="lucide:layout-dashboard"></span>
                Dashboard ao vivo
              </a>
            </div>

            {/* tag line */}
            <p style={{marginTop:'32px',fontSize:'11px',color:'rgba(255,255,255,0.2)',letterSpacing:'0.08em',textTransform:'uppercase'}}>
              FIAP Post-Tech MLET · Datathon Tech Challenge Fase 5 · 2025
            </p>


          </div>
        </div>
      </section>

      {/* ============================================================
          FOOTER — rendered globally via layout.tsx
          ============================================================ */}

      <Script src="/ds-assets/iconify_654a1ef798a3.js" strategy="afterInteractive" />
      <Script src="/ds-assets/resource_3fa48481346f.es" strategy="afterInteractive" />
    </div>
  );
}
