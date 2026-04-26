import Link from "next/link";

const NAV_LINKS = [
  { href: "/home", label: "Home" },
  { href: "/predictions", label: "Predictions" },
  { href: "/metrics", label: "Metrics" },
  { href: "/evaluation", label: "Evaluation" },
  { href: "/observability", label: "Observability" },
  { href: "/agent", label: "Agent IA" },
];

const MORE_LINKS = [
  { href: "/mlops", label: "MLOps" },
  { href: "/logs", label: "Logs" },
  { href: "/model-schema", label: "Model Schema" },
  { href: "/architecture", label: "Architecture" },
  { href: "/next-steps", label: "Next Steps" },
];

const RESOURCE_LINKS = [
  {
    href: "https://github.com/LucasTechAI/nvidia-mlops-platform",
    label: "GitHub",
    external: true,
  },
  {
    href: "https://github.com/LucasTechAI/nvidia-mlops-platform#-documentation",
    label: "Documentação",
    external: true,
  },
  {
    href: "https://www.linkedin.com/in/lucas-mendes-barbosa/",
    label: "LinkedIn",
    external: true,
  },
];

export default function Footer() {
  return (
    <footer className="mt-16 border-t border-white/10 bg-black/30 backdrop-blur-sm">
      <div className="mx-auto max-w-7xl px-8 py-12">
        <div className="grid grid-cols-1 gap-10 sm:grid-cols-2 lg:grid-cols-4">
          {/* Brand */}
          <div className="lg:col-span-1">
            <div className="mb-4 flex items-center gap-3">
              <img
                src="/lucas.png"
                alt="Lucas"
                className="h-9 w-9 flex-shrink-0 rounded-full border border-teal-500/30 object-cover object-top"
              />
              <div>
                <div className="text-sm font-bold tracking-widest text-white">
                  NVDA · MLOps
                </div>
                <div className="text-[10px] text-teal-400/70">
                  por Lucas · FIAP Post-Tech MLET
                </div>
              </div>
            </div>
            <p className="text-xs leading-relaxed text-white/40">
              Datathon FIAP Post-Tech MLET — Tech Challenge Fase&nbsp;5.
              Plataforma MLOps end-to-end para forecasting de NVDA com LSTM,
              MLflow, agente RAG e observabilidade.
            </p>
          </div>

          {/* Navigation */}
          <div>
            <div className="mb-4 text-xs font-semibold uppercase tracking-widest text-teal-400">
              Navegação
            </div>
            <ul className="space-y-2">
              {NAV_LINKS.map((l) => (
                <li key={l.href}>
                  <Link
                    href={l.href}
                    className="text-sm text-white/50 transition-colors hover:text-teal-400"
                  >
                    {l.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* More */}
          <div>
            <div className="mb-4 text-xs font-semibold uppercase tracking-widest text-teal-400">
              Mais
            </div>
            <ul className="space-y-2">
              {MORE_LINKS.map((l) => (
                <li key={l.href}>
                  <Link
                    href={l.href}
                    className="text-sm text-white/50 transition-colors hover:text-teal-400"
                  >
                    {l.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Resources */}
          <div>
            <div className="mb-4 text-xs font-semibold uppercase tracking-widest text-teal-400">
              Recursos
            </div>
            <ul className="space-y-2">
              {RESOURCE_LINKS.map((l) => (
                <li key={l.href}>
                  <a
                    href={l.href}
                    target="_blank"
                    rel="noreferrer"
                    className="text-sm text-white/50 transition-colors hover:text-teal-400"
                  >
                    {l.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="mt-10 flex flex-col items-center justify-between gap-3 border-t border-white/10 pt-6 sm:flex-row">
          <p className="text-xs text-white/30">
            © Lucas · Datathon FIAP MLET · Tech Challenge Fase 5
          </p>
          <div className="flex gap-4">
            <a
              href="https://github.com/LucasTechAI"
              target="_blank"
              rel="noreferrer"
              className="text-white/30 transition-colors hover:text-teal-400"
              aria-label="GitHub"
            >
              <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
              </svg>
            </a>
            <a
              href="https://www.linkedin.com/in/lucas-mendes-barbosa/"
              target="_blank"
              rel="noreferrer"
              className="text-white/30 transition-colors hover:text-teal-400"
              aria-label="LinkedIn"
            >
              <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
              </svg>
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
