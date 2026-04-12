"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import {
  BarChart3,
  LineChart,
  Brain,
  Search,
  ClipboardList,
  Bot,
  Activity,
  Home,
  GitBranch,
  Rocket,
  Shield,
} from "lucide-react";
import { clsx } from "clsx";
import { api } from "@/lib/api";

const NAV_ITEMS = [
  // 1. Abertura — contexto e visão geral
  { href: "/home", label: "Home", icon: Home },
  // 2. Como o sistema foi construído (pipeline)
  { href: "/architecture", label: "Project Architecture", icon: GitBranch },
  // 3. O cérebro — arquitetura do modelo LSTM
  { href: "/model-schema", label: "Model Architecture", icon: Brain },
  // 4. Prova de que funciona — métricas de treino/val/test
  { href: "/metrics", label: "Model Metrics", icon: LineChart },
  // 5. O resultado final — previsões de ações
  { href: "/predictions", label: "Stock Predictions", icon: BarChart3 },
  // 6. Validação independente — avaliação e explainability
  { href: "/evaluation", label: "Evaluation", icon: ClipboardList },
  // 7. Monitoramento em produção — drift, champion-challenger
  { href: "/observability", label: "Observability", icon: Search },
  // 8. Governança e SLAs
  { href: "/mlops", label: "MLOps & SLA", icon: Shield },
  // 9. Diferencial — agente AI com RAG
  { href: "/agent", label: "AI Agent", icon: Bot },
  // 10. Transparência operacional
  { href: "/logs", label: "System Logs", icon: Activity },
  // 11. Encerramento — visão de futuro
  { href: "/next-steps", label: "Next Steps", icon: Rocket },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [stats, setStats] = useState<{
    rmse: number | null;
    r2: number | null;
    mape: number | null;
    sharpe: number | null;
    dirAcc: number | null;
    status: string | null;
  }>({ rmse: null, r2: null, mape: null, sharpe: null, dirAcc: null, status: null });

  useEffect(() => {
    async function load() {
      try {
        const [info, health] = await Promise.all([
          api.model.info() as Promise<{
            test_metrics?: {
              rmse?: number;
              r2_score?: number;
              mape?: number;
              sharpe_ratio?: number;
              directional_accuracy?: number;
            };
          }>,
          api.health.check() as Promise<{ status?: string }>,
        ]);
        const m = info.test_metrics || {};
        setStats({
          rmse: m.rmse ?? null,
          r2: m.r2_score ?? null,
          mape: m.mape ?? null,
          sharpe: m.sharpe_ratio ?? null,
          dirAcc: m.directional_accuracy ?? null,
          status: health.status as string ?? null,
        });
      } catch {
        /* silent */
      }
    }
    load();
  }, []);

  return (
    <aside className="fixed left-0 top-0 z-40 flex h-screen w-64 flex-col border-r border-surface-border bg-surface-card">
      {/* Logo */}
      <div className="flex items-center gap-3 border-b border-surface-border px-5 py-5">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-nvidia/20">
          <span className="text-xl font-bold text-nvidia">N</span>
        </div>
        <div>
          <h1 className="text-sm font-bold text-white">NVIDIA MLOps</h1>
          <p className="text-xs text-white/40">Stock Prediction Platform</p>
        </div>
      </div>

      {/* Status */}
      <div className="border-b border-surface-border px-5 py-3">
        <div className="flex items-center gap-2">
          <Activity className="h-3.5 w-3.5 text-nvidia" />
          <span className="text-xs font-medium text-nvidia">Model Active</span>
          <span className="ml-auto h-2 w-2 animate-pulse rounded-full bg-nvidia" />
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-4">
        {NAV_ITEMS.map((item) => {
          const isActive =
            pathname === item.href || pathname.startsWith(item.href + "/");
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={clsx(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200",
                isActive
                  ? "border border-nvidia/30 bg-nvidia/10 text-nvidia glow-green"
                  : "text-white/60 hover:bg-surface-hover hover:text-white"
              )}
            >
              <Icon className="h-4 w-4 flex-shrink-0" />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      {/* Quick Stats */}
      <div className="border-t border-surface-border px-5 py-4">
        <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-white/30">
          Model Stats
        </p>
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-xs text-white/50">Status</span>
            <span className={clsx("text-xs font-semibold", stats.status === "healthy" ? "text-green-400" : stats.status === "degraded" ? "text-amber-400" : "text-white/30")}>
              {stats.status ? (stats.status === "healthy" ? "● Healthy" : stats.status === "degraded" ? "● Degraded" : "● " + stats.status) : "—"}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-white/50">R²</span>
            <span className="text-xs font-semibold text-nvidia">{stats.r2 != null ? stats.r2.toFixed(3) : "—"}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-white/50">RMSE</span>
            <span className="text-xs font-semibold text-nvidia">{stats.rmse != null ? `$${stats.rmse.toFixed(2)}` : "—"}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-white/50">MAPE</span>
            <span className="text-xs font-semibold text-nvidia">{stats.mape != null ? `${stats.mape.toFixed(1)}%` : "—"}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-white/50">Sharpe</span>
            <span className="text-xs font-semibold text-nvidia">{stats.sharpe != null ? stats.sharpe.toFixed(2) : "—"}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-white/50">Dir. Acc.</span>
            <span className="text-xs font-semibold text-nvidia">{stats.dirAcc != null ? `${stats.dirAcc.toFixed(1)}%` : "—"}</span>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-surface-border px-5 py-3">
        <p className="text-[10px] text-white/20">
          v1.0.0 • Next.js Dashboard
        </p>
      </div>
    </aside>
  );
}
