"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  LineChart,
  Brain,
  Search,
  ClipboardList,
  Bot,
  Activity,
} from "lucide-react";
import { clsx } from "clsx";

const NAV_ITEMS = [
  { href: "/predictions", label: "Stock Predictions", icon: BarChart3, emoji: "📊" },
  { href: "/metrics", label: "Model Metrics", icon: LineChart, emoji: "📈" },
  { href: "/model-schema", label: "Model Architecture", icon: Brain, emoji: "🧠" },
  { href: "/observability", label: "Observability", icon: Search, emoji: "🔍" },
  { href: "/evaluation", label: "Evaluation", icon: ClipboardList, emoji: "📋" },
  { href: "/agent", label: "AI Agent", icon: Bot, emoji: "🤖" },
];

export default function Sidebar() {
  const pathname = usePathname();

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
          Quick Stats
        </p>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-white/50">Accuracy</span>
            <span className="text-xs font-semibold text-nvidia">95.1%</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-white/50">R² Score</span>
            <span className="text-xs font-semibold text-nvidia">0.91</span>
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
