"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
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
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { clsx } from "clsx";

const NAV_ITEMS = [
  { href: "/home", label: "Home", icon: Home },
  { href: "/architecture", label: "Project Architecture", icon: GitBranch },
  { href: "/model-schema", label: "Model Architecture", icon: Brain },
  { href: "/metrics", label: "Model Metrics", icon: LineChart },
  { href: "/predictions", label: "Stock Predictions", icon: BarChart3 },
  { href: "/evaluation", label: "Evaluation", icon: ClipboardList },
  { href: "/observability", label: "Observability", icon: Search },
  { href: "/mlops", label: "MLOps & SLA", icon: Shield },
  { href: "/agent", label: "AI Agent", icon: Bot },
  { href: "/logs", label: "System Logs", icon: Activity },
  { href: "/next-steps", label: "Next Steps", icon: Rocket },
];

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

export default function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const pathname = usePathname();

  return (
    <aside
      className={clsx(
        "fixed left-0 top-0 z-40 flex h-screen flex-col border-r transition-all duration-300",
        "border-teal-500/10 bg-[#0a0b0d]",
        collapsed ? "w-16" : "w-64"
      )}
    >
      {/* Logo */}
      <div
        className={clsx(
          "flex items-center border-b border-teal-500/10 py-5 transition-all duration-300",
          collapsed ? "justify-center px-0" : "px-5"
        )}
      >
        {collapsed ? (
          <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-lg bg-teal-500/15 ring-1 ring-teal-500/20">
            <span className="text-base font-bold text-teal-400">T</span>
          </div>
        ) : (
          <Link href="/home" className="min-w-0">
            <Image
              src="/logo.png"
              alt="TradeOps"
              width={200}
              height={80}
              className="h-20 w-auto object-contain"
              priority
            />
            <p className="mt-0.5 truncate text-xs text-white/40">Stock Prediction Platform</p>
          </Link>
        )}
      </div>

      {/* Status */}
      <div
        className={clsx(
          "border-b border-teal-500/10 py-3 transition-all duration-300",
          collapsed ? "flex justify-center px-0" : "px-5"
        )}
      >
        {collapsed ? (
          <span
            title="Model Active"
            className="flex h-7 w-7 items-center justify-center rounded-full bg-teal-500/10"
          >
            <span className="h-2 w-2 animate-pulse rounded-full bg-teal-400" />
          </span>
        ) : (
          <div className="flex items-center gap-2">
            <Activity className="h-3.5 w-3.5 text-teal-400" />
            <span className="text-xs font-medium text-teal-400">Model Active</span>
            <span className="ml-auto h-2 w-2 animate-pulse rounded-full bg-teal-400" />
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 overflow-y-auto py-4 px-2">
        {NAV_ITEMS.map((item) => {
          const isActive =
            pathname === item.href || pathname.startsWith(item.href + "/");
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              title={collapsed ? item.label : undefined}
              className={clsx(
                "flex items-center rounded-lg py-2.5 text-sm font-medium transition-all duration-200",
                collapsed ? "justify-center px-0" : "gap-3 px-3",
                isActive
                  ? "border border-teal-500/20 bg-teal-500/10 text-teal-400 shadow-[0_0_12px_rgba(20,184,166,0.12)]"
                  : "text-white/50 hover:bg-white/5 hover:text-white/90"
              )}
            >
              <Icon className="h-4 w-4 flex-shrink-0" />
              {!collapsed && <span className="truncate uppercase tracking-wider">{item.label}</span>}
            </Link>
          );
        })}
      </nav>


      {/* Footer + Collapse toggle */}
      <div className="border-t border-teal-500/10 px-3 py-3 flex items-center justify-between">
        {!collapsed && (
          <p className="text-[10px] text-white/20">v1.0.0 · MLOps Platform</p>
        )}
        <button
          onClick={onToggle}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          className={clsx(
            "flex items-center justify-center rounded-lg p-1.5 text-white/30 transition-all duration-200 hover:bg-white/5 hover:text-teal-400",
            collapsed && "mx-auto"
          )}
        >
          {collapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </button>
      </div>
    </aside>
  );
}
