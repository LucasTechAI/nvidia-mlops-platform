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
} from "lucide-react";
import { clsx } from "clsx";

const NAV_ITEMS = [
  { href: "/home",          label: "Home",          icon: Home },
  { href: "/architecture",  label: "Architecture",  icon: GitBranch },
  { href: "/model-schema",  label: "Model",         icon: Brain },
  { href: "/metrics",       label: "Metrics",       icon: LineChart },
  { href: "/predictions",   label: "Predictions",   icon: BarChart3 },
  { href: "/evaluation",    label: "Evaluation",    icon: ClipboardList },
  { href: "/observability", label: "Observability", icon: Search },
  { href: "/mlops",         label: "MLOps",         icon: Shield },
  { href: "/agent",         label: "Agent",         icon: Bot },
  { href: "/logs",          label: "Logs",          icon: Activity },
  { href: "/next-steps",    label: "Next Steps",    icon: Rocket },
];

export default function TopNav() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 flex h-14 items-center border-b border-teal-500/10 bg-[#0a0b0d]">
      {/* Logo */}
      <div className="flex flex-shrink-0 items-center border-r border-teal-500/10 px-5 h-full">
        <Link href="/home">
          <Image
            src="/logo.png"
            alt="TradeOps"
            width={200}
            height={80}
            className="h-20 w-auto object-contain"
            priority
          />
        </Link>
      </div>

      {/* Nav items — centered */}
      <nav className="flex flex-1 items-center justify-center gap-0 overflow-x-auto scrollbar-hide h-full">
        {NAV_ITEMS.map((item) => {
          const isActive =
            pathname === item.href || pathname.startsWith(item.href + "/");
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={clsx(
                "relative flex h-full flex-shrink-0 items-center gap-1 px-2.5 text-[11px] font-medium transition-colors duration-200",
                isActive
                  ? "text-teal-400"
                  : "text-white/50 hover:text-white/90"
              )}
            >
              <span className="whitespace-nowrap uppercase tracking-wider">{item.label}</span>
              {/* Active underline */}
              {isActive && (
                <span className="absolute bottom-0 left-1.5 right-1.5 h-0.5 rounded-full bg-teal-400" />
              )}
            </Link>
          );
        })}
      </nav>

      {/* Status indicator */}
      <div className="flex flex-shrink-0 items-center gap-2 border-l border-teal-500/10 px-4 h-full">
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-teal-400" />
        <span className="hidden text-[11px] font-medium text-teal-400 sm:block whitespace-nowrap">
          Live
        </span>
      </div>
    </header>
  );
}
