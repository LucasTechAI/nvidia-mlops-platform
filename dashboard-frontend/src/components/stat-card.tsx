"use client";

import { clsx } from "clsx";
import { Info } from "lucide-react";
import { type ReactNode, useState } from "react";

interface StatCardProps {
  label: string;
  value: string | number;
  subtitle?: string;
  icon?: ReactNode;
  accentColor?: string;
  delta?: string;
  deltaType?: "positive" | "negative" | "neutral";
  tooltip?: string;
}

export default function StatCard({
  label,
  value,
  subtitle,
  icon,
  accentColor = "#76B900",
  delta,
  deltaType = "neutral",
  tooltip,
}: StatCardProps) {
  const [showTip, setShowTip] = useState(false);

  return (
    <div className="stat-card group">
      <div
        className="absolute left-0 top-0 h-full w-1 rounded-l-xl"
        style={{ background: accentColor }}
      />
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-1.5">
            <p className="text-xs font-medium uppercase tracking-wider text-white/40">
              {label}
            </p>
            {tooltip && (
              <div
                className="relative"
                onMouseEnter={() => setShowTip(true)}
                onMouseLeave={() => setShowTip(false)}
              >
                <Info className="h-3 w-3 cursor-help text-white/25 transition-colors hover:text-white/60" />
                {showTip && (
                  <div className="absolute bottom-full left-1/2 z-50 mb-2 w-56 -translate-x-1/2 rounded-lg border border-surface-border bg-[#1a1c24] px-3 py-2 text-[11px] font-normal normal-case tracking-normal text-white/70 shadow-xl">
                    {tooltip}
                    <div className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-[#1a1c24]" />
                  </div>
                )}
              </div>
            )}
          </div>
          <p className="mt-1 text-2xl font-bold text-white">{value}</p>
          {delta && (
            <p
              className={clsx(
                "mt-1 text-xs font-medium",
                deltaType === "positive" && "text-green-400",
                deltaType === "negative" && "text-red-400",
                deltaType === "neutral" && "text-white/50"
              )}
            >
              {delta}
            </p>
          )}
          {subtitle && (
            <p className="mt-1 text-xs text-white/40">{subtitle}</p>
          )}
        </div>
        {icon && (
          <div
            className="flex h-10 w-10 items-center justify-center rounded-lg"
            style={{ background: `${accentColor}20` }}
          >
            {icon}
          </div>
        )}
      </div>
    </div>
  );
}
