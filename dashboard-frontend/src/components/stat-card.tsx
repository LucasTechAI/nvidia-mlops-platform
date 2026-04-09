import { clsx } from "clsx";
import type { ReactNode } from "react";

interface StatCardProps {
  label: string;
  value: string | number;
  subtitle?: string;
  icon?: ReactNode;
  accentColor?: string;
  delta?: string;
  deltaType?: "positive" | "negative" | "neutral";
}

export default function StatCard({
  label,
  value,
  subtitle,
  icon,
  accentColor = "#76B900",
  delta,
  deltaType = "neutral",
}: StatCardProps) {
  return (
    <div className="stat-card group">
      <div
        className="absolute left-0 top-0 h-full w-1"
        style={{ background: accentColor }}
      />
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-white/40">
            {label}
          </p>
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
