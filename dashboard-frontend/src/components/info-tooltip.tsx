import { Info } from "lucide-react";

export function InfoTooltip({ text }: { text: string }) {
  return (
    <div className="group relative inline-flex">
      <Info className="h-3.5 w-3.5 cursor-help text-white/25 transition hover:text-white/60" />
      <div className="pointer-events-none absolute bottom-full left-1/2 z-50 mb-2 w-60 -translate-x-1/2 rounded-lg border border-surface-border bg-[#0f0f1a] px-3 py-2 text-xs leading-relaxed text-white/70 opacity-0 shadow-xl transition-opacity group-hover:opacity-100">
        {text}
        <div className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-[#0f0f1a]" />
      </div>
    </div>
  );
}
