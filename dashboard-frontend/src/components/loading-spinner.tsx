"use client";

import { Loader2 } from "lucide-react";

export default function LoadingSpinner({ text = "Loading..." }: { text?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16">
      <Loader2 className="h-8 w-8 animate-spin text-nvidia" />
      <p className="text-sm text-white/50">{text}</p>
    </div>
  );
}
