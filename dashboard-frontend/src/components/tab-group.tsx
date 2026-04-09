"use client";

import { clsx } from "clsx";
import { useState } from "react";

interface Tab {
  id: string;
  label: string;
  icon?: string;
}

interface TabGroupProps {
  tabs: Tab[];
  children: (activeTab: string) => React.ReactNode;
}

export default function TabGroup({ tabs, children }: TabGroupProps) {
  const [active, setActive] = useState(tabs[0]?.id ?? "");

  return (
    <div>
      <div className="mb-6 flex gap-1 border-b border-surface-border">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActive(tab.id)}
            className={clsx(
              "px-4 py-2.5 text-sm font-medium transition-all duration-200",
              active === tab.id ? "tab-active" : "tab-inactive"
            )}
          >
            {tab.icon && <span className="mr-1.5">{tab.icon}</span>}
            {tab.label}
          </button>
        ))}
      </div>
      <div>{children(active)}</div>
    </div>
  );
}
