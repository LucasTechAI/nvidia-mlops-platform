"use client";

import { useState, useEffect } from "react";
import Sidebar from "@/components/sidebar";

const STORAGE_KEY = "sidebar-collapsed";

export default function SidebarLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [collapsed, setCollapsed] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored !== null) setCollapsed(stored === "true");
    setMounted(true);
  }, []);

  function toggle() {
    setCollapsed((prev) => {
      localStorage.setItem(STORAGE_KEY, String(!prev));
      return !prev;
    });
  }

  // Avoid layout shift on SSR — render with default (expanded) until mounted
  const sidebarWidth = mounted && collapsed ? "ml-16" : "ml-64";

  return (
    <>
      <Sidebar collapsed={mounted ? collapsed : false} onToggle={toggle} />
      <main
        className={`${sidebarWidth} min-h-screen p-8 transition-all duration-300`}
      >
        {children}
      </main>
    </>
  );
}
