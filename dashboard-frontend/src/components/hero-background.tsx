"use client";

import { usePathname } from "next/navigation";
import HeroCanvas from "./hero-canvas";
import "./hero-background.css";

/** Full-viewport animated background (canvas particles + orbs + stars + orbits).
 *  Skips rendering on "/" because the home page has its own hero section. */
export default function HeroBackground() {
  const pathname = usePathname();
  if (pathname === "/" || pathname === "/home") return null;

  return (
    <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
      <HeroCanvas className="opacity-30" />

      {/* Blurred colour orbs */}
      <div className="hero-orb hero-orb-1" />
      <div className="hero-orb hero-orb-2" />
      <div className="hero-orb hero-orb-3" />

      {/* Shooting stars */}
      <div className="hero-stars">
        <div className="hero-star hero-star-1" />
        <div className="hero-star hero-star-2" />
        <div className="hero-star hero-star-3" />
        <div className="hero-star hero-star-4" />
        <div className="hero-star hero-star-5" />
        <div className="hero-star hero-star-6" />
      </div>

      {/* Orbiting rings with dots */}
      <div className="hero-orbits">
        <div className="hero-orbit hero-orbit-1"><div className="hero-orbit-dot" /></div>
        <div className="hero-orbit hero-orbit-2"><div className="hero-orbit-dot" /></div>
        <div className="hero-orbit hero-orbit-3"><div className="hero-orbit-dot" /></div>
      </div>
    </div>
  );
}
