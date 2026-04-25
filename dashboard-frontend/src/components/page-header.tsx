import type { LucideIcon } from "lucide-react";
import "@/app/home/design-system.css";

interface PageHeaderProps {
  label: string;
  title: string;
  gradient: string;
  subtitle: string;
  icon: LucideIcon;
}

export function PageHeader({ label, title, gradient, subtitle, icon: Icon }: PageHeaderProps) {
  return (
    <div style={{ textAlign: "center" }}>
      <span
        style={{
          display: "inline-block",
          fontFamily: "'Inter', sans-serif",
          fontSize: "0.68rem",
          fontWeight: 500,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          color: "#14B8A6",
          marginBottom: "12px",
        }}
      >
        {label}
      </span>
      <h2
        style={{
          fontFamily: "'Outfit', sans-serif",
          fontWeight: 200,
          letterSpacing: "-0.03em",
          lineHeight: 1.15,
          color: "#FAFAFA",
          fontSize: "clamp(1.8rem, 3vw, 2.2rem)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: "12px",
        }}
      >
        <Icon style={{ color: "#14B8A6", width: "28px", height: "28px", flexShrink: 0 }} />
        {title && `${title} `}
        <span
          style={{
            background: "linear-gradient(135deg, #2DD4BF 0%, #14B8A6 50%, #5EEAD4 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            backgroundClip: "text",
          }}
        >
          {gradient}
        </span>
      </h2>
      <p
        style={{
          fontFamily: "'Inter', sans-serif",
          color: "rgba(255,255,255,0.45)",
          fontWeight: 300,
          fontSize: "0.86rem",
          maxWidth: "540px",
          margin: "12px auto 0",
        }}
      >
        {subtitle}
      </p>
    </div>
  );
}
