import type { Metadata } from "next";
import TopNav from "@/components/top-nav";
import Footer from "@/components/footer";
import HeroBackground from "@/components/hero-background";
import "./globals.css";

export const metadata: Metadata = {
  title: "TradeOps Dashboard",
  description: "Stock Prediction Platform — TradeOps",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-surface text-white antialiased">
        <HeroBackground />
        <div className="relative z-10">
          <TopNav />
          <main className="min-h-screen p-8">{children}</main>
          <Footer />
        </div>
      </body>
    </html>
  );
}
