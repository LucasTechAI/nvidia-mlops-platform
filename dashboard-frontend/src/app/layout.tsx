import type { Metadata } from "next";
import TopNav from "@/components/top-nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "NVIDIA MLOps Dashboard",
  description: "Stock Prediction Platform — NVIDIA MLOps",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-surface text-white antialiased">
        <TopNav />
        <main className="min-h-screen p-8">{children}</main>
      </body>
    </html>
  );
}
