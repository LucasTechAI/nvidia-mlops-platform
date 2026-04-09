import type { Metadata } from "next";
import Sidebar from "@/components/sidebar";
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
        <Sidebar />
        <main className="ml-64 min-h-screen p-8">{children}</main>
      </body>
    </html>
  );
}
