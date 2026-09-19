import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DocAI — AI Knowledge Assistant",
  description: "Advanced production-grade RAG knowledge assistant with multi-format document support, hybrid retrieval, and real-time thinking traces.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-[#FAF9F5] text-[#1F1E1D] min-h-screen antialiased selection:bg-[#F0E6DE] selection:text-[#CC785C]">
        {children}
      </body>
    </html>
  );
}
