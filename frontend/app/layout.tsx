import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DocAI — Claude-Style Knowledge Assistant",
  description: "Advanced RAG knowledge assistant with Claude aesthetic, document attachments, and thinking trace.",
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
