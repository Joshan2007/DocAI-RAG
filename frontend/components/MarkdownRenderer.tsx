"use client";

import React, { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownRendererProps {
  content: string;
}

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  const formattedContent = useMemo(() => {
    if (!content) return "";

    // 1. Strip raw <thought> tags if they ever appear in content
    let clean = content
      .replace(/<thought>[\s\S]*?<\/thought>/gi, "")
      .replace(/<\/?thought>/gi, "");

    // 2. Extract inline [Doc: filename, Page X] or [📄 filename, pg. X] citations from body
    const docPages: Record<string, Set<string>> = {};

    clean = clean.replace(
      /\[(?:📄\s*)?(?:Doc:\s*)?([^,\]]+),\s*(?:Page|pg\.?)\s*([0-9,\s&]+)\]\s*/gi,
      (match, doc, pagesRaw) => {
        const docName = doc.trim();
        if (!docPages[docName]) {
          docPages[docName] = new Set();
        }
        const pages = pagesRaw.split(/[,&]/);
        for (const p of pages) {
          const pClean = p.trim().replace(/^p\./i, "").trim();
          if (pClean) {
            docPages[docName].add(pClean);
          }
        }
        return ""; // Cleanly remove clumsy inline citation from the sentence
      }
    );

    // 3. Fix awkward spacing before punctuation caused by removed brackets (e.g. "word . " -> "word. ")
    clean = clean.replace(/\s+([.,!?;:])/g, "$1").trim();

    // 4. If citations were extracted and no "### Sources" section already exists, append clean sources at the end
    const hasSourcesSection = /#+\s*Sources/i.test(clean);
    if (Object.keys(docPages).length > 0 && !hasSourcesSection) {
      let sourcesSection = "\n\n---\n\n### Sources\n";
      for (const [docName, pagesSet] of Object.entries(docPages)) {
        const sortedPages = Array.from(pagesSet).sort((a, b) => {
          const numA = parseInt(a, 10);
          const numB = parseInt(b, 10);
          if (!isNaN(numA) && !isNaN(numB)) return numA - numB;
          return a.localeCompare(b);
        });
        sourcesSection += `* **${docName}** — Page${sortedPages.length > 1 ? "s" : ""} ${sortedPages.join(", ")}\n`;
      }
      clean += sourcesSection;
    }

    return clean;
  }, [content]);

  return (
    <div className="prose prose-stone max-w-none text-[#1F1E1D] text-sm sm:text-base leading-relaxed">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => (
            <p className="mb-3 leading-relaxed text-[#1F1E1D] last:mb-0">
              {children}
            </p>
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-[#1F1E1D]">
              {children}
            </strong>
          ),
          em: ({ children }) => (
            <em className="italic text-[#383633]">{children}</em>
          ),
          ul: ({ children }) => (
            <ul className="my-2.5 ml-5 list-disc space-y-1.5 text-[#2E2C29]">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="my-2.5 ml-5 list-decimal space-y-1.5 text-[#2E2C29]">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="leading-relaxed pl-1">{children}</li>
          ),
          h1: ({ children }) => (
            <h1 className="text-xl sm:text-2xl font-serif font-bold text-[#1F1E1D] mt-5 mb-2.5 tracking-tight border-b border-[#E5E3DC] pb-1.5">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-lg sm:text-xl font-serif font-semibold text-[#1F1E1D] mt-4 mb-2 tracking-tight">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-base font-serif font-semibold text-[#1F1E1D] mt-4 mb-1.5">
              {children}
            </h3>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-[#CC785C] bg-[#F7F5EE] pl-3.5 py-1.5 my-2.5 rounded-r-lg italic text-[#52504C] text-sm">
              {children}
            </blockquote>
          ),
          hr: () => <hr className="my-4 border-[#E5E3DC]" />,
          code: ({ node, className, children, ...props }: any) => {
            const match = /language-(\w+)/.exec(className || "");
            const isInline = !match && !String(children).includes("\n");
            return isInline ? (
              <code
                className="bg-[#EFECE6] text-[#8C4A32] px-1.5 py-0.5 rounded text-xs font-mono font-medium"
                {...props}
              >
                {children}
              </code>
            ) : (
              <pre className="bg-[#2B2927] text-[#F5F3EC] p-3.5 rounded-xl overflow-x-auto text-xs font-mono my-3 shadow-inner">
                <code className={className} {...props}>
                  {children}
                </code>
              </pre>
            );
          },
          table: ({ children }) => (
            <div className="overflow-x-auto my-3 rounded-xl border border-[#E5E3DC]">
              <table className="min-w-full divide-y divide-[#E5E3DC] text-xs sm:text-sm">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-[#F5F3EC] font-semibold text-[#1F1E1D]">
              {children}
            </thead>
          ),
          tbody: ({ children }) => (
            <tbody className="divide-y divide-[#EFECE6] bg-white">
              {children}
            </tbody>
          ),
          tr: ({ children }) => <tr>{children}</tr>,
          th: ({ children }) => (
            <th className="px-3 py-2 text-left text-xs font-semibold uppercase tracking-wider text-[#6B6963]">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-3 py-2 text-xs sm:text-sm text-[#383633]">
              {children}
            </td>
          ),
        }}
      >
        {formattedContent}
      </ReactMarkdown>
    </div>
  );
}