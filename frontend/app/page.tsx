"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Paperclip,
  ArrowUp,
  ChevronDown,
  Sparkles,
  FileText,
  X,
  CheckCircle2,
  Brain,
  BookOpen,
  ExternalLink,
} from "lucide-react";
import MarkdownRenderer from "../components/MarkdownRenderer";

const getApiUrl = (path: string) => {
  if (process.env.NEXT_PUBLIC_API_URL) {
    return `${process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, "")}${path}`;
  }
  if (
    typeof window !== "undefined" &&
    (window.location.hostname === "localhost" ||
      window.location.hostname === "127.0.0.1")
  ) {
    return `http://127.0.0.1:8000${path}`;
  }
  return path;
};

interface DocumentMeta {
  filename: string;
  total_chunks?: number;
  total_pages?: number;
  size_bytes?: number;
}

interface Citation {
  source: string;
  page: number;
  chunk_id: string;
  score: number;
  text_snippet: string;
  retrieval_method: string;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  thinkingContent?: string;
  thinkingDuration?: number;
  isThinkingDone?: boolean;
  citations?: Citation[];
  evaluation?: {
    relevance: number;
    groundedness: number;
    citation_count: number;
    retrieval_ms: number;
    generation_ms: number;
    is_grounded: boolean;
    summary: string;
  };
}

export interface ModelOption {
  id: string;
  name: string;
  desc: string;
}

const AVAILABLE_MODELS: ModelOption[] = [
  { id: "gemini-1.5-flash", name: "Gemini 1.5 Flash", desc: "Recommended • 1,500 req/day quota" },
  { id: "gemini-2.0-flash", name: "Gemini 2.0 Flash", desc: "Next-Gen Flash • 1,500 req/day quota" },
  { id: "gemini-1.5-pro", name: "Gemini 1.5 Pro", desc: "Deep reasoning • 50 req/day quota" },
  { id: "gemini-2.5-flash", name: "Gemini 2.5 Flash", desc: "Experimental • 20 req/day quota" },
];

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedModel, setSelectedModel] = useState("gemini-1.5-flash");
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);
  const [attachedFiles, setAttachedFiles] = useState<DocumentMeta[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [openThinkingIds, setOpenThinkingIds] = useState<Record<string, boolean>>({});
  const [openSourceIds, setOpenSourceIds] = useState<Record<string, boolean>>({});
  const [geminiApiKey, setGeminiApiKey] = useState("");
  const [isKeyModalOpen, setIsKeyModalOpen] = useState(false);
  const [geminiKeyInput, setGeminiKeyInput] = useState("");
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Load API key & selected model from localStorage on mount
  useEffect(() => {
    const savedGemini = localStorage.getItem("docai_gemini_api_key");
    if (savedGemini) {
      setGeminiApiKey(savedGemini);
      setGeminiKeyInput(savedGemini);
    }
    const savedModel = localStorage.getItem("docai_selected_model");
    if (savedModel && AVAILABLE_MODELS.some((m) => m.id === savedModel)) {
      setSelectedModel(savedModel);
    }
  }, []);

  // Auto scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming]);

  // Fetch initial documents
  useEffect(() => {
    fetch(getApiUrl("/api/documents"))
      .then((res) => res.json())
      .then((data) => {
        if (data.documents) {
          setAttachedFiles(data.documents);
        }
      })
      .catch(() => {});
  }, []);

  // Reusable upload function for file picker and drag-and-drop
  const uploadFiles = async (files: FileList | File[]) => {
    if (!files || files.length === 0) return;

    setIsUploading(true);
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }

    try {
      const res = await fetch(getApiUrl("/api/upload"), {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        throw new Error(`Upload returned status ${res.status}`);
      }
      const data = await res.json();
      if (data.uploaded) {
        const successes = data.uploaded.filter(
          (u: any) => u.status === "success"
        );
        const failures = data.uploaded.filter(
          (u: any) => u.status === "error"
        );

        if (successes.length > 0) {
          setAttachedFiles((prev) => {
            const newFiles = successes.filter(
              (u: any) => !prev.some((p) => p.filename === u.filename)
            );
            return [...prev, ...newFiles];
          });
        }

        if (failures.length > 0) {
          alert(`Failed to attach: ${failures.map((f: any) => f.filename).join(", ")}\nError: ${failures[0].error}`);
        }
      }
    } catch (err: any) {
      console.error("Upload failed", err);
      alert(`Document upload failed: ${err.message || "Could not connect to backend server"}`);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) uploadFiles(e.target.files);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      uploadFiles(e.dataTransfer.files);
    }
  };

  // Remove individual document via X button
  const handleRemoveDoc = async (filename: string) => {
    try {
      await fetch(getApiUrl(`/api/documents/${encodeURIComponent(filename)}`), {
        method: "DELETE",
      });
      setAttachedFiles((prev) => prev.filter((f) => f.filename !== filename));
    } catch (err) {
      console.error("Delete doc failed", err);
    }
  };

  // Submit Message
  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!input.trim() || isStreaming) return;

    const userText = input.trim();
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    const userMessage: Message = {
      id: "user-" + Date.now(),
      role: "user",
      content: userText,
    };

    const assistantMsgId = "asst-" + Date.now();
    const initialAssistantMsg: Message = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      thinkingContent: "",
      thinkingDuration: 0,
      isThinkingDone: false,
    };

    setMessages((prev) => [...prev, userMessage, initialAssistantMsg]);
    setIsStreaming(true);
    setOpenThinkingIds((prev) => ({ ...prev, [assistantMsgId]: true }));

    try {
      const response = await fetch(getApiUrl("/api/chat"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userText,
          model: selectedModel,
          api_key: geminiApiKey || undefined,
        }),
      });

      if (!response.body) {
        throw new Error("No response body");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const eventStr of events) {
          if (!eventStr.trim()) continue;
          const lines = eventStr.split("\n");
          let eventType = "";
          let dataStr = "";

          for (const line of lines) {
            if (line.startsWith("event: ")) {
              eventType = line.replace("event: ", "").trim();
            } else if (line.startsWith("data: ")) {
              dataStr = line.replace("data: ", "").trim();
            }
          }

          if (!dataStr) continue;
          const payload = JSON.parse(dataStr);

          if (eventType === "thinking_start") {
            setOpenThinkingIds((prev) => ({ ...prev, [assistantMsgId]: true }));
          } else if (eventType === "thinking_token") {
            setMessages((prev) =>
              prev.map((msg) => {
                if (msg.id === assistantMsgId) {
                  return {
                    ...msg,
                    thinkingContent: (msg.thinkingContent || "") + payload.token,
                  };
                }
                return msg;
              })
            );
          } else if (eventType === "thinking_end") {
            setMessages((prev) =>
              prev.map((msg) => {
                if (msg.id === assistantMsgId) {
                  return {
                    ...msg,
                    thinkingDuration: payload.duration,
                    isThinkingDone: true,
                  };
                }
                return msg;
              })
            );
            // Smoothly collapse thinking drawer once thoughts finish
            setOpenThinkingIds((prev) => ({ ...prev, [assistantMsgId]: false }));
          } else if (eventType === "token") {
            setMessages((prev) =>
              prev.map((msg) => {
                if (msg.id === assistantMsgId) {
                  return {
                    ...msg,
                    content: msg.content + payload.text,
                    isThinkingDone: true,
                  };
                }
                return msg;
              })
            );
          } else if (eventType === "sources") {
            setMessages((prev) =>
              prev.map((msg) => {
                if (msg.id === assistantMsgId) {
                  return {
                    ...msg,
                    citations: payload.citations,
                    evaluation: payload.evaluation,
                  };
                }
                return msg;
              })
            );
          }
        }
      }
    } catch (err: any) {
      setMessages((prev) =>
        prev.map((msg) => {
          if (msg.id === assistantMsgId) {
            return {
              ...msg,
              content: msg.content + `\n\n[Connection Error: ${err.message}]`,
            };
          }
          return msg;
        })
      );
    } finally {
      setIsStreaming(false);
    }
  };

  // Auto resize textarea
  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-[#FAF9F5] text-[#1F1E1D]">
      {/* Minimal Top Navbar */}
      <header className="sticky top-0 z-30 flex items-center justify-between px-6 py-3.5 bg-[#FAF9F5]/90 backdrop-blur border-b border-[#ECEAE4]">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-[#CC785C] flex items-center justify-center text-white font-serif font-bold text-lg shadow-sm">
            D
          </div>
          <div>
            <h1 className="font-serif font-semibold text-lg tracking-tight text-[#1F1E1D]">
              DocAI
            </h1>
            <p className="text-xs text-[#82807A]">Mini AI Knowledge Assistant</p>
          </div>
        </div>

        {/* Right Status: API Key Trigger Only (Clean Header) */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setIsKeyModalOpen(true)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full border text-xs font-medium transition-all ${
              geminiApiKey
                ? "bg-[#F3F8F2] border-[#C2E0C6] text-[#2E7D32] hover:bg-[#E8F5E9]"
                : "bg-[#FFF9F0] border-[#FCE2BF] text-[#B45309] hover:bg-[#FEF3C7]"
            }`}
            title="Configure your Google Gemini API Key"
          >
            <span
              className={`w-2 h-2 rounded-full ${
                geminiApiKey ? "bg-[#2E7D32]" : "bg-[#F59E0B]"
              }`}
            ></span>
            <span>{geminiApiKey ? "Gemini Key Active" : "Set Gemini Key"}</span>
          </button>
        </div>
      </header>

      {/* Main Chat Area */}
      <main className="flex-1 max-w-4xl w-full mx-auto px-4 sm:px-6 py-8 flex flex-col justify-between">
        {messages.length === 0 ? (
          /* Empty State - Minimal Aesthetic */
          <div className="flex-1 flex flex-col items-center justify-center text-center my-auto py-12 animate-fade-in-up">
            <div className="w-14 h-14 rounded-2xl bg-[#F0E6DE] text-[#CC785C] flex items-center justify-center mb-6 shadow-sm">
              <Sparkles className="w-7 h-7" />
            </div>
            <h2 className="text-3xl sm:text-4xl font-serif font-normal text-[#1F1E1D] tracking-tight mb-3">
              What would you like to explore?
            </h2>
            <p className="text-sm sm:text-base text-[#6B6963] max-w-md mx-auto mb-8 font-sans">
              Attach any document, ask complex questions, and inspect verified step-by-step thinking traces.
            </p>

            {/* Document Badges with Individual X Deletion */}
            {attachedFiles.length > 0 && (
              <div className="max-w-xl w-full mb-6">
                <p className="text-xs text-[#82807A] mb-2 font-medium">Uploaded Documents:</p>
                <div className="flex flex-wrap gap-2 justify-center">
                  {attachedFiles.map((doc) => (
                    <div
                      key={doc.filename}
                      className="group flex items-center gap-2 px-3 py-1.5 rounded-xl bg-white border border-[#E5E3DC] text-xs text-[#383633] shadow-sm hover:border-[#CC785C]/50 transition-all"
                    >
                      <FileText className="w-3.5 h-3.5 text-[#CC785C]" />
                      <span className="font-medium max-w-[200px] truncate" title={doc.filename}>
                        {doc.filename}
                      </span>
                      {doc.total_pages && (
                        <span className="text-[11px] text-[#82807A]">
                          ({doc.total_pages} pg)
                        </span>
                      )}
                      <button
                        onClick={() => handleRemoveDoc(doc.filename)}
                        className="text-[#9E9B93] hover:text-[#C55030] hover:bg-[#FEE2E2] p-1 rounded-md transition-colors"
                        title="Delete this document"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Message Stream List */
          <div className="space-y-6 mb-6">
            {messages.map((msg) => (
              <div key={msg.id} className="space-y-2 animate-fade-in-up">
                {msg.role === "user" ? (
                  /* User Bubble */
                  <div className="flex justify-end">
                    <div className="max-w-2xl bg-[#EBE7DF] text-[#1F1E1D] rounded-2xl px-5 py-3.5 text-sm sm:text-base font-sans leading-relaxed shadow-sm">
                      {msg.content}
                    </div>
                  </div>
                ) : (
                  /* Assistant Bubble */
                  <div className="flex gap-4 items-start">
                    <div className="w-8 h-8 rounded-lg bg-[#CC785C] flex-shrink-0 flex items-center justify-center text-white font-serif font-bold text-sm mt-1">
                      D
                    </div>
                    <div className="flex-1 space-y-3 overflow-hidden">
                      {/* Dynamic Thinking Accordion */}
                      {(msg.thinkingContent || !msg.isThinkingDone) && (
                        <div className="rounded-xl border border-[#E5E3DC] bg-[#F7F5EE] overflow-hidden transition-all">
                          <button
                            onClick={() =>
                              setOpenThinkingIds((prev) => ({
                                ...prev,
                                [msg.id]: !prev[msg.id],
                              }))
                            }
                            className="w-full flex items-center justify-between px-4 py-2.5 text-xs text-[#52504C] hover:bg-[#EFECE3] transition-colors"
                          >
                            <div className="flex items-center gap-2">
                              <Brain className="w-3.5 h-3.5 text-[#CC785C]" />
                              <span className="font-medium font-sans">
                                {!msg.isThinkingDone ? (
                                  <span className="thinking-shimmer flex items-center gap-1.5 font-semibold">
                                    <span>Thinking through document...</span>
                                  </span>
                                ) : (
                                  `Thought for ${msg.thinkingDuration || 1.4} seconds`
                                )}
                              </span>
                            </div>
                            <ChevronDown
                              className={`w-3.5 h-3.5 text-[#82807A] transition-transform duration-200 ${
                                openThinkingIds[msg.id] ? "rotate-180" : ""
                              }`}
                            />
                          </button>

                          {openThinkingIds[msg.id] && (
                            <div className="px-4 py-3.5 border-t border-[#E5E3DC] bg-[#FAF9F5]">
                              <div className="border-l-2 border-[#CC785C] pl-3.5 text-[#5C5A55] text-xs sm:text-[13px] leading-relaxed whitespace-pre-wrap font-sans italic">
                                {msg.thinkingContent || "Synthesizing document context..."}
                                {!msg.isThinkingDone && <span className="animate-cursor ml-1" />}
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Main Message Content */}
                      <div className="text-sm sm:text-base text-[#1F1E1D] leading-relaxed font-sans">
                        {msg.content ? (
                          <div>
                            <MarkdownRenderer content={msg.content} />
                            {isStreaming && msg.isThinkingDone && msg.id === messages[messages.length - 1].id && (
                              <span className="animate-cursor inline-block" />
                            )}
                          </div>
                        ) : msg.isThinkingDone ? (
                          <span className="thinking-shimmer text-xs sm:text-sm">
                            Generating grounded response...
                          </span>
                        ) : null}
                      </div>

                      {/* Verified Sources & Evaluation Drawer */}
                      {msg.citations && msg.citations.length > 0 && (
                        <div className="pt-2">
                          <button
                            onClick={() =>
                              setOpenSourceIds((prev) => ({
                                ...prev,
                                [msg.id]: !prev[msg.id],
                              }))
                            }
                            className="flex items-center gap-1.5 text-xs text-[#82807A] hover:text-[#1F1E1D] font-medium transition-colors"
                          >
                            <BookOpen className="w-3.5 h-3.5" />
                            <span>
                              {msg.citations.length} Verified{" "}
                              {msg.citations.length === 1 ? "Source" : "Sources"} & Telemetry
                            </span>
                            <ChevronDown
                              className={`w-3 h-3 transition-transform ${
                                openSourceIds[msg.id] ? "rotate-180" : ""
                              }`}
                            />
                          </button>

                          {openSourceIds[msg.id] && (
                            <div className="mt-2.5 p-3.5 rounded-xl border border-[#E5E3DC] bg-white space-y-3 shadow-sm animate-fade-in-up">
                              {msg.evaluation && (
                                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pb-3 border-b border-[#ECEAE4] text-xs">
                                  <div>
                                    <span className="text-[#82807A] block">Retrieval</span>
                                    <span className="font-semibold text-[#1F1E1D]">
                                      {msg.evaluation.retrieval_ms} ms
                                    </span>
                                  </div>
                                  <div>
                                    <span className="text-[#82807A] block">Generation</span>
                                    <span className="font-semibold text-[#1F1E1D]">
                                      {msg.evaluation.generation_ms} ms
                                    </span>
                                  </div>
                                  <div>
                                    <span className="text-[#82807A] block">Relevance</span>
                                    <span className="font-semibold text-[#1F1E1D]">
                                      {Math.round(msg.evaluation.relevance * 100)}%
                                    </span>
                                  </div>
                                  <div>
                                    <span className="text-[#82807A] block">Groundedness</span>
                                    <span className="font-semibold text-[#15803D]">
                                      {Math.round(msg.evaluation.groundedness * 100)}% Verified
                                    </span>
                                  </div>
                                </div>
                              )}

                              <div className="space-y-2">
                                {msg.citations.map((cite, cIdx) => (
                                  <div
                                    key={cIdx}
                                    className="p-2.5 rounded-lg bg-[#FAF9F5] border border-[#ECEAE4] text-xs text-[#383633]"
                                  >
                                    <div className="flex items-center justify-between font-medium text-[#1F1E1D] mb-1">
                                      <span>
                                        [{cIdx + 1}] {cite.source} (Page {cite.page})
                                      </span>
                                      <span className="text-[10px] text-[#82807A] uppercase tracking-wider">
                                        {cite.retrieval_method.replace("_", " ")}
                                      </span>
                                    </div>
                                    <p className="text-[#52504C] italic line-clamp-2">
                                      "{cite.text_snippet}"
                                    </p>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}

        <div className="sticky bottom-4 z-20 w-full pt-4">
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`rounded-2xl border transition-all p-3 ${
              isDragging
                ? "border-[#CC785C] ring-2 ring-[#CC785C]/40 bg-[#FDFBF7]"
                : "bg-white border-[#D5D3CC] shadow-md focus-within:border-[#CC785C] focus-within:ring-1 focus-within:ring-[#CC785C]/30"
            }`}
          >
            {/* Attached Documents Pills with Individual X Delete Button */}
            {attachedFiles.length > 0 && (
              <div className="flex flex-wrap gap-1.5 mb-2 pb-2 border-b border-[#F0EEE8]">
                {attachedFiles.map((doc) => (
                  <div
                    key={doc.filename}
                    className="group flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[#F5F3EC] border border-[#E5E3DC] text-xs text-[#383633] transition-all hover:border-[#CC785C]/60"
                  >
                    <FileText className="w-3.5 h-3.5 text-[#CC785C]" />
                    <span className="font-medium max-w-[170px] truncate" title={doc.filename}>
                      {doc.filename}
                    </span>
                    <button
                      onClick={() => handleRemoveDoc(doc.filename)}
                      className="text-[#9E9B93] hover:text-[#C55030] hover:bg-[#FEE2E2] p-0.5 rounded transition-colors ml-1"
                      title="Remove this document"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Input Textarea */}
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={handleTextareaChange}
              onKeyDown={handleKeyDown}
              placeholder={
                attachedFiles.length === 0
                  ? "Attach documents to get started, or ask a question..."
                  : "Reply to DocAI or ask anything..."
              }
              className="w-full bg-transparent resize-none border-none outline-none text-[#1F1E1D] placeholder-[#9E9B93] text-sm sm:text-base leading-relaxed py-1 px-1 min-h-[44px] max-h-[200px]"
            />

            {/* Bottom Actions Row: Attach Button + Model Picker + Send Button */}
            <div className="flex items-center justify-between pt-2 border-t border-[#F5F3EC] mt-1">
              {/* Left Actions: Attach File + Model Selector */}
              <div className="flex items-center gap-2">
                {/* File Upload Trigger */}
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileUpload}
                  multiple
                  accept=".pdf,.docx,.doc,.xlsx,.xls,.csv,.txt,.md,.markdown,.json,.py,.js,.ts,.html,.css,.xml"
                  className="hidden"
                />
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isUploading}
                  className="flex items-center gap-1.5 text-xs font-medium text-[#52504C] hover:text-[#1F1E1D] px-2.5 py-1.5 rounded-lg hover:bg-[#F3F1EC] transition-colors border border-transparent hover:border-[#E5E3DC]"
                  title="Attach PDF, Word (.docx), Excel (.xlsx), CSV, Markdown, TXT, or Code files"
                >
                  <Paperclip className="w-4 h-4 text-[#82807A]" />
                  <span>{isUploading ? "Uploading..." : "Attach"}</span>
                </button>

                {/* Model Selector Dropdown */}
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setIsModelMenuOpen(!isModelMenuOpen)}
                    className="flex items-center gap-1.5 text-xs font-medium text-[#52504C] hover:text-[#1F1E1D] px-2.5 py-1.5 rounded-lg hover:bg-[#F3F1EC] transition-colors border border-transparent hover:border-[#E5E3DC]"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-[#CC785C]"></span>
                    <span>
                      {AVAILABLE_MODELS.find((m) => m.id === selectedModel)?.name || selectedModel}
                    </span>
                    <ChevronDown className="w-3.5 h-3.5 text-[#82807A]" />
                  </button>

                  {/* Dropdown Menu */}
                  {isModelMenuOpen && (
                    <div className="absolute bottom-full left-0 mb-2 w-64 rounded-xl bg-white border border-[#E5E3DC] shadow-xl p-1.5 z-50 text-xs animate-fade-in-up">
                      {AVAILABLE_MODELS.map((model) => (
                        <button
                          key={model.id}
                          onClick={() => {
                            setSelectedModel(model.id);
                            localStorage.setItem("docai_selected_model", model.id);
                            setIsModelMenuOpen(false);
                          }}
                          className={`w-full text-left px-3 py-2 rounded-lg transition-colors flex items-center justify-between ${
                            selectedModel === model.id
                              ? "bg-[#F7F5EE] text-[#CC785C] font-semibold"
                              : "text-[#383633] hover:bg-[#F5F3EC]"
                          }`}
                        >
                          <div>
                            <div>{model.name}</div>
                            <div className="text-[10px] text-[#82807A] font-normal">
                              {model.desc}
                            </div>
                          </div>
                          {selectedModel === model.id && (
                            <CheckCircle2 className="w-4 h-4 text-[#CC785C]" />
                          )}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Right Action: Send Arrow Button */}
              <button
                type="button"
                onClick={() => handleSubmit()}
                disabled={!input.trim() || isStreaming}
                className={`w-8 h-8 rounded-full flex items-center justify-center transition-all ${
                  input.trim() && !isStreaming
                    ? "bg-[#CC785C] text-white hover:bg-[#B8664B] shadow-sm cursor-pointer"
                    : "bg-[#EAE8E1] text-[#9E9B93] cursor-not-allowed"
                }`}
              >
                <ArrowUp className="w-4 h-4" />
              </button>
            </div>
          </div>

          <p className="text-[11px] text-center text-[#9E9B93] mt-2 font-sans">
            DocAI is grounded on your documents. Verify critical findings using inline citations.
          </p>
        </div>
      </main>

      {/* API Key Modal Dialog */}
      {isKeyModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4 animate-fade-in-up">
          <div className="w-full max-w-md bg-white rounded-2xl border border-[#E5E3DC] shadow-2xl p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded-lg bg-[#CC785C] text-white flex items-center justify-center text-xs font-serif font-bold">
                  ✦
                </div>
                <h3 className="font-serif font-semibold text-lg text-[#1F1E1D]">
                  Configure Gemini API Key
                </h3>
              </div>
              <button
                onClick={() => setIsKeyModalOpen(false)}
                className="text-[#82807A] hover:text-[#1F1E1D] p-1 rounded-md"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <p className="text-xs text-[#6B6963] leading-relaxed">
              Enter your Google Gemini API key to enable live factual completions. Your key is stored locally in your browser and sent securely only to your local backend.
            </p>

            <div className="space-y-1.5">
              <input
                type="password"
                value={geminiKeyInput}
                onChange={(e) => setGeminiKeyInput(e.target.value)}
                placeholder="AIzaSy..."
                className="w-full px-3.5 py-2.5 rounded-xl border border-[#D5D3CC] focus:border-[#CC785C] focus:ring-1 focus:ring-[#CC785C] outline-none text-xs font-mono"
              />
              <div className="flex justify-between items-center pt-1 text-[11px]">
                <a
                  href="https://aistudio.google.com/app/apikey"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[#CC785C] hover:underline flex items-center gap-1"
                >
                  <span>Get free key at Google AI Studio</span>
                  <ExternalLink className="w-3 h-3" />
                </a>
                {geminiApiKey && (
                  <button
                    onClick={() => {
                      setGeminiApiKey("");
                      setGeminiKeyInput("");
                      localStorage.removeItem("docai_gemini_api_key");
                    }}
                    className="text-[#C55030] hover:underline"
                  >
                    Remove key
                  </button>
                )}
              </div>
            </div>

            <div className="flex items-center justify-end gap-2 pt-3 border-t border-[#F0EEE8]">
              <button
                onClick={() => setIsKeyModalOpen(false)}
                className="px-3.5 py-2 rounded-xl text-xs font-medium text-[#52504C] hover:bg-[#F3F1EC] transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={async () => {
                  const trimmed = geminiKeyInput.trim();
                  setGeminiApiKey(trimmed);
                  if (trimmed) {
                    localStorage.setItem("docai_gemini_api_key", trimmed);
                  } else {
                    localStorage.removeItem("docai_gemini_api_key");
                  }

                  try {
                    await fetch(getApiUrl("/api/keys"), {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({
                        gemini_api_key: trimmed || null,
                      }),
                    });
                  } catch (e) {
                    // Ignore background sync errors
                  }

                  setIsKeyModalOpen(false);
                }}
                className="px-4 py-2 rounded-xl text-xs font-medium bg-[#CC785C] text-white hover:bg-[#B8664B] transition-colors shadow-sm"
              >
                Save & Connect
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
