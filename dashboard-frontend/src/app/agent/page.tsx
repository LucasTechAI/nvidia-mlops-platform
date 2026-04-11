"use client";

import { useRef, useState, useEffect } from "react";
import { Send, Trash2, Sparkles } from "lucide-react";
import { api } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  sources?: string[];
  reasoningCount?: number;
  elapsedTime?: number;
}

const EXAMPLE_QUERIES = [
  "What is the current NVIDIA stock trend?",
  "Analyze NVIDIA's financial performance this quarter",
  "What are the main risks for NVIDIA stock?",
  "Compare NVIDIA with AMD in the GPU market",
];

export default function AgentPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async (query?: string) => {
    const text = query || input.trim();
    if (!text || loading) return;

    const userMsg: Message = {
      role: "user",
      content: text,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await api.agent.query({
        query: text,
        use_guardrails: true,
        temperature: 0.1,
        max_iterations: 8,
      });

      const agentRes = res as {
        answer?: string;
        response?: string;
        tools_used?: string[];
        reasoning_steps?: number;
        elapsed_time?: number;
        model_used?: string;
      };

      const assistantMsg: Message = {
        role: "assistant",
        content: agentRes.answer || agentRes.response || JSON.stringify(res),
        timestamp: new Date(),
        sources: agentRes.tools_used,
        reasoningCount: agentRes.reasoning_steps ?? 0,
        elapsedTime: agentRes.elapsed_time,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const errorMsg: Message = {
        role: "assistant",
        content: `❌ Error: ${err instanceof Error ? err.message : "Failed to get response"}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => setMessages([]);

  return (
    <div className="flex h-[calc(100vh-4rem)] flex-col">
      {/* Header */}
      <div className="flex items-center justify-between pb-4">
        <div>
          <h2 className="text-2xl font-semibold">🤖 AI Agent</h2>
          <p className="mt-1 text-sm text-white/50">
            ReAct agent with RAG for financial analysis and Q&amp;A
          </p>
        </div>
        {messages.length > 0 && (
          <button
            onClick={clearChat}
            className="flex items-center gap-2 rounded-lg border border-surface-border bg-surface-hover px-3 py-2 text-xs text-white/50 hover:text-white"
          >
            <Trash2 className="h-3.5 w-3.5" /> Clear Chat
          </button>
        )}
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto rounded-xl border border-surface-border bg-surface-card p-4">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center">
            <Sparkles className="mb-4 h-12 w-12 text-nvidia/30" />
            <h3 className="text-lg font-semibold text-white/30">
              Ask me anything about NVIDIA
            </h3>
            <p className="mt-2 text-sm text-white/20">
              Financial analysis, stock predictions, market insights
            </p>
            <div className="mt-6 grid grid-cols-1 gap-2 sm:grid-cols-2">
              {EXAMPLE_QUERIES.map((q) => (
                <button
                  key={q}
                  onClick={() => sendMessage(q)}
                  className="rounded-lg border border-surface-border bg-surface-hover px-4 py-2.5 text-left text-sm text-white/50 transition-all hover:border-nvidia/30 hover:text-white"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[80%] rounded-xl px-4 py-3 ${
                    msg.role === "user"
                      ? "bg-nvidia/20 text-white"
                      : "bg-surface-hover text-white/90"
                  }`}
                >
                  <div className="flex items-center gap-2 text-xs text-white/30">
                    <span>{msg.role === "user" ? "You" : "🤖 Agent"}</span>
                    <span>
                      {msg.timestamp.toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </span>
                  </div>
                  <div className="mt-1 whitespace-pre-wrap text-sm">
                    {msg.content}
                  </div>

                  {/* Reasoning info */}
                  {msg.reasoningCount !== undefined && msg.reasoningCount > 0 && (
                    <div className="mt-2 flex items-center gap-3 text-[10px] text-white/30">
                      <span>🧠 {msg.reasoningCount} reasoning steps</span>
                      {msg.elapsedTime !== undefined && (
                        <span>⏱ {msg.elapsedTime.toFixed(2)}s</span>
                      )}
                    </div>
                  )}

                  {/* Tools used */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {msg.sources.map((src, j) => (
                        <span
                          key={j}
                          className="rounded bg-nvidia/10 px-2 py-0.5 text-[10px] text-nvidia"
                        >
                          {src}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="rounded-xl bg-surface-hover px-4 py-3">
                  <div className="flex items-center gap-2 text-sm text-white/50">
                    <div className="h-2 w-2 animate-pulse rounded-full bg-nvidia" />
                    <span>Thinking...</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="mt-4 flex gap-3">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage()}
          placeholder="Ask about NVIDIA stock, financials, or market analysis..."
          disabled={loading}
          className="flex-1 rounded-xl border border-surface-border bg-surface-card px-4 py-3 text-sm text-white placeholder-white/30 outline-none focus:border-nvidia/50 disabled:opacity-50"
        />
        <button
          onClick={() => sendMessage()}
          disabled={loading || !input.trim()}
          className="flex items-center gap-2 rounded-xl bg-nvidia px-5 py-3 text-sm font-semibold text-black transition-all hover:bg-nvidia-dark disabled:opacity-50"
        >
          <Send className="h-4 w-4" />
          Send
        </button>
      </div>
    </div>
  );
}
