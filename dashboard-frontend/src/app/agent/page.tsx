"use client";

import { useRef, useState, useEffect } from "react";
import { Send, Trash2, Sparkles, Bot, Brain, XCircle, Cpu, Zap } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/page-header";

/* ── LLM Model Info (read from env via API or hardcoded fallback) ── */
const MODEL_INFO: Record<string, { name: string; provider: string; strengths: string; speed: string; context: string }> = {
  "google/gemini-2.0-flash-001": { name: "Gemini 2.0 Flash", provider: "Google (via OpenRouter)", strengths: "Rápido, econômico, ótimo para Q&A estruturado", speed: "~0.3s/req", context: "1M tokens" },
  "google/gemini-2.5-pro-preview": { name: "Gemini 2.5 Pro", provider: "Google (via OpenRouter)", strengths: "Raciocínio avançado, análise profunda", speed: "~1.5s/req", context: "1M tokens" },
  "openai/gpt-4o": { name: "GPT-4o", provider: "OpenAI (via OpenRouter)", strengths: "Multimodal, raciocínio geral forte", speed: "~0.8s/req", context: "128K tokens" },
  "openai/gpt-4o-mini": { name: "GPT-4o Mini", provider: "OpenAI (via OpenRouter)", strengths: "Equilíbrio custo/desempenho", speed: "~0.3s/req", context: "128K tokens" },
  "anthropic/claude-sonnet-4": { name: "Claude Sonnet 4", provider: "Anthropic (via OpenRouter)", strengths: "Excelente em análise e textos longos", speed: "~1.0s/req", context: "200K tokens" },
  "meta-llama/llama-4-maverick": { name: "Llama 4 Maverick", provider: "Meta (via OpenRouter)", strengths: "Open-source, inferência rápida", speed: "~0.2s/req", context: "128K tokens" },
};

interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  sources?: string[];
  reasoningCount?: number;
  elapsedTime?: number;
}

const EXAMPLE_QUERIES = [
  "Qual é a tendência atual das ações da NVIDIA?",
  "Analise o desempenho financeiro da NVIDIA neste trimestre",
  "Quais são os principais riscos para as ações da NVIDIA?",
  "Compare NVIDIA e AMD no mercado de GPUs",
];

export default function AgentPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [modelId, setModelId] = useState<string>("google/gemini-2.0-flash-001");
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Fetch actual model from config
  useEffect(() => {
    api.health.check().then((res) => {
      const cfg = res as Record<string, unknown>;
      if (cfg.llm_model) setModelId(String(cfg.llm_model));
    }).catch(() => {});
  }, []);

  const modelInfo = MODEL_INFO[modelId] ?? { name: modelId.split("/").pop() ?? modelId, provider: "OpenRouter", strengths: "LLM-powered agent", speed: "—", context: "—" };

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
        content: `Error: ${err instanceof Error ? err.message : "Falha ao obter resposta"}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const clearChat = () => setMessages([]);

  return (
    <div className="mx-auto max-w-7xl flex h-[calc(100vh-4rem)] flex-col">
      {/* Header */}
      <div className="pb-2">
        <PageHeader
          label="RAG · Conversacional"
          title="Agente de"
          gradient="IA"
          subtitle="Agente ReAct com RAG para análise financeira e perguntas e respostas."
          icon={Bot}
        />
        {messages.length > 0 && (
          <div className="mt-3 flex justify-center">
            <button
              onClick={clearChat}
              className="flex items-center gap-2 rounded-lg border border-surface-border bg-surface-hover px-3 py-2 text-xs text-white/50 hover:text-white"
            >
              <Trash2 className="h-3.5 w-3.5" /> Limpar chat
            </button>
          </div>
        )}
      </div>

      {/* Active Model Card */}
      <div className="mb-4 rounded-xl border border-nvidia/20 bg-nvidia/5 p-3">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <Cpu className="h-4 w-4 text-nvidia" />
            <span className="text-xs font-semibold text-white/60">Modelo Ativo</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="rounded-full bg-nvidia/20 px-2.5 py-0.5 text-xs font-bold text-nvidia">{modelInfo.name}</span>
            <span className="text-[11px] text-white/30">{modelInfo.provider}</span>
          </div>
          <div className="hidden items-center gap-3 text-[11px] text-white/40 sm:flex">
            <span className="flex items-center gap-1"><Zap className="h-3 w-3 text-amber-400" />{modelInfo.speed}</span>
            <span>Contexto: {modelInfo.context}</span>
            <span className="text-white/25">|</span>
            <span>{modelInfo.strengths}</span>
          </div>
          <span className="ml-auto font-mono text-[10px] text-white/20">{modelId}</span>
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto rounded-xl border border-surface-border bg-surface-card p-4">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center">
            <Sparkles className="mb-4 h-12 w-12 text-nvidia/30" />
            <h3 className="text-lg font-semibold text-white/30">
              Pergunte-me qualquer coisa sobre NVIDIA
            </h3>
            <p className="mt-2 text-sm text-white/20">
              Análise financeira, predições de preço, insights de mercado
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
                    <span className="flex items-center gap-1">{msg.role === "user" ? "Você" : <><Bot className="inline h-3 w-3" /> Agente</>}</span>
                    <span>
                      {msg.timestamp.toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </span>
                  </div>
                  {msg.role === "assistant" ? (
                    <div className="mt-1 text-sm prose prose-invert prose-sm max-w-none prose-p:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0.5 prose-headings:text-white prose-strong:text-nvidia prose-a:text-nvidia">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                  ) : (
                    <div className="mt-1 whitespace-pre-wrap text-sm">
                      {msg.content}
                    </div>
                  )}

                  {/* Reasoning info */}
                  {msg.reasoningCount !== undefined && msg.reasoningCount > 0 && (
                    <div className="mt-2 flex items-center gap-3 text-[10px] text-white/30">
                      <span className="flex items-center gap-1"><Brain className="inline h-3 w-3" /> {msg.reasoningCount} etapas de raciocínio</span>
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
                    <span>Processando...</span>
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
          placeholder="Pergunte sobre ações da NVIDIA, financeiro ou análise de mercado..."
          disabled={loading}
          className="flex-1 rounded-xl border border-surface-border bg-surface-card px-4 py-3 text-sm text-white placeholder-white/30 outline-none focus:border-nvidia/50 disabled:opacity-50"
        />
        <button
          onClick={() => sendMessage()}
          disabled={loading || !input.trim()}
          className="flex items-center gap-2 rounded-xl bg-nvidia px-5 py-3 text-sm font-semibold text-black transition-all hover:bg-nvidia-dark disabled:opacity-50"
        >
          <Send className="h-4 w-4" />
          Enviar
        </button>
      </div>
    </div>
  );
}
