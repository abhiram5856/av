"use client";

import { useState, useRef, useEffect } from "react";
import { Send, User, Search, Bot } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { fetchFromAPI } from "@/lib/api-client";
import { useTranslation } from "@/lib/i18n";
import { useAppStore } from "@/lib/store";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

export default function AssistantPage() {
  const { t } = useTranslation();
  
  const initialMessages: Message[] = [
    {
      id: "welcome-message",
      role: "assistant",
      content: "WELCOME", // We will translate this dynamically in the render loop
    },
  ];

  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const currentLanguage = useAppStore.getState().language;
      const data = await fetchFromAPI("/api/chat", {
        method: "POST",
        body: JSON.stringify({ query: input, language: currentLanguage }),
      });

      const botMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.response || "Sorry, I couldn't generate a response.",
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (error) {
      console.error("Chat error:", error);
      const botMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: t("assistant.error") || "I'm having trouble connecting to the Zenith AgriBot server.",
      };
      setMessages((prev) => [...prev, botMsg]);
    } finally {
      setIsLoading(false);
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-10rem)] lg:h-[calc(100vh-8rem)] max-w-3xl mx-auto w-full">
      {/* ─── Header ──────────────────────────────────────────────────────── */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="border-b pb-4 mb-6"
      >
        <h1 className="text-2xl font-semibold tracking-tight text-foreground">
          {t("assistant.title") || "AI Farm Assistant"}
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          {t("assistant.subtitle") || "Ask questions about crop health, weather, and general farming."}
        </p>
      </motion.div>

      {/* ─── Chat Container ──────────────────────────────────────────────── */}
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1 }}
        className="flex flex-col flex-1 overflow-hidden rounded-lg border bg-card shadow-sm"
      >
        
        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6 scroll-smooth">
          <AnimatePresence initial={false}>
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ type: "spring", stiffness: 400, damping: 25 }}
                className={`flex gap-4 max-w-[85%] ${
                  message.role === "user" ? "ml-auto flex-row-reverse" : ""
                }`}
              >
                <div
                  className={`shrink-0 h-8 w-8 rounded-full flex items-center justify-center mt-0.5 shadow-sm ${
                    message.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted border text-foreground"
                  }`}
                >
                  {message.role === "user" ? (
                    <User className="h-4 w-4" />
                  ) : (
                    <Bot className="h-4 w-4 text-primary" />
                  )}
                </div>
                <div
                  className={`px-4 py-3 text-sm leading-relaxed rounded-2xl shadow-sm ${
                    message.role === "user"
                      ? "bg-primary text-primary-foreground rounded-tr-sm"
                      : "bg-muted/50 border rounded-tl-sm prose prose-sm dark:prose-invert"
                  }`}
                >
                  {message.role === "assistant" ? (
                    message.id === "welcome-message" ? (
                      <ReactMarkdown>{t("assistant.welcome") || "Hello! I am your AI Farm Assistant. How can I help you today?"}</ReactMarkdown>
                    ) : (
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    )
                  ) : (
                    message.content
                  )}
                </div>
              </motion.div>
            ))}
            
            {/* Typing Indicator */}
            {isLoading && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="flex gap-4 max-w-[85%]"
              >
                <div className="shrink-0 h-8 w-8 rounded-full flex items-center justify-center mt-0.5 bg-muted border text-foreground shadow-sm">
                  <Bot className="h-4 w-4 text-primary" />
                </div>
                <div className="px-4 py-4 rounded-2xl rounded-tl-sm bg-muted/50 border flex items-center gap-1.5 shadow-sm">
                  <motion.div 
                    animate={{ y: [0, -5, 0] }} 
                    transition={{ repeat: Infinity, duration: 0.6, ease: "easeInOut" }} 
                    className="h-1.5 w-1.5 bg-foreground/40 rounded-full" 
                  />
                  <motion.div 
                    animate={{ y: [0, -5, 0] }} 
                    transition={{ repeat: Infinity, duration: 0.6, delay: 0.2, ease: "easeInOut" }} 
                    className="h-1.5 w-1.5 bg-foreground/40 rounded-full" 
                  />
                  <motion.div 
                    animate={{ y: [0, -5, 0] }} 
                    transition={{ repeat: Infinity, duration: 0.6, delay: 0.4, ease: "easeInOut" }} 
                    className="h-1.5 w-1.5 bg-foreground/40 rounded-full" 
                  />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Input Form */}
        <div className="border-t p-3 bg-card/80 backdrop-blur-sm">
          <form onSubmit={handleSend} className="flex items-center gap-2">
            <Input
              ref={inputRef}
              placeholder={t("assistant.placeholder") || "Ask about your crops, weather, or farming advice..."}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isLoading}
              className="flex-1 rounded-full px-4 shadow-sm focus-visible:ring-primary/50"
            />
            <Button
              type="submit"
              size="icon"
              disabled={!input.trim() || isLoading}
              className="rounded-full shrink-0 shadow-sm transition-transform active:scale-95"
            >
              <Send className="h-4 w-4" />
            </Button>
          </form>
        </div>
      </motion.div>
    </div>
  );
}
