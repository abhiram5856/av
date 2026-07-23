"use client";

import { useState } from "react";
import { Bot, Send, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { fetchFromAPI } from "@/lib/api-client";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

const initialMessages: Message[] = [
  {
    id: "1",
    role: "assistant",
    content: "Hello! I'm your AgriVision AI assistant. How can I help you with your farm today?",
  },
];

export default function AssistantPage() {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg: Message = { id: Date.now().toString(), role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const data = await fetchFromAPI('/api/chat', {
        method: 'POST',
        body: JSON.stringify({ query: input }),
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
        content: "I'm having trouble connecting to the Zenith AgriBot server. Please ensure the backend is running.",
      };
      setMessages((prev) => [...prev, botMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)] max-w-4xl mx-auto gap-4">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">AI Farm Assistant</h1>
        <p className="text-muted-foreground mt-2">Ask questions about crop health, weather impacts, and general farming advice.</p>
      </div>

      <Card className="flex flex-col flex-1 overflow-hidden shadow-sm">
        <CardHeader className="border-b bg-muted/30 pb-4">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Bot className="h-6 w-6 text-primary" />
            AgriVision Assistant
          </CardTitle>
        </CardHeader>
        
        <CardContent className="flex-1 overflow-y-auto p-4 space-y-6">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-4 max-w-[80%] ${
                message.role === "user" ? "ml-auto flex-row-reverse" : ""
              }`}
            >
              <div className={`shrink-0 rounded-full h-8 w-8 flex items-center justify-center mt-1 ${
                message.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted text-foreground border"
              }`}>
                {message.role === "user" ? <User className="h-5 w-5" /> : <Bot className="h-5 w-5" />}
              </div>
              <div className={`rounded-lg p-4 text-sm ${
                message.role === "user" 
                  ? "bg-primary text-primary-foreground" 
                  : "bg-muted/50 border"
              }`}>
                {message.content}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex gap-4 max-w-[80%]">
              <div className="shrink-0 rounded-full h-8 w-8 flex items-center justify-center mt-1 bg-muted text-foreground border">
                <Bot className="h-5 w-5" />
              </div>
              <div className="rounded-lg p-4 text-sm bg-muted/50 border flex items-center gap-2">
                <div className="h-2 w-2 bg-foreground/30 rounded-full animate-bounce" />
                <div className="h-2 w-2 bg-foreground/30 rounded-full animate-bounce [animation-delay:-0.15s]" />
                <div className="h-2 w-2 bg-foreground/30 rounded-full animate-bounce [animation-delay:-0.3s]" />
              </div>
            </div>
          )}
        </CardContent>

        <CardFooter className="border-t p-4 bg-background">
          <form onSubmit={handleSend} className="flex w-full items-center gap-2">
            <Input
              placeholder="Ask about your crops, weather, or farming advice..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isLoading}
              className="flex-1"
            />
            <Button type="submit" size="icon" disabled={!input.trim() || isLoading}>
              <Send className="h-4 w-4" />
              <span className="sr-only">Send message</span>
            </Button>
          </form>
        </CardFooter>
      </Card>
    </div>
  );
}
