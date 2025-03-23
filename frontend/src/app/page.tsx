// app/page.tsx
"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Bot, User, Send, Loader2, Plus, Menu, X, Book, Sun, Moon, MessageSquare } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { Textarea } from "@/components/ui/textarea"
import ReactMarkdown from 'react-markdown'
import { useTheme } from "next-themes"

type Message = {
  id: string
  role: "user" | "assistant"
  content: string
  sources?: string[]
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api"

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const { theme, setTheme } = useTheme()
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "60px"
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }, [input])

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement | HTMLInputElement>) => {
    setInput(e.target.value)
  }

  const getChatHistory = () => {
    return messages.map(msg => ({
      role: msg.role,
      content: msg.content
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!input.trim()) return

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: input.trim(),
          chat_history: getChatHistory()
        }),
      })

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
      }

      const data = await response.json()

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.response,
        sources: data.sources || []
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      console.error("Error:", error)
      
      // Add error message to chat
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Sorry, I encountered an error while processing your request. Please try again later."
      }
      
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const startNewChat = () => {
    setMessages([])
  }

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark")
  }

  return (
    <div className="flex h-[100dvh] bg-background text-foreground">
      {/* Mobile sidebar toggle */}
      <Button
        variant="ghost"
        size="icon"
        className="absolute top-4 left-4 md:hidden z-50"
        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
      >
        {isSidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
      </Button>

      {/* Sidebar */}
      <div
        className={cn(
          "w-64 border-r border-border flex-shrink-0 flex flex-col",
          "fixed md:static inset-y-0 left-0 z-40 bg-background",
          "transform transition-transform duration-300 ease-in-out",
          isSidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0",
        )}
      >
        <div className="p-4 border-b border-border flex items-center justify-between">
          <Button
            variant="outline"
            className="w-full justify-start gap-2 text-foreground"
            onClick={startNewChat}
          >
            <Plus size={16} />
            New chat
          </Button>
        </div>
        <div className="flex-1 overflow-auto p-2">
          <div className="space-y-1 mt-2">
            {/* Chat history items will be populated dynamically */}
          </div>
        </div>
        <div className="p-4 border-t border-border space-y-4">
          <Button 
            variant="ghost" 
            size="sm" 
            className="w-full justify-start gap-2 text-muted-foreground"
            onClick={toggleTheme}
          >
            {theme === "dark" ? (
              <>
                <Sun size={16} />
                <span>Light mode</span>
              </>
            ) : (
              <>
                <Moon size={16} />
                <span>Dark mode</span>
              </>
            )}
          </Button>
          <div className="text-xs text-muted-foreground flex items-center gap-2">
            <Book size={12} />
            <span>Books Knowledge Assistant</span>
          </div>
        </div>
      </div>

      {/* Chat overlay to close sidebar on mobile */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/20 z-30 md:hidden" 
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Main chat area */}
      <div className="flex-1 flex flex-col max-w-4xl mx-auto w-full">
        {/* Chat messages */}
        <div className="flex-1 overflow-auto py-4 px-2 md:px-4">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center p-4 md:p-8">
              <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-6">
                <Book className="h-8 w-8 text-primary" />
              </div>
              <h1 className="text-2xl font-semibold mb-3">Personal Knowledge Assistant</h1>
              <div className="max-w-md text-center text-muted-foreground">
                <p>Ask me anything from your knowledge base.</p>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={cn(
                    "flex items-start gap-4 px-4 py-6 rounded-lg message-animate-in",
                    message.role === "assistant" && "bg-card",
                  )}
                >
                  <div className="flex-shrink-0 mt-0.5">
                    {message.role === "user" ? (
                      <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center">
                        <User className="h-5 w-5" />
                      </div>
                    ) : (
                      <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                        <Bot className="h-5 w-5 text-primary-foreground" />
                      </div>
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <ReactMarkdown
                      components={{
                        div: ({ node, ...props }) => (
                          <div className="prose prose-sm dark:prose-invert max-w-none" {...props} />
                        )
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                    
                    {message.sources && message.sources.length > 0 && (
                      <div className="mt-4 pt-3 border-t border-border">
                        <p className="text-xs font-medium text-muted-foreground mb-1">Sources:</p>
                        <ul className="text-xs text-muted-foreground space-y-1 pl-5 list-disc">
                          {message.sources.map((source, index) => (
                            <li key={index}>{source}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex items-start gap-4 px-4 py-6 bg-card rounded-lg message-animate-in">
                  <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                    <Bot className="h-5 w-5 text-primary-foreground" />
                  </div>
                  <div className="flex items-center gap-3">
                    <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">Generating response...</span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input form */}
        <div className="border-t border-border p-3 md:p-4 bg-background">
          <form onSubmit={handleSubmit} className="flex gap-2 items-end max-w-4xl mx-auto">
            <div className="relative flex-1">
              <Textarea
                ref={textareaRef}
                value={input}
                onChange={handleInputChange}
                placeholder="Ask about your books..."
                className="resize-none pr-10 py-3 min-h-[60px] max-h-[200px] rounded-xl border border-input focus:border-primary focus:ring-1 focus:ring-primary focus:ring-offset-0"
                rows={1}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault()
                    handleSubmit(e as any)
                  }
                }}
              />
              <Button
                type="submit"
                size="icon"
                className="absolute right-2 bottom-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90"
                disabled={isLoading || !input.trim()}
              >
                <Send className="h-4 w-4" />
                <span className="sr-only">Send</span>
              </Button>
            </div>
          </form>
          <p className="text-xs text-center mt-2 text-muted-foreground">
            The assistant may produce inaccurate information. Verify important information.
          </p>
        </div>
      </div>
    </div>
  )
}