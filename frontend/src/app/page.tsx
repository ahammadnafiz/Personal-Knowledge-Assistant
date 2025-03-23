// app/page.tsx
"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Bot, User, Send, Loader2, Plus, Menu, X, Book } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { Textarea } from "@/components/ui/textarea"
import ReactMarkdown from 'react-markdown'

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
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

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

  return (
    <div className="flex h-[100dvh] bg-white dark:bg-black text-black dark:text-white">
      {/* Mobile sidebar toggle */}
      <Button
        variant="ghost"
        size="icon"
        className="absolute top-3 left-3 md:hidden z-50"
        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
      >
        {isSidebarOpen ? <X /> : <Menu />}
      </Button>

      {/* Sidebar */}
      <div
        className={cn(
          "w-64 border-r border-gray-200 dark:border-gray-800 flex-shrink-0 flex flex-col",
          "fixed md:static inset-y-0 left-0 z-40 bg-white dark:bg-black",
          "transform transition-transform duration-300 ease-in-out",
          isSidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0",
        )}
      >
        <div className="p-4 border-b border-gray-200 dark:border-gray-800">
          <Button
            variant="outline"
            className="w-full justify-start gap-2 text-gray-700 dark:text-gray-300"
            onClick={startNewChat}
          >
            <Plus size={16} />
            New chat
          </Button>
        </div>
        <div className="flex-1 overflow-auto p-2">{/* Chat history would go here */}</div>
        <div className="p-4 border-t border-gray-200 dark:border-gray-800">
          <div className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-2">
            <Book size={12} />
            <span>Books Knowledge Assistant</span>
          </div>
        </div>
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col max-w-3xl mx-auto w-full">
        {/* Chat messages */}
        <div className="flex-1 overflow-auto p-4">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center">
              <h1 className="text-2xl font-semibold mb-6">Book Knowledge Assistant</h1>
              <div className="max-w-md text-center text-gray-500 dark:text-gray-400">
                <p>Ask me anything about the books in your knowledge base.</p>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={cn(
                    "flex items-start gap-4 px-4 py-6 rounded-lg",
                    message.role === "assistant" && "bg-gray-50 dark:bg-gray-900",
                  )}
                >
                  <div className="flex-shrink-0 mt-0.5">
                    {message.role === "user" ? (
                      <div className="w-8 h-8 rounded-full bg-gray-300 dark:bg-gray-700 flex items-center justify-center">
                        <User className="h-5 w-5" />
                      </div>
                    ) : (
                      <div className="w-8 h-8 rounded-full bg-black dark:bg-white flex items-center justify-center">
                        <Bot className="h-5 w-5 text-white dark:text-black" />
                      </div>
                    )}
                  </div>
                  <div className="flex-1">
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
                      <div className="mt-4 pt-3 border-t border-gray-200 dark:border-gray-700">
                        <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Sources:</p>
                        <ul className="text-xs text-gray-500 dark:text-gray-400 space-y-1 pl-5 list-disc">
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
                <div className="flex items-start gap-4 px-4 py-6 bg-gray-50 dark:bg-gray-900 rounded-lg">
                  <div className="w-8 h-8 rounded-full bg-black dark:bg-white flex items-center justify-center">
                    <Bot className="h-5 w-5 text-white dark:text-black" />
                  </div>
                  <Loader2 className="h-5 w-5 animate-spin" />
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input form */}
        <div className="border-t border-gray-200 dark:border-gray-800 p-4">
          <form onSubmit={handleSubmit} className="flex gap-2 items-end max-w-3xl mx-auto">
            <div className="relative flex-1">
              <Textarea
                value={input}
                onChange={handleInputChange}
                placeholder="Ask about your books..."
                className="resize-none pr-10 py-3 min-h-[60px] max-h-[200px] rounded-xl border-gray-300 dark:border-gray-700 focus:border-black dark:focus:border-white focus:ring-0"
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
                className="absolute right-2 bottom-2 rounded-lg bg-black dark:bg-white text-white dark:text-black hover:bg-gray-800 dark:hover:bg-gray-200"
                disabled={isLoading || !input.trim()}
              >
                <Send className="h-4 w-4" />
                <span className="sr-only">Send</span>
              </Button>
            </div>
          </form>
          <p className="text-xs text-center mt-2 text-gray-500">
            The assistant may produce inaccurate information. Verify important information.
          </p>
        </div>
      </div>
    </div>
  )
}