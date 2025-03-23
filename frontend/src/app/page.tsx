"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Bot, User, Send, Loader2, Plus, Menu, X, Book, Sun, Moon, ChevronLeft, Upload, MessageSquare, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { Textarea } from "@/components/ui/textarea"
import ReactMarkdown from 'react-markdown'
import { useTheme } from "next-themes"
import UploadButton from "@/components/ui/UploadButton"

type Message = {
  id: string
  role: "user" | "assistant"
  content: string
  sources?: string[]
}

type ChatSession = {
  id: string
  title: string
  messages: Message[]
  createdAt: Date
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api"

export default function ChatInterface() {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false)
  const { theme, setTheme } = useTheme()
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Initialize with a new session on first load
  useEffect(() => {
    if (sessions.length === 0) {
      const newSessionId = Date.now().toString()
      const newSession: ChatSession = {
        id: newSessionId,
        title: "New Chat",
        messages: [],
        createdAt: new Date()
      }
      setSessions([newSession])
      setActiveSessionId(newSessionId)
    }
  }, [sessions])

  // Load the active session's messages
  useEffect(() => {
    if (activeSessionId) {
      const activeSession = sessions.find(session => session.id === activeSessionId)
      if (activeSession) {
        setMessages(activeSession.messages)
      }
    }
  }, [activeSessionId, sessions])

  // Generate a title for the chat after the first exchange
  const generateChatTitle = async (userQuery: string, aiResponse: string) => {
    try {
      // For simplicity, we'll just use the first 30 chars of the user's first message
      return userQuery.length > 30 
        ? userQuery.substring(0, 30) + "..." 
        : userQuery
        
      // In a real implementation, you might call an API to generate a title
      // const response = await fetch(`${API_URL}/generate-title`, {
      //   method: "POST",
      //   headers: { "Content-Type": "application/json" },
      //   body: JSON.stringify({ userQuery, aiResponse }),
      // })
      // const data = await response.json()
      // return data.title
    } catch (error) {
      console.error("Error generating title:", error)
      return "New Chat"
    }
  }

  // Update session title after first exchange
  const updateSessionTitle = async (sessionId: string, userMessage: Message, aiMessage: Message) => {
    const session = sessions.find(s => s.id === sessionId)
    if (session && session.title === "New Chat" && session.messages.length === 0) {
      const title = await generateChatTitle(userMessage.content, aiMessage.content)
      setSessions(prev => prev.map(s => 
        s.id === sessionId ? { ...s, title } : s
      ))
    }
  }

  // Save messages to the current session
  const saveMessagesToSession = (sessionId: string, updatedMessages: Message[]) => {
    setSessions(prev => prev.map(session => 
      session.id === sessionId 
        ? { ...session, messages: updatedMessages } 
        : session
    ))
  }

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

  // Add global styles for custom scrollbar on component mount
  useEffect(() => {
    const style = document.createElement('style');
    style.innerHTML = `
      /* Custom scrollbar styles */
      .chat-scrollbar::-webkit-scrollbar {
        width: 7px;
      }
      
      .chat-scrollbar::-webkit-scrollbar-track {
        background: transparent;
      }
      
      .chat-scrollbar::-webkit-scrollbar-thumb {
        background-color: rgba(155, 155, 155, 0.5);
        border-radius: 10px;
      }
      
      .chat-scrollbar::-webkit-scrollbar-thumb:hover {
        background-color: rgba(155, 155, 155, 0.7);
      }
      
      /* For Firefox */
      .chat-scrollbar {
        scrollbar-width: thin;
        scrollbar-color: rgba(155, 155, 155, 0.5) transparent;
      }
      
      /* Hide scrollbar on mobile but show on hover */
      @media (max-width: 768px) {
        .chat-scrollbar::-webkit-scrollbar {
          width: 0px;
        }
        
        .chat-scrollbar:hover::-webkit-scrollbar {
          width: 7px;
        }
      }
    `;
    document.head.appendChild(style);
    
    return () => {
      document.head.removeChild(style);
    };
  }, []);

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

    if (!input.trim() || !activeSessionId) return

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
    }

    const updatedMessages = [...messages, userMessage]
    setMessages(updatedMessages)
    saveMessagesToSession(activeSessionId, updatedMessages)
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

      const newUpdatedMessages = [...updatedMessages, assistantMessage]
      setMessages(newUpdatedMessages)
      saveMessagesToSession(activeSessionId, newUpdatedMessages)
      
      // Update session title if this is the first exchange
      updateSessionTitle(activeSessionId, userMessage, assistantMessage)
    } catch (error) {
      console.error("Error:", error)
      
      // Add error message to chat
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Sorry, I encountered an error while processing your request. Please try again later."
      }
      
      const newUpdatedMessages = [...updatedMessages, errorMessage]
      setMessages(newUpdatedMessages)
      saveMessagesToSession(activeSessionId, newUpdatedMessages)
    } finally {
      setIsLoading(false)
    }
  }

  const startNewChat = () => {
    const newSessionId = Date.now().toString()
    const newSession: ChatSession = {
      id: newSessionId,
      title: "New Chat",
      messages: [],
      createdAt: new Date()
    }
    setSessions(prev => [newSession, ...prev])
    setActiveSessionId(newSessionId)
    setMessages([])
  }

  const switchSession = (sessionId: string) => {
    setActiveSessionId(sessionId)
    setIsSidebarOpen(false) // Close sidebar on mobile after selecting a chat
  }

  const deleteSession = (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setSessions(prev => prev.filter(session => session.id !== sessionId))
    
    // If we deleted the active session, switch to the first available one
    if (sessionId === activeSessionId) {
      const remainingSessions = sessions.filter(session => session.id !== sessionId)
      if (remainingSessions.length > 0) {
        setActiveSessionId(remainingSessions[0].id)
      } else {
        // If no sessions left, create a new one
        startNewChat()
      }
    }
  }

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark")
  }

  const toggleSidebar = () => {
    setIsSidebarCollapsed(!isSidebarCollapsed)
  }

  const handleUploadSuccess = () => {
    // Optionally add a system message to indicate the knowledge base has been updated
    const systemMessage: Message = {
      id: Date.now().toString(),
      role: "assistant",
      content: "New documents have been added to the knowledge base and are being processed. You can now ask questions about the new content!"
    }
    
    if (activeSessionId) {
      const newMessages = [...messages, systemMessage]
      setMessages(newMessages)
      saveMessagesToSession(activeSessionId, newMessages)
    }
  }

  // Format date for display in chat history
  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', { 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    }).format(date)
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
          "border-r border-border flex-shrink-0 flex flex-col",
          "fixed md:static inset-y-0 left-0 z-40 bg-background",
          "transform transition-all duration-300 ease-in-out",
          isSidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0",
          isSidebarCollapsed ? "w-16" : "w-64",
        )}
      >
        {/* Sidebar header */}
        <div className={cn(
          "border-b border-border flex items-center", 
          isSidebarCollapsed ? "justify-center p-3" : "justify-between p-4"
        )}>
          {!isSidebarCollapsed ? (
            <Button
              variant="outline"
              className="w-full justify-start gap-2 text-foreground hover:bg-primary/10 hover:text-primary transition-colors"
              onClick={startNewChat}
            >
              <Plus size={16} />
              New chat
            </Button>
          ) : (
            <Button
              variant="outline"
              size="icon"
              className="text-foreground hover:bg-primary/10 hover:text-primary transition-colors"
              onClick={startNewChat}
            >
              <Plus size={16} />
            </Button>
          )}
        </div>
        
        {/* Sidebar content */}
        <div className="flex-1 overflow-auto p-2 chat-scrollbar">
          <div className="space-y-1 mt-2">
            {/* Chat history items */}
            {!isSidebarCollapsed && (
              <div className="mb-4">
                <h3 className="text-xs font-semibold text-muted-foreground mb-2 px-2">Chat History</h3>
                <div className="space-y-1">
                  {sessions.map(session => (
                    <div
                      key={session.id}
                      className={cn(
                        "group w-full text-left rounded-lg flex items-center gap-2 text-sm transition-colors",
                        activeSessionId === session.id
                          ? "bg-primary/10 text-primary"
                          : "hover:bg-muted text-foreground"
                      )}
                    >
                      <button
                        className="flex-1 flex items-start gap-2 p-3 truncate text-left"
                        onClick={() => switchSession(session.id)}
                      >
                        <MessageSquare size={16} className="mt-0.5 flex-shrink-0" />
                        <div className="flex-1 flex flex-col overflow-hidden">
                          <span className="truncate font-medium">{session.title}</span>
                          <span className="text-xs text-muted-foreground truncate">{formatDate(session.createdAt)}</span>
                        </div>
                      </button>
                      <button 
                        className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive mr-2 p-1 rounded-md hover:bg-destructive/10 transition-colors"
                        onClick={(e) => deleteSession(session.id, e)}
                        aria-label="Delete chat"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
        
        {/* Sidebar footer with upload button and theme toggle */}
        <div className={cn(
          "border-t border-border", 
          isSidebarCollapsed ? "p-2" : "p-4",
          "space-y-4"
        )}>
          {!isSidebarCollapsed ? (
            <>
              {/* Upload button in sidebar footer */}
              <div className="mb-4">
                <UploadButton onSuccess={handleUploadSuccess} />
              </div>
              <Button 
                variant="ghost" 
                size="sm" 
                className="w-full justify-start gap-2 text-muted-foreground hover:bg-muted transition-colors"
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
                <span>Personal Knowledge Assistant</span>
              </div>
            </>
          ) : (
            <>
              {/* Upload button in collapsed sidebar */}
              <Button 
                variant="ghost" 
                size="icon" 
                className="w-full flex justify-center text-muted-foreground hover:bg-muted transition-colors"
                onClick={() => setIsSidebarCollapsed(false)}
                title="Upload documents"
              >
                <Upload size={16} />
              </Button>
              <Button 
                variant="ghost" 
                size="icon" 
                className="w-full flex justify-center text-muted-foreground hover:bg-muted transition-colors"
                onClick={toggleTheme}
                title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
              >
                {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
              </Button>
              <div className="flex justify-center">
                <Book size={16} className="text-muted-foreground" />
              </div>
            </>
          )}
        </div>
        
        {/* Improved collapse toggle button */}
        <Button 
          variant="ghost" 
          size="icon" 
          className="absolute top-1/2 -right-3 transform -translate-y-1/2 h-6 w-6 rounded-full shadow-md hidden md:flex bg-background border border-border hover:bg-muted"
          onClick={toggleSidebar}
          title={isSidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <ChevronLeft 
            size={14} 
            className={cn(
              "transition-transform duration-300",
              isSidebarCollapsed ? "rotate-180" : ""
            )} 
          />
        </Button>
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
        <div className="flex-1 overflow-auto py-4 px-2 md:px-4 chat-scrollbar">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center p-4 md:p-8">
              <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-6">
                <Book className="h-8 w-8 text-primary" />
              </div>
              <h1 className="text-2xl font-semibold mb-3">Personal Knowledge Assistant</h1>
              <div className="max-w-md text-center text-muted-foreground">
                <p>Ask me anything from your knowledge base.</p>
                <p className="mt-2">
                  Use the <span className="font-medium">Upload</span> button in the sidebar to add new documents.
                </p>
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
                placeholder="Ask to Brain..."
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