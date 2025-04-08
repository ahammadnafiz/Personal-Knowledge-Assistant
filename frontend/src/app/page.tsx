"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import {
  Bot,
  User,
  Loader2,
  Menu,
  X,
  Book,
  Sun,
  Moon,
  ChevronLeft,
  Upload,
  MessageSquare,
  Trash2,
  Sparkles,
  Send,
  Copy,
  Check,
  FileText,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import ReactMarkdown from "react-markdown"
import remarkMath from "remark-math"
import rehypeKatex from "rehype-katex"
import rehypeHighlight from "rehype-highlight"
import "katex/dist/katex.min.css"
import "highlight.js/styles/github-dark.css"
import { useTheme } from "next-themes"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@/libs/utils"

type Message = {
  id: string
  role: "user" | "assistant"
  content: string
  sources?: string[]
  isTyping?: boolean
}

type ChatSession = {
  id: string
  title: string
  messages: Message[]
  createdAt: Date
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api"

export default function ModernChatInterface() {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false)
  const [isTypingAnimation, setIsTypingAnimation] = useState(false)
  const [currentTypingMessage, setCurrentTypingMessage] = useState<string>("")
  const [typingIndex, setTypingIndex] = useState(0)
  const { theme, setTheme } = useTheme()
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const chatContainerRef = useRef<HTMLDivElement>(null)
  const [isBrowser, setIsBrowser] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [copiedText, setCopiedText] = useState<string | null>(null)

  useEffect(() => {
    setIsBrowser(true)
    const checkMobile = () => {
      if (window.innerWidth < 768) {
        setIsSidebarOpen(false)
      } else {
        setIsSidebarOpen(true)
      }
    }
    checkMobile()
    window.addEventListener("resize", checkMobile)
    return () => window.removeEventListener("resize", checkMobile)
  }, [])

  // Initialize with a new session on first load
  useEffect(() => {
    if (sessions.length === 0) {
      const newSessionId = Date.now().toString()
      const newSession: ChatSession = {
        id: newSessionId,
        title: "New chat",
        messages: [],
        createdAt: new Date(),
      }
      setSessions([newSession])
      setActiveSessionId(newSessionId)
    }
  }, [sessions])

  // Load the active session's messages
  useEffect(() => {
    if (activeSessionId) {
      const activeSession = sessions.find((session) => session.id === activeSessionId)
      if (activeSession) {
        setMessages(activeSession.messages)
      }
    }
  }, [activeSessionId, sessions])

  // Add custom styling
  useEffect(() => {
    const style = document.createElement("style")
    style.innerHTML = `
      @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
      }
      
      .typing-cursor {
        display: inline-block;
        width: 2px;
        height: 1em;
        background-color: currentColor;
        margin-left: 2px;
        vertical-align: text-bottom;
        animation: blink 0.8s infinite;
      }
      
      .hljs {
        background: transparent !important;
        padding: 0 !important;
      }
      
      /* Enhanced KaTeX styling */
      .katex-display {
        overflow-x: auto;
        overflow-y: hidden;
        padding: 0.5rem 0;
        margin: 0 !important;
      }
  
      .katex {
        font-size: 1.1em;
        text-rendering: auto;
      }
  
      /* Math display container */
      .math-display {
        width: 100%;
        overflow-x: auto;
        margin: 1rem 0;
        padding: 0.5rem 0;
        background-color: rgba(0, 0, 0, 0.02);
        border-radius: 0.5rem;
      }
  
      .dark .math-display {
        background-color: rgba(255, 255, 255, 0.02);
      }
  
      /* Inline math styling */
      .math-inline .katex {
        font-size: 1.05em;
        display: inline-block;
      }
  
      .math-inline {
        background-color: rgba(0, 0, 0, 0.02);
        border-radius: 0.25rem;
        padding: 0 0.25rem;
      }
  
      .dark .math-inline {
        background-color: rgba(255, 255, 255, 0.02);
      }
  
      /* Code highlighting */
      pre code.hljs {
        padding: 1rem !important;
        border-radius: 0.5rem;
      }

      /* Custom scrollbar styling */
      .chat-scrollbar::-webkit-scrollbar {
        width: 6px;
        height: 6px;
      }

      .chat-scrollbar::-webkit-scrollbar-track {
        background: transparent;
      }

      .chat-scrollbar::-webkit-scrollbar-thumb {
        background-color: rgba(155, 155, 155, 0.5);
        border-radius: 20px;
      }

      .chat-scrollbar::-webkit-scrollbar-thumb:hover {
        background-color: rgba(155, 155, 155, 0.7);
      }

      .chat-scrollbar {
        scrollbar-width: thin;
        scrollbar-color: rgba(155, 155, 155, 0.5) transparent;
      }

      /* Ensure scrollbar is always on the right */
      .chat-scrollbar {
        overflow-y: scroll;
        margin-right: 0;
        padding-right: 0;
        scrollbar-gutter: stable;
        position: relative;
      }

      /* Code block styling */
      .code-block {
        position: relative;
        margin: 1rem 0;
      }

      .code-block pre {
        padding-top: 2.5rem !important;
      }

      .code-header {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 1rem;
        background: rgba(0, 0, 0, 0.1);
        border-top-left-radius: 0.5rem;
        border-top-right-radius: 0.5rem;
        font-family: monospace;
        font-size: 0.8rem;
      }

      .dark .code-header {
        background: rgba(255, 255, 255, 0.1);
      }

      .copy-button {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0.25rem;
        border-radius: 0.25rem;
        cursor: pointer;
        transition: all 0.2s;
        background: transparent;
        border: none;
        color: inherit;
      }

      .copy-button:hover {
        background: rgba(0, 0, 0, 0.1);
      }

      .dark .copy-button:hover {
        background: rgba(255, 255, 255, 0.1);
      }

      /* Text copy button */
      .message-content {
        position: relative;
      }

      .message-copy-button {
        position: absolute;
        top: 0.5rem;
        right: 0.5rem;
        opacity: 0;
        transition: opacity 0.2s;
      }

      .message-bubble:hover .message-copy-button {
        opacity: 1;
      }
    `
    document.head.appendChild(style)

    return () => {
      document.head.removeChild(style)
    }
  }, [])

  // Reset copied text after 2 seconds
  useEffect(() => {
    if (copiedText) {
      const timer = setTimeout(() => {
        setCopiedText(null)
      }, 2000)
      return () => clearTimeout(timer)
    }
  }, [copiedText])

  // Text typing animation effect
  useEffect(() => {
    if (isTypingAnimation && currentTypingMessage) {
      const timer = setTimeout(() => {
        setIsTypingAnimation(false)
        setMessages((prev) => prev.map((msg) => (msg.isTyping ? { ...msg, isTyping: false } : msg)))
      }, 500)

      return () => clearTimeout(timer)
    }
  }, [isTypingAnimation, currentTypingMessage])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "24px"
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }, [input])

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileSelection = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    setIsUploading(true)
    setIsSidebarOpen(false)

    try {
      const formData = new FormData()
      Array.from(files).forEach((file) => {
        formData.append("files", file)
      })

      const response = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) throw new Error("Upload failed")
      handleUploadSuccess()
    } catch (error) {
      console.error("Upload error:", error)
    } finally {
      setIsUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ""
    }
  }

  const handleUploadSuccess = () => {
    const systemMessage: Message = {
      id: Date.now().toString(),
      role: "assistant",
      content:
        "New documents have been added to the knowledge base and are being processed. You can now ask questions about the new content!",
    }

    if (activeSessionId) {
      const newMessages = [...messages, systemMessage]
      setMessages(newMessages)
      saveMessagesToSession(activeSessionId, newMessages)
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
  }

  const getChatHistory = () => {
    return messages.map((msg) => ({
      role: msg.role,
      content: msg.content,
    }))
  }

  const simulateTypingAnimation = (content: string) => {
    setCurrentTypingMessage(content)
    setTypingIndex(0)
    setIsTypingAnimation(true)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!input.trim() || !activeSessionId) return

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
          chat_history: getChatHistory(),
        }),
      })

      if (!response.ok) throw new Error(`API error: ${response.status}`)
      const data = await response.json()

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.response,
        sources: data.sources || [],
        isTyping: true,
      }

      const newUpdatedMessages = [...updatedMessages, assistantMessage]
      setMessages(newUpdatedMessages)
      saveMessagesToSession(activeSessionId, newUpdatedMessages)
      simulateTypingAnimation(data.response)
      generateChatTitle(activeSessionId, userMessage.content, assistantMessage.content)
    } catch (error) {
      console.error("Error:", error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Sorry, I encountered an error while processing your request. Please try again later.",
      }

      const newUpdatedMessages = [...updatedMessages, errorMessage]
      setMessages(newUpdatedMessages)
      saveMessagesToSession(activeSessionId, newUpdatedMessages)
    } finally {
      setIsLoading(false)
    }
  }

  const generateChatTitle = async (sessionId: string, userQuery: string, aiResponse: string) => {
    const session = sessions.find((s) => s.id === sessionId)
    if (session && session.title === "New chat" && session.messages.length === 0) {
      const title = userQuery.length > 30 ? userQuery.substring(0, 30) + "..." : userQuery

      setSessions((prev) => prev.map((s) => (s.id === sessionId ? { ...s, title } : s)))
    }
  }

  const saveMessagesToSession = (sessionId: string, updatedMessages: Message[]) => {
    setSessions((prev) =>
      prev.map((session) => (session.id === sessionId ? { ...session, messages: updatedMessages } : session)),
    )
  }

  const startNewChat = () => {
    const newSessionId = Date.now().toString()
    const newSession: ChatSession = {
      id: newSessionId,
      title: "New chat",
      messages: [],
      createdAt: new Date(),
    }
    setSessions((prev) => [newSession, ...prev])
    setActiveSessionId(newSessionId)
    setMessages([])
  }

  const switchSession = (sessionId: string) => {
    setActiveSessionId(sessionId)
    if (isBrowser && window.innerWidth < 768) {
      setIsSidebarOpen(false)
    }
  }

  const deleteSession = (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setSessions((prev) => prev.filter((session) => session.id !== sessionId))

    if (sessionId === activeSessionId) {
      const remainingSessions = sessions.filter((session) => session.id !== sessionId)
      if (remainingSessions.length > 0) {
        setActiveSessionId(remainingSessions[0].id)
      } else {
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

  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
    }).format(date)
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    setCopiedText(text)
  }

  // Updated renderTypingAnimation function with proper type checking
  const renderTypingAnimation = (content: string) => {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="prose prose-sm dark:prose-invert max-w-none"
      >
        <ReactMarkdown
          remarkPlugins={[remarkMath]}
          rehypePlugins={[rehypeKatex, rehypeHighlight]}
          components={{
            code({ node, inline, className, children, ...props }: any) {
              const match = /language-(\w+)/.exec(className || "")
              return !inline && match ? (
                <div className="relative my-4">
                  <div className="absolute right-2 top-2 text-xs text-muted-foreground">{match[1]}</div>
                  <pre className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 overflow-x-auto">
                    <code className={`language-${match[1]}`} {...props}>
                      {children}
                    </code>
                  </pre>
                </div>
              ) : (
                <code className="bg-muted px-1.5 py-0.5 rounded text-sm" {...props}>
                  {children}
                </code>
              )
            },
            // Fix for math display
            div: ({ node, className, children, ...props }: any) => {
              // Check if this is a math display node
              const isMathDisplay = node?.properties?.className
                ? Array.isArray(node.properties.className)
                  ? node.properties.className.includes("math-display")
                  : node.properties.className.includes("math-display")
                : false

              if (isMathDisplay) {
                return <div className="math-display overflow-x-auto py-2 my-4">{children}</div>
              }
              return (
                <div className={className} {...props}>
                  {children}
                </div>
              )
            },
            // Fix for inline math
            span: ({ node, className, children, ...props }: any) => {
              // Check if this is a math inline node
              const isMathInline = node?.properties?.className
                ? Array.isArray(node.properties.className)
                  ? node.properties.className.includes("math-inline")
                  : node.properties.className.includes("math-inline")
                : false

              if (isMathInline) {
                return <span className="math-inline mx-1 inline-block">{children}</span>
              }
              return (
                <span className={className} {...props}>
                  {children}
                </span>
              )
            },
          }}
        >
          {content}
        </ReactMarkdown>
      </motion.div>
    )
  }

  const isDesktop = isBrowser ? window.innerWidth >= 768 : false
  const isDarkTheme = theme === "dark"

  return (
    <div className="flex h-[100dvh] bg-background text-foreground overflow-hidden">
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        multiple
        onChange={handleFileSelection}
        accept=".pdf,.txt,.md,.docx,.pptx"
      />

      <AnimatePresence>
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.2 }}
          className="absolute top-4 left-4 md:hidden z-50"
        >
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="rounded-full bg-background/90 backdrop-blur-md shadow-lg"
          >
            {isSidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </Button>
        </motion.div>
      </AnimatePresence>

      <motion.div
        initial={false}
        animate={{
          width: isSidebarCollapsed ? "4rem" : "16rem",
          x: isSidebarOpen || isDesktop ? 0 : "-100%",
        }}
        transition={{ duration: 0.3, ease: "easeInOut" }}
        className={cn(
          "border-r border-border flex-shrink-0 flex flex-col",
          "fixed md:static inset-y-0 left-0 z-40 bg-background/95 backdrop-blur-md",
        )}
      >
        <div
          className={cn(
            "border-b border-border/50 flex items-center py-4",
            isSidebarCollapsed ? "justify-center px-2" : "px-4",
          )}
        >
          {!isSidebarCollapsed ? (
            <div className="flex items-center justify-between w-full">
              <Button
                variant="outline"
                className="flex-1 justify-start gap-2 text-foreground hover:bg-primary/10 hover:text-primary transition-all rounded-xl"
                onClick={startNewChat}
              >
                <Sparkles size={16} className="animate-pulse" />
                New chat
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="ml-2 h-8 w-8 rounded-full hover:bg-muted"
                onClick={toggleSidebar}
                title="Collapse sidebar"
              >
                <ChevronLeft size={16} />
              </Button>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <Button
                variant="outline"
                size="icon"
                className="text-foreground hover:bg-primary/10 hover:text-primary transition-all rounded-full"
                onClick={startNewChat}
              >
                <Sparkles size={16} className="animate-pulse" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 rounded-full hover:bg-muted"
                onClick={toggleSidebar}
                title="Expand sidebar"
              >
                <ChevronLeft size={16} className="rotate-180" />
              </Button>
            </div>
          )}
        </div>

        <div className="flex-1 overflow-y-auto p-2 chat-scrollbar">
          <AnimatePresence>
            {!isSidebarCollapsed && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="space-y-1 mt-2"
              >
                <div className="mb-4">
                  <h3 className="text-xs font-semibold text-muted-foreground mb-2 px-2">Chat History</h3>
                  <motion.div
                    className="space-y-1"
                    initial="hidden"
                    animate="visible"
                    variants={{
                      visible: { transition: { staggerChildren: 0.05 } },
                    }}
                  >
                    {sessions.map((session) => (
                      <motion.div
                        key={session.id}
                        variants={{
                          hidden: { opacity: 0, x: -20 },
                          visible: {
                            opacity: 1,
                            x: 0,
                            transition: { duration: 0.3 },
                          },
                        }}
                        className={cn(
                          "group w-full text-left rounded-xl flex items-center gap-2 text-sm transition-all",
                          activeSessionId === session.id
                            ? "bg-primary/10 text-primary shadow-md"
                            : "hover:bg-muted text-foreground",
                        )}
                      >
                        <button
                          className="flex-1 flex items-start gap-2 p-3 truncate text-left"
                          onClick={() => switchSession(session.id)}
                        >
                          <MessageSquare size={16} className="mt-0.5 flex-shrink-0" />
                          <div className="flex-1 flex flex-col overflow-hidden">
                            <span className="truncate font-medium">{session.title}</span>
                            <span className="text-xs text-muted-foreground truncate">
                              {formatDate(session.createdAt)}
                            </span>
                          </div>
                        </button>
                        <motion.button
                          whileHover={{ scale: 1.1 }}
                          whileTap={{ scale: 0.95 }}
                          className="opacity-0 group-hover:opacity-100 text-muted-foreground hover:text-destructive mr-2 p-1 rounded-md hover:bg-destructive/10 transition-colors"
                          onClick={(e) => deleteSession(session.id, e)}
                          aria-label="Delete chat"
                        >
                          <Trash2 size={16} />
                        </motion.button>
                      </motion.div>
                    ))}
                  </motion.div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className={cn("border-t border-border/50", isSidebarCollapsed ? "p-2" : "p-4", "space-y-4")}>
          {!isSidebarCollapsed ? (
            <>
              <div className="mb-4">
                <Button
                  variant="outline"
                  className="w-full justify-start gap-2"
                  onClick={handleUploadClick}
                  disabled={isUploading}
                >
                  <Upload size={16} />
                  <span>{isUploading ? "Uploading..." : "Upload Documents"}</span>
                </Button>
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start gap-2 text-muted-foreground hover:bg-muted transition-colors rounded-xl"
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
              <Button
                variant="ghost"
                size="icon"
                className="w-full flex justify-center text-muted-foreground hover:bg-muted transition-colors rounded-full"
                onClick={handleUploadClick}
                disabled={isUploading}
                title="Upload documents"
              >
                {isUploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload size={16} />}
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="w-full flex justify-center text-muted-foreground hover:bg-muted transition-colors rounded-full"
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
      </motion.div>

      {isSidebarOpen && !isDesktop && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 bg-black/20 backdrop-blur-sm z-30 md:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      <div className="flex-1 flex flex-col relative">
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          <div className="absolute -top-32 -right-32 w-64 h-64 bg-primary/5 rounded-full blur-3xl"></div>
          <div className="absolute top-1/4 -left-32 w-64 h-64 bg-primary/5 rounded-full blur-3xl"></div>
        </div>

        <div ref={chatContainerRef} className="flex-1 overflow-y-auto py-6 chat-scrollbar relative">
          {messages.length === 0 ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="h-full flex flex-col items-center justify-center p-4 md:p-8"
            >
              <motion.div
                whileHover={{ scale: 1.05, rotate: 5 }}
                transition={{ type: "spring", stiffness: 400, damping: 10 }}
                className="w-20 h-20 rounded-full bg-blue-100 flex items-center justify-center mb-8"
              >
                <FileText className="h-10 w-10 text-blue-500" />
              </motion.div>
              <h1 className="text-3xl font-bold mb-3 text-blue-500">Personal Knowledge Assistant</h1>
              <div className="max-w-md text-center text-muted-foreground">
                <p className="text-lg">Ask me anything from your knowledge base.</p>
                <p className="mt-2">
                  Use the <span className="font-medium">Upload</span> button to add new documents.
                </p>
              </div>
            </motion.div>
          ) : (
            <div className="space-y-6 max-w-3xl mx-auto w-full">
              <AnimatePresence>
                {messages.map((message) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                    className={cn(
                      "message-bubble flex w-full items-start gap-4 px-6 py-4",
                      message.role === "user" ? "bg-primary/5" : "bg-card/95",
                      message.isTyping && "glassmorphism",
                    )}
                  >
                    <div className="flex-shrink-0 mt-1">
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
                    <div className="flex-1 min-w-0 message-content">
                      <button
                        onClick={() => copyToClipboard(message.content)}
                        className="message-copy-button p-1.5 rounded-full bg-background/80 text-muted-foreground hover:text-foreground hover:bg-background transition-colors"
                        title="Copy message"
                      >
                        {copiedText === message.content ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                      </button>

                      {message.isTyping ? (
                        <div className="prose prose-sm dark:prose-invert max-w-none">
                          {renderTypingAnimation(message.content)}
                        </div>
                      ) : (
                        <ReactMarkdown
                          remarkPlugins={[remarkMath]}
                          rehypePlugins={[rehypeKatex, rehypeHighlight]}
                          components={{
                            code({ node, inline, className, children, ...props }: any) {
                              const match = /language-(\w+)/.exec(className || "")
                              const codeContent = String(children).replace(/\n$/, "")

                              return !inline && match ? (
                                <div className="code-block">
                                  <div className="code-header">
                                    <span>{match[1]}</span>
                                    <button
                                      onClick={() => copyToClipboard(codeContent)}
                                      className="copy-button"
                                      title="Copy code"
                                    >
                                      {copiedText === codeContent ? (
                                        <Check className="h-4 w-4" />
                                      ) : (
                                        <Copy className="h-4 w-4" />
                                      )}
                                    </button>
                                  </div>
                                  <pre className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 overflow-x-auto">
                                    <code className={`language-${match[1]}`} {...props}>
                                      {children}
                                    </code>
                                  </pre>
                                </div>
                              ) : (
                                <code className="bg-muted px-1.5 py-0.5 rounded text-sm" {...props}>
                                  {children}
                                </code>
                              )
                            },
                            // Fix for math display
                            div: ({ node, className, children, ...props }: any) => {
                              // Check if this is a math display node
                              const isMathDisplay = node?.properties?.className
                                ? Array.isArray(node.properties.className)
                                  ? node.properties.className.includes("math-display")
                                  : node.properties.className.includes("math-display")
                                : false

                              if (isMathDisplay) {
                                return <div className="math-display overflow-x-auto py-2 my-4">{children}</div>
                              }
                              return (
                                <div className={className} {...props}>
                                  {children}
                                </div>
                              )
                            },
                            // Fix for inline math
                            span: ({ node, className, children, ...props }: any) => {
                              // Check if this is a math inline node
                              const isMathInline = node?.properties?.className
                                ? Array.isArray(node.properties.className)
                                  ? node.properties.className.includes("math-inline")
                                  : node.properties.className.includes("math-inline")
                                : false

                              if (isMathInline) {
                                return <span className="math-inline mx-1 inline-block">{children}</span>
                              }
                              return (
                                <span className={className} {...props}>
                                  {children}
                                </span>
                              )
                            },
                          }}
                        >
                          {message.content}
                        </ReactMarkdown>
                      )}

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
                  </motion.div>
                ))}
              </AnimatePresence>

              {isLoading && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="message-bubble flex w-full bg-card/70 items-start gap-4 px-6 py-4 glassmorphism"
                >
                  <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                    <Bot className="h-5 w-5 text-primary-foreground" />
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex space-x-1">
                      <motion.div
                        animate={{ y: [0, -5, 0] }}
                        transition={{
                          duration: 0.5,
                          repeat: Number.POSITIVE_INFINITY,
                          repeatDelay: 0.1,
                        }}
                        className="w-2 h-2 rounded-full bg-primary"
                      />
                      <motion.div
                        animate={{ y: [0, -5, 0] }}
                        transition={{
                          duration: 0.5,
                          repeat: Number.POSITIVE_INFINITY,
                          repeatDelay: 0.2,
                        }}
                        className="w-2 h-2 rounded-full bg-primary"
                      />
                      <motion.div
                        animate={{ y: [0, -5, 0] }}
                        transition={{
                          duration: 0.5,
                          repeat: Number.POSITIVE_INFINITY,
                          repeatDelay: 0.3,
                        }}
                        className="w-2 h-2 rounded-full bg-primary"
                      />
                    </div>
                    <span className="text-sm text-muted-foreground">Thinking...</span>
                  </div>
                </motion.div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="p-3 md:p-4"
        >
          <div className="max-w-3xl mx-auto">
            <div
              className={cn(
                "relative rounded-2xl shadow-lg",
                isDarkTheme ? "bg-gray-900" : "bg-white border border-gray-200",
              )}
            >
              <form onSubmit={handleSubmit} className="relative">
                <div className="absolute left-3 bottom-3 flex items-center">
                  <button
                    type="button"
                    onClick={handleUploadClick}
                    className={cn(
                      "p-1.5 rounded-full transition-colors",
                      isDarkTheme
                        ? "text-gray-400 hover:bg-gray-800 hover:text-gray-300"
                        : "text-gray-500 hover:bg-gray-100 hover:text-gray-700",
                    )}
                    disabled={isUploading}
                  >
                    {isUploading ? <Loader2 className="h-[18px] w-[18px] animate-spin" /> : <Upload size={18} />}
                  </button>
                </div>

                <Textarea
                  ref={textareaRef}
                  value={input}
                  onChange={handleInputChange}
                  placeholder="Ask anything"
                  className={cn(
                    "resize-none py-4 pl-12 pr-12 min-h-[56px] max-h-[200px] rounded-2xl border-0 focus:ring-0",
                    isDarkTheme
                      ? "bg-transparent text-white placeholder:text-gray-400"
                      : "bg-transparent text-gray-900 placeholder:text-gray-500",
                  )}
                  rows={1}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault()
                      handleSubmit(e as any)
                    }
                  }}
                />

                <div className="absolute right-3 bottom-3 flex items-center">
                  <button
                    type="submit"
                    className={cn(
                      "p-1.5 rounded-full transition-colors",
                      isDarkTheme
                        ? "text-gray-400 hover:bg-gray-800 hover:text-gray-300"
                        : "text-gray-500 hover:bg-gray-100 hover:text-gray-700",
                      !input.trim() && (isDarkTheme ? "text-gray-600" : "text-gray-300"),
                    )}
                    disabled={isLoading || !input.trim()}
                  >
                    <Send size={18} className={input.trim() ? (isDarkTheme ? "text-white" : "text-blue-500") : ""} />
                  </button>
                </div>
              </form>
            </div>
            <div className="text-center mt-2 text-xs text-muted-foreground">
              <span>Personal Knowledge Assistant can make mistakes. Check important info.</span>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
