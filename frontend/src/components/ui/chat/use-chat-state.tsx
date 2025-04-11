"use client"

import type React from "react"

import { useState, useEffect } from "react"

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

export function useChatState(apiUrl: string) {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isTypingAnimation, setIsTypingAnimation] = useState(false)
  const [currentTypingMessage, setCurrentTypingMessage] = useState<string>("")
  const [typingIndex, setTypingIndex] = useState(0)
  const [copiedText, setCopiedText] = useState<string | null>(null)
  const [isWebSearchEnabled, setIsWebSearchEnabled] = useState(false)

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
      console.log("Sending request with web search enabled:", isWebSearchEnabled)

      const response = await fetch(`${apiUrl}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: input.trim(),
          chat_history: getChatHistory(),
          web_search_enabled: isWebSearchEnabled,
        }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        console.error("API error:", response.status, errorText)
        throw new Error(`API error: ${response.status} - ${errorText}`)
      }

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

  const toggleWebSearch = () => {
    setIsWebSearchEnabled((prev) => !prev)
  }

  return {
    sessions,
    setSessions,
    activeSessionId,
    setActiveSessionId,
    messages,
    setMessages,
    input,
    setInput,
    isLoading,
    isTypingAnimation,
    currentTypingMessage,
    typingIndex,
    copiedText,
    setCopiedText,
    isWebSearchEnabled,
    toggleWebSearch,
    handleSubmit,
    saveMessagesToSession,
    startNewChat,
    switchSession,
    deleteSession,
    handleUploadSuccess,
    simulateTypingAnimation,
  }
}

export type { Message, ChatSession }
