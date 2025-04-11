"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Menu, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { motion, AnimatePresence } from "framer-motion"
import { useTheme } from "next-themes"
import { ChatSidebar } from "./chat-sidebar"
import { ChatMessages } from "./chat-messages"
import { ChatInput } from "./chat-input"
import { EmptyState } from "./empty-state"
import { useChatState } from "./use-chat-state"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api"

export default function ModernChatInterface() {
  const {
    sessions,
    activeSessionId,
    messages,
    isLoading,
    input,
    setInput,
    copiedText,
    setCopiedText,
    isWebSearchEnabled,
    toggleWebSearch,
    handleSubmit,
    startNewChat,
    switchSession,
    deleteSession,
    saveMessagesToSession,
    handleUploadSuccess,
  } = useChatState(API_URL)

  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [isBrowser, setIsBrowser] = useState(false)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null) as React.RefObject<HTMLDivElement>
  const chatContainerRef = useRef<HTMLDivElement>(null)

  const { theme, setTheme } = useTheme()

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

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Reset copied text after 2 seconds
  useEffect(() => {
    if (copiedText) {
      const timer = setTimeout(() => {
        setCopiedText(null)
      }, 2000)
      return () => clearTimeout(timer)
    }
  }, [copiedText, setCopiedText])

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

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark")
  }

  const toggleSidebar = () => {
    setIsSidebarCollapsed(!isSidebarCollapsed)
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

      <ChatSidebar
        isSidebarOpen={isSidebarOpen}
        isSidebarCollapsed={isSidebarCollapsed}
        isDesktop={isDesktop}
        toggleSidebar={toggleSidebar}
        sessions={sessions}
        activeSessionId={activeSessionId}
        startNewChat={startNewChat}
        switchSession={switchSession}
        deleteSession={deleteSession}
        handleUploadClick={handleUploadClick}
        isUploading={isUploading}
        toggleTheme={toggleTheme}
        theme={theme}
      />

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
            <EmptyState />
          ) : (
            <ChatMessages
              messages={messages}
              isLoading={isLoading}
              copiedText={copiedText}
              setCopiedText={setCopiedText}
              messagesEndRef={messagesEndRef}
            />
          )}
        </div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="p-3 md:p-4"
        >
          <ChatInput
            input={input}
            setInput={setInput}
            handleSubmit={handleSubmit}
            isLoading={isLoading}
            handleUploadClick={handleUploadClick}
            isUploading={isUploading}
            isDarkTheme={isDarkTheme}
            isWebSearchEnabled={isWebSearchEnabled}
            toggleWebSearch={toggleWebSearch}
          />
        </motion.div>
      </div>
    </div>
  )
}
