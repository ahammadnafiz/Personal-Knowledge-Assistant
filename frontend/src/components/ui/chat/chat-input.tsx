"use client"

import type React from "react"

import { useRef, useEffect } from "react"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/libs/utils"
import { Send, Upload, Loader2, Globe } from "lucide-react"

interface ChatInputProps {
  input: string
  setInput: (value: string) => void
  handleSubmit: (e: React.FormEvent) => void
  isLoading: boolean
  handleUploadClick: () => void
  isUploading: boolean
  isDarkTheme: boolean
  isWebSearchEnabled: boolean
  toggleWebSearch: () => void
}

export function ChatInput({
  input,
  setInput,
  handleSubmit,
  isLoading,
  handleUploadClick,
  isUploading,
  isDarkTheme,
  isWebSearchEnabled,
  toggleWebSearch,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "24px"
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }, [input])

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
  }

  return (
    <div className="max-w-3xl mx-auto">
      <div
        className={cn(
          "relative rounded-2xl shadow-lg",
          isDarkTheme ? "bg-gray-900" : "bg-white border border-gray-200",
        )}
      >
        <form onSubmit={handleSubmit} className="relative">
          <div className="absolute left-3 bottom-3 flex items-center space-x-2">
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
              title="Upload documents"
            >
              {isUploading ? <Loader2 className="h-[18px] w-[18px] animate-spin" /> : <Upload size={18} />}
            </button>

            <button
              type="button"
              onClick={toggleWebSearch}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1.5 rounded-full transition-colors text-sm",
                isWebSearchEnabled
                  ? isDarkTheme
                    ? "bg-blue-600 text-white hover:bg-blue-700"
                    : "bg-transparent border border-blue-500 text-blue-600 hover:bg-blue-50"
                  : isDarkTheme
                    ? "bg-gray-800 text-gray-300 hover:bg-gray-700"
                    : "bg-transparent border border-gray-300 text-gray-600 hover:bg-gray-50",
              )}
              title={isWebSearchEnabled ? "Web search enabled" : "Web search disabled"}
            >
              <Globe size={14} className={isWebSearchEnabled ? (isDarkTheme ? "text-white" : "text-blue-500") : ""} />
              <span>Search</span>
            </button>
          </div>

          <Textarea
            ref={textareaRef}
            value={input}
            onChange={handleInputChange}
            placeholder="Ask anything"
            className={cn(
              "resize-none py-4 pl-36 pr-12 min-h-[56px] max-h-[200px] rounded-2xl border-0 focus:ring-0",
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
  )
}
