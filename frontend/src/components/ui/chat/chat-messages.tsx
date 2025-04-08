"use client"

import type React from "react"

import { motion, AnimatePresence } from "framer-motion"
import { Bot, User, Copy, Check } from "lucide-react"
import { cn } from "@/libs/utils"
import ReactMarkdown from "react-markdown"
import remarkMath from "remark-math"
import rehypeKatex from "rehype-katex"
import rehypeHighlight from "rehype-highlight"
import "katex/dist/katex.min.css"
import "highlight.js/styles/github-dark.css"
import type { Message } from "./use-chat-state"

interface ChatMessagesProps {
  messages: Message[]
  isLoading: boolean
  copiedText: string | null
  setCopiedText: (text: string | null) => void
  messagesEndRef: React.RefObject<HTMLDivElement>
}

export function ChatMessages({ messages, isLoading, copiedText, setCopiedText, messagesEndRef }: ChatMessagesProps) {
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
    setCopiedText(text)
  }

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
            div: ({ node, className, children, ...props }: any) => {
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
            span: ({ node, className, children, ...props }: any) => {
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

  return (
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
                    div: ({ node, className, children, ...props }: any) => {
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
                    span: ({ node, className, children, ...props }: any) => {
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
  )
}