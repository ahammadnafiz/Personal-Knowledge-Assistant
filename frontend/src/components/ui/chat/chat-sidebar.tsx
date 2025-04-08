"use client"

import type React from "react"

import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/button"
import { cn } from "@/libs/utils"
import { ChevronLeft, Sparkles, MessageSquare, Trash2, Upload, Sun, Moon, Book, Loader2 } from "lucide-react"
import type { ChatSession } from "./use-chat-state"

interface ChatSidebarProps {
  isSidebarOpen: boolean
  isSidebarCollapsed: boolean
  isDesktop: boolean
  toggleSidebar: () => void
  sessions: ChatSession[]
  activeSessionId: string | null
  startNewChat: () => void
  switchSession: (sessionId: string) => void
  deleteSession: (sessionId: string, e: React.MouseEvent) => void
  handleUploadClick: () => void
  isUploading: boolean
  toggleTheme: () => void
  theme: string | undefined
}

export function ChatSidebar({
  isSidebarOpen,
  isSidebarCollapsed,
  isDesktop,
  toggleSidebar,
  sessions,
  activeSessionId,
  startNewChat,
  switchSession,
  deleteSession,
  handleUploadClick,
  isUploading,
  toggleTheme,
  theme,
}: ChatSidebarProps) {
  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
    }).format(date)
  }

  return (
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
  )
}