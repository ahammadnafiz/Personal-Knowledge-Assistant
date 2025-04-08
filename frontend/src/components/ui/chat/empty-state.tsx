"use client"

import { motion } from "framer-motion"
import { FileText } from "lucide-react"

export function EmptyState() {
  return (
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
  )
}