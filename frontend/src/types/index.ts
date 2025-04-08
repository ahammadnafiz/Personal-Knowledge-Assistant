export type Message = {
    id: string
    role: "user" | "assistant"
    content: string
    sources?: string[]
    isTyping?: boolean
  }
  
  export type ChatSession = {
    id: string
    title: string
    messages: Message[]
    createdAt: Date
  }  