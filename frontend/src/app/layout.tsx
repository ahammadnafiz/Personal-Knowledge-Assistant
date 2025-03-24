// app/layout.tsx
import "./globals.css"
import type { Metadata } from "next"
import { Quicksand } from "next/font/google"
import { ThemeProvider } from "next-themes"

const quicksand = Quicksand({ 
  subsets: ["latin"],
  variable: '--font-quicksand',
})

export const metadata: Metadata = {
  title: "Knowledge Assistant",
  description: "AI-powered knowledge assistant using RAG",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${quicksand.className} antialiased`}>
        <ThemeProvider 
          attribute="class" 
          defaultTheme="system" 
          enableSystem 
          disableTransitionOnChange
        >
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}