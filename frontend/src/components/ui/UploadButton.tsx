// components/UploadButton.tsx
"use client"

import React, { useRef, useState } from 'react'
import { Upload, Check, AlertCircle, Loader2 } from 'lucide-react'
import { Button } from "@/components/ui/button"
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle,
  DialogTrigger,
  DialogFooter,
  DialogClose
} from "@/components/ui/dialog"
import { Alert, AlertDescription } from "@/components/ui/alert"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api"

type UploadStatus = 'idle' | 'uploading' | 'success' | 'error'

export default function UploadButton({ onSuccess }: { onSuccess?: () => void }) {
  const [isOpen, setIsOpen] = useState(false)
  const [files, setFiles] = useState<File[]>([])
  const [status, setStatus] = useState<UploadStatus>('idle')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      // Convert FileList to Array
      const fileArray = Array.from(e.target.files)
      
      // Filter for PDFs only
      const pdfFiles = fileArray.filter(file => file.type === 'application/pdf')
      
      // Set the files state
      setFiles(pdfFiles)
      
      // Show error if some files were filtered out
      if (pdfFiles.length < fileArray.length) {
        setErrorMessage('Only PDF files are supported.')
      } else {
        setErrorMessage(null)
      }
    }
  }

  const triggerFileInput = () => {
    fileInputRef.current?.click()
  }

  const uploadFiles = async () => {
    if (files.length === 0) return
    
    setStatus('uploading')
    setErrorMessage(null)
    
    const formData = new FormData()
    files.forEach(file => {
      formData.append('files', file)
    })
    
    try {
      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Upload failed')
      }
      
      setStatus('success')
      
      // Call onSuccess callback if provided
      if (onSuccess) {
        onSuccess()
      }
      
      // Reset after 3 seconds
      setTimeout(() => {
        setFiles([])
        setStatus('idle')
        setIsOpen(false)
      }, 3000)
      
    } catch (error) {
      console.error('Upload error:', error)
      setStatus('error')
      setErrorMessage(error instanceof Error ? error.message : 'An unknown error occurred')
    }
  }

  // Reset when dialog closes
  const handleOpenChange = (open: boolean) => {
    setIsOpen(open)
    if (!open) {
      setFiles([])
      setStatus('idle')
      setErrorMessage(null)
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        <Button
          variant="outline"
          className="justify-start gap-2 text-foreground"
          onClick={() => setIsOpen(true)}
        >
          <Upload size={16} />
          Upload files
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Upload files to knowledge base</DialogTitle>
        </DialogHeader>
        
        <div className="flex flex-col gap-4 py-4">
          {status === 'error' && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                {errorMessage || 'There was an error uploading your files.'}
              </AlertDescription>
            </Alert>
          )}
          
          {status === 'success' && (
            <Alert className="bg-green-500/10 text-green-500 border-green-500/20">
              <Check className="h-4 w-4" />
              <AlertDescription>
                Files uploaded successfully! Processing has started.
              </AlertDescription>
            </Alert>
          )}
          
          <div
            className="border-2 border-dashed rounded-lg p-6 text-center cursor-pointer hover:border-primary/50 transition-colors"
            onClick={triggerFileInput}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              className="hidden"
              multiple
              accept=".pdf"
            />
            
            <div className="flex flex-col items-center gap-2">
              <Upload className="h-10 w-10 text-muted-foreground" />
              <div className="text-lg font-medium">Click to upload PDFs</div>
              <p className="text-sm text-muted-foreground">
                Drag and drop not supported. PDF files only.
              </p>
            </div>
          </div>
          
          {files.length > 0 && (
            <div className="mt-2">
              <div className="font-medium mb-2">Selected files:</div>
              <ul className="text-sm space-y-1">
                {files.map((file, index) => (
                  <li key={index} className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded-full bg-primary/20 flex items-center justify-center">
                      <Check className="h-3 w-3 text-primary" />
                    </div>
                    <span className="truncate">{file.name}</span>
                    <span className="text-muted-foreground text-xs">
                      ({Math.round(file.size / 1024)} KB)
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
        
        <DialogFooter className="flex items-center justify-between sm:justify-between">
          <DialogClose asChild>
            <Button variant="outline" disabled={status === 'uploading'}>
              Cancel
            </Button>
          </DialogClose>
          
          <Button
            onClick={uploadFiles}
            disabled={files.length === 0 || status === 'uploading' || status === 'success'}
          >
            {status === 'uploading' ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Uploading...
              </>
            ) : (
              'Upload'
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}