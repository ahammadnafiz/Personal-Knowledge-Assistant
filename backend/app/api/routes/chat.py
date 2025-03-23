# app/api/routes/chat.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any

from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag import RAGService

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message using the RAG system and return a response
    """
    try:
        rag_service = RAGService()
        response, sources = rag_service.process_query(
            query=request.query,
            chat_history=request.chat_history
        )
        
        return ChatResponse(
            response=response,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@router.get("/chat/history")
async def get_chat_history():
    """
    Get the current chat history
    """
    try:
        rag_service = RAGService()
        return rag_service.get_chat_history()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")

@router.delete("/chat/history")
async def clear_chat_history():
    """
    Clear the current chat history
    """
    try:
        rag_service = RAGService()
        rag_service.clear_memory()
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing chat history: {str(e)}")