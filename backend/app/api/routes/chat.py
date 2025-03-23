# app/api/routes/chat.py
from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag import RAGService

router = APIRouter()
rag_service = RAGService()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response, sources = rag_service.process_query(request.query, request.chat_history)
        return ChatResponse(response=response, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
