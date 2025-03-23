# app/models/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = None