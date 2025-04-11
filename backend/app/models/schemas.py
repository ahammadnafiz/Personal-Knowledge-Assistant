# app/models/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict[str, Any]]] = None
    web_search_enabled: Optional[bool] = Field(default=None)

class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []
