# app/models/schemas.py
from typing import List, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[dict]] = []


class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = None
