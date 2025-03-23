# app/main.py (Updated to include upload router)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.api.routes.chat import router as chat_router
from app.api.routes.upload import router as upload_router
from app.core.config import settings

app = FastAPI(title="Personal Knowledge Assistant")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")
app.include_router(upload_router, prefix="/api")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}