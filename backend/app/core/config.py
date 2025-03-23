# app/core/config.py
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "Personal Knowledge Assistant"
    API_PREFIX: str = "/api"
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # Vector store settings
    VECTOR_STORE_PATH: str = "data/vector_store"
    
    # Embedding settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # LLM settings
    LLM_MODEL: str = "qwen-2.5-32b"
    
    # RAG settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 5

settings = Settings()