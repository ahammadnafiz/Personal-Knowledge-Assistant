# app/core/config.py
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "Personal Knowledge Assistant"
    API_PREFIX: str = "/api"
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")  
    # Vector store settings
    VECTOR_STORE_PATH: str = "data/vector_store"
    
    # Embedding settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    # EMBEDDING_MODEL: str = "intfloat/e5-small"
    # EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    
    # LLM settings
    # LLM_MODEL: str = "gemini-1.5-flash"
    LLM_MODEL: str = "gemini-2.0-flash"
    
    # RAG settings
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 8

settings = Settings()