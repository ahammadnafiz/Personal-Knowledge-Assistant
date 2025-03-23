# app/api/routes/upload.py
import os
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List

from app.services.rag import RAGService
from app.core.config import settings

router = APIRouter()

# Ensure the upload directory exists
UPLOAD_DIR = "books"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    Upload files to the knowledge base and trigger ingestion into the vector store
    """
    # Validate file types
    valid_extensions = [".pdf"]
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in valid_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Only {', '.join(valid_extensions)} are supported."
            )
    
    # Save uploaded files
    saved_files = []
    for file in files:
        try:
            # Generate a unique filename to avoid conflicts
            unique_filename = f"{uuid.uuid4()}-{file.filename}"
            file_path = os.path.join(UPLOAD_DIR, unique_filename)
            
            # Save the file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            saved_files.append(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    # Trigger background task to ingest the files
    background_tasks.add_task(ingest_files, saved_files)
    
    return {"message": f"Successfully uploaded {len(saved_files)} files. Processing has started."}

def ingest_files(file_paths: List[str]):
    """
    Ingest specific files into the vector store
    """
    try:
        rag_service = RAGService()
        # Get just the filenames for specific file ingestion
        filenames = [os.path.basename(path) for path in file_paths]
        num_chunks = rag_service.ingest_documents(UPLOAD_DIR, specific_files=filenames)
        print(f"Successfully ingested documents. Created {num_chunks} text chunks in the vector store.")
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")