import os
import shutil
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List, Optional

from app.services.rag import RAGService
from app.core.config import settings

router = APIRouter()

# Ensure the upload directories exist
UPLOAD_DIR = "knowledge_base"
IMAGE_DIR = "image_base"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

@router.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    Upload files to the knowledge base and trigger ingestion into the vector store
    """
    # Validate file types
    valid_document_extensions = [".pdf", ".txt", ".md", ".docx"]
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp"]
    
    document_files = []
    image_files = []
    
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in valid_document_extensions:
            document_files.append(file)
        elif ext in valid_image_extensions:
            image_files.append(file)
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Only {', '.join(valid_document_extensions + valid_image_extensions)} are supported."
            )
    
    # Save uploaded files
    saved_documents = []
    saved_images = []
    
    # Save document files
    for file in document_files:
        try:
            # Generate a unique filename to avoid conflicts
            unique_filename = f"{uuid.uuid4()}-{file.filename}"
            file_path = os.path.join(UPLOAD_DIR, unique_filename)
            
            # Save the file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            saved_documents.append(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving document file: {str(e)}")
    
    # Save image files
    for file in image_files:
        try:
            # Generate a unique filename to avoid conflicts
            unique_filename = f"{uuid.uuid4()}-{file.filename}"
            file_path = os.path.join(IMAGE_DIR, unique_filename)
            
            # Save the file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            saved_images.append(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving image file: {str(e)}")
    
    # Trigger background task to ingest the document files
    if saved_documents:
        background_tasks.add_task(ingest_files, saved_documents)
    
    # Process images if needed (this would depend on your RAG implementation)
    if saved_images:
        background_tasks.add_task(process_images, saved_images)
    
    return {
        "message": f"Successfully uploaded {len(saved_documents)} documents and {len(saved_images)} images. Processing has started.",
        "documents": [os.path.basename(path) for path in saved_documents],
        "images": [os.path.basename(path) for path in saved_images]
    }

def ingest_files(file_paths: List[str]):
    """
    Ingest specific files into the vector store
    """
    try:
        rag_service = RAGService(google_api_key=settings.GOOGLE_API_KEY)
        # Get just the filenames for specific file ingestion
        filenames = [os.path.basename(path) for path in file_paths]
        num_chunks = rag_service.ingest_documents(UPLOAD_DIR, specific_files=filenames)
        print(f"Successfully ingested documents. Created {num_chunks} text chunks in the vector store.")
    except Exception as e:
        print(f"Error during document ingestion: {str(e)}")

def process_images(file_paths: List[str]):
    """
    Process images for the RAG system
    This function would depend on your specific implementation for handling images
    """
    try:
        # This is a placeholder - you would need to implement image processing
        # based on your specific RAG implementation
        print(f"Processing {len(file_paths)} images")
        # Example: You might extract text from images using OCR
        # or store image metadata in a database
    except Exception as e:
        print(f"Error during image processing: {str(e)}")

@router.get("/images")
async def list_images():
    """
    List all available images
    """
    try:
        images = []
        for filename in os.listdir(IMAGE_DIR):
            if os.path.isfile(os.path.join(IMAGE_DIR, filename)):
                images.append({
                    "filename": filename,
                    "path": f"/api/images/{filename}"
                })
        return {"images": images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing images: {str(e)}")

@router.get("/images/{filename}")
async def get_image(filename: str):
    """
    Serve an image file
    """
    file_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Determine content type based on file extension
    ext = os.path.splitext(filename)[1].lower()
    content_type = "image/jpeg"  # Default
    if ext == ".png":
        content_type = "image/png"
    elif ext == ".gif":
        content_type = "image/gif"
    elif ext == ".webp":
        content_type = "image/webp"
    
    # Return the file as a response
    from fastapi.responses import FileResponse
    return FileResponse(file_path, media_type=content_type)