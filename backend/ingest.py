# ingest.py
import argparse
import sys
import os

# Add the current directory to the Python path to make app package importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.rag import RAGService

def main():
    parser = argparse.ArgumentParser(description='Ingest books into the knowledge base')
    parser.add_argument('--dir', type=str, required=True, help='Directory containing PDF books')
    args = parser.parse_args()
    
    rag_service = RAGService()
    num_chunks = rag_service.ingest_documents(args.dir)
    print(f"Successfully ingested documents. Created {num_chunks} text chunks in the vector store.")

if __name__ == "__main__":
    main()