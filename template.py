import os

def create_files_and_dirs(base_path, structure):
    for path in structure:
        full_path = os.path.join(base_path, path)
        if path.endswith("/"):
            os.makedirs(full_path, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write("")

if __name__ == "__main__":
    base_directory = "backend"
    if not os.path.exists(base_directory):
        print(f"Error: The directory '{base_directory}' does not exist. Please create it first.")
    else:
        structure = [
            "app/main.py",
            "app/api/__init__.py",
            "app/api/routes/__init__.py",
            "app/api/routes/chat.py",
            "app/core/__init__.py",
            "app/core/config.py",
            "app/core/security.py",
            "app/db/__init__.py",
            "app/db/vector_store.py",
            "app/models/__init__.py",
            "app/models/schemas.py",
            "app/services/__init__.py",
            "app/services/rag.py",
            "app/services/llm.py",
            "app/utils/__init__.py",
            "app/utils/text_processing.py",
            "data/embeddings/",
            "ingest.py",
            "requirements.txt",
            ".env",
        ]
        
        create_files_and_dirs(base_directory, structure)
        print("Project structure created successfully!")
