
<h1 align="center">Personal Knowledge Assistant</h1>


Welcome to the **Personal Knowledge Assistant** project! This guide will walk you through setting up both the frontend and backend components of the system, a RAG-based platform for querying books and personal knowledge.

---

## Prerequisites

Before starting, ensure you have the following tools installed:

- **Python 3.9+** for the backend
- **Node.js 18+** for the frontend
- **Git** (optional)
- **PDF books** you want to include in your knowledge base

---

## Backend Setup

### 1. Clone the repository (or set up a new project)

```bash
mkdir personal-knowledge-assistant
cd personal-knowledge-assistant
mkdir backend frontend
```

### 2. Set up the backend folder structure

Organize your project directory as follows:

```
backend/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   └── chat.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── security.py
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   └── vector_store.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── rag.py
│   │   └── llm.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── text_processing.py
│
├── data/
│   └── embeddings/
│
├── ingest.py
├── requirements.txt
└── .env
```

### 3. Create a virtual environment and install dependencies

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Create requirements.txt with the following content
```

Add the dependencies to your `requirements.txt`:

```
fastapi
uvicorn
pydantic
pydantic-settings
langchain
langchain-groq
langchain-community
langchain-huggingface
faiss-cpu
python-dotenv
pypdf
sentence-transformers
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the backend directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Create empty `__init__.py` files

```bash
touch app/__init__.py
touch app/api/__init__.py
touch app/api/routes/__init__.py
touch app/core/__init__.py
touch app/db/__init__.py
touch app/models/__init__.py
touch app/services/__init__.py
touch app/utils/__init__.py
```

### 6. Ingest your books

Place your PDF books in a directory and ingest them:

```bash
mkdir books
# Copy your PDF books into the books directory

python ingest.py --dir books
```

### 7. Run the backend server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Frontend Setup

### 1. Set up the Next.js project

```bash
cd ../frontend
npx create-next-app@latest .
# Select Yes for TypeScript
# Select Yes for ESLint
# Select Yes for Tailwind CSS
# Select Yes for src/ directory
# Select Yes for App Router
# Select Yes for import alias
```

### 2. Install additional dependencies

```bash
npm install lucide-react react-markdown
```

### 3. Install shadcn/ui components

```bash
npx shadcn-ui@latest init
# Select Default for style
# Select Default for baseColor
# Select Yes for CSS variables
# Use App dir structure
# Select src/components for components directory
# Select @/components for import alias
# Select Yes for React Server Components
# Select Yes for tailwind.config.ts
# Select @/lib/utils for utils

# Install the required components
npx shadcn-ui@latest add button textarea card
```

### 4. Set up environment variables

Create a `.env.local` file in the frontend directory:

```
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

### 5. Update the code

Replace the contents of the following files with the provided code:

- `src/app/page.tsx`
- `src/app/layout.tsx`
- `src/app/globals.css`
- `tailwind.config.ts`

### 6. Run the frontend

```bash
npm run dev
```

Your application should now be running at `http://localhost:3000`.

---

## Using the Application

1. Navigate to `http://localhost:3000` in your web browser.
2. Ask questions about the books you've ingested.
3. The application will search through the book content and provide relevant answers.

---

## Troubleshooting

### Vector Store Issues

If you encounter issues with the vector store:

```bash
rm -rf data/vector_store
python ingest.py --dir books
```

### API Connection Issues

If the frontend can't connect to the backend:

1. Ensure the backend is running on port `8000`.
2. Check that CORS is properly configured.
3. Verify your `.env.local` file has the correct API URL.

### Model API Key Issues

If you encounter authentication errors:

1. Double-check your Groq API key in the `.env` file.
2. Ensure your HuggingFace token has the necessary permissions.

---

## Customization

### Changing the LLM Model

To change the LLM model, edit `app/core/config.py`:

```python
LLM_MODEL: str = "your-preferred-model"  # e.g., "llama3-8b-8192" for a smaller model
```

### Adjusting RAG Parameters

Edit `app/core/config.py` to customize the RAG behavior:

```python
CHUNK_SIZE: int = 1000  # Increase for larger contexts
CHUNK_OVERLAP: int = 200  # Adjust to reduce information loss at chunk boundaries
TOP_K_RESULTS: int = 5  # Increase for more comprehensive context
```

### Changing the Embedding Model

Edit `app/core/config.py` to use a different embedding model:

```python
EMBEDDING_MODEL: str = "your-preferred-embedding-model"  # e.g., "sentence-transformers/all-mpnet-base-v2"
```

---

<p align="center">
  <img src="https://img.shields.io/badge/Ahammad%20Nafiz-Blue.svg" alt="Powered by ahammadnafiz"/>
</p>
