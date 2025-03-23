# app/services/rag.py updates
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from app.core.config import settings
from typing import List, Tuple, Dict, Any, Optional

class RAGService:
    def __init__(self):
        self.embeddings = self._initialize_embeddings()
        self.vector_store = self._initialize_or_load_vector_store()
        self.llm = self._initialize_llm()
        self.memory = self._initialize_memory()
    
    def _initialize_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    
    def _initialize_or_load_vector_store(self):
        if os.path.exists(settings.VECTOR_STORE_PATH):
            return FAISS.load_local(
                settings.VECTOR_STORE_PATH, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            # If vector store doesn't exist, initialize with empty data
            # It will be populated later with the ingest_documents method
            return FAISS.from_texts(["placeholder"], self.embeddings)
    
    def _initialize_llm(self):
        return ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.LLM_MODEL,
            temperature=0.0
        )
    
    def _initialize_memory(self):
        """Initialize conversation memory for chat history tracking"""
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    def ingest_documents(self, directory_path: str, specific_files: Optional[List[str]] = None):
        """
        Process PDF documents in the specified directory and add them to the vector store.
        
        Args:
            directory_path: Path to the directory containing PDF documents
            specific_files: If provided, only process these specific files
        
        Returns:
            Number of chunks added to the vector store
        """
        if specific_files:
            # Load specific files
            chunks = []
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            
            for filename in specific_files:
                file_path = os.path.join(directory_path, filename)
                if os.path.exists(file_path) and filename.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                    file_chunks = text_splitter.split_documents(documents)
                    chunks.extend(file_chunks)
        else:
            # Load all documents in the directory
            loader = DirectoryLoader(
                directory_path, 
                glob="**/*.pdf", 
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            chunks = text_splitter.split_documents(documents)
        
        # No documents found
        if not chunks:
            return 0
        
        # Create or update the vector store
        if os.path.exists(settings.VECTOR_STORE_PATH):
            # If it exists, add documents to it
            vector_store = FAISS.load_local(settings.VECTOR_STORE_PATH, self.embeddings, allow_dangerous_deserialization = True)
            vector_store.add_documents(chunks)
            vector_store.save_local(settings.VECTOR_STORE_PATH)
            self.vector_store = vector_store
        else:
            # If it doesn't exist, create it
            os.makedirs(os.path.dirname(settings.VECTOR_STORE_PATH), exist_ok=True)
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            vector_store.save_local(settings.VECTOR_STORE_PATH)
            self.vector_store = vector_store
        
        return len(chunks)
    
    def load_chat_history(self, chat_history: Optional[List[Dict[str, Any]]] = None):
        """Load chat history from a list of message dictionaries into memory"""
        if not chat_history:
            # Reset memory if no history is provided
            self.memory.clear()
            return
            
        # Clear existing memory
        self.memory.clear()
        
        # Add messages to memory
        for message in chat_history:
            if message.get("role") == "user":
                self.memory.chat_memory.add_user_message(message.get("content", ""))
            elif message.get("role") == "assistant":
                self.memory.chat_memory.add_ai_message(message.get("content", ""))
            elif message.get("role") == "system":
                self.memory.chat_memory.add_message(SystemMessage(content=message.get("content", "")))
    
    def process_query(self, query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[str]]:
        """Process a query using RAG and return the answer and sources."""
        # Load chat history into memory if provided
        self.load_chat_history(chat_history)
        
        # Create a retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.TOP_K_RESULTS}
        )
        
        # Create the prompt template with chat history context
        prompt = ChatPromptTemplate.from_template("""
        You are a knowledgeable assistant that helps users find information from books.
        
        Previous conversation:
        {chat_history}
        
        Answer the following question based on the provided context. If the answer is not in the context, 
        say "I don't have enough information to answer this question" and don't try to make up an answer.
        
        Context:
        {context}
        
        Question:
        {input}
        
        Answer:
        """)
        
        # Create the document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Create the retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Process the query with memory
        response = retrieval_chain.invoke({
            "input": query,
            "chat_history": self.memory.load_memory_variables({})["chat_history"]
        })
        
        # Save the interaction to memory
        self.memory.save_context({"input": query}, {"answer": response["answer"]})
        
        # Extract sources if available (document metadata often contains the source)
        sources = []
        if "context" in response and response["context"]:
            for doc in response["context"]:
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    source = doc.metadata["source"]
                    if source not in sources:
                        sources.append(source)
        
        return response["answer"], sources
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get the current chat history as a list of message dictionaries"""
        history = []
        for message in self.memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage):
                history.append({"role": "system", "content": message.content})
        return history
    
    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear()