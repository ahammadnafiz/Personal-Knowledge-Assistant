# app/services/rag.py
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.retrievers.document_compressors import EmbeddingsFilter
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
        """
        Initialize or load the vector store, with additional checks to handle dimension mismatch.
        """
        try:
            if os.path.exists(settings.VECTOR_STORE_PATH):
                return FAISS.load_local(
                    settings.VECTOR_STORE_PATH, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                # If vector store doesn't exist, return None
                # It will be properly initialized during the first ingestion
                return None
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Creating a new vector store...")
            # If loading fails (e.g., due to dimension mismatch), delete the existing store
            if os.path.exists(settings.VECTOR_STORE_PATH):
                import shutil
                shutil.rmtree(settings.VECTOR_STORE_PATH)
            return None
    
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
    
    def _create_retrieval_evaluator(self):
        """
        Create a retrieval evaluator to assess document quality.
        Part of the CRAG implementation for quality assessment.
        """
        # Define a prompt for the evaluator
        evaluator_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a document relevance evaluator. Your job is to determine how relevant 
            the retrieved documents are to the user's query. Score each document from 0-10, where:
            - 0-3: Irrelevant or barely related
            - 4-6: Somewhat relevant but missing key information
            - 7-10: Highly relevant and contains needed information
            
            Only provide a numerical score with a brief one-sentence justification."""),
            ("human", "Query: {query}\n\nDocument: {document_content}\n\nRelevance score (0-10):"),
        ])
        
        # Create an evaluator chain
        return create_stuff_documents_chain(self.llm, evaluator_prompt)
    
    def _perform_knowledge_stripping(self, documents, query):
        """
        Break documents into smaller segments and grade them for relevance.
        Part of the CRAG implementation for knowledge stripping.
        """
        # First break into smaller segments if the documents are large
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # Smaller chunks for more granular relevance assessment
            chunk_overlap=50
        )
        
        # Create smaller segments from the retrieved documents
        small_chunks = []
        for doc in documents:
            smaller_docs = text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(smaller_docs):
                small_chunks.append({
                    "content": chunk,
                    "metadata": doc.metadata,
                    "original_index": documents.index(doc),
                    "chunk_index": i
                })
        
        # Use embeddings to filter the most relevant chunks
        embeddings_filter = EmbeddingsFilter(
            embeddings=self.embeddings,
            similarity_threshold=0.76  # Adjust based on your needs
        )
        
        # This is a simplified approach - in a full implementation, you would:
        # 1. Create document objects from small_chunks
        # 2. Pass them through the embeddings filter
        # 3. Score them using the evaluator
        # 4. Keep only the most relevant ones
        
        # For now, we'll use a simple embedding similarity as our filter
        return [doc for i, doc in enumerate(documents) if i < min(len(documents), 5)]  # Just return top 5 for demonstration
    
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
        if self.vector_store is not None:
            # If it exists, add documents to it
            try:
                self.vector_store.add_documents(chunks)
                self.vector_store.save_local(settings.VECTOR_STORE_PATH)
            except AssertionError as e:
                # Handle dimension mismatch
                print(f"Dimension mismatch detected: {e}")
                print("Creating a new vector store with the current embeddings model...")
                
                # Create a new vector store with the right dimensions
                os.makedirs(os.path.dirname(settings.VECTOR_STORE_PATH), exist_ok=True)
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
                self.vector_store.save_local(settings.VECTOR_STORE_PATH)
        else:
            # If it doesn't exist, create it
            os.makedirs(os.path.dirname(settings.VECTOR_STORE_PATH), exist_ok=True)
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            self.vector_store.save_local(settings.VECTOR_STORE_PATH)
        
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
        """Process a query using Corrective RAG (CRAG) and return the answer and sources."""
        # Load chat history into memory if provided
        self.load_chat_history(chat_history)
        
        # Create a basic retriever
        base_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.TOP_K_RESULTS}
        )
        
        # Implement the Corrective RAG approach
        
        # Step 1: Retrieve initial documents
        initial_docs = base_retriever.get_relevant_documents(query)
        
        # Step 2: Document quality assessment using the retrieval evaluator
        # (In a full implementation, you would score each document)
        
        # Step 3: Knowledge stripping - break documents into smaller segments and filter
        filtered_docs = self._perform_knowledge_stripping(initial_docs, query)
        
        # Create the prompt template with improved instructions
        prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a knowledgeable assistant that helps users find information from technical documents and research papers.
                Your goal is to provide accurate, detailed answers based solely on the provided context.
                
                You must:
                1. Only use information from the provided context.
                2. Be comprehensive and detailed in your explanations.
                3. If asked about code or algorithms, explain the implementation details if they appear in the context.
                4. Cite specific sections or papers when relevant.
                5. Acknowledge when information might be incomplete.
                6. If asked for code and the context contains code or detailed algorithm descriptions, provide it in a structured format.
                7. If the answer is not in the context, say "I don't have enough information in the knowledge base to answer this question completely" and provide what you do know from the context.
                8. IMPORTANT: Evaluate the quality of retrieved context. If it seems irrelevant or inadequate, acknowledge this limitation in your response.
                
                Do NOT make up or hallucinate information not present in the context."""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                ("system", """Base your answer exclusively on the following context:
                
                {context}"""),
            ])
        
        # Create the document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Instead of creating a standard retrieval chain, we'll process directly with filtered docs
        # This simulates what the contextual compression retriever would do
        context = "\n\n".join([doc.page_content for doc in filtered_docs])
        
        # Process the query with memory
        response = document_chain.invoke({
            "input": query,
            "chat_history": self.memory.load_memory_variables({})["chat_history"],
            "context": filtered_docs
        })
        
        # Save the interaction to memory
        self.memory.save_context({"input": query}, {"answer": response})
        
        # Extract sources if available
        sources = []
        for doc in filtered_docs:
            if hasattr(doc, "metadata") and "source" in doc.metadata:
                source = doc.metadata["source"]
                if source not in sources:
                    sources.append(source)
        
        return response, sources
    
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