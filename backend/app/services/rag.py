# app/services/rag.py
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.schema.document import Document
from langchain.retrievers.document_compressors import EmbeddingsFilter, LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from pydantic import BaseModel, Field
from app.core.config import settings
from typing import List, Tuple, Dict, Any, Optional

class RetrievalEvaluatorInput(BaseModel):
    """
    Model for capturing the relevance score of a document to a query.
    """
    relevance_score: float = Field(..., description="Relevance score between 0 and 1, "
                                                   "indicating the document's relevance to the query.")

class QueryRewriterInput(BaseModel):
    """
    Model for capturing a rewritten query suitable for web search.
    """
    query: str = Field(..., description="The query rewritten for better search results.")

class KnowledgeRefinementInput(BaseModel):
    """
    Model for extracting key points from a document.
    """
    key_points: str = Field(..., description="Key information extracted from the document in bullet-point form.")

class RAGService:
    def __init__(self):
        self.embeddings = self._initialize_embeddings()
        self.vector_store = self._initialize_or_load_vector_store()
        self.llm = self._initialize_llm()
        self.memory = self._initialize_memory()
        
        # Thresholds for CRAG implementation
        self.lower_threshold = 0.3
        self.upper_threshold = 0.7
    
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
    
    def retrieval_evaluator(self, query, document):
        """
        Evaluate the relevance of a document to a query.
        Returns a relevance score between 0 and 1.
        """
        prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="On a scale from 0 to 1, how relevant is the following document to the query? "
                     "Query: {query}\nDocument: {document}\nRelevance score:"
        )
        chain = prompt | self.llm.with_structured_output(RetrievalEvaluatorInput)
        input_variables = {"query": query, "document": document}
        result = chain.invoke(input_variables).relevance_score
        return result
    
    def knowledge_refinement(self, document):
        """Extract key points from a document in bullet-point form."""
        prompt = PromptTemplate(
            input_variables=["document"],
            template="Extract the key information from the following document in bullet points:"
                     "\n{document}\nKey points:"
        )
        chain = prompt | self.llm.with_structured_output(KnowledgeRefinementInput)
        input_variables = {"document": document}
        result = chain.invoke(input_variables).key_points
        return [point.strip() for point in result.split('\n') if point.strip()]
    
    def rewrite_query(self, query):
        """Rewrite a query to make it more suitable for retrieval."""
        prompt = PromptTemplate(
            input_variables=["query"],
            template="Rewrite the following query to make it more suitable for document retrieval:"
                     "\n{query}\nRewritten query:"
        )
        chain = prompt | self.llm.with_structured_output(QueryRewriterInput)
        input_variables = {"query": query}
        return chain.invoke(input_variables).query.strip()
    
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
    
    def create_contextually_compressed_retriever(self, query):
        """
        Create a contextually compressed retriever that combines CRAG evaluation
        with contextual compression techniques.
        """
        # First, create base retriever
        base_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.TOP_K_RESULTS}
        )
        
        # Create document compressors
        
        # 1. Embeddings filter to remove irrelevant portions
        embeddings_filter = EmbeddingsFilter(
            embeddings=self.embeddings,
            similarity_threshold=0.75  # Adjust threshold as needed
        )
        
        # 2. LLM extractor to pull out only the most relevant information
        
        # Modified to accept 'question' and 'context' instead of 'query' and 'document'
        extractor_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""Given the following question and context, extract only the parts of the context 
            that are directly relevant to answering the question. Focus on maintaining key information while removing 
            irrelevant content.
            
            Question: {question}
            Context: {context}
            
            Relevant content:"""
        )
        
        llm_extractor = LLMChainExtractor.from_llm(
            self.llm,
            prompt=extractor_prompt
        )
        
        # Chain the compressors
        from langchain.retrievers.document_compressors import DocumentCompressorPipeline
        compression_pipeline = DocumentCompressorPipeline(
            transformers=[embeddings_filter, llm_extractor]
        )
        
        # Create contextual compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compression_pipeline,
            base_retriever=base_retriever
        )
        
        return compression_retriever
    
    def process_query(self, query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[str]]:
        """Process a query using Combined CRAG and Contextual Compression RAG."""
        # Load chat history into memory if provided
        self.load_chat_history(chat_history)
        
        # Step 1: Query rewriting (from CRAG) to improve retrieval
        rewritten_query = self.rewrite_query(query)
        print(f"Original query: {query}")
        print(f"Rewritten query: {rewritten_query}")
        
        # Step 2: Retrieve documents using contextual compression
        compression_retriever = self.create_contextually_compressed_retriever(rewritten_query)
        retrieved_docs = compression_retriever.invoke(rewritten_query)  # Updated to use invoke
        
        # Step 3: Evaluate document relevance (from CRAG)
        eval_scores = [self.retrieval_evaluator(query, doc.page_content) for doc in retrieved_docs]
        print(f"Evaluation scores: {eval_scores}")
        
        # Step 4: Implement CRAG's adaptive retrieval strategy based on confidence
        max_score = max(eval_scores) if eval_scores else 0
        sources = []
        final_knowledge_text = ""  # Initialize this variable
        
        if max_score > self.upper_threshold:
            # High confidence - use best document directly
            print("Action: High Confidence - Using top retrieved document")
            best_doc_index = eval_scores.index(max_score)
            best_doc = retrieved_docs[best_doc_index]
            
            # Extract key knowledge using knowledge refinement
            key_points = self.knowledge_refinement(best_doc.page_content)
            final_knowledge_text = "\n".join([f"- {point}" for point in key_points])
            
            # Track source
            if hasattr(best_doc, "metadata") and "source" in best_doc.metadata:
                sources.append(best_doc.metadata["source"])
        
        elif max_score < self.lower_threshold:
            # Low confidence - reformulate approach
            print("Action: Low Confidence - Using broader context approach")
            
            # For documents with low relevance, we'll use knowledge refinement 
            # to extract anything potentially useful
            all_key_points = []
            for doc in retrieved_docs:
                key_points = self.knowledge_refinement(doc.page_content)
                all_key_points.extend(key_points)
                
                # Track source
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    source = doc.metadata["source"]
                    if source not in sources:
                        sources.append(source)
            
            final_knowledge_text = "\n".join([f"- {point}" for point in all_key_points])
        
        else:
            # Medium confidence - use a combination approach
            print("Action: Medium Confidence - Using selective knowledge approach")
            
            # Sort documents by relevance score
            sorted_docs = [doc for _, doc in sorted(
                zip(eval_scores, retrieved_docs), 
                key=lambda x: x[0], 
                reverse=True
            )]
            
            # Process top documents with knowledge refinement
            all_key_points = []
            for doc in sorted_docs[:3]:  # Focus on top 3 documents
                key_points = self.knowledge_refinement(doc.page_content)
                all_key_points.extend(key_points)
                
                # Track source
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    source = doc.metadata["source"]
                    if source not in sources:
                        sources.append(source)
            
            final_knowledge_text = "\n".join([f"- {point}" for point in all_key_points])
        
        # Convert the text to a Document object
        final_knowledge = [Document(page_content=final_knowledge_text)]
        
        # Create the final prompt template with improved instructions
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
            8. IMPORTANT: The system has assessed the confidence of the retrieved information as follows:
            - High confidence (>0.7): Information is highly relevant
            - Medium confidence (0.3-0.7): Information is partially relevant
            - Low confidence (<0.3): Information may not be directly relevant
            
            Do NOT make up or hallucinate information not present in the context."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("system", """Base your answer exclusively on the following context:
            
            {context}
            
            Confidence level: {confidence_level}"""),
        ])
        
        # Determine confidence level message
        if max_score > self.upper_threshold:
            confidence_level = "High confidence - the information provided is highly relevant to the query"
        elif max_score < self.lower_threshold:
            confidence_level = "Low confidence - the information may not directly address the query"
        else:
            confidence_level = "Medium confidence - the information is partially relevant to the query"
        
        # Create the document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Process the query with memory and our processed knowledge
        response = document_chain.invoke({
            "input": query,
            "chat_history": self.memory.load_memory_variables({})["chat_history"],
            "context": final_knowledge,
            "confidence_level": confidence_level
        })
        
        # Save the interaction to memory
        self.memory.save_context({"input": query}, {"answer": response})
        
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