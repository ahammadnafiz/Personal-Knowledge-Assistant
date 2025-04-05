import os
import numpy as np
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage, Document
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
    
    def _create_document_evaluator(self):
        """
        Create a document evaluator to assess document quality.
        Part of the CRAG implementation for quality assessment.
        """
        evaluator_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a document relevance evaluator. Your job is to determine how relevant 
            the retrieved document is to the user's query. Score the document from 0-10, where:
            - 0-3: Irrelevant or barely related
            - 4-6: Somewhat relevant but missing key information
            - 7-10: Highly relevant and contains needed information
            
            ONLY return a numerical score and nothing else."""),
            ("human", "Query: {query}\n\nDocument: {document_content}\n\nRelevance score (0-10):"),
        ])
        
        # Create an evaluator chain
        return create_stuff_documents_chain(self.llm, evaluator_prompt)
    
    def _evaluate_documents(self, documents, query):
        """
        Evaluate each document's relevance to the query.
        Returns documents with scores added to metadata.
        """
        scored_documents = []
        
        for doc in documents:
            try:
                # Use a simpler approach with direct LLM call
                prompt = f"""Query: {query}

                        Document: {doc.page_content}

                        On a scale of 0-10, how relevant is this document to the query? 
                        0-3: Irrelevant or barely related
                        4-6: Somewhat relevant but missing key information
                        7-10: Highly relevant and contains needed information

                        Relevance score (0-10):"""
                
                # Call LLM directly
                score_response = self.llm.invoke(prompt)
                
                # Extract the numerical score
                try:
                    score = float(score_response.content.strip())
                except ValueError:
                    # If we can't parse the response as a float, default to a middle score
                    score = 5.0
                
                # Create a new document with the score in metadata
                metadata = dict(doc.metadata)
                metadata["relevance_score"] = score
                
                new_doc = Document(
                    page_content=doc.page_content,
                    metadata=metadata
                )
                scored_documents.append(new_doc)
                
            except Exception as e:
                print(f"Error evaluating document: {e}")
                # Keep the document but with a neutral score
                metadata = dict(doc.metadata)
                metadata["relevance_score"] = 5.0
                
                new_doc = Document(
                    page_content=doc.page_content,
                    metadata=metadata
                )
                scored_documents.append(new_doc)
        
        return scored_documents
    
    def _hypothetical_answer_generation(self, query):
        """
        Generate a hypothetical answer to guide better retrieval.
        This is part of the CRAG implementation (Query Transformation).
        """
        # Use a direct prompt to avoid dictionary input errors
        prompt = f"""You are an expert research assistant. Given a user's question, 
        generate what you think would be a comprehensive and accurate answer based on your knowledge.
        This hypothetical answer will be used to guide document retrieval.
        Keep your answer concise but informative, focusing on key points that should be covered.

        User question: {query}

        Your hypothetical answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"Error generating hypothetical answer: {e}")
            return query  # Fallback to original query
    
    def _query_transformation(self, query, chat_history=None):
        """
        Transform the query using chat history and hypothetical answers.
        This is part of the CRAG implementation.
        """
        # Extract previous interactions if available
        recent_history = []
        if chat_history:
            # Get the last 3 exchanges (max)
            for i in range(min(6, len(chat_history))):
                message = chat_history[-(i+1)]
                if i < 6:  # Limit to 3 exchanges (6 messages)
                    recent_history.insert(0, f"{message['role']}: {message['content']}")
        
        history_text = "\n".join(recent_history) if recent_history else "No previous conversation."
        
        # Generate a hypothetical answer to guide retrieval
        hypothetical_answer = self._hypothetical_answer_generation(query)
        
        # Direct approach to avoid dictionary input errors
        prompt = f"""You are an expert at reformulating search queries to improve document retrieval.
        Your task is to enhance the original query by:
        1. Incorporating relevant context from the conversation history
        2. Adding key terms from the hypothetical answer
        3. Expanding abbreviations and technical terms
        4. Using synonyms for important concepts
        5. Breaking complex questions into key retrieval components
        
        Return ONLY the enhanced query text without explanation.

        Original query: {query}
        
        Recent conversation history:
        {history_text}
        
        Hypothetical answer points:
        {hypothetical_answer}
        
        Enhanced query:"""
        
        try:
            response = self.llm.invoke(prompt)
            enhanced_query = response.content.strip()
            return enhanced_query
        except Exception as e:
            print(f"Error transforming query: {e}")
            return query  # Fallback to original query
    
    def _knowledge_stripping(self, documents, query):
        """
        Break documents into smaller segments and evaluate their relevance.
        This is part of the CRAG implementation.
        """
        if not documents:
            return []
            
        # First break into smaller segments
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # Smaller chunks for more granular relevance assessment
            chunk_overlap=50
        )
        
        # Create smaller segments from the retrieved documents
        small_chunks = []
        
        for doc in documents:
            # Split the text
            splits = text_splitter.split_text(doc.page_content)
            
            # Create new document objects for each chunk
            for i, chunk in enumerate(splits):
                metadata = dict(doc.metadata)
                metadata["chunk_index"] = i
                metadata["original_doc_id"] = id(doc)
                
                chunk_doc = Document(
                    page_content=chunk,
                    metadata=metadata
                )
                small_chunks.append(chunk_doc)
        
        # Get embeddings for query and chunks
        query_embedding = self.embeddings.embed_query(query)
        
        # Calculate similarity and rank chunks
        ranked_chunks = []
        for chunk in small_chunks:
            chunk_embedding = self.embeddings.embed_query(chunk.page_content)
            similarity = self._calculate_similarity(query_embedding, chunk_embedding)
            
            # Add similarity score to metadata
            metadata = dict(chunk.metadata)
            metadata["similarity_score"] = similarity
            
            new_chunk = Document(
                page_content=chunk.page_content,
                metadata=metadata
            )
            ranked_chunks.append(new_chunk)
        
        # Sort by similarity score
        ranked_chunks.sort(key=lambda x: x.metadata["similarity_score"], reverse=True)
        
        # Get top chunks
        top_chunks = ranked_chunks[:min(len(ranked_chunks), settings.TOP_K_RESULTS * 2)]
        
        # Final step: Evaluate each top chunk with the LLM
        evaluated_chunks = self._evaluate_documents(top_chunks, query)
        
        # Sort by relevance score
        evaluated_chunks.sort(key=lambda x: x.metadata["relevance_score"], reverse=True)
        
        # Return top chunks by relevance score
        final_chunks = evaluated_chunks[:min(len(evaluated_chunks), settings.TOP_K_RESULTS)]
        
        return final_chunks
    
    def _calculate_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot_product / (norm1 * norm2)
    
    def _answer_refinement(self, initial_answer, query, documents):
        """
        Refine the initial answer for accuracy and completeness.
        This is part of the CRAG implementation (post-generation verification).
        """
        # Extract the most relevant excerpts from documents
        evidence = []
        for doc in documents:
            # Add source information if available
            source_info = f" (Source: {doc.metadata.get('source', 'Unknown')})"
            evidence.append(f"{doc.page_content}{source_info}")
        
        evidence_text = "\n\n".join(evidence)
        
        # Direct approach to avoid dictionary input errors
        prompt = f"""You are a fact-checking assistant that ensures answers are accurate and supported by evidence.
        Your task is to analyze an initial answer against the provided evidence and:
        1. Verify that all claims in the answer are supported by the evidence
        2. Remove or correct any unsupported claims
        3. Add any important information from the evidence that was missed
        4. Ensure proper attribution to sources
        5. Maintain a helpful, informative tone
        
        Produce a refined answer that is maximally accurate and helpful based on ONLY the provided evidence.

        User query: {query}
        
        Initial answer: {initial_answer}
        
        Evidence from documents:
        {evidence_text}
        
        Refined answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"Error refining answer: {e}")
            return initial_answer  # Fallback to initial answer
    
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
        """Process a query using full Corrective RAG (CRAG) implementation."""
        # Load chat history into memory if provided
        self.load_chat_history(chat_history)
        
        # CRAG Step 1: Query Transformation
        enhanced_query = self._query_transformation(query, chat_history)
        print(f"Enhanced query: {enhanced_query}")
        
        # Make sure the vector store is initialized
        if self.vector_store is None:
            return "I don't have any documents in my knowledge base yet. Please add some documents first.", []
        
        # Create a basic retriever
        base_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.TOP_K_RESULTS * 2}  # Get more documents initially
        )
        
        # CRAG Step 2: Initial Retrieval with enhanced query
        try:
            # Use invoke instead of get_relevant_documents
            retrieval_result = base_retriever.invoke(enhanced_query)
            initial_docs = retrieval_result if isinstance(retrieval_result, list) else [retrieval_result]
        except Exception as e:
            print(f"Error during retrieval: {e}")
            # Fall back to regular retrieval
            initial_docs = base_retriever.get_relevant_documents(enhanced_query)
        
        # CRAG Step 3: Document Quality Assessment
        if not initial_docs:
            return "No relevant documents found in the knowledge base for your query.", []
            
        scored_docs = self._evaluate_documents(initial_docs, enhanced_query)
        
        # Sort by relevance score and keep top K
        scored_docs.sort(key=lambda x: x.metadata.get("relevance_score", 0), reverse=True)
        filtered_docs = scored_docs[:settings.TOP_K_RESULTS]
        
        # CRAG Step 4: Knowledge Stripping for granular relevance
        if len(filtered_docs) > 0:
            stripped_docs = self._knowledge_stripping(filtered_docs, enhanced_query)
        else:
            stripped_docs = []
        
        # If no documents survived the filtering process
        if not stripped_docs:
            return "I couldn't find information relevant to your query in my knowledge base.", []
        
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
        
        # CRAG Step 5: Generate Initial Answer
        try:
            chat_history_messages = self.memory.load_memory_variables({}).get("chat_history", [])
            
            initial_response = document_chain.invoke({
                "input": query,
                "chat_history": chat_history_messages,
                "context": stripped_docs
            })
        except Exception as e:
            print(f"Error generating initial response: {e}")
            # Fallback to a simpler approach
            context_text = "\n\n".join([doc.page_content for doc in stripped_docs])
            system_prompt = f"""You are a helpful assistant answering a question based solely on the provided context.
            
            Context:
            {context_text}
            
            Answer the following question using only information from the context. If the information isn't in the context, say you don't have enough information."""
            
            human_prompt = query
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt}
            ]
            
            response = self.llm.invoke(messages)
            initial_response = response.content
        
        # CRAG Step 6: Answer Refinement (post-generation verification)
        final_answer = self._answer_refinement(initial_response, query, stripped_docs)
        
        # Save the interaction to memory
        self.memory.save_context({"input": query}, {"answer": final_answer})
        
        # Extract sources if available
        sources = []
        for doc in stripped_docs:
            if hasattr(doc, "metadata") and "source" in doc.metadata:
                source = doc.metadata["source"]
                if source not in sources:
                    sources.append(source)
        
        return final_answer, sources
    
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

# Example usage
if __name__ == "__main__":
    rag_service = RAGService()
    
    # Ingest documents from a directory
    num_chunks = rag_service.ingest_documents('/media/nafiz/NewVolume/Personal-Knowledge-Assistant/demo_data')
    print(f"Ingested {num_chunks} chunks.")
    
    # Process a query
    answer, sources = rag_service.process_query("What is the significance of Agent in AI?")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")
    
    # Get chat history
    chat_history = rag_service.get_chat_history()
    print(f"Chat History: {chat_history}")
    
    # Clear memory
    rag_service.clear_memory()