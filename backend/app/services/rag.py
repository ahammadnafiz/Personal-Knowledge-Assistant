import os
import requests
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.schema.document import Document
from langchain.retrievers.document_compressors import EmbeddingsFilter, LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

from app.core.config import settings

from dotenv import load_dotenv
from textwrap import dedent

# Try to import agno tools - make web search optional if not available
try:
    from agno.agent import Agent
    from agno.models.google import Gemini
    from agno.tools.duckduckgo import DuckDuckGoTools
    from agno.tools.newspaper4k import Newspaper4kTools
    AGNO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: agno tools not available: {e}")
    print("Web search functionality will be disabled.")
    AGNO_AVAILABLE = False
    Agent = None
    Gemini = None
    DuckDuckGoTools = None
    Newspaper4kTools = None

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# Pydantic models for structured outputs
class RetrievalEvaluatorOutput(BaseModel):
    """Model for capturing the relevance score of a document to a query."""
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score between 0 and 1")
    reliability_score: float = Field(..., ge=0, le=1, description="Reliability score between 0 and 1")
    reasoning: str = Field(..., description="Reasoning behind the scores")

class QueryRewriterOutput(BaseModel):
    """Model for capturing a rewritten query suitable for web search."""
    query: str = Field(..., description="The query rewritten for better search results")
    search_terms: List[str] = Field(..., description="Key search terms extracted from the query")

class KnowledgeStripOutput(BaseModel):
    """Model for representing decomposed knowledge strips."""
    strips: List[str] = Field(..., description="Individual knowledge strips extracted from the document")
    strip_scores: List[float] = Field(..., description="Relevance scores for each knowledge strip")

class WebSearchResult(BaseModel):
    """Model for web search results."""
    title: str = Field(..., description="Title of the search result")
    snippet: str = Field(..., description="Snippet or summary of the search result")
    url: str = Field(..., description="URL of the search result")

class RAGService:
    """
    Self-Corrective RAG (CRAG) Service implementing the complete CRAG pipeline.
    """
    def __init__(self, google_api_key: str = None):
        # Initialize components
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.embeddings = self._initialize_embeddings()
        self.vector_store = self._initialize_or_load_vector_store()
        self.llm = self._initialize_llm()
        self.memory = self._initialize_memory()
        
        # CRAG thresholds
        self.high_confidence_threshold = 0.7  # Score above which documents are considered highly relevant
        self.low_confidence_threshold = 0.3   # Score below which documents are considered irrelevant
        
        # CRAG parameters
        self.top_k_results = settings.TOP_K_RESULTS
        self.enable_web_search = False  # Set to False if web search is not available
        
        # Track latest CRAG action for logging/debugging
        self.latest_crag_action = None
    
    def _initialize_embeddings(self):
        """Initialize the embedding model."""
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    
    def _initialize_or_load_vector_store(self):
        """Initialize or load the vector store with error handling."""
        try:
            if os.path.exists(settings.VECTOR_STORE_PATH):
                return FAISS.load_local(
                    settings.VECTOR_STORE_PATH, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                return None
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Creating a new vector store...")
            if os.path.exists(settings.VECTOR_STORE_PATH):
                import shutil
                shutil.rmtree(settings.VECTOR_STORE_PATH)
            return None
    
    def _initialize_llm(self):
        """Initialize the language model."""
        return ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=self.google_api_key,
            temperature=0.2,  # Slightly higher for more natural, less templated responses
            max_tokens=8192,  # Increased for longer, more detailed responses
            timeout=120,  # Longer timeout for detailed processing
            max_retries=3,
            request_timeout=60
        )
    
    def _initialize_memory(self):
        """Initialize conversation memory for chat history tracking."""
        return ConversationBufferWindowMemory(
            k=5, 
            memory_key="chat_history", 
            return_messages=True
        )
        
    def set_web_search_enabled(self, enabled: bool):
        """Set whether web search is enabled."""
        self.enable_web_search = enabled
    
    def evaluate_retrieval(self, query: str, document: str) -> RetrievalEvaluatorOutput:
        """
        Step 2 of CRAG: Evaluate the relevance and reliability of a document for the query.
        Enhanced with more robust scoring methodology.
        
        Args:
            query: User query
            document: Document content to evaluate
            
        Returns:
            RetrievalEvaluatorOutput with relevance and reliability scores
        """
        # Truncate document if too long to avoid token limits
        max_doc_length = 3000
        if len(document) > max_doc_length:
            document = document[:max_doc_length] + "..."
            
        prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="""
            You are an expert document evaluator. Analyze the document's relevance and reliability for answering the query.
            
            QUERY: {query}
            
            DOCUMENT CONTENT: {document}
            
            EVALUATION CRITERIA:
            
            RELEVANCE (0.0-1.0):
            - 1.0: Document directly answers the query with specific information
            - 0.8: Document contains most information needed to answer the query
            - 0.6: Document contains some relevant information but incomplete
            - 0.4: Document has limited relevance, only tangentially related
            - 0.2: Document mentions topics from query but doesn't help answer it
            - 0.0: Document is completely irrelevant to the query
            
            RELIABILITY (0.0-1.0):
            - 1.0: Information appears factual, well-sourced, and authoritative
            - 0.8: Information seems accurate with good supporting details
            - 0.6: Information appears reasonable but lacks depth
            - 0.4: Information has some questionable aspects or lacks context
            - 0.2: Information seems outdated or potentially inaccurate
            - 0.0: Information appears unreliable or contradictory
            
            ANALYSIS STEPS:
            1. Identify key concepts in the query
            2. Check if document addresses these concepts
            3. Assess the completeness of information
            4. Evaluate the quality and credibility of information
            5. Consider recency and accuracy indicators
            
            RESPONSE FORMAT:
            Relevance: [score between 0.0 and 1.0]
            Reliability: [score between 0.0 and 1.0]
            Reasoning: [2-3 sentences explaining your scoring with specific examples from the document]
            """
        )
        
        try:
            chain = prompt | self.llm.with_structured_output(RetrievalEvaluatorOutput)
            result = chain.invoke({"query": query, "document": document})
            
            # Validate scores are within bounds
            result.relevance_score = max(0.0, min(1.0, result.relevance_score))
            result.reliability_score = max(0.0, min(1.0, result.reliability_score))
            
            return result
        except Exception as e:
            print(f"Error in retrieval evaluation: {e}")
            # Return default moderate scores if evaluation fails
            return RetrievalEvaluatorOutput(
                relevance_score=0.5,
                reliability_score=0.5,
                reasoning=f"Evaluation failed due to error: {str(e)}. Using default scores."
            )
    
    def rewrite_query(self, query: str) -> QueryRewriterOutput:
        """
        Step 4 of CRAG (part 1): Rewrite the query to be more effective for search.
        Enhanced with better query understanding and expansion.
        
        Args:
            query: Original user query
            
        Returns:
            QueryRewriterOutput with rewritten query and search terms
        """
        prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            You are an expert query optimizer for document retrieval systems. Your task is to transform user queries into more effective search queries.
            
            ORIGINAL QUERY: {query}
            
            OPTIMIZATION GOALS:
            1. Expand abbreviations and acronyms
            2. Add relevant synonyms and related terms
            3. Convert questions into keyword-focused statements
            4. Include technical and domain-specific terminology
            5. Make implicit concepts explicit
            
            ANALYSIS STEPS:
            1. Identify the main topic/domain of the query
            2. Extract key concepts and entities
            3. Consider alternative phrasings and terminology
            4. Add context-specific keywords that might appear in relevant documents
            5. Structure for optimal semantic similarity matching
            
            OPTIMIZATION EXAMPLES:
            - "What is machine learning?" → "machine learning definition algorithms supervised unsupervised deep learning neural networks"
            - "How does photosynthesis work?" → "photosynthesis process chlorophyll light energy glucose carbon dioxide plants cellular respiration"
            - "Python error handling" → "Python exception handling try catch except finally error management debugging"
            
            RESPONSE FORMAT:
            Rewritten query: [Optimized query with expanded terms and concepts]
            Search terms: [List of 5-8 key terms that should appear in relevant documents]
            
            INSTRUCTIONS:
            - Keep the core intent of the original query
            - Add domain-specific terminology likely to appear in relevant documents  
            - Include both general and specific terms
            - Focus on information-rich keywords
            - Avoid overly complex or unnatural phrasing
            """
        )
        
        try:
            chain = prompt | self.llm.with_structured_output(QueryRewriterOutput)
            result = chain.invoke({"query": query})
            
            # Validate and clean up the results
            if not result.query:
                result.query = query  # Fallback to original
            if not result.search_terms:
                result.search_terms = query.split()  # Basic fallback
                
            return result
        except Exception as e:
            print(f"Query rewriting failed: {e}")
            # Fallback to basic query processing
            basic_terms = query.lower().replace('?', '').replace('.', '').split()
            return QueryRewriterOutput(
                query=query,
                search_terms=basic_terms[:5]
            )
    
    def decompose_knowledge(self, document: str, query: str) -> KnowledgeStripOutput:
        """
        Step 3 of CRAG (part 1): Decompose a document into knowledge strips.
        Enhanced to preserve more context and detail.
        
        Args:
            document: Document content to decompose
            query: The query to evaluate strips against
            
        Returns:
            KnowledgeStripOutput with knowledge strips and their relevance scores
        """
        # Truncate document if too long
        max_doc_length = 4000
        if len(document) > max_doc_length:
            document = document[:max_doc_length] + "..."
            
        prompt = PromptTemplate(
            input_variables=["document", "query"],
            template="""
            You are an expert information analyst. Extract comprehensive knowledge strips from the document that are relevant to the query.
            
            DOCUMENT CONTENT: {document}
            
            USER QUERY: {query}
            
            TASK: Decompose the document into detailed "knowledge strips" - coherent pieces of information that maintain context and detail.
            
            GUIDELINES FOR KNOWLEDGE STRIPS:
            1. Each strip should be 2-5 sentences that form a complete thought
            2. Preserve technical details, definitions, and context
            3. Include supporting information and explanations
            4. Maintain connections between related concepts
            5. Keep important qualifiers, examples, and specifics
            6. Ensure each strip can stand alone as meaningful information
            
            RELEVANCE SCORING (0.0 to 1.0):
            - 1.0: Directly answers the query with comprehensive detail
            - 0.9: Highly relevant with important supporting information
            - 0.8: Relevant with good context and detail
            - 0.7: Moderately relevant with useful information
            - 0.6: Somewhat relevant with background context
            - 0.5: Tangentially relevant but provides context
            - Below 0.5: Limited relevance
            
            FORMAT (Extract 6-10 knowledge strips):
            Strip 1: [Comprehensive information with context and details]
            Score 1: [relevance score]
            
            Strip 2: [Comprehensive information with context and details]
            Score 2: [relevance score]
            
            Continue for all relevant information...
            
            FOCUS: Prioritize comprehensive, detailed strips that provide deep understanding rather than brief mentions.
            """
        )
        
        try:
            # First get the raw output
            raw_output = self.llm.invoke(prompt.format(document=document, query=query))
            
            # Parse the raw output into strips and scores
            strips = []
            strip_scores = []
            
            lines = raw_output.content.split('\n')
            for i in range(0, len(lines) - 1, 2):
                if i + 1 < len(lines) and lines[i].startswith("Strip") and lines[i+1].startswith("Score"):
                    # Extract the strip content (everything after the colon)
                    strip_content = ":".join(lines[i].split(":", 1)[1:]).strip()
                    if strip_content and len(strip_content) > 20:  # Ensure substantial content
                        strips.append(strip_content)
                        
                        # Extract the score (everything after the colon)
                        score_text = ":".join(lines[i+1].split(":", 1)[1:]).strip()
                        try:
                            score = float(score_text)
                            strip_scores.append(min(max(score, 0.0), 1.0))  # Ensure score is between 0 and 1
                        except ValueError:
                            strip_scores.append(0.6)  # Default to moderate score if parsing fails
            
            # If no strips were extracted, create a fallback
            if not strips:
                strips = [document[:1000] + "..." if len(document) > 1000 else document]
                strip_scores = [0.7]
                
            return KnowledgeStripOutput(strips=strips, strip_scores=strip_scores)
            
        except Exception as e:
            print(f"Error in knowledge decomposition: {e}")
            # Fallback to using the document as-is
            return KnowledgeStripOutput(
                strips=[document[:1500] + "..." if len(document) > 1500 else document],
                strip_scores=[0.6]
            )
    
    def filter_knowledge_strips(self, strips_output: KnowledgeStripOutput, threshold: float = 0.4) -> str:
        """
        Step 3 of CRAG (part 2): Filter knowledge strips based on relevance score.
        Enhanced to preserve more detailed information.
        
        Args:
            strips_output: Output from decompose_knowledge
            threshold: Minimum score to keep a strip (lowered for more comprehensive responses)
            
        Returns:
            String of filtered high-quality knowledge with detailed formatting
        """
        high_quality_strips = [
            strip for strip, score in zip(strips_output.strips, strips_output.strip_scores)
            if score >= threshold
        ]
        
        if not high_quality_strips:
            # If no strips meet threshold, include the highest scoring ones
            if strips_output.strips:
                max_score_idx = strips_output.strip_scores.index(max(strips_output.strip_scores))
                high_quality_strips = [strips_output.strips[max_score_idx]]
            else:
                return "No relevant information found in the document."
        
        # Format the strips with better structure for comprehensive responses
        formatted_strips = []
        for i, strip in enumerate(high_quality_strips):
            # Add section markers for better organization in long responses
            if len(high_quality_strips) > 3:
                formatted_strips.append(f"**Section {i+1}**: {strip}")
            else:
                formatted_strips.append(f"• {strip}")
        
        return "\n\n".join(formatted_strips)

    def perform_web_search(self, search_query: str, num_results: int = 5) -> str:
        """
        Step 4 of CRAG (part 2): Perform web search for external knowledge using DuckDuckGo.
        
        Args:
            search_query: The search query
            num_results: Number of search results to retrieve
            
        Returns:
            String with search results or error message
        """
        
        if not AGNO_AVAILABLE:
            return "Web search is not available due to missing dependencies. Please install: pip install newspaper4k lxml_html_clean agno"
            
        try:
            research_agent = Agent(
             model=Gemini(
                id="gemini-2.0-flash",
                api_key=api_key,
            ),
            tools=[
                DuckDuckGoTools(),
                Newspaper4kTools(),
            ],
            description=dedent("""\
                    You are an elite research specialist with expertise in comprehensive information gathering and analysis. Your skills include:
                    🔍 Core Competencies:
                    - Thorough information collection and verification
                    - Critical evaluation of sources
                    - Extracting key insights from multiple resources
                    - Identifying reliable and authoritative content
                    - Synthesizing complex information
                    - Recognizing patterns and connections
                    - Providing balanced and objective overviews
                    - Contextualizing information relevance
                """),
                instructions=dedent("""\
                    1. Research Process:
                    - Search for multiple authoritative sources on the topic
                    - Prioritize recent and relevant information
                    - Collect information from diverse perspectives
                    
                    2. Analysis & Synthesis:
                    - Verify information across multiple sources
                    - Extract key facts and concepts
                    - Identify consensus views and disagreements
                    - Organize information logically
                    
                    3. Response Format:
                    - Provide a concise overview of findings
                    - Present information in clear, structured sections
                    - Include important facts, statistics and quotes
                    - Maintain objectivity throughout
                    
                    4. Quality Standards:
                    - Ensure accuracy of all information
                    - Focus on relevance to the query
                    - Present a balanced perspective
                    - Identify any significant information gaps
                """),
            markdown=True,
            show_tool_calls=False,  # Hide tool calls in the response
            add_datetime_to_instructions=True,
            )

            response = research_agent.run(search_query)
            return response
                    
        except Exception as e:
            print(f"Web search error: {e}")
            return f'Web search failed: {str(e)}'
    
    def ingest_documents(self, directory_path: str, specific_files: Optional[List[str]] = None) -> int:
        """
        Process PDF documents in the specified directory and add them to the vector store.
        
        Args:
            directory_path: Path to the directory containing PDF documents
            specific_files: If provided, only process these specific files
        
        Returns:
            Number of chunks added to the vector store
        """
        # Document chunking configuration
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        # Load documents
        if specific_files:
            chunks = []
            for filename in specific_files:
                file_path = os.path.join(directory_path, filename)
                if os.path.exists(file_path) and filename.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                    file_chunks = text_splitter.split_documents(documents)
                    chunks.extend(file_chunks)
        else:
            loader = DirectoryLoader(
                directory_path, 
                glob="**/*.pdf", 
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            documents = loader.load()
            chunks = text_splitter.split_documents(documents)
        
        # No documents found
        if not chunks:
            return 0
        
        # Create or update the vector store
        if self.vector_store is not None:
            try:
                self.vector_store.add_documents(chunks)
                self.vector_store.save_local(settings.VECTOR_STORE_PATH)
            except AssertionError as e:
                print(f"Dimension mismatch detected: {e}")
                print("Creating a new vector store with the current embeddings model...")
                
                os.makedirs(os.path.dirname(settings.VECTOR_STORE_PATH), exist_ok=True)
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
                self.vector_store.save_local(settings.VECTOR_STORE_PATH)
        else:
            os.makedirs(os.path.dirname(settings.VECTOR_STORE_PATH), exist_ok=True)
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            self.vector_store.save_local(settings.VECTOR_STORE_PATH)
        
        return len(chunks)
    
    def load_chat_history(self, chat_history: Optional[List[Dict[str, Any]]] = None):
        """Load chat history from a list of message dictionaries into memory."""
        if not chat_history:
            self.memory.clear()
            return
            
        self.memory.clear()
        
        for message in chat_history:
            if message.get("role") == "user":
                self.memory.chat_memory.add_user_message(message.get("content", ""))
            elif message.get("role") == "assistant":
                self.memory.chat_memory.add_ai_message(message.get("content", ""))
            elif message.get("role") == "system":
                self.memory.chat_memory.add_message(SystemMessage(content=message.get("content", "")))
    
    def create_contextual_compression_retriever(self):
        """Create a contextually compressed retriever for better retrieval quality."""
        if not self.vector_store:
            raise ValueError("Vector store is not initialized. Please ingest documents first.")
            
        # Use MMR for more diverse results
        base_retriever = self.vector_store.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": self.top_k_results * 2,  # Fetch more candidates
                "lambda_mult": 0.7,  # Balance between relevance and diversity
                "fetch_k": self.top_k_results * 3  # Initial fetch size
            }
        )
        
        # Create more selective document compressors
        embeddings_filter = EmbeddingsFilter(
            embeddings=self.embeddings,
            similarity_threshold=0.65,  # Lower threshold for more inclusive filtering
            k=self.top_k_results
        )
        
        extractor_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""You are an expert information extractor. Your task is to identify and extract ONLY the parts of the context that are directly relevant to answering the specific question.

            QUESTION: {question}
            
            CONTEXT: {context}
            
            INSTRUCTIONS:
            1. Read the question carefully to understand what information is being sought
            2. Scan the context for information that directly addresses the question
            3. Extract complete sentences or paragraphs that contain relevant information
            4. Include supporting details that provide context to the main answer
            5. If no relevant information exists, return "No relevant information found"
            6. Do not add any information not present in the context
            7. Preserve the original wording and technical terms
            
            RELEVANT EXTRACTED CONTENT:"""
        )
        
        llm_extractor = LLMChainExtractor.from_llm(
            self.llm,
            prompt=extractor_prompt
        )
        
        # Chain the compressors - order matters
        compression_pipeline = DocumentCompressorPipeline(
            transformers=[embeddings_filter, llm_extractor]
        )
        
        # Create contextual compression retriever
        return ContextualCompressionRetriever(
            base_compressor=compression_pipeline,
            base_retriever=base_retriever
        )
    
    def process_query(self, query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Process a query using the complete CRAG pipeline.
        
        Args:
            query: User query
            chat_history: Optional chat history
            
        Returns:
            Tuple of (response, sources, debug_info)
        """
        
        if not self.vector_store:
            return "Please ingest documents first.", [], {"error": "No documents ingested"}
        
        # Load chat history
        self.load_chat_history(chat_history)
        
        # Debug information dictionary
        debug_info = {
            "original_query": query,
            "crag_action": None,
            "evaluation_scores": [],
            "knowledge_sources": [],
            "confidence_level": None
        }
        
        # Step 1: Rewrite query for better retrieval
        query_rewrite_output = self.rewrite_query(query)
        rewritten_query = query_rewrite_output.query
        debug_info["rewritten_query"] = rewritten_query
        debug_info["search_terms"] = query_rewrite_output.search_terms
        
        print(f"Original query: {query}")
        print(f"Rewritten query: {rewritten_query}")
        
        if self.enable_web_search:
            # Step 2: Local document retrieval with contextual compression
            sources = []
            local_knowledge = ""
            best_score = 0.5  # Default medium confidence
            
            # Only attempt vector retrieval if we have a vector store
            if self.vector_store:
                retriever = self.create_contextual_compression_retriever()
                retrieved_docs = retriever.invoke(rewritten_query)
                
                # Evaluate retrieved documents if any were found
                doc_evaluations = []
                if retrieved_docs:
                    for doc in retrieved_docs:
                        eval_result = self.evaluate_retrieval(query, doc.page_content)
                        doc_evaluations.append({
                            "document": doc,
                            "relevance": eval_result.relevance_score,
                            "reliability": eval_result.reliability_score,
                            "combined_score": (eval_result.relevance_score + eval_result.reliability_score) / 2,
                            "reasoning": eval_result.reasoning
                        })
                    
                    # Sort documents by combined score (descending)
                    doc_evaluations.sort(key=lambda x: x["combined_score"], reverse=True)
                    
                    # Record evaluation scores for debugging
                    debug_info["evaluation_scores"] = [{
                        "relevance": eval["relevance"], 
                        "reliability": eval["reliability"],
                        "combined_score": eval["combined_score"],
                        "reasoning": eval["reasoning"]
                    } for eval in doc_evaluations]
                    
                    # Use the best score for confidence level
                    if doc_evaluations:
                        best_score = doc_evaluations[0]["combined_score"]
                    
                    # Process document knowledge
                    doc_knowledge = []
                    for eval in doc_evaluations:
                        doc = eval["document"]
                        strips_output = self.decompose_knowledge(doc.page_content, query)
                        filtered_content = self.filter_knowledge_strips(strips_output, threshold=0.5)
                        doc_knowledge.append(filtered_content)
                        
                        # Track source
                        if hasattr(doc, "metadata") and "source" in doc.metadata:
                            source = doc.metadata["source"]
                            if source not in sources:
                                sources.append(source)
                    
                    local_knowledge = "\n\n".join(doc_knowledge)
            
            # Step 3: Always perform web search
            web_knowledge = ""
            print("Performing web search...")
            web_knowledge  = self.perform_web_search(rewritten_query)
            
            # Add web search as a source
            if "Web search" not in sources:
                sources.append("Web search")
        
            # Step 4: Combine knowledge from both sources
            if local_knowledge and web_knowledge:
                final_knowledge = "Information from documents:\n\n" + local_knowledge + "\n\nUp-to-date information from web search:\n\n" + str(web_knowledge)
            elif local_knowledge:
                final_knowledge = local_knowledge
            elif web_knowledge:
                final_knowledge = str(web_knowledge)
            else:
                final_knowledge = "No relevant information found from either documents or web search."
            
            # Record knowledge sources
            debug_info["knowledge_sources"] = sources
            
            # Step 5: Generate final response
            response = self._generate_response(query, final_knowledge, best_score, sources)
            
            # Save to memory
            self.memory.chat_memory.add_user_message(query)
            self.memory.chat_memory.add_ai_message(response)
            
            return response, sources, debug_info
        
        # If web search is not enabled, only use local documents
        else:
            
            # Step 2: Enhanced retrieval with hybrid approach
            print("Using hybrid retrieval for better document matching...")
            retrieved_docs = self.hybrid_retrieval(query, rewritten_query)
            
            # If no documents were retrieved, handle empty case
            if not retrieved_docs:
                print("No documents retrieved from vector store using hybrid approach.")
                
                # Try fallback with basic similarity search
                try:
                    fallback_docs = self.vector_store.similarity_search(query, k=self.top_k_results)
                    if fallback_docs:
                        retrieved_docs = fallback_docs
                        print(f"Fallback retrieval found {len(fallback_docs)} documents")
                    else:
                        self.latest_crag_action = "web_search_fallback"
                        debug_info["crag_action"] = "web_search_fallback"
                        
                        # Perform web search
                        external_knowledge = self.perform_web_search(rewritten_query)
                        
                        # Create a response with web search results
                        response = self._generate_response(query, external_knowledge, 0.5, ["Web search"])
                        return response, ["Web search"], debug_info
                except Exception as e:
                    print(f"Even fallback retrieval failed: {e}")
                    return "Unable to retrieve any relevant documents. Please check if documents are properly ingested.", [], {"error": "Retrieval failure"}
            
            print(f"Retrieved {len(retrieved_docs)} documents for evaluation")
            
            # Step 3: Enhanced document evaluation with batch processing
            doc_evaluations = []
            for i, doc in enumerate(retrieved_docs):
                try:
                    print(f"Evaluating document {i+1}/{len(retrieved_docs)}...")
                    eval_result = self.evaluate_retrieval(query, doc.page_content)
                    doc_evaluations.append({
                        "document": doc,
                        "relevance": eval_result.relevance_score,
                        "reliability": eval_result.reliability_score,
                        "combined_score": (eval_result.relevance_score + eval_result.reliability_score) / 2,
                        "reasoning": eval_result.reasoning,
                        "retrieval_method": doc.metadata.get('retrieval_method', 'unknown')
                    })
                except Exception as e:
                    print(f"Error evaluating document {i+1}: {e}")
                    # Add with default scores if evaluation fails
                    doc_evaluations.append({
                        "document": doc,
                        "relevance": 0.4,
                        "reliability": 0.4,
                        "combined_score": 0.4,
                        "reasoning": f"Evaluation failed: {str(e)}",
                        "retrieval_method": doc.metadata.get('retrieval_method', 'unknown')
                    })
            
            # Sort documents by combined score (descending), with tie-breaking by retrieval method
            def sort_key(eval_dict):
                base_score = eval_dict["combined_score"]
                # Bonus for certain retrieval methods
                method_bonus = {
                    'similarity_original': 0.05,
                    'similarity_rewritten': 0.03,
                    'mmr': 0.02,
                    'keyword_expanded': 0.01
                }.get(eval_dict.get("retrieval_method", ""), 0)
                return base_score + method_bonus
            
            doc_evaluations.sort(key=sort_key, reverse=True)
            
            # Record evaluation scores for debugging
            debug_info["evaluation_scores"] = [{
                "relevance": eval["relevance"], 
                "reliability": eval["reliability"],
                "combined_score": eval["combined_score"],
                "reasoning": eval["reasoning"],
                "retrieval_method": eval.get("retrieval_method", "unknown")
            } for eval in doc_evaluations]
            
            # Get the best document and its score
            best_eval = doc_evaluations[0] if doc_evaluations else None
            best_score = best_eval["combined_score"] if best_eval else 0
            
            print(f"Best document score: {best_score:.3f} (method: {best_eval.get('retrieval_method', 'unknown') if best_eval else 'none'})")
            
            # Step 4: Determine CRAG action based on evaluation scores
            sources = []
            final_knowledge = ""
            
            if best_score >= self.high_confidence_threshold:
                # CORRECT action: Use knowledge refinement
                self.latest_crag_action = "correct"
                debug_info["crag_action"] = "correct"
                print(f"CRAG Action: CORRECT (Score: {best_score})")
                
                # Use the top documents
                top_docs = [eval["document"] for eval in doc_evaluations[:3]]
                
                # Apply knowledge decomposition and filtering to each document
                refined_knowledge = []
                for doc in top_docs:
                    strips_output = self.decompose_knowledge(doc.page_content, query)
                    filtered_content = self.filter_knowledge_strips(strips_output, threshold=0.4)  # Lower threshold for more content
                    refined_knowledge.append(filtered_content)
                    
                    # Track source
                    if hasattr(doc, "metadata") and "source" in doc.metadata:
                        source = doc.metadata["source"]
                        if source not in sources:
                            sources.append(source)
                
                # Combine refined knowledge
                final_knowledge = "\n\n".join(refined_knowledge)
                
            elif best_score <= self.low_confidence_threshold:
                # INCORRECT action: Use web search if enabled
                self.latest_crag_action = "incorrect"
                debug_info["crag_action"] = "incorrect"
                print(f"CRAG Action: INCORRECT (Score: {best_score})")
                
                # Perform web search
                search_results = self.perform_web_search(rewritten_query)
                
                # Still include whatever limited information we got from documents
                doc_knowledge = []
                for eval in doc_evaluations:
                    doc = eval["document"]
                    strips_output = self.decompose_knowledge(doc.page_content, query)
                    filtered_content = self.filter_knowledge_strips(strips_output, threshold=0.3)  # Lower threshold for ambiguous cases
                    doc_knowledge.append(filtered_content)
                    
                    # Track source
                    if hasattr(doc, "metadata") and "source" in doc.metadata:
                        source = doc.metadata["source"]
                        if source not in sources:
                            sources.append(source)
                
                # Add web search as a source
                sources.append("Web search")
                
                # Combine knowledge
                final_knowledge = "Information from documents:\n\n" + "\n\n".join(doc_knowledge) + "\n\nInformation from web search:\n\n" + str(search_results)
            
            else:
                # AMBIGUOUS action: Use hybrid approach
                self.latest_crag_action = "ambiguous"
                debug_info["crag_action"] = "ambiguous"
                print(f"CRAG Action: AMBIGUOUS (Score: {best_score})")
                
                # Process document knowledge
                doc_knowledge = []
                for eval in doc_evaluations:
                    doc = eval["document"]
                    strips_output = self.decompose_knowledge(doc.page_content, query)
                    filtered_content = self.filter_knowledge_strips(strips_output, threshold=0.3)  # Lower threshold for comprehensive coverage
                    doc_knowledge.append(filtered_content)
                    
                    # Track source
                    if hasattr(doc, "metadata") and "source" in doc.metadata:
                        source = doc.metadata["source"]
                        if source not in sources:
                            sources.append(source)
                
                # Optionally supplement with web search in ambiguous cases
                if best_score < 0.6:  # Only use web search if confidence is on the lower end
                    search_results = self.perform_web_search(rewritten_query)
                    
                    # Add web search as a source
                    sources.append("Web search")
                    
                    final_knowledge = "Information from documents:\n\n" + "\n\n".join(doc_knowledge) + "\n\nAdditional information:\n\n" + str(search_results)
                else:
                    final_knowledge = "\n\n".join(doc_knowledge)
            
            # Record knowledge sources
            debug_info["knowledge_sources"] = sources
            
            # Step 6: Generate final response
            response = self._generate_response(query, final_knowledge, best_score, sources)
            
            # Save to memory
            self.memory.chat_memory.add_user_message(query)
            self.memory.chat_memory.add_ai_message(response)
            
            return response, sources, debug_info
    
    def _generate_response(self, query: str, knowledge: str, confidence_score: float, sources: List[str]) -> str:
        """
        Generate a final response using the refined knowledge.
        
        Args:
            query: Original user query
            knowledge: Refined knowledge
            confidence_score: Confidence score (0-1)
            sources: List of sources
            
        Returns:
            Final response string
        """
        # Convert confidence score to text level
        if confidence_score >= self.high_confidence_threshold:
            confidence_level = f"High confidence ({confidence_score:.2f})"
        elif confidence_score <= self.low_confidence_threshold:
            confidence_level = f"Low confidence ({confidence_score:.2f})"
        else:
            confidence_level = f"Medium confidence ({confidence_score:.2f})"
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an Expert Technical Assistant specializing in providing comprehensive, practical answers based on document analysis.

**Your Role**: Respond directly and naturally to the user's specific question using the information from the provided documents. Do not use templates or forced structures - instead, tailor your response to what the user is actually asking for.

**Response Principles**:
1. **Answer the Actual Question**: Focus on what the user specifically wants to know
2. **Be Comprehensive but Natural**: Provide thorough information in a conversational, educational style
3. **Include Practical Details**: When asked about implementation, provide code examples, algorithms, steps, and practical guidance
4. **Use Document Content**: Base your response entirely on the provided documents
5. **Maintain Educational Depth**: Explain concepts thoroughly but organically based on the query

**For Implementation/Coding Questions**:
- Provide actual code examples and implementation details
- Break down algorithmic steps clearly
- Explain the mathematical foundations when relevant
- Include practical considerations and best practices
- Show how different components work together

**For Conceptual Questions**:
- Start with clear definitions
- Explain underlying principles and mechanisms
- Provide context and applications
- Compare with related approaches when relevant

**For Explanatory Questions**:
- Give intuitive explanations first
- Follow with technical details
- Use examples to illustrate concepts
- Connect to broader context when helpful

**Quality Standards**:
- Accuracy: Only use information from the provided documents
- Completeness: Address all aspects of the user's question
- Clarity: Use clear, accessible language while maintaining technical precision
- Practicality: Focus on actionable, useful information
- Depth: Provide sufficient detail for thorough understanding

**Critical Instructions**:
- NEVER use rigid templates or forced section headers unless the user specifically asks for structured output
- Respond naturally to what the user is asking
- If they want code, give them code with explanations
- If they want explanations, focus on clear conceptual understanding
- If they want implementation guidance, provide step-by-step practical advice
- Always acknowledge when information is incomplete or when documents don't contain specific details"""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "Query: {input}"),
                ("system", """**Document Content**:
{context}

**Information Sources**: {sources}
**Confidence Level**: {confidence_level}

**Instructions**: 
Respond directly to the user's query using the document content above. Tailor your response to what they're specifically asking for:

- If they want to know "how to code" something, focus on implementation details, code examples, and algorithmic steps
- If they want explanations, provide clear conceptual understanding with technical depth
- If they want to understand mechanisms, explain how things work internally
- Always base your response on the document content and cite sources appropriately

Make your response comprehensive and educational, but structure it naturally around their specific question rather than using predefined templates."""),
        ])
        
        # Create document chain with error handling
        try:
            document_chain = create_stuff_documents_chain(self.llm, prompt)
            
            # Create a proper Document object
            documents = [Document(page_content=knowledge)]
            
            # Generate response
            response = document_chain.invoke({
                "input": query,
                "chat_history": self.memory.chat_memory.messages,
                "context": documents,
                "confidence_level": confidence_level,
                "sources": ", ".join(sources) if sources else "No specific sources"
            })
            
            return response
            
        except Exception as e:
            print(f"Error in response generation: {e}")
            # Fallback to a simpler response
            fallback_response = f"""I encountered an error while processing your query: {str(e)}

Based on the available information from the documents, I can provide the following:

{knowledge[:1000]}{'...' if len(knowledge) > 1000 else ''}

**References**: {', '.join(sources) if sources else 'No specific sources'}

Please note: This response may be incomplete due to the processing error."""
            
            return fallback_response
    
    def _initialize_conversation_chain(self):
        """Initialize conversation chain for basic chat functionality."""
        stage1_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a helpful AI assistant. You can engage in conversation and answer questions based on context and previous conversation history."""),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ])
        
        return ConversationChain(
            llm=self.llm,
            prompt=stage1_prompt,
            verbose=True,
            memory=self.memory,
            input_key="human_input",
        )
    
    def get_conversation_chain(self):
        """Get the conversation chain for direct conversation."""
        return self._initialize_conversation_chain()
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get the current chat history as a list of message dictionaries."""
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
        """Clear the conversation memory."""
        self.memory.clear()
    
    def hybrid_retrieval(self, query: str, rewritten_query: str) -> List[Document]:
        """
        Perform hybrid retrieval using multiple strategies for better recall.
        
        Args:
            query: Original query
            rewritten_query: Optimized query
            
        Returns:
            List of documents ranked by relevance
        """
        if not self.vector_store:
            return []
            
        all_docs = []
        seen_content = set()
        
        try:
            # Strategy 1: Similarity search with original query
            similarity_docs = self.vector_store.similarity_search(
                query, 
                k=self.top_k_results
            )
            
            for doc in similarity_docs:
                content_hash = hash(doc.page_content[:200])  # Hash first 200 chars for deduplication
                if content_hash not in seen_content:
                    doc.metadata['retrieval_method'] = 'similarity_original'
                    doc.metadata['retrieval_score'] = 1.0
                    all_docs.append(doc)
                    seen_content.add(content_hash)
            
            # Strategy 2: Similarity search with rewritten query
            rewritten_docs = self.vector_store.similarity_search(
                rewritten_query, 
                k=self.top_k_results
            )
            
            for doc in rewritten_docs:
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_content:
                    doc.metadata['retrieval_method'] = 'similarity_rewritten'
                    doc.metadata['retrieval_score'] = 0.9
                    all_docs.append(doc)
                    seen_content.add(content_hash)
            
            # Strategy 3: MMR search for diversity
            try:
                mmr_docs = self.vector_store.max_marginal_relevance_search(
                    rewritten_query,
                    k=self.top_k_results,
                    lambda_mult=0.8  # High relevance, some diversity
                )
                
                for doc in mmr_docs:
                    content_hash = hash(doc.page_content[:200])
                    if content_hash not in seen_content:
                        doc.metadata['retrieval_method'] = 'mmr'
                        doc.metadata['retrieval_score'] = 0.8
                        all_docs.append(doc)
                        seen_content.add(content_hash)
            except Exception as e:
                print(f"MMR search failed: {e}")
            
            # Strategy 4: Keyword-based similarity with expanded terms
            query_terms = query.lower().split()
            expanded_query = " ".join(query_terms + rewritten_query.lower().split())
            
            keyword_docs = self.vector_store.similarity_search(
                expanded_query,
                k=self.top_k_results // 2
            )
            
            for doc in keyword_docs:
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_content:
                    doc.metadata['retrieval_method'] = 'keyword_expanded'
                    doc.metadata['retrieval_score'] = 0.7
                    all_docs.append(doc)
                    seen_content.add(content_hash)
            
            # Rank documents by a combination of retrieval score and content length (favor more informative docs)
            def rank_document(doc):
                base_score = doc.metadata.get('retrieval_score', 0.5)
                content_length_bonus = min(len(doc.page_content) / 1000, 0.2)  # Max 0.2 bonus
                return base_score + content_length_bonus
            
            all_docs.sort(key=rank_document, reverse=True)
            
            # Return top documents, limited by TOP_K_RESULTS
            return all_docs[:self.top_k_results]
            
        except Exception as e:
            print(f"Hybrid retrieval error: {e}")
            # Fallback to simple similarity search
            try:
                return self.vector_store.similarity_search(query, k=self.top_k_results)
            except Exception as fallback_e:
                print(f"Fallback retrieval also failed: {fallback_e}")
                return []