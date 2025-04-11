import os
import requests
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.schema.document import Document
from langchain.retrievers.document_compressors import EmbeddingsFilter, LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

from app.core.config import settings

from dotenv import load_dotenv
from textwrap import dedent
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from textwrap import dedent

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
    def __init__(self):
        # Initialize components
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
        return ChatGroq(
            groq_api_key=settings.GROQ_API_KEY_MODEL,
            model_name=settings.LLM_MODEL,
            temperature=0.6
        )
    
    def _initialize_memory(self):
        """Initialize conversation memory for chat history tracking."""
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    def evaluate_retrieval(self, query: str, document: str) -> RetrievalEvaluatorOutput:
        """
        Step 2 of CRAG: Evaluate the relevance and reliability of a document for the query.
        
        Args:
            query: User query
            document: Document content to evaluate
            
        Returns:
            RetrievalEvaluatorOutput with relevance and reliability scores
        """
        prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="""
            Evaluate the following document's relevance and reliability for answering the query.
            
            Query: {query}
            
            Document: {document}
            
            Provide:
            1. Relevance score (0-1): How directly the document addresses the query
            2. Reliability score (0-1): How accurate and up-to-date the information seems
            3. Brief reasoning for your evaluation
            
            Format your response as:
            Relevance: [score]
            Reliability: [score]
            Reasoning: [your reasoning]
            """
        )
        
        chain = prompt | self.llm.with_structured_output(RetrievalEvaluatorOutput)
        result = chain.invoke({"query": query, "document": document})
        
        return result
    
    def rewrite_query(self, query: str) -> QueryRewriterOutput:
        """
        Step 4 of CRAG (part 1): Rewrite the query to be more effective for search.
        
        Args:
            query: Original user query
            
        Returns:
            QueryRewriterOutput with rewritten query and search terms
        """
        prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            Rewrite the following query to make it more effective for document retrieval and web search:
            
            Query: {query}
            
            Provide:
            1. A rewritten version of the query that would yield better search results
            2. A list of 3-5 key search terms extracted from the query
            
            Format as:
            Rewritten query: [rewritten query]
            Search terms: [term1, term2, term3, ...]
            """
        )
        
        chain = prompt | self.llm.with_structured_output(QueryRewriterOutput)
        result = chain.invoke({"query": query})
        
        return result
    
    def decompose_knowledge(self, document: str, query: str) -> KnowledgeStripOutput:
        """
        Step 3 of CRAG (part 1): Decompose a document into knowledge strips.
        
        Args:
            document: Document content to decompose
            query: The query to evaluate strips against
            
        Returns:
            KnowledgeStripOutput with knowledge strips and their relevance scores
        """
        prompt = PromptTemplate(
            input_variables=["document", "query"],
            template="""
            Decompose the following document into individual "knowledge strips" (discrete facts or pieces of information).
            Then rate each strip's relevance to the query on a scale of 0-1.
            
            Document: {document}
            
            Query: {query}
            
            Format as:
            Strip 1: [fact/information]
            Score 1: [relevance score]
            
            Strip 2: [fact/information]
            Score 2: [relevance score]
            
            And so on...
            """
        )
        
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
                strips.append(strip_content)
                
                # Extract the score (everything after the colon)
                score_text = ":".join(lines[i+1].split(":", 1)[1:]).strip()
                try:
                    score = float(score_text)
                    strip_scores.append(min(max(score, 0.0), 1.0))  # Ensure score is between 0 and 1
                except ValueError:
                    strip_scores.append(0.5)  # Default to 0.5 if parsing fails
        
        return KnowledgeStripOutput(strips=strips, strip_scores=strip_scores)
    
    def filter_knowledge_strips(self, strips_output: KnowledgeStripOutput, threshold: float = 0.5) -> str:
        """
        Step 3 of CRAG (part 2): Filter knowledge strips based on relevance score.
        
        Args:
            strips_output: Output from decompose_knowledge
            threshold: Minimum score to keep a strip
            
        Returns:
            String of filtered high-quality knowledge
        """
        high_quality_strips = [
            strip for strip, score in zip(strips_output.strips, strips_output.strip_scores)
            if score >= threshold
        ]
        
        if not high_quality_strips:
            return "No relevant information found."
        
        return "\n".join([f"- {strip}" for strip in high_quality_strips])

    def perform_web_search(self, search_query: str, num_results: int = 5) -> str:
        """
        Step 4 of CRAG (part 2): Perform web search for external knowledge using DuckDuckGo.
        
        Args:
            search_query: The search query
            num_results: Number of search results to retrieve
            
        Returns:
            List of WebSearchResult objects
        """
        if not self.enable_web_search:
            print("Web search is disabled.")
            return []
            
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
            return ''
    
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
            
        base_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k_results}
        )
        
        # Create document compressors
        embeddings_filter = EmbeddingsFilter(
            embeddings=self.embeddings,
            similarity_threshold=0.75
        )
        
        extractor_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""Given the following question and context, extract only the parts of the context 
            that are directly relevant to answering the question.
            
            Question: {question}
            Context: {context}
            
            Relevant content:"""
        )
        
        llm_extractor = LLMChainExtractor.from_llm(
            self.llm,
            prompt=extractor_prompt
        )
        
        # Chain the compressors
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
            self.memory.save_context({"input": query}, {"answer": response})
            
            return response, sources, debug_info
        
        # If web search is not enabled, only use local documents
        else:
            
            # Step 2: Initial retrieval with contextual compression
            retriever = self.create_contextual_compression_retriever()
            retrieved_docs = retriever.invoke(rewritten_query)
            
            # If no documents were retrieved, handle empty case
            if not retrieved_docs:
                print("No documents retrieved from vector store.")

                self.latest_crag_action = "web_search_fallback"
                debug_info["crag_action"] = "web_search_fallback"
                
                # Perform web search
                external_knowledge = self.perform_web_search(rewritten_query)
                
                # Create a response with web search results
                response = self._generate_response(query, external_knowledge, 0.5, ["Web search"])
                return response, ["Web search"], debug_info
            
            # Step 3: Evaluate retrieved documents
            doc_evaluations = []
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
            
            # Get the best document and its score
            best_eval = doc_evaluations[0] if doc_evaluations else None
            best_score = best_eval["combined_score"] if best_eval else 0
            
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
                    filtered_content = self.filter_knowledge_strips(strips_output, threshold=0.6)
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
                    filtered_content = self.filter_knowledge_strips(strips_output, threshold=0.4)  # Lower threshold
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
                    filtered_content = self.filter_knowledge_strips(strips_output, threshold=0.5)
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
            self.memory.save_context({"input": query}, {"answer": response})
            
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
            ("system", """**Role**: You are My Personal Knowledge Architect, designed to analyze and synthesize information from my private document collection.

            **Core Capabilities**:
            1. Cross-Document Synthesis: Connect concepts across different files/books/papers
            2. Adaptive Explanation: Adjust depth based on content type:
            - Technical Papers: Math/formulas/citations
            - Books: Chapter summaries & thematic analysis
            - Scripts: Code structure & implementation logic
            - Notes: Contextual reconstruction
            3. Source Tracing: Always reference filename + page/line numbers
            4. Knowledge Gaps: Flag missing information between documents

            **Operational Rules**:
            1. Absolute Truthfulness: Never invent beyond provided documents
            2. Concept Mapping: Show relationships between ideas from different sources
            3. Layered Explanations:
            - deep dive with examples
            4. Special Handling For:
            ```[Code Blocks] → Preserve original formatting + explain logic
            [Diagrams/Images] → Describe visual elements verbatim
            [Handwritten Notes] → Decipher ambiguous text cautiously```

            **Confidence Framework**:
            - High (70-100%): Exact matches in multiple documents
            - Medium (30-69%): Partial matches requiring inference
            - Low (<30%): Tentative connections

            **Response Template**:
            "[Brief Answer]
            **Full Analysis**:  
            [Detailed explanation with quotes/excerpts]
            **Sources**: [File Names + Locations]  
            **Connections**: [Cross-document links]  
            **Gaps**: [Missing knowledge alerts]  
            "  

            **Critical Directives**:
            - If uncertain: "My documents suggest... [but are incomplete]"
            - For conflicting info: "Document A says X, while Document B states Y"
            - Preserve original document phrasing when crucial
            - Never assume access to external knowledge"""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                ("system", """**Active Documents**:
            {context}

    METADATA:
    - Confidence Level: {confidence_level}
    - Sources: {sources}

    Respond directly to the user's query using ONLY the information provided above. 
    If the information is incomplete or uncertain, acknowledge this fact.
    Always cite your sources at the end of your response in a "References" section.
    """),
        ])
        
        # Create document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Create a proper Document object
        documents = [Document(page_content=knowledge)]
        
        # Generate response
        response = document_chain.invoke({
            "input": query,
            "chat_history": self.memory.load_memory_variables({})["chat_history"],
            "context": documents,  # Pass a list of Document objects
            "confidence_level": confidence_level,
            "sources": ", ".join(sources) if sources else "No specific sources"
        })
        
        return response
    
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
        
        
if __name__ == '__main__':
    # Example usage of the RAGService
    rag_service = RAGService()
    
    # Ingest documents from a directory
    # num_chunks = rag_service.ingest_documents("path/to/pdf/directory")
    # print(f"Number of chunks added: {num_chunks}")
    
    # Process a query
    response, sources, debug_info = rag_service.process_query("Explain me DeepSeekPaper?")
    print(f"Response: {response}")
    print(f"Sources: {sources}")
    # print(f"Debug Info: {debug_info}")