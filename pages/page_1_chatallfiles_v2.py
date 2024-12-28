import os
import sys
import json
import logging
import asyncio
import aiohttp
import re
from typing import Dict, List, Optional, Any, Callable, AsyncIterator, Union
from enum import Enum
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

import streamlit as st
import cohere
from pydantic import BaseModel, Field
from llama_index.core import Document, Settings
from llama_index.retrievers.bm25 import BM25Retriever
from qdrant_client import QdrantClient, models
from openai import OpenAI as OG_OpenAI
from pydantic_ai import Agent, RunContext
import nest_asyncio
nest_asyncio.apply()
from llama_index.core.prompts import ChatPromptTemplate, ChatMessage

import os
import sys
import logging
import asyncio 
import streamlit as st
from typing import Dict, List, Optional, Any, Union
import cohere
from openai import OpenAI as OG_OpenAI
from llama_parse import LlamaParse
from qdrant_client import QdrantClient, AsyncQdrantClient, models
from llama_index.core import Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import VectorStoreIndex, StorageContext
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from pydantic_ai import Agent, RunContext, ModelRetry
from datetime import datetime
import tempfile
from tavily import TavilyClient, AsyncTavilyClient
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# ===== Section: Global Configurations and Constants =====
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure LlamaIndex Settings
Settings.llm = OpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_API_KEY"])
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=st.secrets["OPENAI_API_KEY"]
)
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
Settings.num_output = 512
Settings.context_window = 3900

tavily_client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Constants
class FileType(str, Enum):
    """Supported file types"""
    PDF = "pdf"
    IMAGE = "image"
    EXCEL = "excel"
    CSV = "csv"
    HTML = "html"
    TXT = "txt"
    OTHER = "other"

    @classmethod
    def from_extension(cls, ext: str) -> 'FileType':
        """Get FileType from file extension"""
        ext = ext.lower().lstrip('.')
        if ext in ['pdf']:
            return cls.PDF
        elif ext in ['jpg', 'jpeg', 'png', 'gif']:
            return cls.IMAGE
        elif ext in ['xlsx', 'xls']:
            return cls.EXCEL
        elif ext in ['csv']:
            return cls.CSV
        elif ext in ['html', 'htm']:
            return cls.HTML
        elif ext in ['txt']:
            return cls.TXT
        else:
            return cls.OTHER

SUPPORTED_EXTENSIONS = {
    FileType.PDF: ['.pdf'],
    FileType.IMAGE: ['.png', '.jpg', '.jpeg'],
    FileType.EXCEL: ['.xlsx'],
    FileType.CSV: ['.csv', '.tsv'],
    FileType.HTML: ['.html'],
    FileType.TXT: ['.txt'],
    FileType.OTHER: ['.docx', '.doc', '.odt', '.pptx', '.ppt', '.eml', '.msg', '.rtf', '.epub', '.xml']
}

# Initialize global clients
qdrant_client = QdrantClient(
    url=st.secrets["qdrant_url"],
    api_key=st.secrets["qdrant_api_key"]
)
async_qdrant_client = AsyncQdrantClient(
    url=st.secrets["qdrant_url"],
    api_key=st.secrets["qdrant_api_key"]
)
openai_client = OG_OpenAI()
cohere_client = cohere.Client(st.secrets["COHERE_API_KEY"]) if "COHERE_API_KEY" in st.secrets else None
llama_parser = None  # Initialized on demand


# ===== Section: Data Models for Document Chunks =====

class DocumentChunk(BaseModel):
    content: str
    source: str
    chunk_type: str = Field(description="Type of chunk (e.g., 'experience', 'education', 'skills')")
    relevance_score: float = Field(description="Relevance score for this chunk")

class ChunkSearchResult(BaseModel):
    chunks: List[DocumentChunk]
    source_files: List[str]
    total_chunks: int

class ResponseSection(BaseModel):
    title: str
    content: str
    examples: List[str] = Field(description="Specific examples from the documents")
    suggested_changes: List[str] = Field(description="Suggested improvements or changes")
    confidence: float = Field(description="Confidence score for this section", ge=0.0, le=1.0)

class StructuredResponse(BaseModel):
    question_summary: str
    key_findings: List[str]
    sections: List[ResponseSection]
    sources_used: List[str]
    overall_confidence: float

class SectionOutput(BaseModel):
    """The metadata for a given section"""
    section_name: str = Field(..., description="Section number (e.g. '3.2')")
    section_title: str = Field(..., description="Section title (e.g. 'Experimental Results')")
    start_page_number: int = Field(..., description="Start page number")
    is_subsection: bool = Field(..., description="True if subsection (e.g. '3.2')")
    description: Optional[str] = Field(None, description="Source text indicating section")

    def get_section_id(self) -> str:
        """Get section identifier"""
        return f"{self.section_name}: {self.section_title}"

class SectionsOutput(BaseModel):
    """List of all sections"""
    sections: List[SectionOutput]

class ValidSections(BaseModel):
    """List of valid section indexes"""
    valid_indexes: List[int] = Field(..., description="List of valid section indexes")

class DocumentMetadata(BaseModel):
    """Metadata for a document"""
    doc_id: str = Field(..., description="Unique identifier for the document")
    title: str = Field(..., description="Document title")
    file_path: str = Field(..., description="Path to the document file")
    file_type: FileType = Field(..., description="Type of document")

class SearchQuery(BaseModel):
    """Search query parameters"""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=5, description="Number of results to return")
    filter_types: Optional[List[FileType]] = Field(default=None, description="Filter by file types")

class SearchResult(BaseModel):
    """Search result structure"""
    content: str = Field(..., description="Retrieved content")
    score: float = Field(..., description="Relevance score")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    section_id: Optional[str] = Field(None, description="Section identifier if applicable")


# ===== Section: Web Search Agent =====

# Web Search Agent Models and Types
class WebSearchResultItem(BaseModel):
    """Individual web search result with metadata"""
    result_title: str = Field(description="Title of the search result")
    result_content: str = Field(description="Main content or summary of the result")
    result_url: str = Field(description="URL of the source")
    result_type: str = Field(description="Type of the source (e.g., Website, News, Academic)")
    result_score: float = Field(ge=0.0, le=1.0, description="Relevance score of the result (0.0 to 1.0)")
    result_date: Optional[str] = Field(None, description="Publication or last updated date of the result")

class WebSearchResponse(BaseModel):
    """Complete web search response including analysis"""
    search_summary: str = Field(min_length=50, description="AI-generated summary of all search results")
    search_findings: List[str] = Field(min_items=1, description="List of key findings from the search results")
    search_results: List[WebSearchResultItem] = Field(min_items=1, description="List of relevant search results")
    follow_up_queries: List[str] = Field(min_items=1, description="Suggested follow-up queries for more information")
    search_timestamp: str = Field(description="Timestamp when the search was performed")

class WebSearchParameters(BaseModel):
    """Input parameters for web search"""
    search_query: str = Field(min_length=3, description="The search query")
    max_result_count: int = Field(default=5, ge=1, le=10, description="Maximum number of results to return")
    search_date: str = Field(description="Date when search is performed")
    include_images: bool = Field(default=False, description="Whether to include image results")
    search_depth: str = Field(default="advanced", description="Search depth (basic/advanced)")

# Initialize Web Search Agent
web_search_agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=WebSearchParameters,
    result_type=WebSearchResponse,
    system_prompt=(
        "You are a web search specialist focused on accurate information retrieval and analysis. Your tasks:\n"
        "1. Process and validate search results from multiple sources\n"
        "2. Generate a comprehensive, fact-based summary\n"
        "3. Extract specific, actionable key findings\n"
        "4. Evaluate and rank results by relevance\n"
        "5. Generate targeted follow-up queries for deeper research\n"
        "Ensure all outputs strictly follow the specified schema and maintain high information quality."
    )
)


@web_search_agent.tool
async def execute_web_search(search_context: RunContext[WebSearchParameters]) -> dict:
    """Execute web search using Tavily API with error handling"""
    try:
        # Input validation
        search_query = search_context.deps.search_query.strip()
        if not search_query:
            raise ValueError("Search query cannot be empty")
            
        # Execute search with configurable parameters
        search_results = await tavily_client.search(
            query=search_query,
            max_results=search_context.deps.max_result_count,
            search_depth=search_context.deps.search_depth,
            include_images=search_context.deps.include_images
        )
        
        # Validate and process response
        if not search_results:
            logger.warning(f"Empty response for query: {search_query}")
            return {"results": [], "error": "Empty response from search API"}
            
        results = search_results.get('results', [])
        if not results:
            logger.warning(f"No results found for query: {search_query}")
            return {"results": [], "error": "No search results found"}
            
        # Add metadata to results
        processed_results = []
        for result in results:
            processed_result = {
                "title": result.get("title", "Untitled"),
                "content": result.get("content", "No content available"),
                "url": result.get("url", ""),
                "source_type": result.get("type", "Website"),
                "score": float(result.get("score", 0.0)),
                "date": result.get("published_date", "")
            }
            processed_results.append(processed_result)
            
        return {
            "results": processed_results,
            "answer": search_results.get("answer", ""),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        error_message = f"Web search error: {str(e)}"
        logger.error(error_message)
        return {"error": error_message, "results": []}

async def process_web_search_results(user_query: str, message_placeholder: st.empty) -> str:
    """Process web search results and generate structured response"""
    try:
        # Initialize search parameters with defaults
        search_params = WebSearchParameters(
            search_query=user_query,
            max_result_count=5,
            search_date=datetime.now().strftime("%Y-%m-%d"),
            include_images=False,
            search_depth="advanced"
        )
        
        # Execute search and process results
        search_response = await web_search_agent.run(user_query, deps=search_params)
        
        if not search_response or not search_response.data:
            raise ValueError("No valid response from web search agent")
            
        # Format response with detailed sections
        formatted_response = f"""
            ### 📝 Search Summary
            {search_response.data.search_summary}

            ### 🎯 Key Findings
            {chr(10).join(f'- {finding}' for finding in search_response.data.search_findings)}

            ### 🌐 Relevant Sources
            """
        
        # Add individual search results with improved formatting
        for idx, result_item in enumerate(search_response.data.search_results, 1):
            formatted_response += f"""
                #### {idx}. {result_item.result_title}
                {result_item.result_content[:500]}{"..." if len(result_item.result_content) > 500 else ""}

                🔗 [View Source]({result_item.result_url})
                **Source Type**: {result_item.result_type}
                **Relevance Score**: {result_item.result_score:.2f}
                {f'**Date**: {result_item.result_date}' if result_item.result_date else ""}

                ---
                """
        
        # Add follow-up suggestions with improved formatting
        formatted_response += f"""
            ### 🔍 Suggested Follow-up Questions
            {chr(10).join(f'- {query}' for query in search_response.data.follow_up_queries)}

            *Search performed at: {search_response.data.search_timestamp}*
            """
        
        return formatted_response
        
    except ValueError as ve:
        logger.warning(f"Validation error in web search: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error processing web search results: {str(e)}")
        raise


# ===== Section: Document Section Processing =====

async def extract_sections(doc_text: str, llm=None) -> List[SectionOutput]:
    """Extract sections from document text"""
    llm = llm or OpenAI(model="gpt-4o-mini")
    system_prompt = """Extract section metadata from document text. Only extract if text contains section beginning.
    - Valid section MUST begin with hashtag (#) and have number
    - Extract multiple sections if present
    - Do NOT extract if no sections begin in text
    - Figures/Tables do NOT count as sections
    """
    
    chat_template = ChatPromptTemplate([
        ChatMessage.from_str(system_prompt, "system"),
        ChatMessage.from_str("Document text: {doc_text}", "user"),
    ])
    
    result = await llm.astructured_predict(
        SectionsOutput,
        chat_template,
        doc_text=doc_text
    )
    return result.sections

async def refine_sections(sections: List[SectionOutput], llm=None) -> List[SectionOutput]:
    """Review and correct extracted sections"""
    llm = llm or OpenAI(model="gpt-4o-mini")
    system_prompt = """Review and correct extracted sections.
    Check for:
    - False positive sections
    - Incorrect subsection marking
    - Valid section text in description
    Return only valid section indexes.
    """
    
    chat_template = ChatPromptTemplate([
        ChatMessage.from_str(system_prompt, "system"),
        ChatMessage.from_str("Sections:\n{sections}", "user"),
    ])
    
    section_texts = "\n".join([
        f"{idx}: {json.dumps(s.model_dump())}" 
        for idx, s in enumerate(sections)
    ])
    
    result = await llm.astructured_predict(
        ValidSections,
        chat_template,
        sections=section_texts
    )
    
    return [s for idx, s in enumerate(sections) if idx in result.valid_indexes]

def annotate_chunks(chunks: List[TextNode], sections: List[SectionOutput]):
    """Annotate chunks with section metadata"""
    main_sections = [s for s in sections if not s.is_subsection]
    sub_sections = sections  # Include all sections
    
    main_idx, sub_idx = 0, 0
    for chunk in chunks:
        cur_page = chunk.metadata["page_num"]
        
        # Find current main section
        while (main_idx + 1 < len(main_sections) and 
               main_sections[main_idx + 1].start_page_number <= cur_page):
            main_idx += 1
            
        # Find current subsection
        while (sub_idx + 1 < len(sub_sections) and
               sub_sections[sub_idx + 1].start_page_number <= cur_page):
            sub_idx += 1
            
        # Add section metadata
        chunk.metadata["section_id"] = main_sections[main_idx].get_section_id()
        chunk.metadata["sub_section_id"] = sub_sections[sub_idx].get_section_id()


# ===== Section: Query Types Agent =====

class QueryType(str, Enum):
    FORM_FILLING = "form_filling"
    DOCUMENT_ANALYSIS = "document_analysis"
    COMPARISON = "comparison"
    QA = "qa"
    EXTRACTION = "extraction"
    SUMMARY = "summary"

class ContentMatch(BaseModel):
    text: str
    source: str
    context: str = Field(description="Surrounding context of the match")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FieldSuggestion(BaseModel):
    field_name: str
    suggested_value: str
    alternative_values: List[str] = Field(default_factory=list)
    source_matches: List[ContentMatch] = Field(default_factory=list)
    explanation: Optional[str] = None

class AnalysisResult(BaseModel):
    content: str
    matches: List[ContentMatch] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

class QueryIntent(BaseModel):
    query_type: QueryType
    target_fields: Optional[List[str]] = None
    context_requirements: Optional[List[str]] = None
    expected_format: Optional[Dict[str, str]] = None

class DynamicResponse(BaseModel):
    query_understanding: QueryIntent
    extracted_fields: Dict[str, FieldSuggestion] = Field(default_factory=dict)
    analysis_results: List[AnalysisResult] = Field(default_factory=list)
    suggested_actions: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
   
@dataclass
class ResponseProcessorDependencies:
    query: str
    has_documents: bool
    todays_date: str

class ResponseProcessorResult(BaseModel):
    dynamic_response: DynamicResponse = Field(
        description="Structured response using the DynamicResponse model"
    )

response_processor_agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=ResponseProcessorDependencies,
    result_type=ResponseProcessorResult,
    system_prompt=(
        "You are a response processing agent that analyzes queries and documents. "
        "Your task is to understand queries, extract relevant information, "
        "and provide structured responses using the DynamicResponse format. "
        "For each query, determine its type (FORM_FILLING, DOCUMENT_ANALYSIS, etc.), "
        "extract relevant fields, and provide structured analysis results."
    )
)

@response_processor_agent.tool
async def process_query(ctx: RunContext[ResponseProcessorDependencies], content: str) -> dict:
    """Process the query and content to generate a structured response"""
    # Use the query directly in the response
    query_type = QueryType.DOCUMENT_ANALYSIS  # Default type
    if any(keyword in ctx.deps.query.lower() for keyword in ['form', 'fill']):
        query_type = QueryType.FORM_FILLING
    elif any(keyword in ctx.deps.query.lower() for keyword in ['compare', 'difference']):
        query_type = QueryType.COMPARISON
    elif any(keyword in ctx.deps.query.lower() for keyword in ['extract', 'find']):
        query_type = QueryType.EXTRACTION
    elif any(keyword in ctx.deps.query.lower() for keyword in ['summarize', 'summary']):
        query_type = QueryType.SUMMARY
    elif '?' in ctx.deps.query:
        query_type = QueryType.QA

    query_intent = QueryIntent(
        query_type=query_type,
        target_fields=None,  # Would be populated based on form fields if present
        context_requirements=None,  # Would be populated based on query analysis
        expected_format=None  # Would be populated based on query requirements
    )
    
    # Create analysis results from content
    analysis_results = []
    if content:
        analysis_results.append(
            AnalysisResult(
                content=content[:200] + "..." if len(content) > 200 else content,  # Basic summary
                matches=[],  # Would contain relevant matches from content
                suggestions=[]  # Would contain suggestions based on content
            )
        )
    
    # Build response
    return {
        "dynamic_response": DynamicResponse(
            query_understanding=query_intent,
            analysis_results=analysis_results,
            extracted_fields={},  # Would be populated with extracted fields
            suggested_actions=[
                "Review the analyzed content",
                "Verify extracted information" if query_type == QueryType.EXTRACTION else None,
                "Check for missing context" if not ctx.deps.has_documents else None
            ],
            missing_information=[
                "No documents loaded" if not ctx.deps.has_documents else None
            ]
        )
    }

async def process_document_with_agent(user_query: str, document_content: str) -> DynamicResponse:
    """Process a document with the response processor agent"""
    deps = ResponseProcessorDependencies(
        query=user_query,
        has_documents=bool(document_content),
        todays_date=datetime.now().strftime("%Y-%m-%d")
    )

    result = await response_processor_agent.run(
        document_content,
        deps=deps
    )

    return result.data.dynamic_response

# ===== Section: Response Classifier Agent =====


class ResponseClassifierType(str, Enum):
    SIMPLE = "simple"  # For basic interactions like greetings, thanks
    DOCUMENT = "document"  # For document-based queries
    WEB = "web"  # For queries requiring web search
    HYBRID = "hybrid"  # For queries needing both document and web search

class ResponseClassifierInput(BaseModel):
    query: str = Field(..., description="User's input query")
    has_documents: bool = Field(..., description="Whether there are documents loaded")
    
class ResponseClassifierOutput(BaseModel):
    response_type: ResponseClassifierType = Field(..., description="Type of response to generate")
    explanation: str = Field(..., description="Explanation for the chosen response type")
    requires_web_search: bool = Field(..., description="Whether web search is needed")
    
response_classifier_agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=ResponseClassifierInput,
    result_type=ResponseClassifierOutput,
    system_prompt=(
        'You are a query classifier that determines how to best respond to user input. '
        'Analyze the query and available context to decide between simple responses, '
        'document search, web search, or hybrid approaches.'
    ),
)

@response_classifier_agent.system_prompt
async def add_document_context(ctx: RunContext[ResponseClassifierInput]) -> str:
    has_docs = "has_documents" if ctx.deps.has_documents else "does not have"
    return f"The system {has_docs} documents loaded for searching."

@response_classifier_agent.tool
async def classify_response(ctx: RunContext[ResponseClassifierInput]) -> ResponseClassifierOutput:
    """Classify the type of response needed for the user query"""
    # Simple patterns for basic interactions
    simple_patterns = {
        r'^(hi|hello|hey|thanks|thank you|bye|goodbye|ok|okay|yes|no|sure)$': True,
        r'^how are you': True,
        r'^\\s*$': True  # Empty or whitespace-only input
    }
    
    # Check for simple patterns first
    query_lower = ctx.deps.query.lower().strip()
    for pattern, is_simple in simple_patterns.items():
        if re.match(pattern, query_lower):
            return ResponseClassifierOutput(
                response_type=ResponseClassifierType.SIMPLE,
                explanation="This is a basic interaction that doesn't require complex processing",
                requires_web_search=False
            )
    
    # For document-specific queries when documents are loaded
    if ctx.deps.has_documents and any(word in query_lower for word in ['document', 'file', 'text', 'content', 'uploaded']):
        return ResponseClassifierOutput(
            response_type=ResponseClassifierType.DOCUMENT,
            explanation="This query specifically references the loaded documents",
            requires_web_search=False
        )
    
    # For queries needing current information
    if any(word in query_lower for word in ['latest', 'current', 'news', 'today', 'recent']):
        return ResponseClassifierOutput(
            response_type=ResponseClassifierType.WEB,
            explanation="This query requires current information from the web",
            requires_web_search=True
        )
    
    # Default to HYBRID if documents are loaded, WEB if not
    if ctx.deps.has_documents:
        return ResponseClassifierOutput(
            response_type=ResponseClassifierType.HYBRID,
            explanation="This query might benefit from both document and web search",
            requires_web_search=True
        )
    else:
        return ResponseClassifierOutput(
            response_type=ResponseClassifierType.WEB,
            explanation="No documents loaded, using web search",
            requires_web_search=True
        )


async def execute_search_flow_with_response_classification(question: str, message_placeholder: st.empty):
    """Execute the complete search flow with response classification"""
    try:
        with st.status("🔍 Processing your question...", expanded=True) as status:
            # Initial setup status
            status.write("🚀 Initializing search process...")
            status.write("📚 Checking available documents...")
            has_documents = "documents" in st.session_state and len(st.session_state.documents) > 0
            status.write(f"{'✅' if has_documents else '⚠️'} Documents found: {len(st.session_state.documents) if has_documents else 0}")

            # Classification phase
            status.write("\n🤔 Phase 1: Analyzing your question...")
            classifier_input = ResponseClassifierInput(
                query=question,
                has_documents=has_documents
            )
            
            status.write("⚙️ Running response classifier...")
            classification = await response_classifier_agent.run(
                question,
                deps=classifier_input
            )
            
            status.write(f"✅ Response type determined: {classification.data.response_type}")
            status.write(f"📝 Analysis: {classification.data.explanation}")
            
            if classification.data.response_type == ResponseClassifierType.SIMPLE:
                status.write("\n💬 Phase 2: Generating conversational response...")
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}"
                }
                
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful and friendly AI assistant. Respond naturally to greetings and basic interactions."
                        },
                        {
                            "role": "user",
                            "content": question
                        }
                    ],
                    "temperature": 0.7,
                    "stream": True
                }

                status.write("🎯 Generating response...")
                full_response = ""
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=payload
                    ) as response:
                        async for line in response.content:
                            if line:
                                try:
                                    json_response = json.loads(line.decode('utf-8').strip('data: ').strip())
                                    if json_response["choices"][0]["delta"].get("content"):
                                        content = json_response["choices"][0]["delta"]["content"]
                                        full_response += content
                                        message_placeholder.markdown(f"AI: {full_response}")
                                except Exception as e:
                                    continue

                status.update(label="✨ Response generated successfully!", state="complete")
                return full_response
                
            elif classification.data.response_type in [ResponseClassifierType.DOCUMENT, ResponseClassifierType.HYBRID]:
                status.write("\n📑 Phase 2: Processing document-based response...")
                
                # Document search phase
                if has_documents:
                    status.write("🔍 Searching through documents...")
                    status.write("⚡ Running BM25 search...")
                    bm25_results = []
                    if bm25_retriever:
                        bm25_results = await files_bm25_search(question, bm25_retriever)
                    
                    status.write("🔄 Running vector search...")
                    vector_results = await search_documents(question)
                    
                    status.write("🔗 Combining search results...")
                    document_content = ""
                    if bm25_results or vector_results:
                        combined_results = await combine_search_results(bm25_results, vector_results, question)
                        document_content = "\n".join([r.content for r in combined_results])
                        status.write(f"✅ Found {len(combined_results)} relevant document sections")
                
                # Analysis phase
                status.write("\n🧠 Phase 3: Analyzing content...")
                processor_deps = ResponseProcessorDependencies(
                    query=question,
                    has_documents=has_documents,
                    todays_date=datetime.now().strftime("%Y-%m-%d")
                )
                
                status.write("⚙️ Processing with AI agent...")
                dynamic_response = await process_document_with_agent(question, document_content)
                
                status.write("📝 Formatting response...")
                formatted_response = f"""
                    **Query Understanding**
                    Type: {dynamic_response.query_understanding.query_type.value}
                    Topics: {', '.join(filter(None, dynamic_response.query_understanding.target_fields or []))}

                    **Analysis**:
                    {dynamic_response.analysis_results[0].content if dynamic_response.analysis_results else 'No analysis results available.'}

                    **Evidence**:
                    {dynamic_response.analysis_results[0].matches[0].text if dynamic_response.analysis_results and dynamic_response.analysis_results[0].matches else 'No specific matches found.'}

                    **Suggested Actions**:
                    {chr(10).join('- ' + action for action in dynamic_response.suggested_actions if action)}

                    **Additional Information Needed**:
                    {chr(10).join('- ' + info for info in dynamic_response.missing_information if info)}
                    """
                message_placeholder.markdown(formatted_response)
                status.update(label="✨ Document analysis completed successfully!", state="complete")
                return formatted_response
                
            else:  # WEB type
                status.write("\n🌐 Phase 2: Web Search Integration")
                status.write("🔍 Initiating web search...")
                
                try:
                    # Process web search with enhanced error handling
                    status.write("⚡ Executing search and analyzing results...")
                    formatted_response = await process_web_search_results(question, message_placeholder)
                    
                    if formatted_response:
                        message_placeholder.markdown(formatted_response)
                        status.update(label="✨ Web search analysis completed!", state="complete")
                        return formatted_response
                    else:
                        error_msg = "No valid response generated from web search"
                        message_placeholder.warning(error_msg)
                        status.update(label="⚠️ No valid search results", state="complete")
                        return error_msg
                        
                except ValueError as ve:
                    error_msg = f"Invalid search parameters: {str(ve)}"
                    message_placeholder.warning(error_msg)
                    status.update(label="⚠️ Search parameter error", state="error")
                    return {"error": error_msg}
                    
                except Exception as e:
                    error_msg = f"Web search processing error: {str(e)}"
                    message_placeholder.error(error_msg)
                    status.update(label="❌ Search processing failed", state="error")
                    return {"error": error_msg}
            
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        message_placeholder.error(error_msg)
        logger.error(f"Error in execute_search_flow_with_response_classification: {str(e)}")
        status.update(label="❌ Error occurred during processing", state="error")
        return {"error": error_msg}

# ===== Section: Document Parsing =====

def get_llama_parser():
    """Get or initialize LlamaParse instance"""
    global llama_parser
    if llama_parser is None and st.secrets.get("LLAMA_CLOUD_API_KEY"):
        logger.debug("Initializing LlamaParse with API key")
        llama_parser = LlamaParse(
            api_key=st.secrets["LLAMA_CLOUD_API_KEY"],
            result_type="markdown"  # Changed to json
        )
        logger.debug("LlamaParse initialized successfully")
    return llama_parser

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def parse_file(file_path: str):
    try:
        logger.debug(f"Starting parse_file for: {file_path}")
        parser = get_llama_parser()
        if not parser:
            logger.error("LlamaParse not initialized - missing API key")
            raise ValueError("LlamaParse not initialized - missing API key")
        
        try:
            # Get JSON result from LlamaParse
            json_data = parser.get_json_result(file_path=file_path)
            logger.debug(f"JSON result received for {file_path}")
            
            # Create documents for each page
            documents = []
            for document_json in json_data:
                for page in document_json["pages"]:
                    doc = Document(
                        text=page["text"],
                        metadata={
                            'title': Path(file_path).stem,
                            'page_number': page["page"],
                            'filename': Path(file_path).name,
                            'doc_id': f"{Path(file_path).stem}_page_{page['page']}",
                            'file_path': str(file_path),
                            'file_type': FileType.from_extension(Path(file_path).suffix),
                            'source': Path(file_path).name,
                            'summary': '',
                            'hashtags': [],
                            'author': '',
                            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                    )
                    documents.append(doc)
            
            if not documents:
                logger.warning("No documents created from JSON data")
                return None
            
            # Combine all text for the final result
            combined_text = "\n\n".join(doc.text for doc in documents)
            
            # Create final result object
            result = type('DocumentResult', (), {
                'text': combined_text,
                'metadata': {
                    'title': Path(file_path).stem,
                    'summary': '',
                    'pages': len(documents),
                    'page_texts': [doc.text for doc in documents],
                    'page_numbers': [doc.metadata['page_number'] for doc in documents],
                    'doc_id': Path(file_path).stem,
                    'file_path': str(file_path),
                    'file_type': FileType.from_extension(Path(file_path).suffix),
                    'source': Path(file_path).name,
                    'hashtags': [],
                    'author': '',
                    'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'updated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            })
            
            logger.debug(f"Successfully created document with {len(documents)} pages")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
        
    except Exception as e:
        logger.error(f"Error in parse_file: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error traceback: ", exc_info=True)
        raise

# ===== Section: Search Operations =====

async def setup_hybrid_collection(collection_name: str):
    """Ensures collection exists with proper hybrid search configuration"""
    try:
        collection_exists = await async_qdrant_client.get_collection(collection_name)
        if not collection_exists:
            await async_qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "text-dense": models.VectorParams(
                        size=1536,  # Azure OpenAI embedding size
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "text-sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                        )
                    )
                },
            )
    except Exception as e:
        logger.error(f"Error setting up hybrid collection: {str(e)}")
        raise

async def process_documents(documents: List[Document], collection_name: str) -> VectorStoreIndex:
    """Process documents for hybrid search using LlamaIndex"""
    try:
        # Setup collection with hybrid configuration
        await setup_hybrid_collection(collection_name)
        
        # Initialize vector store with hybrid search enabled
        vector_store = QdrantVectorStore(
            collection_name=collection_name,
            client=qdrant_client,
            aclient=async_qdrant_client,
            enable_hybrid=True,
            batch_size=20,
            fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
        )
        
        # Create index directly
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents=documents,
            embed_model=azure_openai_embed_model,
            storage_context=storage_context,
        )
        
        return index
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise

async def process_parsed_documents(json_data: List[dict], collection_name: str):
    """Process parsed documents for hybrid search"""
    try:
        # Create Document objects from JSON data
        documents = []
        for document_json in json_data:
            for page in document_json["pages"]:
                documents.append(
                    Document(
                        text=page["text"],
                        metadata={
                            "filename": document_json.get("filename", ""),
                            "page_number": page["page"]
                        }
                    )
                )
        
        # Process documents and create index
        index = await process_documents(documents, collection_name)
        return documents, index
    except Exception as e:
        logger.error(f"Error processing parsed documents: {str(e)}")
        raise

async def search_documents(query: str, collection_name: str = "test_collection", limit: int = 5) -> List[SearchResult]:
    """Search documents using LlamaIndex query engine"""
    try:
        # Initialize vector store with hybrid search enabled
        vector_store = QdrantVectorStore(
            collection_name=collection_name,
            client=qdrant_client,
            aclient=async_qdrant_client,
            enable_hybrid=True,
            batch_size=20,
            fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
        )
        
        # Initialize storage context and create index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )
        
        # Initialize query engine with hybrid search
        query_engine = index.as_query_engine(
            similarity_top_k=2,
            sparse_top_k=12,
            vector_store_query_mode="hybrid"
        )
        
        # Execute search
        response = await query_engine.aquery(query)
        
        # Convert response nodes to SearchResult objects
        results = []
        for node in response.source_nodes:
            metadata = DocumentMetadata(
                doc_id=node.id_,
                source=node.metadata.get("source", ""),
                title=node.metadata.get("title", ""),
                file_path=node.metadata.get("file_path", ""),
                file_type=FileType.from_extension(Path(node.metadata.get("file_path", "")).suffix),
                summary=node.metadata.get("summary", ""),
                hashtags=node.metadata.get("hashtags", [])
            )
            
            result = SearchResult(
                content=node.text,
                score=node.score if hasattr(node, 'score') else 0.0,
                metadata=metadata,
                section_id=node.id_
            )
            results.append(result)
            
        return results
    except Exception as e:
        logger.error(f"Error in search_documents: {str(e)}")
        return []

# Initialize BM25 retriever
bm25_retriever = None

def initialize_bm25_retriever(documents: List[Document]):
    """Initialize global BM25 retriever with documents"""
    try:
        global bm25_retriever
        
        # Create sentence splitter and parse documents into nodes
        node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        nodes = node_parser.get_nodes_from_documents(documents=documents)
        
        # Create document store and add nodes
        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)
        
        similarity_top_k_value = min(10, len(nodes))
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=docstore,
            similarity_top_k=similarity_top_k_value,
            stemmer=Stemmer.Stemmer("english"),
            language="english",
        )
        return bm25_retriever
    except Exception as e:
        logger.error(f"Error initializing BM25: {str(e)}")
        raise

async def files_bm25_search(query: str, bm25_retriever: BM25Retriever) -> List[Dict]:
    """Perform BM25 search on documents"""
    try:
        nodes = await bm25_retriever.aretrieve(query)
        return [
            {
                "Query": query,
                "Details": node.text,
                "Confidence": node.score,
            }
            for node in nodes
        ]
    except Exception as e:
        logger.error(f"Error in BM25 search: {str(e)}")
        return []

async def cohere_rerank(query: str, documents: List[Dict], top_n: int = 10) -> List[Dict]:
    """Rerank search results using Cohere"""
    try:
        co = cohere.Client(st.secrets["COHERE_API_KEY"])
        rerank_docs = [{'text': str({"result": doc}).replace('$', '\$')} for doc in documents]
        
        results = co.rerank(
            query=query,
            documents=rerank_docs,
            top_n=top_n,
            model='rerank-english-v3.5'
        )
        
        # Map reranked results back to original documents with scores
        reranked_docs = []
        for result in results:
            doc = documents[result.index]
            doc['relevance_score'] = result.relevance_score  # Update the score
            reranked_docs.append(doc)
            
        return reranked_docs
    except Exception as e:
        logger.error(f"Error in cohere reranking: {str(e)}")
        return documents[:top_n]


# ===== Section: Response Processing =====

async def process_response(prompt: str, search_results: List[SearchResult], message_placeholder: st.empty) -> str:
    """Process search results and generate response with streaming"""
    try:
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "content": result.content,
                "score": result.score,
                "metadata": {
                    "doc_id": result.metadata.doc_id,
                    "title": result.metadata.title,
                    "file_path": result.metadata.file_path
                }
            })

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}"
        }
        
        system_prompt = """Analyze the search results and generate a comprehensive response to the user's question.
                    Include relevant quotes and citations from the search results.
                    Format the response with these sections:
                    **Question Summary**
                    **Key Topics**
                    **Answer**
                    **Source of Evidence**
                    **Confidence** (low/medium/high)"""
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Question: {prompt}\n\nSearch Results: {json.dumps(formatted_results, indent=2)}"
                }
            ],
            "temperature": 0.7,
            "stream": True,
            "seed": 42
        }

        full_response = ""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                async for line in response.content:
                    if line:
                        try:
                            json_response = json.loads(line.decode('utf-8').strip('data: ').strip())
                            if json_response["choices"][0]["delta"].get("content"):
                                content = json_response["choices"][0]["delta"]["content"]
                                full_response += content
                                message_placeholder.markdown(f"**Final Response**: {full_response}")
                        except Exception as e:
                            continue

        return full_response

    except Exception as e:
        logger.error(f"Error in process_response: {str(e)}")
        return f"Error generating response: {str(e)}"

async def analyze_question(question: str, needs_placeholder: st.empty) -> Dict[str, Any]:
    """Extract search topics and intent from user question using GPT-4"""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Analyze the user's question and extract key search topics and intent. Begin with 'Here is what I am understanding from your question...' Return a JSON with query (rephrased question), topics (list of key topics), and search_params (any specific parameters for search)."},
                {"role": "user", "content": question}
            ],
            "stream": True,
            "seed": 42
        }
        
        understanding = ""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                async for line in response.content:
                    if line:
                        try:
                            line_text = line.decode('utf-8').strip()
                            if line_text.startswith('data: '):
                                json_str = line_text.removeprefix('data: ')
                                if json_str != '[DONE]':
                                    json_response = json.loads(json_str)
                                    if json_response["choices"][0]["delta"].get("content"):
                                        content = json_response["choices"][0]["delta"]["content"]
                                        understanding += content
                                        needs_placeholder.markdown(f"**Understanding Your Question**:\n{understanding}")
                        except Exception as e:
                            continue
                
                # Parse the final response into JSON
                try:
                    # Clean up the understanding text to extract just the JSON part
                    json_str = understanding[understanding.find('{'):understanding.rfind('}')+1]
                    result = json.loads(json_str)
                except:
                    # If the streaming response isn't valid JSON, make a final call to get structured data
                    payload["stream"] = False
                    async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=payload
                    ) as final_response:
                        final_data = await final_response.json()
                        content = final_data['choices'][0]['message']['content']
                        # Extract JSON from the content
                        json_str = content[content.find('{'):content.rfind('}')+1]
                        result = json.loads(json_str)
                
                return result
                
    except Exception as e:
        logger.error(f"Error in question analysis: {str(e)}")
        return {
            "query": question,
            "topics": [],
            "search_params": {}
        }

async def process_files_data(prompt: str, search_results: List[Dict]) -> List[str]:
    """Process search results into structured responses"""
    try:
        system_prompt = """
        Response Structure:
        Please use the appropriate JSON schema below for your response.

        {
        "User Question": "string",
        "Key Words": "string",
        "Response": "string",
        "Source of Evidence": "string",
        }
        User Question: Quote the user's question, no changes.
        Key Words: Summarize the main topics that the answer covers with Keywords.
        Response: Provide answer, focus on the quantitative and qualitative aspects of the answer.
        Source of Evidence: Quote the most relevant part of the search result that answers the user's question.
        """

        headers = {
            "Content-Type": "application/json", 
            "Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}"
        }

        async with aiohttp.ClientSession() as session:
            tasks = []
            for result in search_results:
                payload = {
                    "model": "gpt-4o-mini",
                    "response_format": { "type": "json_object" },
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"User Needs Input: {prompt}, FILES search result: {result}"},
                            ]
                        },
                    ],
                    "seed": 42,
                }
                task = session.post(
                    "https://api.openai.com/v1/chat/completions", 
                    headers=headers, 
                    json=payload
                )
                tasks.append(task)

            responses = await asyncio.gather(*tasks)
            output = []
            for response in responses:
                response_json = await response.json()
                output.append(response_json['choices'][0]['message']['content'])
            
            return output
            
    except Exception as e:
        logger.error(f"Error processing files data: {str(e)}")
        return []

async def execute_search_flow(question: str, message_placeholder: st.empty) -> Dict[str, Any]:
    """Execute the complete search flow with question analysis, multiple search methods, and result combination"""
    try:
        # Initialize response containers
        main_full_response = ""
        
        with st.status("🔍 Processing your question...", expanded=True) as status:
            # 1. Analyze question
            status.write("🤔 Analyzing your question...")
            query_intent_dict = await analyze_question(question, st.empty())
            query = query_intent_dict.get("query", question)  # Use rephrased query if available
            
            # Convert dict to QueryIntent model
            query_intent = QueryIntent(
                query_type=QueryType.QA,  # Default to QA
                target_fields=query_intent_dict.get("target_fields"),
                context_requirements=query_intent_dict.get("context_requirements"),
                expected_format=query_intent_dict.get("expected_format")
            )
            
            status.write("✅ Question analyzed")
            status.write(f"Understanding: Query type: {query_intent.query_type}")
            
            # 2. Run BM25 search + Cohere rerank
            status.write("\n🔍 Searching through documents...")
            if bm25_retriever is None:
                logger.warning("BM25 retriever not initialized")
                bm25_results = []
            else:
                bm25_results = await files_bm25_search(query, bm25_retriever)
                if bm25_results:
                    status.write("BM25 Search Results:")
                    for i, result in enumerate(bm25_results[:3], 1):
                        status.write(f"Document {i}: {result.content[:200]}...")
            
            # 3. Run hybrid vector search
            try:
                vector_results = await search_documents(query)
                if vector_results:
                    status.write("\nVector Search Results:")
                    for i, result in enumerate(vector_results[:3], 1):
                        status.write(f"Document {i}: {result.content[:200]}...")
            except Exception as e:
                logger.error(f"Vector search error: {str(e)}")
                vector_results = []
            
            status.write("✅ Document search complete")
            
            # 4. Combine results
            status.write("\n🔄 Processing search results...")
            combined_results = await combine_search_results(bm25_results, vector_results, query)
            
            # Display combined results with full context
            results_count = len(combined_results)
            if results_count > 0:
                status.write(f"\nFound {results_count} relevant documents:")
                # Group documents by source
                docs_by_source = {}
                for doc in combined_results:
                    source = doc.metadata.source if hasattr(doc.metadata, 'source') else 'Unknown Source'
                    if source not in docs_by_source:
                        docs_by_source[source] = []
                    docs_by_source[source].append(doc)
                
                # Display documents grouped by source
                for source, chunks in docs_by_source.items():
                    status.write(f"\n📂 {os.path.basename(source)}:")
                    for i, chunk in enumerate(chunks, 1):
                        status.write(f"  Chunk {i}: {chunk.content[:200]}...")
            else:
                status.write("No relevant documents found")
            status.write("✅ Results processed")
            
            # 5. Generate response
            status.write("\n✍️ Generating response...")
            final_response = await process_response(question, combined_results, st.empty())
            main_full_response = final_response
            status.write("✅ Response generated")
            
            # Keep status expanded to show process
            status.update(label="✅ Processing complete!", state="complete", expanded=True)
            
            # Display final response outside status expander
            message_placeholder.markdown(f"### Final Response\n{main_full_response}")
            
            return {
                "question_analysis": query_intent_dict,
                "search_results": combined_results,
                "final_response": main_full_response
            }

    except Exception as e:
        logger.error(f"Error in search flow: {str(e)}")
        if 'status' in locals():
            status.update(label=f"❌ Error: {str(e)}", state="error")
        return {
            "error": str(e),
            "question_analysis": None,
            "search_results": [],
            "final_response": None
        }


async def combine_search_results(
    bm25_results: List[Dict],
    vector_results: List[SearchResult],
    query: str
) -> List[SearchResult]:
    """Combine and deduplicate results from different search methods"""
    try:
        # Convert BM25 results to SearchResult format if needed
        formatted_bm25_results = []
        for result in bm25_results:
            if isinstance(result, dict):
                # Create metadata
                metadata = DocumentMetadata(
                    doc_id=result.get("id", "unknown"),
                    title=result.get("title", "unknown"),
                    file_path=result.get("file_path", "unknown"),
                    file_type=FileType.from_extension(os.path.splitext(result.get("file_path", ""))[1])
                )
                
                # Create SearchResult
                search_result = SearchResult(
                    content=result.get("content", ""),
                    score=result.get("score", 0.0),
                    metadata=metadata,
                    section_id=result.get("section_id")
                )
                formatted_bm25_results.append(search_result)
            else:
                formatted_bm25_results.append(result)
        
        # Combine results
        all_results = formatted_bm25_results + vector_results
        
        # Remove duplicates based on content similarity
        unique_results = []
        seen_content = set()
        
        for result in all_results:
            content_hash = hash(result.content[:200])  # Use first 200 chars as signature
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        # Sort by score
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        # Rerank if Cohere client is available
        if cohere_client is not None:
            try:
                # Format for reranking
                docs_to_rerank = [
                    {
                        "text": result.content,
                        "id": str(i)
                    }
                    for i, result in enumerate(unique_results)
                ]
                
                # Rerank
                reranked = cohere_client.rerank(
                    model="rerank-english-v3.0",
                    query=query,
                    documents=docs_to_rerank,
                    top_n=min(10, len(docs_to_rerank)),
                    return_documents=True
                )
                
                # Create final results list preserving original result objects
                final_results = []
                for hit in reranked.results:
                    # Extract the document ID from the text field which contains our ID
                    doc_text = hit.document.text
                    try:
                        # Find the ID in the document text
                        id_str = [d for d in docs_to_rerank if d["text"] == doc_text][0]["id"]
                        idx = int(id_str)
                        result = unique_results[idx]
                        result.score = hit.relevance_score
                        final_results.append(result)
                    except (ValueError, IndexError, KeyError) as e:
                        logger.error(f"Error processing reranked result: {str(e)}")
                        continue
                
                if final_results:
                    return final_results
                else:
                    logger.warning("No valid results after reranking, falling back to original results")
                    return unique_results
            
            except Exception as e:
                logger.error(f"Error in Cohere reranking: {str(e)}")
                return unique_results
        
        return unique_results
        
    except Exception as e:
        logger.error(f"Error combining search results: {str(e)}")
        return vector_results  # Fall back to vector results on error

# ===== Section: Chat Interface =====

async def chatallfiles_page():
    """Main chat interface page"""
    st.title("🌿 Parsely - Document Chat")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing_files" not in st.session_state:
        st.session_state.processing_files = False
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = {}
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = {}
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "clipboard" not in st.session_state:
        st.session_state.clipboard = ""
    if "search_mode" not in st.session_state:
        st.session_state.search_mode = "Standard"
    if "event_loop" not in st.session_state:
        st.session_state.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(st.session_state.event_loop)
    
    # Get documents referred
    documents_referred = [doc.name for doc in st.session_state.get('processed_files', {})]
    
    # Add file upload section
    st.subheader("📁 Upload Files")
    
    # Get all supported extensions
    supported_types = []
    for extensions in SUPPORTED_EXTENSIONS.values():
        supported_types.extend(extensions)
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=[ext[1:] for ext in supported_types]  # Remove leading dots
    )
    
    if uploaded_files:
        st.session_state["uploaded_files"] = uploaded_files
        
        # Process files button
        process_button = st.button("Process Files")
        if process_button:
            with st.status("Processing files...", expanded=True) as status:
                try:
                    # Get document references
                    documents_referred = [file.name for file in uploaded_files]
                    
                    # Initialize vector store
                    status.update(label="Initializing vector store...", state="running")
                    vector_store = QdrantVectorStore(
                        collection_name="test_collection",
                        client=qdrant_client,
                        aclient=async_qdrant_client,
                        enable_hybrid=True,
                        batch_size=20,
                        fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
                    )
                    
                    # Initialize storage context
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    
                    processed_docs = []
                    for uploaded_file in uploaded_files:
                        try:
                            # Process file
                            status.update(label=f"Processing {uploaded_file.name}...", state="running")
                            
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name
                            
                            try:
                                # Parse file using event loop
                                result = parse_file(tmp_path)
                                
                                if result and hasattr(result, 'text'):
                                    # Extract sections using event loop
                                    sections = await extract_sections(result.text)
                                    if sections:
                                        sections = await refine_sections(sections)
                                    
                                    # Store processed chunks in session state
                                    st.session_state.processed_documents[uploaded_file.name] = {
                                        'chunks': result.metadata['page_texts'],  # Use chunks from SentenceSplitter
                                        'metadata': result.metadata,
                                        'sections': sections
                                    }
                                    
                                    # Create Document object for indexing
                                    doc = Document(
                                        text=result.text,
                                        metadata={
                                            'source': uploaded_file.name,
                                            'title': result.metadata.get('title', ''),
                                            'summary': result.metadata.get('summary', ''),
                                            'sections': [s.model_dump() for s in sections] if sections else []
                                        }
                                    )
                                    processed_docs.append(doc)
                                    
                                    status.update(label=f"✅ Processed {uploaded_file.name}", state="running")
                                else:
                                    status.update(label=f"❌ Failed to process {uploaded_file.name}", state="error")
                            finally:
                                # Clean up temp file
                                os.unlink(tmp_path)
                                
                        except Exception as e:
                            status.update(label=f"❌ Error processing {uploaded_file.name}: {str(e)}", state="error")
                            logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
                            continue
                    
                    # Index processed documents
                    if processed_docs:
                        status.update(label="Indexing documents...", state="running")
                        index = VectorStoreIndex.from_documents(
                            processed_docs,
                            storage_context=storage_context
                        )
                        
                        # Store documents in session state and initialize BM25
                        st.session_state.documents = processed_docs
                        status.update(label="Initializing BM25 retriever...", state="running")
                        initialize_bm25_retriever(processed_docs)
                        
                        status.update(label="✅ Documents processed and indexed successfully!", state="complete")
                    else:
                        status.update(label="❌ No documents were successfully processed", state="error")
                        
                except Exception as e:
                    status.update(label=f"❌ Error during processing: {str(e)}", state="error")
                    logger.error(f"Error during processing: {str(e)}")

    # Chat interface
    st.subheader("💬 Chat with Your Documents")
    
    # Add search mode selection
    col1, col2 = st.columns([3, 1])
    with col1:
        st.session_state.search_mode = st.radio(
            "Select Search Mode:",
            ["Standard", "Smart Classification"],
            help="Standard: Traditional document search\nSmart Classification: Intelligent response type classification",
            horizontal=True
        )
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                if st.session_state.search_mode == "Smart Classification":
                    response = await execute_search_flow_with_response_classification(prompt, message_placeholder)
                    if isinstance(response, dict) and "error" in response:
                        message_placeholder.error(f"Error: {response['error']}")
                    elif isinstance(response, str):
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })
                else:
                    response = await execute_search_flow(prompt, message_placeholder)
                    if "error" in response:
                        message_placeholder.error(f"Error: {response['error']}")
                    else:
                        # Add final response to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response["final_response"]
                        })
                    
            except Exception as e:
                message_placeholder.error(f"Error: {str(e)}")
                logger.error(f"Error in chat processing: {str(e)}")


if __name__ == "__main__":
    asyncio.run(chatallfiles_page())
