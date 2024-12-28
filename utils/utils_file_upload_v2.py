"""
Optimized file upload and processing utility for the Parsely project.
This module provides a modular and efficient approach to handling different types of file uploads
and their processing using modern Python practices.
"""

import logging
import asyncio
import aiohttp
import streamlit as st
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from tempfile import NamedTemporaryFile
from qdrant_client import QdrantClient, AsyncQdrantClient, models
# from llama_index.core.schema import Document, TextNode
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores.types import (
    FilterOperator,
    FilterCondition,
    MetadataFilters,
    MetadataFilter,
)
from llama_index.core.vector_stores import VectorStoreQuery
from pydantic import BaseModel, Field, StrictInt, StrictFloat, StrictStr
from enum import Enum
from dataclasses import dataclass, field
import os
from llama_parse import LlamaParse
from pydantic_ai import Agent, RunContext
import concurrent.futures
# import datetime
from datetime import datetime, timezone
from typing import Tuple
import nest_asyncio
from openai import AsyncOpenAI
import hashlib
import pandas as pd

# Apply nest_asyncio to handle nested event loops
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=st.secrets["OPENAI_API_KEY"]
)

USERNAME = "parselyai"

# Initialize OpenAI client in session state
if "openai_client" not in st.session_state:
    st.session_state.openai_client = AsyncOpenAI(
        api_key=st.secrets.get("OPENAI_API_KEY")
    )

class ProcessingMethod(str, Enum):
    """Processing methods for different document types"""
    LLAMA_PARSER = "llama_parser"  # Precision parsing for complex documents
    PARSE_API_URL = "parse_api_url"  # General parsing for simple documents
    COLPALI = "colpali"  # Vision-based parsing for images and diagrams

#########################################################
##### 122324 Change - Agent for Qdrant Filter ###########
#########################################################
# Data class for dependencies
@dataclass
class QdrantFilterDependencies:
    """Dependencies for the Qdrant filter agent."""
    user_query: str
    selected_documents: Optional[List[str]] = None
    limit: int = 5
    session_id: Optional[str] = None
    available_metadata_keys: Optional[List[str]] = None # Add this

class Filter(BaseModel):
    """Represents a single metadata filter."""
    key: str = Field(..., description="The metadata key to filter on.")
    value: Union[StrictInt, StrictFloat, StrictStr, List[StrictStr], List[StrictFloat], List[StrictInt]] = Field(
        ..., description="The value to filter by."
    )
    operator: str = Field(default="==", description="The comparison operator (e.g., '==', '!=', '>', '<', '>=', '<=', 'in', 'nin').")

class QdrantFilterOutput(BaseModel):
    """Output containing structured Qdrant filter conditions."""

    filters: List[Filter] = Field(
        default_factory=list,
        description="List of metadata filters to apply.",
    )
    condition: str = Field(
        default="and",
        description="The logical condition ('and' or 'or') to combine the filters.",
    )
    reasoning: str = Field(
        ..., description="Agent's reasoning for choosing the specific filters and condition."
    )

    def to_metadata_filters(self) -> MetadataFilters:
        """Converts the output to a MetadataFilters object."""
        metadata_filters = []
        for f in self.filters:
            # Convert operator string to FilterOperator enum
            try:
                op = FilterOperator(f.operator)
            except ValueError:
                logger.warning(f"Invalid operator '{f.operator}' in filter. Using '==' as default.")
                op = FilterOperator.EQ

            metadata_filters.append(
                MetadataFilter(key=f.key, value=f.value, operator=op)
            )

        # Convert condition string to FilterCondition enum
        try:
            cond = FilterCondition(self.condition.lower())
        except ValueError:
            logger.warning(f"Invalid condition '{self.condition}'. Using 'and' as default.")
            cond = FilterCondition.AND

        return MetadataFilters(filters=metadata_filters, condition=cond)


# Initialize the Agent for generating Qdrant filter conditions
qdrant_filter_agent = Agent(
    model="openai:gpt-4o-mini",
    deps_type=QdrantFilterDependencies,
    result_type=QdrantFilterOutput,
    system_prompt="""
        You are an expert in constructing metadata filters for a vector database based on user queries.
        Analyze the user's query and the available metadata keys to infer the most relevant filters that should be applied.
        Consider the user's intent and the context provided by the available metadata.
        You have the ability to use all available metadata for filtering, not just 'source_name' and 'username'.
        Use the appropriate operators (e.g., ==, >, <, >=, <=, !=, in, nin, contains) for each filter based on the query.
        Provide a clear reasoning for your choice of filters and the logical condition to combine them.

        User Query: {user_query}
        Available Metadata Keys: {available_metadata_keys}
        Explicit Metadata Filters from User: {metadata_filters}
        Filter Condition (AND/OR) for Combining Filters: {filter_condition}
        Available Metadata Keys: {available_metadata_keys}

        Based on the above information, infer the best metadata filters to apply. Explain your reasoning.
        """,
)

@qdrant_filter_agent.tool
async def generate_qdrant_filters(
    context: RunContext,
) -> QdrantFilterOutput:
    """
    Generates Qdrant filter conditions based on user query and available metadata.
    """
    logger.info("Starting generate_qdrant_filters function")

    # Access dependencies directly from RunContext, which will now be a dictionary
    user_query = context.user_query
    selected_documents = context.selected_documents
    available_metadata_keys = context.available_metadata_keys

    nodes = index.docstore.get_nodes([list(index.index_struct.nodes_dict.keys())[0]])
    if nodes:
        available_metadata_keys = list(nodes[0].metadata.keys())
        logger.info(f"Available metadata keys: {available_metadata_keys}")
    else:
        available_metadata_keys = []
        logger.warning("No nodes found in index. Metadata filtering limited.")

    explicit_filters = [
        MetadataFilter(key='source_name', value=doc, operator=FilterOperator.EQ)
        for doc in (selected_documents or [])
    ]
    logger.debug(f"Explicit filters: {explicit_filters}")

    prompt = f"User Query: {user_query}\nAvailable Metadata Keys: {available_metadata_keys}\n"
    prompt += "Filter Condition (AND/OR) for Combining Filters: AND\n"
    prompt += "Infer best metadata filters. Use appropriate operators. Explain reasoning."

    if explicit_filters:
        explicit_filters_str = ", ".join(
            f"{{key: {f.key}, value: {f.value}, operator: {f.operator}}}"
            for f in explicit_filters
        )
        prompt += f"\nExplicit Metadata Filters: {explicit_filters_str}"

    logger.debug(f"Generated prompt: {prompt}")

    try:
        run_result = await qdrant_filter_agent.run(prompt)
        logger.info("Agent run completed successfully")
    except Exception as e:
        logger.error(f"Error during agent run: {str(e)}")
        raise

    if not hasattr(run_result, "data"):
        logger.error("RunResult object has no 'data' attribute")
        raise AttributeError("RunResult object has no attribute 'data'")

    qdrant_filter_output: QdrantFilterOutput = run_result.data
    qdrant_filter_output.metadata_filters.filters.extend(explicit_filters)
    logger.info("Qdrant filters generated successfully")
    logger.debug(f"Generated filters: {qdrant_filter_output.metadata_filters}")

    return qdrant_filter_output

#########################################################
##### 122324 Change - Agent for Qdrant Filter ###########
#########################################################

# 121924 Change - Initialize session state for current_session_id
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

# 121724 Change - modify the ProcessingConfig class to include session tracking
@dataclass
class ProcessingConfig:
    azure_openai_key: Optional[str] = None
    openai_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    embedding_model: str = "text-embedding-3-small"
    processing_method: ProcessingMethod = ProcessingMethod.PARSE_API_URL
    session_id: str = field(default_factory=lambda: st.session_state.current_session_id)
# Don’t mix @dataclass with pydantic.Field.
# If sticking with dataclass, use dataclasses.field(default_factory=...).
# If using pydantic models, inherit from BaseModel and use pydantic.Field.
# By correcting the ProcessingConfig definition, you'll ensure that all attributes are serializable, resolving the "Object of type FieldInfo is not JSON serializable" error.


# 121724 Change - Let's add a FileSummary class for overall file summaries
class FileSummary(BaseModel):
    """Overall summary of a processed file"""
    file_name: str
    total_chunks: int
    processing_method: ProcessingMethod
    session_id: str
    creation_time: datetime = Field(default_factory=datetime.utcnow)
    summary: str
    key_points: List[str]
    document_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentMetadata(BaseModel):
    """Metadata generated by the agent."""
    title: str
    hashtags: List[str] = Field(description="List of hashtags for the document")
    hypothetical_questions: List[str]
    summary: str

class DocumentInfo(BaseModel):
    """Complete document information combining external and agent-generated data."""
    source_name: str
    index: int
    text_chunk: str
    title: str
    hashtags: List[str]
    hypothetical_questions: List[str]
    summary: str

class ProcessingResult(BaseModel):
    """Processing result with error handling"""
    success: bool
    message: str
    method_used: ProcessingMethod
    document_info: Optional[List[Union[DocumentInfo, Dict[str, Any], TextNode]]] = None
    error: Optional[str] = None

class MethodRecommendationInput(BaseModel):
    user_complexity_preference: str = Field(
        ..., 
        description="User's stated complexity and requirements for the uploaded documents"
    )

class MethodRecommendationOutput(BaseModel):
    recommended_method: ProcessingMethod = Field(
        ..., 
        description="Recommended processing method"
    )
    explanation: str = Field(
        ..., 
        description="Reason for this recommendation"
    )


#########################################################
# Document Processing Method Recommendation Agents Based on Complexity and Needs
#########################################################

# Initialize Method Recommendation Agent
method_recommendation_agent = Agent(
    model="openai:gpt-4o-mini",
    deps_type=MethodRecommendationInput,
    result_type=MethodRecommendationOutput,
    system_prompt="""You are a document processing method recommendation agent. Your task is to recommend the best processing method based on the document complexity provided by the user.

                    For simple text-based documents, recommend PARSE_API_URL.
                    For complex documents without images/diagrams, recommend LLAMA_PARSER.
                    For documents with images/diagrams, recommend COLPALI.

                    Provide clear explanations for your recommendations."""
                    )

@method_recommendation_agent.tool
async def get_method_recommendation(search_context: RunContext[MethodRecommendationInput]) -> MethodRecommendationOutput:
    """Get document processing method recommendation based on user input"""
    try:
        complexity = search_context.deps.user_complexity_preference.strip().lower()
        logger.info(f"Processing complexity: {complexity}")

        if "simple" in complexity:
            logger.info("Recommending PARSE_API_URL for simple document")
            return MethodRecommendationOutput(
                recommended_method=ProcessingMethod.PARSE_API_URL,
                explanation="For simple text-based documents, we use the basic parsing API which is fast and efficient for straightforward content extraction."
            )
        elif "images" in complexity or "diagrams" in complexity:
            logger.info("Recommending COLPALI for document with images/diagrams")
            return MethodRecommendationOutput(
                recommended_method=ProcessingMethod.COLPALI,
                explanation="For documents containing images or diagrams, we use COLPALI which provides advanced vision capabilities for comprehensive document analysis."
            )
        elif "complex" in complexity:
            logger.info("Recommending LLAMA_PARSER for complex document")
            return MethodRecommendationOutput(
                recommended_method=ProcessingMethod.LLAMA_PARSER,
                explanation="For complex documents without images, we use the LLAMA parser which excels at handling intricate document structures and relationships."
            )
        else:
            logger.warning(f"No clear match for complexity: {complexity}, defaulting to PARSE_API_URL")
            return MethodRecommendationOutput(
                recommended_method=ProcessingMethod.PARSE_API_URL,
                explanation="No specific complexity requirements detected, using the basic parser for general document processing."
            )
            
    except Exception as e:
        error_message = f"Method recommendation error: {str(e)}"
        logger.error(error_message)
        return MethodRecommendationOutput(
            recommended_method=ProcessingMethod.PARSE_API_URL,
            explanation="An error occurred during recommendation. Defaulting to basic parser for safety."
        )

async def process_method_recommendation(complexity_option: str) -> Tuple[ProcessingMethod, str]:
    """Process method recommendation and return the method and explanation"""
    try:
        logger.info(f"Processing recommendation for complexity: {complexity_option}")
        
        # Create input parameters
        recommendation_input = MethodRecommendationInput(
            user_complexity_preference=complexity_option
        )
        
        # Run agent with proper deps parameter
        run_result = await method_recommendation_agent.run(
            recommendation_input.user_complexity_preference,
            deps=recommendation_input
        )
        
        if not run_result or not hasattr(run_result, 'data'):
            logger.warning("No valid response from recommendation agent, defaulting to PARSE_API_URL")
            return ProcessingMethod.PARSE_API_URL, "Defaulting to basic parser due to invalid response from recommendation agent."
        
        recommendation_output: MethodRecommendationOutput = run_result.data
        
        logger.info(f"Recommendation result: {recommendation_output.recommended_method}")
        return recommendation_output.recommended_method, recommendation_output.explanation
        
    except Exception as e:
        logger.error(f"Error getting method recommendation: {str(e)}")
        return ProcessingMethod.PARSE_API_URL, "Defaulting to basic parser due to error in recommendation."

# This is where chunk-level concurrency can be introduced:
async def file_processing_pipeline_step4_process_document(
    file_data: bytes,
    filename: str,
    config: ProcessingConfig,
    sem: asyncio.Semaphore,
    update_file_status_func: Optional[callable] = None,
    progress_bar: Optional[st.progress] = None
) -> ProcessingResult:
    logs = []
    tmp_path = None

    try:
        logs.append(f"Starting processing of {filename}")
        if update_file_status_func:
            update_file_status_func(filename, "🔄 Starting processing...", "running")

        with NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name
        logger.debug(f"Created temporary file {tmp_path}")
        if update_file_status_func:
            update_file_status_func(filename, "📝 Created temporary file", "running")

        try:
            if config.processing_method == ProcessingMethod.LLAMA_PARSER:
                if update_file_status_func:
                    update_file_status_func(filename, "🔍 Parsing with LLAMA_PARSER...", "running")
                parser = LlamaParse(api_key=st.secrets.get("LLAMA_CLOUD_API_KEY"))
                json_data = parser.get_json_result(file_path=tmp_path)
                logger.debug("LLAMA_PARSER returned JSON data.")
                if update_file_status_func:
                    update_file_status_func(filename, "✅ LLAMA_PARSER returned data", "running")

                documents = []
                all_pages = [page for doc in json_data if "pages" in doc for page in doc["pages"]]

                if progress_bar:
                    progress_bar.text(f"Processing {filename}: Chunking pages...")

                async def process_page(page_data, page_num):
                    # First generate metadata for the page
                    metadata,embedding = await file_processing_pipeline_step5_generate_metadata(str(page_data["text"]), filename, page_num)
                    if not metadata:
                        logger.error(f"Failed to generate metadata for {filename} page {page_num}")
                        return None

                    # Create TextNode with enriched metadata
                    return TextNode(
                        text=str(page_data["text"]),
                        embedding=embedding,
                        metadata={
                            "source_name": filename,
                            "page_number": page_num,
                            "title": metadata['title'],
                            "hashtags": metadata['hashtags'],
                            "hypothetical_questions": metadata['hypothetical_questions'],
                            "summary": metadata['summary'],
                        }
                    )

                page_tasks = [process_page(page, i + 1) for i, page in enumerate(all_pages)]
                documents = await asyncio.gather(*page_tasks)

                logger.debug(f"Extracted {len(documents)} documents.")
                if update_file_status_func:
                    update_file_status_func(filename, f"✅ Extracted {len(documents)} docs", "running")

                

                st.session_state['document_store'].extend(documents)  # Add new documents
                logger.debug(f"Value of st.session_state['document_store']: {st.session_state.get('document_store')}")
                logger.debug("Stored documents in session.")
                if update_file_status_func:
                    update_file_status_func(filename, "💾 Stored docs in session", "running")

            elif config.processing_method == ProcessingMethod.PARSE_API_URL:
                if update_file_status_func:
                    update_file_status_func(filename, "🔍 Preparing to parse with API...", "running")

                url = st.secrets['parse_api_url']
                async with aiohttp.ClientSession() as session:
                    form_data = aiohttp.FormData()
                    form_data.add_field('file', file_data, filename=filename)

                    if update_file_status_func:
                        update_file_status_func(filename, "📤 Uploading file...", "running")

                    if progress_bar:
                        progress_bar.text(f"Processing {filename}: Uploading...")
                    try:
                        async with sem, session.post(url, data=form_data) as response:
                            if response.status == 200:
                                if update_file_status_func:
                                    update_file_status_func(filename, "✅ File uploaded, processing...", "running")
                                if progress_bar:
                                    progress_bar.text(f"Processing {filename}: Processing on server...")
                                try:
                                    response_data = await response.json()
                                    logger.info(f"API Response Data for {filename}: {response_data}")

                                    texts = []
                                    if isinstance(response_data, dict) and 'extracted_text' in response_data:
                                        if isinstance(response_data['extracted_text'], dict):
                                            # Assuming keys in extracted_text represent chunk numbers (e.g., "0", "1", ...)
                                            for key, text_content in response_data['extracted_text'].items():
                                                texts.append(text_content)
                                        else:
                                            logger.error(f"Unexpected format for 'extracted_text' in API response for {filename}: {response_data['extracted_text']}")
                                            raise ValueError("Unexpected format for 'extracted_text'")
                                    else:
                                        logger.error(f"Unexpected API response format for {filename}: {response_data}")
                                        raise ValueError("Unexpected API response format")

                                    async def process_chunk(text, chunk_num):
                                        # Generate metadata for the chunk
                                        metadata, embedding = await file_processing_pipeline_step5_generate_metadata(text, filename, chunk_num)
                                        if not metadata:
                                            logger.error(f"Failed to generate metadata for {filename} chunk {chunk_num}")
                                            return None

                                        # Create TextNode with enriched metadata
                                        return TextNode(
                                            text=str(text),
                                            embedding=embedding,
                                            metadata={
                                                "source_name": filename,
                                                "chunk_number": chunk_num,  # Use chunk_number instead of page_number
                                                "title": metadata.get('title'),
                                                "hashtags": metadata.get('hashtags'),
                                                "hypothetical_questions": metadata.get('hypothetical_questions'),
                                                "summary": metadata.get('summary'),
                                            }
                                        )

                                    chunk_tasks = [process_chunk(text, i + 1) for i, text in enumerate(texts)]
                                    documents = await asyncio.gather(*chunk_tasks)
                                    documents = [doc for doc in documents if doc] # Remove any None results

                                    logs.append(f"Extracted {len(documents)} documents from API response.")
                                    if update_file_status_func:
                                        update_file_status_func(filename, f"✅ Extracted {len(documents)} docs", "running")

                                    if 'document_store' not in st.session_state:
                                        st.session_state['document_store'] = []

                                    st.session_state['document_store'].extend(documents)

                                except aiohttp.ContentTypeError:
                                    logger.error(f"API response for {filename} was not JSON.")
                                    if update_file_status_func:
                                        update_file_status_func(filename, f"❌ API error: Non-JSON response", "error")
                                    raise
                            else:
                                error_msg = f"API request failed with status code: {response.status}"
                                logger.error(error_msg)
                                if update_file_status_func:
                                    update_file_status_func(filename, f"❌ API error: {response.status}", "error")
                                raise Exception(error_msg)
                    except Exception as e:
                        logger.error(f"Error during API request for {filename}: {e}")
                        if update_file_status_func:
                            update_file_status_func(filename, f"❌ Error: {str(e)}", "error")
                        raise

            elif config.processing_method == ProcessingMethod.COLPALI:
                # TODO 122424 Implement COLPALI later: https://github.com/qdrant/demo-colpali-optimized/blob/master/ColPali%20as%20a%20reranker%20I.ipynb
                if update_file_status_func:
                    update_file_status_func(filename, "🖼️ Processing with COLPALI...", "running")
                if update_file_status_func:
                    update_file_status_func(filename, "✅ Extracted content from images", "running")

            logger.info("All chunks processed successfully.")
            if update_file_status_func:
                update_file_status_func(filename, "✅ All chunks processed", "running")

            return ProcessingResult(
                success=True,
                message=f"Document processed successfully with {len(st.session_state['document_store'])} chunks",
                method_used=config.processing_method,
                document_info=st.session_state['document_store'],
                error=None
            )
        except Exception as e:
            error_msg = f"Error processing document {filename}: {str(e)}"
            logger.error(error_msg)
            if update_file_status_func:
                update_file_status_func(filename, "❌ Error occurred", "error")
            return ProcessingResult(
                success=False,
                message="Error processing document",
                method_used=config.processing_method,
                document_info=None,
                error=str(e)
            )
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    logger.info(f"Deleted temporary file {tmp_path}")
            except Exception as e:
                logger.warning(f"Could not delete temporary file {tmp_path}: {str(e)}")

    except Exception as e:
        error_msg = f"Error processing document {filename}: {str(e)}"
        logger.error(error_msg)
        if update_file_status_func:
            update_file_status_func(filename, "❌ Error occurred", "error")
        return ProcessingResult(
            success=False,
            message="Error processing document",
            method_used=config.processing_method,
            document_info=None,
            error=str(e)
        )


async def file_processing_pipeline_step2_run_all_file_processing(pdf_files, image_files, excel_files, csv_files, other_files, option, update_file_status, progress_bars, processed_files_summary):
    await asyncio.gather(
        file_processing_pipeline_step3_process_pdf_file(pdf_files, option, update_file_status, progress_bars, processed_files_summary),
        file_processing_pipeline_step3_process_image_file(image_files, option, update_file_status, progress_bars, processed_files_summary),
        file_processing_pipeline_step3_process_excel_file(excel_files, option, update_file_status, processed_files_summary),
        file_processing_pipeline_step3_process_csv_file(csv_files, option, update_file_status, processed_files_summary),
        file_processing_pipeline_step3_process_other_file(other_files, option, update_file_status, progress_bars, processed_files_summary),
    )


async def file_processing_pipeline_step3_process_pdf_file(files, option, update_file_status, progress_bars, processed_files_summary):
    config = ProcessingConfig(processing_method=option)

    async def process_single_pdf(file):
        progress = progress_bars.get(file.name, st.empty())
        try:
            result = await file_processing_pipeline_step4_process_document(
                file_data=file.getvalue(),
                filename=file.name,
                config=config,
                sem=st.session_state.sem,
                update_file_status_func=update_file_status,
                progress_bar=progress
            )
            progress.empty()
            logger.debug(f"Result for file {file.name}: {result}")

            # result returns a ProcessingResult object
            # It looks like this:
            # {
            #     "success": True,
            #     "document_info": [
            #         {
            #             "source_name": "example_source",
            #             "index": 0,
            #             "text_chunk": "This is a document chunk.",
            #             "title": "Document Title",
            #             "hashtags": ["#hashtag1", "#hashtag2"],
            #             "hypothetical_questions": ["Question 1", "Question 2"],
            #             "summary": "Summary of the document.",
            #             }
            #         },
            #         ... (more chunks)
            #     ],
            #     "message": "Document processed successfully",
            # }
            # ***Each document_info chunk is a point uploaded to Qdrant***

            if result.success and result.document_info:
                update_file_status(file.name, f"✅ Processed {len(result.document_info)} chunks", state="complete")
                logger.info(f"Successfully processed file: {file.name}, adding to summary: { {'filename': file.name, 'method': config.processing_method.value, 'chunks': len(result.document_info)} }")
                processed_files_summary['successful'].append({
                    'filename': file.name,
                    'method': config.processing_method.value,
                    'chunks': len(result.document_info)
                })
            else:
                error_message = result.error or "Unknown error."
                update_file_status(file.name, f"❌ Error: {error_message}", state="error")
                logger.error(f"Error processing PDF file {file.name}: {error_message}")
                processed_files_summary['failed'].append({'filename': file.name, 'error': error_message})

        except Exception as e:
            # Handle exceptions if needed
            pass

    # Use asyncio.gather to process all PDF files concurrently
    await asyncio.gather(*[process_single_pdf(file) for file in files])



async def file_processing_pipeline_step3_process_image_file(files, option, update_file_status, progress_bars, processed_files_summary):
    config = ProcessingConfig(
        processing_method=option
    )

    async def process_single_image(file):
        progress = progress_bars.get(file.name, st.empty())
        try:
            result = await file_processing_pipeline_step4_process_document(
                file_data=file.getvalue(),
                filename=file.name,
                config=config,
                sem=st.session_state.sem,
                update_file_status_func=update_file_status,
                progress_bar=progress
            )
            progress.empty()
            logger.debug(f"Result for file {file.name}: {result}")

            if result.success and result.document_info:
                update_file_status(file.name, f"✅ Processed {len(result.document_info)} chunks", state="complete")
                logger.info(f"Successfully processed image file: {file.name}, adding to summary: { {'filename': file.name, 'method': config.processing_method.value, 'chunks': len(result.document_info)} }")
                processed_files_summary['successful'].append({
                    'filename': file.name,
                    'method': config.processing_method.value,
                    'chunks': len(result.document_info)
                })
            else:
                error_message = result.error or "Unknown error."
                update_file_status(file.name, f"❌ Error: {error_message}", state="error")
                logger.error(f"Error processing image file {file.name}: {error_message}")
                processed_files_summary['failed'].append({'filename': file.name, 'error': error_message})

        except Exception as e:
            error_message = str(e)
            update_file_status(file.name, f"❌ Error: {error_message}", state="error")
            logger.error(f"Error processing image file {file.name}: {error_message}")
            processed_files_summary['failed'].append({'filename': file.name, 'error': error_message})

    # Use asyncio.gather to process all image files concurrently
    await asyncio.gather(*[process_single_image(file) for file in files])

# ----------------------------------------------------------------------------
# 4) EXCEL FILE PROCESSING WITH CONCURRENT EMBEDDING
# ----------------------------------------------------------------------------
async def file_processing_pipeline_step3_process_excel_file(files, option, update_file_status, processed_files_summary):
    """
    Processes a list of Excel files asynchronously. Reads in chunks,
    generates metadata for each row, then concurrently fetches embeddings.
    """
    config = ProcessingConfig(processing_method=option)

    async def process_single_excel(file):
        update_file_status(file.name, "🔍 Processing Excel file...", "running")
        try:
            loop = asyncio.get_running_loop()
            # Reading the ExcelFile object in a thread executor
            with concurrent.futures.ThreadPoolExecutor() as pool:
                df_file = await loop.run_in_executor(pool, pd.ExcelFile, file)
                sheet_names = await loop.run_in_executor(pool, lambda: df_file.sheet_names)

            # Count total rows to display progress
            total_rows = 0
            for sheet_name in sheet_names:
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    df_chunk = await loop.run_in_executor(pool, pd.read_excel, df_file, sheet_name, nrows=0)
                    total_rows += len(df_chunk)

            processed_rows = 0
            all_nodes = []

            async def process_chunk(sheet, chunk_df, chunk_start_index):
                nonlocal processed_rows

                # 1) CREATE metadata tasks
                metadata_tasks = []
                for row_index, row in chunk_df.iterrows():
                    columns_values = ", ".join([f"{col}: {val}" for col, val in row.items()])
                    text_chunk = f"{file.name} {sheet} row {chunk_start_index + row_index}: {columns_values}"

                    metadata_tasks.append(
                        file_processing_pipeline_step5_generate_metadata(
                            text_chunk, file.name, chunk_start_index + row_index
                        )
                    )

                # 2) AWAIT all metadata tasks together
                metadata_results = await asyncio.gather(*metadata_tasks)

                # 3) CREATE embedding tasks for valid metadata
                embedding_tasks = []
                valid_metadata_indices = []
                for i, metadata in enumerate(metadata_results):
                    if metadata:
                        embedding_tasks.append(
                            generate_embedding(metadata['text_chunk'], "text-embedding-3-small")
                        )
                        valid_metadata_indices.append(i)
                    else:
                        error_message = "Failed to generate metadata"
                        row_num = chunk_start_index + i
                        update_file_status(file.name, f"❌ Error: {error_message} for row {row_num}", "error")
                        logger.error(f"Failed to generate metadata for {file.name} sheet {sheet} row {row_num}")

                # 4) AWAIT all embedding tasks together
                embeddings_results = await asyncio.gather(*embedding_tasks)

                # 5) CREATE the final TextNodes
                chunk_nodes = []
                embedding_idx = 0
                for i, metadata in enumerate(metadata_results):
                    if metadata:
                        embedding = embeddings_results[embedding_idx]
                        embedding_idx += 1
                        node = TextNode(
                            text=str(metadata['text_chunk']),
                            embedding=embedding,
                            metadata={
                                "source_name": metadata['source_name'],
                                "index": metadata['index'],
                                "title": metadata['title'],
                                "hashtags": metadata['hashtags'],
                                "hypothetical_questions": metadata['hypothetical_questions'],
                                "summary": metadata['summary'],
                                "file_type": "excel",
                                "sheet_name": sheet,
                                "row_number": chunk_start_index + i,
                                "original_values": chunk_df.iloc[i].to_dict()
                            }
                        )
                        chunk_nodes.append(node)

                processed_rows += len(chunk_df)
                update_file_status(file.name, f"🔍 Processing Excel file... {processed_rows}/{total_rows} rows", "running")

                return chunk_nodes

            chunk_size = 100  # Adjust as needed
            for sheet in sheet_names:
                # Read Excel in multiple chunks
                for i in range(0, total_rows, chunk_size):
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        chunk_df = await loop.run_in_executor(
                            pool, pd.read_excel, df_file, sheet, skiprows=i, nrows=chunk_size
                        )
                    chunk_nodes = await process_chunk(sheet, chunk_df, i)
                    all_nodes.extend(chunk_nodes)

            # Store all nodes in session state
            if 'document_store' not in st.session_state:
                st.session_state['document_store'] = []
            st.session_state['document_store'].extend(all_nodes)

            # Update status & summary
            if all_nodes:
                update_file_status(file.name, "✅ Excel file processed successfully!", "complete")
                logger.info(
                    f"Successfully processed Excel file: {file.name}, "
                    f"adding to summary: {{'filename': {file.name}, "
                    f"'method': {config.processing_method.value}, 'rows': {len(all_nodes)}}}"
                )
                processed_files_summary['successful'].append({
                    'filename': file.name,
                    'method': config.processing_method.value,
                    'rows': len(all_nodes)
                })
            else:
                error_message = "No valid metadata generated"
                update_file_status(file.name, f"❌ Error: {error_message}", "error")
                logger.error(f"Error processing Excel file {file.name}: {error_message}")
                processed_files_summary['failed'].append({'filename': file.name, 'error': error_message})

        except Exception as e:
            error_message = str(e)
            update_file_status(file.name, f"❌ Error: {error_message}", "error")
            logger.error(f"Error processing Excel file {file.name}: {error_message}")
            processed_files_summary['failed'].append({'filename': file.name, 'error': error_message})

    # Process all Excel files concurrently
    await asyncio.gather(*[process_single_excel(file) for file in files])

# ----------------------------------------------------------------------------
# 5) CSV FILE PROCESSING WITH CONCURRENT EMBEDDING
# ----------------------------------------------------------------------------
async def file_processing_pipeline_step3_process_csv_file(files, option, update_file_status, processed_files_summary):
    """
    Processes a list of CSV files asynchronously. Reads in chunks,
    generates metadata for each row, then concurrently fetches embeddings.
    """
    config = ProcessingConfig(processing_method=option)

    async def process_single_csv(file):
        update_file_status(file.name, "🔍 Processing CSV file...", "running")
        try:
            loop = asyncio.get_running_loop()
            # We can read CSV in chunks. (Note that pd.read_csv isn't natively async.)
            # Here, we read an *iterator* of DataFrame chunks:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                df_chunks = await loop.run_in_executor(pool, lambda: pd.read_csv(file, chunksize=100))

            total_rows = 0
            processed_rows = 0
            all_nodes = []

            async def process_chunk(chunk_df, chunk_start_index):
                nonlocal processed_rows

                # 1) CREATE metadata tasks
                metadata_tasks = []
                for row_index, row in chunk_df.iterrows():
                    columns_values = ", ".join([f"{col}: {val}" for col, val in row.items()])
                    text_chunk = f"{file.name} Row {chunk_start_index + row_index}: {columns_values}"

                    metadata_tasks.append(
                        file_processing_pipeline_step5_generate_metadata(
                            text_chunk, file.name, chunk_start_index + row_index
                        )
                    )

                # 2) AWAIT all metadata tasks
                metadata_results = await asyncio.gather(*metadata_tasks)

                # 3) CREATE embedding tasks for valid metadata
                embedding_tasks = []
                valid_metadata_indices = []
                for i, metadata in enumerate(metadata_results):
                    if metadata:
                        embedding_tasks.append(
                            generate_embedding(metadata['text_chunk'], "text-embedding-3-small")
                        )
                        valid_metadata_indices.append(i)
                    else:
                        error_message = "Failed to generate metadata"
                        row_num = chunk_start_index + i
                        update_file_status(file.name, f"❌ Error: {error_message} for row {row_num}", "error")
                        logger.error(f"Failed to generate metadata for {file.name} row {row_num}")

                # 4) AWAIT all embeddings
                embeddings_results = await asyncio.gather(*embedding_tasks)

                # 5) BUILD final TextNodes
                chunk_nodes = []
                embedding_idx = 0
                for i, metadata in enumerate(metadata_results):
                    if metadata:
                        embedding = embeddings_results[embedding_idx]
                        embedding_idx += 1
                        node = TextNode(
                            text=str(metadata['text_chunk']),
                            embedding=embedding,
                            metadata={
                                "source_name": metadata['source_name'],
                                "index": metadata['index'],
                                "title": metadata['title'],
                                "hashtags": metadata['hashtags'],
                                "hypothetical_questions": metadata['hypothetical_questions'],
                                "summary": metadata['summary'],
                                "file_type": "csv",
                                "row_number": chunk_start_index + i,
                                "original_values": chunk_df.iloc[i].to_dict()
                            }
                        )
                        chunk_nodes.append(node)

                processed_rows += len(chunk_df)
                update_file_status(file.name, f"🔍 Processing CSV file... {processed_rows}/{total_rows} rows", "running")

                return chunk_nodes

            # Process each chunk in the CSV iterator
            chunk_tasks = []
            chunk_index = 0
            for chunk_df in df_chunks:
                chunk_size = len(chunk_df)
                total_rows += chunk_size
                chunk_start_index = chunk_index * 100  # Each chunk is size=100
                chunk_index += 1

                chunk_tasks.append(process_chunk(chunk_df, chunk_start_index))

            # Wait for all chunk processing tasks
            chunk_results = await asyncio.gather(*chunk_tasks)
            for chunk_nodes in chunk_results:
                all_nodes.extend(chunk_nodes)

            # Store all nodes in session state
            if 'document_store' not in st.session_state:
                st.session_state['document_store'] = []
            st.session_state['document_store'].extend(all_nodes)

            # Update status & summary
            if all_nodes:
                update_file_status(file.name, "✅ CSV file processed successfully!", "complete")
                logger.info(
                    f"Successfully processed CSV file: {file.name}, "
                    f"adding to summary: {{'filename': {file.name}, "
                    f"'method': {config.processing_method.value}, 'rows': {len(all_nodes)}}}"
                )
                processed_files_summary['successful'].append({
                    'filename': file.name,
                    'method': config.processing_method.value,
                    'rows': len(all_nodes)
                })
            else:
                error_message = "No valid metadata generated"
                update_file_status(file.name, f"❌ Error: {error_message}", "error")
                logger.error(f"Error processing CSV file {file.name}: {error_message}")
                processed_files_summary['failed'].append({'filename': file.name, 'error': error_message})

        except Exception as e:
            error_message = str(e)
            update_file_status(file.name, f"❌ Error: {error_message}", "error")
            logger.error(f"Error processing CSV file {file.name}: {error_message}")
            processed_files_summary['failed'].append({'filename': file.name, 'error': error_message})

    # Process all CSV files concurrently
    await asyncio.gather(*[process_single_csv(file) for file in files])


async def file_processing_pipeline_step3_process_other_file(files, option, update_file_status, progress_bars, processed_files_summary):
    config = ProcessingConfig(
        processing_method=option
    )

    async def process_single_other(file):
        progress = progress_bars.get(file.name, st.empty())
        try:
            result = await file_processing_pipeline_step4_process_document(
                file_data=file.getvalue(),
                filename=file.name,
                config=config,
                sem=st.session_state.sem,
                update_file_status_func=update_file_status,
                progress_bar=progress
            )
            
            # The result returns a ProcessingResult object
            # It looks like this:
            # {
            #     "success": True,
            #     "document_info": [
            #         {
            #             "source_name": "example_source",
            #             "index": 0,
            #             "text_chunk": "This is a document chunk.",
            #             "title": "Document Title",
            #             "hashtags": ["#hashtag1", "#hashtag2"],
            #             "hypothetical_questions": ["Question 1", "Question 2"],
            #             "summary": "Summary of the document.",
            #             }
            #         },
            #         ... (more chunks)
            #     ],
            #     "error": None
            # }
            
            progress.empty()
            if result.success and result.document_info:
                update_file_status(file.name, f"✅ Processed {len(result.document_info)} chunks", state="complete")
                logger.info(f"Successfully processed other file: {file.name}, adding to summary: { {'filename': file.name, 'method': config.processing_method.value, 'chunks': len(result.document_info)} }")
                processed_files_summary['successful'].append({
                    'filename': file.name,
                    'method': config.processing_method.value,
                    'chunks': len(result.document_info)
                })
            else:
                error_message = result.error or "Unknown error."
                update_file_status(file.name, f"❌ Error: {error_message}", state="error")
                logger.error(f"Error processing other file {file.name}: {error_message}")
                processed_files_summary['failed'].append({'filename': file.name, 'error': error_message})
        except Exception as e:
            error_message = str(e)
            update_file_status(file.name, f"❌ Error: {error_message}", state="error")
            logger.error(f"Error processing other file {file.name}: {error_message}")
            processed_files_summary['failed'].append({'filename': file.name, 'error': error_message})

    # Use asyncio.gather to process all "other" files concurrently
    await asyncio.gather(*[process_single_other(file) for file in files])


################################################################################################
### Agent for document info used to parse and pre-process text extracted from files ###
################################################################################################
import asyncio
import logging
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import streamlit as st
import hashlib

logger = logging.getLogger(__name__)

class AgentMetadata(BaseModel):
    """Metadata generated by the agent."""
    title: str
    hashtags: List[str] = Field(description="List of hashtags for the document")
    hypothetical_questions: List[str]
    summary: str
# Initialize the Agent with the desired result type and system prompt
generate_document_metadata_async_agent = Agent(
    model="openai:gpt-4o-mini",  # Changed from gpt-4o-mini to standard gpt-4
    result_type=AgentMetadata,
    system_prompt=(
        "You are a document analysis assistant. Your task is to analyze document content and generate structured metadata.\n"
        "For each document chunk, you will:\n"
        "1. Create a concise but descriptive title\n"
        "2. Generate relevant hashtags\n"
        "3. Create hypothetical questions that could be answered by the content\n"
        "4. Write a brief summary\n\n"
        "Always ensure your output matches the AgentMetadata schema exactly."
    )
)
print(f"DEBUG: Initialized generate_document_metadata_async_agent with model: openai:gpt-4o-mini")

async def file_processing_pipeline_step5_generate_document_metadata_async(
    message_data: str,
    sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    """
    File Processing Pipeline Step 5: Rate-limited query to the Agent for generating document metadata.
    
    Args:
        message_data (str): The text chunk from the document.
        sem (asyncio.Semaphore): Semaphore to control concurrency.
    
    Returns:
        dict: A dictionary containing the combined metadata.
    """
    print(f"DEBUG: Entering file_processing_pipeline_step5_generate_document_metadata_async")
    async with sem:
        try:
            print(f"DEBUG: Generating metadata for chunk: {message_data[:100]}...")
            message_data_cache_key = f"metadata_cache_{hashlib.sha256(str(message_data).encode()).hexdigest()}"
            print(f"DEBUG: Generated cache key: {message_data_cache_key}")
            
            if 'metadata_cache' not in st.session_state:
                st.session_state.metadata_cache = {}
                print("DEBUG: Initialized metadata_cache in session state")
            
            if message_data_cache_key in st.session_state.metadata_cache:
                print("DEBUG: Using cached metadata")
                return st.session_state.metadata_cache[message_data_cache_key]
            
            print("DEBUG: Calling generate_document_metadata_async_agent.run")
            run_result = await generate_document_metadata_async_agent.run(str(message_data))
            
            if not run_result or not hasattr(run_result, 'data'):
                print("DEBUG: Agent returned invalid result")
                raise ValueError("Agent returned invalid result")
                
            print(f"DEBUG: RunResult data: {run_result.data}")
            
            metadata = run_result.data.model_dump()
            print(f"DEBUG: Converted Pydantic model to dict: {metadata}")
            
            st.session_state.metadata_cache[message_data_cache_key] = metadata
            print(f"DEBUG: Cached metadata with key: {message_data_cache_key}")
            return metadata

        except Exception as e:
            print(f"DEBUG: Error in generate_document_metadata_async: {str(e)}")
            raise

async def file_processing_pipeline_step5_generate_metadata(
    content: str,
    filename: str,
    page_num: int,
) -> Dict[str, Any]:
    """
    File Processing Pipeline Step 5: Generate metadata for a document chunk using Agent-based parsing.
    
    Args:
        content (str): The text content of the document chunk.
        filename (str): The name of the source file.
        page_num (int): The page number of the chunk within the document.
        config (Optional[Any]): Processing configuration.
    
    Returns:
        dict: A dictionary containing the complete metadata.
    """
    print(f"DEBUG: Entering file_processing_pipeline_step5_generate_metadata for {filename}, page {page_num}")
    try:
        content = str(content)
        filename_page_num_cache_key = f"doc_metadata_{hashlib.sha256(f'{filename}_{page_num}'.encode()).hexdigest()}"
        print(f"DEBUG: Generated cache key: {filename_page_num_cache_key}")
        
        if 'doc_metadata_cache' not in st.session_state:
            st.session_state.doc_metadata_cache = {}
            print("DEBUG: Initialized doc_metadata_cache in session state")
            
        if filename_page_num_cache_key in st.session_state.doc_metadata_cache:
            print(f"DEBUG: Using cached document metadata for {filename} chunk {page_num}")
            return st.session_state.doc_metadata_cache[filename_page_num_cache_key]
                
        if 'sem' not in st.session_state:
            max_concurrent_requests = 100
            st.session_state.sem = asyncio.Semaphore(max_concurrent_requests)
            print(f"DEBUG: Initialized semaphore with max_concurrent_requests: {max_concurrent_requests}")
        
        sem = st.session_state.sem
        
        print("DEBUG: Calling file_processing_pipeline_step5_generate_document_metadata_async")
        metadata = await file_processing_pipeline_step5_generate_document_metadata_async(content, sem)
        
        doc_info = DocumentInfo(
            source_name=filename,
            index=page_num,
            text_chunk=content,
            title=metadata["title"],
            hashtags=metadata["hashtags"],
            hypothetical_questions=metadata["hypothetical_questions"],
            summary=metadata["summary"],
        )
        print(f"DEBUG: Created DocumentInfo: {doc_info}")
        
        embedding = await generate_embedding(content)
        
        st.session_state.doc_metadata_cache[filename_page_num_cache_key] = doc_info.model_dump()
        print(f"DEBUG: Cached metadata for {filename} chunk {page_num}")
        return doc_info.model_dump(), embedding

    except Exception as e:
        print(f"DEBUG: Error generating metadata: {str(e)}")
        return {}, []

# 121724 Change - generate the overall document summary using the already processed document metadata from the chunks
class DocumentSummaryMetadata(BaseModel):
    """Overall document summary metadata"""
    summary: str
    key_points: List[str] = Field(description="Key points from all chunks")
    document_type: str
    themes: List[str] = Field(description="Main themes across all chunks")
    all_hashtags: List[str] = Field(description="Combined unique hashtags")
    key_questions: List[str] = Field(description="Selected important questions")

# Initialize the Agent for document summary generation, using GPT-4o for overall document summary
document_summary_agent = Agent(
    model="openai:gpt-4o",
    result_type=DocumentSummaryMetadata,
    system_prompt=(
        "You are an expert document analyzer. Given metadata from multiple document chunks, "
        "create a comprehensive document summary. Output a JSON structure with:\n"
        "1. summary: Overall document summary\n"
        "2. key_points: List of main points from all chunks\n"
        "3. document_type: Document classification\n"
        "4. themes: List of main themes across chunks\n"
        "5. all_hashtags: Combined relevant hashtags\n"
        "6. key_questions: Most important questions about the document\n\n"
        "Focus on finding patterns and connections between chunks.\n"
        "Return only the JSON data in the specified format."
    )
)

async def file_processing_pipeline_generate_document_metadata_overall_summary(
    chunks: List[str],
    filename: str,
    config: ProcessingConfig
) -> FileSummary:
    """Generate overall file summary using existing chunk metadata"""
    try:
        # Create a cache key for the entire document
        cache_key = f"doc_summary_{filename}"
        
        # Initialize summary cache if needed
        if 'doc_summary_cache' not in st.session_state:
            st.session_state.doc_summary_cache = {}
            
        # Return cached summary if available
        if cache_key in st.session_state.doc_summary_cache:
            logger.info(f"Using cached document summary for {filename}")
            return st.session_state.doc_summary_cache[cache_key]
            
        # Collect all chunk metadata for this document and sort by index
        chunk_metadata = []
        for cached_key, cached_value in st.session_state.get('doc_metadata_cache', {}).items():
            if cached_key.startswith(f"doc_metadata_{filename}_"):
                chunk_metadata.append(cached_value)
        
        # Sort chunks by index to maintain document order
        chunk_metadata.sort(key=lambda x: x.get('index', 0))
                
        if not chunk_metadata:
            logger.warning(f"No chunk metadata found for document: {filename}")
            raise ValueError(f"No chunk metadata found for document: {filename}")
            
        # Prepare consolidated metadata for the agent
        consolidated_data = {
            "chunks": len(chunk_metadata),
            "titles": [chunk.get("title", "").strip() for chunk in chunk_metadata if chunk.get("title")],
            "summaries": [chunk.get("summary", "").strip() for chunk in chunk_metadata if chunk.get("summary")],
            "hashtags": list(set(tag.strip() for chunk in chunk_metadata 
                              for tag in chunk.get("hashtags", []) if tag)),
            "questions": list(set(q.strip() for chunk in chunk_metadata 
                               for q in chunk.get("hypothetical_questions", []) if q))
        }
        
        # Validate consolidated data
        if not any([consolidated_data["titles"], consolidated_data["summaries"]]):
            raise ValueError("No valid titles or summaries found in chunk metadata")
        
        # Initialize semaphore if needed
        if 'sem' not in st.session_state:
            st.session_state.sem = asyncio.Semaphore(10)
            
        async with st.session_state.sem:
            # Generate overall summary using the agent
            prompt = (
                f"Generate overall summary for document '{filename}' with the following chunk metadata:\n"
                f"Number of chunks: {consolidated_data['chunks']}\n"
                f"Chunk titles (in order): {consolidated_data['titles']}\n"
                f"Chunk summaries (in order): {consolidated_data['summaries']}\n"
                f"All unique hashtags: {consolidated_data['hashtags']}\n"
                f"All unique questions: {consolidated_data['questions']}\n\n"
                f"Focus on creating a coherent narrative that connects all chunks."
            )
            
            run_result = await document_summary_agent.run(prompt)
            
            if not hasattr(run_result, 'data'):
                raise AttributeError("RunResult object has no attribute 'data'")
                
            summary_data: DocumentSummaryMetadata = run_result.data
            
            # Create FileSummary object with validated data
            summary = FileSummary(
                file_name=filename,
                total_chunks=len(chunks),
                processing_method=config.processing_method,
                session_id=config.session_id,
                summary=summary_data.summary.strip(),
                key_points=[point.strip() for point in summary_data.key_points if point],
                document_type=summary_data.document_type.strip(),
                metadata={
                    "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                    "chunks_analyzed": len(chunk_metadata),
                    "model_used": "gpt-4o-mini",
                    "themes": [theme.strip() for theme in summary_data.themes if theme],
                    "all_hashtags": [tag.strip() for tag in summary_data.all_hashtags if tag],
                    "key_questions": [q.strip() for q in summary_data.key_questions if q],
                    "chunk_indices": [chunk.get("index") for chunk in chunk_metadata]
                }
            )
            
            # Cache the summary
            st.session_state.doc_summary_cache[cache_key] = summary
            logger.info(f"Document summary cached for {filename}")
            
            return summary
            
    except Exception as e:
        logger.error(f"Error generating document summary: {str(e)}")
        # Create a basic summary as fallback
        return FileSummary(
            file_name=filename,
            total_chunks=len(chunks),
            processing_method=config.processing_method,
            session_id=config.session_id,
            summary=f"Error generating summary: {str(e)}",
            key_points=["Unable to process document"],
            document_type="unknown",
            metadata={
                "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "chunks_analyzed": len(chunk_metadata) if 'chunk_metadata' in locals() else 0,
                "available_metadata": bool(st.session_state.get('doc_metadata_cache', {}))
            }
        )


################################################################################################
### Function for generating unified document chunks ###
################################################################################################


def retry_async(retries=3, delay=1):
    """Retry decorator for async functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        logger.error(f"Failed after {retries} attempts: {str(e)}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    await asyncio.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator

openai_client = AsyncOpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

async def generate_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    try:
        response = await openai_client.embeddings.create(input=text, model=model)
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        logging.error(f"Error generating embedding for text '{text}': {e}")
        return None

# 121724 Change - Let's modify the setup_hybrid_collection function to include session tracking
@retry_async(retries=3)
async def file_processing_pipeline_step7_setup_search_index(
    processed_files_summary: Dict,
    config: ProcessingConfig,
    session_id: Optional[str] = None
) -> Optional[VectorStoreIndex]:
    """Set up hybrid collection with vector store"""
    # TODO 122224: Use unique file name from collection to see if already created index for processed files summary filename
    try:
        # Create a unique collection name per user
        collection_name = UNIFIED_COLLECTION_NAME

        # Initialize Qdrant clients
        qdrant_client = QdrantClient(url=st.secrets["qdrant_url"], api_key=st.secrets['qdrant_api_key'])
        qdrant_aclient = AsyncQdrantClient(url=st.secrets["qdrant_url"], api_key=st.secrets['qdrant_api_key'])

        current_session = session_id or config.session_id

        # Check if collection exists
        collection_exists = await qdrant_aclient.collection_exists(collection_name=collection_name)
        
        if not collection_exists:
            # Create collection with proper configuration
            await qdrant_aclient.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "text-dense": models.VectorParams(
                        size=1536,  # OpenAI Embeddings
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "text-sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                        )
                    )
                }
            )
            logger.info(f"Created collection '{collection_name}'")
        else:
            logger.info(f"Collection '{collection_name}' already exists")
        
        # Initialize vector store with hybrid search and metadata
        qdrant_vector_store = QdrantVectorStore(
            collection_name=collection_name,
            client=qdrant_client,
            aclient=qdrant_aclient,
            enable_hybrid=True,
            batch_size=100,
            fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
        )

        # # Initialize storage context and set chunk size
        # storage_context = StorageContext.from_defaults(
        #     vector_store=vector_store,
        # )
        # Settings.chunk_size = 1024

        # # Set up embedding model based on config
        # if config.azure_openai_key:
        #     embed_model = AzureOpenAIEmbedding(
        #         model=config.embedding_model,
        #         deployment_name=config.embedding_model,
        #         api_key=config.azure_openai_key,
        #         azure_endpoint=config.azure_endpoint,
        #     )
        # else:
        #     embed_model = OpenAIEmbedding(
        #         model=config.embedding_model,
        #         api_key=config.openai_key,
        #     )

        # Create index from documents with metadata
        if 'document_store' not in st.session_state or not st.session_state['document_store']:
            logger.error("No documents found in session state for indexing")
            return None

        logger.info(f"Length of document_store: {len(st.session_state.get('document_store', []))}")

        # Add metadata to documents, iterating through the successful files
        for file_info in processed_files_summary["successful"]:
            documents_referred = file_info["filename"]
            for doc in st.session_state['document_store']:
               if doc.metadata.get('source_name') == documents_referred: # Make sure we are updating the right documents metadata
                    doc.metadata.update({
                        "documents_referred": documents_referred,
                        "session_id": current_session,
                        "creation_time": datetime.utcnow().isoformat()
                    })
                
        # index = VectorStoreIndex(
        #     nodes = st.session_state['document_store'],
        #     embed_model=embed_model,
        #     storage_context=storage_context,
        #     use_async=True
        # )
        
        logger.info(f"\n\nAsync_Adding {len(st.session_state.get('document_store', []))} nodes to vector store\n\n")
        qdrant_vector_store_add_confirmation = await qdrant_vector_store.async_add(nodes=st.session_state['document_store'])
        logger.info(f"Async_Added {qdrant_vector_store_add_confirmation} nodes to vector store\n\n")

        logger.info(f"Initialized VectorStoreIndex for collection '{collection_name}'")
        return qdrant_vector_store

    except Exception as e:
        logger.error(f"Error setting up hybrid collection: {str(e)}")
        raise


async def get_unique_document_names(collection_name: str, qdrant_client: QdrantClient) -> List[str]:
    """Retrieve unique document names from the Qdrant collection."""
    try:
        unique_names = set()
        page_size = 100  # Adjust based on your collection size
        offset = 0

        while True:
            # Fetch a batch of points with only the 'source_name' payload
            points = qdrant_client.scroll(
                collection_name=collection_name,
                filter=None,
                limit=page_size,
                offset=offset,
                with_payload=["source_name"]
            )
            if not points:
                break
            for point in points:
                source_name = point.payload.get("source_name")
                if source_name:
                    unique_names.add(source_name)
            offset += page_size
            if len(points) < page_size:
                break

        return sorted(list(unique_names))
    except Exception as e:
        logger.error(f"Error retrieving unique document names: {str(e)}")
        return []

@st.cache_data(show_spinner=False)
def cached_get_unique_document_names(collection_name: str, qdrant_client: QdrantClient) -> List[str]:
    return asyncio.run(get_unique_document_names(collection_name, qdrant_client))

UNIFIED_COLLECTION_NAME = f"parsely_hybrid_search_index_{USERNAME}"

#########################################################
# Streamlit UI
#########################################################

# Define a mapping for predefined complexity options
PREDEFINED_RECOMMENDATIONS = {
    "Simple text-based document": {
        "display_name": "Parsely Parser",
        "method": ProcessingMethod.PARSE_API_URL,
        "explanation": "For simple text-based documents, the basic parsing API is fast and efficient for straightforward content extraction."
    },
    "Complex document (no images/diagrams)": {
        "display_name": "LLAMA Parser",
        "method": ProcessingMethod.LLAMA_PARSER,
        "explanation": "For complex documents without images, the LLAMA parser excels at handling intricate structures and relationships."
    },
    "Complex document with images/diagrams": {
        "display_name": "COLPALI Vision Parser",
        "method": ProcessingMethod.COLPALI,
        "explanation": "For documents containing images or diagrams, COLPALI provides advanced vision capabilities for comprehensive analysis."
    }
}

def get_predefined_method_recommendation(option: str) -> Tuple[ProcessingMethod, str]:
    """Retrieve the processing method and explanation based on predefined options."""
    recommendation = PREDEFINED_RECOMMENDATIONS.get(option)
    if recommendation:
        return recommendation["method"], recommendation["explanation"]
    else:
        # Default fallback
        return ProcessingMethod.PARSE_API_URL, "Defaulting to basic parser for general document processing."

# Define file type categories
FILE_CATEGORIES = {
    "pdf": ["pdf", "docx", "doc", "odt", "pptx", "ppt"],
    "image": ["png", "jpg", "jpeg"],
    "excel": ["xlsx"],
    "csv": ["csv", "tsv"],
    "other": ["eml", "msg", "rtf", "epub", "html", "xml", "txt"]
}

def file_processing_pipeline_step1_categorize_files(uploaded_files):
    """
    File Processing Pipeline Step 1: Categorize uploaded files based on their extensions.
    
    Args:
        uploaded_files (List): List of uploaded file objects.
    
    Returns:
        dict: Dictionary with categories as keys and lists of files as values.
    """
    categorized = {
        "pdf": [],
        "image": [],
        "excel": [],
        "csv": [],
        "other": []
    }
    
    for file in uploaded_files:
        ext = Path(file.name).suffix.lower().strip('.')
        categorized_found = False
        for category, extensions in FILE_CATEGORIES.items():
            if ext in extensions:
                categorized[category].append(file)
                categorized_found = True
                break
        if not categorized_found:
            categorized["other"].append(file)
    
    return categorized

##### Initialization Steps #####

##### Step 1: Setup UI and Initial State #####
@st.fragment
async def display_file_upload_ui():
    """Display the file upload UI and handle document processing"""
    st.title("📄 Document Processing System")

    # Initialize search query cache in session state if not already present
    if 'search_query_cache' not in st.session_state:
        st.session_state['search_query_cache'] = {}
        # Consider adding cache invalidation logic (e.g., expiration time)

    # Initialize processed_files_summary in session state if not already present
    if 'processed_files_summary' not in st.session_state:
        st.session_state['processed_files_summary'] = {"successful": [], "failed": []}

    # Initialize semaphore in session state if not already present
    if 'sem' not in st.session_state:
        max_concurrent_requests = 10  # Adjust based on your rate limits
        st.session_state.sem = asyncio.Semaphore(max_concurrent_requests)

    ##### Step 2: Load Document Sources #####
    # Upon app launch, load unique document sources from qdrant database under username
    if "all_unique_document_sources" not in st.session_state:
        unique_sources = await get_unique_document_sources()
        st.session_state.all_unique_document_sources = unique_sources

    ##### User Input Steps #####

    ##### Step 1: File Upload Interface #####
    # File uploader
    uploaded_files = st.file_uploader(
        "📂 Upload documents",
        accept_multiple_files=True,
        type=[
            "pdf", "docx", "doc", "odt", "pptx", "ppt", "xlsx", "csv",
            "tsv", "eml", "msg", "rtf", "epub", "html", "xml",
            "png", "jpg", "jpeg", "txt"
        ],
        help="Select multiple files to upload for processing."
    )

    ##### Post-Upload Steps #####

    if uploaded_files:
        st.markdown("### 📝 Document Processing")

        ##### Step 1: Pre-check for already processed files #####
        already_processed = []
        new_files = []
        processed_sources = st.session_state.get('all_unique_document_sources', set())

        for file in uploaded_files:
            if file.name in processed_sources:
                already_processed.append(file.name)
            else:
                new_files.append(file)


        # ##### Step 1.1: Handle already processed files and provide search functionality #####
        if already_processed:
            st.success(f"The following files have already been processed: {', '.join(already_processed)}")
            if not new_files:
                st.info("All uploaded files have already been processed. Would you like to search through them?")

                # Add search functionality for already processed files
                st.markdown("### 🔍 Search Through Processed Files")

                # Initialize VectorStoreIndex for the processed files
                already_processed_vector_store = QdrantVectorStore(
                    collection_name=UNIFIED_COLLECTION_NAME,
                    client=QdrantClient(url=st.secrets["qdrant_url"], api_key=st.secrets["qdrant_api_key"]),
                    aclient=AsyncQdrantClient(
                            url=st.secrets["qdrant_url"],
                            api_key=st.secrets["qdrant_api_key"]
                    ),
                    enable_hybrid=True,
                    batch_size=100,
                    embed_model="text-embedding-3-small",
                    fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
                )


                already_processed_storage_context = StorageContext.from_defaults(vector_store=already_processed_vector_store)
                already_processed_index = VectorStoreIndex([], storage_context=already_processed_storage_context)

                # Create search input
                search_query = st.text_input("🔎 Ask a question about your processed files:",
                                        placeholder="e.g., What is the main topic of these documents?")
                
                search_query_result_placeholder = st.empty()
                
                # ##### Step 1.3: Check if the current search query exists in the cache keys #####
                if search_query not in st.session_state['search_query_cache']:
                    st.info("You entered a new search query.")
                    st.session_state['search_query_cache'][search_query] = {
                        "query": search_query,
                        "response": None,
                        "source_nodes": [],
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    with st.spinner("🔍 Searching through your processed files..."):
                        try:
                            retrieval_pipeline_hybrid_search_response, retrieval_pipeline_hybrid_search_source_nodes = await retrieval_pipeline_hybrid_search(
                                query=search_query,
                                index=already_processed_index,
                                limit=5
                            )
                            logger.info(f"Retrieval Pipeline Hybrid Search Response: {retrieval_pipeline_hybrid_search_response}")

                            # Update cache with new results
                            st.session_state['search_query_cache'][search_query].update({
                                "response": retrieval_pipeline_hybrid_search_response.response,
                                "source_nodes": retrieval_pipeline_hybrid_search_response.source_nodes,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })

                            if retrieval_pipeline_hybrid_search_response:
                                with search_query_result_placeholder.container():
                                    st.write("### 📄 Response:")
                                    st.write(f"{retrieval_pipeline_hybrid_search_response.response}")
                                    for index, doc in enumerate(st.session_state['search_query_cache'][search_query]["source_nodes"], start=1):
                                        with st.expander(f"📄 {doc.get('source_name', 'Unknown')} - Chunk {index}"):
                                            st.markdown(f"**Content:** {doc.get('text_chunk', 'No content available')}")
                                            if doc.get('summary'):
                                                st.markdown(f"**Summary:** {doc['summary']}")
                                            if doc.get('hashtags'):
                                                st.markdown(f"**Tags:** {', '.join(doc['hashtags'])}")
                            else:
                                search_query_result_placeholder.info("ℹ️ No relevant content found in your processed files.")

                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                            logger.error(f"Error during hybrid search: {e}")

                elif search_query and len(search_query) > 1:  # The search query has more than 1 char and exists in the cache
                    # ##### Step 1.4: Handle case where the search query is the same as the cached query #####
                    st.info("You entered the same search query as in the previous search.")
                    if st.session_state['search_query_cache'][search_query]["response"]:
                        with search_query_result_placeholder.container():
                            st.write("### 📄 Previous Search Response:")
                            st.write(f"{st.session_state['search_query_cache'][search_query]['response']}")
                            for index, doc in enumerate(st.session_state['search_query_cache'][search_query]["source_nodes"], start=1):
                                with st.expander(f"📄 {doc.get('source_name', 'Unknown')} - Chunk {index}"):
                                    st.markdown(f"**Content:** {doc.get('text_chunk', 'No content available')}")
                                    if doc.get('summary'):
                                        st.markdown(f"**Summary:** {doc['summary']}")
                                    if doc.get('hashtags'):
                                        st.markdown(f"**Tags:** {', '.join(doc['hashtags'])}")
                            else:
                                search_query_result_placeholder.info("No results found in the previous search.")

                    # ##### Step 1.5: Provide option to find more results for the same query #####
                    if search_query_result_placeholder.button("Find more results"):
                        search_result_limit = 10
                        with st.spinner("🔍 Searching through your processed files..."):
                            try:
                                response, source_nodes = await retrieval_pipeline_hybrid_search(
                                    query=search_query,
                                    index=already_processed_index,
                                    limit=search_result_limit
                                )

                                # Update cache with new results
                                st.session_state['search_query_cache'][search_query].update({
                                    "response": response,
                                    "source_nodes": source_nodes,
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                })

                                with search_query_result_placeholder.container():
                                    st.write("### 📄 Response:")
                                    st.write(f"{st.session_state['search_query_cache'][search_query]['response']}")
                                    for index, doc in enumerate(st.session_state['search_query_cache'][search_query]["source_nodes"], start=1):
                                        with st.expander(f"📄 {doc.get('source_name', 'Unknown')} - Chunk {index}"):
                                            st.markdown(f"**Content:** {doc.get('text_chunk', 'No content available')}")
                                            if doc.get('summary'):
                                                st.markdown(f"**Summary:** {doc['summary']}")
                                            if doc.get('hashtags'):
                                                st.markdown(f"**Tags:** {', '.join(doc['hashtags'])}")
                            except Exception as e:
                                logger.error(f"Error searching processed files: {str(e)}")
                                search_query_result_placeholder.error("❌ An error occurred while searching. Please try again.")
                else:
                    st.info("Here is a summary of the documents in your processed files.")
                    summary_search_query = "Provide a summary:"
                    with st.spinner("🔍 Summarizing your processed files..."):
                        try:
                            # If summary_search_query for the file_name exists in the cache, use it
                            
                            response, source_nodes = await retrieval_pipeline_hybrid_search(
                                query=summary_search_query,
                                index=already_processed_index,
                                limit=5
                            )
                            
                            logger.info(f"Response: {response}")
                            logger.info(f"Source Nodes: {source_nodes}")

                            # Update cache with new results
                            st.session_state['search_query_cache'][summary_search_query] = {
                                "response": response,
                                "source_nodes": source_nodes,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }

                            if st.session_state['search_query_cache'][summary_search_query]["response"]:
                                with search_query_result_placeholder.container():
                                    st.write("### 📄 Summary:")
                                    st.write(st.session_state['search_query_cache'][summary_search_query]["response"].response)
                                    
                                    st.write("### 📚 Source Documents:")
                                    for node in st.session_state['search_query_cache'][summary_search_query]["response"].source_nodes:
                                        with st.expander(f"📄 {node.metadata.get('source_name', 'Unknown Source')}"):
                                            st.write(f"**Content:** {node.text[:500]}...")
                                            st.write(f"**Metadata:** {node.metadata}")
                                    
                                    # st.write("### get_formatted_sources")
                                    # st.write(st.session_state['search_query_cache'][summary_search_query]["response"].get_formatted_sources())
                            else:
                                st.warning("No summary could be generated. The index might be empty.")
                        except Exception as e:
                            logger.error(f"Error summarizing processed files: {str(e)}")
                            search_query_result_placeholder.error("❌ An error occurred while summarizing. Please try again.")

                return  # Exit early if all files were already processed

        st.divider()



        ##### Step 2: Method Selection and Configuration #####
        st.subheader("🤌 Method Selection and Configuration")
        # Input Method Selection
        input_method = st.selectbox(
            "🔍 Choose input method for document complexity:",
            ("Provide a textual description", "Select from predefined options"),
            help="Choose how you'd like to specify the complexity of your documents."
        )

        # Initialize session state for method and explanation
        if 'method' not in st.session_state:
            st.session_state.method = ProcessingMethod.PARSE_API_URL
        if 'explanation' not in st.session_state:
            st.session_state.explanation = ""

        ##### Step 3: Process Method Selection #####
        if input_method == "Provide a textual description":
            complexity_description = st.text_area(
                "✍️ Describe the complexity and characteristics of your document:",
                help="Provide details such as the presence of images, diagrams, the structure's intricacy, etc."
            )

            complexity_description_submit = st.button("✅ Submit Description", type="primary")

            if complexity_description_submit:
                # Check if the description is sufficiently detailed
                if len(complexity_description.strip()) > 5:
                    # Get processing method recommendation using the agent
                    with st.spinner("🔄 Analyzing your description to recommend a processing method..."):
                        try:
                            st.session_state.method, st.session_state.explanation = await process_method_recommendation(complexity_description)
                        except Exception as e:
                            logger.error(f"Error in method recommendation: {e}")
                            st.error(f"❌ Failed to get method recommendation: {e}")
                else:
                    st.warning("⚠️ Please provide a more detailed description of your document complexity.")

        elif input_method == "Select from predefined options":
            # manual_method_option = st.selectbox(
            #     "📊 Select document complexity:",
            #     list(PREDEFINED_RECOMMENDATIONS.keys()),
            #     help="This helps us choose the best processing method for your documents."
            # )
            #use radio button instead
            manual_method_option = st.radio(
                "📊 Select document complexity:",
                list(PREDEFINED_RECOMMENDATIONS.keys()),
                help="This helps us choose the best processing method for your documents."
            )
            
            st.session_state.method = PREDEFINED_RECOMMENDATIONS.get(manual_method_option)["method"]
            st.session_state.explanation = PREDEFINED_RECOMMENDATIONS.get(manual_method_option)["explanation"]

        ##### Step 4: Display Processing Method #####
        # After submission, display the recommendation if available
        if st.session_state.method and st.session_state.explanation:
            st.info(f"**✅ Recommended Processing Method:** {PREDEFINED_RECOMMENDATIONS.get(manual_method_option)['display_name']}")
            st.write(f"**📝 Reason:** {st.session_state.explanation}")

            # Add confirmation step
            st.write("### 🔄 Confirm Processing")
            st.write(f"**📄 Number of files to process:** {len(uploaded_files)}")
            for file in uploaded_files:
                st.write(f"- {file.name}")

            col1, col2 = st.columns(2)
            with col1:
                confirm = st.button("✅ Confirm and Process Files", key="confirm_btn", type="primary")
            with col2:
                cancel = st.button("❌ Cancel", key="cancel_btn", type="secondary")

            comfirm_and_process_files_status_placeholder = st.empty()

            ##### Step 5: File Processing Setup #####
            if confirm:
                try:
                    comfirm_and_process_files_status_placeholder.status("🔄 Setting up config...")
                    logger.info("Setting up config...")
                    # Initialize config
                    st.session_state['config'] = ProcessingConfig(
                        azure_openai_key=st.secrets.get("AZURE_OPENAI_API_KEY"),
                        azure_endpoint=st.secrets.get("AZURE_OPENAI_ENDPOINT"),
                        openai_key=st.secrets.get("OPENAI_API_KEY"),
                        processing_method=st.session_state.method
                    )

                    ##### Step 6: Execute File Processing #####

                    if 'document_store' not in st.session_state:
                        st.session_state['document_store'] = []

                    ##### Step 6.1: Sort Files by Type #####
                    # Categorize uploaded files
                    comfirm_and_process_files_status_placeholder.status("🗃️ Categorizing uploaded files...")
                    logger.info("Categorizing uploaded files...")
                    categorized_files = file_processing_pipeline_step1_categorize_files(uploaded_files)
                    pdf_files = categorized_files.get("pdf", [])
                    image_files = categorized_files.get("image", [])
                    excel_files = categorized_files.get("excel", [])
                    csv_files = categorized_files.get("csv", [])
                    other_files = categorized_files.get("other", [])
                    comfirm_and_process_files_status_placeholder.status("📂 Files categorized")
                    logger.info(f"PDF files: {pdf_files}")
                    logger.info(f"Image files: {image_files}")
                    logger.info(f"Excel files: {excel_files}")
                    logger.info(f"CSV files: {csv_files}")
                    logger.info(f"Other files: {other_files}")
                    
                    # Create a dictionary to hold individual file statuses
                    file_status_mapping = {}
                    for file in uploaded_files:
                        file_status_mapping[file.name] = st.empty()
                        file_status_mapping[file.name].text(f"⏳ Waiting to process {file.name}...")
                        comfirm_and_process_files_status_placeholder.status(f"⏳ Waiting to process {file.name}...")
                    # Define a callback that process_document can use to update status
                    def update_file_status(filename: str, message: str, state: str = "running"):
                        """
                        Update the processing status for a specific file.

                        Args:
                            filename (str): Name of the file being processed.
                            message (str): Status message to display.
                            state (str): State of the processing ('running', 'error', 'complete').
                        """
                        if filename in file_status_mapping:
                            # Prepend emojis based on state
                            emoji = "🔄" if state == "running" else "❌" if state == "error" else "✅"
                            file_status_mapping[filename].markdown(f"{emoji} {message}")
                            comfirm_and_process_files_status_placeholder.status(f"{emoji} {message}")

                    # Initialize summary lists
                    if 'processed_files_summary' not in st.session_state:
                        st.session_state['processed_files_summary'] = {
                            "successful": [],
                            "failed": []
                        }

                    # Initialize progress bars
                    progress_bars = {file.name: st.empty() for file in uploaded_files}

                    ##### Step 6.2: Process All File Types #####
                    # Run all file processing
                    with st.spinner("🔄 Processing files..."):
                        asyncio.run(file_processing_pipeline_step2_run_all_file_processing(
                            pdf_files=pdf_files,
                            image_files=image_files,
                            excel_files=excel_files,
                            csv_files=csv_files,
                            other_files=other_files,
                            option=st.session_state.method,
                            update_file_status=update_file_status,  # Pass the callback here
                            progress_bars=progress_bars,
                            processed_files_summary=st.session_state['processed_files_summary']
                        ))


                except Exception as e:
                    error_msg = f"❌ Error during document processing: {str(e)}"
                    logger.error(error_msg)
                    st.error(error_msg)

            elif cancel:
                st.warning("🚫 File processing canceled.")
                st.stop()

            ##### Step 6.3: Display Results #####
            # Display processing summary
            st.write("### 📊 Processing Summary")
            logger.info(f"Processed files summary: {st.session_state['processed_files_summary']}")
            if st.session_state['processed_files_summary']["successful"]:
                st.success(f"Successfully processed {len(st.session_state['processed_files_summary']['successful'])} files.")
                with st.expander("Details of successful files"):
                    st.json(st.session_state['processed_files_summary']["successful"])
            if st.session_state['processed_files_summary']["failed"]:
                st.error(f"Failed to process {len(st.session_state['processed_files_summary']['failed'])} files.")
                with st.expander("Details of failed files"):
                    st.json(st.session_state['processed_files_summary']["failed"])

            # Structure of the processed_files_summary['successful']
            # processed_files_summary['successful'].append({
            #     'filename': file.name,
            #     'method': config.processing_method.value,
            #     'chunks': len(result.document_info)
            # })
                
            ###### Step 7: Setup Search Index #####
            # After processing, set up the search index using the processed files
            if "processed_files_qdrant_vector_store" not in st.session_state:
                st.session_state["processed_files_qdrant_vector_store"] = None
            
            if st.button("Setup Search Index"):
                with st.spinner("🔍 Setting up search index..."):
                    try:
                        st.session_state["processed_files_qdrant_vector_store"] = await file_processing_pipeline_step7_setup_search_index(
                            processed_files_summary=st.session_state['processed_files_summary'],
                            config=st.session_state['config'] ,
                            session_id=st.session_state.get('current_session_id')
                        )

                        if st.session_state["processed_files_qdrant_vector_store"]:
                            st.success(f"Successfully added nodes to {USERNAME}'s collection {UNIFIED_COLLECTION_NAME}.")
                        else:
                            st.error(f"⚠️ Failed to add nodes to {USERNAME}'s collection {UNIFIED_COLLECTION_NAME}.")
                    except Exception as e:
                        error_msg = f"❌ Error adding nodes to {USERNAME}'s collection {UNIFIED_COLLECTION_NAME}: {str(e)}"
                        logger.error(error_msg)
                        st.error(error_msg)
            
            if st.session_state["processed_files_qdrant_vector_store"] is not None:
                st.subheader("🔍 Llama_Index Hybrid Search Query Engine")

                processed_files_hybrid_search_query_input = st.text_input("🔍 Enter your question:", value="What is the purpose of this document?"
                                                                            , key="processed_files_hybrid_search_query_input")

                processed_files_hybrid_search_query_input_embedding = await generate_embedding(processed_files_hybrid_search_query_input)

                processed_files_hybrid_search_query =  VectorStoreQuery(
                    query_str=processed_files_hybrid_search_query_input,
                    query_embedding=processed_files_hybrid_search_query_input_embedding,
                    similarity_top_k=10,     # Dense vector top k
                    sparse_top_k=20,        # Sparse vector top k
                    mode="hybrid"
                )

                processed_files_hybrid_search_query_response = await st.session_state["processed_files_qdrant_vector_store"].aquery(
                    query=processed_files_hybrid_search_query)
                
                # Show query parameters
                with st.container(border=True):
                    st.subheader("Query Details")
                    st.write("**Query String:**", processed_files_hybrid_search_query_input)
                    st.write("**Top K:**", processed_files_hybrid_search_query.similarity_top_k)
                    st.write("**Mode:**", processed_files_hybrid_search_query.mode)
                
                # Display results
                if processed_files_hybrid_search_query_response and processed_files_hybrid_search_query_response.nodes:
                    st.write(f"Found {len(processed_files_hybrid_search_query_response.nodes)} relevant documents")
                    
                    for i, (node, similarity) in enumerate(zip(processed_files_hybrid_search_query_response.nodes, processed_files_hybrid_search_query_response.similarities)):
                        with st.expander(f"📄 Result {i+1} (Similarity: {similarity:.3f})"):
                            st.write("**Source:**", node.metadata.get('source_name', 'Unknown'))
                            st.write("**Content:**")
                            st.markdown(node.text)
                            
                            # Show metadata
                            st.write("**Metadata:**")
                            metadata_df = pd.DataFrame([node.metadata]).T
                            st.dataframe(metadata_df)
                else:
                    st.info("No results found for the query.")
                

##### Document Management Steps #####
async def get_unique_document_sources() -> List[str]:
    """Retrieves unique 'source_name' values from the 'metadata' field in the unified collection."""
    unique_sources = set()
    try:
        next_page_offset = None
        while True:
            scroll_result = await AsyncQdrantClient(url=st.secrets["qdrant_url"], api_key=st.secrets["qdrant_api_key"]).scroll(
                collection_name=UNIFIED_COLLECTION_NAME,
                limit=100,
                with_payload=True,  # Fetch the entire payload
                with_vectors=False,
                offset=next_page_offset
            )
            logger.debug(f"Scroll result: {scroll_result}")

            points, next_page_offset = scroll_result

            if not points:
                logger.debug("No more points found.")
                break

            for record in points:
                unique_sources.add(record.payload["source_name"])

            if next_page_offset is None:
                logger.debug("End of scroll.")
                break

        logger.info(f"Unique document sources found: {sorted(list(unique_sources))}")
        return sorted(list(unique_sources))

    except Exception as e:
        logger.error(f"Error getting unique document sources: {e}")
        return []

@st.fragment
async def sidebar_doc_selection_component():
    """Handle document selection in sidebar"""
    try:
        st.header("💬 Document Chat")
        
        # Initialize session state
        if 'doc_selection_initialized' not in st.session_state:
            st.session_state.doc_selection_initialized = False
        if 'selected_sources' not in st.session_state:
            st.session_state.selected_sources = []
            
        with st.status("🔄 Setting up document selection...", expanded=True) as status:
            # Load document sources
            if st.button("Refresh Document List") or 'all_unique_document_sources' not in st.session_state:
                status.update(label="🔍 Loading document sources...", state="running")
                unique_sources = await get_unique_document_sources()
                st.session_state.all_unique_document_sources = unique_sources
                status.update(label="✅ Document sources loaded", state="complete")
            
            # Display document selection
            if st.session_state.get('all_unique_document_sources'):
                status.update(label="📂 Select documents to chat with", state="running")
                selected = st.multiselect(
                    "📂 Select document sources:",
                    options=st.session_state.all_unique_document_sources,
                    default=st.session_state.selected_sources or st.session_state.all_unique_document_sources[0],
                    help="Choose documents to include in your chat"
                )
                
                status.update(label="✅ Document selection ready", state="complete")

                return selected
            else:
                status.update(label="ℹ️ No documents available", state="error")
                st.info("Please upload some documents first")
                
        st.session_state.doc_selection_initialized = True
        
    except Exception as e:
        logger.error(f"Error in document selection: {str(e)}")
        st.error("❌ An error occurred during document selection")
        return False

@st.fragment
async def sidebar_query_engine_initialization_component():
    """Initialize query engine for selected documents"""
    try:
        if not st.session_state.get('selected_sources'):
            return False
            
        with st.status("🔄 Initializing search engine...", expanded=True) as status:            
            # Initialize vector store
            status.update(label="🔄 Setting up vector store...", state="running")
            vector_store = QdrantVectorStore(
                    collection_name=UNIFIED_COLLECTION_NAME,
                    client=QdrantClient(url=st.secrets["qdrant_url"], api_key=st.secrets["qdrant_api_key"]),
                    aclient=AsyncQdrantClient(
                            url=st.secrets["qdrant_url"],
                            api_key=st.secrets["qdrant_api_key"]
                    ),
                    enable_hybrid=True,
                    batch_size=100,
                    fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
                )
            st.session_state.qdrant_vector_store = vector_store
            st.session_state.qdrant_vector_store_initialized = True
            logger.info(f"Qdrant vector store initialized: {vector_store}")
            
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            status.update(label="✅ Vector store ready", state="complete")

            # Retrieve nodes for selected documents
            status.update(label="🔍 Loading document content...", state="running")
            all_nodes = []
            total_sources = len(st.session_state.selected_sources)

            for idx, source in enumerate(st.session_state.selected_sources, 1):
                status.update(label=f"📄 Processing {source} ({idx}/{total_sources})", state="running")
                source_filter = MetadataFilters(
                    filters=[MetadataFilter(key='source_name', operator=FilterOperator.EQ, value=source)]
                )
                nodes = await vector_store.aget_nodes(filters=source_filter)
                logger.info(f"Number of nodes for {source}: {len(nodes)}")
                all_nodes.extend(nodes)
                
            if not all_nodes:
                status.update(label="⚠️ No content found in selected documents", state="error")
                return False
                
            status.update(label=f"✅ Loaded {len(all_nodes)} content blocks", state="complete")
            
            # Initialize search index
            status.update(label="🔄 Building search index...", state="running")
            index = VectorStoreIndex(all_nodes, storage_context=storage_context)
            
            # Setup query engine
            status.update(label="🎯 Configuring search engine...", state="running")
            st.session_state.query_engine = index.as_query_engine(
                similarity_top_k=10,
                sparse_top_k=20,
                vector_store_query_mode="hybrid"
            )
            
            status.update(label="✨ Search engine ready!", state="complete")
            st.session_state.query_engine_initialized = True
            return True

    except Exception as e:
        logger.error(f"Error initializing query engine: {str(e)}")
        st.error("❌ Failed to initialize search engine")
        return False

@st.fragment
async def sidebar_content_fragment_PydanticAIAgentChat_response_component():
    try:
        user_input = st.text_area("💭 Ask a question about the selected documents:", key="chat_input_sidebar")

        if user_input:
            with st.spinner("🤔 Processing your question..."):
                # Initialize VectorStoreIndex on the fly for the unified collection

                logger.info(f"User Input: {user_input}")
                logger.info(f"Selected Sources: {st.session_state.selected_sources}")
                logger.info(f"Current Session ID: {st.session_state.get('current_session_id')}")

                source_name_filtered_query_engine_results = asyncio.run(st.session_state.query_engine.aquery(user_input)) 
                
                # Log the response and source nodes
                logger.info(f"Retrieval Pipeline Hybrid Search Source Nodes: {source_name_filtered_query_engine_results}")

                if source_name_filtered_query_engine_results:
                    st.write("### 🔍 Search Results")
                    st.write(source_name_filtered_query_engine_results.response)
                    if source_name_filtered_query_engine_results.source_nodes:
                        st.write("### Sources")
                        for node in source_name_filtered_query_engine_results.source_nodes:
                            with st.expander(f"📄 {node.metadata.get('source_name', 'Unknown Source')}"):
                                st.write(f"**Content:** {node.text}")
                                if node.metadata.get('title'):
                                    st.write(f"**Title:** {node.metadata['title']}")
                                if node.metadata.get('hashtags'):
                                    st.write(f"**Tags:** {', '.join(node.metadata['hashtags'])}")
                else:
                    st.info("ℹ️ No relevant documents found in the selected documents.")

    except Exception as e:
        logger.error(f"Error in document chat component: {str(e)}")
        st.error("An error occurred in the chat interface. Please try again.")

# Initialize Streamlit UI
if __name__ == "__main__":
    asyncio.run(display_file_upload_ui())

    # Add Chat Input on the Sidebar
    with st.sidebar:
        # Initialize session state variables
        for key in ['doc_selection_initialized', 'qdrant_vector_store_initialized','query_engine_initialized', 'selected_sources', 'qdrant_vector_store', 'query_engine']:
            if key not in st.session_state:
                st.session_state[key] = False if key.endswith('initialized') else [] if key == 'selected_sources' else None

        logger.debug(f"Doc selection initialized: {st.session_state.doc_selection_initialized}")
        logger.debug(f"Query engine initialized: {st.session_state.query_engine_initialized}")
        
        # Run document selection component
        recent_selected_sources = asyncio.run(sidebar_doc_selection_component())
        
        # Only run query engine initialization if documents are selected and engine needs initialization
        if recent_selected_sources != st.session_state.selected_sources:
            st.session_state.selected_sources = recent_selected_sources
        
            if not st.session_state.query_engine_initialized:
                asyncio.run(sidebar_query_engine_initialization_component())
        
        if st.session_state.qdrant_vector_store_initialized == True:
            # Track time
            import time

            from openai import OpenAI, AsyncOpenAI
            client = AsyncOpenAI()

            response = asyncio.run(client.embeddings.create(
                input="Hello, world",
                model="text-embedding-3-small"
            ))

            # Test async add
            start_time = time.time()
            test_async_add = asyncio.run(st.session_state.qdrant_vector_store.async_add([
                TextNode(
                text="Hello, world!",
                embedding=response.data[0].embedding,
                metadata={
                    'source_name': "test_source",
                    "index": 0,
                    "title": "Test Document",
                    "hashtags": ["#test", "#hello"],
                    "hypothetical_questions": ["What is the purpose of this document?", "What is the main theme of this document?"],
                    "summary": "This is a test document.",
                    "text_chunk": "Hello, world!"
                }
            )]))
            logger.info(f"Test async add: {test_async_add}\n\n time: {time.time() - start_time}")
            
            # Test async get nodes
            start_time = time.time()
            test_adelete_nodes = asyncio.run(st.session_state.qdrant_vector_store.adelete_nodes(
                filters=MetadataFilters(
                    filters=[MetadataFilter(key='source_name', operator=FilterOperator.EQ, value="test_source")]
                )
            ))
            logger.info(f"Test async delete nodes: {test_adelete_nodes}\n\n time: {time.time() - start_time}")

            # Test async get nodes
            start_time = time.time()
            query_str = "What is the purpose of this document?"
            query_str_embedding = asyncio.run(client.embeddings.create(
                input=query_str,
                model="text-embedding-3-small"
            )).data[0].embedding
            
            from llama_index.core.vector_stores import VectorStoreQuery
            query = VectorStoreQuery(
                query_str=query_str,
                query_embedding=query_str_embedding,
                similarity_top_k=2,
                sparse_top_k=12,
                mode="hybrid"
            )
            test_aquery = asyncio.run(st.session_state.qdrant_vector_store.aquery(
                query = query, 
                filters=MetadataFilters(
                    filters=[MetadataFilter(key='source_name', operator=FilterOperator.EQ, value="test_source")]
                )))
            logger.info(f"Test aquery: {test_aquery} \n\n time: {time.time() - start_time}")
            
            # Display query results in Streamlit
            st.subheader("🔍 Query Results")
            
            # Show query parameters
            with st.expander("Query Details"):
                st.write("**Query String:**", query_str)
                st.write("**Top K:**", query.similarity_top_k)
                st.write("**Mode:**", query.mode)
            
            # Display results
            if test_aquery and test_aquery.nodes:
                st.write(f"Found {len(test_aquery.nodes)} relevant documents")
                
                for i, (node, similarity) in enumerate(zip(test_aquery.nodes, test_aquery.similarities)):
                    with st.expander(f"📄 Result {i+1} (Similarity: {similarity:.3f})"):
                        st.write("**Source:**", node.metadata.get('source_name', 'Unknown'))
                        st.write("**Content:**")
                        st.markdown(node.text)
                        
                        # Show metadata
                        st.write("**Metadata:**")
                        metadata_df = pd.DataFrame([node.metadata]).T
                        st.dataframe(metadata_df)
            else:
                st.info("No results found for the query.")

            
            
        # Show chat interface only when everything is initialized
        if st.session_state.doc_selection_initialized and st.session_state.query_engine_initialized:
            asyncio.run(sidebar_content_fragment_PydanticAIAgentChat_response_component())