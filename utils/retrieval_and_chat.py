"""
Retrieval and chat module for handling document search and chat interactions.
"""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from .qdrant_filter_agent import generate_qdrant_filters, QdrantFilterDependencies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

USERNAME = "parselyai"
UNIFIED_COLLECTION_NAME = f"parsely_hybrid_search_index_{USERNAME}"

async def retrieval_pipeline_hybrid_search(
    query: str,
    index: VectorStoreIndex,
    selected_documents: Optional[List[str]] = None,
    limit: int = 5,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Performs a hybrid search with Qdrant filters based on selected documents and inferred metadata filters.
    
    Args:
        query: User's search query
        index: VectorStoreIndex instance
        selected_documents: Optional list of document names to filter by
        limit: Maximum number of results to return
        session_id: Optional session identifier
        
    Returns:
        Dictionary containing search results and metadata
    """
    try:
        # Get available metadata keys from the index
        nodes = index.as_retriever().retrieve(query)
        available_metadata_keys = set()
        for node in nodes:
            if isinstance(node, TextNode) and node.metadata:
                available_metadata_keys.update(node.metadata.keys())
        
        # Generate Qdrant filters
        filter_deps = QdrantFilterDependencies(
            user_query=query,
            selected_documents=selected_documents,
            limit=limit,
            session_id=session_id,
            available_metadata_keys=list(available_metadata_keys)
        )
        
        filters = await generate_qdrant_filters(filter_deps)
        metadata_filters = filters.to_metadata_filters() if filters.filters else None
        
        # Log the generated filters
        logger.info(f"Generated filters for query '{query}': {filters}")
        logger.info(f"Filter reasoning: {filters.reasoning}")
        
        # Apply filters to the retriever
        retriever = index.as_retriever(
            similarity_top_k=limit,
            filters=metadata_filters
        )
        
        # Perform the search
        nodes = retriever.retrieve(query)
        
        # Process and return results
        results = []
        for node in nodes:
            if isinstance(node, TextNode):
                result = {
                    "text": node.text,
                    "metadata": node.metadata,
                    "score": getattr(node, "score", None),
                    "source": node.metadata.get("source", "Unknown"),
                    "chunk_index": node.metadata.get("chunk_index", 0),
                    "file_type": node.metadata.get("file_type", "unknown")
                }
                results.append(result)
        
        return {
            "success": True,
            "results": results,
            "filters_applied": filters.dict() if filters.filters else None,
            "filter_reasoning": filters.reasoning,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in hybrid search: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "filters_applied": None,
            "total_results": 0
        }

async def get_unique_document_sources(qdrant_client: AsyncQdrantClient) -> List[str]:
    """
    Retrieves unique 'source' values from the 'metadata' field in the unified collection.
    
    Args:
        qdrant_client: AsyncQdrantClient instance
        
    Returns:
        List of unique document sources
    """
    try:
        # Scroll through all points to get unique sources
        sources = set()
        offset = None
        limit = 100
        
        while True:
            response = await qdrant_client.scroll(
                collection_name=UNIFIED_COLLECTION_NAME,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not response.points:
                break
                
            for point in response.points:
                if point.payload and "source" in point.payload:
                    sources.add(point.payload["source"])
                    
            if len(response.points) < limit:
                break
                
            offset = response.next_page_offset
            
        return sorted(list(sources))
        
    except Exception as e:
        logger.error(f"Error getting unique document sources: {str(e)}")
        return []

async def chat_with_documents(
    query: str,
    index: VectorStoreIndex,
    selected_documents: Optional[List[str]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Chat with documents using hybrid search and conversation history.
    
    Args:
        query: User's chat query
        index: VectorStoreIndex instance
        selected_documents: Optional list of document names to filter by
        chat_history: Optional list of previous chat messages
        session_id: Optional session identifier
        
    Returns:
        Dictionary containing chat response and relevant context
    """
    try:
        # Get relevant context using hybrid search
        search_results = await retrieval_pipeline_hybrid_search(
            query=query,
            index=index,
            selected_documents=selected_documents,
            limit=5,
            session_id=session_id
        )
        
        if not search_results["success"]:
            return {
                "success": False,
                "error": search_results["error"],
                "response": None,
                "context": None
            }
        
        # Format context from search results
        context = "\n\n".join([
            f"From {result['source']}:\n{result['text']}"
            for result in search_results["results"]
        ])
        
        # Format chat history if provided
        formatted_history = ""
        if chat_history:
            formatted_history = "\n".join([
                f"User: {msg['user']}\nAssistant: {msg['assistant']}"
                for msg in chat_history[-5:]  # Use last 5 messages for context
            ])
        
        # Generate chat response using the context
        chat_prompt = f"""Based on the following context and chat history, provide a helpful response to the user's query.

Context:
{context}

Chat History:
{formatted_history}

User Query: {query}

Please provide a clear and concise response that directly addresses the user's query while incorporating relevant information from the context."""
        
        # Use your preferred chat model to generate the response
        # For example, using OpenAI's GPT model
        response = "Generated response based on context and query"  # Replace with actual chat model call
        
        return {
            "success": True,
            "response": response,
            "context": {
                "search_results": search_results["results"],
                "filters_applied": search_results["filters_applied"],
                "filter_reasoning": search_results.get("filter_reasoning")
            }
        }
        
    except Exception as e:
        logger.error(f"Error in chat with documents: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "response": None,
            "context": None
        }
