"""
Main UI module for handling file uploads and user interactions.
"""

import logging
import streamlit as st
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from .method_recommendation import (
    ProcessingMethod,
    get_method_recommendation,
    get_predefined_method_recommendation,
    get_method_display_name
)
from .file_processing import (
    ProcessingConfig,
    process_files,
    FileSummary
)
from .retrieval_and_chat import (
    retrieval_pipeline_hybrid_search,
    get_unique_document_sources,
    chat_with_documents
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def update_file_status(filename: str, status: str, progress: Optional[float] = None):
    """Update the status of a file being processed."""
    if "file_status" not in st.session_state:
        st.session_state.file_status = {}
    
    st.session_state.file_status[filename] = {
        "status": status,
        "progress": progress if progress is not None else 0.0
    }

def check_already_processed_files(uploaded_files: List[Any]) -> Tuple[List[str], List[Any]]:
    """
    Check which files have already been processed.
    
    Args:
        uploaded_files: List of uploaded file objects
        
    Returns:
        Tuple containing:
        - List of already processed file names
        - List of new files to process
    """
    already_processed = []
    new_files = []
    processed_sources = st.session_state.get('all_unique_document_sources', set())
    
    for file in uploaded_files:
        if file.name in processed_sources:
            already_processed.append(file.name)
        else:
            new_files.append(file)
            
    return already_processed, new_files

async def display_file_upload_ui():
    """Display the file upload UI and handle document processing."""
    st.title("Document Processing System")
    
    # File upload section
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files to process",
        accept_multiple_files=True,
        type=["pdf", "docx", "doc", "txt", "csv", "xlsx", "jpg", "png", "jpeg"]
    )
    
    if not uploaded_files:
        st.info("Please upload some documents to begin.")
        return
        
    # Processing method selection
    st.subheader("Select Processing Method")
    complexity_options = [
        "Simple text-based document",
        "Complex document with tables and formatting",
        "Document with diagrams and images"
    ]
    selected_option = st.selectbox(
        "Choose document complexity",
        options=complexity_options
    )
    
    # Get processing method
    method, explanation = get_predefined_method_recommendation(selected_option)
    st.write(f"Recommended method: {get_method_display_name(method)}")
    st.write(f"Reason: {explanation}")
    
    # Process button
    if st.button("Process Documents"):
        try:
            # Check for already processed files
            already_processed, new_files = check_already_processed_files(uploaded_files)
            if already_processed:
                st.warning(f"The following files have already been processed: {', '.join(already_processed)}")
            
            if not new_files:
                st.info("No new files to process.")
                return
            
            # Initialize progress tracking
            progress_bar = st.progress(0)
            status_container = st.empty()
            
            # Create processing config
            config = ProcessingConfig(
                processing_method=method,
                session_id=st.session_state.get("current_session_id")
            )
            
            # Process files
            with st.spinner("Processing documents..."):
                processed_files = await process_files(
                    new_files,
                    config,
                    update_file_status,
                    progress_bar
                )
                
                # Display results
                st.success("Processing complete!")
                for filename, summary in processed_files.items():
                    with st.expander(f"Summary for {filename}"):
                        st.write(f"Total chunks: {summary.total_chunks}")
                        st.write(f"Document type: {summary.document_type}")
                        st.write("Key points:")
                        for point in summary.key_points:
                            st.write(f"- {point}")
                
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            logger.error(f"Document processing error: {str(e)}")

def display_error_message(error: Exception, context: str = "") -> None:
    """
    Display a user-friendly error message in the Streamlit UI.
    
    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
    """
    if isinstance(error, RateLimitError):
        st.error("🚫 Rate limit exceeded. Please try again in a few minutes.")
    elif isinstance(error, ProcessingError):
        st.error(f"❌ Processing error: {str(error)}")
    else:
        error_msg = f"An unexpected error occurred{f' during {context}' if context else ''}"
        st.error(f"⚠️ {error_msg}: {str(error)}")
        logger.error(f"Error {context}: {str(error)}", exc_info=True)

def display_success_message(message: str, icon: str = "✅") -> None:
    """
    Display a success message in the Streamlit UI.
    
    Args:
        message: Success message to display
        icon: Optional icon to show before the message
    """
    st.success(f"{icon} {message}")

async def sidebar_content_fragment_PydanticAIAgentChat_component():
    """Streamlit component for document chat interface with async document selection."""
    st.sidebar.header("Document Chat")
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Get unique document sources
    if "qdrant_client" in st.session_state:
        sources = await get_unique_document_sources(st.session_state.qdrant_client)
        
        # Document selection
        selected_docs = st.sidebar.multiselect(
            "Filter by documents",
            options=sources,
            default=None,
            help="Select specific documents to chat with"
        )
        
        # Chat input
        with st.container():
            # Display chat history
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.write(f"You: {message['content']}")
                else:
                    st.write(f"Assistant: {message['content']}")
            
            # Chat input
            chat_input = st.text_input("Ask a question about your documents")
            
            if chat_input:
                try:
                    # Add user message to history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": chat_input
                    })
                    
                    # Get chat response
                    response = await chat_with_documents(
                        query=chat_input,
                        index=st.session_state.index,
                        selected_documents=selected_docs if selected_docs else None,
                        chat_history=[
                            {"user": msg["content"], "assistant": msg.get("response", "")}
                            for msg in st.session_state.chat_history
                            if msg["role"] == "user"
                        ]
                    )
                    
                    if response["success"]:
                        # Add assistant response to history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response["response"],
                            "context": response["context"]
                        })
                        
                        # Display sources used
                        with st.expander("View sources"):
                            for result in response["context"]["search_results"]:
                                st.write(f"From {result['source']}:")
                                st.write(result["text"])
                                st.write("---")
                    else:
                        st.error(f"Error: {response.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"Error processing chat: {str(e)}")
                    logger.error(f"Chat error: {str(e)}")

async def sidebar_direct_search_component():
    """Streamlit component for direct search on the Qdrant collection."""
    st.sidebar.header("Document Search")
    
    # Get unique document sources
    if "qdrant_client" in st.session_state:
        sources = await get_unique_document_sources(st.session_state.qdrant_client)
        
        # Document selection
        selected_docs = st.sidebar.multiselect(
            "Filter by documents",
            options=sources,
            default=None,
            help="Select specific documents to search within"
        )
        
        # Search input
        search_query = st.sidebar.text_input("Search documents")
        
        if search_query:
            try:
                # Perform search
                results = await retrieval_pipeline_hybrid_search(
                    query=search_query,
                    index=st.session_state.index,
                    selected_documents=selected_docs if selected_docs else None
                )
                
                if results["success"]:
                    st.sidebar.write(f"Found {len(results['results'])} results")
                    
                    # Display results
                    for result in results["results"]:
                        with st.sidebar.expander(f"From {result['source']}"):
                            st.write(result["text"])
                            st.write(f"Score: {result.get('score', 'N/A')}")
                            
                    # Display filter information
                    if results.get("filters_applied"):
                        with st.sidebar.expander("Search filters"):
                            st.write("Applied filters:")
                            st.json(results["filters_applied"])
                            st.write("Reasoning:")
                            st.write(results.get("filter_reasoning", "No reasoning provided"))
                else:
                    st.sidebar.error(f"Search error: {results.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.sidebar.error(f"Error performing search: {str(e)}")
                logger.error(f"Search error: {str(e)}")

# Initialize Streamlit UI
if __name__ == "__main__":
    asyncio.run(display_file_upload_ui())
    asyncio.run(sidebar_content_fragment_PydanticAIAgentChat_component())
    asyncio.run(sidebar_direct_search_component())
