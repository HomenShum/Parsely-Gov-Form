"""
Common constants and configurations for the document processing system.
"""

import streamlit as st
from openai import AsyncOpenAI
from enum import Enum

# User and collection configuration
USERNAME = "parselyai"
UNIFIED_COLLECTION_NAME = f"parsely_hybrid_search_index_{USERNAME}"

# File type categories
FILE_CATEGORIES = {
    "pdf": ["pdf", "docx", "doc", "odt", "pptx", "ppt"],
    "image": ["png", "jpg", "jpeg"],
    "excel": ["xlsx", "xls"],
    "csv": ["csv"],
    "other": ["txt", "json", "xml", "html", "md"]
}

# Initialize OpenAI client in session state
if "openai_client" not in st.session_state:
    st.session_state.openai_client = AsyncOpenAI(
        api_key=st.secrets.get("OPENAI_API_KEY")
    )

# Common logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
}

# API endpoints and configuration
LLAMA_PARSE_API_URL = "https://api.llamaparse.com/v1/parse"
COLPALI_API_URL = "https://api.colpali.ai/v1/process"

# Default processing parameters
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
MAX_CONCURRENT_REQUESTS = 5
RATE_LIMIT_DELAY = 1  # seconds
