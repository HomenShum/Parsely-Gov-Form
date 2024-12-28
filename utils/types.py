"""
Common type definitions for the document processing system.
"""

from typing import Dict, List, Optional, Union, Any, TypeVar, Callable, Awaitable
from typing_extensions import TypedDict, Protocol
from datetime import datetime
from pydantic import BaseModel

# Type aliases
FileContent = Union[str, bytes]
Metadata = Dict[str, Any]
ProgressCallback = Callable[[str, float], None]
AsyncFunc = TypeVar('AsyncFunc', bound=Callable[..., Awaitable[Any]])

# Structured types
class FileStatus(TypedDict):
    """File processing status information."""
    status: str
    message: str
    progress: float
    state: str
    timestamp: datetime

class ProcessingSummary(TypedDict):
    """Summary of file processing results."""
    successful: List[Dict[str, Any]]
    failed: List[Dict[str, str]]
    total_chunks: int
    total_files: int
    processing_time: float

class SearchResult(BaseModel):
    """Search result information."""
    text: str
    metadata: Metadata
    score: Optional[float] = None
    source: str
    chunk_index: int
    file_type: str

class ChatMessage(BaseModel):
    """Chat message structure."""
    role: str
    content: str
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None

# Protocol classes
class DocumentProcessor(Protocol):
    """Protocol for document processors."""
    async def process(
        self,
        content: FileContent,
        filename: str,
        **kwargs: Any
    ) -> Dict[str, Any]: ...

class MetadataGenerator(Protocol):
    """Protocol for metadata generators."""
    async def generate(
        self,
        text: str,
        **kwargs: Any
    ) -> Metadata: ...
