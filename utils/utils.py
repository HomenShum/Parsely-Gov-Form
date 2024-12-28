"""
Common utility functions for the document processing system.
"""

import asyncio
from functools import wraps
import logging
from typing import Any, Callable, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def retry_async(retries: int = 3, delay: int = 1):
    """
    Retry decorator for async functions.
    
    Args:
        retries: Number of retries
        delay: Delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {retries} attempts failed. Last error: {str(e)}")
            raise last_exception
        return wrapper
    return decorator

def get_current_timestamp() -> str:
    """Get current timestamp in ISO format with timezone."""
    return datetime.now(timezone.utc).isoformat()

def create_unique_key(source_name: str, index: int, session_id: Optional[str] = None) -> str:
    """Create a unique key for document tracking."""
    base_key = f"{source_name}_{index}"
    return f"{base_key}_{session_id}" if session_id else base_key

def format_file_size(size_bytes: int) -> str:
    """Format file size in bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def safe_get_extension(filename: str) -> str:
    """Safely get file extension in lowercase."""
    try:
        return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    except Exception:
        return ''

def update_progress(
    progress_bar: Any,
    status_text: str,
    progress: float,
    update_func: Optional[Callable] = None
) -> None:
    """Update progress bar and status text."""
    if progress_bar is not None:
        progress_bar.progress(progress)
    if update_func is not None:
        update_func(status_text)
    logger.info(f"Progress: {progress:.1%} - {status_text}")
