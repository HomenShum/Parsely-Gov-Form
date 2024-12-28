"""
Logging configuration for the document processing system.
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

def setup_logging(module_name: str) -> logging.Logger:
    """
    Set up logging for a module with file and console handlers.
    
    Args:
        module_name: Name of the module for logger
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler - Rotating file handler with date in filename
    log_file = logs_dir / f"{module_name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_logger(module_name: str) -> logging.Logger:
    """
    Get or create a logger for a module.
    
    Args:
        module_name: Name of the module
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(module_name)
    if not logger.handlers:
        logger = setup_logging(module_name)
    return logger
