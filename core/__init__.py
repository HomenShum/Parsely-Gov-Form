"""Core functionality for Parsely document processing system."""

from .config import Config
from .parsers import Parser
from .data_gatherer import DataGatherer

__all__ = [
    'Config',
    'Parser',
    'DataGatherer',
]

# Version information
__version__ = '1.0.0'
__author__ = 'Homen Shum'