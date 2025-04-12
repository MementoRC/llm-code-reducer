"""
Code Reducer MCP Server.

A Model Context Protocol server providing tools for reducing code context to minimize token usage with LLMs.
"""

__version__ = "0.1.0"

from .__main__ import main
from .server import serve
from .models import (
    Position,
    MappingData,
    ProcessedResult,
    ProcessedFileRecord,
    TransformationType,
    TranslatePositionRequest,
    TranslatedPosition,
)
from .database import CodeReducerDatabase