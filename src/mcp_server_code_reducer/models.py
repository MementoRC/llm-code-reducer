"""
Data models for the Code Reducer MCP Server.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel, Field
from datetime import datetime


class Position(BaseModel):
    """Represents a position in a source file."""
    line: int
    character: int

    def __str__(self) -> str:
        return f"{self.line}:{self.character}"


class MappingData(BaseModel):
    """Bidirectional mapping between positions in original and processed code."""
    original_to_processed: Dict[str, Position]
    processed_to_original: Dict[str, Position]


class TransformationType(str, Enum):
    """Types of transformations that can be applied to code."""
    COMMENTS = "comments"
    TYPE_HINTS = "type_hints"
    DOCSTRINGS = "docstrings"
    DEBUG_FUNCTIONS = "debug_functions"
    WHITESPACE = "whitespace"
    UNUSED_IMPORTS = "unused_imports"
    FUNCTION_SIGNATURES = "function_signatures"
    CONDITIONALS = "conditionals"
    MINIFY_NAMES = "minify_names"


class ProcessedResult(BaseModel):
    """Result model containing processed code and statistics."""
    processed_content: str
    original_lines: int
    processed_lines: int
    reduction_percentage: float
    mapping_data: MappingData
    file_id: Optional[str] = None
    transformations: List[str] = []


class ProcessedFileRecord(BaseModel):
    """Database record for a processed file."""
    file_id: str
    original_content: str
    processed_content: str
    file_name: Optional[str]
    original_lines: int
    processed_lines: int
    reduction_percentage: float
    transformations: List[str]
    mapping_data: MappingData
    created_at: str
    updated_at: str


class ProcessPythonRequest(BaseModel):
    """Request model for processing Python code."""
    content: str
    file_name: Optional[str] = None
    strip_comments: bool = False
    strip_docstrings: bool = False
    strip_type_hints: bool = True
    strip_debug: bool = False
    debug_functions: Optional[List[str]] = None
    optimize_whitespace: bool = True
    remove_unused_imports: bool = False
    optimize_function_signatures: bool = False
    optimize_conditionals: bool = False
    minify_names: bool = False


class TranslatePositionRequest(BaseModel):
    """Request model for translating positions between original and processed code."""
    file_id: str
    position: Position
    to_original: bool = True


class TranslatedPosition(BaseModel):
    """Result model containing translated position information."""
    original_position: Optional[Position]
    processed_position: Optional[Position]
    file_id: str


# Context Management Models

class FileRelationship(str, Enum):
    """Types of relationships between files in a context."""
    IMPORTS = "imports"
    IMPORTED_BY = "imported_by"
    REFERENCES = "references"
    REFERENCED_BY = "referenced_by"
    EXTENDS = "extends"
    EXTENDED_BY = "extended_by"
    IMPLEMENTS = "implements"
    IMPLEMENTED_BY = "implemented_by"


class ContextFileMetadata(BaseModel):
    """Metadata for a file within a context."""
    file_id: str
    file_path: str
    file_name: str
    importance: float = 1.0  # Higher values indicate more important files
    relationships: Dict[str, List[FileRelationship]] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class ContextConfiguration(BaseModel):
    """Configuration for a context."""
    strip_comments: bool = False
    strip_docstrings: bool = False
    strip_type_hints: bool = True
    max_files_per_query: int = 10
    prioritize_by_importance: bool = True
    include_summaries: bool = False
    token_budget: Optional[int] = None


class ContextSummary(BaseModel):
    """Summary information about a context."""
    total_files: int
    total_original_size: int
    total_processed_size: int
    overall_reduction_percentage: float
    file_types: Dict[str, int]  # Count of each file extension
    most_important_files: List[str]  # List of most important file_ids


class Context(BaseModel):
    """Represents a group of related files as a context."""
    context_id: str
    name: str
    description: Optional[str] = None
    files: Dict[str, ContextFileMetadata] = Field(default_factory=dict)
    configuration: ContextConfiguration = Field(default_factory=ContextConfiguration)
    created_at: str
    updated_at: str
    summary: Optional[ContextSummary] = None


class CreateContextRequest(BaseModel):
    """Request model for creating a new context."""
    name: str
    description: Optional[str] = None
    configuration: Optional[ContextConfiguration] = None


class AddFilesToContextRequest(BaseModel):
    """Request model for adding files to a context."""
    context_id: str
    files: List[Dict[str, Any]]  # List of files with content and metadata


class UpdateContextConfigRequest(BaseModel):
    """Request model for updating context configuration."""
    context_id: str
    configuration: ContextConfiguration


class GetContextFilesRequest(BaseModel):
    """Request model for getting files in a context."""
    context_id: str
    limit: int = 10
    offset: int = 0
    importance_threshold: Optional[float] = None
    include_content: bool = False
    include_relationships: bool = True