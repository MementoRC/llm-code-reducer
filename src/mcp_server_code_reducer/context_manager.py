"""
Context Manager Service for Code Reducer MCP Server.

This module provides a service for managing code contexts, allowing grouping
of related files and handling their relationships to provide more efficient
token usage in LLM interactions.
"""

import logging
import os
import ast
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timezone
from pathlib import Path

from .models import (
    Context,
    ContextConfiguration,
    ContextFileMetadata,
    ContextSummary,
    FileRelationship,
    ProcessedResult
)
from .database import CodeReducerDatabase

logger = logging.getLogger("mcp_context_manager")


class ContextManager:
    """
    Service for managing code contexts within the Code Reducer.
    
    This service provides functionality to:
    1. Create and manage contexts (groups of related files)
    2. Analyze file relationships
    3. Generate context summaries
    4. Prioritize files within contexts
    5. Select optimal files for a given token budget
    """
    
    def __init__(self, db: CodeReducerDatabase):
        """
        Initialize the Context Manager.
        
        Args:
            db: Database manager for storing context data
        """
        self.db = db
        
    async def create_context(
        self,
        name: str,
        description: Optional[str] = None,
        configuration: Optional[ContextConfiguration] = None
    ) -> Optional[str]:
        """
        Create a new context for managing related files.
        
        Args:
            name: Name of the context
            description: Optional description
            configuration: Optional configuration settings
            
        Returns:
            str: Context ID if created successfully, None otherwise
        """
        return self.db.create_context(name, description, configuration)
    
    async def add_file_to_context(
        self,
        context_id: str,
        file_id: str,
        file_path: str,
        file_name: str,
        importance: float = 1.0,
        tags: Optional[List[str]] = None,
        analyze_relationships: bool = True
    ) -> bool:
        """
        Add a processed file to a context with optional relationship analysis.
        
        Args:
            context_id: ID of the context
            file_id: ID of the processed file
            file_path: Path of the file in its original location
            file_name: Name of the file
            importance: Importance score for this file
            tags: Optional list of tags for categorization
            analyze_relationships: Whether to analyze and store file relationships
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        # Add the file to the context
        success = self.db.add_file_to_context(
            context_id, file_id, file_path, file_name, importance, tags
        )
        
        if not success:
            return False
            
        # If relationship analysis is requested
        if analyze_relationships:
            await self._analyze_file_relationships(context_id, file_id)
            
        # Update context summary
        await self._update_context_summary(context_id)
        
        return True
    
    async def _analyze_file_relationships(self, context_id: str, file_id: str) -> None:
        """
        Analyze relationships between the given file and other files in the context.
        
        Args:
            context_id: ID of the context
            file_id: ID of the file to analyze
        """
        # Get the file record
        file_record = self.db.get_processed_file(file_id)
        if not file_record:
            logger.error(f"File not found for relationship analysis: {file_id}")
            return
            
        # Get other files in the context
        context_files = self.db.get_context_files(context_id, limit=1000, include_content=True)
        
        if not context_files:
            return
            
        # Skip if this is the only file
        if len(context_files) <= 1:
            return
            
        # For Python files, try to analyze imports and other relationships
        if file_record.file_name and file_record.file_name.endswith('.py'):
            try:
                # Parse the original content as Python code
                tree = ast.parse(file_record.original_content)
                
                # Analyze imports
                imports = self._extract_python_imports(tree)
                
                # For each import, find matching files in the context
                for import_name in imports:
                    # Find files that match this import
                    for other_file in context_files:
                        # Skip self
                        if other_file['file_id'] == file_id:
                            continue
                            
                        # Check if this file matches the import
                        if self._file_matches_import(other_file, import_name):
                            # Add import relationship
                            self.db.add_file_relationship(
                                context_id,
                                file_id,
                                other_file['file_id'],
                                FileRelationship.IMPORTS.value
                            )
                            
                            # Add imported_by relationship in the other direction
                            self.db.add_file_relationship(
                                context_id,
                                other_file['file_id'],
                                file_id,
                                FileRelationship.IMPORTED_BY.value
                            )
            except SyntaxError:
                logger.warning(f"Syntax error when analyzing imports in file: {file_id}")
            except Exception as e:
                logger.error(f"Error analyzing file relationships: {str(e)}")
    
    def _extract_python_imports(self, tree: ast.Module) -> Set[str]:
        """
        Extract import names from a Python AST.
        
        Args:
            tree: AST of Python code
            
        Returns:
            Set of import names
        """
        imports = set()
        
        for node in ast.walk(tree):
            # Handle 'import module' statements
            if isinstance(node, ast.Import):
                for name in node.names:
                    # Just get the top-level module
                    module_parts = name.name.split('.')
                    imports.add(module_parts[0])
                    
                    # Also add the full import path
                    imports.add(name.name)
            
            # Handle 'from module import name' statements
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Just get the top-level module
                    module_parts = node.module.split('.')
                    imports.add(module_parts[0])
                    
                    # Also add the full import path
                    imports.add(node.module)
        
        return imports
    
    def _file_matches_import(self, file_data: Dict[str, Any], import_name: str) -> bool:
        """
        Check if a file matches an import name.
        
        Args:
            file_data: File record from get_context_files
            import_name: Import name to check
            
        Returns:
            True if the file matches the import, False otherwise
        """
        file_path = file_data['file_path']
        file_name = file_data['file_name']
        
        # Convert file path to potential module path
        path_parts = Path(file_path).parts
        if file_name.endswith('.py'):
            base_name = file_name[:-3]  # Remove .py extension
        else:
            base_name = file_name
        
        # Check different ways the file could match the import
        
        # Direct match with file name (e.g., "utils.py" matches "utils")
        if base_name == import_name:
            return True
            
        # Match as a module path (e.g., "mypackage/utils.py" matches "mypackage.utils")
        # Construct a dotted module path from the file path
        for i in range(len(path_parts)):
            module_path = '.'.join(path_parts[i:] + (base_name,))
            if module_path == import_name:
                return True
                
            # Also check just the last part to handle relative imports
            if i == len(path_parts) - 1 and base_name == import_name:
                return True
                
        return False
    
    async def _update_context_summary(self, context_id: str) -> None:
        """
        Generate and update the summary for a context.
        
        Args:
            context_id: ID of the context to update
        """
        try:
            # Get all files in the context
            files = self.db.get_context_files(context_id, limit=1000)
            
            if not files:
                return
                
            # Initialize summary values
            total_files = len(files)
            total_original_size = 0
            total_processed_size = 0
            file_types = {}
            
            # Track file IDs and importance scores for finding most important files
            importance_scores = {}
            
            # Process each file
            for file_data in files:
                file_id = file_data['file_id']
                importance_scores[file_id] = file_data['importance']
                
                # Get file extension
                file_name = file_data['file_name'] or ''
                file_ext = os.path.splitext(file_name)[1]
                if file_ext:
                    file_ext = file_ext[1:]  # Remove the dot
                    
                # Count file types
                if file_ext:
                    file_types[file_ext] = file_types.get(file_ext, 0) + 1
                
                # Get file record for size information
                file_record = self.db.get_processed_file(file_id)
                if file_record:
                    total_original_size += len(file_record.original_content)
                    total_processed_size += len(file_record.processed_content)
            
            # Calculate overall reduction percentage
            if total_original_size == 0:
                overall_reduction = 0.0
            else:
                overall_reduction = (total_original_size - total_processed_size) / total_original_size * 100
                
            # Sort files by importance and get top files
            most_important = sorted(
                importance_scores.keys(),
                key=lambda k: importance_scores[k],
                reverse=True
            )[:5]  # Get top 5 most important files
            
            # Create summary
            summary = ContextSummary(
                total_files=total_files,
                total_original_size=total_original_size,
                total_processed_size=total_processed_size,
                overall_reduction_percentage=round(overall_reduction, 2),
                file_types=file_types,
                most_important_files=most_important
            )
            
            # Update in database
            self.db.update_context_summary(context_id, summary)
            
        except Exception as e:
            logger.error(f"Error updating context summary: {str(e)}")
    
    async def select_files_for_context(
        self,
        context_id: str,
        token_budget: Optional[int] = None,
        importance_threshold: Optional[float] = None,
        max_files: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Select files from a context optimized for a token budget.
        
        Args:
            context_id: ID of the context
            token_budget: Maximum number of tokens to use
            importance_threshold: Minimum importance score for included files
            max_files: Maximum number of files to include
            
        Returns:
            List of file records
        """
        try:
            # Get context to check configuration
            context = self.db.get_context(context_id)
            if not context:
                logger.error(f"Context not found: {context_id}")
                return []
                
            # Use context configuration if parameters not specified
            if token_budget is None and context.configuration.token_budget:
                token_budget = context.configuration.token_budget
                
            if max_files is None:
                max_files = context.configuration.max_files_per_query
                
            # Get all files in the context
            all_files = self.db.get_context_files(
                context_id, 
                limit=1000,
                importance_threshold=importance_threshold,
                include_content=True
            )
            
            if not all_files:
                return []
                
            # If we have a token budget, we need to select files to fit
            if token_budget:
                return self._select_files_by_token_budget(
                    all_files, token_budget, max_files, context.configuration.prioritize_by_importance
                )
            
            # Otherwise, just return up to max_files
            if max_files and len(all_files) > max_files:
                # Sort by importance first if configured
                if context.configuration.prioritize_by_importance:
                    all_files.sort(key=lambda f: f['importance'], reverse=True)
                return all_files[:max_files]
                
            return all_files
            
        except Exception as e:
            logger.error(f"Error selecting files for context: {str(e)}")
            return []
    
    def _select_files_by_token_budget(
        self,
        files: List[Dict[str, Any]],
        token_budget: int,
        max_files: int,
        prioritize_by_importance: bool
    ) -> List[Dict[str, Any]]:
        """
        Select files to fit within a token budget.
        
        Args:
            files: List of file records with content
            token_budget: Maximum number of tokens
            max_files: Maximum number of files
            prioritize_by_importance: Whether to prioritize by importance score
            
        Returns:
            List of selected files
        """
        # Sort files by priority
        if prioritize_by_importance:
            files.sort(key=lambda f: f['importance'], reverse=True)
            
        # Estimate tokens using a simple approximation
        # (more sophisticated methods would use a tokenizer)
        def estimate_tokens(text: str) -> int:
            # Simple approximation: ~4 characters per token on average
            return len(text) // 4
            
        selected_files = []
        total_tokens = 0
        
        for file in files:
            if 'processed_content' not in file:
                continue
                
            # Estimate tokens for this file
            file_tokens = estimate_tokens(file['processed_content'])
            
            # If adding this file would exceed budget, skip it
            if total_tokens + file_tokens > token_budget:
                continue
                
            # Add the file
            selected_files.append(file)
            total_tokens += file_tokens
            
            # If we've reached max files, stop
            if max_files and len(selected_files) >= max_files:
                break
                
        return selected_files
    
    async def analyze_file_importance(self, context_id: str) -> bool:
        """
        Analyze and update importance scores for files in a context.
        
        Args:
            context_id: ID of the context
            
        Returns:
            bool: True if analysis succeeded, False otherwise
        """
        try:
            # Get all files in the context with relationships
            files = self.db.get_context_files(context_id, limit=1000)
            
            if not files:
                return False
                
            # Count relationships for centrality measurement
            relationship_counts = {}
            
            for file in files:
                file_id = file['file_id']
                relationship_counts[file_id] = 0
                
                # Count relationships
                if 'relationships' in file:
                    for target_id, rel_types in file['relationships'].items():
                        relationship_counts[file_id] += len(rel_types)
            
            # Find max relationship count for normalization
            max_count = max(relationship_counts.values()) if relationship_counts else 1
            if max_count == 0:  # Avoid division by zero
                max_count = 1
                
            # Calculate importance based on relationship centrality
            # High centrality = important in the dependency graph
            for file_id, count in relationship_counts.items():
                # Base importance (0.5 to 2.0)
                base_importance = 0.5
                
                # Add relationship bonus (up to 1.5)
                relationship_bonus = (count / max_count) * 1.5
                
                # Calculate final importance score
                importance = base_importance + relationship_bonus
                
                # Update in database
                self.db.update_file_importance(context_id, file_id, importance)
                
            return True
            
        except Exception as e:
            logger.error(f"Error analyzing file importance: {str(e)}")
            return False
    
    async def generate_context_summary_text(self, context_id: str) -> Optional[str]:
        """
        Generate a human-readable text summary of the context.
        
        Args:
            context_id: ID of the context
            
        Returns:
            str: Summary text or None if failed
        """
        try:
            # Get context with summary
            context = self.db.get_context(context_id)
            if not context:
                return None
                
            # If no summary available, update it
            if not context.summary:
                await self._update_context_summary(context_id)
                context = self.db.get_context(context_id)
                if not context or not context.summary:
                    return None
            
            # Build summary text
            summary_lines = [
                f"Context: {context.name}",
                "-" * 40
            ]
            
            if context.description:
                summary_lines.append(f"Description: {context.description}")
                summary_lines.append("")
            
            summary = context.summary
            summary_lines.extend([
                f"Total Files: {summary.total_files}",
                f"Total Size: {summary.total_original_size:,} chars (original), {summary.total_processed_size:,} chars (processed)",
                f"Token Reduction: {summary.overall_reduction_percentage:.2f}%",
                ""
            ])
            
            # File types
            if summary.file_types:
                summary_lines.append("File Types:")
                for ext, count in summary.file_types.items():
                    summary_lines.append(f"  {ext}: {count} files")
                summary_lines.append("")
            
            # Most important files
            if summary.most_important_files:
                summary_lines.append("Key Files:")
                for file_id in summary.most_important_files:
                    if file_id in context.files:
                        file = context.files[file_id]
                        summary_lines.append(f"  {file.file_path} (importance: {file.importance:.2f})")
                        
                        # Show some relationships
                        if file.relationships:
                            rel_count = sum(len(rels) for rels in file.relationships.values())
                            if rel_count > 0:
                                summary_lines.append(f"    - Has {rel_count} relationships with other files")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            logger.error(f"Error generating context summary text: {str(e)}")
            return None