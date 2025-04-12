"""
Database manager for the Code Reducer MCP Server.
"""

import os
import sqlite3
import json
import logging
from contextlib import closing
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timezone
import uuid

from .models import (
    ProcessedFileRecord, 
    TransformationType, 
    MappingData, 
    Context, 
    ContextConfiguration, 
    ContextFileMetadata,
    ContextSummary,
    FileRelationship
)

logger = logging.getLogger("mcp_code_reducer_db")

class CodeReducerDatabase:
    """Manages the SQLite database for the Code Reducer MCP Server."""

    def __init__(self, db_path: str = None):
        """
        Initialize the database connection and create tables if they don't exist.
        
        Args:
            db_path: Path to the SQLite database file. If None, uses default location.
        """
        if not db_path:
            # Use a default path in the user's home directory
            db_path = os.path.expanduser("~/.code_reducer/code_reducer.db")
        
        # Ensure the directory exists if not in-memory db
        if db_path != ":memory:":
            db_dir = os.path.dirname(db_path)
            os.makedirs(db_dir, exist_ok=True)
        
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the database connection and create tables if they don't exist."""
        logger.debug(f"Initializing database at {self.db_path}")
        
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            with closing(conn.cursor()) as cursor:
                # Create processed_files table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS processed_files (
                    file_id TEXT PRIMARY KEY,
                    original_content TEXT NOT NULL,
                    processed_content TEXT NOT NULL,
                    file_name TEXT,
                    original_lines INTEGER NOT NULL,
                    processed_lines INTEGER NOT NULL,
                    reduction_percentage REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                ''')
                
                # Create transformations table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS transformations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id TEXT NOT NULL,
                    transformation_type TEXT NOT NULL,
                    FOREIGN KEY (file_id) REFERENCES processed_files (file_id) ON DELETE CASCADE
                )
                ''')
                
                # Create mappings table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS mappings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id TEXT NOT NULL,
                    mapping_data TEXT NOT NULL,
                    FOREIGN KEY (file_id) REFERENCES processed_files (file_id) ON DELETE CASCADE
                )
                ''')
                
                # Create contexts table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS contexts (
                    context_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    configuration TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    summary TEXT
                )
                ''')
                
                # Create context_files table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS context_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_id TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    importance REAL DEFAULT 1.0,
                    tags TEXT,
                    FOREIGN KEY (context_id) REFERENCES contexts (context_id) ON DELETE CASCADE,
                    FOREIGN KEY (file_id) REFERENCES processed_files (file_id) ON DELETE CASCADE,
                    UNIQUE(context_id, file_id)
                )
                ''')
                
                # Create file_relationships table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_id TEXT NOT NULL,
                    source_file_id TEXT NOT NULL,
                    target_file_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    FOREIGN KEY (context_id) REFERENCES contexts (context_id) ON DELETE CASCADE,
                    FOREIGN KEY (source_file_id) REFERENCES processed_files (file_id) ON DELETE CASCADE,
                    FOREIGN KEY (target_file_id) REFERENCES processed_files (file_id) ON DELETE CASCADE,
                    UNIQUE(context_id, source_file_id, target_file_id, relationship_type)
                )
                ''')
                
                conn.commit()
    
    def store_processed_file(
        self,
        file_id: str,
        original_content: str,
        processed_content: str,
        file_name: Optional[str],
        original_lines: int,
        processed_lines: int,
        reduction_percentage: float,
        transformations: List[str],
        mapping_data: Dict
    ) -> bool:
        """
        Store a processed file and its metadata in the database.
        
        Args:
            file_id: Unique identifier for the file
            original_content: Original content of the file
            processed_content: Processed content of the file
            file_name: Name of the file (optional)
            original_lines: Number of lines in the original file
            processed_lines: Number of lines in the processed file
            reduction_percentage: Percentage of reduction
            transformations: List of transformations applied
            mapping_data: Mapping data between original and processed positions
            
        Returns:
            bool: True if stored successfully, False otherwise
        """
        current_time = datetime.now(timezone.utc).isoformat()
        
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                with closing(conn.cursor()) as cursor:
                    # Insert into processed_files
                    cursor.execute('''
                    INSERT OR REPLACE INTO processed_files (
                        file_id, original_content, processed_content, file_name,
                        original_lines, processed_lines, reduction_percentage,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        file_id, original_content, processed_content, file_name,
                        original_lines, processed_lines, reduction_percentage,
                        current_time, current_time
                    ))
                    
                    # Delete existing transformations for this file
                    cursor.execute('DELETE FROM transformations WHERE file_id = ?', (file_id,))
                    
                    # Insert transformations
                    for transformation in transformations:
                        cursor.execute('''
                        INSERT INTO transformations (file_id, transformation_type)
                        VALUES (?, ?)
                        ''', (file_id, transformation))
                    
                    # Delete existing mappings for this file
                    cursor.execute('DELETE FROM mappings WHERE file_id = ?', (file_id,))
                    
                    # Insert mapping data
                    cursor.execute('''
                    INSERT INTO mappings (file_id, mapping_data)
                    VALUES (?, ?)
                    ''', (file_id, json.dumps(mapping_data)))
                    
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Error storing processed file: {str(e)}")
            return False
    
    def get_processed_file(self, file_id: str) -> Optional[ProcessedFileRecord]:
        """
        Retrieve a processed file record by its ID.
        
        Args:
            file_id: Unique identifier for the file
            
        Returns:
            ProcessedFileRecord or None if not found
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                with closing(conn.cursor()) as cursor:
                    # Get the file record
                    cursor.execute('''
                    SELECT * FROM processed_files WHERE file_id = ?
                    ''', (file_id,))
                    
                    file_record = cursor.fetchone()
                    if not file_record:
                        return None
                    
                    # Get transformations
                    cursor.execute('''
                    SELECT transformation_type FROM transformations WHERE file_id = ?
                    ''', (file_id,))
                    
                    transformations = [row['transformation_type'] for row in cursor.fetchall()]
                    
                    # Get mapping data
                    cursor.execute('''
                    SELECT mapping_data FROM mappings WHERE file_id = ?
                    ''', (file_id,))
                    
                    mapping_row = cursor.fetchone()
                    mapping_data = json.loads(mapping_row['mapping_data']) if mapping_row else {}
                    
                    # Build the record
                    return ProcessedFileRecord(
                        file_id=file_record['file_id'],
                        original_content=file_record['original_content'],
                        processed_content=file_record['processed_content'],
                        file_name=file_record['file_name'],
                        original_lines=file_record['original_lines'],
                        processed_lines=file_record['processed_lines'],
                        reduction_percentage=file_record['reduction_percentage'],
                        transformations=transformations,
                        mapping_data=MappingData.model_validate(mapping_data),
                        created_at=file_record['created_at'],
                        updated_at=file_record['updated_at']
                    )
        except Exception as e:
            logger.error(f"Error retrieving processed file: {str(e)}")
            return None
    
    def list_processed_files(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List processed files with pagination.
        
        Args:
            limit: Maximum number of records to return
            offset: Offset for pagination
            
        Returns:
            List of file records
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                with closing(conn.cursor()) as cursor:
                    cursor.execute('''
                    SELECT file_id, file_name, original_lines, processed_lines, 
                           reduction_percentage, created_at, updated_at
                    FROM processed_files
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                    ''', (limit, offset))
                    
                    return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error listing processed files: {str(e)}")
            return []
    
    def delete_processed_file(self, file_id: str) -> bool:
        """
        Delete a processed file record.
        
        Args:
            file_id: Unique identifier for the file
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with closing(conn.cursor()) as cursor:
                    cursor.execute('DELETE FROM processed_files WHERE file_id = ?', (file_id,))
                    # Cascade deletes will handle the related records
                    conn.commit()
                    return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting processed file: {str(e)}")
            return False
    
    def get_mapping_data(self, file_id: str) -> Optional[Dict]:
        """
        Get mapping data for a file.
        
        Args:
            file_id: Unique identifier for the file
            
        Returns:
            Dict: Mapping data or None if not found
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                with closing(conn.cursor()) as cursor:
                    cursor.execute('''
                    SELECT mapping_data FROM mappings WHERE file_id = ?
                    ''', (file_id,))
                    
                    row = cursor.fetchone()
                    return json.loads(row['mapping_data']) if row else None
        except Exception as e:
            logger.error(f"Error retrieving mapping data: {str(e)}")
            return None

    # Context Management Methods

    def create_context(
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
        try:
            context_id = f"ctx_{uuid.uuid4().hex[:12]}"
            current_time = datetime.now(timezone.utc).isoformat()
            
            # Use default configuration if not provided
            if not configuration:
                configuration = ContextConfiguration()
                
            with closing(sqlite3.connect(self.db_path)) as conn:
                with closing(conn.cursor()) as cursor:
                    cursor.execute('''
                    INSERT INTO contexts (
                        context_id, name, description, configuration,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        context_id,
                        name,
                        description,
                        json.dumps(configuration.model_dump()),
                        current_time,
                        current_time
                    ))
                    
                    conn.commit()
                    return context_id
        except Exception as e:
            logger.error(f"Error creating context: {str(e)}")
            return None
    
    def get_context(self, context_id: str) -> Optional[Context]:
        """
        Get a context by its ID.
        
        Args:
            context_id: ID of the context
            
        Returns:
            Context object or None if not found
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                with closing(conn.cursor()) as cursor:
                    # Get the context record
                    cursor.execute('''
                    SELECT * FROM contexts WHERE context_id = ?
                    ''', (context_id,))
                    
                    context_record = cursor.fetchone()
                    if not context_record:
                        return None
                    
                    # Get files in this context
                    cursor.execute('''
                    SELECT * FROM context_files 
                    WHERE context_id = ?
                    ''', (context_id,))
                    
                    context_files = {}
                    for row in cursor.fetchall():
                        file_id = row['file_id']
                        
                        # Get file tags
                        tags = []
                        if row['tags']:
                            tags = json.loads(row['tags'])
                            
                        # Create file metadata
                        context_files[file_id] = ContextFileMetadata(
                            file_id=file_id,
                            file_path=row['file_path'],
                            file_name=row['file_name'],
                            importance=row['importance'],
                            tags=tags,
                            relationships={}
                        )
                    
                    # Get file relationships
                    cursor.execute('''
                    SELECT * FROM file_relationships
                    WHERE context_id = ?
                    ''', (context_id,))
                    
                    for rel in cursor.fetchall():
                        source_id = rel['source_file_id']
                        target_id = rel['target_file_id']
                        rel_type = rel['relationship_type']
                        
                        # Skip if file no longer exists in context
                        if source_id not in context_files:
                            continue
                            
                        # Initialize relationships dict if needed
                        if target_id not in context_files[source_id].relationships:
                            context_files[source_id].relationships[target_id] = []
                            
                        # Add relationship
                        context_files[source_id].relationships[target_id].append(
                            FileRelationship(rel_type)
                        )
                    
                    # Get or compute summary
                    summary = None
                    if context_record['summary']:
                        summary = ContextSummary.model_validate(json.loads(context_record['summary']))
                    
                    # Create the context object
                    return Context(
                        context_id=context_id,
                        name=context_record['name'],
                        description=context_record['description'],
                        files=context_files,
                        configuration=ContextConfiguration.model_validate(
                            json.loads(context_record['configuration'])
                        ),
                        created_at=context_record['created_at'],
                        updated_at=context_record['updated_at'],
                        summary=summary
                    )
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return None
    
    def list_contexts(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List available contexts with pagination.
        
        Args:
            limit: Maximum number of records to return
            offset: Offset for pagination
            
        Returns:
            List of context records
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                with closing(conn.cursor()) as cursor:
                    cursor.execute('''
                    SELECT context_id, name, description, created_at, updated_at 
                    FROM contexts
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                    ''', (limit, offset))
                    
                    contexts = []
                    for row in cursor.fetchall():
                        # Count files in this context
                        cursor.execute('''
                        SELECT COUNT(*) as file_count 
                        FROM context_files 
                        WHERE context_id = ?
                        ''', (row['context_id'],))
                        
                        file_count = cursor.fetchone()['file_count']
                        
                        # Add to result list
                        contexts.append({
                            'context_id': row['context_id'],
                            'name': row['name'],
                            'description': row['description'],
                            'file_count': file_count,
                            'created_at': row['created_at'],
                            'updated_at': row['updated_at']
                        })
                    
                    return contexts
        except Exception as e:
            logger.error(f"Error listing contexts: {str(e)}")
            return []
    
    def delete_context(self, context_id: str) -> bool:
        """
        Delete a context and all its file associations.
        
        Args:
            context_id: ID of the context to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with closing(conn.cursor()) as cursor:
                    cursor.execute('DELETE FROM contexts WHERE context_id = ?', (context_id,))
                    # Cascade deletes will handle related records
                    conn.commit()
                    return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting context: {str(e)}")
            return False
    
    def add_file_to_context(
        self, 
        context_id: str, 
        file_id: str, 
        file_path: str,
        file_name: str,
        importance: float = 1.0,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Add a processed file to a context.
        
        Args:
            context_id: ID of the context
            file_id: ID of the processed file
            file_path: Path of the file in its original location
            file_name: Name of the file
            importance: Importance score for this file
            tags: Optional list of tags for categorization
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        try:
            # Check if the context exists
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                with closing(conn.cursor()) as cursor:
                    cursor.execute('SELECT 1 FROM contexts WHERE context_id = ?', (context_id,))
                    if not cursor.fetchone():
                        logger.error(f"Context not found: {context_id}")
                        return False
                    
                    # Check if the file exists
                    cursor.execute('SELECT 1 FROM processed_files WHERE file_id = ?', (file_id,))
                    if not cursor.fetchone():
                        logger.error(f"File not found: {file_id}")
                        return False
                    
                    # Add to context_files
                    try:
                        tags_json = json.dumps(tags or [])
                        cursor.execute('''
                        INSERT OR REPLACE INTO context_files (
                            context_id, file_id, file_path, file_name, importance, tags
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            context_id, file_id, file_path, file_name, importance, tags_json
                        ))
                        
                        # Update the context's updated_at timestamp
                        current_time = datetime.now(timezone.utc).isoformat()
                        cursor.execute('''
                        UPDATE contexts SET updated_at = ? WHERE context_id = ?
                        ''', (current_time, context_id))
                        
                        conn.commit()
                        return True
                    except Exception as e:
                        logger.error(f"Error adding file to context: {str(e)}")
                        return False
        except Exception as e:
            logger.error(f"Error in add_file_to_context: {str(e)}")
            return False
    
    def remove_file_from_context(self, context_id: str, file_id: str) -> bool:
        """
        Remove a file from a context.
        
        Args:
            context_id: ID of the context
            file_id: ID of the file to remove
            
        Returns:
            bool: True if removed successfully, False otherwise
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with closing(conn.cursor()) as cursor:
                    # Remove from context_files
                    cursor.execute('''
                    DELETE FROM context_files 
                    WHERE context_id = ? AND file_id = ?
                    ''', (context_id, file_id))
                    
                    # Remove any relationships involving this file
                    cursor.execute('''
                    DELETE FROM file_relationships 
                    WHERE context_id = ? AND (source_file_id = ? OR target_file_id = ?)
                    ''', (context_id, file_id, file_id))
                    
                    # Update the context's updated_at timestamp
                    current_time = datetime.now(timezone.utc).isoformat()
                    cursor.execute('''
                    UPDATE contexts SET updated_at = ? WHERE context_id = ?
                    ''', (current_time, context_id))
                    
                    conn.commit()
                    # Return True if we deleted anything
                    return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error removing file from context: {str(e)}")
            return False
    
    def add_file_relationship(
        self, 
        context_id: str, 
        source_file_id: str, 
        target_file_id: str, 
        relationship_type: str
    ) -> bool:
        """
        Add a relationship between two files in a context.
        
        Args:
            context_id: ID of the context
            source_file_id: ID of the source file
            target_file_id: ID of the target file
            relationship_type: Type of relationship (see FileRelationship enum)
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with closing(conn.cursor()) as cursor:
                    # Check if both files are in the context
                    cursor.execute('''
                    SELECT COUNT(*) as count FROM context_files 
                    WHERE context_id = ? AND file_id IN (?, ?)
                    ''', (context_id, source_file_id, target_file_id))
                    
                    if cursor.fetchone()['count'] != 2:
                        logger.error(f"Both files must be in the context: {source_file_id}, {target_file_id}")
                        return False
                    
                    # Add the relationship
                    cursor.execute('''
                    INSERT OR REPLACE INTO file_relationships (
                        context_id, source_file_id, target_file_id, relationship_type
                    ) VALUES (?, ?, ?, ?)
                    ''', (context_id, source_file_id, target_file_id, relationship_type))
                    
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Error adding file relationship: {str(e)}")
            return False
    
    def get_context_files(
        self, 
        context_id: str, 
        limit: int = 20, 
        offset: int = 0,
        importance_threshold: Optional[float] = None,
        include_content: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get files in a context with optional filtering.
        
        Args:
            context_id: ID of the context
            limit: Maximum number of files to return
            offset: Offset for pagination
            importance_threshold: Minimum importance score
            include_content: Whether to include file content
            
        Returns:
            List of file records
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                with closing(conn.cursor()) as cursor:
                    # Build the query with optional importance filter
                    query = '''
                    SELECT cf.file_id, cf.file_path, cf.file_name, cf.importance, cf.tags
                    FROM context_files cf
                    WHERE cf.context_id = ?
                    '''
                    
                    params = [context_id]
                    
                    if importance_threshold is not None:
                        query += " AND cf.importance >= ?"
                        params.append(importance_threshold)
                        
                    query += " ORDER BY cf.importance DESC LIMIT ? OFFSET ?"
                    params.extend([limit, offset])
                    
                    cursor.execute(query, params)
                    
                    files = []
                    for row in cursor.fetchall():
                        file_id = row['file_id']
                        file_data = {
                            'file_id': file_id,
                            'file_path': row['file_path'],
                            'file_name': row['file_name'],
                            'importance': row['importance'],
                            'tags': json.loads(row['tags']) if row['tags'] else []
                        }
                        
                        # Include content if requested
                        if include_content:
                            file_record = self.get_processed_file(file_id)
                            if file_record:
                                file_data['original_content'] = file_record.original_content
                                file_data['processed_content'] = file_record.processed_content
                        
                        # Get relationships for this file
                        cursor.execute('''
                        SELECT target_file_id, relationship_type
                        FROM file_relationships
                        WHERE context_id = ? AND source_file_id = ?
                        ''', (context_id, file_id))
                        
                        relationships = {}
                        for rel in cursor.fetchall():
                            target_id = rel['target_file_id']
                            rel_type = rel['relationship_type']
                            
                            if target_id not in relationships:
                                relationships[target_id] = []
                                
                            relationships[target_id].append(rel_type)
                        
                        file_data['relationships'] = relationships
                        files.append(file_data)
                    
                    return files
        except Exception as e:
            logger.error(f"Error getting context files: {str(e)}")
            return []
    
    def update_context_configuration(
        self, 
        context_id: str, 
        configuration: ContextConfiguration
    ) -> bool:
        """
        Update the configuration for a context.
        
        Args:
            context_id: ID of the context
            configuration: New configuration settings
            
        Returns:
            bool: True if updated successfully, False otherwise
        """
        try:
            current_time = datetime.now(timezone.utc).isoformat()
            
            with closing(sqlite3.connect(self.db_path)) as conn:
                with closing(conn.cursor()) as cursor:
                    cursor.execute('''
                    UPDATE contexts 
                    SET configuration = ?, updated_at = ? 
                    WHERE context_id = ?
                    ''', (
                        json.dumps(configuration.model_dump()),
                        current_time,
                        context_id
                    ))
                    
                    conn.commit()
                    return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating context configuration: {str(e)}")
            return False
    
    def update_context_summary(self, context_id: str, summary: ContextSummary) -> bool:
        """
        Update the summary for a context.
        
        Args:
            context_id: ID of the context
            summary: Summary information
            
        Returns:
            bool: True if updated successfully, False otherwise
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with closing(conn.cursor()) as cursor:
                    cursor.execute('''
                    UPDATE contexts 
                    SET summary = ? 
                    WHERE context_id = ?
                    ''', (
                        json.dumps(summary.model_dump()),
                        context_id
                    ))
                    
                    conn.commit()
                    return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating context summary: {str(e)}")
            return False
    
    def update_file_importance(
        self, 
        context_id: str, 
        file_id: str, 
        importance: float
    ) -> bool:
        """
        Update the importance score for a file in a context.
        
        Args:
            context_id: ID of the context
            file_id: ID of the file
            importance: New importance score
            
        Returns:
            bool: True if updated successfully, False otherwise
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with closing(conn.cursor()) as cursor:
                    cursor.execute('''
                    UPDATE context_files 
                    SET importance = ? 
                    WHERE context_id = ? AND file_id = ?
                    ''', (importance, context_id, file_id))
                    
                    conn.commit()
                    return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating file importance: {str(e)}")
            return False