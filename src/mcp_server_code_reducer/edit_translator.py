"""
Edit Translator Service for Code Reducer MCP Server.

This module provides functionality to translate edit operations between 
processed and original code contexts.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass

from .models import Position, MappingData
from .database import CodeReducerDatabase


@dataclass
class EditOperation:
    """Represents an edit operation on code."""
    position: Position  # Starting position of the edit
    content: str        # Text to insert or replace
    length: int = 0     # Length of text to replace (0 for insert)
    
    def __str__(self) -> str:
        op_type = "insert" if self.length == 0 else "replace"
        return f"{op_type} at {self.position.line}:{self.position.character}, length: {self.length}, content: '{self.content[:20]}{'...' if len(self.content) > 20 else ''}'"


class EditTranslator:
    """
    Service for translating edit operations between processed and original code contexts.
    """
    
    def __init__(self, db: CodeReducerDatabase):
        """
        Initialize the EditTranslator.
        
        Args:
            db: Database manager for retrieving file mappings
        """
        self.db = db
    
    def translate_edit(self, file_id: str, edit: EditOperation, to_original: bool = True) -> Optional[EditOperation]:
        """
        Translate an edit operation between processed and original code contexts.
        
        Args:
            file_id: ID of the processed file
            edit: Edit operation to translate
            to_original: If True, translate from processed to original, otherwise from original to processed
            
        Returns:
            Translated edit operation or None if translation fails
        """
        # Get the file record with mappings
        file_record = self.db.get_processed_file(file_id)
        if not file_record:
            return None
        
        # Translate the position
        translated_position = self._translate_position(
            file_record.mapping_data, 
            edit.position, 
            to_original
        )
        
        if not translated_position:
            return None
            
        # For simple inserts, just translate the position
        if edit.length == 0:
            return EditOperation(
                position=translated_position,
                content=edit.content,
                length=0
            )
            
        # For replacements, we need to adjust the length
        translated_length = self._translate_length(
            file_record.mapping_data,
            edit.position,
            edit.length,
            to_original
        )
        
        return EditOperation(
            position=translated_position,
            content=edit.content,
            length=translated_length
        )
    
    def _translate_position(
        self, 
        mapping_data: MappingData, 
        position: Position, 
        to_original: bool
    ) -> Optional[Position]:
        """
        Translate a position between original and processed code.
        
        Args:
            mapping_data: Mapping data for the file
            position: Position to translate
            to_original: If True, translate from processed to original, otherwise from original to processed
            
        Returns:
            Translated position or None if not found
        """
        pos_str = str(position)
        
        if to_original and pos_str in mapping_data.processed_to_original:
            return mapping_data.processed_to_original[pos_str]
        elif not to_original and pos_str in mapping_data.original_to_processed:
            return mapping_data.original_to_processed[pos_str]
            
        # If exact position not found, find the closest position
        return self._find_closest_position(mapping_data, position, to_original)
    
    def _find_closest_position(
        self, 
        mapping_data: MappingData, 
        position: Position, 
        to_original: bool
    ) -> Optional[Position]:
        """
        Find the closest mapped position to the given position.
        
        Args:
            mapping_data: Mapping data for the file
            position: Position to find closest match for
            to_original: If True, translate from processed to original, otherwise from original to processed
            
        Returns:
            Closest position or None if not found
        """
        # Choose the appropriate mapping direction
        if to_original:
            source_map = mapping_data.processed_to_original
        else:
            source_map = mapping_data.original_to_processed
            
        # Find positions on the same line or closest line
        best_line_dist = float('inf')
        best_char_dist = float('inf')
        best_position = None
        
        for pos_str, mapped_pos in source_map.items():
            pos = Position(
                line=int(pos_str.split(':')[0]),
                character=int(pos_str.split(':')[1])
            )
            
            # Calculate distance
            line_dist = abs(pos.line - position.line)
            
            # If this is a better line match
            if line_dist < best_line_dist:
                best_line_dist = line_dist
                best_char_dist = abs(pos.character - position.character)
                best_position = mapped_pos
            # If same line, check character distance
            elif line_dist == best_line_dist:
                char_dist = abs(pos.character - position.character)
                if char_dist < best_char_dist:
                    best_char_dist = char_dist
                    best_position = mapped_pos
                    
        return best_position
    
    def _translate_length(
        self, 
        mapping_data: MappingData, 
        position: Position, 
        length: int, 
        to_original: bool
    ) -> int:
        """
        Translate a text length from one context to another.
        
        Args:
            mapping_data: Mapping data for the file
            position: Starting position of the text
            length: Length of the text in source context
            to_original: If True, translate from processed to original, otherwise from original to processed
            
        Returns:
            Translated length
        """
        # This is a simplification. In a more sophisticated implementation,
        # we would look at the actual content differences between the contexts.
        
        # Get the end position in the source context
        end_position = Position(
            line=position.line,
            character=position.character + length
        )
        
        # Translate start and end positions
        translated_start = self._translate_position(mapping_data, position, to_original)
        translated_end = self._translate_position(mapping_data, end_position, to_original)
        
        if not translated_start or not translated_end:
            # Fallback: use the same length
            return length
            
        # If they're on the same line, calculate character difference
        if translated_start.line == translated_end.line:
            return translated_end.character - translated_start.character
            
        # For multi-line edits, this is a simplification
        # In a real implementation, we'd need to consider the actual content
        return length
    
    def apply_edit_to_content(self, content: str, edit: EditOperation) -> str:
        """
        Apply an edit operation to content.
        
        Args:
            content: The content to edit
            edit: The edit operation to apply
            
        Returns:
            The edited content
        """
        lines = content.split('\n')
        
        # Make sure we have enough lines
        if edit.position.line >= len(lines):
            return content
            
        line = lines[edit.position.line]
        
        # Make sure the position isn't beyond the line length
        if edit.position.character > len(line):
            return content
            
        # Simple insert or replace within a single line
        if edit.length == 0:
            # Insert
            new_line = line[:edit.position.character] + edit.content + line[edit.position.character:]
            lines[edit.position.line] = new_line
        else:
            # Replace
            end_pos = min(edit.position.character + edit.length, len(line))
            new_line = line[:edit.position.character] + edit.content + line[end_pos:]
            lines[edit.position.line] = new_line
            
        return '\n'.join(lines)
    
    def translate_and_apply_edit(
        self, 
        file_id: str, 
        content: str, 
        edit: EditOperation, 
        to_original: bool = True
    ) -> Tuple[Optional[str], Optional[EditOperation]]:
        """
        Translate an edit operation and apply it to the content.
        
        Args:
            file_id: ID of the processed file
            content: Content to apply the edit to
            edit: Edit operation to translate and apply
            to_original: If True, translate from processed to original, otherwise from original to processed
            
        Returns:
            Tuple of (edited content, translated edit) or (None, None) if translation fails
        """
        translated_edit = self.translate_edit(file_id, edit, to_original)
        if not translated_edit:
            return None, None
            
        edited_content = self.apply_edit_to_content(content, translated_edit)
        return edited_content, translated_edit