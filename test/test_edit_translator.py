#!/usr/bin/env python3

"""
Test the Edit Translator Service.
"""

import os
import tempfile
import unittest
from pathlib import Path

from mcp_server_code_reducer.database import CodeReducerDatabase
from mcp_server_code_reducer.models import MappingData, Position
from mcp_server_code_reducer.edit_translator import EditTranslator, EditOperation


class TestEditTranslator(unittest.TestCase):
    """Test cases for the Edit Translator Service."""

    def setUp(self):
        """Set up a temporary database and sample data for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_db.sqlite")
        self.db = CodeReducerDatabase(self.db_path)
        self.edit_translator = EditTranslator(self.db)
        
        # Create sample data
        self.file_id = "test_edit_file"
        self.original_content = """
def greet(name: str) -> str:
    # This is a comment
    return f"Hello, {name}!"
"""
        self.processed_content = """
def greet(name):
    return f"Hello, {name}!"
"""
        self.mapping_data = MappingData(
            original_to_processed={
                "0:0": Position(line=0, character=0),
                "1:0": Position(line=1, character=0),
                "2:0": Position(line=1, character=0),
                "3:0": Position(line=2, character=0)
            },
            processed_to_original={
                "0:0": Position(line=0, character=0),
                "1:0": Position(line=1, character=0),
                "2:0": Position(line=3, character=0)
            }
        )
        
        # Store the file
        self.db.store_processed_file(
            file_id=self.file_id,
            original_content=self.original_content,
            processed_content=self.processed_content,
            file_name="test.py",
            original_lines=4,
            processed_lines=3,
            reduction_percentage=25.0,
            transformations=["comments", "type_hints"],
            mapping_data=self.mapping_data.model_dump()
        )

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_translate_edit_insert(self):
        """Test translating an insert edit from processed to original."""
        # Create an edit in the processed code
        edit = EditOperation(
            position=Position(line=1, character=10),
            content="new_",
            length=0  # Insert
        )
        
        # Translate to original
        translated = self.edit_translator.translate_edit(
            self.file_id, edit, to_original=True
        )
        
        self.assertIsNotNone(translated)
        # In our mapping, line 1 in processed maps to line 1 in original
        self.assertEqual(translated.position.line, 1)
        self.assertEqual(translated.content, "new_")
        self.assertEqual(translated.length, 0)  # Still an insert

    def test_translate_edit_replace(self):
        """Test translating a replace edit from processed to original."""
        # Create a replace edit in the processed code
        edit = EditOperation(
            position=Position(line=1, character=4),
            content="getName",
            length=4  # Replace "name"
        )
        
        # Translate to original
        translated = self.edit_translator.translate_edit(
            self.file_id, edit, to_original=True
        )
        
        self.assertIsNotNone(translated)
        self.assertEqual(translated.position.line, 1)
        self.assertEqual(translated.content, "getName")
        # Length might be adjusted based on context differences

    def test_apply_edit(self):
        """Test applying an edit to content."""
        content = "def example():\n    return 42"
        
        # Insert edit
        insert_edit = EditOperation(
            position=Position(line=0, character=12),
            content="value",
            length=0
        )
        
        result = self.edit_translator.apply_edit_to_content(content, insert_edit)
        self.assertEqual(result, "def example(value):\n    return 42")
        
        # Replace edit
        replace_edit = EditOperation(
            position=Position(line=1, character=11),
            content="100",
            length=2
        )
        
        result = self.edit_translator.apply_edit_to_content(content, replace_edit)
        self.assertEqual(result, "def example():\n    return 100")

    def test_translate_and_apply_edit(self):
        """Test translating and applying an edit."""
        # Create an edit in the processed code - replace name with person
        edit = EditOperation(
            position=Position(line=1, character=10),
            content="person",
            length=4  # Replace "name"
        )
        
        # Translate and apply
        edited_content, translated_edit = self.edit_translator.translate_and_apply_edit(
            self.file_id,
            self.original_content,
            edit,
            to_original=True
        )
        
        self.assertIsNotNone(edited_content)
        self.assertIsNotNone(translated_edit)
        
        # Check that the content was modified in some way
        self.assertNotEqual(edited_content, self.original_content)
        
        # Since we're replacing "name" with "person", the edit should contain "person"
        self.assertIn("person", edited_content)


if __name__ == "__main__":
    unittest.main()