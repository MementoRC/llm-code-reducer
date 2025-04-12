#!/usr/bin/env python3

"""
Test database integration for the Code Reducer MCP Server.
"""

import os
import tempfile
import unittest
from pathlib import Path

from mcp_server_code_reducer.database import CodeReducerDatabase
from mcp_server_code_reducer.models import MappingData, Position


class TestCodeReducerDatabase(unittest.TestCase):
    """Test cases for the Code Reducer Database."""

    def setUp(self):
        """Set up a temporary database for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_db.sqlite")
        self.db = CodeReducerDatabase(self.db_path)

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_store_and_retrieve_file(self):
        """Test storing and retrieving a processed file."""
        # Create sample data
        file_id = "test_file_1"
        original_content = """
def greet(name: str) -> str:
    # This is a comment
    return f"Hello, {name}!"
"""
        processed_content = """
def greet(name):
    return f"Hello, {name}!"
"""
        mapping_data = MappingData(
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
        success = self.db.store_processed_file(
            file_id=file_id,
            original_content=original_content,
            processed_content=processed_content,
            file_name="test.py",
            original_lines=4,
            processed_lines=3,
            reduction_percentage=25.0,
            transformations=["comments", "type_hints"],
            mapping_data=mapping_data.model_dump()
        )
        self.assertTrue(success, "Failed to store file")

        # Retrieve the file
        file_record = self.db.get_processed_file(file_id)
        self.assertIsNotNone(file_record, "Failed to retrieve file")
        self.assertEqual(file_record.file_id, file_id)
        self.assertEqual(file_record.original_content, original_content)
        self.assertEqual(file_record.processed_content, processed_content)
        self.assertEqual(file_record.file_name, "test.py")
        self.assertEqual(file_record.original_lines, 4)
        self.assertEqual(file_record.processed_lines, 3)
        self.assertEqual(file_record.reduction_percentage, 25.0)
        self.assertEqual(set(file_record.transformations), {"comments", "type_hints"})

        # Check mapping data
        self.assertEqual(
            file_record.mapping_data.original_to_processed["0:0"].line,
            mapping_data.original_to_processed["0:0"].line
        )
        self.assertEqual(
            file_record.mapping_data.processed_to_original["1:0"].line,
            mapping_data.processed_to_original["1:0"].line
        )

    def test_list_processed_files(self):
        """Test listing processed files."""
        # Store multiple files
        for i in range(5):
            file_id = f"test_file_{i}"
            self.db.store_processed_file(
                file_id=file_id,
                original_content=f"content_{i}",
                processed_content=f"processed_{i}",
                file_name=f"test_{i}.py",
                original_lines=10,
                processed_lines=8,
                reduction_percentage=20.0,
                transformations=["comments"],
                mapping_data=MappingData(
                    original_to_processed={},
                    processed_to_original={}
                ).model_dump()
            )

        # List files with pagination
        files = self.db.list_processed_files(limit=3, offset=0)
        self.assertEqual(len(files), 3, "Should return 3 files")
        
        # Get next page
        files_page2 = self.db.list_processed_files(limit=3, offset=3)
        self.assertEqual(len(files_page2), 2, "Should return 2 files")

    def test_delete_processed_file(self):
        """Test deleting a processed file."""
        # Store a file
        file_id = "test_delete"
        self.db.store_processed_file(
            file_id=file_id,
            original_content="content",
            processed_content="processed",
            file_name="delete.py",
            original_lines=10,
            processed_lines=8,
            reduction_percentage=20.0,
            transformations=["comments"],
            mapping_data=MappingData(
                original_to_processed={},
                processed_to_original={}
            ).model_dump()
        )
        
        # Verify file exists
        file_record = self.db.get_processed_file(file_id)
        self.assertIsNotNone(file_record, "File should exist")
        
        # Delete the file
        success = self.db.delete_processed_file(file_id)
        self.assertTrue(success, "Delete operation should succeed")
        
        # Verify file no longer exists
        file_record = self.db.get_processed_file(file_id)
        self.assertIsNone(file_record, "File should be deleted")


if __name__ == "__main__":
    unittest.main()