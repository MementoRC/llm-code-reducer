#!/usr/bin/env python3

"""
Test script for the code reducer MCP server.

This script tests the Python code processing functionality.
"""

import os
import json
import subprocess
import sys
from pathlib import Path
import tempfile

# Path setup is handled by conftest.py


def test_process_python():
    """Test the MCP server's process_python function using the CLI."""
    
    # Test code with comments and type hints
    test_code = """
# This is a comment
def example_function(arg1: str, arg2: int = 0) -> bool:
    # Process the arguments
    result = len(arg1) > arg2  # Another comment
    return result  # Return the result
"""
    
    # Create a temporary file for the test input and output
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+") as input_file, \
         tempfile.NamedTemporaryFile(suffix=".json", mode="w+") as output_file:
        
        # Create input JSON for the MCP server
        input_data = {
            "function": "process_python",
            "arguments": {
                "content": test_code,
                "strip_comments": True,
                "strip_type_hints": True
            }
        }
        
        # Write input JSON to file
        json.dump(input_data, input_file)
        input_file.flush()
        
        # Run the MCP server CLI command
        try:
            cmd = [
                "python", "-m", "mcp_server_code_reducer.__main__",
                "--input", input_file.name,
                "--output", output_file.name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check if the command succeeded
            if result.returncode != 0:
                print(f"Command failed with return code {result.returncode}")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
                return
            
            # Read and parse output JSON
            output_file.seek(0)
            output_data = json.load(output_file)
            
            # Print results
            print("\n=== Original Code ===")
            print(test_code)
            
            print("\n=== Processed Code ===")
            print(output_data["processed_content"])
            print(f"Reduction: {output_data['reduction_percentage']}%")
            print(f"Original Lines: {output_data['original_lines']}")
            print(f"Processed Lines: {output_data['processed_lines']}")
            print(f"Transformations: {', '.join(output_data['transformations'])}")
            
        except Exception as e:
            print(f"Error testing MCP server: {str(e)}")


if __name__ == "__main__":
    test_process_python()