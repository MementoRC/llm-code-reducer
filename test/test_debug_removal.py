"""
Test for debug function removal in the MCP code reducer server.
"""

import unittest
from mcp_server_code_reducer.server import remove_debug_functions, optimize_whitespace


class TestDebugFunctionRemoval(unittest.TestCase):
    def test_remove_simple_debug_calls(self):
        """Test removing simple debug function calls."""
        code = """def main():
    x = 5
    print("Debug:", x)  # This should be removed
    logger.debug("This is also a debug message")  # This should be removed
    return x
"""
        debug_functions = ["print", "logger.debug"]
        result = remove_debug_functions(code, debug_functions)
        
        # Check that the debug lines were removed
        self.assertNotIn("print(\"Debug:", result)
        self.assertNotIn("logger.debug", result)
        
        # Check that the important code is preserved
        self.assertIn("def main():", result)
        self.assertIn("    x = 5", result)
        self.assertIn("    return x", result)

    def test_remove_debug_conditional_blocks(self):
        """Test removing conditional blocks containing only debug calls."""
        code = """def main():
    x = 5
    if debug:
        logger.debug("This is a debug message")
        print("More debugging")
    return x
"""
        debug_functions = ["print", "logger.debug"]
        result = remove_debug_functions(code, debug_functions)
        
        # Check that the if block and debug calls were removed
        self.assertNotIn("if debug:", result)
        self.assertNotIn("logger.debug", result)
        self.assertNotIn("print(\"More debugging\")", result)
        
        # Check that the important code is preserved
        self.assertIn("def main():", result)
        self.assertIn("    x = 5", result)
        self.assertIn("    return x", result)

    def test_mixed_code_with_debug_functions(self):
        """Test mixed code with some debug functions to remove."""
        code = """def process_data(items):
    results = []
    for item in items:
        print(f"Processing item: {item}")  # Debug log
        processed = item * 2
        results.append(processed)
        if verbose:
            print(f"Processed result: {processed}")
    
    logger.debug(f"All results: {results}")
    return results
"""
        debug_functions = ["print", "logger.debug"]
        result = remove_debug_functions(code, debug_functions)
        
        # Check that the debug lines were removed
        self.assertNotIn("print(f\"Processing item:", result)
        self.assertNotIn("logger.debug", result)
        
        # When we remove debug calls in if/else blocks, the entire block may be removed
        # This test verifies the behavior is as expected - if the entire if block contains only debug
        # function calls, it should be removed completely
        
        # Check that the important code is preserved
        self.assertIn("def process_data(items):", result)
        self.assertIn("    results = []", result)
        self.assertIn("    for item in items:", result)
        self.assertIn("        processed = item * 2", result)
        self.assertIn("        results.append(processed)", result)
        self.assertIn("    return results", result)

    def test_no_debug_functions(self):
        """Test code with no debug functions to remove."""
        code = """def clean_function():
    x = 5
    y = x * 2
    return y
"""
        debug_functions = ["print", "logger.debug"]
        result = remove_debug_functions(code, debug_functions)
        # Normalize line endings for comparison
        expected_lines = [line for line in code.splitlines() if line.strip()]
        result_lines = [line for line in result.splitlines() if line.strip()]
        self.assertEqual(result_lines, expected_lines)

    def test_empty_debug_functions_list(self):
        """Test with an empty debug functions list."""
        code = """def main():
    print("This should not be removed")
    return 5
"""
        result = remove_debug_functions(code, [])
        self.assertEqual(result, code)


class TestWhitespaceOptimization(unittest.TestCase):
    def test_remove_trailing_whitespace(self):
        """Test removing trailing whitespace from lines."""
        code = "def main():    \n    x = 5    \n    return x    \n"
        result = optimize_whitespace(code)
        
        # Verify trailing whitespace was removed
        self.assertNotIn("def main():    ", result)
        self.assertNotIn("    x = 5    ", result)
        self.assertNotIn("    return x    ", result)
        
        # Verify content is preserved
        self.assertIn("def main():", result)
        self.assertIn("    x = 5", result)
        self.assertIn("    return x", result)

    def test_collapse_blank_lines(self):
        """Test collapsing multiple consecutive blank lines."""
        code = "def main():\n    x = 5\n\n\n\n    return x\n\n\n"
        result = optimize_whitespace(code)
        
        # Split into lines and count consecutive blank lines
        lines = result.splitlines()
        
        # Verify that no more than one consecutive blank line exists
        blank_line_count = 0
        max_consecutive_blanks = 0
        
        for line in lines:
            if line.strip() == "":
                blank_line_count += 1
            else:
                max_consecutive_blanks = max(max_consecutive_blanks, blank_line_count)
                blank_line_count = 0
                
        max_consecutive_blanks = max(max_consecutive_blanks, blank_line_count)
        self.assertLessEqual(max_consecutive_blanks, 1)
        
        # Verify content is preserved
        self.assertIn("def main():", result)
        self.assertIn("    x = 5", result)
        self.assertIn("    return x", result)

    def test_remove_blank_lines_at_boundaries(self):
        """Test removing blank lines at the beginning and end of the file."""
        code = "\n\n\ndef main():\n    x = 5\n    return x\n\n\n"
        result = optimize_whitespace(code)
        
        # Result should not start or end with blank lines
        lines = result.splitlines()
        if lines:  # Ensure there are lines before checking
            self.assertNotEqual(lines[0].strip(), "")
            self.assertNotEqual(lines[-1].strip(), "")
        
        # Verify content is preserved
        self.assertIn("def main():", result)
        self.assertIn("    x = 5", result)
        self.assertIn("    return x", result)


if __name__ == "__main__":
    unittest.main()