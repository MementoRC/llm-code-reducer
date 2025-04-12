"""
Test for the improved debug function removal.
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mcp_server_code_reducer.debug_handler import improved_remove_debug_functions


class TestImprovedDebugRemoval(unittest.TestCase):
    
    def test_standalone_debug_removal(self):
        """Test that all debug calls are removed regardless of location."""
        code = """def test_function():
    print("Debug output")  # A standalone debug statement
    
    if verbose:
        print(f"This should be removed too")  # Inside conditional
        
    logger.debug("More debug data")  # Another debug call
    
    return True
"""
        # Process with the improved function
        processed = improved_remove_debug_functions(code, ["print", "logger.debug"])
        
        # All debug statements should be removed
        self.assertNotIn('print("Debug output")', processed)
        self.assertNotIn('logger.debug("More debug data")', processed)
        self.assertNotIn('print(f"This should be removed too")', processed)
        
        # The conditional structure should still exist
        self.assertIn("if verbose:", processed)
        
        # Check that we have correct code structure
        self.assertIn("def test_function():", processed)
        self.assertIn("return True", processed)
    
    def test_complex_case(self):
        """Test with a more complex code example."""
        code = """
def process_data(items, verbose=False):
    results = []
    print("Starting processing...")  # Debug statement
    
    for i, item in enumerate(items):
        logger.debug(f"Processing item {i}: {item}")  # Debug statement
        if verbose:
            print(f"Now processing: {item}")  # Should be removed too
        
        processed = item * 2
        results.append(processed)
        
        if processed > 10:
            console.log(f"Large value detected: {processed}")  # Debug statement
    
    if debug_mode:
        print("Debug results:", results)  # Debug in conditional
        
    return results
"""
        # Process with the improved function
        processed = improved_remove_debug_functions(code, ["print", "logger.debug", "console.log"])
        
        # All debug statements should be removed regardless of location
        self.assertNotIn('print("Starting processing...")', processed)
        self.assertNotIn('logger.debug(f"Processing item', processed)
        self.assertNotIn('console.log(f"Large value detected:', processed)
        self.assertNotIn('print(f"Now processing: {item}")', processed)
        self.assertNotIn('print("Debug results:"', processed)
        
        # The conditional structures should still exist
        self.assertIn("if verbose:", processed)
        self.assertIn("if debug_mode:", processed)
        self.assertIn("if processed > 10:", processed)
        
        # Core functionality should remain
        self.assertIn("processed = item * 2", processed)
        self.assertIn("results.append(processed)", processed)
        self.assertIn("return results", processed)


if __name__ == "__main__":
    unittest.main()