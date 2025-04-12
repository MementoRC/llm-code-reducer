"""
pytest configuration for code_reducer tests.
"""

import os
import sys
import pytest

# Add the src directory to the Python path for tests
# This ensures proper module resolution for the src-style package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))