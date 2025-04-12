"""
Improved debug function removal module for code reducer.
"""

import re
from typing import List


def improved_remove_debug_functions(source_code: str, debug_functions: List[str]) -> str:
    """
    Improved version that completely removes all debug function calls
    to maximize token reduction.
    
    Args:
        source_code: Python source code
        debug_functions: List of function names to be treated as debug functions 
                        (e.g., ["log_debug", "logger.debug", "console.print"])
    
    Returns:
        Processed code with all debug function calls removed
    """
    if not debug_functions:
        return source_code
    
    lines = source_code.splitlines()
    output_lines = []
    i = 0
    
    # Create pattern for debug function calls
    debug_pattern = '|'.join(re.escape(f) for f in debug_functions)
    
    # Track control structures and empty blocks
    indentation_stack = []
    empty_block_candidates = []
    
    while i < len(lines):
        line = lines[i]
        line_stripped = line.strip()
        current_indent = len(line) - len(line.lstrip()) if line else 0
        
        # Skip empty lines
        if not line_stripped:
            output_lines.append(line)
            i += 1
            continue
        
        # Check if this is a new control structure
        if line_stripped.endswith(':'):
            # Track this new block's indentation level
            indentation_stack.append(current_indent)
            
            # Start tracking a potential empty block candidate
            empty_block_candidates.append({
                'start_line': i,
                'indent': current_indent,
                'has_non_debug_content': False
            })
            
            # Add the line to output
            output_lines.append(line)
            i += 1
            continue
        
        # Check if we're exiting blocks based on indentation
        while indentation_stack and current_indent <= indentation_stack[-1]:
            indentation_stack.pop()
        
        # Check if this is a debug function call
        is_debug_call = re.search(rf'^\s*({debug_pattern})\s*\(', line)
        
        if is_debug_call:
            # This is a debug call, skip it
            i += 1
            
            # Update empty block tracking - if we're in a block and skipping a debug call,
            # don't mark the block as having non-debug content
            continue
        else:
            # Non-debug line - mark all containing blocks as having content
            for block in empty_block_candidates:
                if current_indent > block['indent']:
                    block['has_non_debug_content'] = True
                    
            # Keep this line
            output_lines.append(line)
            i += 1
    
    return '\n'.join(output_lines)