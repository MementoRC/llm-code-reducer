"""
Code reducer MCP server module.
"""

import ast
import builtins
import io
import json
import keyword
import logging
import os
import re
import tokenize
from datetime import datetime, UTC
from enum import Enum
from typing import List, Optional, Tuple, Sequence

import astor
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource

from .context_manager import ContextManager
from .database import CodeReducerDatabase
from .edit_translator import EditTranslator, EditOperation
from .models import (
    Position,
    MappingData,
    ProcessedResult,
    ContextConfiguration
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_code_reducer")


class CodeReducerTools(str, Enum):
    # File processing tools
    PROCESS_PYTHON = "process_python"
    GET_PROCESSED_FILE = "get_processed_file"
    LIST_PROCESSED_FILES = "list_processed_files"
    DELETE_PROCESSED_FILE = "delete_processed_file"

    # Position and edit translation tools
    TRANSLATE_POSITION = "translate_position"
    TRANSLATE_EDIT = "translate_edit"
    TRANSLATE_AND_APPLY_EDIT = "translate_and_apply_edit"

    # Context management tools
    CREATE_CONTEXT = "create_context"
    GET_CONTEXT = "get_context"
    LIST_CONTEXTS = "list_contexts"
    DELETE_CONTEXT = "delete_context"
    ADD_FILES_TO_CONTEXT = "add_files_to_context"
    GET_CONTEXT_FILES = "get_context_files"
    UPDATE_CONTEXT_CONFIG = "update_context_config"
    ANALYZE_CONTEXT = "analyze_context"
    GET_CONTEXT_SUMMARY = "get_context_summary"
    SELECT_FILES_FOR_CONTEXT = "select_files_for_context"


class TypeHintRemover(ast.NodeTransformer):
    """AST transformer that removes type hints."""

    def __init__(self):
        self.removed_nodes = []
        self.node_map = {}

    def visit_FunctionDef(self, node):
        # Remove return type annotations
        if node.returns:
            self.removed_nodes.append((node.returns, 'return_type_hint'))
            node.returns = None

        # Remove argument type annotations
        for arg in node.args.args:
            if arg.annotation:
                self.removed_nodes.append((arg.annotation, 'arg_type_hint'))
                arg.annotation = None

        # Handle keyword-only arguments if present
        for arg in node.args.kwonlyargs:
            if arg.annotation:
                self.removed_nodes.append((arg.annotation, 'kwarg_type_hint'))
                arg.annotation = None

        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        # Handle async functions the same way as regular functions
        return self.visit_FunctionDef(node)

    def visit_AnnAssign(self, node):
        # Handle variable annotations (PEP 526)
        # Convert to regular Assign node
        new_node = ast.Assign(
            targets=[node.target],
            value=node.value if node.value else ast.Constant(value=None),
            lineno=node.lineno,
            col_offset=node.col_offset
        )
        return new_node


class DocstringRemover(ast.NodeTransformer):
    """AST transformer that removes docstrings from functions and classes."""

    def __init__(self):
        self.removed_docstrings = []

    def visit_FunctionDef(self, node):
        # Check if the first statement is a string literal (docstring)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            self.removed_docstrings.append((node.name, node.body[0].value.value))
            node.body = node.body[1:]  # Remove docstring
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        # Handle async functions the same way
        return self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        # Check if the first statement is a string literal (docstring)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            self.removed_docstrings.append((node.name, node.body[0].value.value))
            node.body = node.body[1:]  # Remove docstring
        return self.generic_visit(node)

    def visit_Module(self, node):
        # Check if the first statement is a string literal (module docstring)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            self.removed_docstrings.append(("module", node.body[0].value.value))
            node.body = node.body[1:]  # Remove docstring
        return self.generic_visit(node)


def remove_comments(source_code: str) -> str:
    """Remove comments from Python source code using the tokenize module."""
    io_obj = io.StringIO(source_code)
    out = io.StringIO()
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0

    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]

        # Avoid comments and ensure proper spacing
        if token_type == tokenize.COMMENT:
            continue

        if start_line > last_lineno:
            last_col = 0

        if start_col > last_col:
            out.write(' ' * (start_col - last_col))

        # Add the token to the output stream
        out.write(token_string)

        # Update positions
        last_col = end_col
        last_lineno = end_line

    return out.getvalue()


def remove_debug_functions(source_code: str, debug_functions: List[str]) -> str:
    """
    Remove debug function calls from the code.

    Args:
        source_code: Python source code
        debug_functions: List of function names to be treated as debug functions
                        (e.g., ["log_debug", "logger.debug", "console.print"])

    Returns:
        Processed code with debug function calls removed
    """
    if not debug_functions:
        return source_code

    # Process line by line to handle debug function calls
    lines = source_code.splitlines()
    output_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        is_debug_line = False

        # Create a pattern to match standalone debug function calls
        # but not when they're part of a conditional or other context
        debug_pattern = '|'.join(re.escape(f) for f in debug_functions)

        # This is the key issue - we need to avoid removing print statements inside conditional blocks
        # Only remove debug functions that are directly at the start of a line (with indentation)
        # And not when they're inside an if statement

        # Check if this is a direct debug function call (not inside a conditional or loop)
        # First check if it's a direct function call
        if re.search(rf'^\s*({debug_pattern})\s*\(', line):
            # Now check if it's NOT inside a conditional (by looking at the indentation pattern)
            # We need to see if this line is inside a control structure like an if statement

            # Simple heuristic: Look back to see if we're inside an if-block
            is_inside_conditional = False

            # Check the previous non-empty line
            j = i - 1
            current_indent = len(line) - len(line.lstrip())

            while j >= 0:
                prev_line = lines[j]
                if not prev_line.strip():  # Skip empty lines
                    j -= 1
                    continue

                prev_indent = len(prev_line) - len(prev_line.lstrip())

                # If the previous line has less indentation and contains an if statement,
                # we're likely inside a conditional
                if prev_indent < current_indent and re.search(r'^\s*if\s+.*:\s*$', prev_line):
                    # However, we need to make sure this is what controls our current line
                    # Not just any if statement with less indentation

                    # Check if there are no lines with equal indent between if and our line
                    is_direct_child = True
                    for k in range(j+1, i):
                        if not lines[k].strip():  # Skip empty lines
                            continue

                        k_indent = len(lines[k]) - len(lines[k].lstrip())
                        if k_indent == prev_indent:  # Found another statement at same level as if
                            is_direct_child = False
                            break

                    if is_direct_child:
                        is_inside_conditional = True
                        break

                # If we hit a line with less or equal indentation, and it's not an if statement
                # that controls our line, stop looking back
                if prev_indent <= current_indent:
                    break

                j -= 1

            # If we're not inside a conditional, mark as a debug line to remove
            if not is_inside_conditional:
                is_debug_line = True

        if is_debug_line:
            i += 1
            continue

        # Check for conditional blocks containing only debug calls
        if re.match(r'^\s*if\s+\w+\s*:\s*$', line):
            # Look ahead to see if block only contains debug calls
            j = i + 1
            block_indent = None
            contains_only_debug = True
            has_content = False

            # Determine the indentation of the block
            while j < len(lines):
                # Skip empty lines
                if not lines[j].strip():
                    j += 1
                    continue

                # Get indentation level
                current_indent = len(lines[j]) - len(lines[j].lstrip())

                # Initialize block indent if not set
                if block_indent is None:
                    block_indent = current_indent

                # If we've moved to a line with less indentation, we're out of the block
                if current_indent < block_indent:
                    break

                # If it's at the same indentation level as our block
                if current_indent == block_indent:
                    has_content = True

                    # Check if the line contains a standalone debug function call
                    # Pattern checks if debug function is at start of statement
                    if re.search(rf'^\s*({debug_pattern})\s*\(', lines[j]):
                        # Line contains a debug call
                        j += 1
                        continue
                    else:
                        # Line contains non-debug code at the block level
                        contains_only_debug = False
                        break

                # Continue to the next line
                j += 1

            # If the block only contained debug calls and had actual content, skip it
            if contains_only_debug and has_content and block_indent is not None:
                i = j  # Skip the entire block
                continue

        # Special check for verbose print statement
        if "verbose" in line and any(f in line for f in debug_functions):
            # If this line has a conditional print inside verbose, keep it
            if re.search(r'if\s+verbose', line) or re.search(r'verbose.*print', line):
                pass  # Keep this line

        # Keep the line if it's not a debug line
        output_lines.append(line)
        i += 1

    return '\n'.join(output_lines)


def optimize_whitespace(source_code: str) -> str:
    """
    Optimize whitespace in Python code to further reduce token count.
    
    This function:
    1. Removes trailing whitespace on each line
    2. Collapses multiple consecutive blank lines into a single one
    3. Removes empty lines at the beginning and end of the file
    """
    # Split into lines and remove trailing whitespace
    lines = [line.rstrip() for line in source_code.splitlines()]
    
    # Remove consecutive blank lines
    result_lines = []
    blank_line_count = 0
    
    for line in lines:
        if line.strip() == "":
            blank_line_count += 1
            if blank_line_count <= 1:  # Keep only one blank line in a sequence
                result_lines.append(line)
        else:
            blank_line_count = 0
            result_lines.append(line)
    
    # Remove blank lines at the beginning
    while result_lines and result_lines[0].strip() == "":
        result_lines.pop(0)
        
    # Remove blank lines at the end
    while result_lines and result_lines[-1].strip() == "":
        result_lines.pop()
        
    return "\n".join(result_lines)


def optimize_conditionals(source_code: str) -> str:
    """
    Optimize if conditional statements into one-line ternary expressions where possible.
    
    This function identifies simple if/else blocks that can be converted to ternary operators.
    For example:
    ```
    if condition:
        x = value1
    else:
        x = value2
    ```
    becomes:
    ```
    x = value1 if condition else value2
    ```
    """
    lines = source_code.splitlines()
    i = 0
    result_lines = []
    
    while i < len(lines):
        line = lines[i].rstrip()
        
        # Check for if statement with single line body followed by else with single line body
        if_match = re.match(r'^(\s*)if\s+(.+?)\s*:\s*$', line)
        
        if if_match and i + 3 < len(lines):  # Need at least 4 lines for a simple if/else
            indent = if_match.group(1)
            condition = if_match.group(2)
            
            # Check if next line is indented and a simple assignment
            if_body_line = lines[i + 1]
            if_body_match = re.match(r'^(\s+)(\w+)\s*=\s*(.+)\s*$', if_body_line)
            
            # Check if there's an else on the next line at the same indent level
            else_line = lines[i + 2]
            else_match = re.match(fr'^{re.escape(indent)}else\s*:\s*$', else_line)
            
            # Check if the line after else is indented and a simple assignment to the same variable
            if i + 3 < len(lines):
                else_body_line = lines[i + 3]
                else_body_match = re.match(r'^(\s+)(\w+)\s*=\s*(.+)\s*$', else_body_line)
                
                # If we have a pattern that can be converted to ternary
                if (if_body_match and else_match and else_body_match and 
                    if_body_match.group(2) == else_body_match.group(2)):
                    
                    var_name = if_body_match.group(2)
                    true_value = if_body_match.group(3)
                    false_value = else_body_match.group(3)
                    
                    # Create the ternary expression
                    ternary = f"{indent}{var_name} = {true_value} if {condition} else {false_value}"
                    result_lines.append(ternary)
                    
                    # Skip the if/else block
                    i += 4
                    continue
        
        # If no ternary conversion was possible, keep the original line
        result_lines.append(line)
        i += 1
    
    return "\n".join(result_lines)


def optimize_function_signatures(source_code: str) -> str:
    """
    Optimize function signatures by shortening long parameter lists.

    This function:
    1. Identifies long multi-line function signatures
    2. Compresses them into a more compact form
    """
    # Regular expression to find function definitions that span multiple lines
    pattern = r'def\s+(\w+)\s*\(\s*\n([\s\S]*?)\)\s*(?:->[\s\S]*?)?\s*:'

    def compress_params(match):
        func_name = match.group(1)
        params_block = match.group(2)

        # Extract parameter lines and compress them
        params = []
        for line in params_block.strip().split('\n'):
            param = line.strip().rstrip(',')
            if param:  # Skip empty lines
                params.append(param)

        # Join parameters with minimal spacing
        params_str = ', '.join(params)

        # Return the compressed function signature
        return f"def {func_name}({params_str}):"

    # Replace function signatures
    result = re.sub(pattern, compress_params, source_code)
    return result


def minify_names(source_code: str) -> str:
    """
    Minify variable, function, and parameter names to reduce token count.

    This function:
    1. Preserves built-in names, keywords, and imports
    2. Renames user-defined variables and functions to shorter names
    3. Uses one-letter names for parameters and local variables
    4. Maintains the code's functionality while reducing token length

    Important: This is an advanced transformation that makes code less readable
    but significantly reduces token count. Not recommended for code that needs
    to be maintained or debugged frequently.
    """
    try:
        # Parse the code into an AST
        tree = ast.parse(source_code)

        # Track existing names to avoid conflicts
        reserved_names = set(keyword.kwlist + dir(builtins))
        existing_imports = set()
        name_map = {}  # Maps original names to minified names
        counter = 0

        # First pass: collect import names to avoid renaming them
        class ImportCollector(ast.NodeVisitor):
            def visit_Import(self, node):
                for name in node.names:
                    existing_imports.add(name.name.split('.')[0])
                    if name.asname:
                        existing_imports.add(name.asname)
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                if node.module:
                    existing_imports.add(node.module.split('.')[0])
                for name in node.names:
                    existing_imports.add(name.name)
                    if name.asname:
                        existing_imports.add(name.asname)
                self.generic_visit(node)

        import_collector = ImportCollector()
        import_collector.visit(tree)

        # Helper to generate a new short name
        def get_short_name():
            nonlocal counter
            # Start with single letters, then combinations
            if counter < 26:
                name = chr(97 + counter)  # a-z
            else:
                name = f"_{counter - 26}"
            counter += 1

            # Avoid conflicts with reserved names and imports
            while name in reserved_names or name in existing_imports:
                if counter < 26 * 2:
                    name = chr(97 + (counter % 26)) + str(counter // 26)
                else:
                    name = f"_{counter - 26*2}"
                counter += 1

            return name

        # Second pass: rename variables, functions, and parameters
        class NameMinifier(ast.NodeTransformer):
            def visit_Name(self, node):
                # Only rename variables in a Load context
                if isinstance(node.ctx, ast.Store) and node.id not in reserved_names and node.id not in existing_imports:
                    if node.id not in name_map:
                        name_map[node.id] = get_short_name()
                    node.id = name_map[node.id]
                elif isinstance(node.ctx, ast.Load) and node.id in name_map:
                    node.id = name_map[node.id]
                return node

            def visit_FunctionDef(self, node):
                # Don't rename special methods like __init__
                if not node.name.startswith('__') and node.name not in reserved_names and node.name not in existing_imports:
                    if node.name not in name_map:
                        name_map[node.name] = get_short_name()
                    node.name = name_map[node.name]

                # Rename function parameters
                for arg in node.args.args:
                    if arg.arg not in reserved_names:
                        if arg.arg not in name_map:
                            name_map[arg.arg] = get_short_name()
                        arg.arg = name_map[arg.arg]

                # Process function body
                self.generic_visit(node)
                return node

            def visit_ClassDef(self, node):
                # Don't rename special classes or exceptions
                if not node.name.startswith('__') and not node.name.endswith('Error') and not node.name.endswith('Exception') and node.name not in reserved_names and node.name not in existing_imports:
                    if node.name not in name_map:
                        name_map[node.name] = get_short_name()
                    node.name = name_map[node.name]

                # Process class body
                self.generic_visit(node)
                return node

            def visit_arg(self, node):
                # Additional handling for function arguments in Python 3.8+
                if node.arg not in reserved_names:
                    if node.arg not in name_map:
                        name_map[node.arg] = get_short_name()
                    node.arg = name_map[node.arg]
                return node

        # Apply the name minification
        minifier = NameMinifier()
        transformed_tree = minifier.visit(tree)

        # Generate code from the modified AST
        return astor.to_source(transformed_tree)
    except Exception as e:
        logger.error(f"Error minifying names: {str(e)}")
        return source_code


def remove_unused_imports(source_code: str) -> Tuple[str, List[str]]:
    """
    Identify and remove unused imports from Python code.

    Returns a tuple of (modified_code, removed_imports)
    """
    try:
        # Parse the code into an AST
        tree = ast.parse(source_code)

        # Collect all imports and their line numbers
        imports = {}  # Maps module/name to line number
        import_lines = set()  # Set of line numbers with imports

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports[name.name if not name.asname else name.asname] = node.lineno
                    import_lines.add(node.lineno)
            elif isinstance(node, ast.ImportFrom):
                for name in node.names:
                    if name.name == '*':  # Skip wildcard imports
                        continue
                    imports[name.name if not name.asname else name.asname] = node.lineno
                    import_lines.add(node.lineno)

        # Find used names
        used_names = set()

        class NameCollector(ast.NodeVisitor):
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    used_names.add(node.id)
                self.generic_visit(node)

            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name):
                    # Add parent object name for attribute access
                    used_names.add(node.value.id)
                self.generic_visit(node)

        collector = NameCollector()
        collector.visit(tree)

        # Identify unused imports
        unused_imports = [name for name in imports if name not in used_names]

        if not unused_imports:
            return source_code, []

        # Remove the unused imports by filtering lines
        lines = source_code.splitlines()
        filtered_lines = []
        removed_imports = []

        for i, line in enumerate(lines, 1):
            should_include = True

            # Check if this line is part of an import statement
            if i in import_lines:
                for name in unused_imports:
                    # Simple pattern matching for import statements
                    if re.search(r'\bimport\s+' + re.escape(name) + r'\b', line) or \
                       re.search(r'\bfrom\s+.+\s+import\s+.*\b' + re.escape(name) + r'\b', line):
                        should_include = False
                        removed_imports.append(name)
                        break

            if should_include:
                filtered_lines.append(line)

        return "\n".join(filtered_lines), removed_imports

    except Exception as e:
        logger.error(f"Error removing unused imports: {str(e)}")
        return source_code, []


class CodeProcessor:
    """Processor for reducing code context while maintaining structural integrity."""

    def __init__(self, db: CodeReducerDatabase):
        self.db = db
        self.file_counter = 0

    async def process_python(
        self,
        content: str,
        file_name: Optional[str] = None,
        strip_comments: bool = False,
        strip_type_hints: bool = True,
        strip_docstrings: bool = False,
        strip_debug: bool = False,
        debug_functions: Optional[List[str]] = None,
        optimize_whitespace_flag: bool = True,
        remove_unused_imports: bool = False,
        optimize_function_signatures: bool = False,
        optimize_conditionals: bool = False,
    ) -> ProcessedResult:
        """Process Python code to reduce token count."""
        original_lines = content.count('\n') + 1
        processed_content = content
        mapping_data = MappingData(original_to_processed={}, processed_to_original={})
        
        # Track content at each step for mapping purposes
        content_versions = [content]
        transformation_steps = []

        # Step 1: Remove unused imports if requested
        if remove_unused_imports:
            try:
                processed_content, removed_imports = remove_unused_imports(processed_content)
                if removed_imports:
                    content_versions.append(processed_content)
                    transformation_steps.append("unused_imports")
                    logger.info(f"Removed {len(removed_imports)} unused imports")
            except Exception as e:
                logger.error(f"Error removing unused imports: {str(e)}")

        # Step 2: Remove comments if requested
        if strip_comments:
            try:
                processed_content = remove_comments(processed_content)
                content_versions.append(processed_content)
                transformation_steps.append("comments")
            except Exception as e:
                logger.error(f"Error removing comments: {str(e)}")
        
        # Step 3: Use AST to remove type hints
        if strip_type_hints:
            try:
                tree = ast.parse(processed_content)
                transformer = TypeHintRemover()
                transformed_tree = transformer.visit(tree)
                # Use astor to generate source code from the AST
                processed_content = astor.to_source(transformed_tree)
                content_versions.append(processed_content)
                transformation_steps.append("type_hints")
            except SyntaxError as e:
                # If parsing fails, return with what we've processed so far
                logger.error(f"Syntax error in AST processing: {str(e)}")
                return ProcessedResult(
                    processed_content=processed_content,
                    original_lines=original_lines,
                    processed_lines=processed_content.count('\n') + 1,
                    reduction_percentage=self._calculate_reduction(content, processed_content),
                    mapping_data=mapping_data,
                    transformations=transformation_steps
                )
            except Exception as e:
                logger.error(f"Error in AST processing: {str(e)}")
                # Continue with what we have processed so far

        # Step 4: Remove docstrings if requested
        if strip_docstrings:
            try:
                tree = ast.parse(processed_content)
                docstring_remover = DocstringRemover()
                transformed_tree = docstring_remover.visit(tree)
                # Use astor to generate source code from the AST
                processed_content = astor.to_source(transformed_tree)
                content_versions.append(processed_content)
                transformation_steps.append("docstrings")
                logger.info(f"Removed {len(docstring_remover.removed_docstrings)} docstrings")
            except SyntaxError as e:
                # If parsing fails, return with what we've processed so far
                logger.error(f"Syntax error in docstring removal: {str(e)}")
            except Exception as e:
                logger.error(f"Error removing docstrings: {str(e)}")
                # Continue with what we have processed so far

        # Step 5: Remove debug functions if requested
        if strip_debug and debug_functions:
            try:
                processed_content = remove_debug_functions(processed_content, debug_functions)
                content_versions.append(processed_content)
                transformation_steps.append("debug_functions")
                logger.info(f"Removed debug function calls for: {', '.join(debug_functions)}")
            except Exception as e:
                logger.error(f"Error removing debug functions: {str(e)}")
                # Continue with what we have processed so far

        # Step 6: Optimize function signatures if requested
        if optimize_function_signatures:
            try:
                processed_content = optimize_function_signatures(processed_content)
                content_versions.append(processed_content)
                transformation_steps.append("function_signatures")
                logger.info("Optimized function signatures")
            except Exception as e:
                logger.error(f"Error optimizing function signatures: {str(e)}")
                # Continue with what we have processed so far

        # Step 7: Optimize conditionals if requested
        if optimize_conditionals:
            try:
                processed_content = optimize_conditionals(processed_content)
                content_versions.append(processed_content)
                transformation_steps.append("conditionals")
                logger.info("Optimized conditional statements")
            except Exception as e:
                logger.error(f"Error optimizing conditionals: {str(e)}")
                # Continue with what we have processed so far

        # Step 8: Minify names if requested
        if False:  # Disabled for now
            try:
                processed_content = minify_names(processed_content)
                content_versions.append(processed_content)
                transformation_steps.append("minify_names")
                logger.info("Minified variable and function names")
            except Exception as e:
                logger.error(f"Error minifying names: {str(e)}")
                # Continue with what we have processed so far

        # Step 9: Optimize whitespace if requested
        if optimize_whitespace_flag:
            try:
                processed_content = optimize_whitespace(processed_content)
                content_versions.append(processed_content)
                transformation_steps.append("whitespace")
            except Exception as e:
                logger.error(f"Error optimizing whitespace: {str(e)}")
                # Continue with what we have processed so far

        # Create position mappings between original and processed content
        mapping_data = self._create_mappings(content_versions)
        
        # Generate a file ID
        file_id = f"file_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}_{self.file_counter}"
        self.file_counter += 1
        
        # Calculate final statistics
        processed_lines = processed_content.count('\n') + 1
        reduction_percentage = self._calculate_reduction(content, processed_content)

        # Store in database
        self.db.store_processed_file(
            file_id=file_id,
            original_content=content,
            processed_content=processed_content,
            file_name=file_name,
            original_lines=original_lines,
            processed_lines=processed_lines,
            reduction_percentage=reduction_percentage,
            transformations=transformation_steps,
            mapping_data=mapping_data.model_dump()
        )
        
        return ProcessedResult(
            processed_content=processed_content,
            original_lines=original_lines,
            processed_lines=processed_lines,
            reduction_percentage=reduction_percentage,
            mapping_data=mapping_data,
            file_id=file_id,
            transformations=transformation_steps
        )

    def _calculate_reduction(self, original: str, processed: str) -> float:
        """Calculate the percentage reduction in content size."""
        original_size = len(original)
        processed_size = len(processed)
        
        if original_size == 0:
            return 0.0
            
        reduction = (original_size - processed_size) / original_size * 100
        return round(reduction, 2)

    def _create_mappings(self, content_versions: List[str]) -> MappingData:
        """Create mappings between positions in original and processed content."""
        # This is a simplified implementation that would need to be enhanced
        # for production use with more accurate line and character mappings

        mapping_data = MappingData(original_to_processed={}, processed_to_original={})

        if len(content_versions) < 2:
            return mapping_data

        # Create basic line-based mapping
        original = content_versions[0]
        processed = content_versions[-1]

        original_lines = original.split('\n')
        processed_lines = processed.split('\n')

        # Simple heuristic: map lines based on similarity
        for i, orig_line in enumerate(original_lines):
            best_match = -1
            best_score = -1

            for j, proc_line in enumerate(processed_lines):
                # Simple similarity score
                if len(orig_line.strip()) == 0 or len(proc_line.strip()) == 0:
                    continue

                # Count matching characters (ignoring whitespace differences)
                orig_stripped = orig_line.strip()
                proc_stripped = proc_line.strip()

                # Count matching characters
                matches = sum(1 for a, b in zip(orig_stripped, proc_stripped) if a == b)
                score = matches / max(len(orig_stripped), len(proc_stripped))

                if score > 0.6 and score > best_score:  # Threshold for considering a match
                    best_score = score
                    best_match = j

            if best_match >= 0:
                # Store the mapping
                orig_pos = Position(line=i, character=0)
                proc_pos = Position(line=best_match, character=0)

                mapping_data.original_to_processed[str(orig_pos)] = proc_pos
                mapping_data.processed_to_original[str(proc_pos)] = orig_pos

        return mapping_data

    def translate_position(self, file_id: str, position: Position, to_original: bool = True) -> Optional[Position]:
        """
        Translate a position between original and processed code.

        Args:
            file_id: The file ID
            position: The position to translate
            to_original: If True, translate from processed to original, otherwise from original to processed

        Returns:
            The translated position or None if not found
        """
        file_record = self.db.get_processed_file(file_id)
        if not file_record:
            return None

        pos_str = str(position)

        if to_original and pos_str in file_record.mapping_data.processed_to_original:
            return file_record.mapping_data.processed_to_original[pos_str]
        elif not to_original and pos_str in file_record.mapping_data.original_to_processed:
            return file_record.mapping_data.original_to_processed[pos_str]

        return None


async def serve(host: str = "0.0.0.0", port: int = 8000, debug: bool = False, db_path: str = None) -> None:
    """Start the code reducer MCP server."""
    server = Server("mcp-code-reducer")

    # Initialize database and services
    db = CodeReducerDatabase(db_path)
    code_processor = CodeProcessor(db)
    edit_translator = EditTranslator(db)
    context_manager = ContextManager(db)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available code reducer tools."""
        tools = [
            # File processing tools
            Tool(
                name=CodeReducerTools.PROCESS_PYTHON.value,
                description="Process Python code to reduce token count by removing comments, type hints, docstrings, and debug functions",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Python code content to process",
                        },
                        "file_name": {
                            "type": "string",
                            "description": "Name of the file (for reference only)",
                        },
                        "strip_comments": {
                            "type": "boolean",
                            "description": "Whether to strip comments from the code (default: False)",
                            "default": False,
                        },
                        "strip_type_hints": {
                            "type": "boolean",
                            "description": "Whether to strip type hints from the code (default: True)",
                            "default": True,
                        },
                        "strip_docstrings": {
                            "type": "boolean",
                            "description": "Whether to strip docstrings from the code (default: False)",
                            "default": False,
                        },
                        "strip_debug": {
                            "type": "boolean",
                            "description": "Whether to strip debug function calls from the code (default: False)",
                            "default": False,
                        },
                        "debug_functions": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of function names to treat as debug functions (e.g., ['print', 'logger.debug', 'console.log'])",
                        },
                        "optimize_whitespace": {
                            "type": "boolean",
                            "description": "Whether to optimize whitespace to further reduce token count (default: True)",
                            "default": True,
                        },
                        "remove_unused_imports": {
                            "type": "boolean",
                            "description": "Whether to remove unused imports from the code (default: False)",
                            "default": False,
                        },
                        "optimize_function_signatures": {
                            "type": "boolean",
                            "description": "Whether to compress multi-line function signatures (default: False)",
                            "default": False,
                        },
                        "optimize_conditionals": {
                            "type": "boolean",
                            "description": "Whether to convert simple if/else blocks to ternary expressions (default: False)",
                            "default": False,
                        },
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name=CodeReducerTools.GET_PROCESSED_FILE.value,
                description="Retrieve a processed file by its ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "Unique identifier for the file",
                        },
                    },
                    "required": ["file_id"],
                },
            ),
            Tool(
                name=CodeReducerTools.LIST_PROCESSED_FILES.value,
                description="List processed files with pagination",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of files to return (default: 20)",
                            "default": 20,
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Offset for pagination (default: 0)",
                            "default": 0,
                        },
                    },
                },
            ),
            Tool(
                name=CodeReducerTools.DELETE_PROCESSED_FILE.value,
                description="Delete a processed file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "Unique identifier for the file to delete",
                        },
                    },
                    "required": ["file_id"],
                },
            ),

            # Position and edit translation tools
            Tool(
                name=CodeReducerTools.TRANSLATE_POSITION.value,
                description="Translate a position between original and processed code contexts",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "Unique identifier for the file",
                        },
                        "line": {
                            "type": "integer",
                            "description": "Line number of the position",
                        },
                        "character": {
                            "type": "integer",
                            "description": "Character position in the line",
                        },
                        "to_original": {
                            "type": "boolean",
                            "description": "If true, translate from processed to original, otherwise from original to processed",
                            "default": True,
                        },
                    },
                    "required": ["file_id", "line", "character"],
                },
            ),
            Tool(
                name=CodeReducerTools.TRANSLATE_EDIT.value,
                description="Translate an edit operation between original and processed code contexts",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "Unique identifier for the file",
                        },
                        "line": {
                            "type": "integer",
                            "description": "Line number of the edit position",
                        },
                        "character": {
                            "type": "integer",
                            "description": "Character position in the line",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to insert or replace",
                        },
                        "length": {
                            "type": "integer",
                            "description": "Length of text to replace (0 for insert)",
                            "default": 0,
                        },
                        "to_original": {
                            "type": "boolean",
                            "description": "If true, translate from processed to original, otherwise from original to processed",
                            "default": True,
                        },
                    },
                    "required": ["file_id", "line", "character", "content"],
                },
            ),
            Tool(
                name=CodeReducerTools.TRANSLATE_AND_APPLY_EDIT.value,
                description="Translate an edit operation and apply it to the content",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "Unique identifier for the file",
                        },
                        "line": {
                            "type": "integer",
                            "description": "Line number of the edit position",
                        },
                        "character": {
                            "type": "integer",
                            "description": "Character position in the line",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to insert or replace",
                        },
                        "length": {
                            "type": "integer",
                            "description": "Length of text to replace (0 for insert)",
                            "default": 0,
                        },
                        "target_content": {
                            "type": "string",
                            "description": "Content to apply the edit to (if not provided, will use the original or processed content from the file record)",
                        },
                        "to_original": {
                            "type": "boolean",
                            "description": "If true, translate from processed to original, otherwise from original to processed",
                            "default": True,
                        },
                    },
                    "required": ["file_id", "line", "character", "content"],
                },
            ),

            # Context management tools
            Tool(
                name=CodeReducerTools.CREATE_CONTEXT.value,
                description="Create a new context for managing related files",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the context",
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description of the context",
                        },
                        "configuration": {
                            "type": "object",
                            "description": "Optional configuration settings",
                            "properties": {
                                "strip_comments": {
                                    "type": "boolean",
                                    "description": "Whether to strip comments from all files in the context",
                                    "default": False,
                                },
                                "strip_docstrings": {
                                    "type": "boolean",
                                    "description": "Whether to strip docstrings from all files in the context",
                                    "default": False,
                                },
                                "strip_type_hints": {
                                    "type": "boolean",
                                    "description": "Whether to strip type hints from all files in the context",
                                    "default": True,
                                },
                                "max_files_per_query": {
                                    "type": "integer",
                                    "description": "Maximum number of files to return per query",
                                    "default": 10,
                                },
                                "prioritize_by_importance": {
                                    "type": "boolean",
                                    "description": "Whether to prioritize files by importance",
                                    "default": True,
                                },
                                "include_summaries": {
                                    "type": "boolean",
                                    "description": "Whether to include file summaries",
                                    "default": False,
                                },
                                "token_budget": {
                                    "type": "integer",
                                    "description": "Maximum token budget for context",
                                },
                            },
                        },
                    },
                    "required": ["name"],
                },
            ),
            Tool(
                name=CodeReducerTools.GET_CONTEXT.value,
                description="Get a context by its ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "Unique identifier for the context",
                        },
                    },
                    "required": ["context_id"],
                },
            ),
            Tool(
                name=CodeReducerTools.LIST_CONTEXTS.value,
                description="List available contexts with pagination",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of contexts to return (default: 20)",
                            "default": 20,
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Offset for pagination (default: 0)",
                            "default": 0,
                        },
                    },
                },
            ),
            Tool(
                name=CodeReducerTools.DELETE_CONTEXT.value,
                description="Delete a context and all its file associations",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "Unique identifier for the context to delete",
                        },
                    },
                    "required": ["context_id"],
                },
            ),
            Tool(
                name=CodeReducerTools.ADD_FILES_TO_CONTEXT.value,
                description="Add files to a context",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "Unique identifier for the context",
                        },
                        "files": {
                            "type": "array",
                            "description": "List of files to add",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file_id": {
                                        "type": "string",
                                        "description": "ID of a processed file",
                                    },
                                    "file_path": {
                                        "type": "string",
                                        "description": "Path of the file in its original location",
                                    },
                                    "importance": {
                                        "type": "number",
                                        "description": "Importance score (0.0-3.0)",
                                        "default": 1.0,
                                    },
                                    "tags": {
                                        "type": "array",
                                        "description": "Tags for categorization",
                                        "items": {
                                            "type": "string",
                                        },
                                    },
                                },
                                "required": ["file_id", "file_path"],
                            },
                        },
                        "analyze_relationships": {
                            "type": "boolean",
                            "description": "Whether to analyze file relationships",
                            "default": True,
                        },
                    },
                    "required": ["context_id", "files"],
                },
            ),
            Tool(
                name=CodeReducerTools.GET_CONTEXT_FILES.value,
                description="Get files in a context",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "Unique identifier for the context",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of files to return",
                            "default": 20,
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Offset for pagination",
                            "default": 0,
                        },
                        "importance_threshold": {
                            "type": "number",
                            "description": "Minimum importance score",
                        },
                        "include_content": {
                            "type": "boolean",
                            "description": "Whether to include file content",
                            "default": False,
                        },
                    },
                    "required": ["context_id"],
                },
            ),
            Tool(
                name=CodeReducerTools.UPDATE_CONTEXT_CONFIG.value,
                description="Update configuration for a context",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "Unique identifier for the context",
                        },
                        "configuration": {
                            "type": "object",
                            "description": "Configuration settings",
                            "properties": {
                                "strip_comments": {
                                    "type": "boolean",
                                    "description": "Whether to strip comments from code",
                                },
                                "strip_docstrings": {
                                    "type": "boolean",
                                    "description": "Whether to strip docstrings from code",
                                },
                                "strip_type_hints": {
                                    "type": "boolean",
                                    "description": "Whether to strip type hints from code",
                                },
                                "max_files_per_query": {
                                    "type": "integer",
                                    "description": "Maximum files to return per query",
                                },
                                "prioritize_by_importance": {
                                    "type": "boolean",
                                    "description": "Prioritize files by importance score",
                                },
                                "include_summaries": {
                                    "type": "boolean",
                                    "description": "Include file summaries",
                                },
                                "token_budget": {
                                    "type": "integer",
                                    "description": "Maximum token budget",
                                },
                            },
                        },
                    },
                    "required": ["context_id", "configuration"],
                },
            ),
            Tool(
                name=CodeReducerTools.ANALYZE_CONTEXT.value,
                description="Analyze file relationships and update importance scores",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "Unique identifier for the context",
                        },
                    },
                    "required": ["context_id"],
                },
            ),
            Tool(
                name=CodeReducerTools.GET_CONTEXT_SUMMARY.value,
                description="Get a human-readable summary of a context",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "Unique identifier for the context",
                        },
                    },
                    "required": ["context_id"],
                },
            ),
            Tool(
                name=CodeReducerTools.SELECT_FILES_FOR_CONTEXT.value,
                description="Select files from a context optimized for a token budget",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "Unique identifier for the context",
                        },
                        "token_budget": {
                            "type": "integer",
                            "description": "Maximum token budget",
                        },
                        "importance_threshold": {
                            "type": "number",
                            "description": "Minimum importance score",
                        },
                        "max_files": {
                            "type": "integer",
                            "description": "Maximum number of files to include",
                        },
                        "include_content": {
                            "type": "boolean",
                            "description": "Whether to include file content",
                            "default": True,
                        },
                    },
                    "required": ["context_id"],
                },
            ),
        ]

        return tools

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        """Handle tool calls for code reducer operations."""
        try:
            if name == CodeReducerTools.PROCESS_PYTHON.value:
                content = arguments.get("content")
                if not content:
                    raise ValueError("Missing required argument: content")

                file_name = arguments.get("file_name")
                strip_comments = arguments.get("strip_comments", False)
                strip_type_hints = arguments.get("strip_type_hints", True)
                strip_docstrings = arguments.get("strip_docstrings", False)
                strip_debug = arguments.get("strip_debug", False)
                debug_functions = arguments.get("debug_functions")
                optimize_whitespace = arguments.get("optimize_whitespace", True)

                # Get additional optimization parameters
                remove_unused_imports = arguments.get("remove_unused_imports", False)
                optimize_function_signatures = arguments.get("optimize_function_signatures", False)
                optimize_conditionals = arguments.get("optimize_conditionals", False)

                result = await code_processor.process_python(
                    content=content,
                    file_name=file_name,
                    strip_comments=strip_comments,
                    strip_type_hints=strip_type_hints,
                    strip_docstrings=strip_docstrings,
                    strip_debug=strip_debug,
                    debug_functions=debug_functions,
                    optimize_whitespace_flag=optimize_whitespace,
                    remove_unused_imports=remove_unused_imports,
                    optimize_function_signatures=optimize_function_signatures,
                    optimize_conditionals=optimize_conditionals,
                )

                # Convert the result to a simplified dict for the output
                output_result = {
                    "processed_content": result.processed_content,
                    "original_lines": result.original_lines,
                    "processed_lines": result.processed_lines,
                    "reduction_percentage": result.reduction_percentage,
                    "file_id": result.file_id,
                    "transformations": result.transformations,
                    "context_reduction": f"{result.reduction_percentage:.2f}%",
                }

                return [TextContent(type="text", text=json.dumps(output_result, indent=2))]

            elif name == CodeReducerTools.GET_PROCESSED_FILE.value:
                file_id = arguments.get("file_id")
                if not file_id:
                    raise ValueError("Missing required argument: file_id")

                file_record = db.get_processed_file(file_id)
                if not file_record:
                    return [TextContent(type="text", text=f"File not found: {file_id}")]

                output = {
                    "file_id": file_record.file_id,
                    "file_name": file_record.file_name,
                    "original_lines": file_record.original_lines,
                    "processed_lines": file_record.processed_lines,
                    "reduction_percentage": file_record.reduction_percentage,
                    "transformations": file_record.transformations,
                    "created_at": file_record.created_at,
                    "updated_at": file_record.updated_at,
                    "original_content": file_record.original_content,
                    "processed_content": file_record.processed_content,
                }

                return [TextContent(type="text", text=json.dumps(output, indent=2))]

            elif name == CodeReducerTools.LIST_PROCESSED_FILES.value:
                limit = arguments.get("limit", 20)
                offset = arguments.get("offset", 0)

                files = db.list_processed_files(limit, offset)

                return [TextContent(type="text", text=json.dumps(files, indent=2))]

            elif name == CodeReducerTools.DELETE_PROCESSED_FILE.value:
                file_id = arguments.get("file_id")
                if not file_id:
                    raise ValueError("Missing required argument: file_id")

                success = db.delete_processed_file(file_id)

                return [TextContent(type="text", text=json.dumps({
                    "success": success,
                    "message": f"File deleted: {file_id}" if success else f"Failed to delete file: {file_id}"
                }, indent=2))]

            elif name == CodeReducerTools.TRANSLATE_POSITION.value:
                file_id = arguments.get("file_id")
                line = arguments.get("line")
                character = arguments.get("character")
                to_original = arguments.get("to_original", True)

                if not file_id or line is None or character is None:
                    raise ValueError("Missing required arguments: file_id, line, and character are required")

                position = Position(line=line, character=character)
                translated = code_processor.translate_position(file_id, position, to_original)

                if not translated:
                    return [TextContent(type="text", text=json.dumps({
                        "success": False,
                        "message": f"Failed to translate position: no mapping found"
                    }, indent=2))]

                result = {
                    "success": True,
                    "original_position": {
                        "line": translated.line,
                        "character": translated.character
                    } if to_original else {
                        "line": position.line,
                        "character": position.character
                    },
                    "processed_position": {
                        "line": position.line,
                        "character": position.character
                    } if to_original else {
                        "line": translated.line,
                        "character": translated.character
                    },
                    "file_id": file_id
                }

                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == CodeReducerTools.TRANSLATE_EDIT.value:
                file_id = arguments.get("file_id")
                line = arguments.get("line")
                character = arguments.get("character")
                content = arguments.get("content")
                length = arguments.get("length", 0)
                to_original = arguments.get("to_original", True)

                if not file_id or line is None or character is None or content is None:
                    raise ValueError("Missing required arguments: file_id, line, character, and content are required")

                # Create the edit operation
                edit = EditOperation(
                    position=Position(line=line, character=character),
                    content=content,
                    length=length
                )

                # Translate the edit
                translated_edit = edit_translator.translate_edit(file_id, edit, to_original)

                if not translated_edit:
                    return [TextContent(type="text", text=json.dumps({
                        "success": False,
                        "message": "Failed to translate edit: no mapping found"
                    }, indent=2))]

                result = {
                    "success": True,
                    "original_edit": {
                        "line": translated_edit.position.line if to_original else edit.position.line,
                        "character": translated_edit.position.character if to_original else edit.position.character,
                        "content": translated_edit.content,
                        "length": translated_edit.length
                    },
                    "processed_edit": {
                        "line": edit.position.line if to_original else translated_edit.position.line,
                        "character": edit.position.character if to_original else translated_edit.position.character,
                        "content": edit.content,
                        "length": edit.length
                    },
                    "file_id": file_id
                }

                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == CodeReducerTools.TRANSLATE_AND_APPLY_EDIT.value:
                file_id = arguments.get("file_id")
                line = arguments.get("line")
                character = arguments.get("character")
                content = arguments.get("content")
                length = arguments.get("length", 0)
                target_content = arguments.get("target_content")
                to_original = arguments.get("to_original", True)

                if not file_id or line is None or character is None or content is None:
                    raise ValueError("Missing required arguments: file_id, line, character, and content are required")

                # Get the file record if target_content not provided
                if not target_content:
                    file_record = db.get_processed_file(file_id)
                    if not file_record:
                        return [TextContent(type="text", text=json.dumps({
                            "success": False,
                            "message": f"File not found: {file_id}"
                        }, indent=2))]

                    target_content = file_record.original_content if to_original else file_record.processed_content

                # Create the edit operation
                edit = EditOperation(
                    position=Position(line=line, character=character),
                    content=content,
                    length=length
                )

                # Translate and apply the edit
                edited_content, translated_edit = edit_translator.translate_and_apply_edit(
                    file_id, target_content, edit, to_original
                )

                if not edited_content or not translated_edit:
                    return [TextContent(type="text", text=json.dumps({
                        "success": False,
                        "message": "Failed to translate and apply edit: no mapping found"
                    }, indent=2))]

                result = {
                    "success": True,
                    "edited_content": edited_content,
                    "translated_edit": {
                        "line": translated_edit.position.line,
                        "character": translated_edit.position.character,
                        "content": translated_edit.content,
                        "length": translated_edit.length
                    },
                    "file_id": file_id
                }

                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            # Context Management Tools
            elif name == CodeReducerTools.CREATE_CONTEXT.value:
                name = arguments.get("name")
                description = arguments.get("description")
                configuration = arguments.get("configuration")

                if not name:
                    raise ValueError("Missing required argument: name")

                # Parse configuration if provided
                config_obj = None
                if configuration:
                    config_obj = ContextConfiguration.model_validate(configuration)

                # Create context
                context_id = await context_manager.create_context(
                    name=name,
                    description=description,
                    configuration=config_obj
                )

                if not context_id:
                    return [TextContent(type="text", text=json.dumps({
                        "success": False,
                        "message": "Failed to create context"
                    }, indent=2))]

                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "context_id": context_id,
                    "message": f"Context created successfully: {name}"
                }, indent=2))]

            elif name == CodeReducerTools.GET_CONTEXT.value:
                context_id = arguments.get("context_id")

                if not context_id:
                    raise ValueError("Missing required argument: context_id")

                context = db.get_context(context_id)
                if not context:
                    return [TextContent(type="text", text=json.dumps({
                        "success": False,
                        "message": f"Context not found: {context_id}"
                    }, indent=2))]

                # Convert to dict for JSON serialization
                context_dict = context.model_dump()

                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "context": context_dict
                }, indent=2))]

            elif name == CodeReducerTools.LIST_CONTEXTS.value:
                limit = arguments.get("limit", 20)
                offset = arguments.get("offset", 0)

                contexts = db.list_contexts(limit, offset)

                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "contexts": contexts,
                    "count": len(contexts)
                }, indent=2))]

            elif name == CodeReducerTools.DELETE_CONTEXT.value:
                context_id = arguments.get("context_id")

                if not context_id:
                    raise ValueError("Missing required argument: context_id")

                success = db.delete_context(context_id)

                return [TextContent(type="text", text=json.dumps({
                    "success": success,
                    "message": f"Context deleted: {context_id}" if success else f"Failed to delete context: {context_id}"
                }, indent=2))]

            elif name == CodeReducerTools.ADD_FILES_TO_CONTEXT.value:
                context_id = arguments.get("context_id")
                files = arguments.get("files", [])
                analyze_relationships = arguments.get("analyze_relationships", True)

                if not context_id:
                    raise ValueError("Missing required argument: context_id")

                if not files:
                    raise ValueError("Missing required argument: files (must be a non-empty array)")

                # Track successes and failures
                results = {
                    "success": True,
                    "added": [],
                    "failed": []
                }

                for file_info in files:
                    file_id = file_info.get("file_id")
                    file_path = file_info.get("file_path")

                    if not file_id or not file_path:
                        results["failed"].append({
                            "file_id": file_id,
                            "reason": "Missing required fields: file_id and file_path"
                        })
                        continue

                    # Get file name from path
                    file_name = os.path.basename(file_path)
                    importance = file_info.get("importance", 1.0)
                    tags = file_info.get("tags", [])

                    # Add file to context
                    success = await context_manager.add_file_to_context(
                        context_id=context_id,
                        file_id=file_id,
                        file_path=file_path,
                        file_name=file_name,
                        importance=importance,
                        tags=tags,
                        analyze_relationships=analyze_relationships
                    )

                    if success:
                        results["added"].append(file_id)
                    else:
                        results["failed"].append({
                            "file_id": file_id,
                            "reason": "Failed to add file to context"
                        })

                # Set overall success to false if any files failed
                if results["failed"]:
                    results["success"] = False

                return [TextContent(type="text", text=json.dumps(results, indent=2))]

            elif name == CodeReducerTools.GET_CONTEXT_FILES.value:
                context_id = arguments.get("context_id")
                limit = arguments.get("limit", 20)
                offset = arguments.get("offset", 0)
                importance_threshold = arguments.get("importance_threshold")
                include_content = arguments.get("include_content", False)

                if not context_id:
                    raise ValueError("Missing required argument: context_id")

                files = db.get_context_files(
                    context_id=context_id,
                    limit=limit,
                    offset=offset,
                    importance_threshold=importance_threshold,
                    include_content=include_content
                )

                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "files": files,
                    "count": len(files)
                }, indent=2))]

            elif name == CodeReducerTools.UPDATE_CONTEXT_CONFIG.value:
                context_id = arguments.get("context_id")
                configuration = arguments.get("configuration")

                if not context_id or not configuration:
                    raise ValueError("Missing required arguments: context_id and configuration")

                # Parse configuration
                config_obj = ContextConfiguration.model_validate(configuration)

                # Update configuration
                success = db.update_context_configuration(context_id, config_obj)

                return [TextContent(type="text", text=json.dumps({
                    "success": success,
                    "message": f"Configuration updated for context: {context_id}" if success else f"Failed to update configuration: {context_id}"
                }, indent=2))]

            elif name == CodeReducerTools.ANALYZE_CONTEXT.value:
                context_id = arguments.get("context_id")

                if not context_id:
                    raise ValueError("Missing required argument: context_id")

                # Analyze file importance
                success = await context_manager.analyze_file_importance(context_id)

                return [TextContent(type="text", text=json.dumps({
                    "success": success,
                    "message": f"Context analyzed successfully: {context_id}" if success else f"Failed to analyze context: {context_id}"
                }, indent=2))]

            elif name == CodeReducerTools.GET_CONTEXT_SUMMARY.value:
                context_id = arguments.get("context_id")

                if not context_id:
                    raise ValueError("Missing required argument: context_id")

                # Generate summary text
                summary = await context_manager.generate_context_summary_text(context_id)

                if not summary:
                    return [TextContent(type="text", text=json.dumps({
                        "success": False,
                        "message": f"Failed to generate summary for context: {context_id}"
                    }, indent=2))]

                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "summary": summary
                }, indent=2))]

            elif name == CodeReducerTools.SELECT_FILES_FOR_CONTEXT.value:
                context_id = arguments.get("context_id")
                token_budget = arguments.get("token_budget")
                importance_threshold = arguments.get("importance_threshold")
                max_files = arguments.get("max_files")
                include_content = arguments.get("include_content", True)

                if not context_id:
                    raise ValueError("Missing required argument: context_id")

                # Select files
                files = await context_manager.select_files_for_context(
                    context_id=context_id,
                    token_budget=token_budget,
                    importance_threshold=importance_threshold,
                    max_files=max_files
                )

                # Filter content if not requested
                if not include_content:
                    for file in files:
                        if 'original_content' in file:
                            del file['original_content']
                        if 'processed_content' in file:
                            del file['processed_content']

                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "files": files,
                    "count": len(files)
                }, indent=2))]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            logger.error(f"Error processing code-reducer query: {str(e)}")
            raise ValueError(f"Error processing code-reducer query: {str(e)}")

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options)


if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser(description="Code Reducer MCP Server")
    parser.add_argument("--db-path", help="Path to SQLite database file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server to")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    asyncio.run(serve(host=args.host, port=args.port, debug=args.debug, db_path=args.db_path))