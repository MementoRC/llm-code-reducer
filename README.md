# Code Reducer MCP Server

The Code Reducer MCP server reduces Python code context to minimize token usage in LLM interactions while maintaining edit capability.

## Core Features

- **Comment Removal**: Eliminates code comments while preserving code structure
- **Type Hints Removal**: Strips Python type annotations from function signatures and variable declarations
- **Position Mapping**: Maintains bidirectional mappings between original and reduced code
- **Context Statistics**: Provides reduction metrics and transformation details
- **Database Persistence**: Stores processed files and mappings for future reference
- **Edit Translation**: Translates edit operations between original and processed code contexts
- **Context Management**: Groups related files and manages their relationships for optimized token usage

## MCP Integration

This server implements the Model Context Protocol (MCP) for use with Claude Code and similar LLM tools. It provides tools for reducing Python code to minimize token usage.

### Using with Claude Code

To use the code reducer with Claude Code:

1. Start the server:
   ```bash
   mcp --launch code_reducer
   ```

2. In Claude Code, you can use the code_reducer tool:
   ```
   /use code_reducer
   ```

## API Tools

### File Processing Tools

#### process_python

Process Python code to reduce token count by removing comments and type hints.

**Input:**
```json
{
  "content": "# Python code here\ndef example(arg: str) -> bool:\n    return True",
  "file_name": "example.py",
  "strip_comments": true,
  "strip_type_hints": true,
  "strip_docstrings": false
}
```

**Output:**
```json
{
  "processed_content": "def example(arg):\n    return True",
  "original_lines": 3,
  "processed_lines": 2,
  "reduction_percentage": 30.25,
  "file_id": "file_20250322123456_1",
  "transformations": ["comments", "type_hints"],
  "context_reduction": "30.25%"
}
```

#### get_processed_file

Retrieve a processed file by its ID.

**Input:**
```json
{
  "file_id": "file_20250322123456_1"
}
```

#### list_processed_files

List processed files with pagination.

**Input:**
```json
{
  "limit": 20,
  "offset": 0
}
```

#### delete_processed_file

Delete a processed file.

**Input:**
```json
{
  "file_id": "file_20250322123456_1"
}
```

### Edit Translation Tools

#### translate_position

Translate a position between original and processed code contexts.

**Input:**
```json
{
  "file_id": "file_20250322123456_1",
  "line": 10,
  "character": 5,
  "to_original": true
}
```

#### translate_edit

Translate an edit operation between original and processed code contexts.

**Input:**
```json
{
  "file_id": "file_20250322123456_1",
  "line": 10,
  "character": 5,
  "content": "newCode();",
  "length": 0,
  "to_original": true
}
```

#### translate_and_apply_edit

Translate an edit operation and apply it to the content.

**Input:**
```json
{
  "file_id": "file_20250322123456_1",
  "line": 10,
  "character": 5,
  "content": "newCode();",
  "length": 8,
  "target_content": "if (condition) {\n  oldCode();\n}",
  "to_original": true
}
```

### Context Management Tools

#### create_context

Create a new context for managing related files.

**Input:**
```json
{
  "name": "My Project",
  "description": "Core files for my Python project",
  "configuration": {
    "strip_comments": true,
    "strip_type_hints": true,
    "max_files_per_query": 10,
    "token_budget": 8000
  }
}
```

#### get_context

Get a context by its ID.

**Input:**
```json
{
  "context_id": "ctx_abc123def456"
}
```

#### add_files_to_context

Add files to a context with optional relationship analysis.

**Input:**
```json
{
  "context_id": "ctx_abc123def456",
  "files": [
    {
      "file_id": "file_20250322123456_1",
      "file_path": "/path/to/main.py",
      "importance": 2.0,
      "tags": ["core", "entry-point"]
    },
    {
      "file_id": "file_20250322123457_2",
      "file_path": "/path/to/utils.py",
      "importance": 1.5,
      "tags": ["helper"]
    }
  ],
  "analyze_relationships": true
}
```

#### get_context_files

Get files in a context with optional filtering.

**Input:**
```json
{
  "context_id": "ctx_abc123def456",
  "limit": 20,
  "offset": 0,
  "importance_threshold": 1.0,
  "include_content": true
}
```

#### analyze_context

Analyze file relationships and update importance scores.

**Input:**
```json
{
  "context_id": "ctx_abc123def456"
}
```

#### get_context_summary

Get a human-readable summary of a context.

**Input:**
```json
{
  "context_id": "ctx_abc123def456"
}
```

#### select_files_for_context

Select files from a context optimized for a token budget.

**Input:**
```json
{
  "context_id": "ctx_abc123def456",
  "token_budget": 8000,
  "importance_threshold": 1.0,
  "max_files": 10
}
```

## Database Integration

The Code Reducer MCP server stores processed files and their mappings in a SQLite database for persistence. This allows for:

- Retrieving previously processed files
- Translating positions between original and processed code
- Tracking transformation history
- Managing processed files over time
- Grouping related files into contexts
- Analyzing file relationships

### Database Schema

- **processed_files**: Stores file metadata and content
- **transformations**: Tracks which transformations were applied to each file
- **mappings**: Stores bidirectional position mappings between original and processed code
- **contexts**: Stores context metadata and configuration
- **context_files**: Associates files with contexts and stores importance scores
- **file_relationships**: Tracks relationships between files within contexts

## Edit Translation Service

The Edit Translation Service provides a crucial bridge between code that an LLM sees (processed code with reduced tokens) and the actual code that needs to be edited (original code). It works with the following components:

### Edit Operations

An edit operation consists of:
- A position (line and character)
- Content to insert or replace
- Length of text to replace (0 for insert)

### Translation Process

The service can:
1. Translate an edit operation from processed code to original code (or vice versa)
2. Apply a translated edit to the appropriate content
3. Maintain correct edit semantics across different code representations

### Use Cases

- An LLM reviews token-reduced code and suggests edits
- The Edit Translator converts those edits to the original code context
- The edits can be applied to the original code with proper positioning
- The results maintain the semantic meaning intended by the LLM

## Context Manager Service

The Context Manager Service provides a way to group related files together and manage them as a coherent unit. This allows for more efficient token usage and better context management in LLM interactions.

### Key Features

- **Context Creation**: Group related files into a named context with custom configuration
- **File Relationships**: Automatically analyze and track relationships between files (imports, references, etc.)
- **Importance Scoring**: Assign and automatically update importance scores for files based on centrality
- **Token Budget Management**: Select optimal files to fit within a token budget
- **Context Summarization**: Generate human-readable summaries of contexts

### Use Cases

- Working with multi-file projects in LLM interactions
- Prioritizing important files when token limits are a constraint
- Understanding dependencies between files in a codebase
- Organizing code by functional areas or features
- Generating project overviews for initial LLM context

## Installation

### From PyPI

```bash
pip install mcp-server-code-reducer
```

### From Source

```bash
git clone https://github.com/anthropics/claude-code.git
cd claude-code/git_servers/src/code_reducer
pip install -e .
```

## Running the Server

### As a standalone service:

```bash
mcp-server-code-reducer [--db-path /path/to/database.db]
```

Options:
- `--db-path`: Path to the SQLite database file (default: ~/.code_reducer/code_reducer.db)
- `--debug`: Enable debug logging
- `--host`: Host to bind server to (default: 0.0.0.0)
- `--port`: Port to bind server to (default: 8000)

### Using Docker:

```bash
docker build -t mcp-server-code-reducer .
docker run -v $(pwd)/data:/data -p 8000:8000 -it mcp-server-code-reducer
```

The database will be persisted in the `./data` directory on the host.

### With MCP:

```bash
mcp --launch code_reducer
```

## Development

### Using Pixi (Recommended)

This project now supports [Pixi](https://pixi.sh) for dependency management:

```bash
# Install dependencies
pixi install

# Activate the Pixi environment
pixi shell

# Run tests
pixi run test

# Run linter
pixi run lint

# Run type check
pixi run typecheck

# Run all checks
pixi run check
```

### Using uv (Alternative)

Install development dependencies:

```bash
uv pip install -e ".[dev]"
```

Run tests:

```bash
pytest test/
```