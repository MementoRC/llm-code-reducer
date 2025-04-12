"""
Command-line entry point for the Code Reducer MCP Server.
"""

import sys
import asyncio
import argparse
import logging
from pathlib import Path

from .server import serve


def main():
    """Entry point for the code reducer MCP server."""
    parser = argparse.ArgumentParser(description="Code Reducer MCP Server")
    parser.add_argument(
        "--db-path", 
        help="Path to SQLite database file (default: ~/.code_reducer/code_reducer.db)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    parser.add_argument(
        "--host", 
        default="0.0.0.0", 
        help="Host to bind server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to bind server to (default: 8000)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Ensure database directory exists if specified
    if args.db_path:
        db_dir = Path(args.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run the server
        asyncio.run(
            serve(
                host=args.host, 
                port=args.port, 
                debug=args.debug, 
                db_path=args.db_path
            )
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()