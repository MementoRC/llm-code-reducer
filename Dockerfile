# Use Python 3.10 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY test/ ./test/

# Install dependencies
RUN pip install --no-cache-dir -e .

# Create a volume for persisting database
VOLUME /data

# Expose port
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["mcp-server-code-reducer", "--db-path", "/data/code_reducer.db"]