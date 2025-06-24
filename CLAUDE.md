# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Code-Aware RAG (Retrieval-Augmented Generation) system designed specifically for code repositories. Unlike traditional RAG systems, it uses tree-sitter for intelligent AST-based code chunking and combines vector search (FAISS) with sparse search (BM25) using Reciprocal Rank Fusion.

## Common Commands

### Development Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
python download_nltk_data.py  # Download NLTK data

# Configuration
cp config.example.yaml config.yaml  # Edit as needed
# cp .env.example .env  # Optional, for API keys
```

### Running the Application
```bash
# Start the FastAPI server
python main.py

# Alternative ways to run
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

The API server runs at http://0.0.0.0:8000 by default (configurable via config.yaml or environment variables).

### Docker
```bash
# Build and run with Docker Compose
docker-compose up --build
```

## Architecture Overview

### Core Pipeline (RAGPipeline)
- **Entry Point**: `src/pipeline.py` - Orchestrates the entire RAG workflow
- **Repository Processing**: Clones/loads repositories → chunks code → builds indexes
- **Query Processing**: Takes queries → retrieves relevant chunks → generates responses

### Key Components

**Data Processing (`src/data_processing/`)**:
- `document_loader.py`: Loads and filters files from repositories
- `chunkers.py`: AST-based intelligent code chunking using tree-sitter

**Indexing (`src/indexing/`)**:
- `vector_index.py`: FAISS-based semantic vector search
- `sparse_index.py`: BM25-based keyword search

**Retrieval (`src/retrieval/`)**:
- `retriever.py`: HybridRetriever combining vector + sparse search with RRF

**Generation (`src/generation/`)**:
- `generator.py`: LLM interaction for query rewriting and response generation

**API (`src/api.py`)**:
- FastAPI application with async endpoints
- Background repository setup with status tracking
- Streaming query responses

### Configuration System
- **Primary**: `config.yaml` for application settings
- **Secondary**: `.env` for API keys (environment variables take precedence)
- **Management**: `src/config.py` handles all configuration loading with fallbacks

### Key API Endpoints
- `POST /v1/code-rag/repository/setup` - Setup and index a repository (background task)
- `GET /v1/code-rag/repository/status/{repo_id}` - Check setup status
- `POST /v1/code-rag/query/stream` - Query with streaming response

## Important Implementation Details

### Tree-sitter Integration
- Supports multiple languages (Python, Java, JavaScript, TypeScript)
- Language configurations in `config.yaml` under `tree_sitter.languages`
- AST nodes chunked by logical units (functions, classes) with configurable token limits

### Search Strategy
- **Vector Search**: Uses embeddings for semantic similarity (default: text-embedding-3-small)
- **Sparse Search**: BM25 for exact keyword matching
- **Fusion**: Reciprocal Rank Fusion combines both approaches
- Configurable top-k values for each search method

### Authentication
- If API keys not configured, service runs in "no-configured apikey mode"
- Clients must provide API key via `Authorization: Bearer {apikey}` header
- OpenAI API key can be set via config.yaml or OPENAI_API_KEY environment variable

### File Processing
- Excludes common directories: `.git`, `node_modules`, `venv`, `__pycache__`, etc.
- Max file size limit: 5MB by default
- Supports custom inclusion/exclusion patterns

## Testing Status

**No testing infrastructure currently exists.** This is a significant gap that should be addressed.

To add testing:
1. Add testing dependencies: `pytest`, `pytest-asyncio`, `httpx`, `pytest-cov`
2. Create `tests/` directory with test files
3. Test the API endpoints, pipeline components, and core functionality

## Dependencies

Key libraries:
- **FastAPI + Uvicorn**: Web framework and ASGI server
- **Tree-sitter**: Code parsing and AST construction
- **FAISS**: Vector similarity search
- **OpenAI**: Embeddings and LLM generation
- **GitPython**: Git repository interaction
- **Loguru**: Enhanced logging
- **Jinja2**: Prompt templating

## Configuration Notes

- All paths are configurable via `config.yaml`
- Default data directory: `./data/` (contains repositories and indexes)
- Logging configured in `src/config.py` with file rotation
- Model names and API endpoints configurable for different providers