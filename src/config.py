import os
from pathlib import Path
from dotenv import load_dotenv
import yaml
from typing import List, Dict, Optional, Any
from loguru import logger


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

env_path = PROJECT_ROOT / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"Loaded environment variables from: {env_path}")
else:
    logger.info(
        f".env file not found at {env_path}. "
        "Relying on system environment variables."
    )


# ============================================================
# YAML CONFIGURATION
# ============================================================

CONFIG_FILE_PATH = PROJECT_ROOT / "config.yaml"
APP_CONFIG: Dict[str, Any] = {}

if CONFIG_FILE_PATH.exists():
    try:
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            APP_CONFIG = yaml.safe_load(f) or {}

        logger.info(
            f"Loaded application configuration from: {CONFIG_FILE_PATH}"
        )

    except Exception as e:
        logger.error(
            f"Error loading {CONFIG_FILE_PATH}: {e}. "
            "Using defaults and environment variables."
        )
else:
    logger.info(
        f"Configuration file {CONFIG_FILE_PATH} not found. "
        "Using defaults and environment variables."
    )


# ============================================================
# CONFIGURATION HELPER
# ============================================================

def get_config_value(
    key_path: str,
    default: Optional[Any] = None,
    env_var: Optional[str] = None
) -> Any:
    """
    Retrieves a configuration value in this priority:

    1. Environment variable
    2. config.yaml
    3. Default value
    """

    # --------------------------------------------------------
    # 1. Environment variable
    # --------------------------------------------------------

    if env_var:
        value = os.getenv(env_var)

        if value is not None:

            if value.lower() in ["true", "false"]:
                return value.lower() == "true"

            if value.isdigit():
                return int(value)

            try:
                return float(value)
            except ValueError:
                pass

            return value

    # --------------------------------------------------------
    # 2. YAML configuration
    # --------------------------------------------------------

    keys = key_path.split(".")
    value = APP_CONFIG

    try:
        for key in keys:
            value = value[key]

        if value is not None:
            return value

    except (KeyError, TypeError):
        pass

    # --------------------------------------------------------
    # 3. Default
    # --------------------------------------------------------

    return default


# ============================================================
# API KEYS & BASE URLS
# ============================================================

# Default API Key to "ollama" if omitted
OPENAI_API_KEY = get_config_value(
    "api_keys.openai",
    "ollama",
    env_var="OPENAI_API_KEY"
)

GOOGLE_API_KEY = get_config_value(
    "api_keys.google",
    env_var="GOOGLE_API_KEY"
)

# Default OpenAI Base URL to local Ollama OpenAI-compatible endpoint
OPENAI_BASE_URL = get_config_value(
    "url.openai",
    "http://localhost:11434/v1",
    env_var="OPENAI_BASE_URL"
)


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = get_config_value(
    "url.ollama",
    "http://localhost:11434",
    "OLLAMA_BASE_URL"
)

OLLAMA_EMBEDDING_MODEL = get_config_value(
    "embedding.ollama_model",
    "nomic-embed-text",
    "OLLAMA_EMBEDDING_MODEL"
)


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path(
    get_config_value(
        "paths.data_dir",
        PROJECT_ROOT / "data",
        "RAG_DATA_DIR"
    )
)

REPOS_DIR = Path(
    get_config_value(
        "paths.repos_dir",
        DATA_DIR / "repositories",
        "RAG_REPOS_DIR"
    )
)

INDEX_DIR = Path(
    get_config_value(
        "paths.index_dir",
        DATA_DIR / "indexes",
        "RAG_INDEX_DIR"
    )
)

LOG_DIR = Path(
    get_config_value(
        "paths.log_dir",
        PROJECT_ROOT / "logs",
        "RAG_LOG_DIR"
    )
)

GRAMMAR_DIR = Path(
    get_config_value(
        "paths.grammar_dir",
        PROJECT_ROOT / "grammars",
        "RAG_GRAMMAR_DIR"
    )
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

DATA_DIR.mkdir(parents=True, exist_ok=True)
REPOS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
GRAMMAR_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = get_config_value(
    "logging.level",
    "INFO",
    "RAG_LOG_LEVEL"
).upper()

LOG_FILE_PATH = LOG_DIR / get_config_value(
    "logging.file_name",
    "rag_app.log",
    "RAG_LOG_FILE"
)

logger.add(
    LOG_FILE_PATH,
    level=LOG_LEVEL,
    rotation="10 MB",
    retention="7 days",
    compression="zip"
)

logger.info(
    f"Logging configured. Level: {LOG_LEVEL}, "
    f"File: {LOG_FILE_PATH}"
)


# ============================================================
# EMBEDDING CONFIGURATION
# ============================================================

EMBEDDING_MODEL_PROVIDER = get_config_value(
    "embedding.provider",
    "ollama",
    "RAG_EMBEDDING_PROVIDER"
)

EMBEDDING_MODEL_NAME = get_config_value(
    "embedding.model_name",
    "nomic-embed-text",
    "RAG_EMBEDDING_MODEL_NAME"
)

EMBEDDING_DIMENSIONS = get_config_value(
    "embedding.dimensions",
    768,
    "RAG_EMBEDDING_DIMENSIONS"
)

EMBEDDING_BATCH_SIZE = get_config_value(
    "embedding.batch_size",
    32,
    "RAG_EMBEDDING_BATCH_SIZE"
)


# ============================================================
# GENERATOR / LLM CONFIGURATION
# ============================================================

GENERATOR_MODEL_NAME = get_config_value(
    "generator.model_name",
    "llama3",
    "RAG_GENERATOR_MODEL_NAME"
)

GENERATOR_TEMPERATURE = get_config_value(
    "generator.temperature",
    0.7,
    "RAG_GENERATOR_TEMPERATURE"
)

GENERATOR_PROMPT = get_config_value(
    "generator.prompt",
    """
You are an expert AI assistant specializing in analyzing
and explaining code repositories.

Your responses should be accurate, concise, and directly
answer the user's query based on the provided context.

Format your answers using Markdown.

Cite specific file paths, classes, methods, and code
constructs when relevant.
""",
    "RAG_GENERATOR_PROMPT"
)


# ============================================================
# TREE-SITTER CONFIGURATION
# ============================================================

TREE_SITTER_LANGUAGES: Dict[str, Dict[str, Any]] = get_config_value(
    "tree_sitter.languages",
    {
        "python": {
            "extensions": [".py"],
            "grammar_name": "python"
        },

        "java": {
            "extensions": [".java"],
            "grammar_name": "java"
        },

        "javascript": {
            "extensions": [".js", ".jsx"],
            "grammar_name": "javascript"
        },

        "typescript": {
            "extensions": [".ts", ".tsx"],
            "grammar_name": "typescript"
        }
    }
)


AST_CHUNK_MAX_TOKENS = get_config_value(
    "tree_sitter.ast_chunk_max_tokens",
    1000,
    "RAG_AST_CHUNK_MAX_TOKENS"
)

AST_CHUNK_MIN_TOKENS = get_config_value(
    "tree_sitter.ast_chunk_min_tokens",
    30,
    "RAG_AST_CHUNK_MIN_TOKENS"
)


# ============================================================
# TEXT SPLITTER CONFIGURATION
# ============================================================

TEXT_CHUNK_SIZE = get_config_value(
    "text_splitter.chunk_size",
    500,
    "RAG_TEXT_CHUNK_SIZE"
)

TEXT_CHUNK_OVERLAP = get_config_value(
    "text_splitter.chunk_overlap",
    50,
    "RAG_TEXT_CHUNK_OVERLAP"
)


# ============================================================
# INDEXING CONFIGURATION
# ============================================================

FAISS_INDEX_FILENAME = get_config_value(
    "indexing.faiss.filename",
    "vector_index.faiss"
)

METADATA_FILENAME = get_config_value(
    "indexing.faiss.metadata_filename",
    "metadata.json"
)

BM25_INDEX_FILENAME = get_config_value(
    "indexing.bm25.filename",
    "bm25_index.pkl"
)


# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

RETRIEVAL_VECTOR_TOP_K = get_config_value(
    "retrieval.vector_top_k",
    20,
    "RAG_RETRIEVAL_VECTOR_TOP_K"
)

RETRIEVAL_BM25_TOP_K = get_config_value(
    "retrieval.bm25_top_k",
    5,
    "RAG_RETRIEVAL_BM25_TOP_K"
)

RRF_CONSTANT_K = get_config_value(
    "retrieval.rrf_k_constant",
    60
)

RETRIEVAL_INDEXES = get_config_value(
    "retrieval.indexes",
    ["vector", "bm25"],
    "RAG_RETRIEVAL_INDEXES"
)


# ============================================================
# FILE PROCESSING CONFIGURATION
# ============================================================

DEFAULT_EXCLUDED_DIRS: List[str] = get_config_value(
    "file_processing.default_excluded_dirs",
    [
        ".git",
        ".idea",
        ".vscode",
        "__pycache__",
        "node_modules",
        "venv",
        ".venv",
        "target",
        "build",
        "dist",
        "docs",
        "examples",
        "tests",
        "test"
    ]
)


DEFAULT_EXCLUDED_FILES: List[str] = get_config_value(
    "file_processing.default_excluded_files",
    [
        "*.min.js",
        "*.min.css",
        "*.lock",
        "*.log",
        ".*",
        "LICENSE"
    ]
)


MAX_FILE_SIZE_MB = get_config_value(
    "file_processing.max_file_size_mb",
    5,
    "RAG_MAX_FILE_SIZE_MB"
)


# ============================================================
# API SERVER CONFIGURATION
# ============================================================

API_HOST = get_config_value(
    "api_server.host",
    "0.0.0.0",
    "RAG_API_HOST"
)

API_PORT = get_config_value(
    "api_server.port",
    8000,
    "RAG_API_PORT"
)

API_RELOAD = get_config_value(
    "api_server.reload",
    False,
    "RAG_API_RELOAD"
)


# ============================================================
# CONFIGURATION LOGGING
# ============================================================

def log_important_configs():

    logger.info("========================================")
    logger.info("        CODE-AWARE RAG CONFIG")
    logger.info("========================================")

    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Data Directory: {DATA_DIR}")
    logger.info(f"Repository Directory: {REPOS_DIR}")
    logger.info(f"Index Directory: {INDEX_DIR}")
    logger.info(f"Embedding Provider: {EMBEDDING_MODEL_PROVIDER}")
    logger.info(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    logger.info(f"Embedding Dimensions: {EMBEDDING_DIMENSIONS}")
    logger.info(f"Embedding Batch Size: {EMBEDDING_BATCH_SIZE}")
    logger.info(f"Ollama URL: {OLLAMA_BASE_URL}")
    logger.info(f"Generator Model: {GENERATOR_MODEL_NAME}")
    logger.info(f"OpenAI Base URL: {OPENAI_BASE_URL}")
    logger.info(f"Supported AST Languages: {list(TREE_SITTER_LANGUAGES.keys())}")
    logger.info(f"API Server: http://{API_HOST}:{API_PORT}")
    logger.info("========================================")


# ============================================================
# OPENAI LLM CLIENT (ROUTED TO LOCAL OLLAMA BY DEFAULT)
# ============================================================

def get_openai_llm_client(
    apikey: Optional[str] = None,
    baseurl: Optional[str] = None
):
    """
    Creates an AsyncOpenAI client routed to local Ollama.
    """
    key = apikey or OPENAI_API_KEY or "ollama"
    url = baseurl or OPENAI_BASE_URL or "http://localhost:11434/v1"

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=key,
            base_url=url
        )

        logger.info(f"AsyncOpenAI client initialized for base_url: {url}")
        return client

    except ImportError:
        logger.error("OpenAI package not installed. Run: pip install openai")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize AsyncOpenAI client: {e}")
        raise


# ============================================================
# OPENAI EMBEDDING CLIENT
# ============================================================

def get_openai_embeddings_client(
    apikey: Optional[str] = None,
    baseurl: Optional[str] = None
):
    """
    Creates an OpenAI client. Kept for backwards compatibility.
    """
    key = apikey or OPENAI_API_KEY or "ollama"
    url = baseurl or OPENAI_BASE_URL or "http://localhost:11434/v1"

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=key,
            base_url=url
        )

        logger.info("OpenAI embedding client initialized.")
        return client

    except ImportError:
        logger.error("OpenAI package not installed. Run: pip install openai")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        raise


# ============================================================
# STARTUP CONFIG LOG
# ============================================================

log_important_configs()