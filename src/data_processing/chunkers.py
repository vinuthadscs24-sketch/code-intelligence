import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Any

import tiktoken
from loguru import logger
from pydantic import Field
from tree_sitter import Language, Parser, Node

from src import config
from src.data_processing.document_loader import LoadedDocument


# ============================================================
# Token Counting
# ============================================================

_token_encoders: Dict[str, Any] = {}


def count_document_tokens(
    text: str,
    model_name: str = config.EMBEDDING_MODEL_NAME
) -> int:
    """Count tokens using tiktoken."""

    if model_name not in _token_encoders:
        try:
            _token_encoders[model_name] = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning(
                f"Model {model_name} not found. Using cl100k_base encoding."
            )
            _token_encoders[model_name] = tiktoken.get_encoding("cl100k_base")

    try:
        return len(_token_encoders[model_name].encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {e}. Approximating with len/4.")
        return len(text) // 4


# ============================================================
# Document Chunk Model
# ============================================================

class DocumentChunk(LoadedDocument):
    """Represents a chunk of a document ready for embedding."""

    original_file_path: Path = Field(
        description="Relative path of the original file."
    )

    chunk_id: int = Field(
        description="Sequential ID of the chunk within the original document."
    )

    start_char_offset: Optional[int] = Field(
        None,
        description="Start character/byte offset in original content."
    )

    end_char_offset: Optional[int] = Field(
        None,
        description="End character/byte offset in original content."
    )

    code_construct_type: Optional[str] = Field(
        None,
        description="Type of code construct such as function or class."
    )

    code_construct_name: Optional[str] = Field(
        None,
        description="Name of the code construct."
    )

    start_line: Optional[int] = Field(
        None,
        description="Start line number."
    )

    end_line: Optional[int] = Field(
        None,
        description="End line number."
    )

    class Config:
        arbitrary_types_allowed = True


# ============================================================
# Base Chunker
# ============================================================

class BaseChunker(ABC):

    @abstractmethod
    def chunk_document(
        self,
        document: LoadedDocument
    ) -> List[DocumentChunk]:
        """Chunk one document."""
        pass

    def chunk_documents(
        self,
        documents: List[LoadedDocument]
    ) -> List[DocumentChunk]:

        all_chunks = []

        for doc in documents:
            try:
                chunks = self.chunk_document(doc)
                all_chunks.extend(chunks)

            except Exception as e:
                logger.error(
                    f"Failed to chunk document {doc.file_path}: {e}"
                )

        return all_chunks


# ============================================================
# Tree-sitter Global Caches
# ============================================================

_ts_languages: Dict[str, Language] = {}
_ts_parsers: Dict[str, Parser] = {}


# ============================================================
# Tree-sitter Parser Loader
# ============================================================

def get_tree_sitter_parser(
    language_name: str
) -> Optional[Parser]:
    """
    Load Tree-sitter parser using modern language packages.

    Supported:
        python
        java
        javascript
        typescript
    """

    # Return cached parser
    if language_name in _ts_parsers:
        return _ts_parsers[language_name]

    # --------------------------------------------------------
    # Check configuration
    # --------------------------------------------------------

    lang_config = config.TREE_SITTER_LANGUAGES.get(language_name)

    if not lang_config:
        logger.warning(
            f"No tree-sitter configuration found for language: "
            f"{language_name}"
        )
        return None

    # --------------------------------------------------------
    # Modern Tree-sitter language packages
    # --------------------------------------------------------

    try:

        # Python
        if language_name == "python":
            import tree_sitter_python
            language = Language(tree_sitter_python.language())

        # Java
        elif language_name == "java":
            import tree_sitter_java
            language = Language(tree_sitter_java.language())

        # JavaScript
        elif language_name == "javascript":
            import tree_sitter_javascript
            language = Language(tree_sitter_javascript.language())

        # TypeScript
        elif language_name == "typescript":
            import tree_sitter_typescript

            language = Language(
                tree_sitter_typescript.language_typescript()
            )

        else:
            logger.warning(
                f"No modern Tree-sitter package handler "
                f"implemented for: {language_name}"
            )
            return None

        # Cache language
        _ts_languages[language_name] = language

        # ----------------------------------------------------
        # Create parser
        # ----------------------------------------------------

        parser = Parser(language)

        _ts_parsers[language_name] = parser

        logger.info(
            f"Successfully initialized Tree-sitter parser "
            f"for language: {language_name}"
        )

        return parser

    except ImportError as e:

        logger.error(
            f"Tree-sitter grammar package missing for "
            f"{language_name}: {e}"
        )

    except Exception as e:

        logger.error(
            f"Failed to initialize Tree-sitter parser "
            f"for {language_name}: {e}"
        )

    return None


# ============================================================
# TreeSitterChunker
# ============================================================

class TreeSitterChunker(BaseChunker):

    """
    Chunks source code using Tree-sitter AST nodes.
    """

    def __init__(
        self,
        max_chunk_tokens: int = config.AST_CHUNK_MAX_TOKENS,
        min_chunk_tokens: int = config.AST_CHUNK_MIN_TOKENS,
        fallback_chunker: Optional[BaseChunker] = None
    ):

        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.fallback_chunker = fallback_chunker

        logger.info(
            f"TreeSitterChunker initialized. "
            f"Max tokens: {max_chunk_tokens}, "
            f"Min tokens: {min_chunk_tokens}"
        )

        # Significant AST nodes
        self.significant_node_types_map: Dict[str, List[str]] = {

            "python": [
                "function_definition",
                "class_definition"
            ],

            "java": [
                "method_declaration",
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
                "constructor_declaration"
            ],

            "javascript": [
                "function_declaration",
                "class_declaration",
                "method_definition",
                "lexical_declaration",
                "variable_declaration"
            ],

            "typescript": [
                "function_declaration",
                "class_declaration",
                "method_definition",
                "interface_declaration",
                "enum_declaration",
                "lexical_declaration",
                "variable_declaration"
            ]
        }


    # ========================================================
    # Extract Construct Name
    # ========================================================

    def _get_node_name(
        self,
        node: Node,
        language: str
    ) -> str:

        try:

            name_node = node.child_by_field_name("name")

            if name_node:

                return name_node.text.decode("utf-8")

        except Exception:
            pass

        return "anonymous_construct"


    # ========================================================
    # Recursive AST Traversal
    # ========================================================

    def _extract_chunks_from_node(
        self,
        node: Node,
        document: LoadedDocument,
        chunk_id_counter: List[int]
    ) -> List[DocumentChunk]:

        chunks: List[DocumentChunk] = []

        language = document.language

        if not language:
            return chunks

        significant_node_types = (
            self.significant_node_types_map.get(
                language,
                []
            )
        )

        try:
            node_text = node.text.decode("utf-8")
        except Exception:
            return chunks

        node_tokens = count_document_tokens(node_text)

        is_significant_node = (
            node.type in significant_node_types
        )

        # ----------------------------------------------------
        # Valid AST chunk
        # ----------------------------------------------------

        if (
            is_significant_node
            and node_tokens >= self.min_chunk_tokens
        ):

            # Node is small enough
            if node_tokens <= self.max_chunk_tokens:

                chunk_id_counter[0] += 1

                chunks.append(
                    DocumentChunk(

                        file_path=Path(
                            f"{document.file_path}"
                            f"_chunk_{chunk_id_counter[0]}"
                        ),

                        absolute_path=document.absolute_path,

                        content=node_text,

                        language=language,

                        size_bytes=len(
                            node_text.encode("utf-8")
                        ),

                        original_file_path=document.file_path,

                        chunk_id=chunk_id_counter[0],

                        start_char_offset=node.start_byte,

                        end_char_offset=node.end_byte,

                        code_construct_type=node.type,

                        code_construct_name=self._get_node_name(
                            node,
                            language
                        ),

                        start_line=node.start_point[0] + 1,

                        end_line=node.end_point[0] + 1
                    )
                )

                # Don't recursively split this node
                return chunks

            # ------------------------------------------------
            # Node too large
            # ------------------------------------------------

            else:

                logger.debug(
                    f"Node '{node.type}' in "
                    f"{document.file_path} is too large "
                    f"({node_tokens} tokens). "
                    f"Processing children."
                )

        # ----------------------------------------------------
        # Traverse children
        # ----------------------------------------------------

        for child_node in node.children:

            chunks.extend(
                self._extract_chunks_from_node(
                    child_node,
                    document,
                    chunk_id_counter
                )
            )

        return chunks


    # ========================================================
    # Chunk Document
    # ========================================================

    def chunk_document(
        self,
        document: LoadedDocument
    ) -> List[DocumentChunk]:

        # ----------------------------------------------------
        # Language not supported
        # ----------------------------------------------------

        if (
            not document.language
            or document.language
            not in config.TREE_SITTER_LANGUAGES
        ):

            logger.debug(
                f"Language '{document.language}' not configured "
                f"for Tree-sitter. Using fallback for: "
                f"{document.file_path}"
            )

            if self.fallback_chunker:

                return self.fallback_chunker.chunk_document(
                    document
                )

            return [
                DocumentChunk(
                    **document.model_dump(),

                    original_file_path=document.file_path,

                    chunk_id=0,

                    start_char_offset=0,

                    end_char_offset=len(
                        document.content.encode("utf-8")
                    )
                )
            ]

        # ----------------------------------------------------
        # Get parser
        # ----------------------------------------------------

        parser = get_tree_sitter_parser(
            document.language
        )

        if not parser:

            logger.warning(
                f"No Tree-sitter parser for language "
                f"'{document.language}'. "
                f"Using fallback for: {document.file_path}"
            )

            if self.fallback_chunker:

                return self.fallback_chunker.chunk_document(
                    document
                )

            return [
                DocumentChunk(
                    **document.model_dump(),

                    original_file_path=document.file_path,

                    chunk_id=0,

                    start_char_offset=0,

                    end_char_offset=len(
                        document.content.encode("utf-8")
                    )
                )
            ]

        # ----------------------------------------------------
        # Parse AST
        # ----------------------------------------------------

        logger.debug(
            f"AST Chunking: "
            f"{document.file_path} "
            f"(Language: {document.language})"
        )

        try:

            tree = parser.parse(
                bytes(
                    document.content,
                    "utf-8"
                )
            )

            root_node = tree.root_node

            chunk_id_counter = [0]

            ast_chunks = (
                self._extract_chunks_from_node(
                    root_node,
                    document,
                    chunk_id_counter
                )
            )

            # ------------------------------------------------
            # No AST chunks
            # ------------------------------------------------

            if not ast_chunks:

                logger.info(
                    f"AST chunking yielded no specific "
                    f"chunks for {document.file_path}. "
                    f"Using fallback."
                )

                if self.fallback_chunker:

                    return self.fallback_chunker.chunk_document(
                        document
                    )

                doc_tokens = count_document_tokens(
                    document.content
                )

                if doc_tokens <= self.max_chunk_tokens:

                    return [
                        DocumentChunk(
                            **document.model_dump(),

                            original_file_path=document.file_path,

                            chunk_id=0,

                            start_char_offset=0,

                            end_char_offset=len(
                                document.content.encode(
                                    "utf-8"
                                )
                            )
                        )
                    ]

                return [
                    DocumentChunk(
                        **document.model_dump(),

                        original_file_path=document.file_path,

                        chunk_id=0,

                        start_char_offset=0,

                        end_char_offset=len(
                            document.content.encode(
                                "utf-8"
                            )
                        )
                    )
                ]

            # ------------------------------------------------
            # Post-processing
            # ------------------------------------------------

            final_chunks = []

            for chunk in ast_chunks:

                chunk_tokens = count_document_tokens(
                    chunk.content
                )

                # AST chunk too large
                if (
                    chunk_tokens > self.max_chunk_tokens
                    and self.fallback_chunker
                ):

                    logger.debug(
                        f"AST chunk '{chunk.code_construct_name}' "
                        f"is too large "
                        f"({chunk_tokens} tokens). "
                        f"Applying fallback."
                    )

                    temp_doc = LoadedDocument(

                        file_path=chunk.original_file_path,

                        absolute_path=chunk.absolute_path,

                        content=chunk.content,

                        language=chunk.language,

                        size_bytes=chunk.size_bytes
                    )

                    fallback_sub_chunks = (
                        self.fallback_chunker.chunk_document(
                            temp_doc
                        )
                    )

                    for i, sub_chunk in enumerate(
                        fallback_sub_chunks
                    ):

                        sub_chunk.original_file_path = (
                            chunk.original_file_path
                        )

                        sub_chunk.chunk_id = int(
                            f"{chunk.chunk_id}{i:03d}"
                        )

                        sub_chunk.code_construct_type = (
                            f"{chunk.code_construct_type}"
                            f"_sub_split"
                        )

                        sub_chunk.code_construct_name = (
                            chunk.code_construct_name
                        )

                        sub_chunk.start_line = (
                            chunk.start_line
                        )

                        sub_chunk.end_line = (
                            chunk.end_line
                        )

                    final_chunks.extend(
                        fallback_sub_chunks
                    )

                else:

                    final_chunks.append(chunk)

            logger.info(
                f"AST chunking for "
                f"{document.file_path} produced "
                f"{len(final_chunks)} chunks."
            )

            return final_chunks

        except Exception as e:

            logger.error(
                f"Error during AST chunking for "
                f"{document.file_path}: {e}. "
                f"Using fallback."
            )

            if self.fallback_chunker:

                return self.fallback_chunker.chunk_document(
                    document
                )

            return [
                DocumentChunk(
                    **document.model_dump(),

                    original_file_path=document.file_path,

                    chunk_id=0,

                    start_char_offset=0,

                    end_char_offset=len(
                        document.content.encode(
                            "utf-8"
                        )
                    )
                )
            ]


# ============================================================
# Token Splitter
# ============================================================

class TokenSplitter(BaseChunker):

    """
    Token-based text splitter.

    Used for:
        - Markdown
        - Plain text
        - Unsupported languages
        - AST fallback
    """

    def __init__(
        self,
        chunk_size: int = config.TEXT_CHUNK_SIZE,
        chunk_overlap: int = config.TEXT_CHUNK_OVERLAP,
        embedding_model_name: str = config.EMBEDDING_MODEL_NAME
    ):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name

        try:

            self.tokenizer = (
                tiktoken.encoding_for_model(
                    self.embedding_model_name
                )
            )

        except KeyError:

            logger.warning(
                f"Model {self.embedding_model_name} "
                f"not found for tiktoken. "
                f"Using cl100k_base."
            )

            self.tokenizer = (
                tiktoken.get_encoding(
                    "cl100k_base"
                )
            )

        logger.info(
            f"TokenSplitter initialized. "
            f"Chunk size: {chunk_size} tokens, "
            f"Overlap: {chunk_overlap} tokens."
        )


    def chunk_document(
        self,
        document: LoadedDocument
    ) -> List[DocumentChunk]:

        # ----------------------------------------------------
        # Empty document
        # ----------------------------------------------------

        if not document.content.strip():

            logger.debug(
                f"Document {document.file_path} "
                f"is empty or whitespace-only. "
                f"Skipping chunking."
            )

            return []

        # ----------------------------------------------------
        # Encode
        # ----------------------------------------------------

        tokens = self.tokenizer.encode(
            document.content
        )

        chunks = []

        chunk_id = 0

        start_token_idx = 0

        # ----------------------------------------------------
        # Sliding window
        # ----------------------------------------------------

        while start_token_idx < len(tokens):

            end_token_idx = min(
                start_token_idx + self.chunk_size,
                len(tokens)
            )

            chunk_tokens = tokens[
                start_token_idx:end_token_idx
            ]

            # ------------------------------------------------
            # Decode
            # ------------------------------------------------

            try:

                chunk_text = self.tokenizer.decode(
                    chunk_tokens
                )

            except Exception as e:

                logger.error(
                    f"Error decoding token chunk for "
                    f"{document.file_path}: {e}"
                )

                chunk_text = "".join(

                    self.tokenizer
                    .decode_single_token_bytes(token)
                    .decode(
                        "utf-8",
                        errors="replace"
                    )

                    for token in chunk_tokens
                )

            chunk_id += 1

            chunks.append(
                DocumentChunk(

                    file_path=Path(
                        f"{document.file_path}"
                        f"_chunk_{chunk_id}"
                    ),

                    absolute_path=document.absolute_path,

                    content=chunk_text,

                    language=document.language,

                    size_bytes=len(
                        chunk_text.encode("utf-8")
                    ),

                    original_file_path=document.file_path,

                    chunk_id=chunk_id,

                    start_char_offset=None,

                    end_char_offset=None
                )
            )

            # ------------------------------------------------
            # End
            # ------------------------------------------------

            if end_token_idx == len(tokens):

                break

            # ------------------------------------------------
            # Move forward with overlap
            # ------------------------------------------------

            start_token_idx += (
                self.chunk_size
                - self.chunk_overlap
            )

        logger.debug(
            f"TokenSplitter for "
            f"{document.file_path} produced "
            f"{len(chunks)} chunks."
        )

        return chunks


# ============================================================
# Main Code RAG Chunker
# ============================================================

class CodeRAGChunker(BaseChunker):

    """
    Main chunking orchestrator.

    Code:
        Tree-sitter AST chunking

    Other files:
        TokenSplitter

    AST failures:
        TokenSplitter fallback
    """

    def __init__(self):

        self.text_splitter = TokenSplitter()

        self.tree_sitter_chunker = (
            TreeSitterChunker(
                fallback_chunker=self.text_splitter
            )
        )

        logger.info(
            "CodeRAGChunker initialized with "
            "TreeSitterChunker and "
            "TokenSplitter fallback."
        )


    def chunk_document(
        self,
        document: LoadedDocument
    ) -> List[DocumentChunk]:

        if (
            document.language
            in config.TREE_SITTER_LANGUAGES
        ):

            logger.debug(
                f"Using TreeSitterChunker for: "
                f"{document.file_path} "
                f"(Lang: {document.language})"
            )

            return (
                self.tree_sitter_chunker
                .chunk_document(document)
            )

        else:

            logger.debug(
                f"Using TokenSplitter fallback for: "
                f"{document.file_path} "
                f"(Lang: "
                f"{document.language or 'unknown'})"
            )

            return (
                self.text_splitter
                .chunk_document(document)
            )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    logger.remove()

    logger.add(
        lambda msg: print(msg, end=""),
        level="DEBUG"
    )

    # --------------------------------------------------------
    # Python test
    # --------------------------------------------------------

    dummy_py_content = """
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


class Calculator:

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
"""

    # --------------------------------------------------------
    # Markdown test
    # --------------------------------------------------------

    dummy_md_content = """
# Title

This is some markdown content.

- Item 1
- Item 2
"""

    documents_to_chunk = [

        LoadedDocument(
            file_path=Path("dummy_module.py"),

            absolute_path=Path(
                "/abs/dummy_module.py"
            ),

            content=dummy_py_content,

            language="python",

            size_bytes=len(
                dummy_py_content.encode()
            )
        ),

        LoadedDocument(
            file_path=Path("notes.md"),

            absolute_path=Path(
                "/abs/notes.md"
            ),

            content=dummy_md_content,

            language="markdown",

            size_bytes=len(
                dummy_md_content.encode()
            )
        ),

        LoadedDocument(
            file_path=Path("empty.py"),

            absolute_path=Path(
                "/abs/empty.py"
            ),

            content="",

            language="python",

            size_bytes=0
        ),

        LoadedDocument(
            file_path=Path("small.py"),

            absolute_path=Path(
                "/abs/small.py"
            ),

            content="x = 1",

            language="python",

            size_bytes=5
        )
    ]

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    main_chunker = CodeRAGChunker()

    all_final_chunks = (
        main_chunker.chunk_documents(
            documents_to_chunk
        )
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    logger.info(
        f"\n--- Total Chunks Produced: "
        f"{len(all_final_chunks)} ---\n"
    )

    for i, chunk in enumerate(
        all_final_chunks
    ):

        logger.info(
            f"Chunk {i + 1}:\n"
        )

        logger.info(
            f"  Original File: "
            f"{chunk.original_file_path}\n"
        )

        logger.info(
            f"  Chunk ID: "
            f"{chunk.chunk_id}\n"
        )

        logger.info(
            f"  Language: "
            f"{chunk.language}\n"
        )

        logger.info(
            f"  Type: "
            f"{chunk.code_construct_type or 'text'}\n"
        )

        if chunk.code_construct_name:

            logger.info(
                f"  Name: "
                f"{chunk.code_construct_name}\n"
            )

        if (
            chunk.start_line
            and chunk.end_line
        ):

            logger.info(
                f"  Lines: "
                f"{chunk.start_line}-"
                f"{chunk.end_line}\n"
            )

        preview = (
            chunk.content[:150]
            .replace(os.linesep, " ")
        )

        logger.info(
            f"  Content Preview: "
            f"{preview}...\n"
        )

        logger.info(
            f"  Token count: "
            f"{count_document_tokens(chunk.content)}\n"
        )

        logger.info(
            "-" * 30
            + "\n"
        )