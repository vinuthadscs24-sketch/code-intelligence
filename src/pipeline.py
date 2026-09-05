import pickle
from src.graph_builder import CodeKnowledgeGraph
from pathlib import Path
from typing import Optional, List, Dict, Any
import pickle
import shutil

from loguru import logger

from src import config

from src.data_processing.document_loader import (
    load_documents_from_repo,
    clone_repository,
    LoadedDocument,
)

from src.data_processing.chunkers import (
    CodeRAGChunker,
    DocumentChunk,
)

from src.indexing.vector_index import FaissVectorIndex
from src.indexing.sparse_index import BM25Index
from src.hybrid_retriever import HybridRetriever



class RAGPipeline:
    """
    Main orchestration layer for the Code Intelligence backend.

    Pipeline:

        Repository
            ↓
        Documents
            ↓
        AST-aware chunks
            ├── FAISS
            ├── BM25
            └── Knowledge Graph
                    ├── Classes
                    ├── Methods
                    ├── HAS_METHOD
                    ├── INJECTS
                    └── CALLS
    """

    KNOWLEDGE_GRAPH_FILENAME = "knowledge_graph.pkl"

    def __init__(
        self,
        repo_id: str,
        indexes: list[str] = config.RETRIEVAL_INDEXES,
        index_base_dir: Path = config.INDEX_DIR,
        repos_base_dir: Path = config.REPOS_DIR,
        model: str = config.GENERATOR_MODEL_NAME,
        temperature: float = config.GENERATOR_TEMPERATURE,
    ):
        if not repo_id:
            raise ValueError("repo_id cannot be empty.")

        self.repo_id = (
            repo_id
            .replace("/", "_")
            .replace(":", "_")
        )

        self.repository_path: Optional[Path] = None

        self.index_dir = (
            Path(index_base_dir) / self.repo_id
        )

        self.cloned_repo_path = (
            Path(repos_base_dir) / self.repo_id
        )

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.cloned_repo_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model = model
        self.temperature = temperature

        logger.info(
            f"RAGPipeline initialized for repo_id: "
            f"'{self.repo_id}'."
        )

        logger.info(
            f"  Index directory: {self.index_dir}"
        )

        logger.info(
            f"  Cloned repo directory: "
            f"{self.cloned_repo_path}"
        )

        # --------------------------------------------------------
        # Core components
        # --------------------------------------------------------

        self.chunker = CodeRAGChunker()

        self.indexes = indexes

        self.retriever = HybridRetriever(
            index_dir=self.index_dir,
            indexes=self.indexes,
        )

        # --------------------------------------------------------
        # Knowledge graph
        # --------------------------------------------------------

        self.graph_file = (
            self.index_dir /
            self.KNOWLEDGE_GRAPH_FILENAME
        )

        self.knowledge_graph: Optional[
            CodeKnowledgeGraph
        ] = None

        self._load_knowledge_graph()

    # ============================================================
    # REPOSITORY SETUP
    # ============================================================

    def setup_repository(
        self,
        repo_url_or_path: str,
        access_token: Optional[str] = None,
        force_reclone: bool = False,
        force_reindex: bool = False,
        apikey: Optional[str] = None,
    ) -> bool:

        logger.info(
            f"Setting up repository: "
            f"{repo_url_or_path} "
            f"for repo_id: {self.repo_id}"
        )

        # --------------------------------------------------------
        # 1. Repository
        # --------------------------------------------------------

        is_url = (
            repo_url_or_path.startswith("http://")
            or repo_url_or_path.startswith("https://")
        )

        if is_url:

            self.repository_path = (
                self.cloned_repo_path
            )

            if (
                force_reclone
                and self.repository_path.exists()
            ):
                logger.info(
                    "Force reclone enabled. "
                    f"Deleting {self.repository_path}"
                )

                try:
                    shutil.rmtree(
                        self.repository_path
                    )

                    self.repository_path.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                except Exception as exc:
                    logger.error(
                        "Failed to delete existing "
                        f"repository: {exc}"
                    )
                    return False

            if not clone_repository(
                repo_url_or_path,
                self.repository_path,
                access_token,
            ):
                logger.error(
                    f"Failed to clone repository: "
                    f"{repo_url_or_path}"
                )
                return False

            logger.info(
                "Repository successfully set up at "
                f"{self.repository_path}"
            )

        else:

            local_path = Path(
                repo_url_or_path
            )

            if (
                not local_path.exists()
                or not local_path.is_dir()
            ):
                logger.error(
                    "Local repository path does not "
                    f"exist: {local_path}"
                )
                return False

            self.repository_path = local_path

            logger.info(
                f"Using local repository at: "
                f"{self.repository_path}"
            )

        # --------------------------------------------------------
        # 2. Existing indexes + graph
        # --------------------------------------------------------

        vector_index_file = (
            self.index_dir /
            config.FAISS_INDEX_FILENAME
        )

        bm25_index_file = (
            self.index_dir /
            config.BM25_INDEX_FILENAME
        )

        metadata_file = (
            self.index_dir /
            config.FAISS_METADATA_FILENAME
        )

        indexes_exist = (
            vector_index_file.exists()
            and bm25_index_file.exists()
            and metadata_file.exists()
        )

        graph_exists = (
            self._knowledge_graph_is_valid()
        )

        # --------------------------------------------------------
        # Existing complete repository
        # --------------------------------------------------------

        if (
            not force_reindex
            and indexes_exist
            and graph_exists
        ):

            logger.info(
                f"Indexes and knowledge graph already "
                f"exist for '{self.repo_id}'."
            )

            if (
                not self.retriever.vector_index
                or not self.retriever.bm25_index
            ):
                logger.info(
                    "Reloading retriever indexes..."
                )

                self.retriever._load_indexes()

            return True

        # --------------------------------------------------------
        # Existing indexes but missing graph
        # --------------------------------------------------------

        if (
            not force_reindex
            and indexes_exist
            and not graph_exists
        ):

            logger.info(
                "Retrieval indexes already exist but "
                "knowledge graph is missing."
            )

            return self._build_knowledge_graph_for_repository()

        # --------------------------------------------------------
        # Full rebuild
        # --------------------------------------------------------

        logger.info(
            "Proceeding with full repository indexing. "
            f"Force reindex: {force_reindex}"
        )

        return self._build_indexes_for_repository(
            force_rebuild=force_reindex,
            apikey=apikey,
        )

    # ============================================================
    # BUILD ALL INDEXES
    # ============================================================

    def _build_indexes_for_repository(
        self,
        force_rebuild: bool = False,
        apikey: Optional[str] = None,
    ) -> bool:

        if not self.repository_path:
            logger.error(
                "Repository path not set. "
                "Cannot build indexes."
            )
            return False

        # --------------------------------------------------------
        # Load documents
        # --------------------------------------------------------

        logger.info(
            f"Loading documents from: "
            f"{self.repository_path}"
        )

        docs_iterator = load_documents_from_repo(
            repo_path=self.repository_path,
            excluded_dirs=config.DEFAULT_EXCLUDED_DIRS,
            excluded_files=config.DEFAULT_EXCLUDED_FILES,
            max_file_size_mb=config.MAX_FILE_SIZE_MB,
        )

        loaded_documents: List[
            LoadedDocument
        ] = list(docs_iterator)

        if not loaded_documents:
            logger.warning(
                "No documents loaded. "
                "Indexing cannot proceed."
            )
            return False

        logger.info(
            f"Loaded {len(loaded_documents)} documents."
        )

        # --------------------------------------------------------
        # Chunk documents
        # --------------------------------------------------------

        logger.info(
            "Chunking documents..."
        )

        document_chunks: List[
            DocumentChunk
        ] = self.chunker.chunk_documents(
            loaded_documents
        )

        if not document_chunks:
            logger.warning(
                "No chunks produced. "
                "Indexing cannot proceed."
            )
            return False

        logger.info(
            f"Produced {len(document_chunks)} chunks."
        )

        # --------------------------------------------------------
        # FAISS
        # --------------------------------------------------------

        vector_indexer = FaissVectorIndex(
            index_dir=self.index_dir,
            embedding_dim=config.EMBEDDING_DIMENSIONS,
        )

        logger.info(
            "Building Vector Index..."
        )

        if not vector_indexer.build_index(
            document_chunks,
            force_rebuild=force_rebuild,
            apikey=apikey,
        ):
            logger.error(
                "Failed to build Vector Index."
            )
            return False

        logger.info(
            "Vector Index built successfully."
        )

        # --------------------------------------------------------
        # BM25
        # --------------------------------------------------------

        bm25_indexer = BM25Index(
            index_dir=self.index_dir
        )

        logger.info(
            "Building BM25 Index..."
        )

        if not bm25_indexer.build_index(
            document_chunks,
            force_rebuild=force_rebuild,
        ):
            logger.error(
                "Failed to build BM25 Index."
            )
            return False

        logger.info(
            "BM25 Index built successfully."
        )

        # --------------------------------------------------------
        # Reload retriever
        # --------------------------------------------------------

        logger.info(
            "Reloading indexes in retriever..."
        )

        self.retriever._load_indexes()

        # --------------------------------------------------------
        # KNOWLEDGE GRAPH
        # --------------------------------------------------------

        logger.info(
            "================================================"
        )

        logger.info(
            "Building Knowledge Graph..."
        )

        logger.info(
            f"Graph source chunks: "
            f"{len(document_chunks)}"
        )

        if not self._build_knowledge_graph_from_chunks(
            document_chunks
        ):
            logger.error(
                "Knowledge graph construction failed."
            )
            return False

        logger.info(
            "Knowledge Graph built successfully."
        )

        # --------------------------------------------------------
        # Final summary
        # --------------------------------------------------------

        if self.knowledge_graph:

            summary = (
                self.knowledge_graph.get_summary()
            )

            logger.info(
                f"Knowledge Graph nodes: "
                f"{summary['total_nodes']}"
            )

            logger.info(
                f"Knowledge Graph edges: "
                f"{summary['total_edges']}"
            )

            logger.info(
                f"Graph node types: "
                f"{summary['node_types']}"
            )

            logger.info(
                f"Graph relationships: "
                f"{summary['relationship_types']}"
            )

        logger.info(
            "================================================"
        )

        return True

    # ============================================================
    # BUILD KNOWLEDGE GRAPH
    # ============================================================

    def _build_knowledge_graph_for_repository(
        self,
    ) -> bool:

        if not self.repository_path:
            logger.error(
                "Repository path not set. "
                "Cannot build knowledge graph."
            )
            return False

        logger.info(
            "Loading repository documents for "
            "knowledge graph..."
        )

        docs_iterator = load_documents_from_repo(
            repo_path=self.repository_path,
            excluded_dirs=config.DEFAULT_EXCLUDED_DIRS,
            excluded_files=config.DEFAULT_EXCLUDED_FILES,
            max_file_size_mb=config.MAX_FILE_SIZE_MB,
        )

        loaded_documents = list(
            docs_iterator
        )

        if not loaded_documents:
            logger.error(
                "No documents available for "
                "knowledge graph."
            )
            return False

        logger.info(
            f"Loaded {len(loaded_documents)} documents "
            "for graph construction."
        )

        document_chunks = (
            self.chunker.chunk_documents(
                loaded_documents
            )
        )

        if not document_chunks:
            logger.error(
                "No chunks available for "
                "knowledge graph."
            )
            return False

        logger.info(
            f"Produced {len(document_chunks)} chunks "
            "for graph construction."
        )

        return self._build_knowledge_graph_from_chunks(
            document_chunks
        )

    # ============================================================
    # BUILD GRAPH FROM CHUNKS
    # ============================================================

    def _build_knowledge_graph_from_chunks(
        self,
        document_chunks: List[DocumentChunk],
    ) -> bool:

        try:

            logger.info(
                "Normalizing chunks for knowledge graph..."
            )

            normalized_chunks = []

            for chunk in document_chunks:

                # ------------------------------------------------
                # DocumentChunk may be a Pydantic/dataclass object
                # ------------------------------------------------

                if isinstance(chunk, dict):

                    chunk_dict = dict(chunk)

                elif hasattr(chunk, "model_dump"):

                    chunk_dict = chunk.model_dump()

                elif hasattr(chunk, "dict"):

                    chunk_dict = chunk.dict()

                elif hasattr(chunk, "__dict__"):

                    chunk_dict = vars(chunk).copy()

                else:

                    logger.warning(
                        "Skipping unsupported chunk type: "
                        f"{type(chunk)}"
                    )

                    continue

                # ------------------------------------------------
                # Normalize common field names
                # ------------------------------------------------

                if (
                    "file_name" not in chunk_dict
                    and "file" not in chunk_dict
                ):
                    if "file_path" in chunk_dict:
                        chunk_dict["file_name"] = str(
                            chunk_dict["file_path"]
                        )

                if (
                    "code_content" not in chunk_dict
                    and "content" in chunk_dict
                ):
                    chunk_dict["code_content"] = (
                        chunk_dict["content"]
                    )

                # ------------------------------------------------
                # Ensure graph builder sees calls
                # ------------------------------------------------

                if "calls" not in chunk_dict:

                    if "method_calls" in chunk_dict:
                        chunk_dict["calls"] = (
                            chunk_dict["method_calls"]
                        )

                    elif "relationships" in chunk_dict:
                        chunk_dict["calls"] = (
                            chunk_dict["relationships"]
                        )

                    else:
                        chunk_dict["calls"] = []

                normalized_chunks.append(
                    chunk_dict
                )

            logger.info(
                f"Normalized "
                f"{len(normalized_chunks)} chunks."
            )

            if not normalized_chunks:
                logger.error(
                    "No compatible chunks available "
                    "for knowledge graph."
                )
                return False

            # ----------------------------------------------------
            # Build graph
            # ----------------------------------------------------

            graph = CodeKnowledgeGraph()

            logger.info(
                "Calling CodeKnowledgeGraph."
                "build_graph_from_chunks()..."
            )

            graph.build_graph_from_chunks(
                normalized_chunks
            )

            self.knowledge_graph = graph

            # ----------------------------------------------------
            # Save graph
            # ----------------------------------------------------

            if not self._save_knowledge_graph():
                logger.error(
                    "Knowledge graph could not be saved."
                )
                return False

            summary = (
                graph.get_summary()
            )

            logger.info(
                "Knowledge graph construction complete."
            )

            logger.info(
                f"Nodes: {summary['total_nodes']}"
            )

            logger.info(
                f"Edges: {summary['total_edges']}"
            )

            logger.info(
                f"Node types: {summary['node_types']}"
            )

            logger.info(
                f"Relationship types: "
                f"{summary['relationship_types']}"
            )

            calls_count = (
                summary[
                    "relationship_types"
                ].get(
                    "CALLS",
                    0,
                )
            )

            logger.info(
                f"CALLS relationships: "
                f"{calls_count}"
            )

            return True

        except Exception as exc:

            logger.exception(
                "Knowledge graph construction failed: "
                f"{exc}"
            )

            return False

    # ============================================================
    # SAVE GRAPH
    # ============================================================

    def _save_knowledge_graph(
        self,
    ) -> bool:

        if not self.knowledge_graph:
            logger.error(
                "No knowledge graph available to save."
            )
            return False

        try:

            self.index_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(
                self.graph_file,
                "wb",
            ) as file:

                pickle.dump(
                    self.knowledge_graph.graph,
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            logger.info(
                f"Knowledge graph saved to: "
                f"{self.graph_file}"
            )

            return True

        except Exception as exc:

            logger.exception(
                "Failed to save knowledge graph: "
                f"{exc}"
            )

            return False

    # ============================================================
    # LOAD GRAPH
    # ============================================================

    def _load_knowledge_graph(
        self,
    ) -> bool:

        if not self.graph_file.exists():
            logger.info(
                "No persisted knowledge graph found."
            )
            return False

        try:

            with open(
                self.graph_file,
                "rb",
            ) as file:

                graph_data = pickle.load(
                    file
                )

            graph = CodeKnowledgeGraph()

            graph.graph = graph_data

            self.knowledge_graph = graph

            logger.info(
                f"Loaded knowledge graph from: "
                f"{self.graph_file}"
            )

            summary = (
                graph.get_summary()
            )

            logger.info(
                f"Loaded graph: "
                f"{summary['total_nodes']} nodes, "
                f"{summary['total_edges']} edges"
            )

            return True

        except Exception as exc:

            logger.exception(
                "Failed to load knowledge graph: "
                f"{exc}"
            )

            self.knowledge_graph = None

            return False

    # ============================================================
    # GRAPH VALIDATION
    # ============================================================

    def _knowledge_graph_is_valid(
        self,
    ) -> bool:

        if not self.graph_file.exists():
            return False

        try:

            with open(
                self.graph_file,
                "rb",
            ) as file:

                graph = pickle.load(
                    file
                )

            if graph is None:
                return False

            if graph.number_of_nodes() == 0:
                return False

            return True

        except Exception:
            return False

    # ============================================================
    # ENSURE GRAPH
    # ============================================================

    def ensure_knowledge_graph(
        self,
    ) -> Optional[CodeKnowledgeGraph]:

        if self.knowledge_graph:
            return self.knowledge_graph

        if self._load_knowledge_graph():
            return self.knowledge_graph

        if self.repository_path:
            if self._build_knowledge_graph_for_repository():
                return self.knowledge_graph

        return None

    # ============================================================
    # QUERY
    # ============================================================

    def query(
        self,
        query_text: str,
        apikey: Optional[str] = None,
        top_n_final: Optional[int] = None,
        vector_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:

        if not self.retriever._load_indexes():

            logger.error(
                f"Failed to load indexes for "
                f"repo_id '{self.repo_id}'."
            )

            return []

        logger.info(
            f"Running hybrid retrieval for: "
            f"{query_text}"
        )

        return self.retriever.retrieve(
            query_text,
            top_n_final=top_n_final,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            apikey=apikey,
        )

    # ============================================================
    # GET KNOWLEDGE GRAPH
    # ============================================================

    def get_knowledge_graph(
        self,
    ) -> Optional[CodeKnowledgeGraph]:

        return self.ensure_knowledge_graph()

    # ============================================================
    # CALLERS
    # ============================================================

    def get_callers(
        self,
        method_name: str,
    ) -> List[str]:

        graph = self.ensure_knowledge_graph()

        if not graph:
            return []

        return graph.get_callers_of(
            method_name
        )

    # ============================================================
    # CALLEES
    # ============================================================

    def get_callees(
        self,
        method_name: str,
    ) -> List[str]:

        graph = self.ensure_knowledge_graph()

        if not graph:
            return []

        return graph.get_calls_from(
            method_name
        )

    # ============================================================
    # IMPACT ANALYZER
    # ============================================================

    def get_impact_analyzer(
        self,
    ):

        graph = self.ensure_knowledge_graph()

        if not graph:
            return None

        from src.impact_analysis import (
            ImpactAnalyzer
        )

        return ImpactAnalyzer(
            graph
        )

    # ============================================================
    # STATUS
    # ============================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:

        graph = self.ensure_knowledge_graph()

        graph_summary = None

        if graph:
            graph_summary = (
                graph.get_summary()
            )

        return {
            "repo_id": self.repo_id,
            "repository_path": (
                str(self.repository_path)
                if self.repository_path
                else str(self.cloned_repo_path)
            ),
            "index_directory": str(
                self.index_dir
            ),
            "vector_index_exists": (
                (
                    self.index_dir /
                    config.FAISS_INDEX_FILENAME
                ).exists()
            ),
            "bm25_index_exists": (
                (
                    self.index_dir /
                    config.BM25_INDEX_FILENAME
                ).exists()
            ),
            "knowledge_graph_exists": (
                self.graph_file.exists()
            ),
            "knowledge_graph_valid": (
                self._knowledge_graph_is_valid()
            ),
            "graph": graph_summary,
        }