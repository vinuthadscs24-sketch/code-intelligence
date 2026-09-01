import asyncio
from typing import Dict, Any, Optional
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from src import config
from src.generation.generator import LLMGenerator
from src.pipeline import RAGPipeline


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Code-Aware RAG API",
    description="API for interacting with the Code-Aware RAG system.",
    version="0.1.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Global State
# ============================================================

pipeline_locks: Dict[str, asyncio.Lock] = {}
setup_status: Dict[str, Dict[str, Any]] = {}


# ============================================================
# Request / Response Models
# ============================================================

class RepositorySetupRequest(BaseModel):
    repo_id: str = Field(
        ...,
        description="Unique identifier for the repository.",
    )

    repo_url_or_path: str = Field(
        ...,
        description="Git repository URL or absolute local repository path.",
    )

    access_token: Optional[str] = Field(
        None,
        description="Access token for private repositories.",
    )

    force_reclone: bool = Field(
        False,
        description="Delete and reclone the repository if it exists.",
    )

    force_reindex: bool = Field(
        False,
        description="Rebuild all indexes.",
    )


class RepositorySetupResponse(BaseModel):
    repo_id: str
    message: str
    index_status: str
    repository_path: Optional[str] = None
    task_id: Optional[str] = None


class RepositoryStatusResponse(BaseModel):
    repo_id: str
    status: str
    message: str
    index_status: Optional[str] = None
    repository_path: Optional[str] = None


class QueryRequest(BaseModel):
    repo_id: str = Field(
        ...,
        description="Repository ID to query.",
    )

    sys_prompt: str = Field(
        config.GENERATOR_PROMPT,
        description="System prompt for LLM generation.",
    )

    query_text: str = Field(
        ...,
        description="Question about the repository.",
    )

    top_n_final: int = Field(
        config.RETRIEVAL_VECTOR_TOP_K,
        description="Number of final context chunks.",
    )

    indexes: list[str] = Field(
        config.RETRIEVAL_INDEXES,
        description="Indexes enabled for retrieval.",
    )

    vector_top_k: int = Field(
        config.RETRIEVAL_VECTOR_TOP_K,
        description="Top K vector results.",
    )

    bm25_top_k: int = Field(
        config.RETRIEVAL_BM25_TOP_K,
        description="Top K BM25 results.",
    )

    rewrite_query: Optional[str] = Field(
        None,
        description="Optional rewritten query.",
    )

    rewrite_prompt: Optional[str] = Field(
        None,
        description="Prompt used to rewrite the query.",
    )


# ============================================================
# Adaptive Response Models
# ============================================================

class AdaptiveQueryRequest(BaseModel):
    repo_id: str = Field(
        ...,
        description="Repository ID to query.",
    )

    query_text: str = Field(
        ...,
        description="Question about the repository.",
    )

    sys_prompt: str = Field(
        config.GENERATOR_PROMPT,
        description="System prompt for LLM generation.",
    )

    top_n_final: int = Field(
        config.RETRIEVAL_VECTOR_TOP_K,
        description="Number of final context chunks.",
    )

    indexes: list[str] = Field(
        config.RETRIEVAL_INDEXES,
        description="Indexes enabled for retrieval.",
    )

    vector_top_k: int = Field(
        config.RETRIEVAL_VECTOR_TOP_K,
        description="Top K vector results.",
    )

    bm25_top_k: int = Field(
        config.RETRIEVAL_BM25_TOP_K,
        description="Top K BM25 results.",
    )


# ============================================================
# Startup / Shutdown
# ============================================================

@app.on_event("startup")
async def startup_event():
    logger.info("Code-Aware RAG API starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Code-Aware RAG API shutting down...")


# ============================================================
# Repository Setup
# ============================================================

@app.post(
    "/v1/code-rag/repository/setup",
    response_model=RepositorySetupResponse,
)
async def setup_repository_endpoint(
    request: RepositorySetupRequest,
):

    repo_id = (
        request.repo_id
        .replace("/", "_")
        .replace(":", "_")
    )

    task_id = f"{repo_id}_{threading.get_ident()}"

    if repo_id not in pipeline_locks:
        pipeline_locks[repo_id] = asyncio.Lock()

    if (
        repo_id in setup_status
        and setup_status[repo_id].get("status") == "pending"
    ):
        return RepositorySetupResponse(
            repo_id=repo_id,
            message="Repository setup already in progress",
            index_status="In Progress",
            task_id=setup_status[repo_id].get("task_id"),
        )

    logger.info(
        f"Received setup request for repo_id={repo_id}, "
        f"source={request.repo_url_or_path}"
    )

    try:

        pipeline = RAGPipeline(
            repo_id=repo_id
        )

        logger.info(
            f"Created RAGPipeline for repo_id={repo_id}"
        )

    except Exception as e:

        logger.exception(
            f"Failed to initialize RAGPipeline for {repo_id}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Pipeline initialization error: {str(e)}",
        )

    setup_status[repo_id] = {
        "status": "pending",
        "message": "Repository setup started",
        "index_status": "In Progress",
        "task_id": task_id,
        "repository_path": None,
    }

    def background_setup():

        try:

            success = pipeline.setup_repository(
                repo_url_or_path=request.repo_url_or_path,
                access_token=request.access_token,
                force_reclone=request.force_reclone,
                force_reindex=request.force_reindex,
                apikey=None,
            )

            if success:

                index_status = "Indexed Successfully"

                faiss_exists = (
                    pipeline.index_dir
                    / config.FAISS_INDEX_FILENAME
                ).exists()

                bm25_exists = (
                    pipeline.index_dir
                    / config.BM25_INDEX_FILENAME
                ).exists()

                if (
                    not request.force_reindex
                    and faiss_exists
                    and bm25_exists
                ):
                    index_status = (
                        "Indexes Already Existed or Verified"
                    )

                setup_status[repo_id] = {
                    "status": "completed",
                    "message": "Repository setup process completed",
                    "index_status": index_status,
                    "task_id": task_id,
                    "repository_path": (
                        str(pipeline.repository_path)
                        if pipeline.repository_path
                        else None
                    ),
                }

                logger.info(
                    f"Repository setup completed: {repo_id}"
                )

            else:

                setup_status[repo_id] = {
                    "status": "failed",
                    "message": "Repository setup failed",
                    "index_status": "Failed",
                    "task_id": task_id,
                    "repository_path": (
                        str(pipeline.repository_path)
                        if pipeline.repository_path
                        else None
                    ),
                }

                logger.error(
                    f"Repository setup failed: {repo_id}"
                )

        except Exception as e:

            logger.exception(
                f"Error during background setup for {repo_id}"
            )

            setup_status[repo_id] = {
                "status": "failed",
                "message": f"Error: {str(e)}",
                "index_status": "Failed",
                "task_id": task_id,
                "repository_path": (
                    str(pipeline.repository_path)
                    if pipeline.repository_path
                    else None
                ),
            }

    thread = threading.Thread(
        target=background_setup,
        daemon=True,
    )

    thread.start()

    return RepositorySetupResponse(
        repo_id=repo_id,
        message="Repository setup started in background",
        index_status="In Progress",
        repository_path=None,
        task_id=task_id,
    )


# ============================================================
# Helper: Load Pipeline
# ============================================================

def load_repository_pipeline(
    repo_id: str,
    indexes: list[str],
) -> RAGPipeline:

    pipeline = RAGPipeline(
        repo_id=repo_id,
        indexes=indexes,
    )

    if (
        not pipeline.retriever.vector_index
        and not pipeline.retriever.bm25_index
    ):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Repository '{repo_id}' not found "
                "or not indexed. Please set it up first."
            ),
        )

    return pipeline


# ============================================================
# Helper: Retrieve Context
# ============================================================

async def retrieve_context(
    pipeline: RAGPipeline,
    request: QueryRequest | AdaptiveQueryRequest,
):

    retriever_query = request.query_text

    if isinstance(request, QueryRequest):

        if request.rewrite_query:

            retriever_query = request.rewrite_query

        elif request.rewrite_prompt:

            retriever_query = (
                await pipeline.retriever.rewrite_query(
                    sys_prompt=request.rewrite_prompt,
                    user_query=request.query_text,
                    apikey=None,
                )
            )

    context_chunks_meta = pipeline.query(
        query_text=retriever_query,
        top_n_final=request.top_n_final,
        vector_top_k=request.vector_top_k,
        bm25_top_k=request.bm25_top_k,
        apikey=None,
    )

    logger.info(
        f"Retrieved {len(context_chunks_meta)} context chunks."
    )

    return context_chunks_meta


# ============================================================
# Normal Query
# ============================================================

@app.post("/v1/code-rag/query")
async def query_repository(
    request: QueryRequest,
):

    repo_id = (
        request.repo_id
        .replace("/", "_")
        .replace(":", "_")
    )

    logger.info(
        f"Received query for repo_id='{repo_id}', "
        f"query='{request.query_text[:50]}...'"
    )

    try:

        pipeline = load_repository_pipeline(
            repo_id,
            request.indexes,
        )

    except HTTPException:
        raise

    except Exception:

        logger.exception(
            f"Failed to load RAGPipeline for {repo_id}"
        )

        raise HTTPException(
            status_code=404,
            detail=(
                f"Repository '{repo_id}' not found "
                "or not indexed."
            ),
        )

    try:

        context_chunks_meta = await retrieve_context(
            pipeline,
            request,
        )

    except Exception as e:

        logger.exception(
            f"Retrieval error for {repo_id}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving context: {str(e)}",
        )

    try:

        llm_generator = LLMGenerator()

    except Exception as e:

        logger.exception(
            "Failed to initialize LLMGenerator"
        )

        raise HTTPException(
            status_code=500,
            detail=f"LLM Generator initialization error: {str(e)}",
        )

    try:

        response = (
            await llm_generator.generate_response_non_streaming(
                apikey=None,
                sys_prompy=request.sys_prompt,
                user_query=request.query_text,
                context_chunks=context_chunks_meta,
            )
        )

        return response

    except Exception as e:

        logger.exception(
            f"LLM generation error for {repo_id}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Error generating LLM response: {str(e)}",
        )


# ============================================================
# Adaptive Query
# ============================================================

@app.post("/v1/code-rag/query/adaptive")
async def adaptive_query(
    request: AdaptiveQueryRequest,
):

    repo_id = (
        request.repo_id
        .replace("/", "_")
        .replace(":", "_")
    )

    logger.info(
        f"Received ADAPTIVE query for repo_id='{repo_id}', "
        f"query='{request.query_text[:80]}...'"
    )

    try:

        pipeline = load_repository_pipeline(
            repo_id,
            request.indexes,
        )

    except HTTPException:
        raise

    except Exception:

        logger.exception(
            f"Failed to load adaptive pipeline for {repo_id}"
        )

        raise HTTPException(
            status_code=404,
            detail=(
                f"Repository '{repo_id}' not found "
                "or not indexed."
            ),
        )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    try:

        context_chunks_meta = pipeline.query(
            query_text=request.query_text,
            top_n_final=request.top_n_final,
            vector_top_k=request.vector_top_k,
            bm25_top_k=request.bm25_top_k,
            apikey=None,
        )

        logger.info(
            f"Adaptive retrieval returned "
            f"{len(context_chunks_meta)} context chunks."
        )

    except Exception as e:

        logger.exception(
            f"Adaptive retrieval error for {repo_id}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving context: {str(e)}",
        )

    # --------------------------------------------------------
    # Generate AI explanation
    # --------------------------------------------------------

    try:

        llm_generator = LLMGenerator()

        response = (
            await llm_generator.generate_response_non_streaming(
                apikey=None,
                sys_prompy=request.sys_prompt,
                user_query=request.query_text,
                context_chunks=context_chunks_meta,
            )
        )

    except Exception as e:

        logger.exception(
            f"Adaptive LLM generation error for {repo_id}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Error generating adaptive response: {str(e)}",
        )

    # --------------------------------------------------------
    # Query type detection
    # --------------------------------------------------------

    query_lower = request.query_text.lower()

    if any(
        word in query_lower
        for word in [
            "caller",
            "callers",
            "who calls",
            "who called",
        ]
    ):

        response_type = "call_graph"

    elif any(
        word in query_lower
        for word in [
            "callee",
            "callees",
            "calls",
            "dependencies",
            "dependency",
        ]
    ):

        response_type = "call_graph"

    elif any(
        word in query_lower
        for word in [
            "commit",
            "commits",
            "git history",
            "history",
            "changed",
        ]
    ):

        response_type = "git_timeline"

    elif any(
        word in query_lower
        for word in [
            "impact",
            "affected",
            "affect",
            "what breaks",
        ]
    ):

        response_type = "impact_analysis"

    else:

        response_type = "code_explanation"

    # --------------------------------------------------------
    # Build structured adaptive response
    # --------------------------------------------------------

    return {
        "response_type": response_type,
        "query": request.query_text,
        "answer": response,
        "repository": repo_id,
        "retrieval": {
            "chunks_retrieved": len(context_chunks_meta),
            "vector_top_k": request.vector_top_k,
            "bm25_top_k": request.bm25_top_k,
            "top_n_final": request.top_n_final,
        },
        "context": context_chunks_meta,
    }


# ============================================================
# Streaming Query
# ============================================================

@app.post(
    "/v1/code-rag/query/stream",
)
async def query_repository_stream(
    request: QueryRequest,
):

    repo_id = (
        request.repo_id
        .replace("/", "_")
        .replace(":", "_")
    )

    logger.info(
        f"Received streaming query for repo_id='{repo_id}', "
        f"query='{request.query_text[:50]}...'"
    )

    try:

        pipeline = load_repository_pipeline(
            repo_id,
            request.indexes,
        )

    except HTTPException:
        raise

    except Exception:

        logger.exception(
            f"Failed to load pipeline for {repo_id}"
        )

        raise HTTPException(
            status_code=404,
            detail=(
                f"Repository '{repo_id}' not found "
                "or not indexed."
            ),
        )

    try:

        context_chunks_meta = await retrieve_context(
            pipeline,
            request,
        )

    except Exception as e:

        logger.exception(
            f"Retrieval error for {repo_id}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving context: {str(e)}",
        )

    try:

        llm_generator = LLMGenerator()

    except Exception as e:

        logger.exception(
            "Failed to initialize LLMGenerator"
        )

        raise HTTPException(
            status_code=500,
            detail=f"LLM Generator initialization error: {str(e)}",
        )

    try:

        response_stream_iterator = (
            llm_generator.generate_response_stream(
                apikey=None,
                sys_prompy=request.sys_prompt,
                user_query=request.query_text,
                context_chunks=context_chunks_meta,
            )
        )

        return StreamingResponse(
            response_stream_iterator,
            media_type="text/plain",
        )

    except Exception as e:

        logger.exception(
            f"LLM streaming error for {repo_id}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Error generating LLM response: {str(e)}",
        )


# ============================================================
# Repository Status
# ============================================================

@app.get(
    "/v1/code-rag/repository/status/{repo_id}",
    response_model=RepositoryStatusResponse,
)
async def check_repository_setup_status(
    repo_id: str,
):

    sanitized_repo_id = (
        repo_id
        .replace("/", "_")
        .replace(":", "_")
    )

    if sanitized_repo_id not in setup_status:

        raise HTTPException(
            status_code=404,
            detail=(
                f"No setup process found for repository ID: "
                f"{repo_id}"
            ),
        )

    status_info = setup_status[sanitized_repo_id]

    return RepositoryStatusResponse(
        repo_id=repo_id,
        status=status_info.get("status", "unknown"),
        message=status_info.get("message", ""),
        index_status=status_info.get("index_status"),
        repository_path=status_info.get("repository_path"),
    )


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "message": "Code-Aware RAG API is running.",
    }


# ============================================================
# Direct Execution
# ============================================================

if __name__ == "__main__":

    import uvicorn

    logger.info(
        "Starting Code-Aware RAG API..."
    )

    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_RELOAD,
    )