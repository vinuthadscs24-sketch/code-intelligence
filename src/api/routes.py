import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

# Import your core modules (update paths according to your project structure)
from src.core.orchestrator import AdaptiveResponseOrchestrator, UnifiedAdaptiveResponse
from src.core.llm import llm_engine
from src.core.analyzer import impact_analyzer
from src.core.git import git_intelligence

logger = logging.getLogger(__name__)

router = APIRouter()

# Global singleton container for the orchestrator
_orchestrator_instance: Optional[AdaptiveResponseOrchestrator] = None

class AdaptiveQueryRequest(BaseModel):
    query: str = Field(..., description="The user's code-RAG query")
    repo_id: str = Field(..., description="Target repository identifier or path")
    debug: bool = Field(default=False, description="Enable verbose debug trace")

def get_orchestrator() -> AdaptiveResponseOrchestrator:
    """Dependency provider that lazily initializes and reuses the Orchestrator."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = AdaptiveResponseOrchestrator(
            llm_engine=llm_engine,
            impact_analyzer=impact_analyzer,
            git_intelligence=git_intelligence
        )
    return _orchestrator_instance

# Use relative path if main.py mounts this router with prefix="/v1/code-rag"
@router.post(
    "/query/adaptive",
    response_model=UnifiedAdaptiveResponse,
    status_code=status.HTTP_200_OK
)
async def query_adaptive_endpoint(
    payload: AdaptiveQueryRequest,
    orchestrator: AdaptiveResponseOrchestrator = Depends(get_orchestrator)
):
    try:
        # Runs blocking LLM calls in a thread pool to keep event loop free
        if asyncio.iscoroutinefunction(orchestrator.process_query):
            return await orchestrator.process_query(
                query=payload.query,
                repo_id=payload.repo_id,
                debug=payload.debug
            )
        else:
            return await asyncio.to_thread(
                orchestrator.process_query,
                query=payload.query,
                repo_id=payload.repo_id,
                debug=payload.debug
            )
    except Exception as err:
        logger.error(f"Error executing adaptive query: {err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Adaptive query processing failed: {str(err)}"
        )