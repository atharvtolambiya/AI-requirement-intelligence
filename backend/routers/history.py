"""
History Router - Prompt history and refinement endpoints.

Endpoints:
    GET    /api/v1/history/sessions              → List all sessions (paginated)
    GET    /api/v1/history/sessions/{session_id} → Get session detail
    DELETE /api/v1/history/sessions/{session_id} → Delete a session
    GET    /api/v1/history/analytics             → Usage analytics
    POST   /api/v1/history/refine                → Refine a prompt with RAG
    GET    /api/v1/history/rag/stats             → RAG collection stats
"""

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.logger import logger
from backend.models.database import get_db_session
from backend.models.schemas import PromptStyle, ScoredPrompt, PromptQualityScore, ScoreDimension
from backend.services.history_service import HistoryService, get_history_service
from backend.services.rag_service import RAGService, get_rag_service
from backend.services.refinement_service import RefinementService, get_refinement_service

# ─── Router Setup ─────────────────────────────────────────────────
router = APIRouter(
    prefix="/history",
    tags=["History & Refinement"],
)


# ─── Request/Response schemas for history router ──────────────────
class RefineRequest(BaseModel):
    """Request to refine a prompt through multi-step RAG refinement."""

    session_id: str | None = None
    prompt_text: str = Field(..., min_length=10, description="Prompt text to refine")
    prompt_style: str = Field(default="zero_shot", description="Style of the prompt")
    original_requirement: str = Field(
        ..., min_length=5, description="Original user requirement"
    )
    current_score: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Current quality score (will be re-scored if not provided)",
    )
    target_score: float = Field(
        default=80.0,
        ge=50.0,
        le=100.0,
        description="Target quality score to reach",
    )
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum refinement iterations",
    )


class RefineResponse(BaseModel):
    """Response from refinement endpoint."""

    success: bool
    session_id: str | None
    original_prompt: str
    refined_prompt: str
    original_score: float
    final_score: float
    total_improvement: float
    grade_before: str
    grade_after: str
    iterations_run: int
    target_reached: bool
    stop_reason: str
    iterations_detail: list[dict[str, Any]]
    total_tokens_used: int
    processing_time_ms: float


# ─── Dependency Injection ──────────────────────────────────────────
def history_dep() -> HistoryService:
    return get_history_service()


def rag_dep() -> RAGService:
    return get_rag_service()


def refinement_dep() -> RefinementService:
    return get_refinement_service()


# ══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@router.get(
    "/sessions",
    status_code=status.HTTP_200_OK,
    summary="List Prompt Sessions",
    description="Returns paginated list of all optimization sessions.",
)
async def list_sessions(
    history: Annotated[HistoryService, Depends(history_dep)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Records per page"),
    min_score: float | None = Query(
        default=None, ge=0.0, le=100.0, description="Filter by minimum score"
    ),
) -> dict[str, Any]:
    """
    Lists prompt optimization sessions with pagination.

    Sorted by creation date (newest first).
    """
    try:
        result = await history.get_session_list(
            db=db,
            page=page,
            page_size=page_size,
            min_score=min_score,
        )
        return result
    except Exception as e:
        logger.exception("Error listing sessions")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sessions: {str(e)}",
        )


@router.get(
    "/sessions/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Get Session Detail",
    description="Returns full detail for a session including all prompts and refinements.",
)
async def get_session(
    session_id: str,
    history: Annotated[HistoryService, Depends(history_dep)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> dict[str, Any]:
    """Returns complete session data with all prompts and refinement logs."""
    try:
        result = await history.get_session_detail(db=db, session_id=session_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session '{session_id}' not found",
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting session | session={s}", s=session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve session: {str(e)}",
        )


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Session",
    description="Deletes a session and all associated prompts and refinements.",
)
async def delete_session(
    session_id: str,
    history: Annotated[HistoryService, Depends(history_dep)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> dict[str, str]:
    """Permanently deletes a session (cascade deletes prompts and refinements)."""
    try:
        deleted = await history.delete_session(db=db, session_id=session_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session '{session_id}' not found",
            )
        return {"message": f"Session '{session_id}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error deleting session | session={s}", s=session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete session: {str(e)}",
        )


@router.get(
    "/analytics",
    status_code=status.HTTP_200_OK,
    summary="Usage Analytics",
    description="Returns simple analytics about prompt history.",
)
async def get_analytics(
    history: Annotated[HistoryService, Depends(history_dep)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> dict[str, Any]:
    """
    Returns usage analytics including:
    - Total sessions and prompts
    - Average quality scores
    - Style distribution
    - Grade distribution
    - Production readiness rate
    """
    try:
        return await history.get_analytics(db=db)
    except Exception as e:
        logger.exception("Error getting analytics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute analytics: {str(e)}",
        )


@router.post(
    "/refine",
    response_model=RefineResponse,
    status_code=status.HTTP_200_OK,
    summary="Refine Prompt with RAG ⭐",
    description=(
        "Runs multi-step prompt refinement using RAG-retrieved knowledge. "
        "Iteratively improves the prompt until target score or max iterations reached."
    ),
)
async def refine_prompt(
    request: RefineRequest,
    refinement_svc: Annotated[RefinementService, Depends(refinement_dep)],
    history: Annotated[HistoryService, Depends(history_dep)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> RefineResponse:
    """
    Multi-step RAG-powered prompt refinement.

    Process:
    1. Score input prompt (if not pre-scored)
    2. Retrieve relevant knowledge from ChromaDB
    3. Apply targeted improvements via LLM
    4. Re-score refined prompt
    5. Repeat until target reached or max iterations done
    6. Log all iterations to history
    """
    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    logger.info(
        "Refine request | session={s} | target={target} | max_iter={max}",
        s=request.session_id,
        target=request.target_score,
        max=request.max_iterations,
    )

    try:
        # ── Parse style safely ────────────────────────────────────
        try:
            style = PromptStyle(request.prompt_style)
        except ValueError:
            style = PromptStyle.ZERO_SHOT

        # ── Build initial scored prompt for refinement input ──────
        # Create a placeholder score for the initial prompt
        from backend.services.prompt_scorer import get_prompt_scorer
        scorer = get_prompt_scorer()

        initial_scored = await scorer.score_single(
            prompt=ScoredPrompt(
                style=style,
                style_label=style.value.replace("_", " ").title(),
                prompt_text=request.prompt_text,
                score=PromptQualityScore(
                    overall_score=request.current_score or 50.0,
                    grade="C",
                    dimensions=[],
                    strengths=[],
                    weaknesses=["Needs evaluation"],
                    hallucination_risk="medium",
                    hallucination_reasons=[],
                    overall_feedback="Initial score pending.",
                    is_production_ready=False,
                ),
                word_count=len(request.prompt_text.split()),
                estimated_tokens=len(request.prompt_text.split()) * 2,
            ),
            original_requirement=request.original_requirement,
        )

        # ── Run refinement loop ───────────────────────────────────
        result = await refinement_svc.refine(
            prompt=initial_scored,
            original_requirement=request.original_requirement,
            target_score=request.target_score,
            max_iterations=request.max_iterations,
            session_id=request.session_id,
        )

        # ── Save each iteration to history ────────────────────────
        for iter_data in result.iterations:
            await history.save_refinement_log(
                db=db,
                session_id=request.session_id,
                iteration=iter_data["iteration"],
                input_prompt=iter_data["prompt_before"],
                refined_prompt=iter_data["prompt_after"],
                score_before=iter_data["score_before"],
                score_after=iter_data["score_after"],
                rag_context=iter_data.get("rag_context", ""),
                feedback_applied=f"Score improved by {iter_data['improvement']:+.1f}",
                rag_docs_count=iter_data.get("rag_docs_retrieved", 0),
            )

        # ── Strip heavy iteration data for response ───────────────
        lite_iterations = [
            {
                "iteration": i["iteration"],
                "score_before": i["score_before"],
                "score_after": i["score_after"],
                "improvement": i["improvement"],
                "grade_before": i["grade_before"],
                "grade_after": i["grade_after"],
                "rag_docs_retrieved": i["rag_docs_retrieved"],
                "time_ms": i["time_ms"],
            }
            for i in result.iterations
        ]

        return RefineResponse(
            success=True,
            session_id=request.session_id,
            original_prompt=result.original_prompt.prompt_text,
            refined_prompt=result.final_prompt.prompt_text,
            original_score=result.original_prompt.score.overall_score,
            final_score=result.final_prompt.score.overall_score,
            total_improvement=result.total_improvement,
            grade_before=result.original_prompt.score.grade,
            grade_after=result.final_prompt.score.grade,
            iterations_run=len(result.iterations),
            target_reached=result.target_reached,
            stop_reason=result.stop_reason,
            iterations_detail=lite_iterations,
            total_tokens_used=result.total_tokens,
            processing_time_ms=result.total_time_ms,
        )

    except RuntimeError as e:
        logger.error("Refine LLM error | session={s} | error={e}",
                     s=request.session_id, e=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service error: {str(e)}",
        )
    except Exception as e:
        logger.exception("Unexpected refine error | session={s}", s=request.session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Refinement failed: {str(e)}",
        )


@router.get(
    "/rag/stats",
    status_code=status.HTTP_200_OK,
    summary="RAG Knowledge Base Stats",
    description="Returns statistics about the ChromaDB knowledge base collection.",
)
async def rag_stats(
    rag: Annotated[RAGService, Depends(rag_dep)],
) -> dict[str, Any]:
    """Returns ChromaDB collection stats including document count."""
    try:
        await rag.initialize()
        stats = rag.get_collection_stats()
        return stats
    except Exception as e:
        logger.error("RAG stats error | error={e}", e=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get RAG stats: {str(e)}",
        )