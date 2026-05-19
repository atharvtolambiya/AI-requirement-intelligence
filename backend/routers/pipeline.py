"""
Pipeline Router — Unified Semantic Pipeline Endpoints.

Endpoints:
    POST /api/v1/pipeline/run        → Standard pipeline (no vision)
    POST /api/v1/pipeline/run-vision → Vision-aware pipeline (recommended)
"""

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.logger import logger
from backend.models.schemas import (
    OptimizationPipelineRequest,
    OptimizationPipelineResponse,
    PromptStyle,
)
from backend.services.semantic_pipeline import (
    SemanticPipeline,
    get_semantic_pipeline,
)

# ── Router ────────────────────────────────────────────────────────
router = APIRouter(
    prefix="/pipeline",
    tags=["Semantic Pipeline"],
)

# ── Free tier protection ──────────────────────────────────────────
# Each style = 1 LLM call ≈ 3000 tokens
# Groq free tier = 6000 TPM → max 2 styles to avoid timeout
FREE_TIER_MAX_STYLES = 2


# ══════════════════════════════════════════════════════════════════
# REQUEST MODELS
# ══════════════════════════════════════════════════════════════════

class VisionAwarePipelineRequest(BaseModel):
    """
    Pipeline request that includes a pre-extracted VisionProfile.
    Use after the user has completed /vision/probe → /vision/profile.
    """
    raw_requirement: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="The user's original requirement",
    )
    context: str | None = Field(
        default=None,
        description="Additional context provided by the user",
    )
    user_answers: dict[str, str] | None = Field(
        default=None,
        description="Answers to clarifying questions from gap detection",
    )
    styles: list[PromptStyle] = Field(
        default=[
            PromptStyle.CHAIN_OF_THOUGHT,
            PromptStyle.STRUCTURED,
        ],
        description=(
            "Prompt styles to generate. "
            "Max 2 on Groq free tier to avoid timeout."
        ),
    )
    target_llm: str = Field(
        default="gpt-4o",
        description="Target LLM the generated prompts will be used with",
    )
    vision_profile: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Pre-extracted VisionProfile from /vision/profile endpoint. "
            "Strongly recommended for best results."
        ),
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for tracking. Auto-generated if not provided.",
    )


# ══════════════════════════════════════════════════════════════════
# DEPENDENCY
# ══════════════════════════════════════════════════════════════════

def pipeline_dep() -> SemanticPipeline:
    """FastAPI dependency — returns the pipeline singleton."""
    return get_semantic_pipeline()


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _cap_styles(
    styles: list[PromptStyle],
    session_id: str,
) -> list[PromptStyle]:
    """
    Caps styles to FREE_TIER_MAX_STYLES in Fast Mode.
    If Deep Mode is enabled, bypasses the cap and auto-expands to all 4 styles.
    """
    from backend.config import settings

    if settings.ENABLE_DEEP_REASONING:
        logger.warning(
            "Deep Mode enabled: bypassing free tier limits. "
            "This may increase latency and token usage."
        )
        return [
            PromptStyle.ZERO_SHOT,
            PromptStyle.CHAIN_OF_THOUGHT,
            PromptStyle.ROLE_BASED,
            PromptStyle.STRUCTURED,
        ]

    # Fast Mode: enforce limits
    if len(styles) <= FREE_TIER_MAX_STYLES:
        return styles

    capped = styles[:FREE_TIER_MAX_STYLES]
    logger.warning(
        "Styles capped for free tier (Fast Mode) | session={s} | "
        "requested={r} | using={u} | kept={k}",
        s=session_id,
        r=len(styles),
        u=FREE_TIER_MAX_STYLES,
        k=[s.value for s in capped],
    )
    return capped


def _ensure_session(session_id: str | None) -> str:
    """Returns existing session_id or generates a new UUID."""
    return session_id or str(uuid.uuid4())


# ══════════════════════════════════════════════════════════════════
# POST /pipeline/run  — Standard pipeline (no vision profile)
# ══════════════════════════════════════════════════════════════════

@router.post(
    "/run",
    response_model=OptimizationPipelineResponse,
    status_code=status.HTTP_200_OK,
    summary="Run Standard Pipeline",
    description=(
        "Runs the full 9-stage semantic pipeline WITHOUT a vision profile.\n\n"
        "Semantic analysis is performed from the raw requirement alone. "
        "For best results, use /run-vision after completing the vision "
        "clarification flow."
    ),
)
async def run_pipeline(
    request: OptimizationPipelineRequest,
    pipeline: Annotated[SemanticPipeline, Depends(pipeline_dep)],
) -> OptimizationPipelineResponse:
    """Standard pipeline without pre-extracted vision profile."""

    session_id = _ensure_session(request.session_id)

    # Patch session and cap styles
    request = request.model_copy(update={
        "session_id": session_id,
        "styles":     _cap_styles(request.styles, session_id),
    })

    logger.info(
        "Pipeline run | session={s} | styles={st} | preview={p}",
        s=session_id,
        st=[s.value for s in request.styles],
        p=request.raw_requirement[:60],
    )

    try:
        result = await pipeline.run(
            request=request,
            vision_profile=None,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Pipeline returned no result.",
            )

        return result

    except HTTPException:
        raise

    except RuntimeError as e:
        error_msg = str(e)
        logger.error(
            "Pipeline runtime error | session={s} | error={e}",
            s=session_id,
            e=error_msg,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error_msg,
        )

    except Exception as e:
        logger.exception(
            "Pipeline unexpected error | session={s}",
            s=session_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed: {str(e)}",
        )


# ══════════════════════════════════════════════════════════════════
# POST /pipeline/run-vision  — Vision-aware pipeline (recommended)
# ══════════════════════════════════════════════════════════════════

@router.post(
    "/run-vision",
    response_model=OptimizationPipelineResponse,
    status_code=status.HTTP_200_OK,
    summary="Run Vision-Aware Pipeline (Recommended)",
    description=(
        "Runs the full 9-stage semantic pipeline WITH a pre-extracted "
        "VisionProfile.\n\n"
        "**Recommended flow:**\n"
        "1. `POST /vision/probe` → get clarification questions\n"
        "2. User answers questions\n"
        "3. `POST /vision/profile` → extract VisionProfile\n"
        "4. `POST /pipeline/run-vision` → run this endpoint\n\n"
        "**Pipeline stages:**\n"
        "1. Semantic Intent Extraction\n"
        "2. Innovation Analysis\n"
        "3. Abstraction Classification\n"
        "4. Preservation Rules (rule-based)\n"
        "5. Reasoning Alignment\n"
        "6. Requirement Expansion\n"
        "7. Drift Detection\n"
        "8. Prompt Optimization (sequential per style)\n"
        "9. Prompt Scoring"
    ),
)
async def run_vision_pipeline(
    request: VisionAwarePipelineRequest,
    pipeline: Annotated[SemanticPipeline, Depends(pipeline_dep)],
) -> OptimizationPipelineResponse:
    """Vision-aware pipeline — uses VisionProfile as semantic constraint."""

    session_id = _ensure_session(request.session_id)
    capped_styles = _cap_styles(request.styles, session_id)

    logger.info(
        "Vision pipeline run | session={s} | has_vision={v} | "
        "styles={st} | preview={p}",
        s=session_id,
        v=request.vision_profile is not None,
        st=[s.value for s in capped_styles],
        p=request.raw_requirement[:60],
    )

    # Build standard pipeline request from vision-aware request
    pipeline_request = OptimizationPipelineRequest(
        raw_requirement=request.raw_requirement,
        context=request.context,
        user_answers=request.user_answers,
        styles=capped_styles,
        target_llm=request.target_llm,
        session_id=session_id,
    )

    try:
        result = await pipeline.run(
            request=pipeline_request,
            vision_profile=request.vision_profile,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Pipeline returned no result.",
            )

        logger.info(
            "Vision pipeline success | session={s} | "
            "score={sc:.1f} | grade={g} | time={t}ms",
            s=session_id,
            sc=result.best_prompt.score.overall_score
               if result.best_prompt else 0,
            g=result.best_prompt.score.grade
              if result.best_prompt else "?",
            t=result.total_processing_time_ms,
        )

        return result

    except HTTPException:
        raise

    except RuntimeError as e:
        error_msg = str(e)
        logger.error(
            "Vision pipeline runtime error | session={s} | error={e}",
            s=session_id,
            e=error_msg,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error_msg,
        )

    except Exception as e:
        logger.exception(
            "Vision pipeline unexpected error | session={s}",
            s=session_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vision pipeline failed: {str(e)}",
        )