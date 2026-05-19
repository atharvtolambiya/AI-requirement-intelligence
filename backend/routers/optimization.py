"""
Optimization Router - FastAPI endpoints for the prompt optimization pipeline.

Endpoints:
    POST /api/v1/optimize/expand        → Expand a vague requirement
    POST /api/v1/optimize/prompts       → Generate optimized prompts
    POST /api/v1/optimize/score         → Score one or more prompts
    POST /api/v1/optimize/pipeline      → Run full pipeline (recommended)
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from backend.logger import logger
from backend.models.schemas import (
    OptimizationPipelineRequest,
    OptimizationPipelineResponse,
    PromptOptimizationRequest,
    PromptOptimizationResponse,
    PromptScoringRequest,
    PromptScoringResponse,
    PromptStyle,
    RequirementExpansionRequest,
    RequirementExpansionResponse,
)
from backend.services.requirement_expander import (
    RequirementExpander,
    get_requirement_expander,
)
from backend.services.prompt_optimizer import (
    PromptOptimizer,
    get_prompt_optimizer,
)
from backend.services.prompt_scorer import (
    PromptScorer,
    get_prompt_scorer,
)

# ─── Router Setup ─────────────────────────────────────────────────
router = APIRouter(
    prefix="/optimize",
    tags=["Optimization"],
)


# ─── Dependency Injection ──────────────────────────────────────────
def expander_dep() -> RequirementExpander:
    return get_requirement_expander()


def optimizer_dep() -> PromptOptimizer:
    return get_prompt_optimizer()


def scorer_dep() -> PromptScorer:
    return get_prompt_scorer()


# ══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@router.post(
    "/expand",
    response_model=RequirementExpansionResponse,
    status_code=status.HTTP_200_OK,
    summary="Expand Requirement",
    description=(
        "Expands a vague requirement into a fully structured specification "
        "with objectives, functional/non-functional requirements, and success criteria."
    ),
)
async def expand_requirement(
    request: RequirementExpansionRequest,
    expander: Annotated[RequirementExpander, Depends(expander_dep)],
) -> RequirementExpansionResponse:
    """
    Intelligently expands a vague requirement.

    Example: "build a habit tracker" →
    Full spec with objectives, functional requirements,
    scope definition, and success criteria.
    """
    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    logger.info(
        "Expand request | session={s} | preview={p}",
        s=request.session_id,
        p=request.raw_requirement[:60],
    )

    try:
        return await expander.expand(request)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service error: {str(e)}",
        )
    except Exception as e:
        logger.exception("Expand endpoint error | session={s}", s=request.session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Expansion failed: {str(e)}",
        )


@router.post(
    "/prompts",
    response_model=PromptOptimizationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Optimized Prompts",
    description=(
        "Generates optimized prompts in multiple styles "
        "(Zero-Shot, Chain-of-Thought, Role-Based, Structured, Few-Shot)."
    ),
)
async def generate_prompts(
    request: PromptOptimizationRequest,
    optimizer: Annotated[PromptOptimizer, Depends(optimizer_dep)],
) -> PromptOptimizationResponse:
    """
    Generates prompts in all requested styles concurrently.

    Styles available:
    - zero_shot: Direct instruction
    - chain_of_thought: Step-by-step reasoning
    - few_shot: With examples
    - role_based: Expert persona
    - structured: Organized sections
    """
    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    logger.info(
        "Optimize request | session={s} | styles={styles}",
        s=request.session_id,
        styles=[s.value for s in request.styles],
    )

    try:
        return await optimizer.optimize(request)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service error: {str(e)}",
        )
    except Exception as e:
        logger.exception("Optimize endpoint error | session={s}", s=request.session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}",
        )


@router.post(
    "/score",
    response_model=PromptScoringResponse,
    status_code=status.HTTP_200_OK,
    summary="Score Prompts",
    description=(
        "Evaluates one or more prompts across 5 quality dimensions: "
        "Clarity, Specificity, Completeness, Actionability, and Hallucination Reduction."
    ),
)
async def score_prompts(
    request: PromptScoringRequest,
    scorer: Annotated[PromptScorer, Depends(scorer_dep)],
) -> PromptScoringResponse:
    """
    Scores prompts using LLM-as-judge with structured rubrics.

    Returns:
    - Per-dimension scores (0-10 each)
    - Overall weighted score (0-100)
    - Letter grade (A+ to F)
    - Hallucination risk assessment
    - Actionable improvement suggestions
    """
    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    logger.info(
        "Score request | session={s} | count={c}",
        s=request.session_id,
        c=len(request.prompts),
    )

    try:
        return await scorer.score_prompts(request)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service error: {str(e)}",
        )
    except Exception as e:
        logger.exception("Score endpoint error | session={s}", s=request.session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed: {str(e)}",
        )


@router.post(
    "/pipeline",
    response_model=OptimizationPipelineResponse,
    status_code=status.HTTP_200_OK,
    summary="Full Optimization Pipeline ⭐",
    description=(
        "Runs the COMPLETE optimization pipeline in one call: "
        "1) Expand requirement → 2) Generate prompts → 3) Score all → 4) Return best. "
        "This is the PRIMARY endpoint for end-to-end prompt optimization."
    ),
)
async def run_pipeline(
    request: OptimizationPipelineRequest,
    expander: Annotated[RequirementExpander, Depends(expander_dep)],
    optimizer: Annotated[PromptOptimizer, Depends(optimizer_dep)],
    scorer: Annotated[PromptScorer, Depends(scorer_dep)],
) -> OptimizationPipelineResponse:
    """
    The primary end-to-end optimization pipeline.

    Pipeline stages:
    1. EXPAND: Vague requirement → Structured specification
    2. OPTIMIZE: Specification → Multiple styled prompts
    3. SCORE: All prompts evaluated and ranked
    4. RECOMMEND: Best prompt identified with reasoning

    Single call replaces 3 separate API calls.
    """
    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    logger.info(
        "Pipeline request | session={s} | styles={styles} | preview={p}",
        s=request.session_id,
        styles=[s.value for s in request.styles],
        p=request.raw_requirement[:60],
    )

    total_tokens = 0

    try:
        # Expand Requirement
        logger.info("Pipeline Stage 1: Expanding requirement")
        expand_request = RequirementExpansionRequest(
            raw_requirement=request.raw_requirement,
            user_answers=request.user_answers,
            session_id=request.session_id,
        )
        expansion_response = await expander.expand(expand_request)
        total_tokens += expansion_response.tokens_used

        expanded_dict = expansion_response.expanded.model_dump()

        # Generate Optimized Prompts
        logger.info("Pipeline Stage 2: Generating optimized prompts")
        optimize_request = PromptOptimizationRequest(
            raw_requirement=request.raw_requirement,
            expanded_requirement=expanded_dict,
            styles=request.styles,
            domain=expanded_dict.get("estimated_complexity"),
            target_llm=request.target_llm,
            session_id=request.session_id,
        )
        optimization_response = await optimizer.optimize(optimize_request)
        total_tokens += optimization_response.tokens_used

        # Score All Prompts
        logger.info(
            "Pipeline Stage 3: Scoring {count} prompts",
            count=len(optimization_response.optimized_prompts),
        )
        prompts_for_scoring = [
            {
                "style": p.style.value,
                "style_label": p.style_label,
                "prompt_text": p.prompt_text,
            }
            for p in optimization_response.optimized_prompts
        ]

        score_request = PromptScoringRequest(
            prompts=prompts_for_scoring,
            original_requirement=request.raw_requirement,
            session_id=request.session_id,
        )
        scoring_response = await scorer.score_prompts(score_request)
        total_tokens += scoring_response.tokens_used

        #Compute total time
        total_time_ms = round(
            expansion_response.processing_time_ms
            + optimization_response.processing_time_ms
            + scoring_response.processing_time_ms,
            2,
        )

        logger.info(
            "Pipeline complete | session={s} | best_score={score} | "
            "best_grade={grade} | total_tokens={tokens} | time={time}ms",
            s=request.session_id,
            score=scoring_response.best_prompt.score.overall_score,
            grade=scoring_response.best_prompt.score.grade,
            tokens=total_tokens,
            time=total_time_ms,
        )

        return OptimizationPipelineResponse(
            success=True,
            session_id=request.session_id,
            raw_requirement=request.raw_requirement,
            expanded_requirement=expansion_response.expanded,
            scored_prompts=scoring_response.scored_prompts,
            best_prompt=scoring_response.best_prompt,
            recommended_style=optimization_response.recommended_style,
            recommendation_reason=optimization_response.recommendation_reason,
            total_processing_time_ms=total_time_ms,
            total_tokens_used=total_tokens,
        )

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error(
            "Pipeline LLM error | session={s} | error={e}",
            s=request.session_id,
            e=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service error: {str(e)}",
        )
    except Exception as e:
        logger.exception(
            "Pipeline unexpected error | session={s}", s=request.session_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed: {str(e)}",
        )