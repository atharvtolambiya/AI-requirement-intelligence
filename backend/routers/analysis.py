"""
Analysis Router - FastAPI endpoints for requirement analysis.

Endpoints:
    POST /api/v1/analysis/intent          → Analyze user intent
    POST /api/v1/analysis/gaps            → Detect missing requirements
    POST /api/v1/analysis/full            → Run both analyses together

All endpoints use dependency injection for service access
and return consistent response schemas.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from backend.logger import logger
from backend.models.schemas import (
    ErrorResponse,
    FullAnalysisRequest,
    FullAnalysisResponse,
    IntentAnalysisRequest,
    IntentAnalysisResponse,
    MissingRequirementRequest,
    MissingRequirementResponse,
)
from backend.services.intent_analyzer import IntentAnalyzer, get_intent_analyzer
from backend.services.requirement_detector import (
    RequirementDetector,
    get_requirement_detector,
)

# ─── Router Setup ─────────────────────────────────────────────────
router = APIRouter(
    prefix="/analysis",
    tags=["Analysis"],
)


# ─── Dependency Injection Helpers ─────────────────────────────────
def intent_analyzer_dep() -> IntentAnalyzer:
    return get_intent_analyzer()


def requirement_detector_dep() -> RequirementDetector:
    return get_requirement_detector()


# ══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@router.post(
    "/intent",
    response_model=IntentAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze User Intent",
    description=(
        "Analyzes a vague user requirement to extract structured intent: "
        "primary goal, domain, constraints, complexity, and confidence level."
    ),
    responses={
        200: {"description": "Intent analysis successful"},
        422: {"description": "Validation error in request"},
        500: {"description": "LLM service error"},
    },
)
async def analyze_intent(
    request: IntentAnalysisRequest,
    analyzer: Annotated[IntentAnalyzer, Depends(intent_analyzer_dep)],
) -> IntentAnalysisResponse:
    """
    Analyzes the intent behind a user requirement.

    Takes a raw, potentially vague requirement and extracts:
    - What the user actually wants (primary goal)
    - Domain classification
    - Implied constraints
    - Complexity level
    - Confidence in the analysis

    Example input: "build me an app that tracks stuff"
    """
    # Auto-generate session_id if not provided
    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    logger.info(
        "Intent analysis request | session={session} | req_preview={preview}",
        session=request.session_id,
        preview=request.raw_requirement[:80],
    )

    try:
        result = await analyzer.analyze(request)
        return result

    except RuntimeError as e:
        # LLM API failures
        logger.error(
            "Intent analysis LLM error | session={session} | error={error}",
            session=request.session_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service error: {str(e)}",
        )

    except ValueError as e:
        # JSON parsing failures
        logger.error(
            "Intent analysis parse error | session={session} | error={error}",
            session=request.session_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse LLM response: {str(e)}",
        )

    except Exception as e:
        logger.exception(
            "Unexpected error in intent analysis | session={session}",
            session=request.session_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


@router.post(
    "/gaps",
    response_model=MissingRequirementResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect Missing Requirements",
    description=(
        "Analyzes a requirement to find missing information, "
        "generating prioritized clarifying questions with severity levels."
    ),
    responses={
        200: {"description": "Gap detection successful"},
        422: {"description": "Validation error in request"},
        500: {"description": "LLM service error"},
    },
)
async def detect_gaps(
    request: MissingRequirementRequest,
    detector: Annotated[RequirementDetector, Depends(requirement_detector_dep)],
) -> MissingRequirementResponse:
    """
    Detects gaps and missing information in a requirement.

    Returns a completeness score and prioritized list of
    clarifying questions ranked by severity (critical → optional).

    Example input: "I need a REST API for my project"
    """
    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    logger.info(
        "Gap detection request | session={session} | req_preview={preview}",
        session=request.session_id,
        preview=request.raw_requirement[:80],
    )

    try:
        result = await detector.detect_gaps(request)
        return result

    except RuntimeError as e:
        logger.error(
            "Gap detection LLM error | session={session} | error={error}",
            session=request.session_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service error: {str(e)}",
        )

    except ValueError as e:
        logger.error(
            "Gap detection parse error | session={session} | error={error}",
            session=request.session_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse LLM response: {str(e)}",
        )

    except Exception as e:
        logger.exception(
            "Unexpected error in gap detection | session={session}",
            session=request.session_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


@router.post(
    "/full",
    response_model=FullAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Full Requirement Analysis",
    description=(
        "Runs both intent analysis AND gap detection in sequence. "
        "The intent analysis result feeds into gap detection for better accuracy. "
        "Use this for the complete analysis pipeline."
    ),
)
async def full_analysis(
    request: FullAnalysisRequest,
    analyzer: Annotated[IntentAnalyzer, Depends(intent_analyzer_dep)],
    detector: Annotated[RequirementDetector, Depends(requirement_detector_dep)],
) -> FullAnalysisResponse:
    """
    Complete analysis pipeline: intent + gap detection.

    Step 1: Analyze intent (what does the user want?)
    Step 2: Use intent to guide gap detection (what's missing?)

    This gives better gap detection because the domain and
    primary goal are already established from step 1.

    Example input: "build a website for my restaurant"
    """
    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    logger.info(
        "Full analysis request | session={session} | req_preview={preview}",
        session=request.session_id,
        preview=request.raw_requirement[:80],
    )

    total_tokens = 0

    try:
        # ── Step 1: Intent Analysis ───────────────────────────────
        logger.debug("Full analysis: Step 1 - Intent Analysis")
        intent_request = IntentAnalysisRequest(
            raw_requirement=request.raw_requirement,
            context=request.context,
            session_id=request.session_id,
        )
        intent_response = await analyzer.analyze(intent_request)
        total_tokens += intent_response.tokens_used

        # ── Step 2: Gap Detection (enriched with intent context) ──
        logger.debug("Full analysis: Step 2 - Gap Detection")
        intent = intent_response.intent

        # Build intent summary to guide gap detection
        intent_summary = (
            f"Primary goal: {intent.primary_goal}. "
            f"Secondary goals: {', '.join(intent.secondary_goals[:3])}. "
            f"Constraints detected: {', '.join(intent.constraints[:3])}."
        )

        gap_request = MissingRequirementRequest(
            raw_requirement=request.raw_requirement,
            intent_summary=intent_summary,
            domain=intent.domain.value,
            session_id=request.session_id,
        )
        gap_response = await detector.detect_gaps(gap_request)
        total_tokens += gap_response.tokens_used

        # ── Combine results ───────────────────────────────────────
        total_time_ms = (
            intent_response.processing_time_ms
            + gap_response.processing_time_ms
        )

        logger.info(
            "Full analysis complete | session={session} | "
            "tokens={tokens} | time={time}ms | score={score}",
            session=request.session_id,
            tokens=total_tokens,
            time=round(total_time_ms, 2),
            score=gap_response.gap_analysis.completeness_score,
        )

        return FullAnalysisResponse(
            success=True,
            session_id=request.session_id,
            raw_requirement=request.raw_requirement,
            intent=intent_response.intent,
            gap_analysis=gap_response.gap_analysis,
            total_processing_time_ms=round(total_time_ms, 2),
            total_tokens_used=total_tokens,
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is

    except Exception as e:
        logger.exception(
            "Unexpected error in full analysis | session={session}",
            session=request.session_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Full analysis failed: {str(e)}",
        )