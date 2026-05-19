"""
Drift Router — Phase 4 API Endpoints.

Endpoints:
    POST /api/v1/drift/preserve   → Build preservation rules (no LLM)
    POST /api/v1/drift/align      → Generate reasoning alignment
    POST /api/v1/drift/detect     → Detect drift in expansion
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from backend.logger import logger
from backend.models.schemas import (
    DriftAnalysisRequest,
    DriftAnalysisResponse,
    PreservationRulesRequest,
    PreservationRulesResponse,
    ReasoningAlignmentRequest,
    ReasoningAlignmentResponse,
)
from backend.services.concept_preserver import (
    ConceptPreserver,
    get_concept_preserver,
)
from backend.services.drift_detector import (
    DriftDetector,
    get_drift_detector,
)
from backend.services.reasoning_aligner import (
    ReasoningAligner,
    get_reasoning_aligner,
)


# ─── Router ───────────────────────────────────────────────────────
router = APIRouter(
    prefix="/drift",
    tags=["Drift Prevention"],
)


# ─── Dependencies ─────────────────────────────────────────────────
def preserver_dep() -> ConceptPreserver:
    return get_concept_preserver()


def aligner_dep() -> ReasoningAligner:
    return get_reasoning_aligner()


def detector_dep() -> DriftDetector:
    return get_drift_detector()


# ══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@router.post(
    "/preserve",
    response_model=PreservationRulesResponse,
    status_code=status.HTTP_200_OK,
    summary="Build Preservation Rules",
    description=(
        "Builds enforceable preservation rules from VisionProfile "
        "and semantic analysis. NO LLM call — fast and deterministic. "
        "Returns pre-formatted text blocks ready to inject into "
        "downstream LLM prompts."
    ),
)
async def build_preservation_rules(
    request: PreservationRulesRequest,
    preserver: Annotated[ConceptPreserver, Depends(preserver_dep)],
) -> PreservationRulesResponse:
    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    logger.info(
        "Preservation rules request | session={s}",
        s=request.session_id,
    )

    try:
        return preserver.build_rules(request)
    except Exception as e:
        logger.exception(
            "Preservation rules error | session={s}",
            s=request.session_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preservation rule building failed: {str(e)}",
        )


@router.post(
    "/align",
    response_model=ReasoningAlignmentResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Reasoning Alignment",
    description=(
        "Generates reasoning alignment instructions for the expander. "
        "Tells the expander HOW to think about the problem at the "
        "correct abstraction level using the right reasoning mode. "
        "Output is a complete instruction block ready for prompt injection."
    ),
)
async def generate_alignment(
    request: ReasoningAlignmentRequest,
    aligner: Annotated[ReasoningAligner, Depends(aligner_dep)],
) -> ReasoningAlignmentResponse:
    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    logger.info(
        "Reasoning alignment request | session={s}",
        s=request.session_id,
    )

    try:
        return await aligner.align(request)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM error: {str(e)}",
        )
    except Exception as e:
        logger.exception(
            "Reasoning alignment error | session={s}",
            s=request.session_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reasoning alignment failed: {str(e)}",
        )


@router.post(
    "/detect",
    response_model=DriftAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect Semantic Drift ⭐",
    description=(
        "Audits an expansion for semantic drift against the user's vision. "
        "Detects lost concepts, diluted innovations, abstraction collapse, "
        "and anti-pattern violations. Returns specific warnings with "
        "evidence quotes and required corrections."
    ),
)
async def detect_drift(
    request: DriftAnalysisRequest,
    detector: Annotated[DriftDetector, Depends(detector_dep)],
) -> DriftAnalysisResponse:
    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    logger.info(
        "Drift detection request | session={s}",
        s=request.session_id,
    )

    try:
        return await detector.detect(request)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM error: {str(e)}",
        )
    except Exception as e:
        logger.exception(
            "Drift detection error | session={s}",
            s=request.session_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift detection failed: {str(e)}",
        )