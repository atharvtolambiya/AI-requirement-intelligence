"""
Vision Router - Vision Clarification API Endpoints.

Endpoints:
    POST /api/v1/vision/probe    → Generate vision probe questions
    POST /api/v1/vision/profile  → Extract VisionProfile from answers
    POST /api/v1/vision/analyze  → Full vision flow in one call

These endpoints form STAGE 0 of the new semantic pipeline.
They must be called BEFORE any analysis or optimization.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from backend.logger import logger
from backend.models.schemas import (
    VisionProbeRequest,
    VisionProbeResponse,
    VisionProfileRequest,
    VisionProfileResponse,
)
from backend.services.vision_clarifier import (
    VisionClarifier,
    get_vision_clarifier,
)

#  Router
router = APIRouter(
    prefix="/vision",
    tags=["Vision Clarification"],
)


# Dependency
def vision_dep() -> VisionClarifier:
    return get_vision_clarifier()



@router.post(
    "/probe",
    response_model=VisionProbeResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Vision Probe Questions",
    description=(
        "Analyzes the user's raw input and generates 4-6 adaptive "
        "vision-clarification questions designed to: \n"
        "1. Resolve detected ambiguities\n"
        "2. Prevent semantic collapse into generic patterns\n"
        "3. Extract semantic anchors before any expansion\n\n"
        "**Call this FIRST before any analysis or optimization.**"
    ),
)
async def generate_vision_probe(
    request: VisionProbeRequest,
    clarifier: Annotated[VisionClarifier, Depends(vision_dep)],
) -> VisionProbeResponse:
    """
    Generates adaptive vision-probing questions for the user's input.

    The questions are specifically designed based on:
    - Detected ambiguities in the input
    - Identified collapse patterns to prevent
    - Missing conceptual anchors

    Example:
        Input: "AI-powered decision intelligence system"
        Generates questions about:
        - What decisions it helps make (not what ML models it uses)
        - What makes it different from existing tools
        - What domain knowledge it needs
    """
    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    logger.info(
        "Vision probe request | session={s} | input_preview={p}",
        s=request.session_id,
        p=request.raw_input[:60],
    )

    try:
        return await clarifier.generate_probe(request)

    except RuntimeError as e:
        logger.error(
            "Vision probe LLM error | session={s} | error={e}",
            s=request.session_id,
            e=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service error: {str(e)}",
        )
    except Exception as e:
        logger.exception(
            "Vision probe unexpected error | session={s}",
            s=request.session_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vision probe failed: {str(e)}",
        )


@router.post(
    "/profile",
    response_model=VisionProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract Vision Profile from Answers",
    description=(
        "Processes user's answers to vision probe questions and extracts "
        "a complete VisionProfile containing:\n"
        "- Semantic anchors (non-negotiable conceptual constraints)\n"
        "- Innovation type classification\n"
        "- Abstraction level classification\n"
        "- Differentiation markers\n"
        "- Drift risk assessment\n\n"
        "The VisionProfile is passed as a hard constraint to all "
        "downstream pipeline stages."
    ),
)
async def extract_vision_profile(
    request: VisionProfileRequest,
    clarifier: Annotated[VisionClarifier, Depends(vision_dep)],
) -> VisionProfileResponse:
    """
    Extracts a complete VisionProfile from user's answers.

    The resulting VisionProfile becomes the semantic foundation
    for all downstream processing — requirement expansion,
    prompt optimization, and scoring all use it as a constraint.
    """
    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    logger.info(
        "Vision profile request | session={s} | answers={count}",
        s=request.session_id,
        count=len(request.answers),
    )

    try:
        return await clarifier.extract_profile(request)

    except RuntimeError as e:
        logger.error(
            "Vision profile LLM error | session={s} | error={e}",
            s=request.session_id,
            e=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service error: {str(e)}",
        )
    except Exception as e:
        logger.exception(
            "Vision profile unexpected error | session={s}",
            s=request.session_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vision profile extraction failed: {str(e)}",
        )