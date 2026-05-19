"""
Semantic Router — Phase 3 API Endpoints.

Endpoints:
    POST /api/v1/semantic/intent         → Semantic intent extraction
    POST /api/v1/semantic/innovation     → Innovation analysis
    POST /api/v1/semantic/abstraction    → Abstraction classification
    POST /api/v1/semantic/full           → All three in one call
"""

import asyncio
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from backend.logger import logger
from backend.models.schemas import (
    AbstractionClassificationRequest,
    AbstractionClassificationResponse,
    FullSemanticAnalysisRequest,
    FullSemanticAnalysisResponse,
    InnovationAnalysisRequest,
    InnovationAnalysisResponse,
    SemanticIntentRequest,
    SemanticIntentResponse,
)
from backend.services.abstraction_classifier import (
    AbstractionClassifier,
    get_abstraction_classifier,
)
from backend.services.innovation_analyzer import (
    InnovationAnalyzer,
    get_innovation_analyzer,
)
from backend.services.semantic_intent_extractor import (
    SemanticIntentExtractor,
    get_semantic_intent_extractor,
)

# ─── Router ───────────────────────────────────────────────────────
router = APIRouter(
    prefix="/semantic",
    tags=["Semantic Analysis"],
)


# ─── Dependencies ─────────────────────────────────────────────────
def intent_dep() -> SemanticIntentExtractor:
    return get_semantic_intent_extractor()


def innovation_dep() -> InnovationAnalyzer:
    return get_innovation_analyzer()


def abstraction_dep() -> AbstractionClassifier:
    return get_abstraction_classifier()


# ══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@router.post(
    "/intent",
    response_model=SemanticIntentResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract Semantic Intent",
    description=(
        "Extracts deep conceptual intent beyond keyword matching. "
        "Builds a concept map with collapse risk assessment for each concept."
    ),
)
async def extract_semantic_intent(
    request: SemanticIntentRequest,
    extractor: Annotated[SemanticIntentExtractor, Depends(intent_dep)],
) -> SemanticIntentResponse:
    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    logger.info(
        "Semantic intent request | session={s}",
        s=request.session_id,
    )

    try:
        return await extractor.extract(request)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM error: {str(e)}",
        )
    except Exception as e:
        logger.exception("Semantic intent error | session={s}", s=request.session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic intent extraction failed: {str(e)}",
        )


@router.post(
    "/innovation",
    response_model=InnovationAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Innovation",
    description=(
        "Identifies innovation tier (incremental/novel/transformative), "
        "generates anti-patterns to prevent collapse, "
        "and creates preservation rules for downstream stages."
    ),
)
async def analyze_innovation(
    request: InnovationAnalysisRequest,
    analyzer: Annotated[InnovationAnalyzer, Depends(innovation_dep)],
) -> InnovationAnalysisResponse:
    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    logger.info(
        "Innovation analysis request | session={s}",
        s=request.session_id,
    )

    try:
        return await analyzer.analyze(request)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM error: {str(e)}",
        )
    except Exception as e:
        logger.exception("Innovation analysis error | session={s}", s=request.session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Innovation analysis failed: {str(e)}",
        )


@router.post(
    "/abstraction",
    response_model=AbstractionClassificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify Abstraction Level",
    description=(
        "Detects the abstraction level gap between user intent and "
        "system default. Generates specific correction instructions "
        "for the requirement expander."
    ),
)
async def classify_abstraction(
    request: AbstractionClassificationRequest,
    classifier: Annotated[AbstractionClassifier, Depends(abstraction_dep)],
) -> AbstractionClassificationResponse:
    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    logger.info(
        "Abstraction classification request | session={s}",
        s=request.session_id,
    )

    try:
        return await classifier.classify(request)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM error: {str(e)}",
        )
    except Exception as e:
        logger.exception("Abstraction classification error | session={s}", s=request.session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Abstraction classification failed: {str(e)}",
        )


@router.post(
    "/full",
    response_model=FullSemanticAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Full Semantic Analysis ⭐",
    description=(
        "Runs all three semantic analyses in parallel:\n"
        "1. Semantic Intent Extraction\n"
        "2. Innovation Analysis\n"
        "3. Abstraction Classification\n\n"
        "All three share the vision profile as context. "
        "Results feed directly into Phase 4 drift detection."
    ),
)
async def full_semantic_analysis(
    request: FullSemanticAnalysisRequest,
    extractor:  Annotated[SemanticIntentExtractor, Depends(intent_dep)],
    analyzer:   Annotated[InnovationAnalyzer,      Depends(innovation_dep)],
    classifier: Annotated[AbstractionClassifier,   Depends(abstraction_dep)],
) -> FullSemanticAnalysisResponse:
    """
    Runs all three semantic analyses concurrently.

    Uses asyncio.gather() for parallel execution — all three
    LLM calls happen simultaneously for performance.
    """
    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    logger.info(
        "Full semantic analysis | session={s} | preview={p}",
        s=request.session_id,
        p=request.raw_input[:60],
    )

    try:
        #  Build individual requests
        vp = request.vision_profile

        intent_request = SemanticIntentRequest(
            raw_input=request.raw_input,
            vision_profile=vp,
            session_id=request.session_id,
        )
        innovation_request = InnovationAnalysisRequest(
            raw_input=request.raw_input,
            vision_profile=vp,
            session_id=request.session_id,
        )
        abstraction_request = AbstractionClassificationRequest(
            raw_input=request.raw_input,
            vision_profile=vp,
            session_id=request.session_id,
        )

        #  Run all three concurrently
        logger.info("Running 3 semantic analyses in parallel")

        intent_resp, innovation_resp, abstraction_resp = (
            await asyncio.gather(
                extractor.extract(intent_request),
                analyzer.analyze(innovation_request),
                classifier.classify(abstraction_request),
            )
        )

        # Pass intent into innovation + abstraction for richer results
        # (second pass with more context if needed in future phases)

        total_time = round(
            intent_resp.processing_time_ms
            + innovation_resp.processing_time_ms
            + abstraction_resp.processing_time_ms,
            2,
        )
        total_tokens = (
            intent_resp.tokens_used
            + innovation_resp.tokens_used
            + abstraction_resp.tokens_used
        )

        logger.info(
            "Full semantic analysis complete | session={s} | "
            "tokens={t} | time={ms}ms",
            s=request.session_id,
            t=total_tokens,
            ms=total_time,
        )

        return FullSemanticAnalysisResponse(
            success=True,
            session_id=request.session_id,
            raw_input=request.raw_input,
            semantic_intent=intent_resp.semantic_intent,
            innovation_profile=innovation_resp.innovation_profile,
            classification=abstraction_resp.classification,
            total_processing_time_ms=total_time,
            total_tokens_used=total_tokens,
        )

    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM error: {str(e)}",
        )
    except Exception as e:
        logger.exception(
            "Full semantic analysis error | session={s}",
            s=request.session_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Full semantic analysis failed: {str(e)}",
        )