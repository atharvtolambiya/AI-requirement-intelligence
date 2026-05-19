"""
Health check endpoints for the FastAPI backend.

Endpoints:
    GET /api/v1/health  → Basic liveness check (is the server running?)
    GET /api/v1/ready   → Readiness check (are dependencies available?)

These are used by:
- Streamlit frontend to verify backend connectivity
- Docker health checks (if deployed with Docker)
- Monitoring tools
"""

import time
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from backend.config import settings
from backend.logger import logger

# ─── Router Setup ─────────────────────────────────────────────────
router = APIRouter(tags=["Health"])

# ─── Track server start time for uptime calculation ───────────────
_SERVER_START_TIME = time.time()


# ─── Response Schemas ─────────────────────────────────────────────
class HealthResponse(BaseModel):
    """Response schema for the basic health check endpoint."""
    status: str
    timestamp: str
    uptime_seconds: float
    version: str = "1.0.0"
    service: str = "AI Requirement Intelligence API"


class ReadinessResponse(BaseModel):
    """Response schema for the readiness check endpoint."""
    status: str
    timestamp: str
    checks: dict[str, str]
    llm_provider: str
    active_model: str


# ─── Endpoints ────────────────────────────────────────────────────
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness Check",
    description="Returns 200 if the API server is running.",
)
async def health_check() -> HealthResponse:
    """
    Liveness probe: confirms the server process is alive.
    Should always return 200 if the server is running.
    """
    uptime = round(time.time() - _SERVER_START_TIME, 2)
    logger.debug("Health check called | uptime={uptime}s", uptime=uptime)

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        uptime_seconds=uptime,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness Check",
    description="Returns status of all system dependencies.",
)
async def readiness_check() -> ReadinessResponse:
    """
    Readiness probe: verifies critical dependencies are configured.
    Checks API key presence without making actual LLM calls.
    """
    checks = {}
    overall_status = "ready"

    # ─── Check LLM API Key ────────────────────────────────────────
    if settings.active_api_key:
        checks["llm_api_key"] = "configured"
    else:
        checks["llm_api_key"] = "missing"
        overall_status = "degraded"
        logger.warning("Readiness check: LLM API key is not configured")

    # ─── Check Data Directory ─────────────────────────────────────
    from pathlib import Path
    if Path("./data").exists():
        checks["data_directory"] = "ok"
    else:
        checks["data_directory"] = "missing"
        overall_status = "degraded"

    # ─── Check Log Directory ──────────────────────────────────────
    if Path(settings.LOG_DIR).exists():
        checks["log_directory"] = "ok"
    else:
        checks["log_directory"] = "missing"

    logger.debug(
        "Readiness check | status={status} | checks={checks}",
        status=overall_status,
        checks=checks,
    )

    return ReadinessResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        checks=checks,
        llm_provider=settings.LLM_PROVIDER,
        active_model=settings.active_llm_model,
    )