"""
FastAPI Application — Phase 5 Final Version.

All routers registered. Full semantic pipeline active.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.logger import logger, setup_logger
from backend.routers import (
    health,
    analysis,
    optimization,
    history as history_router,
    vision as vision_router,
    drift as drift_router,
)
from backend.routers import semantic as semantic_router
from backend.routers import pipeline as pipeline_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("🚀 AI Requirement Intelligence — Semantic Pipeline")
    logger.info(f"   Provider : {settings.LLM_PROVIDER.upper()}")
    logger.info(f"   Model    : {settings.active_llm_model}")
    logger.info("=" * 60)

    settings.ensure_directories()

    from backend.models.database import init_db
    await init_db()

    from backend.services.rag_service import get_rag_service
    rag = get_rag_service()
    await rag.initialize()

    logger.info("✅ All systems ready")
    yield
    logger.info("🛑 Shutting down")


def create_app() -> FastAPI:
    setup_logger(log_level=settings.LOG_LEVEL, log_dir=settings.LOG_DIR)

    app = FastAPI(
        title="AI Requirement Intelligence — Semantic Pipeline",
        description=(
            "Full semantic reasoning pipeline with vision preservation, "
            "drift detection, and innovation-aware prompt generation."
        ),
        version="2.0.0",
        docs_url=f"{settings.API_PREFIX}/docs",
        redoc_url=f"{settings.API_PREFIX}/redoc",
        openapi_url=f"{settings.API_PREFIX}/openapi.json",
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            f"http://localhost:{settings.FRONTEND_PORT}",
            "http://localhost:8501",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── All Routers ───────────────────────────────────────────────
    prefix = settings.API_PREFIX

    app.include_router(health.router,          prefix=prefix)
    app.include_router(analysis.router,        prefix=prefix)
    app.include_router(optimization.router,    prefix=prefix)
    app.include_router(history_router.router,  prefix=prefix)
    app.include_router(vision_router.router,   prefix=prefix)
    app.include_router(semantic_router.router, prefix=prefix)
    app.include_router(drift_router.router,    prefix=prefix)
    app.include_router(pipeline_router.router, prefix=prefix)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "version": "2.0.0",
            "docs":     f"{prefix}/docs",
            "pipeline": f"{prefix}/pipeline/run-vision",
            "vision":   f"{prefix}/vision/probe",
            "semantic": f"{prefix}/semantic/full",
            "drift":    f"{prefix}/drift/detect",
        }

    logger.info("✅ FastAPI app ready | all routers registered")
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )