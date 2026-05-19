"""
Database Models & Engine Setup.

Uses SQLAlchemy async ORM with aiosqlite for SQLite.

Tables:
    prompt_sessions  → Stores each full optimization session
    prompt_history   → Individual prompt records with scores
    refinement_log   → Tracks multi-step refinement iterations

The DB file lives at: ./data/app.db
"""

from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship

from backend.config import settings
from backend.logger import logger


# ─── Base Class ───────────────────────────────────────────────────
class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ══════════════════════════════════════════════════════════════════
# ORM MODELS
# ══════════════════════════════════════════════════════════════════

class PromptSession(Base):
    """
    Represents one complete optimization session.
    A session = one user requirement → full pipeline run.
    """
    __tablename__ = "prompt_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), unique=True, nullable=False, index=True)
    raw_requirement = Column(Text, nullable=False)
    context = Column(Text, nullable=True)

    # Analysis results (stored as JSON strings)
    detected_domain = Column(String(100), nullable=True)
    detected_complexity = Column(String(20), nullable=True)
    primary_goal = Column(Text, nullable=True)
    completeness_score = Column(Float, nullable=True)

    # Expansion summary
    expanded_title = Column(String(200), nullable=True)
    expanded_overview = Column(Text, nullable=True)

    # Best prompt result
    best_prompt_style = Column(String(50), nullable=True)
    best_prompt_score = Column(Float, nullable=True)
    best_prompt_grade = Column(String(5), nullable=True)
    best_prompt_text = Column(Text, nullable=True)

    # Metrics
    total_tokens_used = Column(Integer, default=0)
    total_processing_time_ms = Column(Float, default=0.0)

    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    prompts = relationship(
        "PromptRecord",
        back_populates="session",
        cascade="all, delete-orphan",
    )
    refinements = relationship(
        "RefinementLog",
        back_populates="session",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"<PromptSession id={self.id} "
            f"session_id={self.session_id[:8]}... "
            f"score={self.best_prompt_score}>"
        )


class PromptRecord(Base):
    """
    Individual prompt record within a session.
    One record per prompt style generated.
    """
    __tablename__ = "prompt_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(36),
        ForeignKey("prompt_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Prompt details
    style = Column(String(50), nullable=False)
    style_label = Column(String(100), nullable=False)
    prompt_text = Column(Text, nullable=False)

    # Quality scores
    overall_score = Column(Float, nullable=True)
    grade = Column(String(5), nullable=True)
    clarity_score = Column(Float, nullable=True)
    specificity_score = Column(Float, nullable=True)
    completeness_score = Column(Float, nullable=True)
    actionability_score = Column(Float, nullable=True)
    hallucination_reduction_score = Column(Float, nullable=True)
    hallucination_risk = Column(String(20), nullable=True)
    is_production_ready = Column(Boolean, default=False)

    # Refinement tracking
    is_refined = Column(Boolean, default=False)
    refinement_count = Column(Integer, default=0)

    # Metadata
    word_count = Column(Integer, default=0)
    estimated_tokens = Column(Integer, default=0)
    is_best = Column(Boolean, default=False)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    session = relationship("PromptSession", back_populates="prompts")

    def __repr__(self) -> str:
        return (
            f"<PromptRecord id={self.id} "
            f"style={self.style} "
            f"score={self.overall_score} "
            f"grade={self.grade}>"
        )


class RefinementLog(Base):
    """
    Tracks each iteration of multi-step prompt refinement.
    One row per refinement iteration per session.
    """
    __tablename__ = "refinement_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String(36),
        ForeignKey("prompt_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Iteration info
    iteration_number = Column(Integer, nullable=False)
    input_prompt = Column(Text, nullable=False)
    refined_prompt = Column(Text, nullable=False)

    # Scores before and after
    score_before = Column(Float, nullable=True)
    score_after = Column(Float, nullable=True)
    score_delta = Column(Float, nullable=True)      # Improvement amount
    grade_before = Column(String(5), nullable=True)
    grade_after = Column(String(5), nullable=True)

    # RAG context used
    rag_context_used = Column(Text, nullable=True)   # Snippets retrieved
    rag_docs_count = Column(Integer, default=0)

    # Feedback applied
    feedback_applied = Column(Text, nullable=True)   # What was improved

    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    session = relationship("PromptSession", back_populates="refinements")

    def __repr__(self) -> str:
        return (
            f"<RefinementLog id={self.id} "
            f"iteration={self.iteration_number} "
            f"delta={self.score_delta}>"
        )


# ══════════════════════════════════════════════════════════════════
# ENGINE & SESSION FACTORY
# ══════════════════════════════════════════════════════════════════

# Create async engine (lazy - doesn't connect until first use)
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,          # Log SQL queries in debug mode
    connect_args={"check_same_thread": False},  # SQLite specific
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,       # Keep objects usable after commit
    autocommit=False,
    autoflush=False,
)


async def init_db() -> None:
    """
    Creates all database tables if they don't exist.
    Safe to call on every startup (idempotent).
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database tables initialized")


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides an async DB session.
    Automatically commits on success, rolls back on error.

    Usage in FastAPI:
        async def endpoint(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()