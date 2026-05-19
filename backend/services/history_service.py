"""
History Service - Prompt Session & History Management.

Handles all database operations for:
- Saving complete optimization pipeline results
- Loading session history with pagination
- Retrieving individual session details
- Deleting sessions
- Generating simple analytics

Uses SQLAlchemy async ORM with SQLite via aiosqlite.
"""

import json
from datetime import datetime
from typing import Any

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.logger import logger
from backend.models.database import PromptRecord, PromptSession, RefinementLog
from backend.models.schemas import (
    OptimizationPipelineResponse,
    ScoredPrompt,
)


class HistoryService:
    """
    Service for persisting and retrieving prompt optimization history.

    All methods accept an AsyncSession parameter (dependency injected
    from FastAPI endpoints) for clean transaction management.
    """

    async def save_pipeline_result(
        self,
        db: AsyncSession,
        result: OptimizationPipelineResponse,
    ) -> PromptSession:
        """
        Saves a complete pipeline result to the database.

        Creates:
        - One PromptSession record (summary)
        - One PromptRecord per generated prompt style

        Args:
            db: Async database session
            result: Full pipeline response to persist

        Returns:
            Created PromptSession ORM object
        """
        logger.info(
            "Saving pipeline result | session={s} | prompts={count}",
            s=result.session_id,
            count=len(result.scored_prompts),
        )

        # ── Create session record ─────────────────────────────────
        best = result.best_prompt
        session = PromptSession(
            session_id=result.session_id,
            raw_requirement=result.raw_requirement,

            # Expansion data
            expanded_title=result.expanded_requirement.title,
            expanded_overview=result.expanded_requirement.overview,
            detected_complexity=result.expanded_requirement.estimated_complexity.value,

            # Best prompt data
            best_prompt_style=best.style.value if best else None,
            best_prompt_score=best.score.overall_score if best else None,
            best_prompt_grade=best.score.grade if best else None,
            best_prompt_text=best.prompt_text if best else None,

            # Metrics
            total_tokens_used=result.total_tokens_used,
            total_processing_time_ms=result.total_processing_time_ms,
        )
        db.add(session)
        await db.flush()  # Get the session ID without full commit

        # ── Create prompt records ─────────────────────────────────
        best_style = best.style.value if best else None

        for scored_prompt in result.scored_prompts:
            dims = {d.dimension.lower().replace(" ", "_"): d.score
                    for d in scored_prompt.score.dimensions}

            record = PromptRecord(
                session_id=result.session_id,
                style=scored_prompt.style.value,
                style_label=scored_prompt.style_label,
                prompt_text=scored_prompt.prompt_text,
                overall_score=scored_prompt.score.overall_score,
                grade=scored_prompt.score.grade,
                clarity_score=dims.get("clarity", None),
                specificity_score=dims.get("specificity", None),
                completeness_score=dims.get("completeness", None),
                actionability_score=dims.get("actionability", None),
                hallucination_reduction_score=dims.get("hallucination_reduction", None),
                hallucination_risk=scored_prompt.score.hallucination_risk,
                is_production_ready=scored_prompt.score.is_production_ready,
                word_count=scored_prompt.word_count,
                estimated_tokens=scored_prompt.estimated_tokens,
                is_best=(scored_prompt.style.value == best_style),
            )
            db.add(record)

        await db.commit()

        logger.info(
            "Pipeline result saved | session={s} | db_id={id}",
            s=result.session_id,
            id=session.id,
        )
        return session

    async def save_refinement_log(
        self,
        db: AsyncSession,
        session_id: str,
        iteration: int,
        input_prompt: str,
        refined_prompt: str,
        score_before: float,
        score_after: float,
        rag_context: str,
        feedback_applied: str,
        rag_docs_count: int = 0,
    ) -> RefinementLog:
        """
        Saves a single refinement iteration to the log.

        Args:
            db: Async database session
            session_id: Parent session ID
            iteration: Iteration number (1-based)
            input_prompt: Prompt before refinement
            refined_prompt: Prompt after refinement
            score_before: Quality score before
            score_after: Quality score after
            rag_context: RAG knowledge used
            feedback_applied: Description of improvements made
            rag_docs_count: Number of RAG docs retrieved

        Returns:
            Created RefinementLog ORM object
        """
        log = RefinementLog(
            session_id=session_id,
            iteration_number=iteration,
            input_prompt=input_prompt,
            refined_prompt=refined_prompt,
            score_before=score_before,
            score_after=score_after,
            score_delta=round(score_after - score_before, 2),
            grade_before=self._score_to_grade(score_before),
            grade_after=self._score_to_grade(score_after),
            rag_context_used=rag_context[:2000] if rag_context else None,
            rag_docs_count=rag_docs_count,
            feedback_applied=feedback_applied[:1000] if feedback_applied else None,
        )
        db.add(log)
        await db.commit()

        logger.info(
            "Refinement log saved | session={s} | iter={i} | "
            "delta={delta:+.1f} | {grade_b}→{grade_a}",
            s=session_id,
            i=iteration,
            delta=log.score_delta,
            grade_b=log.grade_before,
            grade_a=log.grade_after,
        )
        return log

    async def get_session_list(
        self,
        db: AsyncSession,
        page: int = 1,
        page_size: int = 20,
        min_score: float | None = None,
    ) -> dict[str, Any]:
        """
        Returns paginated list of prompt sessions.

        Args:
            db: Async database session
            page: Page number (1-based)
            page_size: Records per page
            min_score: Optional minimum score filter

        Returns:
            Dict with 'sessions', 'total', 'page', 'pages' keys
        """
        offset = (page - 1) * page_size

        # Build base query
        query = select(PromptSession).order_by(desc(PromptSession.created_at))

        if min_score is not None:
            query = query.where(PromptSession.best_prompt_score >= min_score)

        # Get total count
        count_query = select(func.count()).select_from(
            query.subquery()
        )
        total = await db.scalar(count_query) or 0

        # Get paginated results
        result = await db.execute(
            query.offset(offset).limit(page_size)
        )
        sessions = result.scalars().all()

        return {
            "sessions": [self._serialize_session(s) for s in sessions],
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": max(1, (total + page_size - 1) // page_size),
        }

    async def get_session_detail(
        self,
        db: AsyncSession,
        session_id: str,
    ) -> dict[str, Any] | None:
        """
        Returns full detail for a single session including all prompts.

        Args:
            db: Async database session
            session_id: UUID session identifier

        Returns:
            Full session dict or None if not found
        """
        result = await db.execute(
            select(PromptSession)
            .where(PromptSession.session_id == session_id)
            .options(
                selectinload(PromptSession.prompts),
                selectinload(PromptSession.refinements),
            )
        )
        session = result.scalar_one_or_none()

        if not session:
            return None

        session_dict = self._serialize_session(session)
        session_dict["prompts"] = [
            self._serialize_prompt_record(p) for p in session.prompts
        ]
        session_dict["refinements"] = [
            self._serialize_refinement(r) for r in session.refinements
        ]
        return session_dict

    async def delete_session(
        self,
        db: AsyncSession,
        session_id: str,
    ) -> bool:
        """
        Deletes a session and all associated records (cascade).

        Args:
            db: Async database session
            session_id: UUID session identifier

        Returns:
            True if deleted, False if not found
        """
        result = await db.execute(
            select(PromptSession).where(PromptSession.session_id == session_id)
        )
        session = result.scalar_one_or_none()

        if not session:
            return False

        await db.delete(session)
        await db.commit()

        logger.info("Session deleted | session_id={s}", s=session_id)
        return True

    async def get_analytics(self, db: AsyncSession) -> dict[str, Any]:
        """
        Returns simple analytics about prompt history.

        Returns:
            Dict with counts, averages, and distribution data
        """
        # Total sessions
        total_sessions = await db.scalar(select(func.count(PromptSession.id))) or 0

        # Total prompts
        total_prompts = await db.scalar(select(func.count(PromptRecord.id))) or 0

        # Average best score
        avg_score = await db.scalar(
            select(func.avg(PromptSession.best_prompt_score))
        )

        # Production-ready count
        production_ready = await db.scalar(
            select(func.count(PromptRecord.id)).where(
                PromptRecord.is_production_ready == True  # noqa: E712
            )
        ) or 0

        # Style distribution
        style_result = await db.execute(
            select(PromptRecord.style, func.count(PromptRecord.id))
            .group_by(PromptRecord.style)
            .order_by(desc(func.count(PromptRecord.id)))
        )
        style_distribution = {row[0]: row[1] for row in style_result.all()}

        # Grade distribution
        grade_result = await db.execute(
            select(PromptSession.best_prompt_grade, func.count(PromptSession.id))
            .where(PromptSession.best_prompt_grade.isnot(None))
            .group_by(PromptSession.best_prompt_grade)
        )
        grade_distribution = {row[0]: row[1] for row in grade_result.all()}

        # Total refinements
        total_refinements = await db.scalar(
            select(func.count(RefinementLog.id))
        ) or 0

        return {
            "total_sessions": total_sessions,
            "total_prompts": total_prompts,
            "total_refinements": total_refinements,
            "average_best_score": round(float(avg_score), 1) if avg_score else 0.0,
            "production_ready_count": production_ready,
            "production_ready_rate": (
                round(production_ready / total_prompts * 100, 1)
                if total_prompts > 0 else 0.0
            ),
            "style_distribution": style_distribution,
            "grade_distribution": grade_distribution,
        }

    # ── Private serializers ────────────────────────────────────────

    def _serialize_session(self, session: PromptSession) -> dict[str, Any]:
        """Converts ORM PromptSession to dict."""
        return {
            "id": session.id,
            "session_id": session.session_id,
            "raw_requirement": session.raw_requirement,
            "expanded_title": session.expanded_title,
            "detected_complexity": session.detected_complexity,
            "best_prompt_style": session.best_prompt_style,
            "best_prompt_score": session.best_prompt_score,
            "best_prompt_grade": session.best_prompt_grade,
            "best_prompt_text": session.best_prompt_text,
            "completeness_score": session.completeness_score,
            "total_tokens_used": session.total_tokens_used,
            "total_processing_time_ms": session.total_processing_time_ms,
            "created_at": session.created_at.isoformat() if session.created_at else None,
        }

    def _serialize_prompt_record(self, record: PromptRecord) -> dict[str, Any]:
        """Converts ORM PromptRecord to dict."""
        return {
            "id": record.id,
            "style": record.style,
            "style_label": record.style_label,
            "prompt_text": record.prompt_text,
            "overall_score": record.overall_score,
            "grade": record.grade,
            "clarity_score": record.clarity_score,
            "specificity_score": record.specificity_score,
            "completeness_score": record.completeness_score,
            "actionability_score": record.actionability_score,
            "hallucination_reduction_score": record.hallucination_reduction_score,
            "hallucination_risk": record.hallucination_risk,
            "is_production_ready": record.is_production_ready,
            "is_best": record.is_best,
            "word_count": record.word_count,
            "is_refined": record.is_refined,
            "refinement_count": record.refinement_count,
        }

    def _serialize_refinement(self, log: RefinementLog) -> dict[str, Any]:
        """Converts ORM RefinementLog to dict."""
        return {
            "id": log.id,
            "iteration_number": log.iteration_number,
            "score_before": log.score_before,
            "score_after": log.score_after,
            "score_delta": log.score_delta,
            "grade_before": log.grade_before,
            "grade_after": log.grade_after,
            "rag_docs_count": log.rag_docs_count,
            "feedback_applied": log.feedback_applied,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }

    def _score_to_grade(self, score: float | None) -> str:
        """Converts numeric score to letter grade."""
        if score is None:
            return "?"
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        return "F"


# ─── Singleton ────────────────────────────────────────────────────
_history_service: HistoryService | None = None


def get_history_service() -> HistoryService:
    """Returns the HistoryService singleton."""
    global _history_service
    if _history_service is None:
        _history_service = HistoryService()
    return _history_service