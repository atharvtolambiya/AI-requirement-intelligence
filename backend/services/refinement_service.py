"""
Multi-step Prompt Refinement Service.

Implements an iterative refinement loop that:
1. Takes the best generated prompt
2. Retrieves relevant knowledge from ChromaDB (RAG)
3. Identifies specific weaknesses from scoring feedback
4. Applies targeted improvements using LLM
5. Re-scores the refined prompt
6. Repeats until quality target is met or max iterations reached
7. Logs each iteration for history tracking

Refinement stops when:
- Score reaches target threshold (default: 80)
- Max iterations reached (default: 3)
- Score improvement < minimum delta (default: 3 points)
"""

import time
from typing import Any

from backend.logger import logger
from backend.models.schemas import (
    PromptQualityScore,
    PromptStyle,
    ScoredPrompt,
)
from backend.services.llm_service import get_llm_service
from backend.services.prompt_scorer import get_prompt_scorer
from backend.services.rag_service import get_rag_service


# ─── Refinement Configuration ──────────────────────────────────────
DEFAULT_TARGET_SCORE = 80.0      # Stop refining at this score
DEFAULT_MAX_ITERATIONS = 3       # Maximum refinement passes
DEFAULT_MIN_IMPROVEMENT = 2.0    # Minimum score gain per iteration


# ─── Refinement System Prompt ──────────────────────────────────────
REFINEMENT_SYSTEM_PROMPT = """You are an expert Prompt Engineer specializing in 
iterative prompt improvement using evidence-based techniques.

Your task is to REFINE an existing prompt to improve its quality.
You will be given:
1. The current prompt
2. Quality score breakdown with specific weaknesses
3. Relevant prompt engineering knowledge to apply

REFINEMENT RULES:
1. PRESERVE the core intent and task — do not change what the prompt is asking for
2. ADDRESS each identified weakness specifically
3. APPLY relevant knowledge from the provided best practices
4. IMPROVE clarity: replace vague terms with specific, measurable ones
5. ADD missing constraints: what the LLM should NOT do
6. SPECIFY output format if not already defined
7. REDUCE hallucination risk: add scope limits and uncertainty handling
8. STRENGTHEN role/context if using role-based approach

SPECIFIC IMPROVEMENTS BY WEAKNESS TYPE:
- Low Clarity: Rewrite ambiguous sentences; define technical terms
- Low Specificity: Replace "good/fast/detailed" with concrete values
- Low Completeness: Add missing context (audience, constraints, background)
- Low Actionability: Add clear output format; remove blocking ambiguities
- High Hallucination Risk: Add "only use provided information"; add uncertainty instruction

OUTPUT FORMAT:
Respond ONLY with the refined prompt text.
Do NOT include explanations, commentary, or preamble.
The refined prompt must be immediately usable — no placeholders."""


REFINEMENT_USER_TEMPLATE = """Refine this prompt to address the identified weaknesses:

CURRENT PROMPT:
---
{current_prompt}
---

QUALITY ASSESSMENT:
Overall Score: {overall_score}/100 (Grade: {grade})

WEAKNESSES TO ADDRESS:
{weaknesses}

DIMENSION SCORES (focus on lowest scores):
{dimension_scores}

RELEVANT BEST PRACTICES TO APPLY:
{rag_context}

Generate the refined prompt now (prompt text only, no explanation):"""


class RefinementResult:
    """Holds the complete result of a refinement run."""

    def __init__(
        self,
        original_prompt: ScoredPrompt,
        final_prompt: ScoredPrompt,
        iterations: list[dict[str, Any]],
        total_improvement: float,
        target_reached: bool,
        stop_reason: str,
        total_tokens: int,
        total_time_ms: float,
    ):
        self.original_prompt = original_prompt
        self.final_prompt = final_prompt
        self.iterations = iterations
        self.total_improvement = total_improvement
        self.target_reached = target_reached
        self.stop_reason = stop_reason
        self.total_tokens = total_tokens
        self.total_time_ms = total_time_ms


class RefinementService:
    """
    Iterative prompt refinement using RAG + LLM feedback loop.

    The refinement loop:
    ┌─────────────────────────────────────────────────────┐
    │  Input: ScoredPrompt (with weaknesses from scorer)  │
    │         ↓                                           │
    │  1. Retrieve RAG knowledge for weaknesses           │
    │         ↓                                           │
    │  2. LLM applies improvements + RAG knowledge        │
    │         ↓                                           │
    │  3. Score refined prompt                            │
    │         ↓                                           │
    │  4. Check stop conditions                           │
    │         ↓                                           │
    │  5. Loop or return best result                      │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(self):
        self.llm_service = get_llm_service()
        self.rag_service = get_rag_service()
        self.scorer = get_prompt_scorer()
        logger.info("RefinementService initialized")

    async def refine(
        self,
        prompt: ScoredPrompt,
        original_requirement: str,
        target_score: float = DEFAULT_TARGET_SCORE,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
        session_id: str | None = None,
    ) -> RefinementResult:
        """
        Runs the multi-step refinement loop.

        Args:
            prompt: Initial ScoredPrompt to refine
            original_requirement: The user's original requirement
            target_score: Stop when this score is reached
            max_iterations: Maximum number of refinement passes
            min_improvement: Stop if improvement < this per iteration
            session_id: Optional session ID for logging

        Returns:
            RefinementResult with original, final, and all iterations
        """
        start_time = time.time()
        total_tokens = 0
        iterations: list[dict[str, Any]] = []

        logger.info(
            "Starting refinement | session={s} | initial_score={score} | "
            "target={target} | max_iter={max}",
            s=session_id,
            score=prompt.score.overall_score,
            target=target_score,
            max=max_iterations,
        )

        # ── Check if refinement is needed ─────────────────────────
        if prompt.score.overall_score >= target_score:
            logger.info(
                "Prompt already meets target score ({score} >= {target}) — "
                "skipping refinement",
                score=prompt.score.overall_score,
                target=target_score,
            )
            return RefinementResult(
                original_prompt=prompt,
                final_prompt=prompt,
                iterations=[],
                total_improvement=0.0,
                target_reached=True,
                stop_reason="already_meets_target",
                total_tokens=0,
                total_time_ms=0.0,
            )

        # ── Initialize RAG ────────────────────────────────────────
        await self.rag_service.initialize()

        current_prompt = prompt
        original_prompt = prompt
        stop_reason = "max_iterations_reached"

        for iteration in range(1, max_iterations + 1):
            logger.info(
                "Refinement iteration {iter}/{max} | current_score={score}",
                iter=iteration,
                max=max_iterations,
                score=current_prompt.score.overall_score,
            )

            iter_start = time.time()

            # ── Step 1: Retrieve RAG knowledge ────────────────────
            rag_docs = await self.rag_service.retrieve_for_refinement(
                prompt_text=current_prompt.prompt_text,
                weaknesses=current_prompt.score.weaknesses,
            )
            rag_context = self.rag_service.format_context_for_prompt(
                rag_docs,
                max_chars=1500,
            )

            # ── Step 2: Apply refinement via LLM ──────────────────
            refined_text, tokens = await self._apply_refinement(
                current_prompt=current_prompt,
                rag_context=rag_context,
            )
            total_tokens += tokens

            # ── Step 3: Score the refined prompt ──────────────────
            scored_refined = await self.scorer.score_single(
                prompt=ScoredPrompt(
                    style=current_prompt.style,
                    style_label=current_prompt.style_label,
                    prompt_text=refined_text,
                    score=current_prompt.score,  # Temporary, will be replaced
                    word_count=len(refined_text.split()),
                    estimated_tokens=self.llm_service.count_tokens(refined_text),
                ),
                original_requirement=original_requirement,
            )
            total_tokens += self.llm_service.count_tokens(refined_text)

            score_before = current_prompt.score.overall_score
            score_after = scored_refined.score.overall_score
            improvement = score_after - score_before
            iter_time_ms = round((time.time() - iter_start) * 1000, 2)

            # ── Log this iteration ─────────────────────────────────
            iter_data = {
                "iteration": iteration,
                "score_before": round(score_before, 1),
                "score_after": round(score_after, 1),
                "improvement": round(improvement, 1),
                "grade_before": current_prompt.score.grade,
                "grade_after": scored_refined.score.grade,
                "rag_docs_retrieved": len(rag_docs),
                "rag_context": rag_context,
                "prompt_before": current_prompt.prompt_text,
                "prompt_after": refined_text,
                "time_ms": iter_time_ms,
            }
            iterations.append(iter_data)

            logger.info(
                "Iteration {iter} complete | {grade_b}({score_b}) → "
                "{grade_a}({score_a}) | delta={delta:+.1f}",
                iter=iteration,
                grade_b=iter_data["grade_before"],
                score_b=iter_data["score_before"],
                grade_a=iter_data["grade_after"],
                score_a=iter_data["score_after"],
                delta=improvement,
            )

            # Update current prompt to refined version
            current_prompt = scored_refined

            # ── Check stop conditions ─────────────────────────────
            if score_after >= target_score:
                stop_reason = "target_score_reached"
                logger.info(
                    "Target score reached: {score} >= {target}",
                    score=score_after,
                    target=target_score,
                )
                break

            if improvement < min_improvement and iteration > 1:
                stop_reason = "insufficient_improvement"
                logger.info(
                    "Stopping: improvement {delta:.1f} < min {min}",
                    delta=improvement,
                    min=min_improvement,
                )
                break

        # ── Compute final metrics ─────────────────────────────────
        total_time_ms = round((time.time() - start_time) * 1000, 2)
        total_improvement = round(
            current_prompt.score.overall_score - original_prompt.score.overall_score,
            1,
        )

        logger.info(
            "Refinement complete | session={s} | iterations={iters} | "
            "improvement={delta:+.1f} | {grade_start}→{grade_end} | "
            "stop_reason={reason} | time={time}ms",
            s=session_id,
            iters=len(iterations),
            delta=total_improvement,
            grade_start=original_prompt.score.grade,
            grade_end=current_prompt.score.grade,
            reason=stop_reason,
            time=total_time_ms,
        )

        return RefinementResult(
            original_prompt=original_prompt,
            final_prompt=current_prompt,
            iterations=iterations,
            total_improvement=total_improvement,
            target_reached=(current_prompt.score.overall_score >= target_score),
            stop_reason=stop_reason,
            total_tokens=total_tokens,
            total_time_ms=total_time_ms,
        )

    async def _apply_refinement(
        self,
        current_prompt: ScoredPrompt,
        rag_context: str,
    ) -> tuple[str, int]:
        """
        Calls LLM to apply targeted improvements to the prompt.

        Args:
            current_prompt: The prompt to refine with its score
            rag_context: Retrieved best-practice knowledge

        Returns:
            Tuple of (refined_prompt_text, tokens_used)
        """
        # Format dimension scores for the refinement prompt
        dim_lines = []
        for dim in current_prompt.score.dimensions:
            bar = "█" * int(dim.score) + "░" * (10 - int(dim.score))
            dim_lines.append(
                f"  {dim.dimension}: {dim.score:.1f}/10 [{bar}] — {dim.feedback}"
            )
        dimension_scores = "\n".join(dim_lines)

        # Format weaknesses
        weakness_lines = [f"  • {w}" for w in current_prompt.score.weaknesses[:5]]
        weaknesses = "\n".join(weakness_lines) if weakness_lines else "  • None identified"

        user_message = REFINEMENT_USER_TEMPLATE.format(
            current_prompt=current_prompt.prompt_text,
            overall_score=current_prompt.score.overall_score,
            grade=current_prompt.score.grade,
            weaknesses=weaknesses,
            dimension_scores=dimension_scores,
            rag_context=rag_context or "No specific knowledge retrieved.",
        )

        llm_result = await self.llm_service.invoke(
            system_prompt=REFINEMENT_SYSTEM_PROMPT,
            user_message=user_message,
            temperature_override=0.2,  # Low temp for precise improvements
        )

        refined_text = llm_result["content"].strip()

        # Remove any accidental wrapping (markdown code blocks, quotes)
        if refined_text.startswith("```"):
            lines = refined_text.split("\n")
            refined_text = "\n".join(lines[1:-1])
        refined_text = refined_text.strip('"\'`')

        if not refined_text:
            logger.warning("LLM returned empty refinement — using original")
            refined_text = current_prompt.prompt_text

        return refined_text, llm_result["tokens_used"]


# ─── Singleton ────────────────────────────────────────────────────
_refinement_service: RefinementService | None = None


def get_refinement_service() -> RefinementService:
    """Returns the RefinementService singleton."""
    global _refinement_service
    if _refinement_service is None:
        _refinement_service = RefinementService()
    return _refinement_service