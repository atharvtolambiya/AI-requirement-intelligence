"""
Prompt Scorer — UPGRADED for Phase 5.

New scoring dimensions added:
- Vision Alignment (20%): Does prompt preserve the user's vision?
- Innovation Preservation (15%): Are innovation claims intact?

Existing dimensions adjusted weights:
- Clarity (20%)
- Specificity (20%)
- Completeness (15%)
- Actionability (10%)
- Hallucination Reduction (10%)
- Vision Alignment (20%) ← NEW
- Innovation Preservation (15%) ← NEW ... wait, total = 90%

Adjusted final weights to sum to 1.0:
- Clarity (15%)
- Specificity (15%)
- Completeness (15%)
- Actionability (10%)
- Hallucination Reduction (10%)
- Vision Alignment (20%)
- Innovation Preservation (15%)
Total = 100%
"""

import asyncio
import json
import re
import time
from typing import Any

from backend.logger import logger
from backend.models.schemas import (
    OptimizedPrompt,
    PromptQualityScore,
    PromptScoringRequest,
    PromptScoringResponse,
    PromptStyle,
    ScoreDimension,
    ScoredPrompt,
)
from backend.services.llm_service import get_llm_service


# ── Updated scoring dimensions ─────────────────────────────────────
SCORING_DIMENSIONS = [
    {
        "name": "clarity",
        "weight": 0.15,
        "description": "Unambiguous, clear instructions",
    },
    {
        "name": "specificity",
        "weight": 0.15,
        "description": "Concrete vs vague terms",
    },
    {
        "name": "completeness",
        "weight": 0.15,
        "description": "All necessary context included",
    },
    {
        "name": "actionability",
        "weight": 0.10,
        "description": "LLM can immediately execute",
    },
    {
        "name": "hallucination_reduction",
        "weight": 0.10,
        "description": "Reduces hallucination risk",
    },
    {
        "name": "vision_alignment",
        "weight": 0.20,
        "description": "Preserves user's conceptual vision",
    },
    {
        "name": "innovation_preservation",
        "weight": 0.15,
        "description": "Innovation claims preserved correctly",
    },
]


# ─── Scoring System Prompt ─────────────────────────────────────────
SCORING_SYSTEM_PROMPT = """You are an expert Prompt Quality Evaluator with specialization
in assessing AI prompts for conceptual fidelity and innovation preservation.

SCORING DIMENSIONS (0-10 each):

1. CLARITY (weight: 15%)
   9-10: Crystal clear, zero ambiguity
   5-8:  Mostly clear, minor ambiguity
   0-4:  Multiple unclear sections

2. SPECIFICITY (weight: 15%)
   9-10: All vague terms replaced with concrete ones
   5-8:  Mix of specific and vague
   0-4:  Mostly vague or generic

3. COMPLETENESS (weight: 15%)
   9-10: All necessary context included
   5-8:  Some context missing
   0-4:  Major context gaps

4. ACTIONABILITY (weight: 10%)
   9-10: LLM can immediately produce high-quality output
   5-8:  Minor clarification might help
   0-4:  Cannot be effectively acted upon

5. HALLUCINATION_REDUCTION (weight: 10%)
   9-10: Excellent constraints, scope limits
   5-8:  Some risk areas unaddressed
   0-4:  High hallucination risk

6. VISION_ALIGNMENT (weight: 20%)
   9-10: Prompt perfectly reflects the user's conceptual vision
         at the correct abstraction level
   7-8:  Mostly aligned, minor drift
   5-6:  Some important vision elements missing
   3-4:  Significant vision drift — wrong abstraction level
   0-2:  Prompt describes a completely different product

7. INNOVATION_PRESERVATION (weight: 15%)
   9-10: All innovation claims preserved with original meaning
   7-8:  Most claims preserved, minor dilution
   5-6:  Some claims diluted or oversimplified
   3-4:  Multiple claims replaced with generic equivalents
   0-2:  Innovation completely collapsed into generic pattern

HALLUCINATION RISK: low|medium|high
PRODUCTION READY: score >= 70

RESPOND WITH VALID JSON ONLY:
{
  "dimensions": {
    "clarity": {"score": 0-10, "feedback": "...", "suggestions": ["..."]},
    "specificity": {"score": 0-10, "feedback": "...", "suggestions": ["..."]},
    "completeness": {"score": 0-10, "feedback": "...", "suggestions": ["..."]},
    "actionability": {"score": 0-10, "feedback": "...", "suggestions": ["..."]},
    "hallucination_reduction": {"score": 0-10, "feedback": "...", "suggestions": ["..."]},
    "vision_alignment": {"score": 0-10, "feedback": "...", "suggestions": ["..."]},
    "innovation_preservation": {"score": 0-10, "feedback": "...", "suggestions": ["..."]}
  },
  "strengths": ["..."],
  "weaknesses": ["..."],
  "hallucination_risk": "low|medium|high",
  "hallucination_reasons": ["..."],
  "overall_feedback": "..."
}"""


SCORING_USER_TEMPLATE = """Score this prompt against the original requirement and vision:

ORIGINAL REQUIREMENT:
{original_requirement}

{vision_section}

PROMPT TO SCORE (Style: {style_label}):
---
{prompt_text}
---

Evaluate all 7 dimensions. Pay special attention to vision_alignment
and innovation_preservation — these are weighted highest.

Respond with JSON only:"""


class PromptScorer:
    """
    Vision-aware prompt scorer.

    Upgraded with two new dimensions:
    - vision_alignment: Does the prompt reflect the user's vision?
    - innovation_preservation: Are innovation claims preserved?
    """

    def __init__(self):
        self.llm_service = get_llm_service()
        logger.info("PromptScorer initialized (Phase 5 upgrade)")

    async def score_prompts(
        self,
        request: PromptScoringRequest,
        vision_profile: dict | None = None,
    ) -> PromptScoringResponse:
        """
        Scores prompts with vision alignment awareness.

        Args:
            request: PromptScoringRequest
            vision_profile: VisionProfile for alignment scoring

        Returns:
            PromptScoringResponse with vision-aware scores
        """
        start_time = time.time()
        logger.info(
            "Scoring {c} prompts | has_vision={v}",
            c=len(request.prompts),
            v=vision_profile is not None,
        )

        tasks = [
            self._score_single_prompt(
                prompt_data=p,
                original_requirement=request.original_requirement,
                vision_profile=vision_profile,
            )
            for p in request.prompts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        scored_prompts: list[ScoredPrompt] = []
        total_tokens = 0

        for prompt_data, result in zip(request.prompts, results):
            if isinstance(result, Exception):
                logger.warning(
                    "Scoring failed | error={e}", e=str(result)
                )
                scored_prompts.append(
                    self._create_zero_score_prompt(prompt_data)
                )
            else:
                scored_prompt, tokens = result
                scored_prompts.append(scored_prompt)
                total_tokens += tokens

        # Sort by overall score
        scored_prompts.sort(
            key=lambda x: x.score.overall_score, reverse=True
        )

        best_prompt = scored_prompts[0] if scored_prompts else None
        total_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(
            "Scoring complete | best={s} | grade={g} | time={t}ms",
            s=best_prompt.score.overall_score if best_prompt else 0,
            g=best_prompt.score.grade if best_prompt else "?",
            t=total_ms,
        )

        return PromptScoringResponse(
            success=True,
            session_id=request.session_id,
            scored_prompts=scored_prompts,
            best_prompt=best_prompt,
            processing_time_ms=total_ms,
            tokens_used=total_tokens,
        )

    async def score_single(
        self,
        prompt: OptimizedPrompt,
        original_requirement: str,
        vision_profile: dict | None = None,
    ) -> ScoredPrompt:
        """Scores a single OptimizedPrompt directly."""
        prompt_data = {
            "style": prompt.style.value,
            "prompt_text": prompt.prompt_text,
            "style_label": prompt.style_label,
        }
        scored, _ = await self._score_single_prompt(
            prompt_data=prompt_data,
            original_requirement=original_requirement,
            vision_profile=vision_profile,
        )
        return scored

    async def _score_single_prompt(
        self,
        prompt_data: dict[str, str],
        original_requirement: str,
        vision_profile: dict | None = None,
    ) -> tuple[ScoredPrompt, int]:
        """Scores a single prompt with vision awareness."""
        prompt_text = prompt_data.get("prompt_text", "")
        style_str   = prompt_data.get("style", "zero_shot")
        style_label = prompt_data.get(
            "style_label", style_str.replace("_", " ").title()
        )

        vision_section = self._build_vision_section(vision_profile)

        user_message = SCORING_USER_TEMPLATE.format(
            original_requirement=str(original_requirement)[:4000],
            vision_section=vision_section,
            prompt_text=str(prompt_text)[:4000],
            style_label=style_label,
        )

        llm_result = await self.llm_service.invoke(
            system_prompt=SCORING_SYSTEM_PROMPT,
            user_message=user_message,
            temperature_override=0.1,
        )

        score_data = self._parse_json(llm_result["content"])
        quality_score = self._build_quality_score(score_data)

        try:
            style = PromptStyle(style_str)
        except ValueError:
            style = PromptStyle.ZERO_SHOT

        scored = ScoredPrompt(
            style=style,
            style_label=style_label,
            prompt_text=prompt_text,
            score=quality_score,
            word_count=len(prompt_text.split()),
            estimated_tokens=self.llm_service.count_tokens(prompt_text),
        )

        logger.debug(
            "Scored | style={s} | score={sc} | grade={g} | "
            "vision={v} | innovation={i}",
            s=style_str,
            sc=quality_score.overall_score,
            g=quality_score.grade,
            v=next(
                (
                    d.score for d in quality_score.dimensions
                    if "vision" in d.dimension.lower()
                ),
                0,
            ),
            i=next(
                (
                    d.score for d in quality_score.dimensions
                    if "innovation" in d.dimension.lower()
                ),
                0,
            ),
        )

        return scored, llm_result["tokens_used"]

    def _build_vision_section(self, vision_profile: dict | None) -> str:
        """Formats vision profile for scoring context."""
        if not vision_profile:
            return ""

        lines = ["VISION PROFILE (use for vision_alignment scoring):"]
        for key in [
            "primary_user", "core_transformation",
            "intelligence_description", "innovation_type",
            "dominant_collapse_pattern",
        ]:
            val = vision_profile.get(key, "")
            if val:
                lines.append(f"  {key}: {val}")

        anchors = vision_profile.get("semantic_anchors", [])
        if anchors:
            lines.append("  Anchors that MUST appear:")
            for a in anchors[:4]:
                if isinstance(a, dict):
                    lines.append(f"    - {a.get('concept', '')}")

        return "\n".join(lines) + "\n"

    def _build_quality_score(
        self, data: dict[str, Any]
    ) -> PromptQualityScore:
        """Builds PromptQualityScore from parsed data."""
        raw_dims = data.get("dimensions", {})
        score_dimensions: list[ScoreDimension] = []
        weighted_total = 0.0
        total_weight   = 0.0

        for dim_cfg in SCORING_DIMENSIONS:
            name   = dim_cfg["name"]
            weight = dim_cfg["weight"]
            dd     = raw_dims.get(name, {})

            try:
                score = float(dd.get("score", 5.0))
                score = max(0.0, min(10.0, score))
            except (TypeError, ValueError):
                score = 5.0

            feedback    = str(dd.get("feedback", ""))
            suggestions = [
                str(s) for s in dd.get("suggestions", []) if s
            ]

            score_dimensions.append(
                ScoreDimension(
                    dimension=name.replace("_", " ").title(),
                    score=score,
                    weight=weight,
                    feedback=feedback,
                    suggestions=suggestions,
                )
            )

            weighted_total += score * weight
            total_weight   += weight

        overall = round(
            (weighted_total / total_weight * 10) if total_weight > 0 else 50.0,
            1,
        )
        overall = max(0.0, min(100.0, overall))

        grade = self._score_to_grade(overall)
        is_ready = overall >= 70.0

        hallucination_risk = str(
            data.get("hallucination_risk", "medium")
        ).lower()
        if hallucination_risk not in {"low", "medium", "high"}:
            hallucination_risk = "medium"

        def safe_list(key: str) -> list[str]:
            val = data.get(key, [])
            return (
                [str(s) for s in val if s]
                if isinstance(val, list) else []
            )

        return PromptQualityScore(
            overall_score=overall,
            grade=grade,
            dimensions=score_dimensions,
            strengths=safe_list("strengths"),
            weaknesses=safe_list("weaknesses"),
            hallucination_risk=hallucination_risk,
            hallucination_reasons=safe_list("hallucination_reasons"),
            overall_feedback=str(data.get("overall_feedback", "")),
            is_production_ready=is_ready,
        )

    def _score_to_grade(self, score: float) -> str:
        """Converts score to letter grade."""
        if score >= 90: return "A+"
        if score >= 80: return "A"
        if score >= 70: return "B"
        if score >= 60: return "C"
        if score >= 50: return "D"
        return "F"

    def _create_zero_score_prompt(
        self, prompt_data: dict[str, str]
    ) -> ScoredPrompt:
        """Creates zero-score prompt for failed scoring."""
        style_str = prompt_data.get("style", "zero_shot")
        try:
            style = PromptStyle(style_str)
        except ValueError:
            style = PromptStyle.ZERO_SHOT

        zero = PromptQualityScore(
            overall_score=0.0,
            grade="F",
            dimensions=[
                ScoreDimension(
                    dimension=d["name"].replace("_", " ").title(),
                    score=0.0,
                    weight=d["weight"],
                    feedback="Scoring failed.",
                    suggestions=["Retry"],
                )
                for d in SCORING_DIMENSIONS
            ],
            strengths=[],
            weaknesses=["Scoring failed"],
            hallucination_risk="high",
            hallucination_reasons=["Could not assess"],
            overall_feedback="Scoring failed. Please retry.",
            is_production_ready=False,
        )

        return ScoredPrompt(
            style=style,
            style_label=style_str.replace("_", " ").title(),
            prompt_text=prompt_data.get("prompt_text", ""),
            score=zero,
            word_count=0,
            estimated_tokens=0,
        )

    def _parse_json(self, content: str) -> dict[str, Any]:
        """Robust JSON extraction."""
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass
        match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL
        )
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        start = content.find("{")
        end   = content.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass
        return {}


# ─── Singleton ────────────────────────────────────────────────────
_scorer: PromptScorer | None = None


def get_prompt_scorer() -> PromptScorer:
    global _scorer
    if _scorer is None:
        _scorer = PromptScorer()
    return _scorer