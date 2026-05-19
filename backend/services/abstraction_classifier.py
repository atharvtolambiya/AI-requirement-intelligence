"""
Abstraction Classifier.

PHASE 3 — Stage 3 of semantic reasoning.

Detects the abstraction level gap between:
- What level the USER is thinking at
- What level the SYSTEM defaults to

This gap is the primary source of semantic drift.

Example:
  User thinks at:   SYSTEM level (reasoning architecture)
  System defaults:  FEATURE level (ML model + UI components)
  Gap:              2 levels — causes complete vision collapse

The classifier produces:
- Explicit abstraction level assignment
- Mismatch detection with specific examples
- Correction strategy
- Expansion-level instructions for downstream use
"""

import json
import re
import time
from typing import Any

from backend.logger import logger
from backend.models.schemas import (
    AbstractionClassification,
    AbstractionClassificationRequest,
    AbstractionClassificationResponse,
    AbstractionLevel,
    AbstractionMismatch,
)
from backend.services.llm_service import get_llm_service


# ─── System Prompt ─────────────────────────────────────────────────
ABSTRACTION_CLASSIFIER_SYSTEM_PROMPT = """You are a Systems Architect and Abstraction 
Level Expert specializing in detecting semantic drift caused by abstraction mismatches.

YOUR MISSION:
Identify the abstraction level gap between what the USER intended and what a standard
AI system would produce. This gap is where semantic drift originates.

ABSTRACTION LEVEL DEFINITIONS:
- task:     A single action to automate ("send an email when X happens")
- feature:  A capability within a product ("real-time search with filters")
- product:  A complete standalone product ("project management tool")
- system:   A complex intelligent system with multiple reasoning components
- paradigm: A new way of conceptualizing and solving a class of problems

TYPICAL SYSTEM DEFAULT BEHAVIOR:
Standard AI expansion systems default to PRODUCT or FEATURE level
because these are the most common patterns in training data.

When a user describes a SYSTEM or PARADIGM level idea,
the expansion collapses it to PRODUCT or FEATURE level.

This is the core problem.

ABSTRACTION GAP CALCULATION:
Map levels to numbers: task=0, feature=1, product=2, system=3, paradigm=4

abstraction_gap = user_level_number - system_default_number
Positive gap = user thinks higher than system defaults
Negative gap = user thinks lower (rare but possible)

A gap of 2+ is critical and will cause severe semantic drift.

MISMATCH DETECTION:
For each detected mismatch, provide:
- Which concept was affected
- How the user intended it (high abstraction)
- How the system would interpret it (low abstraction)
- A concrete before/after example
- How to correct it

EXPANSION LEVEL INSTRUCTIONS:
Generate specific, actionable instructions for how the requirement
expander should operate to match the user's abstraction level.

Format: "When expanding [concept], treat it as [abstraction level] 
by [specific instruction]"

RESPONSE FORMAT (strict JSON):
{
  "user_abstraction_level": "task|feature|product|system|paradigm",
  "system_default_level": "task|feature|product|system|paradigm",
  "abstraction_gap": integer (-4 to 4),
  "detected_mismatches": [
    {
      "mismatch_id": "m1",
      "intended_abstraction": "What user meant",
      "detected_abstraction": "What system would produce",
      "concept_affected": "Which concept",
      "example": "Concrete before/after example",
      "correction": "How to fix this",
      "severity": "low|medium|high|critical"
    }
  ],
  "mismatch_count": integer,
  "classification_reasoning": "How level was determined (2-3 sentences)",
  "correction_strategy": "Overall strategy to fix abstraction alignment",
  "expansion_level_instructions": [
    "Specific instruction for expander"
  ]
}"""


ABSTRACTION_CLASSIFIER_USER_TEMPLATE = """Classify the abstraction level for this input:

RAW INPUT:
{raw_input}

{vision_section}

{intent_section}

Identify the user's abstraction level, what the system would default to,
and all specific mismatches that would cause semantic drift.

Respond with JSON only:"""


class AbstractionClassifier:
    """
    Classifies abstraction levels and detects abstraction mismatches.

    The abstraction gap between user intent and system default
    is the root cause of semantic drift. This classifier:
    1. Determines the user's intended abstraction level
    2. Determines the system's default abstraction level
    3. Calculates the gap
    4. Identifies specific mismatch points
    5. Generates correction instructions for downstream stages
    """

    def __init__(self):
        self.llm_service = get_llm_service()
        logger.info("AbstractionClassifier initialized")

    async def classify(
        self,
        request: AbstractionClassificationRequest,
    ) -> AbstractionClassificationResponse:
        """
        Classifies abstraction levels and detects mismatches.

        Args:
            request: AbstractionClassificationRequest

        Returns:
            AbstractionClassificationResponse with full classification
        """
        start_time = time.time()
        logger.info(
            "Classifying abstraction | input_len={l}",
            l=len(request.raw_input),
        )

        vision_section = self._build_vision_section(
            request.vision_profile
        )
        intent_section = self._build_intent_section(
            request.semantic_intent
        )

        user_message = ABSTRACTION_CLASSIFIER_USER_TEMPLATE.format(
            raw_input=request.raw_input,
            vision_section=vision_section,
            intent_section=intent_section,
        )

        llm_result = await self.llm_service.invoke(
            system_prompt=ABSTRACTION_CLASSIFIER_SYSTEM_PROMPT,
            user_message=user_message,
            temperature_override=0.15,  # Very low for precise classification
        )

        data           = self._parse_json(llm_result["content"])
        classification = self._build_classification(
            data, request.session_id
        )

        total_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(
            "Abstraction classified | user={ul} | system={sl} | "
            "gap={gap} | mismatches={mc} | time={t}ms",
            ul=classification.user_abstraction_level,
            sl=classification.system_default_level,
            gap=classification.abstraction_gap,
            mc=classification.mismatch_count,
            t=total_ms,
        )

        return AbstractionClassificationResponse(
            success=True,
            session_id=request.session_id,
            raw_input=request.raw_input,
            classification=classification,
            processing_time_ms=total_ms,
            tokens_used=llm_result["tokens_used"],
        )

    # ── Private helpers ────────────────────────────────────────────

    def _build_vision_section(self, vp: dict | None) -> str:
        """Formats vision profile for LLM."""
        if not vp:
            return ""
        lines = ["VISION PROFILE:"]
        for key in [
            "innovation_type", "abstraction_level",
            "core_transformation", "intelligence_description"
        ]:
            val = vp.get(key, "")
            if val:
                lines.append(f"  {key}: {val}")
        return "\n".join(lines) + "\n"

    def _build_intent_section(self, si: dict | None) -> str:
        """Formats semantic intent for LLM."""
        if not si:
            return ""
        lines = ["SEMANTIC INTENT:"]
        one_line = si.get("one_line_intent", "")
        if one_line:
            lines.append(f"  One-line intent: {one_line}")
        category = si.get("conceptual_category", "")
        if category:
            lines.append(f"  Conceptual category: {category}")
        std_cat = si.get("closest_standard_category", "")
        if std_cat:
            lines.append(f"  Closest standard category: {std_cat}")
        div = si.get("category_divergence", [])
        if div:
            lines.append("  How it diverges:")
            for d in div[:2]:
                lines.append(f"    - {d}")
        return "\n".join(lines) + "\n"

    def _build_classification(
        self, data: dict[str, Any], session_id: str | None
    ) -> AbstractionClassification:
        """Builds validated AbstractionClassification."""

        # User abstraction level
        ul_str = str(
            data.get("user_abstraction_level", "product")
        ).lower()
        try:
            user_level = AbstractionLevel(ul_str)
        except ValueError:
            user_level = AbstractionLevel.PRODUCT

        # System default level
        sl_str = str(
            data.get("system_default_level", "feature")
        ).lower()
        try:
            system_level = AbstractionLevel(sl_str)
        except ValueError:
            system_level = AbstractionLevel.FEATURE

        # Gap
        try:
            gap = int(data.get("abstraction_gap", 0))
            gap = max(-4, min(4, gap))
        except (TypeError, ValueError):
            gap = 0

        # Mismatches
        mismatches: list[AbstractionMismatch] = []
        for i, m in enumerate(data.get("detected_mismatches", [])):
            if not isinstance(m, dict):
                continue
            mismatches.append(
                AbstractionMismatch(
                    mismatch_id=str(
                        m.get("mismatch_id", f"m{i+1}")
                    ),
                    intended_abstraction=str(
                        m.get("intended_abstraction", "")
                    ),
                    detected_abstraction=str(
                        m.get("detected_abstraction", "")
                    ),
                    concept_affected=str(
                        m.get("concept_affected", "")
                    ),
                    example=str(m.get("example", "")),
                    correction=str(m.get("correction", "")),
                    severity=str(
                        m.get("severity", "medium")
                    ).lower(),
                )
            )

        def safe_list(key: str) -> list[str]:
            val = data.get(key, [])
            return (
                [str(v) for v in val if v]
                if isinstance(val, list) else []
            )

        return AbstractionClassification(
            session_id=session_id,
            user_abstraction_level=user_level,
            system_default_level=system_level,
            abstraction_gap=gap,
            detected_mismatches=mismatches,
            mismatch_count=len(mismatches),
            classification_reasoning=str(
                data.get("classification_reasoning", "")
            ),
            correction_strategy=str(
                data.get("correction_strategy", "")
            ),
            expansion_level_instructions=safe_list(
                "expansion_level_instructions"
            ),
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

        logger.error(
            "AbstractionClassifier JSON parse failed | p={p}",
            p=content[:200],
        )
        return {}


# ─── Singleton ────────────────────────────────────────────────────
_classifier: AbstractionClassifier | None = None


def get_abstraction_classifier() -> AbstractionClassifier:
    global _classifier
    if _classifier is None:
        _classifier = AbstractionClassifier()
    return _classifier