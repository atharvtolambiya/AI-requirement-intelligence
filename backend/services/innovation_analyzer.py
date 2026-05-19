"""
Innovation Analyzer.

PHASE 3 — Stage 2 of semantic reasoning.

Identifies what is genuinely novel about a product idea,
classifies its innovation tier, and generates explicit
preservation rules to protect innovation claims during expansion.

The core problem this solves:
  Standard expansion treats ALL ideas as if they are
  incremental improvements on known patterns.
  Novel and transformative ideas need DIFFERENT expansion
  strategies that protect their core innovation claims.
"""

import json
import re
import time
from typing import Any

from backend.logger import logger
from backend.models.schemas import (
    InnovationAnalysisRequest,
    InnovationAnalysisResponse,
    InnovationClaim,
    InnovationProfile,
    InnovationTier,
)
from backend.services.llm_service import get_llm_service


# ─── System Prompt ─────────────────────────────────────────────────
INNOVATION_ANALYSIS_SYSTEM_PROMPT = """You are an Innovation Analyst and AI Systems 
Architect specializing in evaluating the novelty and conceptual integrity of AI products.

YOUR MISSION:
Identify what is genuinely innovative about a product idea, classify its innovation
tier, and generate explicit rules to protect those innovations during requirement
expansion and prompt generation.

INNOVATION TIER DEFINITIONS:
- incremental:    Improves existing patterns (better UI, faster pipeline, more features)
- novel:          New combination of existing concepts in a meaningful way
- transformative: Fundamentally new approach to solving a class of problems

INNOVATION CLAIM TYPES:
- reasoning_advancement:   The system reasons in ways existing systems don't
- ux_innovation:           Fundamentally different user interaction model
- domain_synthesis:        Combines domain knowledge in unprecedented ways
- intelligence_depth:      Understands context at a deeper level than existing tools
- workflow_transformation:  Changes how an entire workflow is done
- decision_support:         New model of human-AI collaboration in decisions

AT-RISK CONCEPT IDENTIFICATION:
Concepts are at risk when they:
1. Have no direct equivalent in standard software patterns
2. Require reasoning about WHY not just WHAT
3. Involve causal or consequence-based thinking
4. Depend on domain-specific judgment that is hard to formalize
5. Represent a new MODEL of human-AI interaction

ANTI-PATTERN GENERATION:
For each at-risk concept, generate explicit anti-patterns:
Format: "Do NOT [generic pattern] — instead [specific requirement]"
Examples:
  "Do NOT reduce consequence reasoning to a prediction score — 
   instead, the system must explain the causal chain of consequences"
  "Do NOT implement as a recommendation list — 
   instead, the system must model human judgment augmentation"

PRESERVATION RULES:
Positive counterparts to anti-patterns:
Format: "ALWAYS [specific requirement]"
Examples:
  "ALWAYS include domain constraint reasoning in requirements"
  "ALWAYS distinguish between automation and augmentation in design"

INNOVATION SCORING (0-100):
0-30:   Incremental improvement, similar to existing tools
31-60:  Novel combination, meaningfully different approach
61-80:  Transformative in specific domain or use case
81-100: Paradigm-shifting, genuinely new class of system

RESPONSE FORMAT (strict JSON):
{
  "innovation_tier": "incremental|novel|transformative",
  "tier_reasoning": "Why this tier was assigned (2-3 sentences)",
  "innovation_claims": [
    {
      "claim_id": "claim_1",
      "claim": "Specific innovation claim",
      "claim_type": "reasoning_advancement|ux_innovation|...",
      "is_verifiable": true,
      "at_risk": true,
      "risk_reason": "Why this is at risk of being lost",
      "preservation_instruction": "How to ensure this survives"
    }
  ],
  "at_risk_concepts": ["concept at risk 1", "concept at risk 2"],
  "safe_concepts": ["safe concept 1"],
  "anti_patterns": [
    "Do NOT [X] — instead [Y]"
  ],
  "preservation_rules": [
    "ALWAYS [specific requirement]"
  ],
  "compared_to_existing": "How this compares to state-of-the-art",
  "innovation_score": 0-100
}"""


INNOVATION_ANALYSIS_USER_TEMPLATE = """Analyze the innovation in this product idea:

RAW INPUT:
{raw_input}

{vision_section}

{intent_section}

Identify all innovation claims, assess which are at risk of collapse,
and generate explicit anti-patterns and preservation rules.

Respond with JSON only:"""


class InnovationAnalyzer:
    """
    Analyzes innovation tier and claims in a product idea.

    Produces:
    - InnovationTier classification with reasoning
    - List of specific innovation claims
    - At-risk concept identification
    - Anti-patterns to prevent collapse
    - Preservation rules for downstream stages
    """

    def __init__(self):
        self.llm_service = get_llm_service()
        logger.info("InnovationAnalyzer initialized")

    async def analyze(
        self,
        request: InnovationAnalysisRequest,
    ) -> InnovationAnalysisResponse:
        """
        Analyzes innovation in a product idea.

        Args:
            request: InnovationAnalysisRequest with input + optional context

        Returns:
            InnovationAnalysisResponse with complete InnovationProfile
        """
        start_time = time.time()
        logger.info(
            "Analyzing innovation | input_len={l} | "
            "has_vision={v} | has_intent={i}",
            l=len(request.raw_input),
            v=request.vision_profile is not None,
            i=request.semantic_intent is not None,
        )

        vision_section = self._build_context_section(
            request.vision_profile, "VISION PROFILE"
        )
        intent_section = self._build_intent_section(
            request.semantic_intent
        )

        user_message = INNOVATION_ANALYSIS_USER_TEMPLATE.format(
            raw_input=request.raw_input,
            vision_section=vision_section,
            intent_section=intent_section,
        )

        llm_result = await self.llm_service.invoke(
            system_prompt=INNOVATION_ANALYSIS_SYSTEM_PROMPT,
            user_message=user_message,
            temperature_override=0.25,
        )

        data    = self._parse_json(llm_result["content"])
        profile = self._build_innovation_profile(
            data, request.session_id
        )

        total_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(
            "Innovation analyzed | tier={tier} | score={score} | "
            "claims={claims} | at_risk={risk} | time={t}ms",
            tier=profile.innovation_tier,
            score=profile.innovation_score,
            claims=len(profile.innovation_claims),
            risk=len(profile.at_risk_concepts),
            t=total_ms,
        )

        return InnovationAnalysisResponse(
            success=True,
            session_id=request.session_id,
            raw_input=request.raw_input,
            innovation_profile=profile,
            processing_time_ms=total_ms,
            tokens_used=llm_result["tokens_used"],
        )

    # ── Private helpers ────────────────────────────────────────────

    def _build_context_section(
        self, data: dict | None, label: str
    ) -> str:
        """Formats a context dict as a labeled section."""
        if not data:
            return ""
        lines = [f"{label}:"]
        for key, val in list(data.items())[:8]:
            if isinstance(val, str) and val:
                lines.append(f"  {key}: {val[:150]}")
            elif isinstance(val, list) and val:
                lines.append(
                    f"  {key}: {', '.join(str(v) for v in val[:3])}"
                )
        return "\n".join(lines) + "\n"

    def _build_intent_section(
        self, semantic_intent: dict | None
    ) -> str:
        """Formats semantic intent for LLM context."""
        if not semantic_intent:
            return ""

        lines = ["SEMANTIC INTENT:"]
        one_line = semantic_intent.get("one_line_intent", "")
        if one_line:
            lines.append(f"  Real Intent: {one_line}")

        category = semantic_intent.get("conceptual_category", "")
        if category:
            lines.append(f"  Category: {category}")

        diff_vec = semantic_intent.get("differentiation_vector", [])
        if diff_vec:
            lines.append("  Key Differentiators:")
            for d in diff_vec[:3]:
                lines.append(f"    - {d}")

        # High/critical collapse risk concepts
        concept_map = semantic_intent.get("concept_map", [])
        at_risk = [
            c for c in concept_map
            if c.get("collapse_risk") in ("high", "critical")
        ]
        if at_risk:
            lines.append("  HIGH COLLAPSE RISK CONCEPTS:")
            for c in at_risk[:4]:
                ge = c.get("generic_equivalent", "generic pattern")
                lines.append(
                    f"    - {c.get('name','?')} "
                    f"→ would collapse to: {ge}"
                )

        return "\n".join(lines) + "\n"

    def _build_innovation_profile(
        self, data: dict[str, Any], session_id: str | None
    ) -> InnovationProfile:
        """Builds validated InnovationProfile from parsed data."""

        # Innovation tier
        tier_str = str(
            data.get("innovation_tier", "incremental")
        ).lower()
        try:
            tier = InnovationTier(tier_str)
        except ValueError:
            tier = InnovationTier.INCREMENTAL

        # Innovation claims
        claims: list[InnovationClaim] = []
        for i, c in enumerate(data.get("innovation_claims", [])):
            if not isinstance(c, dict):
                continue
            claims.append(
                InnovationClaim(
                    claim_id=str(c.get("claim_id", f"claim_{i+1}")),
                    claim=str(c.get("claim", "")),
                    claim_type=str(
                        c.get("claim_type", "reasoning_advancement")
                    ),
                    is_verifiable=bool(c.get("is_verifiable", True)),
                    at_risk=bool(c.get("at_risk", False)),
                    risk_reason=c.get("risk_reason"),
                    preservation_instruction=str(
                        c.get("preservation_instruction", "")
                    ),
                )
            )

        # Safe score
        try:
            score = float(data.get("innovation_score", 30.0))
            score = max(0.0, min(100.0, score))
        except (TypeError, ValueError):
            score = 30.0

        def safe_list(key: str) -> list[str]:
            val = data.get(key, [])
            return (
                [str(v) for v in val if v]
                if isinstance(val, list) else []
            )

        return InnovationProfile(
            session_id=session_id,
            innovation_tier=tier,
            tier_reasoning=str(data.get("tier_reasoning", "")),
            innovation_claims=claims,
            at_risk_concepts=safe_list("at_risk_concepts"),
            safe_concepts=safe_list("safe_concepts"),
            anti_patterns=safe_list("anti_patterns"),
            preservation_rules=safe_list("preservation_rules"),
            compared_to_existing=str(
                data.get("compared_to_existing", "")
            ),
            innovation_score=score,
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
            "InnovationAnalyzer JSON parse failed | p={p}",
            p=content[:200],
        )
        return {}


# ─── Singleton ────────────────────────────────────────────────────
_innovation_analyzer: InnovationAnalyzer | None = None


def get_innovation_analyzer() -> InnovationAnalyzer:
    global _innovation_analyzer
    if _innovation_analyzer is None:
        _innovation_analyzer = InnovationAnalyzer()
    return _innovation_analyzer