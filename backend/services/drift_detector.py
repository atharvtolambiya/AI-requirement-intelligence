"""
Semantic Drift Detector.

PHASE 4 — Stage 3 of drift prevention.

Runs AFTER expansion to detect semantic drift between:
- The user's vision (anchors, intent, innovation claims)
- The actual expansion output

Detects:
- Lost concepts (anchors that disappeared)
- Diluted concepts (anchors that survived but were weakened)
- Generic pattern introduction (forbidden collapses that occurred)
- Anti-pattern violations (rules that were broken)
- Abstraction collapse (level mismatch in output)

Output: DriftAnalysis with specific warnings and corrections
that can trigger re-expansion if drift is too severe.
"""

import json
import re
import time
from typing import Any

from backend.logger import logger
from backend.models.schemas import (
    DriftAnalysis,
    DriftAnalysisRequest,
    DriftAnalysisResponse,
    DriftRisk,
    DriftWarning,
)
from backend.services.llm_service import get_llm_service


# ─── System Prompt ─────────────────────────────────────────────────
DRIFT_DETECTION_SYSTEM_PROMPT = """You are a Semantic Drift Auditor specializing in 
detecting when AI-generated expansions lose, dilute, or distort the user's original 
intent.

YOUR MISSION:
Compare an expansion against the user's original vision and identify EVERY instance 
where conceptual integrity was lost or weakened.

DRIFT TYPES TO DETECT:

1. concept_loss
   A concept from the vision is completely missing from the expansion.
   Severity: high or critical depending on importance.

2. abstraction_collapse
   The expansion operates at a lower abstraction level than the vision.
   Example: vision describes a SYSTEM, expansion describes FEATURES.

3. innovation_dilution
   An innovation claim survived but is weakened or made generic.
   Example: "causal consequence reasoning" → "risk scoring"

4. generic_pattern_replacement
   A novel concept was replaced with a standard generic pattern.
   Example: "domain-aware reasoning" → "configurable rules engine"

5. anti_pattern_violation
   The expansion violates an explicit anti-pattern rule.
   Example: rule says "do not become ML platform" but expansion did.

6. anchor_violation
   A semantic anchor's preserved_meaning is not honored.

DRIFT SCORING (0-100):
- 0-20:   No significant drift
- 21-40:  Minor drift, acceptable
- 41-60:  Moderate drift, correction recommended
- 61-80:  Severe drift, correction required
- 81-100: Critical drift, full re-expansion required

CAN_PROCEED DECISION:
- drift_score < 50 AND no critical warnings → can_proceed: true
- drift_score >= 50 OR any critical warning → can_proceed: false

EVIDENCE REQUIREMENT:
For each drift warning, you MUST quote the specific text from the
expansion that demonstrates the drift. Do not generalize.

CRITICAL: Respond with VALID JSON only.

RESPONSE FORMAT:
{
  "drift_score": 0-100,
  "drift_risk": "low|medium|high|critical",
  "can_proceed": true|false,
  "preserved_concepts": ["concept that survived"],
  "lost_concepts": ["concept that was lost"],
  "diluted_concepts": [
    {"concept": "name", "how_diluted": "explanation"}
  ],
  "introduced_generic_patterns": ["generic pattern that appeared"],
  "warnings": [
    {
      "warning_id": "w1",
      "drift_type": "concept_loss|abstraction_collapse|...",
      "severity": "low|medium|high|critical",
      "original_concept": "What was in vision",
      "drifted_to": "What it became in expansion",
      "location": "Where in expansion (functional_requirements, etc.)",
      "evidence": "Direct quote from expansion showing drift",
      "correction_required": "Specific fix needed",
      "blocked_anti_pattern": "Anti-pattern violated or null"
    }
  ],
  "violated_anti_patterns": ["specific anti-pattern violated"],
  "violated_anchors": ["anchor concept not preserved"],
  "overall_assessment": "Plain-language drift assessment",
  "correction_strategy": "Strategy to fix detected drift"
}"""


DRIFT_DETECTION_USER_TEMPLATE = """Audit this expansion for semantic drift:

ORIGINAL USER INPUT:
{raw_input}

USER'S VISION:
{vision_section}

SEMANTIC INTENT:
{intent_section}

INNOVATION CLAIMS:
{innovation_section}

EXPANSION OUTPUT TO AUDIT:
---
{expansion_text}
---

Identify EVERY concept that was lost, diluted, or replaced with generic patterns.
Quote specific text as evidence for each warning.

Respond with JSON only:"""


class DriftDetector:
    """
    Detects semantic drift in expansion output.

    Compares expansion against vision profile, semantic intent,
    and innovation profile to identify concept loss, dilution,
    abstraction collapse, and anti-pattern violations.

    Returns a complete DriftAnalysis with specific warnings
    and actionable corrections.
    """

    def __init__(self):
        self.llm_service = get_llm_service()
        logger.info("DriftDetector initialized")

    async def detect(
        self,
        request: DriftAnalysisRequest,
    ) -> DriftAnalysisResponse:
        """
        Detects drift between vision and expansion.

        Args:
            request: DriftAnalysisRequest with vision + expansion

        Returns:
            DriftAnalysisResponse with complete drift analysis
        """
        start_time = time.time()
        logger.info(
            "Detecting drift | expansion_len={l}",
            l=len(request.expansion_text),
        )

        vision_section     = self._format_vision(request.vision_profile)
        intent_section     = self._format_intent(request.semantic_intent)
        innovation_section = self._format_innovation(request.innovation_profile)

        # Truncate very long expansion to fit context window
        expansion = request.expansion_text
        if len(expansion) > 6000:
            expansion = expansion[:6000] + "\n... [truncated]"

        user_message = DRIFT_DETECTION_USER_TEMPLATE.format(
            raw_input=str(request.raw_input)[:4000],
            vision_section=str(vision_section)[:2000],
            intent_section=str(intent_section)[:1500],
            innovation_section=str(innovation_section)[:1500],
            expansion_text=expansion,
        )

        llm_result = await self.llm_service.invoke(
            system_prompt=DRIFT_DETECTION_SYSTEM_PROMPT,
            user_message=user_message,
            temperature_override=0.15,  # Very low for consistent detection
        )

        data     = self._parse_json(llm_result["content"])
        analysis = self._build_analysis(data, request.session_id)

        total_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(
            "Drift detection complete | score={s} | risk={r} | "
            "can_proceed={p} | warnings={w} | critical={c} | time={t}ms",
            s=analysis.drift_score,
            r=analysis.drift_risk,
            p=analysis.can_proceed,
            w=analysis.total_warnings_count,
            c=analysis.critical_warnings_count,
            t=total_ms,
        )

        return DriftAnalysisResponse(
            success=True,
            session_id=request.session_id,
            drift_analysis=analysis,
            processing_time_ms=total_ms,
            tokens_used=llm_result["tokens_used"],
        )

    # ── Private helpers ────────────────────────────────────────────

    def _format_vision(self, vision: dict) -> str:
        """Formats vision profile for LLM context."""
        if not vision:
            return "(no vision profile provided)"

        lines = []

        primary_user = vision.get("primary_user", "")
        if primary_user:
            lines.append(f"Primary User: {primary_user}")

        core_transform = vision.get("core_transformation", "")
        if core_transform:
            lines.append(f"Core Transformation: {core_transform}")

        intelligence = vision.get("intelligence_description", "")
        if intelligence:
            lines.append(f"Intelligence: {intelligence}")

        itype = vision.get("innovation_type", "")
        if itype:
            lines.append(f"Innovation Type: {itype}")

        abstraction = vision.get("abstraction_level", "")
        if abstraction:
            lines.append(f"Abstraction Level: {abstraction}")

        # Critical: include semantic anchors
        anchors = vision.get("semantic_anchors", [])
        if anchors:
            lines.append("\nSEMANTIC ANCHORS (must be preserved):")
            for anchor in anchors[:8]:
                if not isinstance(anchor, dict):
                    continue
                concept = anchor.get("concept", "")
                meaning = anchor.get("preserved_meaning", "")
                anti = anchor.get("anti_patterns", [])
                lines.append(f"  - {concept}: {meaning}")
                if anti:
                    lines.append(
                        f"    Anti-patterns: {'; '.join(anti[:2])}"
                    )

        # Dominant collapse pattern
        collapse = vision.get("dominant_collapse_pattern", "")
        if collapse:
            lines.append(f"\nMUST NOT collapse into: {collapse}")

        return "\n".join(lines)

    def _format_intent(self, intent: dict | None) -> str:
        """Formats semantic intent for LLM context."""
        if not intent:
            return "(no semantic intent provided)"

        lines = []
        one_line = intent.get("one_line_intent", "")
        if one_line:
            lines.append(f"Real Intent: {one_line}")

        category = intent.get("conceptual_category", "")
        if category:
            lines.append(f"Conceptual Category: {category}")

        # Concepts with high collapse risk
        concept_map = intent.get("concept_map", [])
        at_risk = [
            c for c in concept_map
            if isinstance(c, dict)
            and c.get("collapse_risk") in ("high", "critical")
        ]
        if at_risk:
            lines.append("HIGH-RISK CONCEPTS (check carefully for drift):")
            for c in at_risk[:6]:
                name = c.get("name", "")
                generic = c.get("generic_equivalent", "")
                lines.append(
                    f"  - {name} (could collapse to: {generic})"
                )

        return "\n".join(lines)

    def _format_innovation(self, innovation: dict | None) -> str:
        """Formats innovation profile for LLM context."""
        if not innovation:
            return "(no innovation profile provided)"

        lines = []
        tier = innovation.get("innovation_tier", "")
        if tier:
            lines.append(f"Innovation Tier: {tier}")

        # At-risk concepts
        at_risk = innovation.get("at_risk_concepts", [])
        if at_risk:
            lines.append("AT-RISK CONCEPTS:")
            for c in at_risk[:6]:
                lines.append(f"  - {c}")

        # Anti-patterns
        anti_patterns = innovation.get("anti_patterns", [])
        if anti_patterns:
            lines.append("ANTI-PATTERNS (check expansion for violations):")
            for ap in anti_patterns[:6]:
                lines.append(f"  - {ap}")

        # Innovation claims
        claims = innovation.get("innovation_claims", [])
        if claims:
            lines.append("KEY INNOVATION CLAIMS:")
            for claim in claims[:5]:
                if not isinstance(claim, dict):
                    continue
                lines.append(f"  - {claim.get('claim', '')}")

        return "\n".join(lines)

    def _build_analysis(
        self, data: dict[str, Any], session_id: str | None
    ) -> DriftAnalysis:
        """Builds validated DriftAnalysis from parsed data."""

        # Drift score
        try:
            score = float(data.get("drift_score", 50.0))
            score = max(0.0, min(100.0, score))
        except (TypeError, ValueError):
            score = 50.0

        # Drift risk
        risk_str = str(data.get("drift_risk", "medium")).lower()
        try:
            drift_risk = DriftRisk(risk_str)
        except ValueError:
            drift_risk = DriftRisk.MEDIUM

        # Build warnings
        warnings: list[DriftWarning] = []
        for i, w in enumerate(data.get("warnings", [])):
            if not isinstance(w, dict):
                continue
            warnings.append(
                DriftWarning(
                    warning_id=str(w.get("warning_id", f"w{i+1}")),
                    drift_type=str(
                        w.get("drift_type", "concept_loss")
                    ).lower(),
                    severity=str(w.get("severity", "medium")).lower(),
                    original_concept=str(w.get("original_concept", "")),
                    drifted_to=str(w.get("drifted_to", "")),
                    location=str(w.get("location", "expansion")),
                    evidence=str(w.get("evidence", ""))[:500],
                    correction_required=str(
                        w.get("correction_required", "")
                    ),
                    blocked_anti_pattern=w.get("blocked_anti_pattern"),
                )
            )

        # Sort warnings by severity (critical first)
        severity_order = {
            "critical": 0, "high": 1, "medium": 2, "low": 3,
        }
        warnings.sort(
            key=lambda w: severity_order.get(w.severity, 4)
        )

        # Count critical warnings
        critical_count = sum(
            1 for w in warnings if w.severity == "critical"
        )

        # Diluted concepts
        diluted: list[dict[str, str]] = []
        for d in data.get("diluted_concepts", []):
            if isinstance(d, dict):
                diluted.append({
                    "concept": str(d.get("concept", "")),
                    "how_diluted": str(d.get("how_diluted", "")),
                })

        def safe_list(key: str) -> list[str]:
            val = data.get(key, [])
            return (
                [str(v) for v in val if v]
                if isinstance(val, list) else []
            )

        # Determine can_proceed (override LLM if needed)
        can_proceed = bool(data.get("can_proceed", True))
        if score >= 50 or critical_count > 0:
            can_proceed = False

        return DriftAnalysis(
            session_id=session_id,
            drift_score=score,
            drift_risk=drift_risk,
            can_proceed=can_proceed,
            preserved_concepts=safe_list("preserved_concepts"),
            lost_concepts=safe_list("lost_concepts"),
            diluted_concepts=diluted,
            introduced_generic_patterns=safe_list(
                "introduced_generic_patterns"
            ),
            warnings=warnings,
            critical_warnings_count=critical_count,
            total_warnings_count=len(warnings),
            violated_anti_patterns=safe_list("violated_anti_patterns"),
            violated_anchors=safe_list("violated_anchors"),
            overall_assessment=str(
                data.get("overall_assessment", "")
            ),
            correction_strategy=str(
                data.get("correction_strategy", "")
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
            "DriftDetector JSON parse failed | p={p}",
            p=content[:200],
        )
        return {}


# ─── Singleton ────────────────────────────────────────────────────
_drift_detector: DriftDetector | None = None


def get_drift_detector() -> DriftDetector:
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = DriftDetector()
    return _drift_detector