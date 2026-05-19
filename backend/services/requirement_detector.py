"""
Missing Requirement Detector Service.

Analyzes user requirements to find:
- Missing technical specifications
- Undefined scope boundaries
- Absent constraints or non-functional requirements
- Missing context about users/audience
- Undefined success criteria
- Security/compliance considerations not mentioned

Generates targeted clarifying questions ranked by severity.
"""

import json
import re
import time
from typing import Any

from backend.config import settings
from backend.logger import logger
from backend.models.schemas import (
    MissingInfoSeverity,
    MissingRequirement,
    MissingRequirementRequest,
    MissingRequirementResponse,
    RequirementGapAnalysis,
)
from backend.services.llm_service import get_llm_service


# System Prompt for Gap Detection
GAP_DETECTION_SYSTEM_PROMPT = """You are an expert Requirements Engineer specializing in 
identifying gaps, ambiguities, and missing information in software and project requirements.

Your job is to thoroughly analyze requirements and find EXACTLY what information is missing 
or undefined that would be needed to create a high-quality, actionable prompt or specification.

GAP CATEGORIES TO CHECK:
1. SCOPE: What is in/out of scope? What are the boundaries?
2. TECHNICAL_STACK: What technologies, languages, frameworks should be used?
3. AUDIENCE: Who are the users? What is their technical level?
4. SCALE: How many users? What volume/size of data?
5. PERFORMANCE: Speed, latency, throughput requirements?
6. SECURITY: Authentication, authorization, data privacy needs?
7. INTEGRATION: What systems must this connect with?
8. SUCCESS_CRITERIA: How do we know when it's done/working?
9. TIMELINE: When is it needed? Any phases?
10. CONSTRAINTS: Budget, team size, existing systems, regulatory?
11. OUTPUT_FORMAT: What should the final deliverable look like?
12. EXAMPLES: Are there reference examples or anti-examples?

SEVERITY LEVELS:
- critical: Without this info, the prompt/requirement is unusable
- important: This significantly impacts the quality of output
- optional: Nice to have, improves output but not blocking

COMPLETENESS SCORING:
- 90-100: Nearly complete, ready for optimization
- 70-89: Mostly complete, a few clarifications needed
- 50-69: Partially complete, several important gaps
- 30-49: Incomplete, many critical pieces missing
- 0-29: Very vague, major gaps throughout

RULES:
- Only flag REAL gaps — don't invent problems that aren't there
- Maximum 8 missing requirements (focus on most important)
- Be specific about WHAT is missing, not just that something is vague
- Provide helpful example answers to guide the user
- Always respond with VALID JSON only

RESPONSE FORMAT (strict JSON):
{
  "completeness_score": 0-100 (float),
  "missing_requirements": [
    {
      "category": "string from GAP CATEGORIES",
      "question": "Specific question to ask the user",
      "why_needed": "Why this matters for prompt quality",
      "severity": "critical|important|optional",
      "example_answer": "Example of a good answer or null",
      "default_assumption": "What will be assumed if not answered or null"
    }
  ],
  "critical_gaps": ["Short summary of most critical gap 1", "gap 2"],
  "can_proceed": true|false,
  "recommendation": "Overall recommendation string (1-2 sentences)"
}"""


GAP_DETECTION_USER_TEMPLATE = """Analyze this requirement for missing information and gaps:

REQUIREMENT:
{requirement}

{intent_section}

{domain_section}

Find all significant gaps and generate targeted clarifying questions.
Respond with JSON only:"""


class RequirementDetector:
    """
    Service for detecting missing information in user requirements.

    Analyzes completeness and generates prioritized clarifying questions
    to help users provide the information needed for high-quality prompt generation.
    """

    def __init__(self):
        self.llm_service = get_llm_service()
        logger.info("RequirementDetector initialized")

    async def detect_gaps(
        self,
        request: MissingRequirementRequest,
    ) -> MissingRequirementResponse:
        """
        Main entry point: Detects gaps in a user requirement.

        Args:
            request: MissingRequirementRequest with requirement and optional context

        Returns:
            MissingRequirementResponse with gap analysis

        Raises:
            RuntimeError: If LLM call fails
        """
        start_time = time.time()
        logger.info(
            "Starting gap detection | req_len={length}",
            length=len(request.raw_requirement),
        )

        # ── Build context sections ────────────────────────────────
        intent_section = ""
        if request.intent_summary:
            intent_section = f"INTENT CONTEXT:\n{request.intent_summary}\n"

        domain_section = ""
        if request.domain:
            domain_section = f"DETECTED DOMAIN: {request.domain}\n"

        user_message = GAP_DETECTION_USER_TEMPLATE.format(
            requirement=request.raw_requirement,
            intent_section=intent_section,
            domain_section=domain_section,
        )

        # ── Call LLM ─────────────────────────────────────────────
        llm_result = await self.llm_service.invoke(
            system_prompt=GAP_DETECTION_SYSTEM_PROMPT,
            user_message=user_message,
        )

        # ── Parse and validate response ───────────────────────────
        gap_data = self._parse_llm_response(llm_result["content"])
        gap_analysis = self._build_gap_analysis(gap_data)

        total_time_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(
            "Gap detection complete | score={score} | gaps={count} | "
            "can_proceed={proceed} | time={time}ms",
            score=gap_analysis.completeness_score,
            count=len(gap_analysis.missing_requirements),
            proceed=gap_analysis.can_proceed,
            time=total_time_ms,
        )

        return MissingRequirementResponse(
            success=True,
            session_id=request.session_id,
            raw_requirement=request.raw_requirement,
            gap_analysis=gap_analysis,
            processing_time_ms=total_time_ms,
            tokens_used=llm_result["tokens_used"],
        )

    def _parse_llm_response(self, content: str) -> dict[str, Any]:
        """
        Parses JSON from LLM response with multiple fallback strategies.

        Args:
            content: Raw LLM response string

        Returns:
            Parsed dictionary

        Raises:
            ValueError: If no valid JSON found
        """
        # Strategy 1: Direct JSON parse
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code blocks
        json_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find JSON object in text
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            try:
                parsed = json.loads(content[start:end])
                if "completeness_score" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        logger.error(
            "Failed to parse gap detection JSON | preview={preview}",
            preview=content[:200],
        )
        raise ValueError(
            f"Could not extract valid JSON from gap detection response. "
            f"Preview: {content[:200]}"
        )

    def _build_gap_analysis(self, data: dict[str, Any]) -> RequirementGapAnalysis:
        """
        Builds validated RequirementGapAnalysis from parsed LLM data.

        Applies safe defaults for all fields.

        Args:
            data: Parsed JSON from LLM response

        Returns:
            Validated RequirementGapAnalysis object
        """
        # ── Parse missing requirements list ───────────────────────
        raw_missing = data.get("missing_requirements", [])
        missing_requirements = []

        for item in raw_missing:
            if not isinstance(item, dict):
                continue

            # Safe severity extraction
            severity_str = item.get("severity", "important").lower().strip()
            try:
                severity = MissingInfoSeverity(severity_str)
            except ValueError:
                severity = MissingInfoSeverity.IMPORTANT

            missing_requirements.append(
                MissingRequirement(
                    category=str(item.get("category", "General")),
                    question=str(item.get("question", "Please provide more details")),
                    why_needed=str(item.get("why_needed", "Improves output quality")),
                    severity=severity,
                    example_answer=item.get("example_answer"),
                    default_assumption=item.get("default_assumption"),
                )
            )

        # ── Sort by severity (critical first) ─────────────────────
        severity_order = {
            MissingInfoSeverity.CRITICAL: 0,
            MissingInfoSeverity.IMPORTANT: 1,
            MissingInfoSeverity.OPTIONAL: 2,
        }
        missing_requirements.sort(key=lambda x: severity_order[x.severity])

        # ── Safe score extraction ──────────────────────────────────
        try:
            score = float(data.get("completeness_score", 50.0))
            score = max(0.0, min(100.0, score))  # Clamp to 0-100
        except (TypeError, ValueError):
            score = 50.0

        # ── Safe critical gaps extraction ─────────────────────────
        critical_gaps = []
        raw_critical = data.get("critical_gaps", [])
        if isinstance(raw_critical, list):
            critical_gaps = [str(g) for g in raw_critical if g]

        # ── Determine if we can proceed ───────────────────────────
        # Override LLM's can_proceed if there are critical gaps
        has_critical = any(
            r.severity == MissingInfoSeverity.CRITICAL
            for r in missing_requirements
        )
        can_proceed = data.get("can_proceed", True)
        if has_critical and score < 40:
            can_proceed = False

        recommendation = str(
            data.get(
                "recommendation",
                "Please provide more details to improve prompt quality.",
            )
        )

        return RequirementGapAnalysis(
            completeness_score=score,
            missing_requirements=missing_requirements,
            critical_gaps=critical_gaps,
            can_proceed=can_proceed,
            recommendation=recommendation,
        )


# ─── Module-level singleton ────────────────────────────────────────
_requirement_detector: RequirementDetector | None = None


def get_requirement_detector() -> RequirementDetector:
    """Returns the RequirementDetector singleton."""
    global _requirement_detector
    if _requirement_detector is None:
        _requirement_detector = RequirementDetector()
    return _requirement_detector