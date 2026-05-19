"""
Intent Analyzer Service.

Analyzes vague user requirements to extract:
- Primary and secondary goals
- Domain/category classification
- Target audience
- Constraints and boundaries
- Hidden assumptions
- Key terminology
- Complexity assessment

Uses structured JSON output prompting to ensure
consistent, parseable responses from the LLM.
"""

import json
import re
import time
from typing import Any

from backend.config import settings
from backend.logger import logger
from backend.models.schemas import (
    ComplexityLevel,
    ConfidenceLevel,
    DetectedIntent,
    DomainType,
    IntentAnalysisRequest,
    IntentAnalysisResponse,
)
from backend.services.llm_service import get_llm_service


# ─── System Prompt for Intent Analysis ────────────────────────────
INTENT_ANALYSIS_SYSTEM_PROMPT = """You are an expert Requirements Engineer and AI Prompt Analyst.

Your job is to deeply analyze user requirements — even vague, incomplete, or poorly 
written ones — and extract structured intent information.

ANALYSIS FRAMEWORK:
1. PRIMARY GOAL: What is the ONE main thing the user wants to achieve?
2. SECONDARY GOALS: What supporting objectives exist?
3. DOMAIN: What field/category does this belong to?
4. TARGET AUDIENCE: Who will use/benefit from the final output?
5. CONSTRAINTS: What limitations, restrictions, or boundaries are mentioned or implied?
6. ASSUMPTIONS: What are you assuming that wasn't explicitly stated?
7. KEYWORDS: What are the most important technical/domain terms?
8. COMPLEXITY: How complex is this requirement (low/medium/high)?
9. CONFIDENCE: How confident are you in this analysis (high/medium/low)?

DOMAIN OPTIONS (choose the BEST match):
- software_development: apps, APIs, coding, databases, web, mobile
- data_science: ML, AI, analytics, data processing, visualization
- content_creation: writing, blogs, marketing, social media, copywriting
- business_analysis: processes, workflows, strategy, reporting
- research: literature review, analysis, investigation, surveys
- education: teaching, learning, curriculum, training
- creative: art, music, storytelling, design, brainstorming
- general: doesn't fit other categories

COMPLEXITY GUIDELINES:
- low: simple, single-step, clear scope
- medium: multiple components, some technical depth
- high: multi-system, complex logic, many unknowns

CONFIDENCE GUIDELINES:
- high: requirement is clear enough to analyze confidently (>80% sure)
- medium: some ambiguity, made reasonable interpretations (50-80% sure)
- low: very vague, many assumptions made (<50% sure)

CRITICAL RULES:
- Always respond with VALID JSON only — no markdown, no explanation outside JSON
- Be specific, not generic in your analysis
- Extract real insights, not obvious observations
- If something is unclear, note it in assumptions

RESPONSE FORMAT (strict JSON):
{
  "primary_goal": "string - specific main objective",
  "secondary_goals": ["string", "string"],
  "domain": "one of the domain options above",
  "target_audience": "string or null",
  "constraints": ["string", "string"],
  "assumptions": ["string", "string"],
  "keywords": ["string", "string"],
  "complexity": "low|medium|high",
  "confidence": "high|medium|low",
  "raw_reasoning": "string - your step-by-step reasoning (2-3 sentences)"
}"""


INTENT_ANALYSIS_USER_TEMPLATE = """Analyze the following user requirement:

REQUIREMENT:
{requirement}

{context_section}

Provide your structured intent analysis as JSON:"""


class IntentAnalyzer:
    """
    Service class for analyzing user intent from raw requirements.

    Uses LangChain + LLM to extract structured intent information.
    Handles JSON parsing, validation, and fallback logic.
    """

    def __init__(self):
        self.llm_service = get_llm_service()
        logger.info("IntentAnalyzer initialized")

    async def analyze(
        self,
        request: IntentAnalysisRequest,
    ) -> IntentAnalysisResponse:
        """
        Main entry point: Analyzes intent from a raw requirement.

        Args:
            request: IntentAnalysisRequest with raw_requirement and optional context

        Returns:
            IntentAnalysisResponse with structured intent analysis

        Raises:
            RuntimeError: If LLM call fails
            ValueError: If response cannot be parsed
        """
        start_time = time.time()
        logger.info(
            "Starting intent analysis | req_len={length}",
            length=len(request.raw_requirement),
        )

        # ── Build user message ────────────────────────────────────
        context_section = ""
        if request.context:
            context_section = f"ADDITIONAL CONTEXT:\n{request.context}\n"

        user_message = INTENT_ANALYSIS_USER_TEMPLATE.format(
            requirement=request.raw_requirement,
            context_section=context_section,
        )

        # ── Call LLM ─────────────────────────────────────────────
        llm_result = await self.llm_service.invoke(
            system_prompt=INTENT_ANALYSIS_SYSTEM_PROMPT,
            user_message=user_message,
        )

        # ── Parse JSON response ───────────────────────────────────
        intent_data = self._parse_llm_response(llm_result["content"])

        # ── Build structured intent object ────────────────────────
        intent = self._build_intent(intent_data)

        # ── Calculate total processing time ───────────────────────
        total_time_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(
            "Intent analysis complete | domain={domain} | complexity={complexity} | "
            "confidence={confidence} | time={time}ms",
            domain=intent.domain,
            complexity=intent.complexity,
            confidence=intent.confidence,
            time=total_time_ms,
        )

        return IntentAnalysisResponse(
            success=True,
            session_id=request.session_id,
            raw_requirement=request.raw_requirement,
            intent=intent,
            processing_time_ms=total_time_ms,
            tokens_used=llm_result["tokens_used"],
        )

    def _parse_llm_response(self, content: str) -> dict[str, Any]:
        """
        Parses JSON from LLM response content.

        Handles cases where LLM wraps JSON in markdown code blocks.

        Args:
            content: Raw string response from LLM

        Returns:
            Parsed dictionary

        Raises:
            ValueError: If JSON cannot be extracted or parsed
        """
        # ── Try direct JSON parse first ───────────────────────────
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass

        # ── Extract JSON from markdown code blocks ────────────────
        # Handles: ```json {...} ``` or ``` {...} ```
        json_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(json_pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # ── Try to find raw JSON object in response ────────────────
        brace_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(brace_pattern, content, re.DOTALL)
        for m in matches:
            try:
                parsed = json.loads(m)
                # Verify it has expected fields
                if "primary_goal" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue

        logger.error(
            "Failed to parse LLM JSON response | content_preview={preview}",
            preview=content[:200],
        )
        raise ValueError(
            f"Could not extract valid JSON from LLM response. "
            f"Response preview: {content[:200]}"
        )

    def _build_intent(self, data: dict[str, Any]) -> DetectedIntent:
        """
        Builds a validated DetectedIntent object from parsed LLM data.

        Applies safe fallbacks for missing or invalid fields.

        Args:
            data: Parsed JSON dictionary from LLM

        Returns:
            Validated DetectedIntent object
        """
        # ── Safe domain extraction with fallback ──────────────────
        domain_str = data.get("domain", "unknown").lower().strip()
        try:
            domain = DomainType(domain_str)
        except ValueError:
            logger.warning(
                "Unknown domain '{domain}', falling back to 'general'",
                domain=domain_str,
            )
            domain = DomainType.GENERAL

        # ── Safe complexity extraction ─────────────────────────────
        complexity_str = data.get("complexity", "medium").lower().strip()
        try:
            complexity = ComplexityLevel(complexity_str)
        except ValueError:
            complexity = ComplexityLevel.MEDIUM

        # ── Safe confidence extraction ─────────────────────────────
        confidence_str = data.get("confidence", "medium").lower().strip()
        try:
            confidence = ConfidenceLevel(confidence_str)
        except ValueError:
            confidence = ConfidenceLevel.MEDIUM

        return DetectedIntent(
            primary_goal=data.get("primary_goal", "Goal not clearly identified"),
            secondary_goals=self._safe_list(data.get("secondary_goals", [])),
            domain=domain,
            target_audience=data.get("target_audience"),
            constraints=self._safe_list(data.get("constraints", [])),
            assumptions=self._safe_list(data.get("assumptions", [])),
            keywords=self._safe_list(data.get("keywords", [])),
            complexity=complexity,
            confidence=confidence,
            raw_reasoning=data.get("raw_reasoning", "No reasoning provided"),
        )

    def _safe_list(self, value: Any) -> list[str]:
        """
        Ensures a value is a list of strings.
        Handles cases where LLM returns unexpected types.
        """
        if not isinstance(value, list):
            if isinstance(value, str):
                return [value] if value else []
            return []
        return [str(item) for item in value if item]


# ─── Module-level singleton ────────────────────────────────────────
_intent_analyzer: IntentAnalyzer | None = None


def get_intent_analyzer() -> IntentAnalyzer:
    """Returns the IntentAnalyzer singleton."""
    global _intent_analyzer
    if _intent_analyzer is None:
        _intent_analyzer = IntentAnalyzer()
    return _intent_analyzer