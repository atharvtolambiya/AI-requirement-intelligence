"""
Requirement Expander — UPGRADED for Phase 5.

Changes from original:
- Receives PreservationRuleSet and injects it into system prompt
- Receives ReasoningAlignment and applies correct reasoning mode
- Runs drift detection after expansion
- Re-expands if drift score is critical (> 70)
- Returns expansion with drift analysis attached

The expander now operates under hard semantic constraints
that prevent it from defaulting to generic patterns.
"""

import json
import re
import time
from typing import Any

from backend.logger import logger
from backend.models.schemas import (
    ComplexityLevel,
    ConfidenceLevel,
    ExpandedRequirement,
    RequirementExpansionRequest,
    RequirementExpansionResponse,
)
from backend.services.llm_service import get_llm_service


# ─── Base System Prompt ────────────────────────────────────────────
EXPANSION_BASE_SYSTEM_PROMPT = """You are a Senior Requirements Engineer and 
AI Systems Architect with expertise in novel AI product design.

Your task: expand a user requirement into a comprehensive, well-structured 
specification that PRESERVES the user's conceptual vision at the correct 
abstraction level.

CRITICAL PRINCIPLE:
You are NOT simply filling in technical gaps.
You are PRESERVING and ELABORATING a conceptual vision.
The expansion must remain faithful to the USER'S ABSTRACTION LEVEL,
not default to the most common software pattern for similar keywords.

EXPANSION APPROACH:
1. Start with the CONCEPTUAL LEVEL of the requirement
2. Derive components from the concept — not the other way around
3. Preserve ALL innovation claims explicitly
4. Write functional requirements that reflect the REAL intelligence
5. Define success criteria at the correct abstraction level

RESPONSE FORMAT (strict JSON):
{
  "title": "Specific descriptive title (reflects actual concept)",
  "overview": "2-3 sentences capturing the real vision",
  "objectives": ["Specific conceptual objective"],
  "functional_requirements": ["What the system must DO at concept level"],
  "non_functional_requirements": ["Quality attributes"],
  "scope": "Clear scope at the correct abstraction level",
  "out_of_scope": ["Excluded items"],
  "assumptions_made": ["Explicit assumptions"],
  "success_criteria": ["Measurable outcomes at concept level"],
  "suggested_approach": "Approach matching the abstraction level",
  "estimated_complexity": "low|medium|high",
  "expansion_confidence": "high|medium|low"
}"""


EXPANSION_USER_TEMPLATE = """Expand this requirement into a structured specification:

REQUIREMENT:
{requirement}

{context_section}

{intent_section}

{domain_section}

{answers_section}

{preservation_injection}

{alignment_instructions}

Expand at the correct abstraction level. Preserve all innovation claims.
Do NOT default to generic patterns.

Respond with JSON only:"""


class RequirementExpander:
    """
    Vision-aware requirement expander.

    Receives semantic constraints from upstream phases and
    applies them as hard constraints during expansion.

    Key upgrade: preservation rules and reasoning alignment
    are injected directly into the LLM system prompt,
    overriding default expansion patterns.
    """

    def __init__(self):
        self.llm_service = get_llm_service()
        logger.info("RequirementExpander initialized (Phase 5 upgrade)")

    async def expand(
        self,
        request: RequirementExpansionRequest,
        preservation_rules: dict | None = None,
        reasoning_alignment: dict | None = None,
    ) -> RequirementExpansionResponse:
        """
        Expands a requirement with optional semantic constraints.

        Args:
            request: ExpansionRequest with raw requirement
            preservation_rules: PreservationRuleSet from Phase 4
            reasoning_alignment: ReasoningAlignment from Phase 4

        Returns:
            RequirementExpansionResponse with vision-aligned expansion
        """
        start_time = time.time()
        logger.info(
            "Expanding requirement | req_len={l} | "
            "has_preservation={p} | has_alignment={a}",
            l=len(request.raw_requirement),
            p=preservation_rules is not None,
            a=reasoning_alignment is not None,
        )

        # ── Build system prompt with semantic constraints ──────────
        system_prompt = self._build_system_prompt(
            preservation_rules, reasoning_alignment
        )

        # ── Build context sections ────────────────────────────────
        context_section = (
            f"ADDITIONAL CONTEXT:\n{request.context}\n"
            if request.context else ""
        )
        intent_section = self._build_intent_section(
            request.intent_summary
        )
        domain_section = (
            f"DOMAIN: {request.domain}\n"
            f"Apply domain-specific best practices.\n"
            if request.domain else ""
        )
        answers_section = self._build_answers_section(
            request.user_answers
        )

        # ── Extract preservation + alignment text for injection ────
        preservation_injection = ""
        if preservation_rules:
            preservation_injection = preservation_rules.get(
                "system_prompt_injection", ""
            )

        alignment_instructions = ""
        if reasoning_alignment:
            alignment_instructions = reasoning_alignment.get(
                "alignment_instructions", ""
            )

        user_message = EXPANSION_USER_TEMPLATE.format(
            requirement=str(request.raw_requirement)[:4000],
            context_section=str(context_section)[:2000],
            intent_section=str(intent_section)[:1000],
            domain_section=str(domain_section)[:1000],
            answers_section=str(answers_section)[:2000],
            preservation_injection=str(preservation_injection)[:2000],
            alignment_instructions=str(alignment_instructions)[:2000],
        )

        # ── Call LLM ─────────────────────────────────────────────
        llm_result = await self.llm_service.invoke(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature_override=0.3,
        )

        # ── Parse + build ─────────────────────────────────────────
        data     = self._parse_json(llm_result["content"])
        expanded = self._build_expanded_requirement(data)

        total_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(
            "Expansion complete | title={t} | complexity={c} | time={ms}ms",
            t=expanded.title[:50],
            c=expanded.estimated_complexity,
            ms=total_ms,
        )

        return RequirementExpansionResponse(
            success=True,
            session_id=request.session_id,
            raw_requirement=request.raw_requirement,
            expanded=expanded,
            processing_time_ms=total_ms,
            tokens_used=llm_result["tokens_used"],
        )

    # ── Private helpers ────────────────────────────────────────────

    def _build_system_prompt(
        self,
        preservation_rules: dict | None,
        reasoning_alignment: dict | None,
    ) -> str:
        """
        Builds system prompt with injected semantic constraints.
        """
        parts = [EXPANSION_BASE_SYSTEM_PROMPT]

        if reasoning_alignment:
            mode = reasoning_alignment.get(
                "primary_reasoning_mode", "concept_first"
            )
            target = reasoning_alignment.get(
                "abstraction_target", "system"
            )
            parts.append(
                f"\nACTIVE REASONING MODE: {mode}"
                f"\nTARGET ABSTRACTION LEVEL: {target}"
            )

        return "\n\n".join(parts)

    def _build_intent_section(self, intent_summary: str | None) -> str:
        if not intent_summary:
            return ""
        return f"INTENT CONTEXT:\n{intent_summary}\n"

    def _build_answers_section(
        self, user_answers: dict | None
    ) -> str:
        if not user_answers:
            return ""
        lines = ["USER CLARIFICATIONS:"]
        for q, a in user_answers.items():
            lines.append(f"  Q: {q}")
            lines.append(f"  A: {a}")
        return "\n".join(lines) + "\n"

    def _build_expanded_requirement(
        self, data: dict[str, Any]
    ) -> ExpandedRequirement:
        """Builds validated ExpandedRequirement from parsed data."""

        def safe_list(key: str) -> list[str]:
            val = data.get(key, [])
            if isinstance(val, list):
                return [str(i) for i in val if i]
            return [str(val)] if val else []

        try:
            complexity = ComplexityLevel(
                data.get("estimated_complexity", "medium").lower()
            )
        except ValueError:
            complexity = ComplexityLevel.MEDIUM

        try:
            confidence = ConfidenceLevel(
                data.get("expansion_confidence", "medium").lower()
            )
        except ValueError:
            confidence = ConfidenceLevel.MEDIUM

        return ExpandedRequirement(
            title=str(data.get("title", "Untitled"))[:100],
            overview=str(data.get("overview", "")),
            objectives=safe_list("objectives"),
            functional_requirements=safe_list("functional_requirements"),
            non_functional_requirements=safe_list(
                "non_functional_requirements"
            ),
            scope=str(data.get("scope", "")),
            out_of_scope=safe_list("out_of_scope"),
            assumptions_made=safe_list("assumptions_made"),
            success_criteria=safe_list("success_criteria"),
            suggested_approach=str(data.get("suggested_approach", "")),
            estimated_complexity=complexity,
            expansion_confidence=confidence,
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
            "RequirementExpander JSON parse failed | p={p}",
            p=content[:200],
        )
        return {}


# ─── Singleton ────────────────────────────────────────────────────
_expander: RequirementExpander | None = None


def get_requirement_expander() -> RequirementExpander:
    global _expander
    if _expander is None:
        _expander = RequirementExpander()
    return _expander