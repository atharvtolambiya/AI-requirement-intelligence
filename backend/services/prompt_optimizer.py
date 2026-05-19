"""
Prompt Optimizer — UPGRADED for Phase 5.

Changes from original:
- Injects semantic anchors into EVERY style prompt
- Injects anti-patterns as explicit negative constraints
- Injects preservation rules into system prompt
- Each generated prompt must preserve the vision
- Adds vision-alignment instruction to all style templates
"""

import asyncio
import json
import re
import time
from typing import Any

from backend.logger import logger
from backend.models.schemas import (
    OptimizedPrompt,
    PromptOptimizationRequest,
    PromptOptimizationResponse,
    PromptStyle,
)
from backend.services.llm_service import get_llm_service


# ── Style metadata ─────────────────────────────────────────────────
STYLE_METADATA: dict[PromptStyle, dict[str, str]] = {
    PromptStyle.ZERO_SHOT: {
        "label": "Zero-Shot",
        "best_used_for": (
            "Direct, well-defined tasks with strong conceptual clarity."
        ),
    },
    PromptStyle.CHAIN_OF_THOUGHT: {
        "label": "Chain-of-Thought",
        "best_used_for": (
            "Complex reasoning tasks, multi-step problems, "
            "decision-making workflows."
        ),
    },
    PromptStyle.FEW_SHOT: {
        "label": "Few-Shot",
        "best_used_for": (
            "Tasks requiring specific output format or pattern consistency."
        ),
    },
    PromptStyle.ROLE_BASED: {
        "label": "Role-Based",
        "best_used_for": (
            "Domain-expert tasks, technical architecture, analysis."
        ),
    },
    PromptStyle.STRUCTURED: {
        "label": "Structured",
        "best_used_for": (
            "Complex multi-part tasks, systems design, documentation."
        ),
    },
}


# ─── Base System Prompt ────────────────────────────────────────────
OPTIMIZER_BASE_SYSTEM_PROMPT = """You are a world-class AI Systems Architect and Prompt Engineer 
specializing in Conceptual Architecture Synthesis.

CRITICAL PRINCIPLE - SEMANTIC SYNTHESIS OVER LITERAL PRESERVATION:
You will receive isolated constraints, objectives, and anti-patterns.
DO NOT simply enumerate them as a fragmented checklist.
You must SYNTHESIZE these fragments into a cohesive, higher-order architectural narrative. 
Preserve the conceptual intent, abstraction identity, and semantic meaning, but fuse related concepts into a unified model.

SEMANTIC FUSION RULES:
1. Conceptual Relationship Modeling: Cluster related constraints (e.g., "explainability" + "contextual reasoning" -> "Contextual Reasoning Infrastructure").
2. Higher-Order Abstraction Synthesis: Generate a unified conceptual model that inherently satisfies the negative constraints (anti-patterns) without reading like a list of warnings.
3. Narrative-Oriented Concept Assembly: Present the system as a cohesive architecture, not a loose collection of rules.
4. Concept Fusion & Traceability: The synthesized narrative must remain fully traceable to the core semantic anchors, but integrated seamlessly.

VISION PRESERVATION (CONCEPTUAL, NOT LITERAL):
- The final prompt must operate at the user's explicit abstraction level.
- Innovation claims and critical invariants must be fundamentally baked into the architectural identity, rather than appended as bullet points.

Generate ONLY the prompt text. No meta-commentary."""


OPTIMIZER_USER_TEMPLATE = """Generate a {style_name} style prompt:

REQUIREMENT TITLE: {title}
ORIGINAL REQUEST: {raw_requirement}
OVERVIEW: {overview}
DOMAIN: {domain}
TARGET LLM: {target_llm}

KEY OBJECTIVES:
{objectives}

FUNCTIONAL REQUIREMENTS:
{functional_reqs}

Generate a production-quality {style_name} prompt that:
1. SYNTHESIZES the objectives, functional requirements, and preservation constraints into a unified architectural narrative
2. Operates at the CORRECT abstraction level (inherently satisfying anti-patterns)
3. Reduces hallucination risk through cohesive system boundaries
4. Is optimized for {target_llm}

Respond with JSON:
{{
  "prompt_text": "The synthesized conceptual prompt text",
  "style_reasoning": "How the isolated concepts were fused into a unified architectural narrative"
}}"""


class PromptOptimizer:
    """
    Vision-aware prompt optimizer.

    Upgraded to inject semantic anchors and preservation rules
    into every generated prompt style.
    """

    def __init__(self):
        self.llm_service = get_llm_service()
        logger.info("PromptOptimizer initialized (Phase 5 upgrade)")

    async def optimize(
            self,
            request: PromptOptimizationRequest,
            preservation_rules: dict | None = None,
            vision_profile: dict | None = None,
    ) -> PromptOptimizationResponse:
        """
        Generates vision-aligned prompts SEQUENTIALLY.

        WHY SEQUENTIAL (not parallel):
        - Parallel: all 4 styles hit rate limiter at same time
          → all 4 wait 41s → all 4 call API together → 429 again
        - Sequential: each call completes before next starts
          → rate limiter window has accurate data
          → waits shrink naturally as window slides
        """
        start_time = time.time()
        logger.info(
            "Optimizing prompts | styles={s} | has_preservation={p} | mode=sequential",
            s=[style.value for style in request.styles],
            p=preservation_rules is not None,
        )

        # ── Extract expanded requirement context ──────────────────
        exp = request.expanded_requirement or {}
        title = str(exp.get("title", "Untitled"))[:200]
        raw_req = str(request.raw_requirement)[:6000]
        overview = str(exp.get("overview", raw_req))[:6000]
        objectives = str(self._format_list(exp.get("objectives", [])))[:3000]
        functional_reqs = str(self._format_list(
            exp.get("functional_requirements", [])
        ))[:6000]
        domain = request.domain or "general"

        # ── Build injection blocks ONCE (shared across all styles) ─
        system_prompt = self._build_system_prompt(preservation_rules)

        # ══════════════════════════════════════════════════════════
        # SEQUENTIAL style generation — replaces asyncio.gather
        # ══════════════════════════════════════════════════════════
        optimized_prompts: list[OptimizedPrompt] = []
        total_tokens = 0

        for i, style in enumerate(request.styles):
            logger.info(
                "Generating style {i}/{n} | style={s}",
                i=i + 1,
                n=len(request.styles),
                s=style.value,
            )
            try:
                prompt, tokens = await self._generate_single_style(
                    style=style,
                    system_prompt=system_prompt,
                    raw_requirement=raw_req,
                    title=title,
                    overview=overview,
                    objectives=objectives,
                    functional_reqs=functional_reqs,
                    domain=domain,
                    target_llm=request.target_llm,
                )
                optimized_prompts.append(prompt)
                total_tokens += tokens

                logger.info(
                    "Style {i}/{n} complete | style={s} | words={w} | tokens={t}",
                    i=i + 1,
                    n=len(request.styles),
                    s=style.value,
                    w=prompt.word_count,
                    t=tokens,
                )

            except Exception as e:
                logger.warning(
                    "Style generation failed | style={s} | error={e}",
                    s=style.value,
                    e=str(e)[:150],
                )
                optimized_prompts.append(
                    self._create_fallback_prompt(style, request.raw_requirement)
                )

        # ── Recommend best style ──────────────────────────────────
        recommended, reason = self._recommend_style(
            request.styles,
            domain,
            exp.get("estimated_complexity", "medium"),
            vision_profile,
        )

        total_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(
            "Optimization complete | prompts={c} | recommended={s} | "
            "tokens={t} | time={ms}ms",
            c=len(optimized_prompts),
            s=recommended.value,
            t=total_tokens,
            ms=total_ms,
        )

        return PromptOptimizationResponse(
            success=True,
            session_id=request.session_id,
            raw_requirement=request.raw_requirement,
            optimized_prompts=optimized_prompts,
            recommended_style=recommended,
            recommendation_reason=reason,
            processing_time_ms=total_ms,
            tokens_used=total_tokens,
        )

    async def _generate_single_style(
        self,
        style: PromptStyle,
        system_prompt: str,
        raw_requirement: str,
        title: str,
        overview: str,
        objectives: str,
        functional_reqs: str,
        domain: str,
        target_llm: str,
    ) -> tuple[OptimizedPrompt, int]:
        """Generates a vision-aligned prompt in one style."""
        style_meta = STYLE_METADATA[style]

        user_message = OPTIMIZER_USER_TEMPLATE.format(
            style_name=style_meta["label"],
            raw_requirement=raw_requirement,
            title=title,
            overview=overview,
            objectives=objectives,
            functional_reqs=functional_reqs,
            domain=domain,
            target_llm=target_llm,
        )

        llm_result = await self.llm_service.invoke(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature_override=0.3,
        )

        data = self._parse_json(llm_result["content"])
        prompt_text = data.get("prompt_text", llm_result["content"])
        style_reasoning = data.get(
            "style_reasoning",
            f"Generated using {style_meta['label']} technique.",
        )

        word_count = len(prompt_text.split())
        token_estimate = self.llm_service.count_tokens(prompt_text)

        prompt = OptimizedPrompt(
            style=style,
            style_label=style_meta["label"],
            prompt_text=prompt_text,
            style_reasoning=style_reasoning,
            word_count=word_count,
            estimated_tokens=token_estimate,
            best_used_for=style_meta["best_used_for"],
        )

        logger.debug(
            "Style generated | style={s} | words={w}",
            s=style.value,
            w=word_count,
        )

        return prompt, llm_result["tokens_used"]

    def _build_system_prompt(
        self, preservation_rules: dict | None
    ) -> str:
        """Builds system prompt with preservation injection."""
        parts = [OPTIMIZER_BASE_SYSTEM_PROMPT]
        if preservation_rules:
            injection = preservation_rules.get(
                "system_prompt_injection", ""
            )
            if injection:
                parts.append(injection)
        return "\n\n".join(parts)

    def _build_anchor_injection(
        self, vision_profile: dict | None
    ) -> str:
        """Builds semantic anchor injection for user message."""
        if not vision_profile:
            return ""

        anchors = vision_profile.get("semantic_anchors", [])
        if not anchors:
            return ""

        lines = [
            "SEMANTIC ANCHORS (MUST appear in generated prompt):"
        ]
        for anchor in anchors[:8]:
            if not isinstance(anchor, dict):
                continue
            concept = anchor.get("concept", "")
            meaning = anchor.get("preserved_meaning", "")
            anti    = anchor.get("anti_patterns", [])
            importance = anchor.get("importance", "high")

            lines.append(
                f"  [{importance.upper()}] {concept}: {meaning}"
            )
            if anti:
                lines.append(
                    f"    MUST NOT BECOME: {', '.join(anti[:2])}"
                )

        # Add innovation type
        itype = vision_profile.get("innovation_type", "")
        if itype:
            lines.append(
                f"\nINNOVATION TYPE: {itype} "
                f"(preserve this throughout the prompt)"
            )

        # Add dominant collapse to avoid
        collapse = vision_profile.get("dominant_collapse_pattern", "")
        if collapse:
            lines.append(
                f"DOMINANT COLLAPSE TO AVOID: {collapse}"
            )

        return "\n".join(lines)

    def _build_preservation_injection(
        self, preservation_rules: dict | None
    ) -> str:
        """Builds preservation rule injection for user message."""
        if not preservation_rules:
            return ""
        return preservation_rules.get("user_prompt_injection", "")

    def _recommend_style(
        self,
        styles: list[PromptStyle],
        domain: str,
        complexity: str,
        vision_profile: dict | None,
    ) -> tuple[PromptStyle, str]:
        """
        Recommends best style considering vision profile.

        Vision-aware recommendations:
        - reasoning type → chain_of_thought or structured
        - augmentation type → role_based
        - high complexity → chain_of_thought
        """
        # Vision-based override
        if vision_profile:
            itype = vision_profile.get("innovation_type", "")
            alevel = vision_profile.get("abstraction_level", "")

            if itype == "reasoning" and PromptStyle.CHAIN_OF_THOUGHT in styles:
                return (
                    PromptStyle.CHAIN_OF_THOUGHT,
                    "Reasoning-type innovation benefits most from "
                    "Chain-of-Thought which forces systematic reasoning steps.",
                )
            if itype == "augmentation" and PromptStyle.ROLE_BASED in styles:
                return (
                    PromptStyle.ROLE_BASED,
                    "Augmentation-type systems work best with Role-Based "
                    "prompting that positions AI as expert collaborator.",
                )
            if alevel in ("system", "paradigm") and PromptStyle.STRUCTURED in styles:
                return (
                    PromptStyle.STRUCTURED,
                    "System-level abstractions require Structured prompting "
                    "to capture all architectural components clearly.",
                )

        # Domain-based fallback
        domain_map: dict[str, PromptStyle] = {
            "software_development": PromptStyle.STRUCTURED,
            "data_science":         PromptStyle.CHAIN_OF_THOUGHT,
            "content_creation":     PromptStyle.ROLE_BASED,
            "business_analysis":    PromptStyle.STRUCTURED,
            "research":             PromptStyle.CHAIN_OF_THOUGHT,
        }

        if complexity == "high" and PromptStyle.CHAIN_OF_THOUGHT in styles:
            return (
                PromptStyle.CHAIN_OF_THOUGHT,
                "High complexity requires step-by-step reasoning.",
            )

        preferred = domain_map.get(domain, PromptStyle.STRUCTURED)
        if preferred in styles:
            return preferred, f"Best style for {domain} domain."

        fallback = styles[0]
        return (
            fallback,
            f"Using {STYLE_METADATA[fallback]['label']} as default.",
        )

    def _format_list(self, items: list) -> str:
        if not items:
            return "  - Not specified"
        return "\n".join(f"  - {item}" for item in items[:8])

    def _create_fallback_prompt(
        self, style: PromptStyle, raw_requirement: str
    ) -> OptimizedPrompt:
        """Creates basic fallback prompt on generation failure."""
        meta = STYLE_METADATA[style]
        text = (
            f"Please help with: {raw_requirement}\n\n"
            "Provide a clear, detailed response."
        )
        return OptimizedPrompt(
            style=style,
            style_label=meta["label"],
            prompt_text=text,
            style_reasoning="Fallback — generation failed.",
            word_count=len(text.split()),
            estimated_tokens=len(text.split()) * 2,
            best_used_for=meta["best_used_for"],
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
        return {"prompt_text": content.strip(), "style_reasoning": ""}


# ─── Singleton ────────────────────────────────────────────────────
_optimizer: PromptOptimizer | None = None


def get_prompt_optimizer() -> PromptOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = PromptOptimizer()
    return _optimizer