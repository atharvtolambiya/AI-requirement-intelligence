"""
Reasoning Alignment Engine.

PHASE 4 — Stage 2 of drift prevention.

Generates pre-expansion reasoning alignment instructions.
This service runs BEFORE the expander to ensure expansion
reasons at the correct abstraction level and preserves intent.

The output is a complete instruction block that gets injected
into the expander's system prompt to align its reasoning mode.

Key insight:
  Standard LLM expansion uses a default reasoning mode
  (typically feature-first, product-level reasoning).
  This service explicitly tells the LLM what reasoning mode
  to use based on the user's actual abstraction level
  and innovation tier.
"""

import json
import re
import time
from typing import Any

from backend.logger import logger
from backend.models.schemas import (
    AbstractionLevel,
    ReasoningAlignment,
    ReasoningAlignmentRequest,
    ReasoningAlignmentResponse,
)
from backend.services.llm_service import get_llm_service


# ─── System Prompt ─────────────────────────────────────────────────
REASONING_ALIGNMENT_SYSTEM_PROMPT = """You are an Operational Intelligence Reasoning Architect
specializing in aligning LLM reasoning modes to generate OPERATIONALLY GROUNDED,
MECHANISM-LEVEL intelligence architecture specifications.

YOUR MISSION:
Generate reasoning alignment instructions that direct a downstream expander to reason
AT THE MECHANISM LEVEL — not just at the conceptual level.

The expander must not only describe WHAT the system is, but HOW it operates:
- HOW intelligence flows through the system
- HOW reasoning propagates across components
- HOW domain constraints interact with execution
- HOW causal inference is executed step by step
- HOW contextual adaptation occurs dynamically
- HOW operational cognition behaves over the intelligence lifecycle

═══════════════════════════════════════════════════════════════
REASONING MODES — 8 Available Modes
═══════════════════════════════════════════════════════════════
CONCEPTUAL MODES (abstract-first):
- concept_first:          Reason about underlying concepts before features
- architecture_first:     Reason about system architecture before components
- feature_first:          Reason about features and capabilities (avoid for novel ideas)
- outcome_first:          Reason about outcomes and value before mechanism
- reasoning_first:        Reason about how the system reasons before what it does

OPERATIONAL MODES (mechanism-first):
- operational_first:      Reason by tracing execution flows and component interactions.
                          Start with HOW data and decisions flow, then derive components.
                          Use for systems where runtime behavior IS the core value.
- causal_execution_first: Reason by constructing causal inference chains explicitly.
                          Trace every input → constraint activation → inference step →
                          output consequence. Use when causal traceability is the
                          innovation (e.g. explainable AI, decision intelligence).
- mechanism_first:        Reason by identifying concrete mechanisms before abstractions.
                          Define trigger → process → output for every intelligence claim.
                          Use when the HOW is more novel than the WHAT.

WHEN TO USE EACH MODE:
- concept_first:          User is at SYSTEM or PARADIGM abstraction level
- architecture_first:     System-level with strong architectural intent
- feature_first:          PRODUCT or FEATURE level (avoid for novel ideas)
- outcome_first:          Values are clear, mechanism is secondary
- reasoning_first:        AI REASONING MECHANISM is the core innovation
- operational_first:      Execution flows and component interactions are the value
- causal_execution_first: Causal traceability or decision explainability is key
- mechanism_first:        The HOW is more differentiated than the WHAT

═══════════════════════════════════════════════════════════════
OPERATIONAL ALIGNMENT REQUIREMENTS
═══════════════════════════════════════════════════════════════
For each concept in the system, the expander must generate language that:

1. MECHANISM-LEVEL SYNTHESIS:
   - Not: "The system supports explainable reasoning."
   - Yes: "The system maintains a causal inference layer that traces prediction
           pathways through domain-specific constraints, contextual dependencies,
           rule-based reasoning modules, and historical observations to explain
           downstream decision consequences."

2. CAUSAL EXECUTION GRAPH:
   - Identify every causal chain: input → trigger → evaluation → consequence
   - Specify constraint activation: which domain rules activate under what conditions
   - Trace decision provenance: how each output is causally explained

3. INTELLIGENCE LIFECYCLE MODELING:
   - Describe how intelligence evolves over time (not just at request time)
   - Model: ingestion phase → learning phase → inference phase → adaptation phase
   - Include how domain feedback loops close and improve reasoning

4. COMPONENT INTERACTION DYNAMICS:
   - For each component pair, describe the interaction interface
   - Specify what signals flow between components and in what direction
   - Model how a state change in one component propagates to others

5. EXECUTION-FLOW DEPENDENCY MODELING:
   - Identify which execution flows are sequential vs parallel
   - Specify dependency: "Component B cannot reason until Component A has resolved X"
   - Trace where bottlenecks or uncertainty propagation occurs

6. CONTEXTUAL DEPENDENCY PROPAGATION:
   - How does domain context enter the system?
   - How does it propagate through the reasoning chain?
   - How does it modulate outputs differently across contexts?

7. OPERATIONAL EXPLAINABILITY:
   - Every intelligence claim must be backed by a traceable mechanism
   - Not: "The system understands context"
   - Yes: "The system maintains a sliding context buffer that dynamically weights
           historical signals, domain constraints, and real-time observations,
           re-calibrating its reasoning pathway on each inference cycle."

ALIGNMENT INSTRUCTION GENERATION:
Generate a complete instruction block that:
1. Specifies the reasoning mode (conceptual AND operational)
2. Lists what aspects to reason about at the MECHANISM level
3. Lists what aspects to avoid reasoning about
4. Identifies protected operational concepts and their execution logic
5. Specifies the expansion approach with operational grounding
6. Provides a pre-expansion checklist covering both concept and mechanism layers
7. Includes operational execution flows to preserve verbatim

EXAMPLE OUTPUT FOR A DECISION INTELLIGENCE SYSTEM:
{
  "primary_reasoning_mode": "causal_execution_first",
  "abstraction_target": "system",
  "must_reason_about": [
    "HOW causal inference chains trace from input to decision consequence",
    "HOW domain constraints activate and propagate through the reasoning pipeline",
    "HOW human judgment and AI inference are combined at each decision stage",
    "HOW the intelligence lifecycle adapts from historical feedback"
  ],
  "must_avoid_reasoning_about": [
    "ML model selection and training infrastructure",
    "Dashboard UI layout and visualization components",
    "Standard CRUD database operations"
  ],
  "protected_concepts": [
    "causal consequence reasoning",
    "domain-constraint-aware inference",
    "human-AI augmentation loop"
  ],
  "protected_operational_flows": [
    "Input → domain constraint evaluation → causal graph traversal → consequence projection → explanation synthesis"
  ],
  "forbidden_collapses": [
    "ML platform",
    "analytics dashboard",
    "recommendation engine"
  ],
  "expansion_approach": "Start with HOW the system executes causal reasoning, then derive conceptual components from those mechanisms",
  "expansion_anti_approach": "Do not start with generic technical components and attempt to make them sound intelligent",
  "pre_expansion_checklist": [
    "Does every intelligence claim trace to a concrete execution mechanism?",
    "Is the causal inference pipeline described with step-by-step traceability?",
    "Are component interactions modeled with explicit signal flows?",
    "Is the intelligence lifecycle (ingestion, inference, adaptation) addressed?",
    "Does the expansion explain HOW reasoning propagates, not just WHAT it produces?"
  ]
}

CRITICAL: Respond with VALID JSON only.

RESPONSE FORMAT:
{
  "primary_reasoning_mode": "concept_first|architecture_first|feature_first|outcome_first|reasoning_first|operational_first|causal_execution_first|mechanism_first",
  "abstraction_target": "task|feature|product|system|paradigm",
  "must_reason_about": ["HOW aspect 1 executes", "HOW aspect 2 propagates"],
  "must_avoid_reasoning_about": ["avoid 1", "avoid 2"],
  "protected_concepts": ["concept 1", "concept 2"],
  "protected_operational_flows": ["Flow 1: A → B → C", "Flow 2: X → Y"],
  "forbidden_collapses": ["collapse pattern 1"],
  "expansion_approach": "Mechanism-grounded approach description",
  "expansion_anti_approach": "Approach to avoid",
  "pre_expansion_checklist": ["check 1 (mechanism level)", "check 2 (execution level)"]
}"""


REASONING_ALIGNMENT_USER_TEMPLATE = """Generate OPERATIONAL reasoning alignment instructions
for this expansion task.

The expander must reason at the MECHANISM LEVEL — describing HOW intelligence
executes, HOW reasoning propagates, and HOW components interact.

RAW INPUT:
{raw_input}

VISION PROFILE:
{vision_section}

SEMANTIC INTENT:
{intent_section}

INNOVATION PROFILE:
{innovation_section}

ABSTRACTION CLASSIFICATION:
{abstraction_section}

OPERATIONAL FLOWS ALREADY IDENTIFIED:
{operational_flows_section}

Based on the above, generate complete reasoning alignment that:
1. Selects the RIGHT reasoning mode (prefer operational_first, causal_execution_first,
   or mechanism_first when the system's HOW is the core innovation)
2. Mandates mechanism-level synthesis in all must_reason_about entries
3. Captures protected_operational_flows that MUST survive into the expansion
4. Sets a pre-expansion checklist that enforces execution-level grounding

Be specific and actionable. Use HOW/WHAT framing consistently.

Respond with JSON only:"""


class ReasoningAligner:
    """
    Generates pre-expansion reasoning alignment instructions.

    Takes all upstream semantic analysis data and produces
    a complete instruction block that tells the expander HOW
    to think about the problem at the right abstraction level
    using the right reasoning mode.
    """

    def __init__(self):
        self.llm_service = get_llm_service()
        logger.info("ReasoningAligner initialized")

    async def align(
        self,
        request: ReasoningAlignmentRequest,
    ) -> ReasoningAlignmentResponse:
        """
        Generates operational reasoning alignment instructions.

        Now passes operational flows already identified by the semantic
        intent extractor into the alignment prompt, so the aligner can
        protect those flows verbatim and mandate mechanism-level reasoning.
        """
        start_time = time.time()
        logger.info(
            "Generating reasoning alignment | input_len={l}",
            l=len(request.raw_input),
        )

        vision_section      = self._format_dict(request.vision_profile, max_keys=8)
        intent_section      = self._format_dict(request.semantic_intent, max_keys=8)
        innovation_section  = self._format_dict(request.innovation_profile, max_keys=6)
        abstraction_section = self._format_dict(
            request.abstraction_classification, max_keys=6
        )

        # Extract operational flows already identified in semantic intent
        operational_flows_section = self._extract_operational_flows(
            request.semantic_intent
        )

        user_message = REASONING_ALIGNMENT_USER_TEMPLATE.format(
            raw_input=request.raw_input,
            vision_section=vision_section,
            intent_section=intent_section,
            innovation_section=innovation_section,
            abstraction_section=abstraction_section,
            operational_flows_section=operational_flows_section,
        )

        llm_result = await self.llm_service.invoke(
            system_prompt=REASONING_ALIGNMENT_SYSTEM_PROMPT,
            user_message=user_message,
            temperature_override=0.2,
        )

        data       = self._parse_json(llm_result["content"])
        alignment  = self._build_alignment(data, request.session_id)

        total_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(
            "Reasoning alignment generated | mode={m} | "
            "target={t} | protected={p} | op_flows={f} | time={ms}ms",
            m=alignment.primary_reasoning_mode,
            t=alignment.abstraction_target,
            p=len(alignment.protected_concepts),
            f=len(alignment.protected_operational_flows),
            ms=total_ms,
        )

        return ReasoningAlignmentResponse(
            success=True,
            session_id=request.session_id,
            alignment=alignment,
            processing_time_ms=total_ms,
            tokens_used=llm_result["tokens_used"],
        )

    # ── Private helpers ────────────────────────────────────────────

    def _extract_operational_flows(self, semantic_intent: dict | None) -> str:
        """Extracts and formats operational flows already identified in semantic intent."""
        if not semantic_intent:
            return "(none identified yet)"
        lines = []
        exec_flows = semantic_intent.get("operational_execution_flows", [])
        if isinstance(exec_flows, list) and exec_flows:
            lines.append("Execution Flows:")
            for f in exec_flows[:4]:
                lines.append(f"  • {str(f)[:300]}")
        causal = semantic_intent.get("causal_inference_pathways", [])
        if isinstance(causal, list) and causal:
            lines.append("Causal Inference Pathways:")
            for p in causal[:3]:
                lines.append(f"  • {str(p)[:300]}")
        ctx = semantic_intent.get("contextual_dependencies", [])
        if isinstance(ctx, list) and ctx:
            lines.append("Contextual Dependencies:")
            for d in ctx[:3]:
                lines.append(f"  • {str(d)[:300]}")
        return "\n".join(lines) if lines else "(none identified yet)"

    def _format_dict(
        self, data: dict | None, max_keys: int = 8
    ) -> str:
        """Formats a context dict for LLM consumption."""
        if not data:
            return "(not provided)"

        lines = []
        count = 0
        for key, val in data.items():
            if count >= max_keys:
                break
            count += 1

            if isinstance(val, str) and val:
                lines.append(f"  {key}: {val[:200]}")
            elif isinstance(val, list) and val:
                items = [str(v) for v in val[:3]]
                lines.append(f"  {key}: {', '.join(items)}")
            elif isinstance(val, dict):
                lines.append(f"  {key}: <{len(val)} fields>")
            elif val is not None:
                lines.append(f"  {key}: {val}")

        return "\n".join(lines) if lines else "(empty)"

    def _build_alignment(
        self, data: dict[str, Any], session_id: str | None
    ) -> ReasoningAlignment:
        """Builds validated ReasoningAlignment with operational fields."""

        # Reasoning mode — now includes 3 new operational modes
        mode = str(
            data.get("primary_reasoning_mode", "concept_first")
        ).lower()
        valid_modes = {
            "concept_first", "architecture_first",
            "feature_first", "outcome_first", "reasoning_first",
            "operational_first", "causal_execution_first", "mechanism_first",
        }
        if mode not in valid_modes:
            mode = "operational_first"  # safe default for novel systems

        # Abstraction target
        target_str = str(
            data.get("abstraction_target", "system")
        ).lower()
        try:
            abstraction_target = AbstractionLevel(target_str)
        except ValueError:
            abstraction_target = AbstractionLevel.SYSTEM

        def safe_list(key: str) -> list[str]:
            val = data.get(key, [])
            return (
                [str(v) for v in val if v]
                if isinstance(val, list) else []
            )

        must_reason_about           = safe_list("must_reason_about")
        must_avoid                  = safe_list("must_avoid_reasoning_about")
        protected_concepts          = safe_list("protected_concepts")
        protected_operational_flows = safe_list("protected_operational_flows")
        forbidden_collapses         = safe_list("forbidden_collapses")
        pre_expansion_checklist     = safe_list("pre_expansion_checklist")

        expansion_approach = str(
            data.get("expansion_approach", "")
        )
        expansion_anti_approach = str(
            data.get("expansion_anti_approach", "")
        )

        # Build complete instruction block
        instructions = self._build_instruction_block(
            mode,
            abstraction_target.value,
            must_reason_about,
            must_avoid,
            protected_concepts,
            protected_operational_flows,
            forbidden_collapses,
            expansion_approach,
            expansion_anti_approach,
            pre_expansion_checklist,
        )

        return ReasoningAlignment(
            session_id=session_id,
            primary_reasoning_mode=mode,
            abstraction_target=abstraction_target,
            must_reason_about=must_reason_about,
            must_avoid_reasoning_about=must_avoid,
            protected_concepts=protected_concepts,
            forbidden_collapses=forbidden_collapses,
            expansion_approach=expansion_approach,
            expansion_anti_approach=expansion_anti_approach,
            pre_expansion_checklist=pre_expansion_checklist,
            alignment_instructions=instructions,
        )

    def _build_instruction_block(
        self,
        mode: str,
        target_level: str,
        must_reason: list[str],
        must_avoid: list[str],
        protected: list[str],
        protected_op_flows: list[str],
        forbidden: list[str],
        approach: str,
        anti_approach: str,
        checklist: list[str],
    ) -> str:
        """Builds complete operational instruction block for prompt injection."""

        # Determine if this is an operational mode
        operational_modes = {
            "operational_first", "causal_execution_first", "mechanism_first"
        }
        is_operational = mode in operational_modes

        lines = [
            "=" * 60,
            "OPERATIONAL REASONING ALIGNMENT INSTRUCTIONS" if is_operational
            else "REASONING ALIGNMENT INSTRUCTIONS",
            "=" * 60,
            "",
            f"REASONING MODE: {mode.upper()}",
            f"ABSTRACTION TARGET: {target_level}",
            "",
        ]

        if is_operational:
            lines += [
                "OPERATIONAL IMPERATIVE:",
                "  Do NOT merely describe what the system IS.",
                "  You MUST describe HOW it executes at every level:",
                "  - HOW intelligence flows (use A → B → C notation)",
                "  - HOW reasoning propagates across components",
                "  - HOW domain constraints activate and modulate outputs",
                "  - HOW causal inference chains from input to consequence",
                "  - HOW the intelligence lifecycle evolves over time",
                "  Every intelligence claim MUST be backed by a traceable mechanism.",
                "",
            ]

        if approach:
            lines.append(f"EXPANSION APPROACH: {approach}")
            lines.append("")
        if anti_approach:
            lines.append(f"AVOID THIS APPROACH: {anti_approach}")
            lines.append("")

        if must_reason:
            lines.append("REASON EXPLICITLY ABOUT (at mechanism level):")
            for item in must_reason:
                lines.append(f"  • {item}")
            lines.append("")

        if must_avoid:
            lines.append("DO NOT REASON ABOUT:")
            for item in must_avoid:
                lines.append(f"  • {item}")
            lines.append("")

        if protected:
            lines.append("PROTECTED CONCEPTS (preserve meaning AND mechanism):")
            for item in protected:
                lines.append(f"  • {item}")
            lines.append("")

        if protected_op_flows:
            lines.append(
                "PROTECTED OPERATIONAL FLOWS (must appear in expansion verbatim or paraphrased):"
            )
            lines.append(
                "  These execution sequences are core architectural identity. "
                "Preserve each flow stage:"
            )
            for item in protected_op_flows:
                lines.append(f"  ► {item}")
            lines.append("")

        if forbidden:
            lines.append("FORBIDDEN COLLAPSES:")
            for item in forbidden:
                lines.append(f"  • Do not become: {item}")
            lines.append("")

        if checklist:
            label = "PRE-OUTPUT CHECKLIST (mechanism layer):"
            lines.append(label)
            for i, item in enumerate(checklist, 1):
                lines.append(f"  {i}. {item}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

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
            "ReasoningAligner JSON parse failed | p={p}",
            p=content[:200],
        )
        return {}


# ─── Singleton ────────────────────────────────────────────────────
_aligner: ReasoningAligner | None = None


def get_reasoning_aligner() -> ReasoningAligner:
    global _aligner
    if _aligner is None:
        _aligner = ReasoningAligner()
    return _aligner