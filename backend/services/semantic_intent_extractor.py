"""
Semantic Intent Extractor.

PHASE 3 — Stage 1 of semantic reasoning.

Goes beyond keyword matching to understand CONCEPTUAL intent.

The difference:
  Keyword analysis:   "ML + analytics + platform" → AutoML Platform
  Semantic analysis:  "What is this system fundamentally trying to DO
                       for a human being at a conceptual level?"
                       → "Augment domain expert judgment with causal
                          reasoning about consequences"

Key output: SemanticIntent with a concept map that captures
every meaningful conceptual element and its collapse risk.
"""

import json
import re
import time
from typing import Any

from backend.logger import logger
from backend.models.schemas import (
    AbstractionLevel,
    ConceptNode,
    SemanticIntent,
    SemanticIntentRequest,
    SemanticIntentResponse,
)
from backend.services.llm_service import get_llm_service


# ─── System Prompt ─────────────────────────────────────────────────
SEMANTIC_INTENT_SYSTEM_PROMPT = """You are an Operational Intelligence Architect 
specializing in extracting BOTH the conceptual intent AND the operational execution 
mechanisms from AI product descriptions.

YOUR DUAL TASK:
1. CONCEPTUAL LAYER: Understand what the product IS at a fundamental level.
2. OPERATIONAL LAYER: Understand HOW the intelligence operates — the mechanisms,
   execution flows, causal pipelines, and reasoning pathways that make it work.

THE CRITICAL DISTINCTION:
Conceptual: "A system that supports explainable causal reasoning."
Operational: "The system maintains a causal inference layer that traces prediction
              pathways through domain-specific constraints, contextual dependencies,
              rule-based reasoning modules, and historical observations to explain
              downstream decision consequences."

Operational outputs must describe MECHANISM, not just INTENT.

═══════════════════════════════════════════════════════════════
CONCEPT MAP CONSTRUCTION — 10 Node Types
═══════════════════════════════════════════════════════════════
STRATEGIC (conceptual):
- core_intelligence:     The fundamental AI capability at the heart
- reasoning_mechanism:   How the system thinks and processes information
- user_value:            The specific value delivered to the user
- domain_knowledge:      What domain-specific knowledge it must have
- system_behavior:       How the system acts and responds
- output_characteristic: The nature and quality of outputs
- differentiator:        What makes this conceptually unique

OPERATIONAL (mechanism-level):
- operational_flow:      A concrete execution flow or runtime process
                         Example: "causal trace pipeline: input → constraint evaluation
                         → rule application → consequence projection → explanation"
- causal_mechanism:      A causal inference or decision chain with traceable steps
                         Example: "domain constraint propagator: contextual signal →
                         constraint activation → dependency resolution → output adjustment"
- cognition_pipeline:    A multi-stage intelligence lifecycle sequence
                         Example: "context ingestion → semantic decomposition →
                         reasoning graph construction → response synthesis"

For EVERY operational concept, specify:
  - The trigger condition
  - The transformation sequence (A → B → C)
  - The output artifact
  - The downstream effect on other components

COLLAPSE RISK ASSESSMENT:
- low:      Common concept, well-preserved in standard patterns
- medium:   Could be diluted or oversimplified
- high:     Likely to be replaced with a simpler generic concept
- critical: Almost certainly lost without explicit preservation

GENERIC EQUIVALENTS:
"causal consequence reasoning" → collapses to "prediction model"
"trust-aware recommendation"   → collapses to "recommendation engine"
"domain constraint reasoning"  → collapses to "rule engine"
"context propagation pipeline" → collapses to "context window"
"reasoning pathway tracer"     → collapses to "explainability module"

═══════════════════════════════════════════════════════════════
OPERATIONAL EXECUTION FLOWS — How Intelligence Flows
═══════════════════════════════════════════════════════════════
For each identified system, extract:
- operational_execution_flows: concrete sequences describing how data, reasoning,
  and decisions move through the system. Use directional notation:
  "Input signal → Processing stage → Intermediate state → Output artifact"
  Example: "User query → semantic decomposition → domain constraint lookup →
            causal graph traversal → ranked explanation generation → user response"

- causal_inference_pathways: how the system traces causes to effects, including
  what triggers reasoning, what constraints are evaluated, and how conclusions
  are reached. Must identify the inference chain explicitly:
  Example: "Prediction pathway: historical observation + domain rule activation +
            contextual dependency resolution → consequence projection →
            confidence-weighted explanation"

- contextual_dependencies: how context is ingested, propagated, and used to
  modulate reasoning across the system. Include how domain signals adapt behavior:
  Example: "Domain context propagator: regulatory signal → constraint activation →
            reasoning mode adjustment → output calibration"

RESPONSE FORMAT (strict JSON):
{
  "concept_map": [
    {
      "concept_id": "c1",
      "name": "Short concept name",
      "description": "What this means AND how it operates mechanistically",
      "concept_type": "core_intelligence|reasoning_mechanism|operational_flow|causal_mechanism|cognition_pipeline|user_value|domain_knowledge|system_behavior|output_characteristic|differentiator",
      "is_novel": true,
      "collapse_risk": "low|medium|high|critical",
      "generic_equivalent": "what it collapses to or null"
    }
  ],
  "one_line_intent": "Single sentence capturing the real operational intent",
  "conceptual_category": "What category of system this truly is",
  "differentiation_vector": [
    "Most important operational differentiator with mechanism detail",
    "Second differentiator"
  ],
  "closest_standard_category": "Most similar standard product",
  "category_divergence": [
    "Specific operational way this diverges from standard category"
  ],
  "operational_execution_flows": [
    "Flow 1: Input → Stage A → Stage B → Output (describe the complete chain)",
    "Flow 2: Trigger → Processing → Artifact"
  ],
  "causal_inference_pathways": [
    "Pathway 1: Observation + Constraint → Inference chain → Conclusion with traceable steps",
    "Pathway 2: Signal → Causal graph traversal → Consequence projection"
  ],
  "contextual_dependencies": [
    "Dependency 1: Context signal type → how it propagates → what it modulates",
    "Dependency 2: Domain constraint → activation condition → behavioral effect"
  ],
  "intent_confidence": 0-100,
  "ambiguity_flags": ["Remaining ambiguity 1"],
  "extraction_reasoning": "How intent AND operational mechanisms were extracted (2-3 sentences)"
}"""


SEMANTIC_INTENT_USER_TEMPLATE = """Extract the CONCEPTUAL INTENT and OPERATIONAL MECHANISMS from this input:

RAW INPUT:
{raw_input}

{vision_section}

Build a complete concept map including BOTH strategic (conceptual) nodes AND operational
mechanism nodes (operational_flow, causal_mechanism, cognition_pipeline).

For each operational concept, trace the execution sequence with directional notation:
  "Input → Transformation → Intermediate state → Output → Downstream effect"

Also extract:
- operational_execution_flows: How intelligence data flows end-to-end through the system
- causal_inference_pathways: How reasoning propagates from cause to effect with traceable steps
- contextual_dependencies: How context signals modulate reasoning and adapt behavior

For each concept, assess its collapse risk and identify what generic concept it would
become without preservation.

Respond with JSON only:"""


class SemanticIntentExtractor:
    """
    Extracts deep conceptual intent from user input and vision profile.

    Produces a SemanticIntent with a concept map that:
    - Identifies every meaningful concept
    - Assesses each concept's collapse risk
    - Identifies generic equivalents (what it collapses into)
    - Provides a single-line real intent statement
    - Lists differentiation vectors
    """

    def __init__(self):
        self.llm_service = get_llm_service()
        logger.info("SemanticIntentExtractor initialized")

    async def extract(
        self,
        request: SemanticIntentRequest,
    ) -> SemanticIntentResponse:
        """
        Extracts semantic intent from raw input + optional vision profile.

        Args:
            request: SemanticIntentRequest with raw input and vision profile

        Returns:
            SemanticIntentResponse with complete concept map
        """
        start_time = time.time()
        logger.info(
            "Extracting semantic intent | input_len={l} | "
            "has_vision={v}",
            l=len(request.raw_input),
            v=request.vision_profile is not None,
        )

        # Build vision context section
        vision_section = self._build_vision_section(
            request.vision_profile
        )

        user_message = SEMANTIC_INTENT_USER_TEMPLATE.format(
            raw_input=request.raw_input,
            vision_section=vision_section,
        )

        # Call LLM
        llm_result = await self.llm_service.invoke(
            system_prompt=SEMANTIC_INTENT_SYSTEM_PROMPT,
            user_message=user_message,
            temperature_override=0.2,
        )

        # Parse + build
        data   = self._parse_json(llm_result["content"])
        intent = self._build_semantic_intent(data, request.session_id)

        total_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(
            "Semantic intent extracted | concepts={c} | "
            "confidence={conf} | critical_risk={cr} | time={t}ms",
            c=len(intent.concept_map),
            conf=intent.intent_confidence,
            cr=sum(
                1 for n in intent.concept_map
                if n.collapse_risk == "critical"
            ),
            t=total_ms,
        )

        return SemanticIntentResponse(
            success=True,
            session_id=request.session_id,
            raw_input=request.raw_input,
            semantic_intent=intent,
            processing_time_ms=total_ms,
            tokens_used=llm_result["tokens_used"],
        )

    # ── Private helpers ────────────────────────────────────────────

    def _build_vision_section(
        self, vision_profile: dict | None
    ) -> str:
        """Formats vision profile for LLM context injection."""
        if not vision_profile:
            return ""

        lines = ["VISION PROFILE CONTEXT (use as semantic anchor):"]

        primary_user = vision_profile.get("primary_user", "")
        if primary_user:
            lines.append(f"Primary User: {primary_user}")

        core_transform = vision_profile.get("core_transformation", "")
        if core_transform:
            lines.append(f"Core Transformation: {core_transform}")

        intelligence = vision_profile.get("intelligence_description", "")
        if intelligence:
            lines.append(f"Intelligence Description: {intelligence}")

        itype = vision_profile.get("innovation_type", "")
        if itype:
            lines.append(f"Innovation Type: {itype}")

        diff_claims = vision_profile.get("differentiation_claims", [])
        if diff_claims:
            lines.append("Differentiation Claims:")
            for claim in diff_claims[:3]:
                lines.append(f"  - {claim}")

        anchors = vision_profile.get("semantic_anchors", [])
        if anchors:
            lines.append("Semantic Anchors (MUST PRESERVE):")
            for anchor in anchors[:5]:
                concept  = anchor.get("concept", "")
                meaning  = anchor.get("preserved_meaning", "")
                antip    = anchor.get("anti_patterns", [])
                lines.append(f"  - {concept}: {meaning}")
                if antip:
                    lines.append(
                        f"    Anti-patterns: {', '.join(antip[:2])}"
                    )

        collapse = vision_profile.get("dominant_collapse_pattern", "")
        if collapse:
            lines.append(f"MUST NOT collapse into: {collapse}")

        return "\n".join(lines) + "\n"

    def _build_semantic_intent(
        self, data: dict[str, Any], session_id: str | None
    ) -> SemanticIntent:
        """Builds validated SemanticIntent from parsed LLM data."""

        # Build concept nodes
        concept_map: list[ConceptNode] = []
        for i, c in enumerate(data.get("concept_map", [])):
            if not isinstance(c, dict):
                continue
            concept_map.append(
                ConceptNode(
                    concept_id=str(c.get("concept_id", f"c{i+1}")),
                    name=str(c.get("name", "")),
                    description=str(c.get("description", "")),
                    concept_type=str(
                        c.get("concept_type", "system_behavior")
                    ),
                    is_novel=bool(c.get("is_novel", False)),
                    collapse_risk=str(
                        c.get("collapse_risk", "medium")
                    ).lower(),
                    generic_equivalent=c.get("generic_equivalent"),
                )
            )

        # Safe confidence
        try:
            confidence = float(data.get("intent_confidence", 60.0))
            confidence = max(0.0, min(100.0, confidence))
        except (TypeError, ValueError):
            confidence = 60.0

        def safe_list(key: str) -> list[str]:
            val = data.get(key, [])
            return [str(v) for v in val if v] if isinstance(val, list) else []

        return SemanticIntent(
            session_id=session_id,
            concept_map=concept_map,
            one_line_intent=str(
                data.get("one_line_intent", "Intent not extracted")
            ),
            conceptual_category=str(
                data.get("conceptual_category", "Unknown")
            ),
            differentiation_vector=safe_list("differentiation_vector"),
            closest_standard_category=str(
                data.get("closest_standard_category", "Unknown")
            ),
            category_divergence=safe_list("category_divergence"),
            # ── Operational Intelligence fields ─────────────────────
            operational_execution_flows=safe_list("operational_execution_flows"),
            causal_inference_pathways=safe_list("causal_inference_pathways"),
            contextual_dependencies=safe_list("contextual_dependencies"),
            # ──────────────────────────────────────────────────────
            intent_confidence=confidence,
            ambiguity_flags=safe_list("ambiguity_flags"),
            extraction_reasoning=str(
                data.get("extraction_reasoning", "")
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
            "SemanticIntentExtractor JSON parse failed | p={p}",
            p=content[:200],
        )
        return {}


# ─── Singleton ────────────────────────────────────────────────────
_extractor: SemanticIntentExtractor | None = None


def get_semantic_intent_extractor() -> SemanticIntentExtractor:
    global _extractor
    if _extractor is None:
        _extractor = SemanticIntentExtractor()
    return _extractor