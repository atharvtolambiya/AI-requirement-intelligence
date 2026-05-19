"""
Vision Clarification Engine.

This is the FIRST stage of the new semantic reasoning pipeline.

Responsibilities:
1. Analyze raw user input for ambiguity and drift risk
2. Generate adaptive vision-probing questions
3. Extract semantic anchors from user answers
4. Build a complete VisionProfile

The VisionProfile becomes the FOUNDATION for all downstream
processing — it is passed as a hard constraint to:
- Semantic Intent Extractor
- Innovation Analyzer
- Requirement Expander
- Prompt Optimizer

Key principle:
  Questions are NOT generic checklists.
  They are ADAPTIVE — generated specifically to address
  the ambiguities and drift risks detected in the input.
"""

import json
import re
import time
import uuid
from typing import Any

from backend.logger import logger
from backend.models.schemas import (
    AbstractionLevel,
    DriftRisk,
    InnovationType,
    SemanticAnchor,
    VisionAnswers,
    VisionProfile,
    VisionProbe,
    VisionProbeRequest,
    VisionProbeResponse,
    VisionProfileRequest,
    VisionProfileResponse,
    VisionQuestion,
)
from backend.services.llm_service import get_llm_service


# ══════════════════════════════════════════════════════════════════
# VISION PROBE GENERATION
# ══════════════════════════════════════════════════════════════════

PROBE_GENERATION_SYSTEM_PROMPT = """You are a Senior AI Requirements Engineer and 
Semantic Analyst specializing in preserving the conceptual integrity of novel AI systems.

YOUR CRITICAL MISSION:
When users describe AI systems, platforms, or products, their ideas often get 
misinterpreted into generic software patterns. Your job is to generate targeted 
questions that extract the USER'S REAL VISION before any interpretation happens.

WHAT YOU DETECT:
1. Ambiguous terms that could mean multiple things
2. Novel concepts that systems typically collapse into generic patterns
3. Missing conceptual anchors that would prevent semantic drift
4. The "dominant collapse pattern" — what generic product this would become
   if not clarified

KNOWN DANGEROUS COLLAPSE PATTERNS:
- "decision intelligence" → collapses to "ML dashboard"
- "reasoning system"      → collapses to "rule engine or chatbot"
- "domain AI"             → collapses to "domain classifier"
- "augmentation system"   → collapses to "automation tool"
- "contextual AI"         → collapses to "personalization engine"
- "knowledge system"      → collapses to "search or RAG app"
- "prediction platform"   → collapses to "analytics dashboard"
- "intelligent assistant" → collapses to "simple chatbot"

QUESTION GENERATION RULES:
1. Generate EXACTLY 4-6 questions — no more, no less
2. Each question targets a SPECIFIC ambiguity or drift risk
3. Questions must be CONCRETE — not vague philosophical probes
4. Each question must prevent a SPECIFIC collapse pattern
5. Questions should feel like a smart colleague asking for clarity
6. At least 2 questions must be marked is_critical: true

QUESTION CATEGORIES:
- user_transformation: Who uses it and what changes for them
- intelligence_type: What kind of AI reasoning is at the core
- differentiation: How it differs from closest existing product
- domain_reasoning: What domain knowledge it must understand
- output_value: What a successful output looks like

RESPONSE FORMAT (strict JSON):
{
  "questions": [
    {
      "id": "q1",
      "category": "intelligence_type",
      "question": "Specific question text",
      "purpose": "What ambiguity this resolves",
      "example_answer": "Example of a specific good answer",
      "is_critical": true,
      "anti_pattern_hint": "What collapse this prevents"
    }
  ],
  "detected_ambiguities": ["ambiguity 1", "ambiguity 2"],
  "initial_drift_risk": "low|medium|high|critical",
  "dominant_pattern": "The generic product this would become without clarification",
  "reasoning": "Why these specific questions were chosen (2-3 sentences)"
}"""


PROBE_GENERATION_USER_TEMPLATE = """Analyze this user input and generate targeted 
vision-clarification questions:

USER INPUT:
{raw_input}

{context_section}

Identify the ambiguities, detect the dominant collapse pattern, assess drift risk,
and generate adaptive questions that will extract the REAL vision.

Respond with JSON only:"""


# ══════════════════════════════════════════════════════════════════
# VISION PROFILE EXTRACTION
# ══════════════════════════════════════════════════════════════════

PROFILE_EXTRACTION_SYSTEM_PROMPT = """You are a Senior AI Systems Architect and 
Semantic Reasoning Expert.

YOUR MISSION:
Extract a complete, structured VisionProfile from a user's answers to vision 
clarification questions. This profile becomes the SEMANTIC FOUNDATION for all 
downstream AI reasoning — it must preserve every nuance of the user's intent.

EXTRACTION PRINCIPLES:
1. LITERAL PRESERVATION: Use the user's exact words when possible
2. ANCHOR EXTRACTION: Find the non-negotiable conceptual claims
3. ANTI-PATTERN GENERATION: For each anchor, define what it must NOT become
4. DRIFT RISK ASSESSMENT: How likely is this to be misinterpreted?
5. CONFIDENCE SCORING: How complete is the extracted vision?

SEMANTIC ANCHOR TYPES:
- intelligence_claim: "The system understands WHY, not just WHAT"
- reasoning_requirement: "Must reason about consequences, not just classify"
- differentiation_marker: "Unlike X, this system does Y"
- domain_constraint: "Must understand [specific domain rule/constraint]"
- user_transformation: "User goes from [state A] to [state B]"
- output_requirement: "Output must include [specific quality]"

INNOVATION TYPE CLASSIFICATION:
- automation:     Removes manual steps entirely
- reasoning:      Thinks through complex multi-step problems
- recommendation: Suggests best option from alternatives
- intelligence:   Understands context at a deep semantic level
- augmentation:   Enhances human capability without replacing judgment
- prediction:     Forecasts future states or outcomes
- orchestration:  Coordinates multiple systems/workflows intelligently
- generation:     Creates novel artifacts (content, code, plans)

ABSTRACTION LEVEL CLASSIFICATION:
- task:     A single automated action
- feature:  A capability within a larger product
- product:  A complete standalone product
- system:   A complex multi-component intelligent system
- paradigm: A new way of approaching a class of problems

DRIFT RISK ASSESSMENT:
- low:      Clear concrete vision, anchors are specific
- medium:   Some ambiguity remains, anchors need reinforcement
- high:     Multiple generic patterns could absorb this idea
- critical: Vision is so novel that any standard pattern will distort it

RESPONSE FORMAT (strict JSON):
{
  "primary_user": "Specific description of who uses this",
  "core_transformation": "User goes from [X] to [Y] — specific",
  "intelligence_description": "What feels intelligent — specific quote",
  "innovation_type": "one of the innovation types",
  "abstraction_level": "one of the abstraction levels",
  "closest_existing_product": "Name of closest existing tool",
  "differentiation_claims": ["specific claim 1", "specific claim 2"],
  "domain_constraints": ["constraint 1", "constraint 2"],
  "consequence_types": ["consequence type 1", "consequence type 2"],
  "semantic_anchors": [
    {
      "anchor_id": "anchor_1",
      "concept": "Short concept name",
      "anchor_type": "intelligence_claim",
      "original_quote": "Exact words from user answer",
      "preserved_meaning": "What this means and must mean",
      "anti_patterns": ["must not become X", "must not reduce to Y"],
      "must_appear_in_prompt": true,
      "importance": "critical|high|medium"
    }
  ],
  "vision_confidence": 0-100,
  "unanswered_critical_questions": ["question that was skipped"],
  "initial_drift_risk": "low|medium|high|critical",
  "dominant_collapse_pattern": "What this would become without anchors"
}"""


PROFILE_EXTRACTION_USER_TEMPLATE = """Extract the VisionProfile from these 
vision clarification answers:

ORIGINAL INPUT:
{raw_input}

QUESTIONS AND ANSWERS:
{qa_pairs}

Extract all semantic anchors, classify the innovation type and abstraction level,
and assess drift risk. Use the user's exact words wherever possible.

Respond with JSON only:"""


class VisionClarifier:
    """
    Vision Clarification Engine.

    Two-step process:
    Step 1: generate_probe()    → creates adaptive questions for the user
    Step 2: extract_profile()   → builds VisionProfile from user's answers

    The VisionProfile is the output — a structured semantic foundation
    that all downstream services use as a hard constraint.
    """

    def __init__(self):
        self.llm_service = get_llm_service()
        logger.info("VisionClarifier initialized")

    # ══════════════════════════════════════════════════════════════
    # STEP 1: GENERATE VISION PROBE
    # ══════════════════════════════════════════════════════════════

    async def generate_probe(
        self,
        request: VisionProbeRequest,
    ) -> VisionProbeResponse:
        """
        Analyzes user input and generates adaptive vision-probing questions.

        Questions are specifically designed to:
        - Resolve detected ambiguities
        - Prevent identified collapse patterns
        - Extract semantic anchors before any expansion

        Args:
            request: VisionProbeRequest with raw user input

        Returns:
            VisionProbeResponse with targeted questions
        """
        start_time = time.time()
        logger.info(
            "Generating vision probe | input_len={l}",
            l=len(request.raw_input),
        )

        # Build context section
        context_section = ""
        if request.context:
            context_section = f"ADDITIONAL CONTEXT:\n{request.context}\n"

        user_message = PROBE_GENERATION_USER_TEMPLATE.format(
            raw_input=request.raw_input,
            context_section=context_section,
        )

        # Call LLM
        llm_result = await self.llm_service.invoke(
            system_prompt=PROBE_GENERATION_SYSTEM_PROMPT,
            user_message=user_message,
            temperature_override=0.3,
        )

        # Parse response
        probe_data = self._parse_json_response(llm_result["content"])
        probe = self._build_vision_probe(
            data=probe_data,
            raw_input=request.raw_input,
            session_id=request.session_id,
        )

        total_time_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(
            "Vision probe generated | questions={q} | "
            "drift_risk={risk} | collapse_pattern={pattern} | "
            "time={t}ms",
            q=len(probe.questions),
            risk=probe.initial_drift_risk,
            pattern=probe.dominant_pattern[:50],
            t=total_time_ms,
        )

        return VisionProbeResponse(
            success=True,
            session_id=request.session_id,
            raw_input=request.raw_input,
            probe=probe,
            processing_time_ms=total_time_ms,
            tokens_used=llm_result["tokens_used"],
        )

    # ══════════════════════════════════════════════════════════════
    # STEP 2: EXTRACT VISION PROFILE
    # ══════════════════════════════════════════════════════════════

    async def extract_profile(
        self,
        request: VisionProfileRequest,
    ) -> VisionProfileResponse:
        """
        Extracts a complete VisionProfile from user's answers.

        Builds structured semantic anchors, classifies innovation type
        and abstraction level, and assesses drift risk.

        Args:
            request: VisionProfileRequest with answers to probe questions

        Returns:
            VisionProfileResponse with complete VisionProfile
        """
        start_time = time.time()
        logger.info(
            "Extracting vision profile | answers={count}",
            count=len(request.answers),
        )

        # Format Q&A pairs for LLM
        qa_pairs = self._format_qa_pairs(request.answers)

        user_message = PROFILE_EXTRACTION_USER_TEMPLATE.format(
            raw_input=request.raw_input,
            qa_pairs=qa_pairs,
        )

        # Call LLM
        llm_result = await self.llm_service.invoke(
            system_prompt=PROFILE_EXTRACTION_SYSTEM_PROMPT,
            user_message=user_message,
            temperature_override=0.2,  # Low temp for precise extraction
        )

        # Parse response
        profile_data = self._parse_json_response(llm_result["content"])
        vision_profile = self._build_vision_profile(
            data=profile_data,
            session_id=request.session_id,
        )

        total_time_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(
            "Vision profile extracted | "
            "innovation_type={itype} | abstraction={level} | "
            "anchors={anchors} | confidence={conf} | "
            "drift_risk={risk} | time={t}ms",
            itype=vision_profile.innovation_type,
            level=vision_profile.abstraction_level,
            anchors=len(vision_profile.semantic_anchors),
            conf=vision_profile.vision_confidence,
            risk=vision_profile.initial_drift_risk,
            t=total_time_ms,
        )

        return VisionProfileResponse(
            success=True,
            session_id=request.session_id,
            raw_input=request.raw_input,
            vision_profile=vision_profile,
            processing_time_ms=total_time_ms,
            tokens_used=llm_result["tokens_used"],
        )

    # ══════════════════════════════════════════════════════════════
    # PRIVATE: BUILD METHODS
    # ══════════════════════════════════════════════════════════════

    def _build_vision_probe(
        self,
        data: dict[str, Any],
        raw_input: str,
        session_id: str | None,
    ) -> VisionProbe:
        """Builds validated VisionProbe from parsed LLM data."""

        # Parse questions
        questions: list[VisionQuestion] = []
        raw_questions = data.get("questions", [])

        for q in raw_questions:
            if not isinstance(q, dict):
                continue
            questions.append(
                VisionQuestion(
                    id=str(q.get("id", f"q{len(questions)+1}")),
                    category=str(q.get("category", "general")),
                    question=str(q.get("question", "")),
                    purpose=str(q.get("purpose", "")),
                    example_answer=q.get("example_answer"),
                    is_critical=bool(q.get("is_critical", False)),
                    anti_pattern_hint=q.get("anti_pattern_hint"),
                )
            )

        # Parse drift risk
        drift_str = str(data.get("initial_drift_risk", "medium")).lower()
        try:
            drift_risk = DriftRisk(drift_str)
        except ValueError:
            drift_risk = DriftRisk.MEDIUM

        # Parse ambiguities
        ambiguities = data.get("detected_ambiguities", [])
        if not isinstance(ambiguities, list):
            ambiguities = []
        ambiguities = [str(a) for a in ambiguities if a]

        return VisionProbe(
            session_id=session_id,
            raw_input=raw_input,
            questions=questions,
            detected_ambiguities=ambiguities,
            initial_drift_risk=drift_risk,
            dominant_pattern=str(
                data.get("dominant_pattern", "generic software platform")
            ),
            reasoning=str(data.get("reasoning", "")),
        )

    def _build_vision_profile(
        self,
        data: dict[str, Any],
        session_id: str | None,
    ) -> VisionProfile:
        """Builds validated VisionProfile from parsed LLM data."""

        # Parse innovation type
        itype_str = str(data.get("innovation_type", "unknown")).lower()
        try:
            innovation_type = InnovationType(itype_str)
        except ValueError:
            innovation_type = InnovationType.UNKNOWN

        # Parse abstraction level
        alevel_str = str(data.get("abstraction_level", "product")).lower()
        try:
            abstraction_level = AbstractionLevel(alevel_str)
        except ValueError:
            abstraction_level = AbstractionLevel.PRODUCT

        # Parse drift risk
        drift_str = str(data.get("initial_drift_risk", "medium")).lower()
        try:
            drift_risk = DriftRisk(drift_str)
        except ValueError:
            drift_risk = DriftRisk.MEDIUM

        # Parse semantic anchors
        anchors: list[SemanticAnchor] = []
        raw_anchors = data.get("semantic_anchors", [])

        for i, a in enumerate(raw_anchors):
            if not isinstance(a, dict):
                continue

            anti_patterns = a.get("anti_patterns", [])
            if not isinstance(anti_patterns, list):
                anti_patterns = []

            anchors.append(
                SemanticAnchor(
                    anchor_id=str(
                        a.get("anchor_id", f"anchor_{i+1}")
                    ),
                    concept=str(a.get("concept", "")),
                    anchor_type=str(
                        a.get("anchor_type", "intelligence_claim")
                    ),
                    original_quote=str(a.get("original_quote", "")),
                    preserved_meaning=str(
                        a.get("preserved_meaning", "")
                    ),
                    anti_patterns=[
                        str(ap) for ap in anti_patterns if ap
                    ],
                    must_appear_in_prompt=bool(
                        a.get("must_appear_in_prompt", True)
                    ),
                    importance=str(a.get("importance", "high")),
                )
            )

        # Parse confidence
        try:
            confidence = float(data.get("vision_confidence", 50.0))
            confidence = max(0.0, min(100.0, confidence))
        except (TypeError, ValueError):
            confidence = 50.0

        # Parse list fields safely
        def safe_list(key: str) -> list[str]:
            val = data.get(key, [])
            if isinstance(val, list):
                return [str(v) for v in val if v]
            return []

        return VisionProfile(
            session_id=session_id,
            primary_user=str(
                data.get("primary_user", "Not specified")
            ),
            core_transformation=str(
                data.get("core_transformation", "Not specified")
            ),
            intelligence_description=str(
                data.get("intelligence_description", "Not specified")
            ),
            innovation_type=innovation_type,
            abstraction_level=abstraction_level,
            closest_existing_product=str(
                data.get("closest_existing_product", "Unknown")
            ),
            differentiation_claims=safe_list("differentiation_claims"),
            domain_constraints=safe_list("domain_constraints"),
            consequence_types=safe_list("consequence_types"),
            semantic_anchors=anchors,
            vision_confidence=confidence,
            unanswered_critical_questions=safe_list(
                "unanswered_critical_questions"
            ),
            initial_drift_risk=drift_risk,
            dominant_collapse_pattern=str(
                data.get(
                    "dominant_collapse_pattern",
                    "generic software platform",
                )
            ),
        )

    def _format_qa_pairs(self, answers: list[VisionAnswers]) -> str:
        """Formats Q&A pairs into a readable string for LLM."""
        lines = []
        for i, ans in enumerate(answers, 1):
            lines.append(f"Q{i}: {ans.question_text}")
            lines.append(f"A{i}: {ans.answer}")
            lines.append("")
        return "\n".join(lines)

    def _parse_json_response(self, content: str) -> dict[str, Any]:
        """
        Robust JSON extraction with multiple fallback strategies.
        Shared across all methods.
        """
        # Strategy 1: Direct parse
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: Code block extraction
        match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            content,
            re.DOTALL,
        )
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find outermost JSON object
        start = content.find("{")
        end   = content.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass

        logger.error(
            "Vision clarifier JSON parse failed | preview={p}",
            p=content[:200],
        )
        return {}


# ─── Singleton ────────────────────────────────────────────────────
_vision_clarifier: VisionClarifier | None = None


def get_vision_clarifier() -> VisionClarifier:
    """Returns the VisionClarifier singleton."""
    global _vision_clarifier
    if _vision_clarifier is None:
        _vision_clarifier = VisionClarifier()
    return _vision_clarifier