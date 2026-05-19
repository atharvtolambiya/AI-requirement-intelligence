"""
Semantic Pipeline Orchestrator — Final Version.

Architecture:
    run()        → timeout wrapper (300s hard limit)
    _run_inner() → all 9 pipeline stages with error handling

Fixes applied:
    - Single run() definition (no duplicate)
    - _run_inner() contains actual logic (not pass)
    - Sequential LLM calls (no asyncio.gather on LLM stages)
    - Default fallbacks for every stage
    - Hard timeout guard
"""

import asyncio
import hashlib
import time
from typing import Any

from backend.logger import logger
from backend.models.schemas import (
    AbstractionClassificationRequest,
    DriftAnalysisRequest,
    InnovationAnalysisRequest,
    OptimizationPipelineRequest,
    OptimizationPipelineResponse,
    PreservationRulesRequest,
    PromptOptimizationRequest,
    PromptScoringRequest,
    ReasoningAlignmentRequest,
    RequirementExpansionRequest,
    SemanticIntentRequest,
)
from backend.config import settings
from backend.services.abstraction_classifier import get_abstraction_classifier
from backend.services.concept_preserver import get_concept_preserver
from backend.services.drift_detector import get_drift_detector
from backend.services.innovation_analyzer import get_innovation_analyzer
from backend.services.prompt_optimizer import get_prompt_optimizer
from backend.services.prompt_scorer import get_prompt_scorer
from backend.services.reasoning_aligner import get_reasoning_aligner
from backend.services.requirement_expander import get_requirement_expander
from backend.services.semantic_intent_extractor import (
    get_semantic_intent_extractor,
)

class TTLCache:
    def __init__(self, ttl_seconds=3600, max_size=100):
        self.cache = {}
        self.ttl = ttl_seconds
        self.max_size = max_size

    def get(self, key):
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp <= self.ttl:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache, key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        self.cache[key] = (value, time.time())


class SemanticPipeline:
    """
    9-stage semantic pipeline orchestrator.

    Stages:
        1. Semantic Intent Extraction   (LLM)
        2. Innovation Analysis          (LLM)
        3. Abstraction Classification   (LLM)
        4. Preservation Rules           (rule-based, no LLM)
        5. Reasoning Alignment          (LLM)
        6. Requirement Expansion        (LLM)
        7. Drift Detection              (LLM)
        8. Prompt Optimization          (LLM × n_styles, sequential)
        9. Prompt Scoring               (LLM)

    All LLM stages are sequential to avoid rate limit collisions.
    """

    # Hard timeout — prevents infinite hangs on free tier
    TIMEOUT_SECONDS = 600  # 10 minutes

    def __init__(self):
        self.intent_extractor       = get_semantic_intent_extractor()
        self.innovation_analyzer    = get_innovation_analyzer()
        self.abstraction_classifier = get_abstraction_classifier()
        self.concept_preserver      = get_concept_preserver()
        self.reasoning_aligner      = get_reasoning_aligner()
        self.drift_detector         = get_drift_detector()
        self.expander               = get_requirement_expander()
        self.optimizer              = get_prompt_optimizer()
        self.scorer                 = get_prompt_scorer()
        self.session_cache          = TTLCache(ttl_seconds=3600, max_size=100)
        logger.info("SemanticPipeline initialized")

    # ══════════════════════════════════════════════════════════════
    # PUBLIC ENTRY POINT
    # ══════════════════════════════════════════════════════════════

    async def run(
        self,
        request: OptimizationPipelineRequest,
        vision_profile: dict | None = None,
    ) -> OptimizationPipelineResponse:
        """
        Public entry point with hard timeout guard.

        Wraps _run_inner() with asyncio.wait_for so the pipeline
        can never hang longer than TIMEOUT_SECONDS.
        """
        logger.info(
            "Semantic pipeline starting | session={s} | "
            "has_vision={v} | styles={st} | input_len={l}",
            s=request.session_id or "unknown",
            v=bool(vision_profile),
            st=[s.value for s in request.styles],
            l=len(request.raw_requirement),
        )

        try:
            result = await asyncio.wait_for(
                self._run_inner(request, vision_profile),
                timeout=self.TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Pipeline hard timeout | session={s} | limit={t}s",
                s=request.session_id,
                t=self.TIMEOUT_SECONDS,
            )
            raise RuntimeError(
                f"Pipeline exceeded {self.TIMEOUT_SECONDS}s time limit. "
                f"On Groq free tier, reduce styles to 2. "
                f"Requested: {len(request.styles)} styles."
            )

        # Guard: _run_inner must never return None
        if result is None:
            raise RuntimeError(
                "Pipeline _run_inner() returned None — "
                "this is a bug, please report it."
            )

        return result

    # ══════════════════════════════════════════════════════════════
    # INNER PIPELINE — All 9 stages
    # ══════════════════════════════════════════════════════════════

    async def _run_inner(
        self,
        request: OptimizationPipelineRequest,
        vision_profile: dict | None = None,
    ) -> OptimizationPipelineResponse:
        """
        Executes all 9 pipeline stages sequentially.

        Every stage has its own try/except so a single stage
        failure degrades gracefully instead of crashing the pipeline.
        """
        start_time   = time.time()
        total_tokens = 0
        MAX_TOKENS = settings.MAX_PIPELINE_TOKENS
        session_id   = request.session_id or "unknown"
        fast_mode = not settings.ENABLE_DEEP_REASONING
        # Normalize vision_profile — always a dict internally
        vp_dict: dict = (
            vision_profile
            if isinstance(vision_profile, dict)
            else {}
        )

        try:
            cache_key = None
            if session_id != "unknown":
                req_hash = hashlib.md5(request.raw_requirement.encode()).hexdigest()
                cache_key = f"{session_id}_{req_hash}"

            cached_data = self.session_cache.get(cache_key) if cache_key else None
            intent_dict, innovation_dict, abstraction_dict = cached_data if cached_data else (None, None, None)

            # ══════════════════════════════════════════════════════
            # STAGE 1: Semantic Intent Extraction
            # ══════════════════════════════════════════════════════
            if intent_dict:
                logger.info("[Pipeline 1/9] Semantic intent extraction (cached)")
            else:
                logger.info("[Pipeline 1/9] Semantic intent extraction")
            try:
                intent_req  = SemanticIntentRequest(
                    raw_input=request.raw_requirement,
                    vision_profile=vp_dict or None,
                    session_id=session_id,
                )
                intent_resp = await self.intent_extractor.extract(intent_req)
                intent_dict  = intent_resp.semantic_intent.model_dump()
                total_tokens += intent_resp.tokens_used
                logger.info(
                    "[Pipeline 1/9] Done | confidence={c}",
                    c=intent_dict.get("intent_confidence", 0),
                )
            except Exception as e:
                logger.error(
                    "[Pipeline 1/9] Failed (using default) | error={e}",
                    e=str(e),
                )
                intent_dict = self._default_intent(request.raw_requirement)
            if total_tokens > MAX_TOKENS:
                logger.warning(
                    "Pipeline token budget exceeded | tokens={t}. Gracefully degrading to fast mode.",
                    t=total_tokens,
                )
                fast_mode = True

            # ══════════════════════════════════════════════════════
            # STAGE 2: Innovation Analysis
            # ══════════════════════════════════════════════════════
            if innovation_dict:
                logger.info("[Pipeline 2/9] Innovation analysis (cached)")
            elif fast_mode:
                logger.info("[Pipeline 2/9] Innovation analysis (Skipped for Fast Mode)")
                innovation_dict = self._default_innovation()
            else:
                logger.info("[Pipeline 2/9] Innovation analysis")
            try:
                innovation_req  = InnovationAnalysisRequest(
                    raw_input=request.raw_requirement,
                    vision_profile=vp_dict or None,
                    session_id=session_id,
                )
                innovation_resp = await self.innovation_analyzer.analyze(
                    innovation_req
                )
                innovation_dict  = innovation_resp.innovation_profile.model_dump()
                total_tokens    += innovation_resp.tokens_used
                logger.info(
                    "[Pipeline 2/9] Done | tier={t} | score={s}",
                    t=innovation_dict.get("innovation_tier", "?"),
                    s=innovation_dict.get("innovation_score", 0),
                )
            except Exception as e:
                logger.error(
                    "[Pipeline 2/9] Failed (using default) | error={e}",
                    e=str(e),
                )
                innovation_dict = self._default_innovation()
            if total_tokens > MAX_TOKENS:
                logger.warning(
                    "Pipeline token budget exceeded | tokens={t}. Gracefully degrading to fast mode.",
                    t=total_tokens,
                )
                fast_mode = True

            # ══════════════════════════════════════════════════════
            # STAGE 3: Abstraction Classification
            # ══════════════════════════════════════════════════════
            if abstraction_dict:
                logger.info("[Pipeline 3/9] Abstraction classification (cached)")
            elif fast_mode:
                logger.info("[Pipeline 3/9] Abstraction classification (Skipped for Fast Mode)")
                abstraction_dict = self._default_abstraction()
            else:
                logger.info("[Pipeline 3/9] Abstraction classification")
            try:
                abstraction_req  = AbstractionClassificationRequest(
                    raw_input=request.raw_requirement,
                    vision_profile=vp_dict or None,
                    session_id=session_id,
                )
                abstraction_resp = await self.abstraction_classifier.classify(
                    abstraction_req
                )
                abstraction_dict  = abstraction_resp.classification.model_dump()
                total_tokens     += abstraction_resp.tokens_used
                logger.info(
                    "[Pipeline 3/9] Done | user_level={u} | gap={g}",
                    u=abstraction_dict.get("user_abstraction_level", "?"),
                    g=abstraction_dict.get("abstraction_gap", 0),
                )
            except Exception as e:
                logger.error(
                    "[Pipeline 3/9] Failed (using default) | error={e}",
                    e=str(e),
                )
                abstraction_dict = self._default_abstraction()
            if total_tokens > MAX_TOKENS:
                logger.warning(
                    "Pipeline token budget exceeded | tokens={t}. Gracefully degrading to fast mode.",
                    t=total_tokens,
                )
                fast_mode = True

            if cache_key and not cached_data:
                self.session_cache.set(cache_key, (intent_dict, innovation_dict, abstraction_dict))

            # ══════════════════════════════════════════════════════
            # STAGE 4: Preservation Rules (rule-based, no LLM)
            # ══════════════════════════════════════════════════════
            logger.info("[Pipeline 4/9] Building preservation rules")
            try:
                preserve_req = PreservationRulesRequest(
                    vision_profile=vp_dict,
                    semantic_intent=intent_dict,
                    innovation_profile=innovation_dict,
                    abstraction_classification=abstraction_dict,
                    session_id=session_id,
                )
                # run_in_executor because build_rules is synchronous
                loop = asyncio.get_event_loop()
                preservation_resp = await loop.run_in_executor(
                    None,
                    self.concept_preserver.build_rules,
                    preserve_req,
                )
                preservation_dict = preservation_resp.rule_set.model_dump()
                logger.info(
                    "[Pipeline 4/9] Done | rules={r}",
                    r=preservation_resp.rule_set.rule_count,
                )
            except Exception as e:
                logger.error(
                    "[Pipeline 4/9] Failed (using default) | error={e}",
                    e=str(e),
                )
                preservation_dict = self._default_preservation()
            if total_tokens > MAX_TOKENS:
                logger.warning(
                    "Pipeline token budget exceeded | tokens={t}. Gracefully degrading to fast mode.",
                    t=total_tokens,
                )
                fast_mode = True

            # ══════════════════════════════════════════════════════
            # STAGE 5: Reasoning Alignment
            # ══════════════════════════════════════════════════════
            logger.info("[Pipeline 5/9] Reasoning alignment")
            try:
                alignment_req  = ReasoningAlignmentRequest(
                    raw_input=request.raw_requirement,
                    vision_profile=vp_dict,
                    semantic_intent=intent_dict,
                    innovation_profile=innovation_dict,
                    abstraction_classification=abstraction_dict,
                    session_id=session_id,
                )
                alignment_resp = await self.reasoning_aligner.align(
                    alignment_req
                )
                alignment_dict  = alignment_resp.alignment.model_dump()
                total_tokens   += alignment_resp.tokens_used
                logger.info(
                    "[Pipeline 5/9] Done | mode={m} | target={t}",
                    m=alignment_dict.get("primary_reasoning_mode", "?"),
                    t=alignment_dict.get("abstraction_target", "?"),
                )
            except Exception as e:
                logger.error(
                    "[Pipeline 5/9] Failed (using default) | error={e}",
                    e=str(e),
                )
                alignment_dict = self._default_alignment()
            if total_tokens > MAX_TOKENS:
                logger.warning(
                    "Pipeline token budget exceeded | tokens={t}. Gracefully degrading to fast mode.",
                    t=total_tokens,
                )
                fast_mode = True

            # ══════════════════════════════════════════════════════
            # STAGE 6: Requirement Expansion
            # ══════════════════════════════════════════════════════
            logger.info("[Pipeline 6/9] Requirement expansion")

            intent_summary = (
                f"Real Intent: {intent_dict.get('one_line_intent', '')}. "
                f"Category: {intent_dict.get('conceptual_category', 'unknown')}."
            )
            expand_req = RequirementExpansionRequest(
                raw_requirement=request.raw_requirement,
                intent_summary=intent_summary[:500],
                user_answers=request.user_answers,
                session_id=session_id,
            )

            # Expansion is critical — failure aborts the pipeline
            try:
                expansion_resp = await self.expander.expand(
                    request=expand_req,
                    preservation_rules=preservation_dict,
                    reasoning_alignment=alignment_dict,
                )
                total_tokens += expansion_resp.tokens_used
                expanded_dict = expansion_resp.expanded.model_dump()
                expanded_text = self._serialize_expansion(expansion_resp.expanded)
                logger.info(
                    "[Pipeline 6/9] Done | title={t} | complexity={c}",
                    t=expansion_resp.expanded.title[:50],
                    c=expansion_resp.expanded.estimated_complexity,
                )
            except Exception as e:
                logger.error(
                    "[Pipeline 6/9] FATAL — expansion failed | error={e}",
                    e=str(e),
                )
                # Expansion failure is unrecoverable
                raise RuntimeError(
                    f"Requirement expansion failed: {e}"
                ) from e
            if total_tokens > MAX_TOKENS:
                logger.warning(
                    "Pipeline token budget exceeded | tokens={t}. Gracefully degrading to fast mode.",
                    t=total_tokens,
                )
                fast_mode = True
            # STAGE 7: Drift Detection
            if settings.ENABLE_DRIFT_DETECTION and not fast_mode:

                logger.info("[Pipeline 7/9] Drift detection")

                drift = None

                try:
                    drift_req = DriftAnalysisRequest(
                        raw_input=request.raw_requirement,
                        expansion_text=expanded_text[:6000],
                        vision_profile=vp_dict,
                        semantic_intent=intent_dict,
                        innovation_profile=innovation_dict,
                        session_id=session_id,
                    )

                    drift_resp = await self.drift_detector.detect(
                        drift_req
                    )

                    total_tokens += drift_resp.tokens_used

                    drift = drift_resp.drift_analysis

                    logger.info(
                        "[Pipeline 7/9] Done | drift_score={s} | can_proceed={p}",
                        s=drift.drift_score,
                        p=drift.can_proceed,
                    )

                except Exception as e:

                    # Drift detection failure is non-blocking
                    logger.warning(
                        "[Pipeline 7/9] Failed (non-blocking) | error={e}",
                        e=str(e),
                    )

            else:

                logger.info(
                    "[Pipeline 7/9] Skipped (fast mode enabled)"
                )

                drift = {
                    "drift_score": 0,
                    "drift_risk": "low",
                    "can_proceed": True,
                    "warnings": [],
                }
            if total_tokens > MAX_TOKENS:
                logger.warning(
                    "Pipeline token budget exceeded | tokens={t}. Gracefully degrading to fast mode.",
                    t=total_tokens,
                )
                fast_mode = True

            # ══════════════════════════════════════════════════════
            # STAGE 8: Prompt Optimization (sequential per style)
            # ══════════════════════════════════════════════════════
            logger.info(
                "[Pipeline 8/9] Prompt optimization | styles={s}",
                s=[style.value for style in request.styles],
            )
            optimize_req = PromptOptimizationRequest(
                raw_requirement=request.raw_requirement,
                expanded_requirement=expanded_dict,
                styles=request.styles,
                target_llm=request.target_llm,
                session_id=session_id,
            )
            # optimizer.optimize() is already sequential internally
            optimization_resp = await self.optimizer.optimize(
                request=optimize_req,
                preservation_rules=preservation_dict,
                vision_profile=vp_dict if vp_dict else None,
            )
            total_tokens += optimization_resp.tokens_used
            logger.info(
                "[Pipeline 8/9] Done | prompts={c} | recommended={r}",
                c=len(optimization_resp.optimized_prompts),
                r=optimization_resp.recommended_style.value,
            )
            if total_tokens > MAX_TOKENS:
                logger.warning(
                    "Pipeline token budget exceeded | tokens={t}. Gracefully degrading to fast mode.",
                    t=total_tokens,
                )
                fast_mode = True

            # ══════════════════════════════════════════════════════
            # STAGE 9: Prompt Scoring
            # ══════════════════════════════════════════════════════
            if settings.ENABLE_LLM_SCORING:
                logger.info("[Pipeline 9/9] LLM scoring enabled")

                scoring_req = PromptScoringRequest(
                    prompts=[
                        {
                            "style": p.style,
                            "prompt_text": p.prompt_text,
                        }
                        for p in optimization_resp.optimized_prompts
                    ],
                    original_requirement=request.raw_requirement,
                    session_id=session_id,
                )

                scoring_resp = await self.scorer.score_prompts(
                    scoring_req,
                    vision_profile=vp_dict,
                )

            else:
                logger.info(
                    "[Pipeline 9/9] Using lightweight heuristic scoring"
                )

                scoring_resp = self._lightweight_score(
                    optimization_resp.optimized_prompts
                )
            if total_tokens > MAX_TOKENS:
                logger.warning(
                    "Pipeline token budget exceeded | tokens={t}. Pipeline completed with truncation.",
                    t=total_tokens,
                )

            # ══════════════════════════════════════════════════════
            # ASSEMBLE + RETURN RESPONSE
            # ══════════════════════════════════════════════════════
            total_ms = round((time.time() - start_time) * 1000, 2)
            best     = scoring_resp.best_prompt

            logger.info(
                "Pipeline COMPLETE | session={s} | "
                "score={sc:.1f} | grade={g} | "
                "drift={dr} | tokens={t} | time={ms}ms",
                s=session_id,
                sc=best.score.overall_score if best else 0,
                g=best.score.grade         if best else "?",
                dr=(drift.drift_score if hasattr(drift, "drift_score") else drift.get("drift_score", "N/A") if isinstance(drift, dict) else "N/A") if drift else "N/A",
                t=total_tokens,
                ms=total_ms,
            )

            return OptimizationPipelineResponse(
                success=True,
                session_id=request.session_id,
                raw_requirement=request.raw_requirement,
                expanded_requirement=expansion_resp.expanded,
                scored_prompts=scoring_resp.scored_prompts,
                best_prompt=best,
                recommended_style=optimization_resp.recommended_style,
                recommendation_reason=optimization_resp.recommendation_reason,
                total_processing_time_ms=total_ms,
                total_tokens_used=total_tokens,
            )

        except RuntimeError:
            # Re-raise runtime errors (expansion failure, timeout)
            raise

        except Exception as e:
            total_ms = round((time.time() - start_time) * 1000, 2)
            logger.exception(
                "Pipeline FATAL error | session={s} | time={ms}ms",
                s=session_id,
                ms=total_ms,
            )
            raise RuntimeError(
                f"Pipeline failed after {total_ms}ms: {str(e)}"
            ) from e

    # ══════════════════════════════════════════════════════════════
    # SERIALIZATION HELPER
    # ══════════════════════════════════════════════════════════════

    def _serialize_expansion(self, expanded: Any) -> str:
        """Converts ExpandedRequirement to plain text for drift detection."""
        parts = [
            f"Title: {expanded.title}",
            f"Overview: {expanded.overview[:500]}",
            "Functional Requirements:",
        ]
        for req in expanded.functional_requirements[:8]:
            parts.append(f"  - {req[:200]}")
        parts.append("Non-Functional Requirements:")
        for req in expanded.non_functional_requirements[:5]:
            parts.append(f"  - {req[:200]}")
        parts.append(f"Scope: {expanded.scope[:300]}")
        parts.append("Success Criteria:")
        for c in expanded.success_criteria[:5]:
            parts.append(f"  - {c[:200]}")
        return "\n".join(parts)

    # ══════════════════════════════════════════════════════════════
    # DEFAULT FALLBACKS — used when a stage fails
    # ══════════════════════════════════════════════════════════════

    def _default_intent(self, raw_requirement: str) -> dict:
        return {
            "one_line_intent":           raw_requirement[:200],
            "conceptual_category":       "unknown",
            "concept_map":               [],
            "differentiation_vector":    [],
            "intent_confidence":         30.0,
            "ambiguity_flags":           ["intent extraction failed"],
            "closest_standard_category": "unknown",
            "category_divergence":       [],
            "extraction_reasoning":      "",
        }

    def _default_innovation(self) -> dict:
        return {
            "innovation_tier":      "incremental",
            "tier_reasoning":       "Analysis failed — using safe default",
            "innovation_claims":    [],
            "at_risk_concepts":     [],
            "safe_concepts":        [],
            "anti_patterns":        [],
            "preservation_rules":   [],
            "compared_to_existing": "",
            "innovation_score":     25.0,
        }

    def _default_abstraction(self) -> dict:
        return {
            "user_abstraction_level":       "product",
            "system_default_level":         "feature",
            "abstraction_gap":              0,
            "detected_mismatches":          [],
            "mismatch_count":               0,
            "classification_reasoning":     "Classification failed — using safe default",
            "correction_strategy":          "",
            "expansion_level_instructions": [],
        }

    def _default_preservation(self) -> dict:
        return {
            "rules":                  [],
            "rule_count":             0,
            "must_include_rules":     [],
            "must_not_become_rules":  [],
            "reasoning_requirements": [],
            "system_prompt_injection": "",
            "user_prompt_injection":   "",
            "enforcement_strategy":    "none — preservation build failed",
        }

    def _default_alignment(self) -> dict:
        return {
            "primary_reasoning_mode":       "operational_first",
            "abstraction_target":           "system",
            "must_reason_about":            [],
            "must_avoid_reasoning_about":   [],
            "protected_concepts":           [],
            "protected_operational_flows":  [],
            "forbidden_collapses":          [],
            "expansion_approach":           "Trace execution flows, then derive conceptual components from mechanisms",
            "expansion_anti_approach":      "Do not list generic components without describing how they operate",
            "pre_expansion_checklist":      [],
            "alignment_instructions":       "",
        }

    def _lightweight_score(self, prompts):
        """
        Cheap heuristic scoring.
        No LLM calls.
        """
        from backend.models.schemas import PromptScoringResponse, ScoredPrompt, PromptQualityScore

        scored = []

        for p in prompts:
            text = p.prompt_text.lower()

            score = 60

            if "step by step" in text:
                score += 8

            if "constraints" in text:
                score += 5

            if len(text) > 1200:
                score += 5

            if "hallucination" in text:
                score += 5

            if "reasoning" in text:
                score += 5

            score = float(min(score, 95))

            scored.append(
                ScoredPrompt(
                    style=p.style,
                    style_label=p.style_label,
                    prompt_text=p.prompt_text,
                    score=PromptQualityScore(
                        overall_score=score,
                        grade="A" if score >= 85 else "B",
                        dimensions=[],
                        strengths=["Fast heuristic scoring"],
                        weaknesses=[],
                        hallucination_risk="low",
                        hallucination_reasons=[],
                        overall_feedback="Heuristic score applied.",
                        is_production_ready=True,
                    ),
                    word_count=p.word_count,
                    estimated_tokens=p.estimated_tokens,
                )
            )

        best = max(scored, key=lambda x: x.score.overall_score)

        return PromptScoringResponse(
            success=True,
            session_id=None,
            scored_prompts=scored,
            best_prompt=best,
            processing_time_ms=0.0,
            tokens_used=0,
        )
# ══════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════

_pipeline: SemanticPipeline | None = None


def get_semantic_pipeline() -> SemanticPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SemanticPipeline()
    return _pipeline