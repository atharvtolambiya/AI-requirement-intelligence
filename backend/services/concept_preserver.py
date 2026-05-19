"""
Concept Preservation Layer.

PHASE 4 — Stage 1 of drift prevention.

Builds enforceable preservation rules from:
- VisionProfile (semantic anchors)
- SemanticIntent (concept map)
- InnovationProfile (anti-patterns + preservation rules)
- AbstractionClassification (level instructions)

Output: PreservationRuleSet with pre-formatted text blocks
ready to inject into downstream LLM calls.

Key principle:
  Rules are not abstract guidelines — they are CONCRETE,
  ENFORCEABLE instructions that get directly injected into
  every system/user prompt for the expander, optimizer, and refiner.
"""

import re
import time
from typing import Any

from backend.logger import logger
from backend.models.schemas import (
    ConceptPreservationRule,
    PreservationRuleSet,
    PreservationRulesRequest,
    PreservationRulesResponse,
)


class ConceptPreserver:
    """
    Builds and formats preservation rules deterministically.

    Note: This service does NOT use an LLM — it transforms
    structured data from previous stages into rule sets and
    pre-formatted injection blocks. This guarantees consistent,
    fast, deterministic rule generation.
    """

    def __init__(self):
        logger.info("ConceptPreserver initialized")

    def build_rules(
        self,
        request: PreservationRulesRequest,
    ) -> PreservationRulesResponse:
        """
        Builds a complete preservation rule set from semantic data.

        Args:
            request: PreservationRulesRequest with vision + semantic data

        Returns:
            PreservationRulesResponse with rules + injection text
        """
        start_time = time.time()
        logger.info("Building preservation rules")

        rules: list[ConceptPreservationRule] = []
        rule_counter = 0

        # ── Source 1: Vision Profile semantic anchors ─────────────
        vision = request.vision_profile or {}
        anchors = vision.get("semantic_anchors", [])

        for anchor in anchors:
            if not isinstance(anchor, dict):
                continue
            rule_counter += 1

            concept = anchor.get("concept", "")
            meaning = anchor.get("preserved_meaning", "")
            anti_patterns = anchor.get("anti_patterns", [])
            importance = anchor.get("importance", "high")
            anchor_id = anchor.get("anchor_id", f"anchor_{rule_counter}")

            # Must-include rule for the concept itself
            if anchor.get("must_appear_in_prompt", True):
                rules.append(
                    ConceptPreservationRule(
                        rule_id=f"r{rule_counter}_include",
                        rule_type="must_include",
                        concept=concept,
                        rule_text=(
                            f"The concept '{concept}' MUST appear in the "
                            f"expansion. Meaning: {meaning}"
                        ),
                        enforcement_level=(
                            "hard" if importance == "critical" else "soft"
                        ),
                        source_anchor_id=anchor_id,
                    )
                )

            # Must-not-become rules from anti-patterns
            for anti in anti_patterns:
                rule_counter += 1
                rules.append(
                    ConceptPreservationRule(
                        rule_id=f"r{rule_counter}_avoid",
                        rule_type="must_not_become",
                        concept=concept,
                        rule_text=(
                            f"The concept '{concept}' MUST NOT be reduced "
                            f"to: {anti}"
                        ),
                        enforcement_level="hard",
                        source_anchor_id=anchor_id,
                    )
                )

        # ── Source 2: Vision dominant_collapse_pattern ────────────
        collapse_pattern = vision.get("dominant_collapse_pattern", "")
        if collapse_pattern:
            rule_counter += 1
            rules.append(
                ConceptPreservationRule(
                    rule_id=f"r{rule_counter}_collapse",
                    rule_type="must_not_become",
                    concept="overall_system",
                    rule_text=(
                        f"The expansion MUST NOT collapse into: "
                        f"{collapse_pattern}"
                    ),
                    enforcement_level="hard",
                )
            )

        # ── Source 3: Innovation Profile anti-patterns ────────────
        innovation = request.innovation_profile or {}
        for anti in innovation.get("anti_patterns", []):
            rule_counter += 1
            rules.append(
                ConceptPreservationRule(
                    rule_id=f"r{rule_counter}_innov_avoid",
                    rule_type="must_not_become",
                    concept="innovation_protection",
                    rule_text=str(anti),
                    enforcement_level="hard",
                )
            )

        # ── Source 4: Innovation preservation rules ───────────────
        for rule in innovation.get("preservation_rules", []):
            rule_counter += 1
            rules.append(
                ConceptPreservationRule(
                    rule_id=f"r{rule_counter}_innov_keep",
                    rule_type="must_preserve_meaning",
                    concept="innovation_preservation",
                    rule_text=str(rule),
                    enforcement_level="hard",
                )
            )

        # ── Source 5: Innovation claims that are at-risk ──────────
        for claim in innovation.get("innovation_claims", []):
            if not isinstance(claim, dict):
                continue
            if not claim.get("at_risk", False):
                continue
            rule_counter += 1
            rules.append(
                ConceptPreservationRule(
                    rule_id=f"r{rule_counter}_claim",
                    rule_type="must_preserve_meaning",
                    concept=claim.get("claim", ""),
                    rule_text=claim.get(
                        "preservation_instruction",
                        f"Preserve claim: {claim.get('claim', '')}",
                    ),
                    enforcement_level="hard",
                )
            )

        # ── Source 6: Semantic Intent collapse-risk concepts ──────
        intent = request.semantic_intent or {}
        for node in intent.get("concept_map", []):
            if not isinstance(node, dict):
                continue
            risk = node.get("collapse_risk", "low")
            if risk not in ("high", "critical"):
                continue

            generic = node.get("generic_equivalent", "")
            name = node.get("name", "")

            if generic and name:
                rule_counter += 1
                rules.append(
                    ConceptPreservationRule(
                        rule_id=f"r{rule_counter}_intent",
                        rule_type="must_not_become",
                        concept=name,
                        rule_text=(
                            f"'{name}' MUST NOT be reduced to '{generic}'"
                        ),
                        enforcement_level=(
                            "hard" if risk == "critical" else "soft"
                        ),
                    )
                )

        # ── Source 7: Abstraction expansion instructions ──────────
        abstraction = request.abstraction_classification or {}
        for instr in abstraction.get("expansion_level_instructions", []):
            rule_counter += 1
            rules.append(
                ConceptPreservationRule(
                    rule_id=f"r{rule_counter}_abstract",
                    rule_type="must_explain_reasoning",
                    concept="abstraction_alignment",
                    rule_text=str(instr),
                    enforcement_level="hard",
                )
            )

        # ── Build categorized lists (with Deduplication) ──────────────
        must_include = self._deduplicate_rules([
            r for r in rules
            if r.rule_type == "must_include"
        ])
        must_not_become = self._deduplicate_rules([
            r for r in rules
            if r.rule_type == "must_not_become"
        ])
        reasoning_reqs = self._deduplicate_rules([
            r for r in rules
            if r.rule_type in (
                "must_explain_reasoning", "must_preserve_meaning"
            )
        ])

        # ── Build injection text blocks ───────────────────────────
        system_injection = self._build_system_injection(
            must_include,
            must_not_become,
            reasoning_reqs,
        )

        user_injection = self._build_user_injection(
            must_include,
            must_not_become,
        )

        # ── Strategy ──────────────────────────────────────────────
        hard_count = sum(
            1 for r in rules if r.enforcement_level == "hard"
        )
        strategy = (
            f"Inject {len(rules)} preservation rules "
            f"({hard_count} hard, {len(rules) - hard_count} soft) "
            "into all downstream expansion and optimization prompts."
        )

        rule_set = PreservationRuleSet(
            session_id=request.session_id,
            rules=rules,
            rule_count=len(rules),
            must_include_rules=[r.rule_text for r in must_include],
            must_not_become_rules=[r.rule_text for r in must_not_become],
            reasoning_requirements=[r.rule_text for r in reasoning_reqs],
            system_prompt_injection=system_injection,
            user_prompt_injection=user_injection,
            enforcement_strategy=strategy,
        )

        total_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(
            "Preservation rules built | total={t} | hard={h} | "
            "must_include={mi} | must_not_become={mnb} | time={ms}ms",
            t=len(rules),
            h=hard_count,
            mi=len(must_include),
            mnb=len(must_not_become),
            ms=total_ms,
        )

        return PreservationRulesResponse(
            success=True,
            session_id=request.session_id,
            rule_set=rule_set,
            processing_time_ms=total_ms,
            tokens_used=0,  # Deterministic — no LLM call
        )

    # ── Private builders ───────────────────────────────────────────

    def _get_salience_tier(self, rule: ConceptPreservationRule) -> int:
        """
        Classifies rule importance into Tiers (1=highest, 3=lowest).
        """
        if rule.enforcement_level == "hard" and rule.concept in (
            "overall_system", "innovation_protection", "innovation_preservation"
        ):
            return 1 # Core invariants, collapse patterns, innovation identity
        
        # Distinct claims and abstraction level alignments
        if rule.enforcement_level == "hard" and rule.rule_type in (
            "must_preserve_meaning", "must_explain_reasoning"
        ):
            return 1
            
        # Concept maps/anchors that are critical
        if rule.enforcement_level == "hard":
            return 2
        
        return 3 # Soft enforcement, general instructions

    def _deduplicate_rules(
        self, rules: list[ConceptPreservationRule]
    ) -> list[ConceptPreservationRule]:
        """
        Priority-aware semantic deduplication.
        Merges redundant constraints while strictly protecting critical invariants.
        """
        unique_rules: list[ConceptPreservationRule] = []

        for rule in rules:
            tier = self._get_salience_tier(rule)
            
            # TIER 1: Critical Invariants -> NO COMPRESSION
            if tier == 1:
                unique_rules.append(rule)
                continue

            text_clean = rule.rule_text.lower().strip()
            # Basic tokenization
            tokens = set(re.findall(r'\b\w+\b', text_clean))
            is_redundant = False

            for existing in unique_rules:
                # Never merge into a Tier 1 rule (protect invariant identity)
                if self._get_salience_tier(existing) == 1:
                    continue

                # Concept-ID-aware merging (only merge if conceptual lineage matches)
                if rule.concept and existing.concept and rule.concept != existing.concept:
                    continue

                existing_clean = existing.rule_text.lower().strip()
                existing_tokens = set(re.findall(r'\b\w+\b', existing_clean))
                
                if not tokens or not existing_tokens:
                    continue

                # Jaccard similarity
                intersection = tokens.intersection(existing_tokens)
                union = tokens.union(existing_tokens)
                similarity = len(intersection) / len(union)

                # High similarity or direct substring match
                if similarity > 0.65 or text_clean in existing_clean:
                    is_redundant = True
                    break

            if not is_redundant:
                unique_rules.append(rule)

        return unique_rules

    def _build_system_injection(
        self,
        must_include: list[ConceptPreservationRule],
        must_not_become: list[ConceptPreservationRule],
        reasoning_reqs: list[ConceptPreservationRule],
    ) -> str:
        """Builds tier-aware system prompt injection text."""
        all_rules = must_include + must_not_become + reasoning_reqs
        
        tier1_rules = [r.rule_text for r in all_rules if self._get_salience_tier(r) == 1]
        tier23_rules = [r.rule_text for r in all_rules if self._get_salience_tier(r) > 1]
        
        sections = [
            "=" * 60,
            "SEMANTIC PRESERVATION CONSTRAINTS (PRIORITY-AWARE)",
            "=" * 60,
            "",
            "These rules OVERRIDE all default expansion patterns.",
            "",
        ]

        if tier1_rules:
            sections.append("CRITICAL INVARIANTS (NEVER COMPROMISE):")
            sections.append("Violating these constitutes a critical failure.")
            for i, rule in enumerate(tier1_rules, 1):
                sections.append(f"  {i}. {rule}")
            sections.append("")

        if tier23_rules:
            sections.append("GENERAL GUIDELINES & PRESERVATIONS:")
            for i, rule in enumerate(tier23_rules[:15], 1):
                sections.append(f"  {i}. {rule}")
            sections.append("")

        sections.append("=" * 60)
        return "\n".join(sections)

    def _build_user_injection(
        self,
        must_include: list[ConceptPreservationRule],
        must_not_become: list[ConceptPreservationRule],
    ) -> str:
        """Builds user message injection text (shorter, focused)."""
        lines = ["IMPORTANT PRESERVATION RULES FOR THIS REQUEST:"]

        if must_include:
            lines.append("MUST INCLUDE:")
            for rule in must_include[:5]:
                lines.append(f"  • {rule.rule_text}")

        if must_not_become:
            lines.append("MUST NOT BECOME:")
            for rule in must_not_become[:5]:
                lines.append(f"  • {rule.rule_text}")

        return "\n".join(lines)


# ─── Singleton ────────────────────────────────────────────────────
_preserver: ConceptPreserver | None = None


def get_concept_preserver() -> ConceptPreserver:
    global _preserver
    if _preserver is None:
        _preserver = ConceptPreserver()
    return _preserver