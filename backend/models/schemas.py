"""
Pydantic schemas for all API request and response models.

Centralizing schemas here ensures:
- Consistent validation across all endpoints
- Single source of truth for data structures
- Easy reuse across routers and services
- Auto-generated Swagger documentation
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any


# ══════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════

class DomainType(str, Enum):
    """Detected domain/category of the user requirement."""
    SOFTWARE_DEVELOPMENT = "software_development"
    DATA_SCIENCE = "data_science"
    CONTENT_CREATION = "content_creation"
    BUSINESS_ANALYSIS = "business_analysis"
    RESEARCH = "research"
    EDUCATION = "education"
    CREATIVE = "creative"
    GENERAL = "general"
    UNKNOWN = "unknown"


class ComplexityLevel(str, Enum):
    """Complexity rating of the requirement."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MissingInfoSeverity(str, Enum):
    """How critical a missing piece of information is."""
    CRITICAL = "critical"      # Prompt cannot work without this
    IMPORTANT = "important"    # Significantly impacts output quality
    OPTIONAL = "optional"      # Nice to have but not blocking


class ConfidenceLevel(str, Enum):
    """AI confidence in its analysis."""
    HIGH = "high"       # > 80% confident
    MEDIUM = "medium"   # 50-80% confident
    LOW = "low"         # < 50% confident


# ══════════════════════════════════════════════════════════════════
# INTENT ANALYSIS SCHEMAS
# ══════════════════════════════════════════════════════════════════

class IntentAnalysisRequest(BaseModel):
    """Request schema for intent analysis endpoint."""

    raw_requirement: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="The vague or incomplete requirement from the user",
        examples=["Build me something that tracks my habits"],
    )
    context: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional additional context provided by the user",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for tracking conversation history",
    )

    @field_validator("raw_requirement")
    @classmethod
    def clean_requirement(cls, v: str) -> str:
        """Strip whitespace and normalize input."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Requirement cannot be empty or whitespace only")
        return cleaned


class DetectedIntent(BaseModel):
    """Structured representation of the analyzed user intent."""

    primary_goal: str = Field(
        description="The main objective the user wants to achieve"
    )
    secondary_goals: list[str] = Field(
        default_factory=list,
        description="Supporting objectives or sub-goals"
    )
    domain: DomainType = Field(
        description="Detected domain/category of the requirement"
    )
    target_audience: str | None = Field(
        default=None,
        description="Who will use or benefit from the output"
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Detected limitations, restrictions, or boundaries"
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions made during intent analysis"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Key terms extracted from the requirement"
    )
    complexity: ComplexityLevel = Field(
        description="Estimated complexity of the requirement"
    )
    confidence: ConfidenceLevel = Field(
        description="AI confidence level in this analysis"
    )
    raw_reasoning: str = Field(
        description="AI explanation of how intent was determined"
    )


class IntentAnalysisResponse(BaseModel):
    """Response schema for intent analysis endpoint."""

    success: bool
    session_id: str | None
    raw_requirement: str
    intent: DetectedIntent
    processing_time_ms: float
    tokens_used: int
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


# ══════════════════════════════════════════════════════════════════
# MISSING REQUIREMENT DETECTION SCHEMAS
# ══════════════════════════════════════════════════════════════════

class MissingRequirement(BaseModel):
    """A single detected gap or missing piece of information."""

    category: str = Field(
        description="Category of missing info (e.g., 'Technical Stack', 'Scope')"
    )
    question: str = Field(
        description="Specific clarifying question to ask the user"
    )
    why_needed: str = Field(
        description="Explanation of why this information matters"
    )
    severity: MissingInfoSeverity = Field(
        description="How critical this missing info is"
    )
    example_answer: str | None = Field(
        default=None,
        description="Example of what a good answer looks like"
    )
    default_assumption: str | None = Field(
        default=None,
        description="What will be assumed if user doesn't answer"
    )


class RequirementGapAnalysis(BaseModel):
    """Full gap analysis result for a requirement."""

    completeness_score: float = Field(
        ge=0.0,
        le=100.0,
        description="How complete the requirement is (0-100)"
    )
    missing_requirements: list[MissingRequirement] = Field(
        description="List of all detected missing information"
    )
    critical_gaps: list[str] = Field(
        description="Short list of the most critical missing items"
    )
    can_proceed: bool = Field(
        description="Whether optimization can proceed with current info"
    )
    recommendation: str = Field(
        description="Overall recommendation on how to proceed"
    )


class MissingRequirementRequest(BaseModel):
    """Request schema for missing requirement detection."""

    raw_requirement: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="The user requirement to analyze for gaps",
    )
    intent_summary: str | None = Field(
        default=None,
        description="Optional: pre-computed intent summary to guide gap detection",
    )
    domain: str | None = Field(
        default=None,
        description="Optional: detected domain to focus gap detection",
    )
    session_id: str | None = None

    @field_validator("raw_requirement")
    @classmethod
    def clean_requirement(cls, v: str) -> str:
        return v.strip()


class MissingRequirementResponse(BaseModel):
    """Response schema for missing requirement detection endpoint."""

    success: bool
    session_id: str | None
    raw_requirement: str
    gap_analysis: RequirementGapAnalysis
    processing_time_ms: float
    tokens_used: int
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


# ══════════════════════════════════════════════════════════════════
# COMBINED FULL ANALYSIS SCHEMA
# ══════════════════════════════════════════════════════════════════

class FullAnalysisRequest(BaseModel):
    """
    Request for running both intent analysis AND gap detection
    in a single API call (more efficient, single LLM context).
    """

    raw_requirement: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="The user requirement to fully analyze",
    )
    context: str | None = Field(
        default=None,
        max_length=2000,
        description="Optional additional context",
    )
    session_id: str | None = None

    @field_validator("raw_requirement")
    @classmethod
    def clean_requirement(cls, v: str) -> str:
        return v.strip()


class FullAnalysisResponse(BaseModel):
    """Combined response with intent + gap analysis."""

    success: bool
    session_id: str | None
    raw_requirement: str
    intent: DetectedIntent
    gap_analysis: RequirementGapAnalysis
    total_processing_time_ms: float
    total_tokens_used: int
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


# ══════════════════════════════════════════════════════════════════
# GENERIC ERROR SCHEMA
# ══════════════════════════════════════════════════════════════════

class ErrorResponse(BaseModel):
    """Standard error response returned on failures."""

    success: bool = False
    error: str
    detail: str | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

# ══════════════════════════════════════════════════════════════════
# MODULE 3: EXPANSION + OPTIMIZATION + SCORING SCHEMAS
# ══════════════════════════════════════════════════════════════════

class PromptStyle(str, Enum):
    """Supported prompt generation styles."""
    ZERO_SHOT = "zero_shot"           # Direct instruction, no examples
    CHAIN_OF_THOUGHT = "chain_of_thought"  # Step-by-step reasoning
    FEW_SHOT = "few_shot"             # With input/output examples
    ROLE_BASED = "role_based"         # Persona/expert role assignment
    STRUCTURED = "structured"         # Markdown sections with headers


# ══════════════════════════════════════════════════════════════════
# REQUIREMENT EXPANSION SCHEMAS
# ══════════════════════════════════════════════════════════════════

class ExpandedRequirement(BaseModel):
    """Fully expanded and structured version of a vague requirement."""

    title: str = Field(description="Short descriptive title for this requirement")
    overview: str = Field(description="Clear 2-3 sentence overview of what is needed")
    objectives: list[str] = Field(description="Specific, measurable objectives")
    functional_requirements: list[str] = Field(
        description="What the system/output must DO"
    )
    non_functional_requirements: list[str] = Field(
        description="Quality attributes: performance, security, usability"
    )
    scope: str = Field(description="Clear definition of what is in scope")
    out_of_scope: list[str] = Field(description="Explicitly what is NOT included")
    assumptions_made: list[str] = Field(
        description="Assumptions applied during expansion"
    )
    success_criteria: list[str] = Field(
        description="How to measure successful completion"
    )
    suggested_approach: str = Field(
        description="Recommended high-level approach or architecture"
    )
    estimated_complexity: ComplexityLevel
    expansion_confidence: ConfidenceLevel


class RequirementExpansionRequest(BaseModel):
    """Request schema for requirement expansion."""

    raw_requirement: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Original vague requirement",
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context to guide expansion"
    )
    intent_summary: str | None = Field(
        default=None,
        description="Pre-analyzed intent summary (from intent analyzer)",
    )
    domain: str | None = Field(
        default=None,
        description="Detected domain to guide expansion",
    )
    user_answers: dict[str, str] | None = Field(
        default=None,
        description="User answers to clarifying questions (gap filling)",
    )
    session_id: str | None = None

    @field_validator("raw_requirement")
    @classmethod
    def clean_requirement(cls, v: str) -> str:
        return v.strip()


class RequirementExpansionResponse(BaseModel):
    """Response schema for requirement expansion."""

    success: bool
    session_id: str | None
    raw_requirement: str
    expanded: ExpandedRequirement
    processing_time_ms: float
    tokens_used: int
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


# ══════════════════════════════════════════════════════════════════
# PROMPT OPTIMIZATION SCHEMAS
# ══════════════════════════════════════════════════════════════════

class OptimizedPrompt(BaseModel):
    """A single generated prompt in a specific style."""

    style: PromptStyle = Field(description="The prompt engineering style used")
    style_label: str = Field(description="Human-readable style name")
    prompt_text: str = Field(description="The full optimized prompt text")
    style_reasoning: str = Field(
        description="Why this style was applied and how it helps"
    )
    word_count: int = Field(description="Word count of the generated prompt")
    estimated_tokens: int = Field(description="Estimated token count")
    best_used_for: str = Field(
        description="When/where this style works best"
    )


class PromptOptimizationRequest(BaseModel):
    """Request schema for prompt optimization."""

    raw_requirement: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Original requirement to optimize",
    )
    expanded_requirement: dict[str, Any] | None = Field(
        default=None,
        description="Pre-expanded requirement (from expander service)",
    )
    styles: list[PromptStyle] = Field(
        default=[
            PromptStyle.CHAIN_OF_THOUGHT,
            PromptStyle.STRUCTURED,
        ],
        description="List of prompt styles to generate",
    )
    domain: str | None = None
    target_llm: str = Field(
        default="gpt-4",
        description="Target LLM the optimized prompt will be used with",
    )
    session_id: str | None = None

    @field_validator("styles")
    @classmethod
    def validate_styles(cls, v: list) -> list:
        if not v:
            raise ValueError("At least one prompt style must be specified")
        if len(v) > 5:
            raise ValueError("Maximum 5 styles allowed per request")
        return v


class PromptOptimizationResponse(BaseModel):
    """Response schema for prompt optimization."""

    success: bool
    session_id: str | None
    raw_requirement: str
    optimized_prompts: list[OptimizedPrompt]
    recommended_style: PromptStyle = Field(
        description="The style recommended as best for this use case"
    )
    recommendation_reason: str
    processing_time_ms: float
    tokens_used: int
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


# ══════════════════════════════════════════════════════════════════
# PROMPT SCORING SCHEMAS
# ══════════════════════════════════════════════════════════════════

class ScoreDimension(BaseModel):
    """Score for a single quality dimension."""

    dimension: str = Field(description="Name of the scoring dimension")
    score: float = Field(ge=0.0, le=10.0, description="Score from 0-10")
    weight: float = Field(ge=0.0, le=1.0, description="Weight in overall score")
    feedback: str = Field(description="Specific feedback for this dimension")
    suggestions: list[str] = Field(
        default_factory=list,
        description="Concrete improvement suggestions",
    )


class PromptQualityScore(BaseModel):
    """Complete quality assessment for a single prompt."""

    overall_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Weighted overall quality score (0-100)",
    )
    grade: str = Field(description="Letter grade: A+, A, B, C, D, F")
    dimensions: list[ScoreDimension] = Field(
        description="Breakdown by scoring dimension"
    )
    strengths: list[str] = Field(description="What the prompt does well")
    weaknesses: list[str] = Field(description="What needs improvement")
    hallucination_risk: str = Field(
        description="Risk level: low, medium, high"
    )
    hallucination_reasons: list[str] = Field(
        description="Specific reasons for hallucination risk rating"
    )
    overall_feedback: str = Field(
        description="Summary feedback paragraph"
    )
    is_production_ready: bool = Field(
        description="Whether this prompt meets production quality bar"
    )


class ScoredPrompt(BaseModel):
    """An optimized prompt with its quality score attached."""

    style: PromptStyle
    style_label: str
    prompt_text: str
    score: PromptQualityScore
    word_count: int
    estimated_tokens: int


class PromptScoringRequest(BaseModel):
    """Request schema for scoring one or more prompts."""

    prompts: list[dict[str, str]] = Field(
        ...,
        description="List of prompts to score. Each dict needs 'style' and 'prompt_text'",
        min_length=1,
    )
    original_requirement: str = Field(
        ...,
        min_length=5,
        description="The original requirement the prompts were generated from",
    )
    session_id: str | None = None


class PromptScoringResponse(BaseModel):
    """Response schema for prompt scoring."""

    success: bool
    session_id: str | None
    scored_prompts: list[ScoredPrompt]
    best_prompt: ScoredPrompt = Field(
        description="The highest-scoring prompt"
    )
    processing_time_ms: float
    tokens_used: int
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


# ══════════════════════════════════════════════════════════════════
# FULL OPTIMIZATION PIPELINE SCHEMA
# ══════════════════════════════════════════════════════════════════

class OptimizationPipelineRequest(BaseModel):
    """
    Single request that runs the COMPLETE pipeline:
    1. Expand requirement
    2. Generate optimized prompts in multiple styles
    3. Score each prompt
    4. Return best + all results

    This is the PRIMARY endpoint users will call.
    """

    raw_requirement: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="The user's original vague requirement",
    )
    context: str | None = Field(
        default=None,
        max_length=2000,
        description="Any additional context the user provides",
    )
    user_answers: dict[str, str] | None = Field(
        default=None,
        description="Answers to clarifying questions (from gap detection)",
    )
    styles: list[PromptStyle] = Field(
        default=[
            PromptStyle.CHAIN_OF_THOUGHT,
            PromptStyle.STRUCTURED,
        ],
        description="Prompt styles to generate and score",
    )
    target_llm: str = Field(
        default="gpt-4",
        description="Target LLM for the generated prompts",
    )
    session_id: str | None = None

    @field_validator("raw_requirement")
    @classmethod
    def clean_requirement(cls, v: str) -> str:
        return v.strip()


class OptimizationPipelineResponse(BaseModel):
    """Complete pipeline result with all stages included."""

    success: bool
    session_id: str | None
    raw_requirement: str

    # Stage results
    expanded_requirement: ExpandedRequirement
    scored_prompts: list[ScoredPrompt]
    best_prompt: ScoredPrompt
    recommended_style: PromptStyle
    recommendation_reason: str

    # Metrics
    total_processing_time_ms: float
    total_tokens_used: int
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

# ══════════════════════════════════════════════════════════════════
# PHASE 2: VISION CLARIFICATION SCHEMAS
# ══════════════════════════════════════════════════════════════════

class InnovationType(str, Enum):
    """
    The fundamental type of intelligence/value the product provides.
    This is the MOST important classification — drives all downstream
    reasoning and prevents collapse into wrong patterns.
    """
    AUTOMATION       = "automation"        # Removes manual work
    REASONING        = "reasoning"         # Thinks through complex problems
    RECOMMENDATION   = "recommendation"    # Suggests best options
    INTELLIGENCE     = "intelligence"      # Understands context deeply
    AUGMENTATION     = "augmentation"      # Enhances human capability
    PREDICTION       = "prediction"        # Forecasts future states
    ORCHESTRATION    = "orchestration"     # Coordinates complex workflows
    GENERATION       = "generation"        # Creates new content/artifacts
    UNKNOWN          = "unknown"           # Not yet determined


class AbstractionLevel(str, Enum):
    """
    The level at which the user is thinking about their product.
    Critical for preventing abstraction collapse.

    task    → "do X automatically"
    feature → "system that has X capability"
    product → "platform for X"
    system  → "architecture that reasons about X"
    paradigm→ "new way of thinking about X"
    """
    TASK      = "task"       # Lowest: single action automation
    FEATURE   = "feature"    # A capability within a product
    PRODUCT   = "product"    # A complete usable product
    SYSTEM    = "system"     # A complex interconnected system
    PARADIGM  = "paradigm"   # A new way of solving a class of problems


class InnovationTier(str, Enum):
    """
    How novel is this idea relative to existing solutions?

    incremental   → improves existing patterns (better dashboard)
    novel         → new combination of existing concepts
    transformative→ fundamentally new approach to a problem
    """
    INCREMENTAL   = "incremental"
    NOVEL         = "novel"
    TRANSFORMATIVE = "transformative"


class DriftRisk(str, Enum):
    """Risk level that semantic drift will occur during expansion."""
    LOW      = "low"       # < 20% concept loss risk
    MEDIUM   = "medium"    # 20-40% concept loss risk
    HIGH     = "high"      # 40-60% concept loss risk
    CRITICAL = "critical"  # > 60% concept loss risk


# ══════════════════════════════════════════════════════════════════
# VISION PROBE — Questions sent to user
# ══════════════════════════════════════════════════════════════════

class VisionQuestion(BaseModel):
    """A single targeted vision-probing question."""

    id: str = Field(
        description="Unique identifier for this question"
    )
    category: str = Field(
        description=(
            "Question category: user_transformation | intelligence_type | "
            "differentiation | domain_reasoning | output_value"
        )
    )
    question: str = Field(
        description="The actual question text shown to user"
    )
    purpose: str = Field(
        description="Why this question matters for semantic preservation"
    )
    example_answer: str | None = Field(
        default=None,
        description="Example of a good, specific answer"
    )
    is_critical: bool = Field(
        default=False,
        description="If True, answer significantly impacts drift prevention"
    )
    anti_pattern_hint: str | None = Field(
        default=None,
        description="What generic pattern this question prevents collapsing into"
    )


class VisionProbe(BaseModel):
    """
    Set of targeted questions generated for a specific input.
    Questions are adaptive — generated based on detected
    ambiguity and drift risk in the user's input.
    """

    session_id: str | None = None
    raw_input: str = Field(description="Original user input")
    questions: list[VisionQuestion] = Field(
        description="Adaptive list of vision-probing questions"
    )
    detected_ambiguities: list[str] = Field(
        description="Ambiguous concepts detected in the input"
    )
    initial_drift_risk: DriftRisk = Field(
        description="Estimated drift risk before answers"
    )
    dominant_pattern: str = Field(
        description=(
            "The generic pattern this input would collapse into "
            "without clarification"
        )
    )
    reasoning: str = Field(
        description="Why these specific questions were chosen"
    )


# ══════════════════════════════════════════════════════════════════
# SEMANTIC ANCHOR — Core preserved concept
# ══════════════════════════════════════════════════════════════════

class SemanticAnchor(BaseModel):
    """
    A single preserved conceptual element extracted from user answers.

    Semantic anchors are NON-NEGOTIABLE constraints that MUST survive
    through requirement expansion and prompt generation.
    They prevent individual concepts from being lost or diluted.
    """

    anchor_id: str = Field(description="Unique anchor identifier")
    concept: str = Field(
        description="The core concept being preserved"
    )
    anchor_type: str = Field(
        description=(
            "Type: intelligence_claim | reasoning_requirement | "
            "differentiation_marker | domain_constraint | "
            "user_transformation | output_requirement"
        )
    )
    original_quote: str = Field(
        description="Exact user words that generated this anchor"
    )
    preserved_meaning: str = Field(
        description="What this anchor means and must mean in the final prompt"
    )
    anti_patterns: list[str] = Field(
        description=(
            "Generic patterns this anchor MUST NOT be reduced to. "
            "e.g. 'do not reduce causal reasoning to rule-based scoring'"
        )
    )
    must_appear_in_prompt: bool = Field(
        default=True,
        description="Whether this concept must explicitly appear in final prompt"
    )
    importance: str = Field(
        default="high",
        description="Importance level: critical | high | medium"
    )


# ══════════════════════════════════════════════════════════════════
# VISION PROFILE — Complete extracted vision
# ══════════════════════════════════════════════════════════════════

class VisionProfile(BaseModel):
    """
    Complete vision profile extracted from user's answers
    to vision probe questions.

    This is the FOUNDATION for all downstream processing.
    Every subsequent stage receives this as a hard constraint.
    """

    session_id: str | None = None

    # Core identity
    primary_user: str = Field(
        description="Who is the primary user of this system"
    )
    core_transformation: str = Field(
        description=(
            "What fundamental change this creates: "
            "'User goes from X to Y'"
        )
    )
    intelligence_description: str = Field(
        description="What should feel intelligent about this system"
    )

    # Classification
    innovation_type: InnovationType = Field(
        description="The fundamental type of value provided"
    )
    abstraction_level: AbstractionLevel = Field(
        description="The level at which this product operates"
    )

    # Differentiation
    closest_existing_product: str = Field(
        description="The existing product most similar to this idea"
    )
    differentiation_claims: list[str] = Field(
        description="Specific ways this differs from closest existing product"
    )

    # Domain intelligence
    domain_constraints: list[str] = Field(
        description="Domain-specific rules/constraints the system must understand"
    )
    consequence_types: list[str] = Field(
        description="What risks or consequences the system reasons about"
    )

    # Semantic anchors
    semantic_anchors: list[SemanticAnchor] = Field(
        description="Extracted non-negotiable conceptual anchors"
    )

    # Confidence
    vision_confidence: float = Field(
        ge=0.0,
        le=100.0,
        description="Confidence in extracted vision (0-100)"
    )
    unanswered_critical_questions: list[str] = Field(
        description="Critical questions that were not answered"
    )

    # Drift prevention
    initial_drift_risk: DriftRisk
    dominant_collapse_pattern: str = Field(
        description=(
            "The generic pattern this idea would collapse into "
            "without anchors"
        )
    )


# ══════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS FOR VISION ENDPOINTS
# ══════════════════════════════════════════════════════════════════

class VisionProbeRequest(BaseModel):
    """Request to generate vision probe questions."""

    raw_input: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="The user's raw idea or requirement",
    )
    context: str | None = Field(
        default=None,
        max_length=2000,
        description="Any additional context provided",
    )
    session_id: str | None = None

    @field_validator("raw_input")
    @classmethod
    def clean_input(cls, v: str) -> str:
        return v.strip()


class VisionProbeResponse(BaseModel):
    """Response containing generated vision probe questions."""

    success: bool
    session_id: str | None
    raw_input: str
    probe: VisionProbe
    processing_time_ms: float
    tokens_used: int
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


class VisionAnswers(BaseModel):
    """User's answers to vision probe questions."""

    question_id: str = Field(description="ID of the question being answered")
    question_text: str = Field(description="The question text")
    answer: str = Field(
        min_length=1,
        description="User's answer",
    )


class VisionProfileRequest(BaseModel):
    """Request to extract vision profile from user answers."""

    raw_input: str = Field(
        ...,
        min_length=10,
        description="Original user input",
    )
    answers: list[VisionAnswers] = Field(
        ...,
        min_length=1,
        description="User's answers to vision probe questions",
    )
    probe: dict[str, Any] = Field(
        description="The original probe that generated the questions",
    )
    session_id: str | None = None


class VisionProfileResponse(BaseModel):
    """Response containing extracted vision profile."""

    success: bool
    session_id: str | None
    raw_input: str
    vision_profile: VisionProfile
    processing_time_ms: float
    tokens_used: int
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

# ══════════════════════════════════════════════════════════════════
# PHASE 3: SEMANTIC ANALYSIS SCHEMAS
# ══════════════════════════════════════════════════════════════════

# ─── Semantic Intent Schemas ───────────────────────────────────────

class ConceptNode(BaseModel):
    """
    A single node in the semantic concept map.
    Represents one meaningful concept extracted from the user's vision.
    """
    concept_id: str
    name: str = Field(description="Short concept name")
    description: str = Field(description="What this concept means in context")
    concept_type: str = Field(
        description=(
            "Type: core_intelligence | reasoning_mechanism | "
            "operational_flow | causal_mechanism | cognition_pipeline | "
            "user_value | domain_knowledge | system_behavior | "
            "output_characteristic | differentiator"
        )
    )
    is_novel: bool = Field(
        description="Whether this concept is novel vs standard"
    )
    collapse_risk: str = Field(
        description="Risk level this concept gets lost: low|medium|high|critical"
    )
    generic_equivalent: str | None = Field(
        default=None,
        description="The generic concept this would collapse into without preservation"
    )


class SemanticIntent(BaseModel):
    """
    Complete conceptual intent map extracted from the vision profile.

    Goes beyond keyword analysis to understand WHAT the system
    is conceptually trying to achieve at each level.
    """
    session_id: str | None = None

    # Core concept map
    concept_map: list[ConceptNode] = Field(
        description="All meaningful concepts extracted from vision"
    )

    # Operational Intelligence
    operational_execution_flows: list[str] = Field(
        default_factory=list,
        description="Identified execution flows and cognitive pipelines"
    )
    causal_inference_pathways: list[str] = Field(
        default_factory=list,
        description="How reasoning propagates and causal inferences are executed"
    )
    contextual_dependencies: list[str] = Field(
        default_factory=list,
        description="How contextual adaptation occurs and propagates"
    )

    # Intent summary
    one_line_intent: str = Field(
        description="Single sentence capturing the REAL intent"
    )
    conceptual_category: str = Field(
        description=(
            "What category of system this truly is — "
            "not just the domain, but the conceptual class"
        )
    )

    # Differentiation vector
    differentiation_vector: list[str] = Field(
        description=(
            "Ordered list of what makes this conceptually unique — "
            "from most to least important"
        )
    )

    # Standard category this resembles + how it differs
    closest_standard_category: str = Field(
        description="Most similar standard product category"
    )
    category_divergence: list[str] = Field(
        description="Specific ways this diverges from that category"
    )

    # Intent confidence
    intent_confidence: float = Field(
        ge=0.0, le=100.0,
        description="Confidence in extracted intent (0-100)"
    )
    ambiguity_flags: list[str] = Field(
        description="Remaining ambiguities that could cause drift"
    )

    # Reasoning trace
    extraction_reasoning: str = Field(
        description="How the intent was extracted — 2-3 sentences"
    )


class SemanticIntentRequest(BaseModel):
    """Request for semantic intent extraction."""
    raw_input: str = Field(..., min_length=10, max_length=5000)
    vision_profile: dict[str, Any] | None = Field(
        default=None,
        description="VisionProfile from Phase 2 (strongly recommended)"
    )
    session_id: str | None = None

    @field_validator("raw_input")
    @classmethod
    def clean(cls, v: str) -> str:
        return v.strip()


class SemanticIntentResponse(BaseModel):
    """Response with extracted semantic intent."""
    success: bool
    session_id: str | None
    raw_input: str
    semantic_intent: SemanticIntent
    processing_time_ms: float
    tokens_used: int
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


# ─── Innovation Analysis Schemas ───────────────────────────────────

class InnovationClaim(BaseModel):
    """A single innovation claim made by the product."""
    claim_id: str
    claim: str = Field(description="The specific innovation claim")
    claim_type: str = Field(
        description=(
            "Type: reasoning_advancement | ux_innovation | "
            "domain_synthesis | intelligence_depth | "
            "workflow_transformation | decision_support"
        )
    )
    is_verifiable: bool = Field(
        description="Whether this claim can be technically verified"
    )
    at_risk: bool = Field(
        description="Whether this claim is at risk of being lost"
    )
    risk_reason: str | None = Field(
        default=None,
        description="Why this claim is at risk of collapse"
    )
    preservation_instruction: str = Field(
        description="How to ensure this claim survives expansion"
    )


class InnovationProfile(BaseModel):
    """
    Complete innovation analysis of the product idea.

    Identifies what is genuinely novel, what tier of innovation
    it represents, and which concepts are most at risk of collapse.
    """
    session_id: str | None = None

    # Innovation tier
    innovation_tier: InnovationTier = Field(
        description="Overall innovation tier: incremental|novel|transformative"
    )
    tier_reasoning: str = Field(
        description="Why this tier was assigned"
    )

    # Claims
    innovation_claims: list[InnovationClaim] = Field(
        description="All identified innovation claims"
    )

    # Risk assessment
    at_risk_concepts: list[str] = Field(
        description="Concepts most likely to be lost during expansion"
    )
    safe_concepts: list[str] = Field(
        description="Concepts unlikely to be distorted"
    )

    # Collapse prevention
    anti_patterns: list[str] = Field(
        description=(
            "Explicit patterns that MUST be avoided during expansion. "
            "e.g. 'Do not reduce causal reasoning to classification'"
        )
    )
    preservation_rules: list[str] = Field(
        description=(
            "Positive rules for preserving innovation. "
            "e.g. 'Always include consequence reasoning in requirements'"
        )
    )

    # Comparison
    compared_to_existing: str = Field(
        description="How this compares to existing state-of-the-art"
    )
    innovation_score: float = Field(
        ge=0.0, le=100.0,
        description="Overall innovation score (0-100)"
    )


class InnovationAnalysisRequest(BaseModel):
    """Request for innovation analysis."""
    raw_input: str = Field(..., min_length=10, max_length=5000)
    vision_profile: dict[str, Any] | None = None
    semantic_intent: dict[str, Any] | None = None
    session_id: str | None = None

    @field_validator("raw_input")
    @classmethod
    def clean(cls, v: str) -> str:
        return v.strip()


class InnovationAnalysisResponse(BaseModel):
    """Response with innovation analysis."""
    success: bool
    session_id: str | None
    raw_input: str
    innovation_profile: InnovationProfile
    processing_time_ms: float
    tokens_used: int
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


# ─── Abstraction Classification Schemas ────────────────────────────

class AbstractionMismatch(BaseModel):
    """
    A detected mismatch between intended and actual abstraction levels.
    These are the exact points where semantic drift occurs.
    """
    mismatch_id: str
    intended_abstraction: str = Field(
        description="What abstraction level the user intended"
    )
    detected_abstraction: str = Field(
        description="What abstraction level the expansion used"
    )
    concept_affected: str = Field(
        description="Which concept was affected"
    )
    example: str = Field(
        description="Concrete example of the mismatch"
    )
    correction: str = Field(
        description="How to correct this mismatch"
    )
    severity: str = Field(
        description="Severity: low|medium|high|critical"
    )


class AbstractionClassification(BaseModel):
    """
    Complete abstraction level analysis.

    Detects what level the user is thinking at vs what level
    the system defaults to — this gap is the source of semantic drift.
    """
    session_id: str | None = None

    # Primary classification
    user_abstraction_level: AbstractionLevel = Field(
        description="The abstraction level the user is thinking at"
    )
    system_default_level: AbstractionLevel = Field(
        description="The level the system would default to without guidance"
    )
    abstraction_gap: int = Field(
        description=(
            "Gap between user and system level "
            "(positive = user thinks higher, negative = user thinks lower)"
        )
    )

    # Mismatch analysis
    detected_mismatches: list[AbstractionMismatch] = Field(
        description="Specific abstraction mismatches detected"
    )
    mismatch_count: int = Field(
        description="Total number of mismatches"
    )

    # Classification reasoning
    classification_reasoning: str = Field(
        description="How the abstraction level was determined"
    )
    correction_strategy: str = Field(
        description="Strategy to align system output with user's abstraction level"
    )

    # Guidance for expansion
    expansion_level_instructions: list[str] = Field(
        description=(
            "Specific instructions for how to expand at the correct "
            "abstraction level"
        )
    )


class AbstractionClassificationRequest(BaseModel):
    """Request for abstraction classification."""
    raw_input: str = Field(..., min_length=10, max_length=5000)
    vision_profile: dict[str, Any] | None = None
    semantic_intent: dict[str, Any] | None = None
    session_id: str | None = None

    @field_validator("raw_input")
    @classmethod
    def clean(cls, v: str) -> str:
        return v.strip()


class AbstractionClassificationResponse(BaseModel):
    """Response with abstraction classification."""
    success: bool
    session_id: str | None
    raw_input: str
    classification: AbstractionClassification
    processing_time_ms: float
    tokens_used: int
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


# ─── Full Semantic Analysis Schema ─────────────────────────────────

class FullSemanticAnalysisRequest(BaseModel):
    """
    Runs all three semantic analyses in one call:
    1. Semantic Intent Extraction
    2. Innovation Analysis
    3. Abstraction Classification

    This is the recommended single endpoint for Phase 3.
    """
    raw_input: str = Field(..., min_length=10, max_length=5000)
    vision_profile: dict[str, Any] | None = Field(
        default=None,
        description="VisionProfile from Phase 2 — strongly recommended"
    )
    session_id: str | None = None

    @field_validator("raw_input")
    @classmethod
    def clean(cls, v: str) -> str:
        return v.strip()


class FullSemanticAnalysisResponse(BaseModel):
    """Combined response from all three semantic analyses."""
    success: bool
    session_id: str | None
    raw_input: str
    semantic_intent: SemanticIntent
    innovation_profile: InnovationProfile
    classification: AbstractionClassification
    total_processing_time_ms: float
    total_tokens_used: int
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

# ══════════════════════════════════════════════════════════════════
# PHASE 4: DRIFT DETECTION + CONCEPT PRESERVATION SCHEMAS
# ══════════════════════════════════════════════════════════════════

# ─── Drift Detection Schemas ───────────────────────────────────────

class DriftWarning(BaseModel):
    """A single detected instance of semantic drift."""

    warning_id: str
    drift_type: str = Field(
        description=(
            "Type of drift detected: "
            "concept_loss | abstraction_collapse | "
            "innovation_dilution | generic_pattern_replacement | "
            "anti_pattern_violation | anchor_violation"
        )
    )
    severity: str = Field(
        description="Severity: low | medium | high | critical"
    )
    original_concept: str = Field(
        description="The original concept that was distorted"
    )
    drifted_to: str = Field(
        description="What the concept became in the expansion"
    )
    location: str = Field(
        description=(
            "Where in the expansion this drift occurs "
            "(e.g. 'functional_requirements', 'overview')"
        )
    )
    evidence: str = Field(
        description="Specific quote from the expansion showing drift"
    )
    correction_required: str = Field(
        description="What must be done to correct this drift"
    )
    blocked_anti_pattern: str | None = Field(
        default=None,
        description="The specific anti-pattern that was violated"
    )


class DriftAnalysis(BaseModel):
    """Complete drift analysis for an expansion."""

    session_id: str | None = None

    # Overall metrics
    drift_score: float = Field(
        ge=0.0, le=100.0,
        description="Overall drift score (0=no drift, 100=total drift)"
    )
    drift_risk: DriftRisk
    can_proceed: bool = Field(
        description="Whether expansion is safe to use without correction"
    )

    # Concept tracking
    preserved_concepts: list[str] = Field(
        description="Concepts from vision that survived expansion"
    )
    lost_concepts: list[str] = Field(
        description="Concepts that were lost or severely diluted"
    )
    diluted_concepts: list[dict[str, str]] = Field(
        description=(
            "Concepts that survived but were weakened. "
            "Each dict has 'concept' and 'how_diluted' keys."
        )
    )
    introduced_generic_patterns: list[str] = Field(
        description="Generic patterns that appeared but should not have"
    )

    # Specific warnings
    warnings: list[DriftWarning] = Field(
        description="All detected drift warnings, sorted by severity"
    )
    critical_warnings_count: int
    total_warnings_count: int

    # Anti-pattern violations
    violated_anti_patterns: list[str] = Field(
        description="Anti-patterns that were violated by the expansion"
    )

    # Anchor violations
    violated_anchors: list[str] = Field(
        description="Semantic anchors that were not preserved"
    )

    # Recommendation
    overall_assessment: str = Field(
        description="Plain-language assessment of drift severity"
    )
    correction_strategy: str = Field(
        description="Strategy to fix the detected drift"
    )


class DriftAnalysisRequest(BaseModel):
    """Request to analyze drift in an expansion."""

    raw_input: str = Field(..., min_length=10, max_length=5000)
    expansion_text: str = Field(
        ...,
        min_length=20,
        description="The expanded requirement text or full prompt"
    )
    vision_profile: dict[str, Any] = Field(
        ...,
        description="VisionProfile from Phase 2 (REQUIRED for drift detection)"
    )
    semantic_intent: dict[str, Any] | None = Field(
        default=None,
        description="SemanticIntent from Phase 3 (recommended)"
    )
    innovation_profile: dict[str, Any] | None = Field(
        default=None,
        description="InnovationProfile from Phase 3 (recommended)"
    )
    session_id: str | None = None


class DriftAnalysisResponse(BaseModel):
    """Response from drift analysis."""

    success: bool
    session_id: str | None
    drift_analysis: DriftAnalysis
    processing_time_ms: float
    tokens_used: int
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


# ─── Concept Preservation Schemas ──────────────────────────────────

class ConceptPreservationRule(BaseModel):
    """
    A single enforceable rule for preserving a concept.

    These rules become NEGATIVE and POSITIVE constraints injected
    into all downstream LLM calls (expander, optimizer, refiner).
    """

    rule_id: str
    rule_type: str = Field(
        description=(
            "Type: must_include | must_not_become | "
            "must_preserve_meaning | must_use_phrase | "
            "must_explain_reasoning"
        )
    )
    concept: str = Field(description="The concept being protected")
    rule_text: str = Field(
        description="The actual rule text to inject into prompts"
    )
    enforcement_level: str = Field(
        description="Enforcement: hard | soft"
    )
    source_anchor_id: str | None = Field(
        default=None,
        description="Source semantic anchor ID this rule is derived from"
    )
    examples: list[str] = Field(
        default_factory=list,
        description="Examples of compliant and non-compliant outputs"
    )


class PreservationRuleSet(BaseModel):
    """
    Complete set of preservation rules for an expansion run.

    This is what gets injected into the expander/optimizer
    as hard constraints to prevent semantic drift.
    """

    session_id: str | None = None

    rules: list[ConceptPreservationRule]
    rule_count: int

    # Categorization
    must_include_rules: list[str] = Field(
        description="Concepts that MUST appear in any expansion"
    )
    must_not_become_rules: list[str] = Field(
        description="What the expansion MUST NOT become"
    )
    reasoning_requirements: list[str] = Field(
        description="Reasoning patterns the expansion must follow"
    )

    # Injection format
    system_prompt_injection: str = Field(
        description=(
            "Pre-formatted text ready to inject into LLM system prompts"
        )
    )
    user_prompt_injection: str = Field(
        description=(
            "Pre-formatted text ready to inject into LLM user messages"
        )
    )

    # Metadata
    enforcement_strategy: str = Field(
        description="How rules will be enforced"
    )


class PreservationRulesRequest(BaseModel):
    """Request to build preservation rules."""

    vision_profile: dict[str, Any] = Field(
        ...,
        description="VisionProfile from Phase 2 (REQUIRED)"
    )
    semantic_intent: dict[str, Any] | None = None
    innovation_profile: dict[str, Any] | None = None
    abstraction_classification: dict[str, Any] | None = None
    session_id: str | None = None


class PreservationRulesResponse(BaseModel):
    """Response with built preservation rules."""

    success: bool
    session_id: str | None
    rule_set: PreservationRuleSet
    processing_time_ms: float
    tokens_used: int
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


# ─── Reasoning Alignment Schemas ───────────────────────────────────

class ReasoningAlignment(BaseModel):
    """
    Pre-expansion reasoning alignment instructions.

    Generated BEFORE expansion to direct the expander to reason
    at the MECHANISM LEVEL — grounding every intelligence claim
    in an operational execution flow or causal inference pathway.
    """

    session_id: str | None = None

    # Core alignment instructions
    primary_reasoning_mode: str = Field(
        description=(
            "Primary reasoning mode the expander should use: "
            "concept_first | architecture_first | feature_first | "
            "outcome_first | reasoning_first | "
            "operational_first | causal_execution_first | mechanism_first"
        )
    )
    abstraction_target: AbstractionLevel = Field(
        description="The target abstraction level for expansion"
    )

    # Alignment guidance
    must_reason_about: list[str] = Field(
        description="Aspects the expander must explicitly reason about at mechanism level"
    )
    must_avoid_reasoning_about: list[str] = Field(
        description="Aspects the expander must NOT focus on"
    )

    # Concept guards
    protected_concepts: list[str] = Field(
        description="Concepts that must appear with their original meaning AND mechanism"
    )
    protected_operational_flows: list[str] = Field(
        default_factory=list,
        description=(
            "Execution flow sequences (A → B → C) that must survive into the expansion. "
            "These are the core operational identity of the system."
        )
    )
    forbidden_collapses: list[str] = Field(
        description="Specific generic patterns to avoid"
    )

    # Expansion approach
    expansion_approach: str = Field(
        description="Mechanism-grounded approach for the expansion"
    )
    expansion_anti_approach: str = Field(
        description="Approach to explicitly avoid"
    )

    # Pre-flight checklist
    pre_expansion_checklist: list[str] = Field(
        description=(
            "Checklist items (mechanism level) the expander must verify before output"
        )
    )

    # Ready-to-inject instruction block
    alignment_instructions: str = Field(
        description=(
            "Complete formatted operational instruction block ready to inject "
            "into the expander's system prompt"
        )
    )


class ReasoningAlignmentRequest(BaseModel):
    """Request to generate reasoning alignment."""

    raw_input: str = Field(..., min_length=10, max_length=5000)
    vision_profile: dict[str, Any] = Field(
        ...,
        description="VisionProfile from Phase 2 (REQUIRED)"
    )
    semantic_intent: dict[str, Any] | None = None
    innovation_profile: dict[str, Any] | None = None
    abstraction_classification: dict[str, Any] | None = None
    session_id: str | None = None


class ReasoningAlignmentResponse(BaseModel):
    """Response with reasoning alignment instructions."""

    success: bool
    session_id: str | None
    alignment: ReasoningAlignment
    processing_time_ms: float
    tokens_used: int
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )