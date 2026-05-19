"""
Reusable UI Components for the Streamlit Dashboard.

IMPORTANT: Streamlit does NOT allow nested expanders.
This file uses tabs, st.info, st.markdown instead of
nested expanders everywhere.
"""

import streamlit as st


# ══════════════════════════════════════════════════════════════════
# SCORE DISPLAY COMPONENTS
# ══════════════════════════════════════════════════════════════════

def render_score_badge(score: float, grade: str) -> None:
    """
    Renders a large score badge with color-coded background.

    Args:
        score: Numeric score 0-100
        grade: Letter grade (A+, A, B, C, D, F)
    """
    color = _grade_to_color(grade)
    emoji = _grade_to_emoji(grade)

    st.markdown(
        f"""
        <div style="
            background: {color};
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            color: white;
            margin-bottom: 10px;
        ">
            <div style="font-size: 3rem; font-weight: bold;">{score:.1f}</div>
            <div style="font-size: 1.5rem;">{emoji} Grade: {grade}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">Quality Score / 100</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dimension_scores(dimensions: list[dict]) -> None:
    """
    Renders a progress-bar breakdown of scoring dimensions.

    Args:
        dimensions: List of dicts with 'dimension', 'score', 'weight', 'feedback'
    """
    if not dimensions:
        st.info("No dimension scores available.")
        return

    st.markdown("**📊 Dimension Breakdown**")

    for dim in dimensions:
        name     = dim.get("dimension", "Unknown")
        score    = float(dim.get("score", 0))
        weight   = float(dim.get("weight", 0))
        feedback = dim.get("feedback", "")

        if score >= 8:
            label_color = "green"
        elif score >= 6:
            label_color = "orange"
        else:
            label_color = "red"

        col_name, col_score, col_bar = st.columns([2, 1, 3])

        with col_name:
            st.markdown(f"**{name}**")
            st.caption(f"Weight: {int(weight * 100)}%")

        with col_score:
            st.markdown(
                f"<span style='color:{label_color}; font-size:1.1rem; "
                f"font-weight:bold'>{score:.1f}/10</span>",
                unsafe_allow_html=True,
            )

        with col_bar:
            st.progress(score / 10)
            if feedback:
                st.caption(feedback)


def render_hallucination_risk_inline(risk: str, reasons: list[str]) -> None:
    """
    Renders hallucination risk WITHOUT an expander.
    Uses st.info / st.warning / st.error instead.

    Args:
        risk: 'low', 'medium', or 'high'
        reasons: List of risk reason strings
    """
    risk_lower = risk.lower() if risk else "medium"

    configs = {
        "low":    ("success", "🟢", "Low Hallucination Risk"),
        "medium": ("warning", "🟡", "Medium Hallucination Risk"),
        "high":   ("error",   "🔴", "High Hallucination Risk"),
    }
    fn_name, emoji, label = configs.get(risk_lower, configs["medium"])

    # Show risk badge
    getattr(st, fn_name)(f"{emoji} **{label}**")

    # Show reasons as bullet list if any
    if reasons:
        for reason in reasons:
            st.caption(f"• {reason}")


def render_hallucination_risk(risk: str, reasons: list[str]) -> None:
    """
    Public alias — always renders inline (no expander).
    Kept for backward compatibility with existing imports.
    """
    render_hallucination_risk_inline(risk, reasons)


def render_production_badge(is_ready: bool) -> None:
    """
    Renders a production readiness badge.

    Args:
        is_ready: Whether the prompt meets production quality bar
    """
    if is_ready:
        st.success("✅ **Production Ready** — Meets quality threshold (≥70/100)")
    else:
        st.warning("⚠️ **Not Production Ready** — Quality improvements recommended")


# ══════════════════════════════════════════════════════════════════
# PROMPT DISPLAY COMPONENTS
# ══════════════════════════════════════════════════════════════════

def render_prompt_card(
    prompt: dict,
    show_score: bool = True,
    is_best: bool = False,
    index: int = 0,
) -> None:
    """
    Renders a complete prompt card with style, text, and score.

    NO nested expanders — score details use st.tabs instead.

    Args:
        prompt: Prompt dict with style, prompt_text, score fields
        show_score: Whether to show the quality score section
        is_best: Whether to highlight this as the best prompt
        index: Card index for unique widget keys
    """
    style_label  = prompt.get("style_label", prompt.get("style", "Unknown"))
    prompt_text  = prompt.get("prompt_text", "")
    score_data   = prompt.get("score", {})
    overall      = score_data.get("overall_score", 0)
    grade        = score_data.get("grade", "?")
    word_count   = prompt.get("word_count", len(prompt_text.split()))
    token_est    = prompt.get("estimated_tokens", word_count * 2)

    # ── Header ────────────────────────────────────────────────────
    header = f"**{style_label}**"
    if is_best:
        header = f"⭐ {header} — Best Prompt"
    st.markdown(f"### {header}")

    # ── Metadata metrics ──────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Score", f"{overall:.1f}/100")
    with m2:
        color = _grade_to_color(grade)
        st.markdown(
            f"<div style='background:{color}; color:white; padding:4px 10px; "
            f"border-radius:6px; text-align:center; font-weight:bold; "
            f"font-size:1.1rem; margin-top:4px;'>{grade}</div>",
            unsafe_allow_html=True,
        )
    with m3:
        st.metric("Words", word_count)
    with m4:
        st.metric("~Tokens", token_est)

    # ── Prompt text area ──────────────────────────────────────────
    st.text_area(
        label="Prompt Text (copy with Ctrl+A → Ctrl+C)",
        value=prompt_text,
        height=220,
        key=f"prompt_text_{index}_{style_label}",
        help="Select all text and copy to use this prompt",
    )

    # ── Score details — use tabs NOT expander ─────────────────────
    if show_score and score_data:
        st.markdown(
            f"**📊 Score Details** — Grade: **{grade}** "
            f"({overall:.1f}/100) &nbsp;|&nbsp; "
            f"Production Ready: "
            f"{'✅ Yes' if score_data.get('is_production_ready') else '❌ No'}"
        )

        score_tab1, score_tab2, score_tab3 = st.tabs([
            "📊 Dimensions",
            "✅ Strengths & Weaknesses",
            "🎯 Feedback",
        ])

        with score_tab1:
            render_dimension_scores(score_data.get("dimensions", []))

        with score_tab2:
            col_s, col_w = st.columns(2)

            with col_s:
                strengths = score_data.get("strengths", [])
                st.markdown("**✅ Strengths**")
                if strengths:
                    for s in strengths:
                        st.markdown(f"• {s}")
                else:
                    st.caption("None identified")

            with col_w:
                weaknesses = score_data.get("weaknesses", [])
                st.markdown("**⚠️ Weaknesses**")
                if weaknesses:
                    for w in weaknesses:
                        st.markdown(f"• {w}")
                else:
                    st.caption("None identified")

        with score_tab3:
            # Hallucination risk — inline, no expander
            render_hallucination_risk_inline(
                score_data.get("hallucination_risk", "medium"),
                score_data.get("hallucination_reasons", []),
            )

            st.markdown("")
            feedback = score_data.get("overall_feedback", "")
            if feedback:
                st.info(f"💬 **Overall Feedback:** {feedback}")

    st.divider()


# ══════════════════════════════════════════════════════════════════
# INTENT DISPLAY COMPONENTS
# ══════════════════════════════════════════════════════════════════

def render_intent_card(intent: dict) -> None:
    """
    Renders the intent analysis result.
    No expanders — uses columns and markdown only.

    Args:
        intent: Intent dict from /analysis/intent response
    """
    st.markdown("### 🎯 Intent Analysis")

    # ── Top metrics ───────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        domain = intent.get("domain", "unknown").replace("_", " ").title()
        st.metric("Domain", domain)

    with col2:
        complexity = intent.get("complexity", "medium")
        emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
            complexity, "🟡"
        )
        st.metric("Complexity", f"{emoji} {complexity.title()}")

    with col3:
        confidence = intent.get("confidence", "medium")
        emoji2 = {"high": "✅", "medium": "⚡", "low": "⚠️"}.get(
            confidence, "⚡"
        )
        st.metric("Confidence", f"{emoji2} {confidence.title()}")

    # ── Primary goal ──────────────────────────────────────────────
    primary = intent.get("primary_goal", "Not identified")
    st.markdown(f"**🎯 Primary Goal:** {primary}")

    # ── Secondary goals ───────────────────────────────────────────
    secondary = intent.get("secondary_goals", [])
    if secondary:
        st.markdown("**📌 Secondary Goals:**")
        for goal in secondary:
            st.markdown(f"&nbsp;&nbsp;&nbsp;• {goal}")

    # ── Two columns: constraints + keywords ───────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        constraints = intent.get("constraints", [])
        if constraints:
            st.markdown("**🚧 Constraints:**")
            for c in constraints:
                st.markdown(f"&nbsp;&nbsp;&nbsp;• {c}")

        audience = intent.get("target_audience")
        if audience:
            st.markdown(f"**👥 Target Audience:** {audience}")

    with col_b:
        keywords = intent.get("keywords", [])
        if keywords:
            st.markdown("**🔑 Key Terms:**")
            tags_html = " ".join(
                f'<span style="background:#e8f4fd; padding:2px 8px; '
                f'border-radius:10px; margin:2px; font-size:0.85rem;">'
                f'{kw}</span>'
                for kw in keywords[:10]
            )
            st.markdown(tags_html, unsafe_allow_html=True)

    # ── Reasoning — use st.info not expander ──────────────────────
    reasoning = intent.get("raw_reasoning", "")
    if reasoning:
        st.markdown("**🔍 AI Reasoning:**")
        st.info(reasoning)


# ══════════════════════════════════════════════════════════════════
# GAP ANALYSIS COMPONENTS
# ══════════════════════════════════════════════════════════════════

def render_gap_analysis(gap_analysis: dict) -> dict[str, str]:
    """
    Renders gap analysis results with input fields.
    No expanders — uses st.container and columns.

    Args:
        gap_analysis: Gap analysis dict from API

    Returns:
        Dict of {question: answer} from user inputs
    """
    st.markdown("### 🔍 Requirement Gap Analysis")

    score       = gap_analysis.get("completeness_score", 0)
    can_proceed = gap_analysis.get("can_proceed", True)

    # Completeness meter
    col1, col2 = st.columns([1, 3])
    with col1:
        color = "🟢" if score >= 70 else "🟡" if score >= 40 else "🔴"
        st.metric("Completeness", f"{color} {score:.0f}/100")
    with col2:
        st.progress(score / 100)
        st.caption(gap_analysis.get("recommendation", ""))

    if not can_proceed:
        st.error(
            "❌ **Critical gaps detected.** "
            "Please answer the questions below before proceeding."
        )

    # Missing requirements
    missing = gap_analysis.get("missing_requirements", [])
    user_answers: dict[str, str] = {}

    if missing:
        st.markdown(f"**Found {len(missing)} gaps to address:**")

        severity_order = {"critical": 0, "important": 1, "optional": 2}
        sorted_missing = sorted(
            missing,
            key=lambda x: severity_order.get(x.get("severity", "optional"), 2),
        )

        for i, req in enumerate(sorted_missing):
            severity = req.get("severity", "optional")
            sev_emoji = {
                "critical":  "🔴",
                "important": "🟡",
                "optional":  "🔵",
            }.get(severity, "🔵")

            category   = req.get("category", "General")
            question   = req.get("question", "")
            why_needed = req.get("why_needed", "")
            example    = req.get("example_answer", "")
            default    = req.get("default_assumption", "")

            # Show each gap inline — no expander
            st.markdown(
                f"{sev_emoji} **[{severity.upper()}] {category}:** {question}"
            )
            if why_needed:
                st.caption(f"Why needed: {why_needed}")

            answer = st.text_input(
                label="Your answer",
                placeholder=example or default or "Type your answer...",
                key=f"gap_render_{i}_{category}",
                help=f"Example: {example}" if example else None,
                label_visibility="collapsed",
            )

            if default and not answer:
                st.caption(f"💡 Will assume: *{default}*")

            if answer:
                user_answers[question] = answer

            st.markdown("---")

    return user_answers


# ══════════════════════════════════════════════════════════════════
# EXPANDED REQUIREMENT COMPONENT
# ══════════════════════════════════════════════════════════════════

def render_expanded_requirement(expanded: dict) -> None:
    """
    Renders the expanded requirement.
    No nested expanders — uses tabs and st.info only.

    Args:
        expanded: Expanded requirement dict from /optimize/expand
    """
    st.markdown(f"**{expanded.get('title', 'Untitled')}**")
    st.info(expanded.get("overview", ""))

    tab1, tab2, tab3 = st.tabs([
        "📌 Requirements",
        "🎯 Scope & Goals",
        "✅ Success Criteria",
    ])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Functional Requirements**")
            func_reqs = expanded.get("functional_requirements", [])
            if func_reqs:
                for req in func_reqs:
                    st.markdown(f"• {req}")
            else:
                st.caption("None specified")

        with col2:
            st.markdown("**Non-Functional Requirements**")
            non_func = expanded.get("non_functional_requirements", [])
            if non_func:
                for req in non_func:
                    st.markdown(f"• {req}")
            else:
                st.caption("None specified")

    with tab2:
        st.markdown("**In Scope:**")
        st.markdown(expanded.get("scope", "Not defined"))

        out = expanded.get("out_of_scope", [])
        if out:
            st.markdown("**Out of Scope:**")
            for item in out:
                st.markdown(f"• ~~{item}~~")

        objectives = expanded.get("objectives", [])
        if objectives:
            st.markdown("**Objectives:**")
            for obj in objectives:
                st.markdown(f"• {obj}")

        # Assumptions — st.info instead of nested expander
        assumptions = expanded.get("assumptions_made", [])
        if assumptions:
            st.markdown("**💡 Assumptions Made:**")
            assumption_text = "\n".join(f"• {a}" for a in assumptions)
            st.info(assumption_text)

    with tab3:
        criteria = expanded.get("success_criteria", [])
        if criteria:
            for criterion in criteria:
                st.markdown(f"✓ {criterion}")
        else:
            st.caption("No success criteria defined")

        approach = expanded.get("suggested_approach", "")
        if approach:
            st.markdown("**Suggested Approach:**")
            st.success(approach)

        col_a, col_b = st.columns(2)
        with col_a:
            complexity = expanded.get("estimated_complexity", "medium")
            emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
                complexity, "🟡"
            )
            st.metric("Complexity", f"{emoji} {complexity.title()}")
        with col_b:
            confidence = expanded.get("expansion_confidence", "medium")
            emoji2 = {"high": "✅", "medium": "⚡", "low": "⚠️"}.get(
                confidence, "⚡"
            )
            st.metric("Confidence", f"{emoji2} {confidence.title()}")


# ══════════════════════════════════════════════════════════════════
# REFINEMENT COMPONENTS
# ══════════════════════════════════════════════════════════════════

def render_refinement_progress(iterations: list[dict]) -> None:
    """
    Renders a visual timeline of refinement iterations.
    No expanders — uses columns and st.info.

    Args:
        iterations: List of iteration dicts
    """
    if not iterations:
        st.info("No refinement iterations recorded.")
        return

    st.markdown("#### 🔄 Refinement Progress")

    # Build score list
    scores = []
    labels = []

    if iterations:
        scores.append(iterations[0]["score_before"])
        labels.append("Initial")

    for it in iterations:
        scores.append(it["score_after"])
        labels.append(f"Iter {it['iteration']}")

    # Metrics row
    cols = st.columns(len(scores))
    for i, (col, score, label) in enumerate(zip(cols, scores, labels)):
        with col:
            delta = None
            if i > 0:
                delta = f"+{score - scores[i-1]:.1f}"
            st.metric(label=label, value=f"{score:.1f}", delta=delta)

    st.progress(min(scores[-1] / 100, 1.0))
    st.caption(
        f"Total improvement: +{scores[-1] - scores[0]:.1f} points"
    )

    # Iteration details — use columns NOT expanders
    for it in iterations:
        st.markdown(
            f"**Iteration {it['iteration']}:** "
            f"{it['grade_before']} ({it['score_before']:.0f}) → "
            f"{it['grade_after']} ({it['score_after']:.0f}) "
            f"[+{it['improvement']:.1f} points] "
            f"| RAG docs: {it.get('rag_docs_retrieved', 0)}"
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Before", f"{it['score_before']:.1f}")
        with col2:
            st.metric("After", f"{it['score_after']:.1f}")
        with col3:
            st.metric("Improvement", f"+{it['improvement']:.1f}")
        with col4:
            st.metric("Time", f"{it.get('time_ms', 0):.0f}ms")
        st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# ANALYTICS COMPONENTS
# ══════════════════════════════════════════════════════════════════

def render_analytics_overview(analytics: dict) -> None:
    """
    Renders the analytics overview metrics.
    No expanders needed here.

    Args:
        analytics: Analytics dict from /history/analytics
    """
    st.markdown("### 📊 Usage Analytics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sessions", analytics.get("total_sessions", 0))
    with col2:
        st.metric("Total Prompts", analytics.get("total_prompts", 0))
    with col3:
        st.metric(
            "Avg Best Score",
            f"{analytics.get('average_best_score', 0):.1f}/100",
        )
    with col4:
        st.metric(
            "Production Ready",
            f"{analytics.get('production_ready_rate', 0):.1f}%",
        )

    col5, col6 = st.columns(2)

    with col5:
        st.markdown("**Prompt Style Distribution**")
        style_dist = analytics.get("style_distribution", {})
        if style_dist:
            max_count = max(style_dist.values()) if style_dist else 1
            for style, count in sorted(
                style_dist.items(), key=lambda x: x[1], reverse=True
            ):
                label = style.replace("_", " ").title()
                st.progress(
                    count / max_count,
                    text=f"{label}: {count}",
                )
        else:
            st.info("No data yet")

    with col6:
        st.markdown("**Grade Distribution**")
        grade_dist  = analytics.get("grade_distribution", {})
        grade_order = ["A+", "A", "B", "C", "D", "F"]
        if grade_dist:
            for grade in grade_order:
                count = grade_dist.get(grade, 0)
                if count > 0:
                    color = _grade_to_color(grade)
                    st.markdown(
                        f'<div style="display:flex; align-items:center; '
                        f'gap:10px; margin:4px 0;">'
                        f'<span style="background:{color}; color:white; '
                        f'padding:2px 10px; border-radius:4px; '
                        f'font-weight:bold; min-width:40px; '
                        f'text-align:center;">{grade}</span>'
                        f'<span>{count} sessions</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No data yet")


# ══════════════════════════════════════════════════════════════════
# SIDEBAR STATUS COMPONENT
# ══════════════════════════════════════════════════════════════════

def render_sidebar_status() -> None:
    """
    Renders the backend + LLM status in the sidebar.
    Used by all pages for consistent status display.
    """
    from frontend.utils.api_client import check_health, check_readiness

    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/brain--v2.png",
            width=70,
        )
        st.title("🧠 AI Req Intel")
        st.caption("Requirement Intelligence System")
        st.divider()

        st.markdown("**🗺️ Navigation**")
        st.page_link("app.py",                        label="🏠 Home")
        st.page_link("pages/1_🔍_Analyzer.py",        label="🔍 Analyzer")
        st.page_link("pages/2_⚡_Optimizer.py",       label="⚡ Optimizer")
        st.page_link("pages/3_✨_Refiner.py",         label="✨ Refiner")
        st.page_link("pages/4_📚_History.py",         label="📚 History")
        st.divider()

        st.markdown("**⚙️ Status**")

        @st.cache_data(ttl=15)
        def _health():
            return check_health()

        @st.cache_data(ttl=15)
        def _ready():
            return check_readiness()

        health = _health()
        ready  = _ready()

        if health:
            uptime = health.get("uptime_seconds", 0)
            st.success(f"✅ Backend Online ({uptime:.0f}s uptime)")
        else:
            st.error("❌ Backend Offline")
            st.caption("`uvicorn backend.main:app --reload`")

        if ready:
            checks = ready.get("checks", {})
            if checks.get("llm_api_key") == "configured":
                provider = ready.get("llm_provider", "?").upper()
                model    = ready.get("active_model", "?")
                st.success(f"✅ {provider} Ready")
                st.caption(f"Model: {model}")
            else:
                st.warning("⚠️ API Key Missing")

        st.divider()
        st.caption("v1.0.0 | Module 5")


# ══════════════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ══════════════════════════════════════════════════════════════════

def _grade_to_color(grade: str) -> str:
    """Maps letter grade to hex color."""
    return {
        "A+": "#1a7a4a",
        "A":  "#27ae60",
        "B":  "#2980b9",
        "C":  "#f39c12",
        "D":  "#e67e22",
        "F":  "#e74c3c",
        "?":  "#95a5a6",
    }.get(grade, "#95a5a6")


def _grade_to_emoji(grade: str) -> str:
    """Maps letter grade to emoji."""
    return {
        "A+": "🏆",
        "A":  "⭐",
        "B":  "👍",
        "C":  "📝",
        "D":  "⚠️",
        "F":  "❌",
        "?":  "❓",
    }.get(grade, "❓")


# ══════════════════════════════════════════════════════════════════
# VISION PROFILE COMPONENTS
# ══════════════════════════════════════════════════════════════════

def render_vision_profile(vision_profile: dict) -> None:
    """
    Renders the extracted VisionProfile in a structured layout.
    Shows innovation type, abstraction level, semantic anchors,
    differentiation claims, and drift risk.
    """
    st.markdown("### 🧠 Vision Profile")

    # ── Top classification metrics ─────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        itype = vision_profile.get("innovation_type", "unknown")
        itype_emoji = {
            "automation":     "⚙️",
            "reasoning":      "🧠",
            "recommendation": "💡",
            "intelligence":   "🔮",
            "augmentation":   "🤝",
            "prediction":     "📈",
            "orchestration":  "🎯",
            "generation":     "✨",
            "unknown":        "❓",
        }.get(itype, "❓")
        st.metric("Innovation Type", f"{itype_emoji} {itype.title()}")

    with c2:
        alevel = vision_profile.get("abstraction_level", "product")
        level_emoji = {
            "task":    "📌",
            "feature": "🔧",
            "product": "📦",
            "system":  "🏗️",
            "paradigm": "🌐",
        }.get(alevel, "📦")
        st.metric("Abstraction Level", f"{level_emoji} {alevel.title()}")

    with c3:
        risk = vision_profile.get("initial_drift_risk", "medium")
        risk_emoji = {
            "low": "🟢", "medium": "🟡",
            "high": "🔴", "critical": "💀",
        }.get(risk, "🟡")
        st.metric("Drift Risk", f"{risk_emoji} {risk.title()}")

    with c4:
        confidence = vision_profile.get("vision_confidence", 0)
        st.metric("Vision Confidence", f"{confidence:.0f}/100")

    st.divider()

    # ── Core vision statements ─────────────────────────────────────
    primary_user = vision_profile.get("primary_user", "")
    if primary_user:
        st.markdown(f"**👤 Primary User:** {primary_user}")

    core_transform = vision_profile.get("core_transformation", "")
    if core_transform:
        st.info(f"🔄 **Core Transformation:** {core_transform}")

    intelligence = vision_profile.get("intelligence_description", "")
    if intelligence:
        st.success(f"🧠 **What Feels Intelligent:** {intelligence}")

    # ── Differentiation claims ─────────────────────────────────────
    diff_claims = vision_profile.get("differentiation_claims", [])
    if diff_claims:
        st.markdown("**🎯 Differentiation Claims:**")
        for claim in diff_claims:
            st.markdown(f"&nbsp;&nbsp;&nbsp;✦ {claim}")

    # ── Closest existing + collapse warning ───────────────────────
    closest = vision_profile.get("closest_existing_product", "")
    collapse = vision_profile.get("dominant_collapse_pattern", "")
    if closest or collapse:
        col_a, col_b = st.columns(2)
        with col_a:
            if closest:
                st.markdown(f"**📌 Closest Existing:** {closest}")
        with col_b:
            if collapse:
                st.warning(f"⚠️ **Would collapse into:** {collapse}")

    # ── Semantic anchors ───────────────────────────────────────────
    anchors = vision_profile.get("semantic_anchors", [])
    if anchors:
        st.divider()
        st.markdown(f"**⚓ Semantic Anchors ({len(anchors)} extracted)**")
        st.caption(
            "These concepts are preserved through ALL pipeline stages."
        )
        render_semantic_anchors(anchors)


def render_semantic_anchors(anchors: list[dict]) -> None:
    """
    Renders semantic anchors as expandable cards.
    Each shows concept, meaning, and anti-patterns.
    """
    importance_colors = {
        "critical": "#e74c3c",
        "high":     "#e67e22",
        "medium":   "#3498db",
    }

    for anchor in anchors:
        importance  = anchor.get("importance", "high")
        concept     = anchor.get("concept", "")
        meaning     = anchor.get("preserved_meaning", "")
        anti        = anchor.get("anti_patterns", [])
        anchor_type = anchor.get("anchor_type", "")
        quote       = anchor.get("original_quote", "")
        color       = importance_colors.get(importance, "#3498db")

        st.markdown(
            f"""
            <div style="border-left: 4px solid {color};
                        padding: 10px 15px;
                        margin: 8px 0;
                        background: #f8f9fa;
                        border-radius: 0 6px 6px 0;">
                <div style="font-weight: bold; color: {color};">
                    [{importance.upper()}] ⚓ {concept}
                </div>
                <div style="font-size: 0.85rem; color: #555; margin-top: 4px;">
                    Type: {anchor_type}
                </div>
                <div style="margin-top: 6px;">
                    <strong>Preserved Meaning:</strong> {meaning}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if quote:
            st.caption(f'💬 From user: "{quote}"')

        if anti:
            anti_text = " | ".join(
                [f"❌ {a}" for a in anti[:3]]
            )
            st.caption(f"Must NOT become: {anti_text}")

        st.markdown("")


# ══════════════════════════════════════════════════════════════════
# SEMANTIC INTENT COMPONENTS
# ══════════════════════════════════════════════════════════════════

def render_semantic_intent(semantic_intent: dict) -> None:
    """
    Renders semantic intent analysis including concept map
    with collapse risk indicators.
    """
    st.markdown("### 🗺️ Semantic Intent Map")

    one_line = semantic_intent.get("one_line_intent", "")
    if one_line:
        st.info(f"**Real Intent:** {one_line}")

    col1, col2 = st.columns(2)
    with col1:
        category = semantic_intent.get("conceptual_category", "")
        if category:
            st.markdown(f"**Conceptual Category:** {category}")
        std_cat = semantic_intent.get("closest_standard_category", "")
        if std_cat:
            st.markdown(f"**Closest Standard:** {std_cat}")

    with col2:
        confidence = semantic_intent.get("intent_confidence", 0)
        st.metric("Intent Confidence", f"{confidence:.0f}/100")
        st.progress(confidence / 100)

    # ── Differentiation vector ─────────────────────────────────────
    diff_vec = semantic_intent.get("differentiation_vector", [])
    if diff_vec:
        st.markdown("**🎯 Differentiation Vector:**")
        for i, d in enumerate(diff_vec, 1):
            st.markdown(f"&nbsp;&nbsp;{i}. {d}")

    # ── Concept map ────────────────────────────────────────────────
    concept_map = semantic_intent.get("concept_map", [])
    if concept_map:
        st.divider()
        st.markdown(
            f"**📊 Concept Map ({len(concept_map)} concepts)**"
        )
        render_concept_map(concept_map)

    # ── Ambiguity flags ────────────────────────────────────────────
    flags = semantic_intent.get("ambiguity_flags", [])
    if flags:
        st.divider()
        st.warning(
            "**⚠️ Remaining Ambiguities:**\n"
            + "\n".join(f"• {f}" for f in flags)
        )


def render_concept_map(concept_map: list[dict]) -> None:
    """
    Renders concept map as a risk-sorted table with color coding.
    """
    risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    risk_colors = {
        "critical": "🔴",
        "high":     "🟠",
        "medium":   "🟡",
        "low":      "🟢",
    }
    novel_icon  = "✨"
    stable_icon = "📦"

    sorted_concepts = sorted(
        concept_map,
        key=lambda c: risk_order.get(
            c.get("collapse_risk", "low"), 3
        ),
    )

    for concept in sorted_concepts:
        name        = concept.get("name", "")
        description = concept.get("description", "")
        ctype       = concept.get("concept_type", "")
        is_novel    = concept.get("is_novel", False)
        risk        = concept.get("collapse_risk", "low")
        generic_eq  = concept.get("generic_equivalent")

        risk_dot  = risk_colors.get(risk, "🟡")
        novel_dot = novel_icon if is_novel else stable_icon

        col1, col2, col3 = st.columns([2, 3, 2])
        with col1:
            st.markdown(
                f"{risk_dot} {novel_dot} **{name}**"
            )
            st.caption(ctype)
        with col2:
            st.markdown(description)
        with col3:
            if generic_eq and risk in ("high", "critical"):
                st.caption(f"⚠️ → *{generic_eq}*")
            elif risk == "low":
                st.caption("✅ Low collapse risk")

        st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# INNOVATION PROFILE COMPONENTS
# ══════════════════════════════════════════════════════════════════

def render_innovation_profile(innovation_profile: dict) -> None:
    """
    Renders innovation analysis including tier, claims,
    anti-patterns, and preservation rules.
    """
    st.markdown("### 🚀 Innovation Analysis")

    tier  = innovation_profile.get("innovation_tier", "incremental")
    score = innovation_profile.get("innovation_score", 0)

    tier_config = {
        "incremental":    ("🔵", "#2980b9", "Incremental"),
        "novel":          ("🟣", "#8e44ad", "Novel"),
        "transformative": ("🔥", "#e74c3c", "Transformative"),
    }
    emoji, color, label = tier_config.get(
        tier, ("🔵", "#2980b9", "Incremental")
    )

    # ── Tier badge ─────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="background:{color}; color:white; padding:12px 20px;
                    border-radius:8px; text-align:center; margin-bottom:12px;">
            <div style="font-size:1.8rem;">{emoji}</div>
            <div style="font-size:1.2rem; font-weight:bold;">
                {label} Innovation
            </div>
            <div style="font-size:0.9rem; opacity:0.9;">
                Score: {score:.0f}/100
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    reasoning = innovation_profile.get("tier_reasoning", "")
    if reasoning:
        st.caption(reasoning)

    st.progress(score / 100)

    col1, col2 = st.columns(2)

    with col1:
        # At-risk concepts
        at_risk = innovation_profile.get("at_risk_concepts", [])
        if at_risk:
            st.markdown("**⚠️ At-Risk Concepts:**")
            for concept in at_risk:
                st.markdown(f"• 🔴 {concept}")

        # Safe concepts
        safe = innovation_profile.get("safe_concepts", [])
        if safe:
            st.markdown("**✅ Safe Concepts:**")
            for concept in safe[:4]:
                st.markdown(f"• 🟢 {concept}")

    with col2:
        # Anti-patterns
        anti_patterns = innovation_profile.get("anti_patterns", [])
        if anti_patterns:
            st.markdown("**🚫 Anti-Patterns to Avoid:**")
            for ap in anti_patterns[:5]:
                st.markdown(
                    f'<div style="background:#fdecea; padding:4px 8px; '
                    f'border-radius:4px; margin:3px 0; font-size:0.85rem;">'
                    f'❌ {ap}</div>',
                    unsafe_allow_html=True,
                )

        # Preservation rules
        rules = innovation_profile.get("preservation_rules", [])
        if rules:
            st.markdown("**✅ Preservation Rules:**")
            for rule in rules[:5]:
                st.markdown(
                    f'<div style="background:#eafaf1; padding:4px 8px; '
                    f'border-radius:4px; margin:3px 0; font-size:0.85rem;">'
                    f'✅ {rule}</div>',
                    unsafe_allow_html=True,
                )

    # ── Innovation claims ──────────────────────────────────────────
    claims = innovation_profile.get("innovation_claims", [])
    if claims:
        st.divider()
        st.markdown(f"**💡 Innovation Claims ({len(claims)}):**")
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            at_risk_flag = "🔴" if claim.get("at_risk") else "🟢"
            st.markdown(
                f"{at_risk_flag} **{claim.get('claim', '')}**"
            )
            instr = claim.get("preservation_instruction", "")
            if instr:
                st.caption(f"   → {instr}")


# ══════════════════════════════════════════════════════════════════
# DRIFT WARNING COMPONENTS
# ══════════════════════════════════════════════════════════════════

def render_drift_analysis(drift_analysis: dict) -> None:
    """
    Renders drift analysis with score, risk level, warnings,
    and concept tracking.
    """
    st.markdown("### 🌊 Semantic Drift Analysis")

    drift_score = drift_analysis.get("drift_score", 0)
    drift_risk  = drift_analysis.get("drift_risk", "medium")
    can_proceed = drift_analysis.get("can_proceed", True)

    risk_config = {
        "low":      ("🟢", "success", "Low Drift"),
        "medium":   ("🟡", "warning", "Medium Drift"),
        "high":     ("🔴", "error",   "High Drift"),
        "critical": ("💀", "error",   "Critical Drift"),
    }
    emoji, fn_name, label = risk_config.get(
        drift_risk, ("🟡", "warning", "Medium Drift")
    )

    # ── Summary metrics ────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Drift Score", f"{drift_score:.1f}/100")
    with m2:
        getattr(st, fn_name)(f"{emoji} {label}")
    with m3:
        cw = drift_analysis.get("critical_warnings_count", 0)
        tw = drift_analysis.get("total_warnings_count", 0)
        st.metric("Warnings", f"{cw} critical / {tw} total")
    with m4:
        st.metric(
            "Can Proceed",
            "✅ Yes" if can_proceed else "❌ Needs Fix"
        )

    st.progress(max(0.0, min(1.0, drift_score / 100)))

    assessment = drift_analysis.get("overall_assessment", "")
    if assessment:
        if drift_score < 40:
            st.success(f"✅ {assessment}")
        elif drift_score < 60:
            st.warning(f"⚠️ {assessment}")
        else:
            st.error(f"❌ {assessment}")

    # ── Concept tracking ───────────────────────────────────────────
    preserved = drift_analysis.get("preserved_concepts", [])
    lost      = drift_analysis.get("lost_concepts", [])
    diluted   = drift_analysis.get("diluted_concepts", [])

    if preserved or lost or diluted:
        st.divider()
        col_p, col_l, col_d = st.columns(3)

        with col_p:
            st.markdown(f"**✅ Preserved ({len(preserved)})**")
            for c in preserved[:5]:
                st.markdown(f"• {c}")

        with col_l:
            st.markdown(f"**❌ Lost ({len(lost)})**")
            for c in lost[:5]:
                st.markdown(
                    f'<span style="color:#e74c3c">• {c}</span>',
                    unsafe_allow_html=True,
                )

        with col_d:
            st.markdown(f"**⚠️ Diluted ({len(diluted)})**")
            for d in diluted[:5]:
                if isinstance(d, dict):
                    st.markdown(f"• {d.get('concept', '')}")
                    st.caption(d.get("how_diluted", ""))

    # ── Warnings ───────────────────────────────────────────────────
    warnings = drift_analysis.get("warnings", [])
    if warnings:
        st.divider()
        st.markdown(
            f"**⚠️ Drift Warnings ({len(warnings)}):**"
        )

        sev_colors = {
            "critical": "#e74c3c",
            "high":     "#e67e22",
            "medium":   "#f39c12",
            "low":      "#27ae60",
        }
        sev_emojis = {
            "critical": "💀", "high": "🔴",
            "medium": "🟡", "low": "🟢",
        }

        for w in warnings[:8]:
            severity = w.get("severity", "medium")
            color    = sev_colors.get(severity, "#f39c12")
            emoji_s  = sev_emojis.get(severity, "🟡")

            st.markdown(
                f"""
                <div style="border-left: 3px solid {color};
                            padding: 8px 12px; margin: 6px 0;
                            background: #fafafa;
                            border-radius: 0 4px 4px 0;">
                    <div style="font-weight:bold; color:{color};">
                        {emoji_s} [{severity.upper()}]
                        {w.get("drift_type","").replace("_"," ").title()}
                    </div>
                    <div style="margin-top:4px;">
                        <strong>{w.get("original_concept","")}</strong>
                        → <em>{w.get("drifted_to","")}</em>
                    </div>
                    <div style="font-size:0.82rem; color:#666; margin-top:4px;">
                        Evidence: "{w.get("evidence","")[:150]}"
                    </div>
                    <div style="font-size:0.82rem; margin-top:4px;">
                        Fix: {w.get("correction_required","")}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════
# ABSTRACTION CLASSIFICATION COMPONENT
# ══════════════════════════════════════════════════════════════════

def render_abstraction_classification(classification: dict) -> None:
    """
    Renders abstraction level classification showing the gap
    between user intent level and system default level.
    """
    st.markdown("### 📐 Abstraction Classification")

    user_level   = classification.get("user_abstraction_level", "product")
    system_level = classification.get("system_default_level", "feature")
    gap          = classification.get("abstraction_gap", 0)

    level_order = ["task", "feature", "product", "system", "paradigm"]
    level_emoji = {
        "task": "📌", "feature": "🔧",
        "product": "📦", "system": "🏗️", "paradigm": "🌐",
    }

    col1, col2, col3 = st.columns(3)
    with col1:
        emoji = level_emoji.get(user_level, "📦")
        st.metric(
            "User Thinks At",
            f"{emoji} {user_level.title()}"
        )
    with col2:
        emoji2 = level_emoji.get(system_level, "🔧")
        st.metric(
            "System Defaults To",
            f"{emoji2} {system_level.title()}"
        )
    with col3:
        gap_color = "normal" if gap == 0 else "off"
        delta_str = (
            f"+{gap} levels" if gap > 0
            else f"{gap} levels" if gap < 0
            else "No gap"
        )
        st.metric(
            "Abstraction Gap",
            f"{abs(gap)} levels",
            delta=delta_str if gap != 0 else None,
        )

    # Gap severity
    if gap == 0:
        st.success("✅ No abstraction gap — system will expand at correct level")
    elif abs(gap) == 1:
        st.warning(
            f"⚠️ Minor abstraction gap ({gap:+d} levels) — "
            "some concepts may be slightly diluted"
        )
    elif abs(gap) == 2:
        st.error(
            f"🔴 Significant abstraction gap ({gap:+d} levels) — "
            "semantic drift correction active"
        )
    else:
        st.error(
            f"💀 Critical abstraction gap ({gap:+d} levels) — "
            "full correction required"
        )

    # Visual level bar
    st.markdown("**Abstraction Level Scale:**")
    cols = st.columns(5)
    for i, level in enumerate(level_order):
        with cols[i]:
            is_user   = (level == user_level)
            is_system = (level == system_level)
            is_both   = is_user and is_system

            if is_both:
                label  = "✅ Both"
                bg     = "#27ae60"
                color  = "white"
            elif is_user:
                label  = "👤 User"
                bg     = "#2980b9"
                color  = "white"
            elif is_system:
                label  = "🤖 System"
                bg     = "#e74c3c"
                color  = "white"
            else:
                label  = ""
                bg     = "#ecf0f1"
                color  = "#666"

            st.markdown(
                f"""
                <div style="background:{bg}; color:{color};
                            padding:8px 4px; border-radius:6px;
                            text-align:center; font-size:0.8rem;">
                    <div style="font-weight:bold;">
                        {level_emoji.get(level,'')} {level.title()}
                    </div>
                    <div style="font-size:0.75rem;">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Mismatches
    mismatches = classification.get("detected_mismatches", [])
    if mismatches:
        st.divider()
        st.markdown(
            f"**🔍 Detected Mismatches ({len(mismatches)}):**"
        )
        for m in mismatches[:4]:
            severity = m.get("severity", "medium")
            sev_emoji = {
                "critical": "💀", "high": "🔴",
                "medium": "🟡", "low": "🟢"
            }.get(severity, "🟡")

            st.markdown(
                f"{sev_emoji} **{m.get('concept_affected','')}:** "
                f"{m.get('example','')}"
            )
            correction = m.get("correction", "")
            if correction:
                st.caption(f"   Fix: {correction}")