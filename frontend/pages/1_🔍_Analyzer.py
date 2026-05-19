"""
Analyzer Page - Intent Analysis & Gap Detection.

Flow:
1. User enters requirement → clicks Analyze
2. Intent analysis + gap detection results shown
3. User fills in gap answer text inputs
4. User clicks Proceed to Optimizer
5. Requirement + gap answers passed via session state
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent.parent.parent))

from frontend.utils.api_client import full_analysis
from frontend.utils.ui_components import (
    render_intent_card,
    render_sidebar_status,
)

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Analyzer | AI Req Intel",
    page_icon="🔍",
    layout="wide",
)

render_sidebar_status()

# ─── Page Header ──────────────────────────────────────────────────
st.title("🔍 Requirement Analyzer")
st.markdown(
    "Paste your vague requirement below. "
    "The AI will analyze your **intent** and identify **missing information**."
)
st.divider()

# ─── Initialize Session State ─────────────────────────────────────
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analyzed_requirement" not in st.session_state:
    st.session_state.analyzed_requirement = ""

if "analyzed_context" not in st.session_state:
    st.session_state.analyzed_context = ""

# ─── STEP 1: Input Form ───────────────────────────────────────────
st.markdown("### Step 1: Enter Your Requirement")

raw_requirement = st.text_area(
    label="📝 Your Requirement",
    placeholder=(
        "Describe what you want to build or accomplish...\n\n"
        "Examples:\n"
        "• Build me an app that tracks my daily habits\n"
        "• I need a REST API for my e-commerce project\n"
        "• Create a dashboard for my sales team"
    ),
    height=130,
    key="raw_req_input",
    value=st.session_state.analyzed_requirement,
)

context = st.text_area(
    label="📌 Additional Context (optional)",
    placeholder=(
        "Any extra info: team size, tech stack, deadline, constraints..."
    ),
    height=70,
    key="context_input",
    value=st.session_state.analyzed_context,
)

analyze_clicked = st.button(
    "🔍 Analyze Requirement",
    type="primary",
    use_container_width=True,
)

# ─── Run Analysis ─────────────────────────────────────────────────
if analyze_clicked:
    if not raw_requirement.strip():
        st.warning("⚠️ Please enter a requirement first.")
        st.stop()

    if len(raw_requirement.strip()) < 10:
        st.warning("⚠️ Requirement is too short. Please add more detail.")
        st.stop()

    with st.spinner(
        "🤖 Analyzing your requirement... (10-25 seconds)"
    ):
        result = full_analysis(
            raw_requirement=raw_requirement.strip(),
            context=context.strip() if context.strip() else None,
        )

    if result:
        # Save to session state
        st.session_state.analysis_result = result
        st.session_state.analyzed_requirement = raw_requirement.strip()
        st.session_state.analyzed_context = context.strip()
        st.success("✅ Analysis complete! Scroll down to review and fill in gaps.")
    else:
        st.error("❌ Analysis failed. Check that the backend is running.")
        st.stop()

# ─── STEP 2: Show Analysis Results ────────────────────────────────
if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    st.divider()

    # ── Metrics row ───────────────────────────────────────────────
    gap_analysis = result.get("gap_analysis", {})
    completeness = gap_analysis.get("completeness_score", 0)
    intent       = result.get("intent", {})

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Completeness", f"{completeness:.0f}/100")
    with mc2:
        st.metric(
            "Domain",
            intent.get("domain", "?").replace("_", " ").title(),
        )
    with mc3:
        st.metric("Complexity", intent.get("complexity", "?").title())
    with mc4:
        st.metric(
            "Tokens Used",
            result.get("total_tokens_used", 0),
        )

    st.divider()

    # ── Intent Analysis ───────────────────────────────────────────
    st.markdown("### Step 2: Review Intent Analysis")
    render_intent_card(intent)

    st.divider()

    # ── Gap Analysis + Answer Form ────────────────────────────────
    st.markdown("### Step 3: Fill In Missing Information")
    st.caption(
        "The AI found these gaps in your requirement. "
        "Fill in what you can — **critical ones matter most**. "
        "Leave optional ones blank if unsure."
    )

    missing = gap_analysis.get("missing_requirements", [])

    # Sort: critical first
    severity_order = {"critical": 0, "important": 1, "optional": 2}
    missing_sorted = sorted(
        missing,
        key=lambda x: severity_order.get(x.get("severity", "optional"), 2),
    )

    # Count by severity for summary
    critical_count  = sum(1 for m in missing if m.get("severity") == "critical")
    important_count = sum(1 for m in missing if m.get("severity") == "important")
    optional_count  = sum(1 for m in missing if m.get("severity") == "optional")

    if missing_sorted:
        st.markdown(
            f"Found **{len(missing_sorted)} gaps**: "
            f"🔴 {critical_count} critical &nbsp;|&nbsp; "
            f"🟡 {important_count} important &nbsp;|&nbsp; "
            f"🔵 {optional_count} optional",
            unsafe_allow_html=True,
        )
        st.markdown("")

        # Render each gap as a labeled input
        for i, gap in enumerate(missing_sorted):
            severity     = gap.get("severity", "optional")
            category     = gap.get("category", "General")
            question     = gap.get("question", "")
            why_needed   = gap.get("why_needed", "")
            example      = gap.get("example_answer", "")
            default_asmp = gap.get("default_assumption", "")

            sev_emoji = {
                "critical":  "🔴",
                "important": "🟡",
                "optional":  "🔵",
            }.get(severity, "🔵")

            sev_label = {
                "critical":  "CRITICAL",
                "important": "IMPORTANT",
                "optional":  "OPTIONAL",
            }.get(severity, "OPTIONAL")

            # Label shown above text input
            input_label = (
                f"{sev_emoji} [{sev_label}] {category}: {question}"
            )

            # Help text shown on hover
            help_text = why_needed
            if example:
                help_text += f"\n\nExample answer: {example}"
            if default_asmp:
                help_text += f"\n\nWill assume: {default_asmp}"

            st.text_input(
                label=input_label,
                placeholder=example or default_asmp or "Type your answer here...",
                key=f"gap_{i}_{severity}_{category}",
                help=help_text,
            )

    else:
        st.success(
            "✅ No significant gaps found! "
            "Your requirement is fairly complete."
        )

    # ── Recommendation ────────────────────────────────────────────
    recommendation = gap_analysis.get("recommendation", "")
    if recommendation:
        st.info(f"💡 **Recommendation:** {recommendation}")

    st.divider()

    # ─── STEP 4: Proceed Button ───────────────────────────────────
    st.markdown("### Step 4: Proceed to Optimizer")

    can_proceed  = gap_analysis.get("can_proceed", True)
    critical_unanswered = []

    # Check which critical gaps are still unanswered
    for i, gap in enumerate(missing_sorted):
        severity = gap.get("severity", "optional")
        category = gap.get("category", "General")
        key      = f"gap_{i}_{severity}_{category}"
        answer   = st.session_state.get(key, "").strip()

        if severity == "critical" and not answer:
            critical_unanswered.append(gap.get("question", ""))

    # Warn if critical gaps unanswered
    if critical_unanswered:
        st.warning(
            f"⚠️ You have **{len(critical_unanswered)} critical gap(s)** "
            "still unanswered. You can still proceed, but prompt quality "
            "may be lower.\n\n"
            + "\n".join(f"• {q}" for q in critical_unanswered)
        )

    # Status summary before button
    answered_count = 0
    for i, gap in enumerate(missing_sorted):
        severity = gap.get("severity", "optional")
        category = gap.get("category", "General")
        key      = f"gap_{i}_{severity}_{category}"
        if st.session_state.get(key, "").strip():
            answered_count += 1

    if missing_sorted:
        st.caption(
            f"You answered {answered_count}/{len(missing_sorted)} gaps. "
            "More answers = better prompts."
        )

    # ── The Proceed Button ────────────────────────────────────────
    proceed_clicked = st.button(
        "⚡ Proceed to Optimizer →",
        type="primary",
        use_container_width=True,
    )

    if proceed_clicked:
        # Collect all gap answers from session state
        user_answers = {}
        for i, gap in enumerate(missing_sorted):
            severity = gap.get("severity", "optional")
            category = gap.get("category", "General")
            question = gap.get("question", "")
            key      = f"gap_{i}_{severity}_{category}"
            answer   = st.session_state.get(key, "").strip()

            if answer:
                user_answers[question] = answer
            elif gap.get("default_assumption"):
                # Auto-fill default assumption if user left blank
                user_answers[question] = (
                    f"[Assumed] {gap['default_assumption']}"
                )

        # Store everything in session state for Optimizer page
        st.session_state.optimizer_requirement = (
            st.session_state.analyzed_requirement
        )
        st.session_state.optimizer_context = (
            st.session_state.analyzed_context
        )
        st.session_state.optimizer_answers  = user_answers
        st.session_state.optimizer_intent   = intent
        st.session_state.optimizer_domain   = intent.get("domain", "general")

        st.success(
            f"✅ Passing your requirement + "
            f"{len(user_answers)} answer(s) to the Optimizer..."
        )

        # Navigate to optimizer
        st.switch_page("pages/2_⚡_Optimizer.py")

    # ── Debug ─────────────────────────────────────────────────────
    with st.expander("🔧 Raw API Response (Debug)"):
        st.json(result)