"""
Vision Clarification Page.

STAGE 0 of the semantic pipeline.

Flow:
1. User enters their idea
2. System generates adaptive probe questions
3. User answers targeted questions
4. System extracts VisionProfile with semantic anchors
5. User proceeds to Optimizer with vision context
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent.parent.parent))

from frontend.utils.api_client import (
    generate_vision_probe,
    extract_vision_profile,
)
from frontend.utils.ui_components import (
    render_sidebar_status,
    render_vision_profile,
)

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Vision Clarifier | AI Req Intel",
    page_icon="🧠",
    layout="wide",
)

render_sidebar_status()


# ══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS — defined FIRST before any usage
# ══════════════════════════════════════════════════════════════════

def _render_question_input(q: dict, index: int = 0) -> None:
    """
    Renders a single vision question with its answer text area.

    Must be defined at the top of the file before any Streamlit
    page rendering code calls it.

    Args:
        q:     Question dict with id, question, purpose, etc.
        index: Fallback index if question has no id
    """
    qid         = q.get("id", f"q{index}")
    question    = q.get("question", "")
    purpose     = q.get("purpose", "")
    example     = q.get("example_answer", "")
    anti_hint   = q.get("anti_pattern_hint", "")
    is_critical = q.get("is_critical", False)
    category    = q.get("category", "general")

    # Category emoji mapping
    cat_emoji = {
        "user_transformation": "👤",
        "intelligence_type":   "🧠",
        "differentiation":     "🎯",
        "domain_reasoning":    "🏢",
        "output_value":        "📋",
    }.get(category, "❓")

    prefix = "🔴" if is_critical else "🔵"

    # Question label
    st.markdown(f"{prefix} {cat_emoji} **{question}**")

    # Supporting info as captions (no nested widgets)
    if purpose:
        st.caption(f"Why this matters: {purpose}")

    if anti_hint:
        st.caption(f"⚠️ Prevents collapse into: *{anti_hint}*")

    # Answer input — key must be unique per question
    st.text_area(
        label="Your answer",
        placeholder=(
            f"Example: {example}"
            if example
            else "Type your specific answer here..."
        ),
        key=f"vq_{qid}",
        height=85,
        label_visibility="collapsed",
    )

    st.markdown("")  # Spacing between questions


def _count_answered(questions: list[dict]) -> int:
    """Counts how many questions have non-empty answers."""
    return sum(
        1 for q in questions
        if st.session_state.get(
            f"vq_{q.get('id', '')}", ""
        ).strip()
    )


def _collect_answers(questions: list[dict]) -> list[dict]:
    """
    Collects all answered questions from session state.

    Returns:
        List of answer dicts with question_id, question_text, answer
    """
    answers = []
    for q in questions:
        qid   = q.get("id", "")
        qtext = q.get("question", "")
        ans   = st.session_state.get(f"vq_{qid}", "").strip()
        if ans:
            answers.append({
                "question_id":   qid,
                "question_text": qtext,
                "answer":        ans,
            })
    return answers


# ══════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════

st.title("🧠 Vision Clarifier")
st.markdown(
    "Describe your idea. The AI will ask **targeted questions** to "
    "understand your **real intent** and prevent your innovation from "
    "being collapsed into a generic software pattern."
)

st.info(
    "💡 **Why this matters:** When you say *'AI decision intelligence system'*, "
    "standard tools turn it into *'ML dashboard'*. "
    "This step extracts your **semantic anchors** to prevent that.",
    icon="🧠",
)

st.divider()

# ══════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════

if "vision_probe"   not in st.session_state:
    st.session_state.vision_probe   = None
if "vision_profile" not in st.session_state:
    st.session_state.vision_profile = None
if "vision_input"   not in st.session_state:
    st.session_state.vision_input   = ""
if "vision_context" not in st.session_state:
    st.session_state.vision_context = ""

# ══════════════════════════════════════════════════════════════════
# STEP 1: Input Form
# ══════════════════════════════════════════════════════════════════

st.markdown("### Step 1: Describe Your Idea")

raw_input = st.text_area(
    label="💡 Your Idea or Product Concept",
    value=st.session_state.vision_input,
    placeholder=(
        "Describe what you want to build — even if it's vague.\n\n"
        "Examples:\n"
        "• AI system that helps doctors make better treatment decisions\n"
        "• Platform that reasons about supply chain risks in real-time\n"
        "• Tool that understands legal documents and explains consequences\n"
        "• System that augments financial analyst judgment with AI reasoning"
    ),
    height=140,
    key="vision_input_field",
)

context_val = st.text_area(
    label="📌 Additional Context (optional)",
    value=st.session_state.vision_context,
    placeholder="Domain, target users, constraints, what makes it unique...",
    height=69,
    key="vision_context_field",
)

probe_clicked = st.button(
    "🔍 Generate Clarification Questions",
    type="primary",
    use_container_width=True,
)

# ─── Handle probe generation ───────────────────────────────────────
if probe_clicked:
    if not raw_input.strip():
        st.warning("⚠️ Please enter your idea first.")
        st.stop()

    if len(raw_input.strip()) < 15:
        st.warning(
            "⚠️ Please provide a more detailed description "
            "(at least 15 characters)."
        )
        st.stop()

    with st.spinner(
        "🤖 Analyzing your idea and generating targeted questions... "
        "(10-20 seconds)"
    ):
        result = generate_vision_probe(
            raw_input=raw_input.strip(),
            context=context_val.strip() if context_val.strip() else None,
        )

    if result and result.get("success"):
        # Save to session state
        st.session_state.vision_probe   = result.get("probe", {})
        st.session_state.vision_input   = raw_input.strip()
        st.session_state.vision_context = context_val.strip()
        st.session_state.vision_profile = None  # Reset on new probe

        probe            = st.session_state.vision_probe
        initial_risk     = probe.get("initial_drift_risk", "medium")
        collapse_pattern = probe.get("dominant_pattern", "")
        question_count   = len(probe.get("questions", []))

        # Status messages
        st.success(
            f"✅ {question_count} clarification questions generated!"
        )

        risk_messages = {
            "low": (
                "success",
                "🟢 Low drift risk — your idea is fairly clear.",
            ),
            "medium": (
                "warning",
                "🟡 Medium drift risk — answering questions will improve results.",
            ),
            "high": (
                "error",
                "🔴 High drift risk — clarification is important for best results.",
            ),
            "critical": (
                "error",
                "💀 Critical drift risk — without clarification, "
                "your idea WILL be misinterpreted.",
            ),
        }
        fn_name, msg = risk_messages.get(
            initial_risk, ("warning", "🟡 Clarification recommended.")
        )
        getattr(st, fn_name)(msg)

        if collapse_pattern:
            st.error(
                f"⚠️ **Without clarification, your idea would become:** "
                f"*{collapse_pattern}*"
            )

        # Show detected ambiguities inline (no expander)
        ambiguities = probe.get("detected_ambiguities", [])
        if ambiguities:
            st.markdown("**🔍 Detected Ambiguities:**")
            for amb in ambiguities:
                st.markdown(f"&nbsp;&nbsp;• {amb}")

    elif result is not None:
        st.error(
            "❌ Failed to generate questions. "
            "Check that the backend is running."
        )

# ══════════════════════════════════════════════════════════════════
# STEP 2: Answer Questions
# ══════════════════════════════════════════════════════════════════

if st.session_state.vision_probe:
    probe     = st.session_state.vision_probe
    questions = probe.get("questions", [])

    # ── Edge case: no questions generated ─────────────────────────
    if not questions:
        st.divider()
        st.success(
            "✅ No clarification questions needed — "
            "your idea is clear enough to proceed!"
        )
        if st.button(
            "⚡ Proceed to Optimizer →",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.optimizer_requirement = (
                st.session_state.vision_input
            )
            st.session_state.optimizer_context       = (
                st.session_state.vision_context
            )
            st.session_state.optimizer_vision_profile = None
            st.session_state.optimizer_answers        = {}
            st.switch_page("pages/2_⚡_Optimizer.py")
        st.stop()

    st.divider()
    st.markdown("### Step 2: Answer the Clarification Questions")
    st.caption(
        "Your answers become **semantic anchors** preserved through "
        "all AI reasoning stages. Answer at least the 🔴 critical ones."
    )

    # Separate by criticality
    critical_qs = [q for q in questions if q.get("is_critical")]
    other_qs    = [q for q in questions if not q.get("is_critical")]

    # ── Critical questions ────────────────────────────────────────
    if critical_qs:
        st.markdown(
            f"**🔴 Critical Questions ({len(critical_qs)}) — "
            "answer these for best results:**"
        )
        for i, q in enumerate(critical_qs):
            _render_question_input(q, index=i)

    # ── Optional questions ────────────────────────────────────────
    if other_qs:
        if critical_qs:
            st.markdown("---")
        st.markdown(
            f"**🔵 Additional Questions ({len(other_qs)}) — optional:**"
        )
        for i, q in enumerate(other_qs):
            _render_question_input(q, index=i + len(critical_qs))

    # ══════════════════════════════════════════════════════════════
    # STEP 3: Extract Profile
    # ══════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("### Step 3: Extract Your Vision Profile")

    # Live answer counts
    answered_count    = _count_answered(questions)
    critical_answered = _count_answered(critical_qs)
    missing_critical  = len(critical_qs) - critical_answered

    col_stat, col_btn = st.columns([2, 1])

    with col_stat:
        st.metric(
            "Questions Answered",
            f"{answered_count} / {len(questions)}",
        )
        if missing_critical > 0:
            st.warning(
                f"⚠️ {missing_critical} critical question(s) unanswered — "
                "results may be less precise."
            )
        elif answered_count > 0:
            st.success("✅ All critical questions answered!")

    with col_btn:
        extract_clicked = st.button(
            "🧠 Extract Vision Profile",
            type="primary",
            use_container_width=True,
            disabled=(answered_count == 0),
            help=(
                "Answer at least one question to enable extraction."
                if answered_count == 0
                else "Click to extract your semantic vision profile."
            ),
        )

    # ── Handle extraction ─────────────────────────────────────────
    if extract_clicked:
        answers_list = _collect_answers(questions)

        if not answers_list:
            st.warning("⚠️ Please answer at least one question.")
            st.stop()

        with st.spinner(
            "🧠 Extracting semantic anchors from your answers... "
            "(10-25 seconds)"
        ):
            result = extract_vision_profile(
                raw_input=st.session_state.vision_input,
                answers=answers_list,
                probe=probe,
            )

        if result and result.get("success"):
            vp = result.get("vision_profile", {})
            st.session_state.vision_profile = vp

            confidence  = vp.get("vision_confidence", 0)
            anchors     = vp.get("semantic_anchors", [])
            drift_risk  = vp.get("initial_drift_risk", "medium")

            # Confidence-based feedback
            if confidence >= 70:
                st.success(
                    f"✅ Vision Profile extracted with **high confidence** "
                    f"({confidence:.0f}/100) | "
                    f"{len(anchors)} semantic anchors created"
                )
            elif confidence >= 40:
                st.warning(
                    f"⚠️ Vision Profile extracted with **moderate confidence** "
                    f"({confidence:.0f}/100) | {len(anchors)} anchors — "
                    "consider answering more questions."
                )
            else:
                st.error(
                    f"❌ Low confidence ({confidence:.0f}/100). "
                    "Please answer more questions for better results."
                )

            # Drift risk feedback
            drift_messages = {
                "low":      "🟢 Low drift risk — proceed with confidence.",
                "medium":   "🟡 Medium drift risk — monitoring active.",
                "high":     "🔴 High drift risk — consider more answers.",
                "critical": "💀 Critical drift risk — answer more questions.",
            }
            msg = drift_messages.get(drift_risk, "")
            if msg:
                if drift_risk == "low":
                    st.success(msg)
                elif drift_risk == "medium":
                    st.info(msg)
                else:
                    st.warning(msg)

        elif result is not None:
            st.error(
                "❌ Vision profile extraction failed. "
                "Check backend connectivity."
            )
        else:
            st.error(
                "❌ No response from backend. "
                "Is the server running on port 8000?"
            )

# ══════════════════════════════════════════════════════════════════
# STEP 4: Show Vision Profile + Proceed
# ══════════════════════════════════════════════════════════════════

if st.session_state.vision_profile:
    vp = st.session_state.vision_profile
    st.divider()

    st.markdown("### Step 4: Review Your Vision Profile")
    render_vision_profile(vp)

    st.divider()
    st.markdown("### Step 5: Proceed to Optimizer")

    # Show any unanswered critical questions
    unanswered = vp.get("unanswered_critical_questions", [])
    if unanswered:
        st.warning(
            "⚠️ **These questions were not answered — "
            "consider going back:**\n"
            + "\n".join(f"• {q}" for q in unanswered[:5])
        )

    col_proceed, col_clear = st.columns([3, 1])

    with col_proceed:
        st.success(
            "✅ Your vision profile is ready. "
            "The Optimizer will use your semantic anchors to "
            "prevent concept drift through all pipeline stages."
        )

    with col_clear:
        if st.button("🔄 Start Over", type="secondary"):
            st.session_state.vision_probe   = None
            st.session_state.vision_profile = None
            st.session_state.vision_input   = ""
            st.rerun()

    if st.button(
        "⚡ Proceed to Optimizer →",
        type="primary",
        use_container_width=True,
    ):
        # Pass vision data to optimizer
        st.session_state.optimizer_requirement    = (
            st.session_state.vision_input
        )
        st.session_state.optimizer_vision_profile = vp
        st.session_state.optimizer_context        = (
            st.session_state.vision_context
        )
        st.session_state.optimizer_answers        = {}
        st.switch_page("pages/2_⚡_Optimizer.py")

    # Debug view
    with st.expander("🔧 Raw Vision Profile JSON"):
        st.json(vp)