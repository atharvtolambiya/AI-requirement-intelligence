"""
Refiner Page - Multi-step RAG Prompt Refinement.

Provides:
1. Input any prompt for refinement
2. Configure target score and iterations
3. Visual refinement progress tracking
4. Side-by-side before/after comparison
5. RAG knowledge base stats
"""

import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent.parent.parent))

from frontend.utils.api_client import refine_prompt, get_rag_stats
from frontend.utils.ui_components import (
    render_refinement_progress,
    render_score_badge,
    render_dimension_scores,
    render_sidebar_status,
    render_hallucination_risk,
    render_production_badge,
)

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Refiner | AI Req Intel",
    page_icon="✨",
    layout="wide",
)

render_sidebar_status()

# ─── Page Header ──────────────────────────────────────────────────
st.title("✨ RAG Prompt Refiner")
st.markdown(
    "Uses **ChromaDB knowledge retrieval** to find relevant prompt engineering "
    "best practices and applies them iteratively to improve your prompt quality."
)
st.divider()

# ─── Session State ────────────────────────────────────────────────
if "refine_result" not in st.session_state:
    st.session_state.refine_result = None

# Pre-fill from Optimizer page if available
default_prompt = st.session_state.get("refiner_prompt", "")
default_style = st.session_state.get("refiner_style", "zero_shot")
default_requirement = st.session_state.get("refiner_requirement", "")
default_score = st.session_state.get("refiner_score", None)

# ─── RAG Stats Sidebar Info ───────────────────────────────────────
with st.sidebar:
    st.divider()
    st.markdown("**📚 Knowledge Base**")

    @st.cache_data(ttl=60)
    def _rag_stats():
        return get_rag_stats()

    stats = _rag_stats()
    if stats:
        st.metric("Documents Indexed", stats.get("document_count", 0))
        st.caption(f"Collection: {stats.get('collection_name', 'N/A')}")
        if stats.get("document_count", 0) > 0:
            st.success("✅ RAG Ready")
        else:
            st.warning("⚠️ No docs indexed")
    else:
        st.error("❌ RAG unavailable")

# ─── Input Form ───────────────────────────────────────────────────
col_form, col_config = st.columns([2, 1])

with col_form:
    st.markdown("#### 📝 Prompt to Refine")

    prompt_text = st.text_area(
        label="Current Prompt",
        value=default_prompt,
        placeholder=(
            "Paste your prompt here...\n\n"
            "Or use the ⚡ Optimizer page to generate prompts first, "
            "then come back here to refine the best one."
        ),
        height=220,
        key="refiner_prompt_input",
    )

    original_requirement = st.text_input(
        label="Original Requirement",
        value=default_requirement,
        placeholder="What was the original user requirement this prompt is for?",
        key="refiner_req_input",
    )

with col_config:
    st.markdown("#### ⚙️ Refinement Config")

    style_options = {
        "zero_shot": "Zero-Shot",
        "chain_of_thought": "Chain-of-Thought",
        "role_based": "Role-Based",
        "structured": "Structured",
        "few_shot": "Few-Shot",
    }
    prompt_style = st.selectbox(
        "Prompt Style",
        options=list(style_options.keys()),
        format_func=lambda x: style_options[x],
        index=list(style_options.keys()).index(default_style)
        if default_style in style_options
        else 0,
    )

    target_score = st.slider(
        "Target Score",
        min_value=60.0,
        max_value=95.0,
        value=80.0,
        step=5.0,
        help="Refinement stops when this score is reached",
    )

    max_iterations = st.slider(
        "Max Iterations",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        help="Maximum number of refinement passes",
    )

    if default_score:
        st.metric("Current Score", f"{default_score:.1f}/100")

    st.info(
        f"🎯 Will refine up to **{max_iterations}** times, "
        f"stopping when score reaches **{target_score:.0f}/100**"
    )

# ─── Refine Button ────────────────────────────────────────────────
st.divider()

refine_clicked = st.button(
    "✨ Start RAG Refinement",
    type="primary",
    use_container_width=True,
)

if refine_clicked:
    errors = []
    if not prompt_text.strip():
        errors.append("Please enter a prompt to refine.")
    if len(prompt_text.strip()) < 10:
        errors.append("Prompt is too short.")
    if not original_requirement.strip():
        errors.append("Please enter the original requirement.")

    if errors:
        for err in errors:
            st.warning(f"⚠️ {err}")
    else:
        session_id = str(uuid.uuid4())

        with st.spinner(
            f"🔄 Running {max_iterations} refinement iterations with RAG knowledge... "
            f"(~{max_iterations * 20}-{max_iterations * 40}s)"
        ):
            result = refine_prompt(
                prompt_text=prompt_text.strip(),
                original_requirement=original_requirement.strip(),
                prompt_style=prompt_style,
                target_score=target_score,
                max_iterations=max_iterations,
                session_id=session_id,
            )

        if result:
            st.session_state.refine_result = result

            improvement = result.get("total_improvement", 0)
            if improvement > 0:
                st.success(
                    f"✅ Refinement complete! "
                    f"Score improved by **+{improvement:.1f} points** "
                    f"({result.get('grade_before', '?')} → {result.get('grade_after', '?')})"
                )
            else:
                st.info(
                    "ℹ️ Refinement complete. "
                    "Score did not improve significantly — prompt may already be optimal."
                )

# ─── Show Results ─────────────────────────────────────────────────
if st.session_state.refine_result:
    result = st.session_state.refine_result
    st.divider()

    # ── Summary Metrics ───────────────────────────────────────────
    st.markdown("### 📊 Refinement Summary")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Initial Score", f"{result.get('original_score', 0):.1f}/100")
    with col2:
        improvement = result.get("total_improvement", 0)
        st.metric(
            "Final Score",
            f"{result.get('final_score', 0):.1f}/100",
            delta=f"+{improvement:.1f}" if improvement > 0 else f"{improvement:.1f}",
        )
    with col3:
        st.metric("Iterations Run", result.get("iterations_run", 0))
    with col4:
        target_reached = result.get("target_reached", False)
        st.metric("Target Reached", "✅ Yes" if target_reached else "❌ No")
    with col5:
        stop_reason = result.get("stop_reason", "unknown")
        reason_labels = {
            "target_score_reached": "Target Hit",
            "max_iterations_reached": "Max Iters",
            "insufficient_improvement": "Converged",
            "already_meets_target": "Pre-Met",
        }
        st.metric("Stop Reason", reason_labels.get(stop_reason, stop_reason))

    # ── Refinement Progress Visual ─────────────────────────────────
    iterations = result.get("iterations_detail", [])
    if iterations:
        render_refinement_progress(iterations)

    st.divider()

    # ── Before / After Comparison ─────────────────────────────────
    st.markdown("### 🔄 Before vs After")

    col_before, col_after = st.columns(2)

    with col_before:
        before_score = result.get("original_score", 0)
        before_grade = result.get("grade_before", "?")
        st.markdown(f"#### Before ({before_grade} — {before_score:.1f}/100)")
        st.text_area(
            label="Original Prompt",
            value=result.get("original_prompt", ""),
            height=350,
            key="before_prompt_display",
            disabled=True,
        )

    with col_after:
        after_score = result.get("final_score", 0)
        after_grade = result.get("grade_after", "?")
        delta = result.get("total_improvement", 0)
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        st.markdown(
            f"#### After ({after_grade} — {after_score:.1f}/100) "
            f"[{delta_str}]"
        )
        refined = result.get("refined_prompt", "")
        st.text_area(
            label="Refined Prompt",
            value=refined,
            height=350,
            key="after_prompt_display",
        )
        st.caption("💡 Click the text area → Ctrl+A → Ctrl+C to copy")

    # ── Copy section ──────────────────────────────────────────────
    st.divider()
    st.markdown("### 📋 Use the Refined Prompt")

    refined_prompt = result.get("refined_prompt", "")
    if refined_prompt:
        st.code(refined_prompt, language=None)
        st.caption("⬆️ Use the copy button (top-right of code block) to copy")

        # Offer to re-refine if target not reached
        target_reached = result.get("target_reached", False)
        final_score = result.get("final_score", 0)

        if not target_reached and final_score < target_score:
            st.warning(
                f"⚠️ Target score of {target_score:.0f} not reached "
                f"(achieved {final_score:.1f}). "
                "You can re-run refinement on the improved prompt."
            )
            if st.button("🔄 Re-Refine This Prompt", type="secondary"):
                st.session_state.refiner_prompt = refined_prompt
                st.session_state.refiner_score = final_score
                st.session_state.refine_result = None
                st.rerun()

    # ── Raw debug ─────────────────────────────────────────────────
    with st.expander("🔧 Raw API Response (Debug)"):
        st.json(result)