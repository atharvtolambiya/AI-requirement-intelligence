"""
Optimizer Page — Updated for Phase 5 semantic pipeline.

Now uses /pipeline/run-vision endpoint when VisionProfile is available.
Falls back to standard pipeline otherwise.
"""

import streamlit as st

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Optimizer | AI Req Intel",
    page_icon="⚡",
    layout="wide",
)

import sys
import uuid
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from frontend.utils.api_client import run_vision_pipeline, run_pipeline
from frontend.utils.ui_components import (
    render_expanded_requirement,
    render_prompt_card,
    render_sidebar_status,
    render_vision_profile,
    render_drift_analysis,
    render_innovation_profile,
)

render_sidebar_status()

# ─── Header ───────────────────────────────────────────────────────
st.title("⚡ Semantic Prompt Optimizer")
st.markdown(
    "Runs the **9-stage semantic pipeline**: "
    "intent analysis → innovation detection → drift prevention → "
    "vision-aligned expansion → optimized prompts → scoring."
)

# ─── Session State ────────────────────────────────────────────────
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

default_requirement  = st.session_state.get("optimizer_requirement", "")
default_context      = st.session_state.get("optimizer_context", "")
default_answers      = st.session_state.get("optimizer_answers", {})
default_vision       = st.session_state.get(
    "optimizer_vision_profile", None
)

# ─── Vision Profile Status ────────────────────────────────────────
if default_vision:
    st.success(
        "🧠 **Vision Profile active** — semantic anchors will be "
        "injected into all pipeline stages."
    )
    with st.expander("👁️ View Active Vision Profile"):
        render_vision_profile(default_vision)
else:
    st.info(
        "ℹ️ No Vision Profile. For best results with novel ideas, "
        "complete the **🧠 Vision Clarifier** page first."
    )

st.divider()

# ─── Configuration ────────────────────────────────────────────────
with st.expander("⚙️ Pipeline Configuration", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        raw_requirement = st.text_area(
            label="📝 Your Requirement",
            value=default_requirement,
            placeholder=(
                "Describe what you want to build...\n"
                "Example: AI-powered decision intelligence system for supply chain"
            ),
            height=120,
            key="optimizer_req",
        )

        context = st.text_area(
            label="📌 Additional Context (optional)",
            value=default_context,
            placeholder="Team size, constraints, tech preferences...",
            height=69,
            key="optimizer_ctx",
        )

    with col2:
        st.markdown("**🎨 Select Prompt Styles**")

        style_options = {
            "zero_shot":        "Zero-Shot (Direct instruction)",
            "chain_of_thought": "Chain-of-Thought (Step-by-step)",
            "role_based":       "Role-Based (Expert persona)",
            "structured":       "Structured (Organized sections)",
            "few_shot":         "Few-Shot (With examples)",
        }
        default_on = {
            "chain_of_thought", "structured",
        }

        selected_styles = [
            style_key
            for style_key, style_label in style_options.items()
            if st.checkbox(
                style_label,
                value=(style_key in default_on),
                key=f"style_{style_key}",
            )
        ]

        target_llm = st.selectbox(
            "🤖 Target LLM",
            options=[
                "gpt-4o", "gpt-4o-mini", "gpt-4",
                "claude-3-opus", "claude-3-sonnet",
                "llama-3.1-70b", "mixtral-8x7b",
                "gemini-1.5-flash",
            ],
            index=0,
        )

if not selected_styles:
    st.warning("⚠️ Select at least one prompt style.")

# ─── Run Button ───────────────────────────────────────────────────
run_clicked = st.button(
    "⚡ Run Semantic Pipeline",
    type="primary",
    use_container_width=True,
    disabled=not selected_styles,
)

if run_clicked:
    if not raw_requirement.strip():
        st.warning("⚠️ Please enter a requirement.")
        st.stop()
    if len(raw_requirement.strip()) < 10:
        st.warning("⚠️ Requirement is too short.")
        st.stop()

    session_id   = str(uuid.uuid4())
    progress_bar = st.progress(0, text="Starting semantic pipeline...")

    with st.spinner(""):
        progress_bar.progress(
            5, text="🧠 Stage 1-3: Semantic analysis..."
        )

        if default_vision:
            result = run_vision_pipeline(
                raw_requirement=raw_requirement.strip(),
                vision_profile=default_vision,
                context=context.strip() if context.strip() else None,
                user_answers=default_answers or None,
                styles=selected_styles,
                target_llm=target_llm,
                session_id=session_id,
            )
        else:
            result = run_pipeline(
                raw_requirement=raw_requirement.strip(),
                context=context.strip() if context.strip() else None,
                user_answers=default_answers or None,
                styles=selected_styles,
                target_llm=target_llm,
                session_id=session_id,
            )

        if result:
            progress_bar.progress(100, text="✅ Pipeline complete!")
            st.session_state.pipeline_result = result
            best_s = (
                result
                .get("best_prompt", {})
                .get("score", {})
                .get("overall_score", 0)
            )
            st.success(
                f"✅ Pipeline complete! "
                f"Generated {len(result.get('scored_prompts', []))} prompts. "
                f"Best score: **{best_s:.1f}/100**"
            )
        else:
            progress_bar.empty()

# ─── Show Results ─────────────────────────────────────────────────
if st.session_state.pipeline_result:
    result = st.session_state.pipeline_result
    st.divider()

    # ── Top metrics ───────────────────────────────────────────────
    best       = result.get("best_prompt", {})
    best_score = best.get("score", {})

    st.markdown("### 📊 Pipeline Results")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(
            "Prompts", len(result.get("scored_prompts", []))
        )
    with m2:
        st.metric(
            "Best Score",
            f"{best_score.get('overall_score',0):.1f}/100",
        )
    with m3:
        st.metric("Grade", best_score.get("grade", "?"))
    with m4:
        st.metric("Tokens", result.get("total_tokens_used", 0))
    with m5:
        t = result.get("total_processing_time_ms", 0)
        st.metric("Time", f"{t/1000:.1f}s")

    # ── Recommendation ────────────────────────────────────────────
    rec_style  = result.get("recommended_style", "")
    rec_reason = result.get("recommendation_reason", "")
    if rec_style:
        st.info(
            f"⭐ **Recommended:** "
            f"{rec_style.replace('_',' ').title()} — {rec_reason}"
        )

    st.divider()

    # ── Results Tabs ──────────────────────────────────────────────
    tab_prompts, tab_expanded, tab_debug = st.tabs([
        "🎯 Generated Prompts",
        "📋 Expanded Requirement",
        "🔧 Debug",
    ])

    with tab_prompts:
        scored_prompts = result.get("scored_prompts", [])
        best_style     = best.get("style", "")

        if scored_prompts:
            sorted_prompts = sorted(
                scored_prompts,
                key=lambda x: x.get("score", {}).get("overall_score", 0),
                reverse=True,
            )

            tab_labels = []
            for p in sorted_prompts:
                label  = p.get("style_label", p.get("style", "?"))
                grade  = p.get("score", {}).get("grade", "?")
                is_top = p.get("style", "") == best_style
                star   = "⭐ " if is_top else ""
                tab_labels.append(f"{star}{label} ({grade})")

            prompt_tabs = st.tabs(tab_labels)
            for idx, (ptab, pdata) in enumerate(
                zip(prompt_tabs, sorted_prompts)
            ):
                with ptab:
                    is_best = pdata.get("style", "") == best_style
                    render_prompt_card(
                        prompt=pdata,
                        show_score=True,
                        is_best=is_best,
                        index=idx,
                    )

            # ── Refine CTA ────────────────────────────────────────
            st.divider()
            best_val = best_score.get("overall_score", 0)
            if best_val < 80:
                st.info(
                    f"Best prompt scored **{best_val:.1f}/100**. "
                    "Use ✨ Refiner to improve."
                )
                if st.button("✨ Refine Best Prompt →"):
                    st.session_state.refiner_prompt = (
                        best.get("prompt_text", "")
                    )
                    st.session_state.refiner_style = (
                        best.get("style", "zero_shot")
                    )
                    st.session_state.refiner_requirement = (
                        result.get("raw_requirement", "")
                    )
                    st.session_state.refiner_score = best_val
                    st.switch_page("pages/3_✨_Refiner.py")
            else:
                st.success(
                    f"🏆 Best prompt: **{best_val:.1f}/100** "
                    f"({best_score.get('grade','?')}). Production ready!"
                )

    with tab_expanded:
        expanded = result.get("expanded_requirement", {})
        if expanded:
            render_expanded_requirement(expanded)
        else:
            st.info("No expanded requirement data available.")

    with tab_debug:
        st.json(result)