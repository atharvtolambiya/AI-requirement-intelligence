"""
History Page - Prompt Session History & Analytics Dashboard.

Provides:
1. Analytics overview with metrics and distributions
2. Paginated session browser
3. Session detail view with all prompts
4. Session deletion
5. RAG knowledge base status
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent.parent.parent))

from frontend.utils.api_client import (
    get_analytics,
    get_session_detail,
    get_sessions,
    delete_session,
    get_rag_stats,
)
from frontend.utils.ui_components import (
    render_analytics_overview,
    render_dimension_scores,
    render_sidebar_status,
)

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="History | AI Req Intel",
    page_icon="📚",
    layout="wide",
)

render_sidebar_status()

# ─── Page Header ──────────────────────────────────────────────────
st.title("📚 Prompt History")
st.markdown(
    "Browse all saved optimization sessions, view prompt details, "
    "track quality scores over time, and manage your prompt library."
)
st.divider()

# ─── Tab Layout ───────────────────────────────────────────────────
tab_analytics, tab_sessions, tab_rag = st.tabs([
    "📊 Analytics",
    "🗂️ Sessions",
    "🧠 RAG Knowledge Base",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1: ANALYTICS
# ══════════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown("### 📊 Usage Analytics Dashboard")

    col_refresh, _ = st.columns([1, 4])
    with col_refresh:
        refresh_analytics = st.button("🔄 Refresh", key="refresh_analytics")

    @st.cache_data(ttl=30)
    def _load_analytics():
        return get_analytics()

    if refresh_analytics:
        st.cache_data.clear()

    analytics = _load_analytics()

    if analytics:
        render_analytics_overview(analytics)

        # Additional insights
        st.divider()
        st.markdown("### 💡 Insights")

        total_sessions = analytics.get("total_sessions", 0)
        avg_score = analytics.get("average_best_score", 0)
        prod_rate = analytics.get("production_ready_rate", 0)
        total_refinements = analytics.get("total_refinements", 0)

        insights = []
        if total_sessions == 0:
            insights.append(
                "🚀 **Get started!** Run your first optimization on the ⚡ Optimizer page."
            )
        else:
            if avg_score >= 80:
                insights.append(
                    f"🏆 **Excellent quality!** Average score of {avg_score:.1f}/100 "
                    "shows consistently high-quality prompt generation."
                )
            elif avg_score >= 65:
                insights.append(
                    f"📈 **Good progress!** Average score of {avg_score:.1f}/100. "
                    "Use the ✨ Refiner to push scores above 80."
                )
            else:
                insights.append(
                    f"⚠️ **Room for improvement.** Average score of {avg_score:.1f}/100. "
                    "Try adding more context to your requirements."
                )

            if prod_rate >= 70:
                insights.append(
                    f"✅ **{prod_rate:.1f}%** of prompts are production-ready. Great work!"
                )
            elif prod_rate > 0:
                insights.append(
                    f"📝 **{prod_rate:.1f}%** of prompts meet production quality. "
                    "Use the Refiner to improve more prompts."
                )

            if total_refinements > 0:
                insights.append(
                    f"🔄 **{total_refinements}** refinement iterations run. "
                    "RAG-powered refinement is helping improve prompt quality."
                )

        for insight in insights:
            st.markdown(f"• {insight}")
    else:
        st.info(
            "📊 No analytics data yet. "
            "Run your first optimization on the ⚡ Optimizer page!"
        )


# ══════════════════════════════════════════════════════════════════
# TAB 2: SESSIONS BROWSER
# ══════════════════════════════════════════════════════════════════
with tab_sessions:
    st.markdown("### 🗂️ Optimization Sessions")

    # ── Filters ──────────────────────────────────────────────────
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        page_num = st.number_input(
            "Page", min_value=1, value=1, step=1, key="sessions_page"
        )
    with filter_col2:
        page_size = st.selectbox(
            "Per Page", options=[10, 20, 50], index=1, key="sessions_size"
        )
    with filter_col3:
        min_score_filter = st.number_input(
            "Min Score Filter",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=10.0,
            key="sessions_min_score",
        )

    refresh_sessions = st.button("🔄 Refresh Sessions", key="refresh_sessions")

    @st.cache_data(ttl=30)
    def _load_sessions(page, size, min_score):
        return get_sessions(
            page=page,
            page_size=size,
            min_score=min_score if min_score > 0 else None,
        )

    if refresh_sessions:
        st.cache_data.clear()

    sessions_data = _load_sessions(
        page_num,
        page_size,
        min_score_filter,
    )

    if sessions_data:
        total = sessions_data.get("total", 0)
        pages = sessions_data.get("pages", 1)
        sessions = sessions_data.get("sessions", [])

        st.caption(
            f"Showing page {page_num}/{pages} | "
            f"{total} total sessions"
        )

        if not sessions:
            st.info(
                "No sessions found. "
                "Run your first optimization on the ⚡ Optimizer page!"
            )
        else:
            # Session list
            for session in sessions:
                session_id = session.get("session_id", "")
                short_id = session_id[:8] if session_id else "?"
                title = session.get("expanded_title") or session.get(
                    "raw_requirement", "Untitled"
                )[:60]
                score = session.get("best_prompt_score", 0) or 0
                grade = session.get("best_prompt_grade", "?") or "?"
                complexity = session.get("detected_complexity", "?") or "?"
                style = (session.get("best_prompt_style") or "?").replace("_", " ").title()
                created = session.get("created_at", "")[:16] if session.get("created_at") else ""

                # Grade color indicator
                grade_colors = {
                    "A+": "🟢", "A": "🟢", "B": "🔵",
                    "C": "🟡", "D": "🟠", "F": "🔴",
                }
                grade_dot = grade_colors.get(grade, "⚪")

                with st.expander(
                    f"{grade_dot} [{grade}] {title[:55]}... | "
                    f"Score: {score:.1f} | {created} | ID: {short_id}"
                ):
                    detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
                    with detail_col1:
                        st.metric("Best Score", f"{score:.1f}/100")
                    with detail_col2:
                        st.metric("Grade", grade)
                    with detail_col3:
                        st.metric("Best Style", style)
                    with detail_col4:
                        st.metric("Complexity", complexity.title())

                    req_preview = session.get("raw_requirement", "")
                    st.markdown(f"**Requirement:** {req_preview[:200]}")

                    btn_col1, btn_col2 = st.columns(2)

                    with btn_col1:
                        if st.button(
                            "🔍 View Full Detail",
                            key=f"view_{session_id}",
                        ):
                            st.session_state.selected_session_id = session_id

                    with btn_col2:
                        if st.button(
                            "🗑️ Delete",
                            key=f"delete_{session_id}",
                            type="secondary",
                        ):
                            result = delete_session(session_id)
                            if result:
                                st.success("✅ Session deleted")
                                st.cache_data.clear()
                                st.rerun()

            # ── Pagination info ───────────────────────────────────
            if pages > 1:
                st.caption(
                    f"Page {page_num} of {pages}. "
                    "Use the Page input above to navigate."
                )

        # ── Session Detail View ───────────────────────────────────
        if "selected_session_id" in st.session_state:
            sel_id = st.session_state.selected_session_id
            st.divider()
            st.markdown(f"### 📄 Session Detail: `{sel_id[:16]}...`")

            detail = get_session_detail(sel_id)
            if detail:
                # Session overview
                d1, d2, d3, d4 = st.columns(4)
                with d1:
                    st.metric(
                        "Best Score",
                        f"{detail.get('best_prompt_score', 0):.1f}/100",
                    )
                with d2:
                    st.metric("Grade", detail.get("best_prompt_grade", "?"))
                with d3:
                    tokens = detail.get("total_tokens_used", 0)
                    st.metric("Tokens Used", tokens)
                with d4:
                    ms = detail.get("total_processing_time_ms", 0)
                    st.metric("Time", f"{ms/1000:.1f}s")

                # Requirement
                st.markdown(
                    f"**Requirement:** {detail.get('raw_requirement', '')}"
                )

                # All prompts
                prompts = detail.get("prompts", [])
                if prompts:
                    st.markdown(f"**Generated Prompts ({len(prompts)}):**")
                    for p in sorted(
                        prompts,
                        key=lambda x: x.get("overall_score", 0),
                        reverse=True,
                    ):
                        p_style = p.get("style_label", p.get("style", "?"))
                        p_score = p.get("overall_score", 0) or 0
                        p_grade = p.get("grade", "?") or "?"
                        p_best = p.get("is_best", False)
                        p_ready = p.get("is_production_ready", False)

                        label = f"{'⭐ ' if p_best else ''}{p_style} — {p_grade} ({p_score:.1f}/100)"
                        if p_ready:
                            label += " ✅"

                        with st.expander(label):
                            dim_data = []
                            for dim_key in [
                                "clarity_score", "specificity_score",
                                "completeness_score", "actionability_score",
                                "hallucination_reduction_score"
                            ]:
                                dim_name = dim_key.replace("_score", "").replace("_", " ").title()
                                dim_data.append({
                                    "dimension": dim_name,
                                    "score": p.get(dim_key, 0) or 0,
                                    "weight": 0.2,
                                    "feedback": "",
                                })
                            render_dimension_scores(dim_data)

                            st.text_area(
                                "Prompt Text",
                                value=p.get("prompt_text", ""),
                                height=150,
                                key=f"history_prompt_{p.get('id', '')}",
                                disabled=True,
                            )

                # Refinement logs
                refinements = detail.get("refinements", [])
                if refinements:
                    st.markdown(f"**Refinement History ({len(refinements)} iterations):**")
                    for ref in refinements:
                        st.markdown(
                            f"• Iteration {ref.get('iteration_number', '?')}: "
                            f"{ref.get('grade_before', '?')}({ref.get('score_before', 0):.0f}) → "
                            f"{ref.get('grade_after', '?')}({ref.get('score_after', 0):.0f}) "
                            f"[{ref.get('score_delta', 0):+.1f}]"
                        )

                if st.button("✖ Close Detail", key="close_detail"):
                    del st.session_state.selected_session_id
                    st.rerun()
            else:
                st.error("Session not found or could not be loaded.")
    else:
        st.info(
            "📭 No sessions in database yet. "
            "Run your first optimization on the ⚡ Optimizer page!"
        )


# ══════════════════════════════════════════════════════════════════
# TAB 3: RAG KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════
with tab_rag:
    st.markdown("### 🧠 RAG Knowledge Base")
    st.markdown(
        "ChromaDB vector store containing prompt engineering best practices "
        "used to guide intelligent prompt refinement."
    )

    refresh_rag = st.button("🔄 Refresh RAG Stats", key="refresh_rag")

    @st.cache_data(ttl=60)
    def _load_rag_stats():
        return get_rag_stats()

    if refresh_rag:
        st.cache_data.clear()

    rag_stats = _load_rag_stats()

    if rag_stats:
        col1, col2, col3 = st.columns(3)
        with col1:
            initialized = rag_stats.get("initialized", False)
            st.metric(
                "Status",
                "✅ Ready" if initialized else "❌ Not Ready",
            )
        with col2:
            st.metric("Documents Indexed", rag_stats.get("document_count", 0))
        with col3:
            st.metric(
                "Collection",
                rag_stats.get("collection_name", "N/A"),
            )

        if rag_stats.get("document_count", 0) > 0:
            st.success(
                f"✅ Knowledge base is active with "
                f"**{rag_stats.get('document_count', 0)} documents** indexed. "
                "RAG retrieval is operational."
            )
        else:
            st.warning(
                "⚠️ Knowledge base is empty. "
                "Ensure `data/knowledge/prompt_engineering.txt` exists "
                "and restart the backend."
            )
    else:
        st.error("❌ Could not connect to RAG service.")

    st.divider()
    st.markdown("### 📖 Knowledge Base Categories")

    categories = [
        ("🏛️ Fundamentals", "fundamentals", "Core prompt engineering principles and best practices"),
        ("🎨 Prompt Styles", "prompt_styles", "Zero-Shot, CoT, Few-Shot, Role-Based, Structured"),
        ("⚡ Quality", "quality", "Hallucination prevention, output format, anti-patterns"),
        ("🚀 Performance", "performance", "Context window optimization, token management"),
        ("🔄 Refinement", "refinement", "Iterative improvement process and measurement"),
        ("🏢 Domain-Specific", "domain_specific", "Software, Data Science, and specialized prompting"),
        ("🔬 Advanced", "advanced", "RAG-enhanced generation and complex techniques"),
    ]

    for emoji_label, category, description in categories:
        st.markdown(f"**{emoji_label}** — {description}")

    st.info(
        "💡 The knowledge base is loaded from "
        "`data/knowledge/prompt_engineering.txt`. "
        "You can add custom knowledge by editing that file and restarting the backend."
    )