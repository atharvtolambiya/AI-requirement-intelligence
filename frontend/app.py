"""
Streamlit Homepage — Phase 6 Update.
Updated navigation includes Vision page.
"""

import sys
from pathlib import Path

import httpx
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
from backend.config import settings

# Use streamlit secrets if configured, otherwise fall back to settings.api_base_url
try:
    if "BACKEND_URL" in st.secrets:
        BASE_URL = st.secrets["BACKEND_URL"].rstrip("/")
    else:
        BASE_URL = settings.api_base_url.rstrip("/")
except (FileNotFoundError, KeyError):
    BASE_URL = settings.api_base_url.rstrip("/")

st.set_page_config(
    page_title="AI Requirement Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/brain--v2.png", width=80
    )
    st.title("🧠 AI Req Intel")
    st.caption("Semantic Intent Preservation System")
    st.divider()

    st.markdown("**🗺️ Navigation**")
    st.page_link("app.py",                          label="🏠 Home")
    st.page_link("pages/0_🧠_Vision.py",            label="🧠 Vision Clarifier")
    st.page_link("pages/2_⚡_Optimizer.py",         label="⚡ Optimizer")
    st.page_link("pages/3_✨_Refiner.py",           label="✨ Refiner")
    st.page_link("pages/4_📚_History.py",           label="📚 History")
    st.divider()

    @st.cache_data(ttl=15)
    def _health():
        try:
            r = httpx.get(
                f"{BASE_URL}/health", timeout=15.0
            )
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    @st.cache_data(ttl=15)
    def _ready():
        try:
            r = httpx.get(
                f"{BASE_URL}/ready", timeout=15.0
            )
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    h = _health()
    r = _ready()

    st.markdown("**⚙️ Status**")
    if h:
        st.success(f"✅ Backend ({h.get('uptime_seconds',0):.0f}s)")
    else:
        st.error("❌ Backend Offline")
        st.caption("`uvicorn backend.main:app --reload`")

    if r:
        checks = r.get("checks", {})
        if checks.get("llm_api_key") == "configured":
            st.success(f"✅ {r.get('llm_provider','').upper()}")
            st.caption(r.get("active_model", ""))
        else:
            st.warning("⚠️ API Key Missing")

    st.divider()
    st.caption("v2.0.0 | Semantic Pipeline")

# ─── Homepage ─────────────────────────────────────────────────────
st.title("🧠 AI Requirement Intelligence")
st.subheader("Semantic Intent Preservation System v2.0")
st.markdown(
    "> Stop losing your innovation to generic software patterns. "
    "This system **preserves your conceptual intent** through every "
    "stage of requirement expansion and prompt generation."
)
st.divider()

# ── New Flow ───────────────────────────────────────────────────────
st.subheader("🚀 Recommended Flow")

c1, c2, c3, c4 = st.columns(4)
cards = [
    ("🧠", "#f8e8ff", "Step 1\nVision Clarifier",
     "Answer targeted questions → extract semantic anchors"),
    ("⚡", "#e8f8e8", "Step 2\nOptimizer",
     "9-stage semantic pipeline → vision-aligned prompts"),
    ("✨", "#fef9e8", "Step 3\nRefiner",
     "RAG-powered iterative improvement"),
    ("📚", "#fde8ff", "Step 4\nHistory",
     "Browse sessions + analytics"),
]

for col, (icon, bg, title, desc) in zip(
    [c1, c2, c3, c4], cards
):
    with col:
        lines = title.split("\n")
        st.markdown(
            f"""
            <div style="background:{bg}; border-radius:10px;
                        padding:16px; text-align:center; height:150px;">
                <div style="font-size:2rem;">{icon}</div>
                <div style="font-weight:bold; font-size:0.9rem;
                            margin:6px 0;">{lines[0]}<br>{lines[1]}</div>
                <div style="font-size:0.78rem; color:#555;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ── What's new in v2.0 ────────────────────────────────────────────
st.subheader("🆕 What's New in v2.0 — Semantic Pipeline")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.success(
        "**🧠 Vision Clarifier**\n"
        "Adaptive questions extract semantic anchors "
        "BEFORE any processing begins."
    )
    st.success(
        "**⚓ Semantic Anchors**\n"
        "Non-negotiable concepts preserved through "
        "ALL 9 pipeline stages."
    )
with col_b:
    st.warning(
        "**🚀 Innovation Analyzer**\n"
        "Classifies innovation tier and identifies "
        "at-risk concepts before expansion."
    )
    st.warning(
        "**📐 Abstraction Classifier**\n"
        "Detects gap between your intent level "
        "and system default level."
    )
with col_c:
    st.error(
        "**🌊 Drift Detector**\n"
        "Audits expansion output for concept loss, "
        "dilution, and anti-pattern violations."
    )
    st.error(
        "**📊 Vision Alignment Score**\n"
        "New scoring dimension: does the prompt "
        "actually reflect YOUR vision?"
    )

st.divider()

with st.expander("🛠️ Quick Start"):
    st.code(
        """
# Terminal 1: Backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
streamlit run frontend/app.py

# New recommended flow:
# 1. Go to 🧠 Vision Clarifier — answer questions
# 2. Go to ⚡ Optimizer — run semantic pipeline
# 3. Go to ✨ Refiner — RAG refinement if needed
        """,
        language="bash",
    )