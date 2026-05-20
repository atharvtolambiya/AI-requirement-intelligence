"""
API Client - Centralized HTTP client for all FastAPI backend calls.

All Streamlit pages import from here instead of making raw httpx calls.
This ensures:
- Consistent error handling across all pages
- Single place to update base URL
- Proper timeout configuration
- Clean response parsing
- User-friendly error messages
"""

import sys
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from backend.config import settings


# ─── Client Configuration ─────────────────────────────────────────
TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=600.0,    # LLM calls can take up to 10 minutes with Groq rate limits
    write=10.0,
    pool=5.0,
)

# Use streamlit secrets if configured, otherwise fall back to settings.api_base_url
try:
    if "BACKEND_URL" in st.secrets:
        BASE_URL = st.secrets["BACKEND_URL"].rstrip("/")
    else:
        BASE_URL = settings.api_base_url.rstrip("/")
except (FileNotFoundError, KeyError):
    BASE_URL = settings.api_base_url.rstrip("/")


# ══════════════════════════════════════════════════════════════════
# GENERIC REQUEST HELPER
# ══════════════════════════════════════════════════════════════════

def _make_request(
    method: str,
    endpoint: str,
    payload: dict | None = None,
    params: dict | None = None,
) -> dict[str, Any] | None:
    """
    Makes a synchronous HTTP request to the FastAPI backend.

    Returns parsed JSON dict on success, None on failure.
    Shows user-friendly error messages via st.error().

    Args:
        method: HTTP method (GET, POST, DELETE)
        endpoint: API endpoint path (e.g., '/analysis/intent')
        payload: Request body for POST requests
        params: Query parameters

    Returns:
        Parsed response dict or None on failure
    """
    url = f"{BASE_URL}{endpoint}"

    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            if method == "GET":
                response = client.get(url, params=params)
            elif method == "POST":
                response = client.post(url, json=payload, params=params)
            elif method == "DELETE":
                response = client.delete(url, params=params)
            else:
                st.error(f"Unsupported HTTP method: {method}")
                return None

        if response.status_code == 200:
            return response.json()

        # Handle specific error codes
        elif response.status_code == 404:
            st.warning("⚠️ Resource not found.")
            return None

        elif response.status_code == 422:
            detail = response.json().get("detail", "Validation error")
            st.error(f"❌ Invalid input: {detail}")
            return None

        elif response.status_code == 503:
            st.error(
                "❌ **LLM Service Unavailable.** "
                "Check your API key in `.env` and ensure the LLM provider is reachable."
            )
            return None

        else:
            try:
                detail = response.json().get("detail", response.text[:200])
            except Exception:
                detail = response.text[:200]
            st.error(f"❌ API Error {response.status_code}: {detail}")
            return None

    except httpx.ConnectError:
        st.error(
            "❌ **Cannot connect to backend.** "
            "Start it with: `uvicorn backend.main:app --reload --port 8000`"
        )
        return None

    except httpx.TimeoutException:
        st.error(
            "⏱️ **Request timed out.** "
            "The LLM is taking longer than expected. Please try again."
        )
        return None

    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None


# ══════════════════════════════════════════════════════════════════
# HEALTH ENDPOINTS
# ══════════════════════════════════════════════════════════════════

def check_health() -> dict | None:
    """Checks backend liveness."""
    return _make_request("GET", "/health")


def check_readiness() -> dict | None:
    """Checks backend readiness including LLM config."""
    return _make_request("GET", "/ready")


# ══════════════════════════════════════════════════════════════════
# ANALYSIS ENDPOINTS
# ══════════════════════════════════════════════════════════════════

def analyze_intent(
    raw_requirement: str,
    context: str | None = None,
    session_id: str | None = None,
) -> dict | None:
    """Calls /analysis/intent endpoint."""
    payload = {
        "raw_requirement": raw_requirement,
        "context": context,
        "session_id": session_id,
    }
    return _make_request("POST", "/analysis/intent", payload=payload)


def detect_gaps(
    raw_requirement: str,
    intent_summary: str | None = None,
    domain: str | None = None,
    session_id: str | None = None,
) -> dict | None:
    """Calls /analysis/gaps endpoint."""
    payload = {
        "raw_requirement": raw_requirement,
        "intent_summary": intent_summary,
        "domain": domain,
        "session_id": session_id,
    }
    return _make_request("POST", "/analysis/gaps", payload=payload)


def full_analysis(
    raw_requirement: str,
    context: str | None = None,
    session_id: str | None = None,
) -> dict | None:
    """Calls /analysis/full endpoint (intent + gaps combined)."""
    payload = {
        "raw_requirement": raw_requirement,
        "context": context,
        "session_id": session_id,
    }
    return _make_request("POST", "/analysis/full", payload=payload)


# ══════════════════════════════════════════════════════════════════
# OPTIMIZATION ENDPOINTS
# ══════════════════════════════════════════════════════════════════

def expand_requirement(
    raw_requirement: str,
    intent_summary: str | None = None,
    domain: str | None = None,
    user_answers: dict | None = None,
    session_id: str | None = None,
) -> dict | None:
    """Calls /optimize/expand endpoint."""
    payload = {
        "raw_requirement": raw_requirement,
        "intent_summary": intent_summary,
        "domain": domain,
        "user_answers": user_answers,
        "session_id": session_id,
    }
    return _make_request("POST", "/optimize/expand", payload=payload)


def run_pipeline(
    raw_requirement: str,
    context: str | None = None,
    user_answers: dict | None = None,
    styles: list[str] | None = None,
    target_llm: str = "gpt-4",
    session_id: str | None = None,
) -> dict | None:
    """Calls /optimize/pipeline endpoint (full end-to-end)."""
    payload = {
        "raw_requirement": raw_requirement,
        "context": context,
        "user_answers": user_answers,
        "styles": styles or ["chain_of_thought", "structured"],
        "target_llm": target_llm,
        "session_id": session_id,
    }
    return _make_request("POST", "/optimize/pipeline", payload=payload)


def score_prompts(
    prompts: list[dict],
    original_requirement: str,
    session_id: str | None = None,
) -> dict | None:
    """Calls /optimize/score endpoint."""
    payload = {
        "prompts": prompts,
        "original_requirement": original_requirement,
        "session_id": session_id,
    }
    return _make_request("POST", "/optimize/score", payload=payload)


# ══════════════════════════════════════════════════════════════════
# HISTORY ENDPOINTS
# ══════════════════════════════════════════════════════════════════

def get_sessions(
    page: int = 1,
    page_size: int = 20,
    min_score: float | None = None,
) -> dict | None:
    """Calls GET /history/sessions endpoint."""
    params = {"page": page, "page_size": page_size}
    if min_score is not None:
        params["min_score"] = min_score
    return _make_request("GET", "/history/sessions", params=params)


def get_session_detail(session_id: str) -> dict | None:
    """Calls GET /history/sessions/{session_id} endpoint."""
    return _make_request("GET", f"/history/sessions/{session_id}")


def delete_session(session_id: str) -> dict | None:
    """Calls DELETE /history/sessions/{session_id} endpoint."""
    return _make_request("DELETE", f"/history/sessions/{session_id}")


def get_analytics() -> dict | None:
    """Calls GET /history/analytics endpoint."""
    return _make_request("GET", "/history/analytics")


def get_rag_stats() -> dict | None:
    """Calls GET /history/rag/stats endpoint."""
    return _make_request("GET", "/history/rag/stats")


def refine_prompt(
    prompt_text: str,
    original_requirement: str,
    prompt_style: str = "zero_shot",
    target_score: float = 80.0,
    max_iterations: int = 3,
    session_id: str | None = None,
) -> dict | None:
    """Calls POST /history/refine endpoint."""
    payload = {
        "prompt_text": prompt_text,
        "original_requirement": original_requirement,
        "prompt_style": prompt_style,
        "target_score": target_score,
        "max_iterations": max_iterations,
        "session_id": session_id,
    }
    return _make_request("POST", "/history/refine", payload=payload)

# ══════════════════════════════════════════════════════════════════
# VISION ENDPOINTS (Phase 2)
# ══════════════════════════════════════════════════════════════════

def generate_vision_probe(
    raw_input: str,
    context: str | None = None,
    session_id: str | None = None,
) -> dict | None:
    """Calls POST /vision/probe — generates clarification questions."""
    payload = {
        "raw_input": raw_input,
        "context": context,
        "session_id": session_id,
    }
    return _make_request("POST", "/vision/probe", payload=payload)


def extract_vision_profile(
    raw_input: str,
    answers: list[dict],
    probe: dict,
    session_id: str | None = None,
) -> dict | None:
    """Calls POST /vision/profile — extracts vision profile from answers."""
    payload = {
        "raw_input": raw_input,
        "answers": answers,
        "probe": probe,
        "session_id": session_id,
    }
    return _make_request("POST", "/vision/profile", payload=payload)


# ══════════════════════════════════════════════════════════════════
# SEMANTIC ANALYSIS ENDPOINTS (Phase 3)
# ══════════════════════════════════════════════════════════════════

def run_semantic_analysis(
    raw_input: str,
    vision_profile: dict | None = None,
    session_id: str | None = None,
) -> dict | None:
    """Calls POST /semantic/full — all three analyses in parallel."""
    payload = {
        "raw_input": raw_input,
        "vision_profile": vision_profile,
        "session_id": session_id,
    }
    return _make_request("POST", "/semantic/full", payload=payload)


# ══════════════════════════════════════════════════════════════════
# DRIFT ENDPOINTS (Phase 4)
# ══════════════════════════════════════════════════════════════════

def build_preservation_rules(
    vision_profile: dict,
    semantic_intent: dict | None = None,
    innovation_profile: dict | None = None,
    session_id: str | None = None,
) -> dict | None:
    """Calls POST /drift/preserve — builds preservation rules."""
    payload = {
        "vision_profile": vision_profile,
        "semantic_intent": semantic_intent,
        "innovation_profile": innovation_profile,
        "session_id": session_id,
    }
    return _make_request("POST", "/drift/preserve", payload=payload)


# ══════════════════════════════════════════════════════════════════
# SEMANTIC PIPELINE ENDPOINTS (Phase 5)
# ══════════════════════════════════════════════════════════════════

def run_vision_pipeline(
    raw_requirement: str,
    vision_profile: dict | None = None,
    context: str | None = None,
    user_answers: dict | None = None,
    styles: list[str] | None = None,
    target_llm: str = "gpt-4o",
    session_id: str | None = None,
) -> dict | None:
    """
    Calls POST /pipeline/run-vision — full 9-stage semantic pipeline.
    Recommended over the old /optimize/pipeline endpoint.
    """
    payload = {
        "raw_requirement": raw_requirement,
        "vision_profile": vision_profile,
        "context": context,
        "user_answers": user_answers,
        "styles": styles or [
            "chain_of_thought", "structured",
        ],
        "target_llm": target_llm,
        "session_id": session_id,
    }
    return _make_request(
        "POST", "/pipeline/run-vision", payload=payload
    )

# ══════════════════════════════════════════════════════════════════
# DEBUG HELPERS
# ══════════════════════════════════════════════════════════════════

def ping_backend() -> tuple[bool, str]:
    """
    Quick connectivity check.
    Returns (is_available, message).
    """
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.get(f"{BASE_URL}/health")
            if r.status_code == 200:
                data = r.json()
                return True, (
                    f"✅ Backend online | uptime: "
                    f"{data.get('uptime_seconds', 0):.0f}s"
                )
            return False, f"Backend returned {r.status_code}"
    except httpx.ConnectError:
        return False, "Cannot connect — is backend running?"
    except Exception as e:
        return False, str(e)