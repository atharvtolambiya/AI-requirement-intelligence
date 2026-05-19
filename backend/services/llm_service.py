"""
LLM Service - Updated with Free Model Support + Rate Limiting.

Supported providers:
    groq      → Llama 3, Mixtral (FREE - https://console.groq.com)
    gemini    → Gemini Flash (FREE - https://aistudio.google.com)

Usage:
    from backend.services.llm_service import get_llm_service
    llm = get_llm_service()
    result = await llm.invoke(system_prompt="...", user_message="...")
"""

import asyncio
import random
import re
import time
from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from backend.config import settings
from backend.logger import logger


# ══════════════════════════════════════════════════════════════════
# RATE LIMITER
# ══════════════════════════════════════════════════════════════════

class TokenBucketRateLimiter:
    """
    Sliding window rate limiter tracking both TPM and RPM.

    Groq free tier hard limits:
        6,000 TPM  → we use 5,000 (buffer)
        30 RPM     → we use 25   (buffer)

    Gemini free tier hard limits:
        15 RPM     → we use 12   (buffer)
    """

    # Conservative limits per provider (below actual limits)
    PROVIDER_LIMITS = {
        "groq": {
            "tpm": 5000,   # actual: 6000
            "rpm": 25,     # actual: 30
        },
        "gemini": {
            "tpm": 999999, # Gemini doesn't publish strict TPM
            "rpm": 12,     # actual: 15
        },
    }

    def __init__(self, provider: str = "groq"):
        limits = self.PROVIDER_LIMITS.get(provider, self.PROVIDER_LIMITS["groq"])
        self.tpm_limit      = limits["tpm"]
        self.rpm_limit      = limits["rpm"]
        self.window_seconds = 60

        # Sliding window state
        # Each entry: (timestamp: float, tokens: int)
        self._token_log:   list[tuple[float, int]] = []
        # Each entry: timestamp: float
        self._request_log: list[float]             = []

        self._lock = asyncio.Lock()

        logger.debug(
            "RateLimiter ready | provider={p} | tpm={t} | rpm={r}",
            p=provider,
            t=self.tpm_limit,
            r=self.rpm_limit,
        )

    async def acquire(self, estimated_tokens: int) -> float:
        """
        Block until this call is safe to make.
        Returns how many seconds we waited.
        """
        async with self._lock:
            wait = self._calculate_wait(estimated_tokens)

            if wait > 0:
                logger.warning(
                    "Pre-emptive rate limit wait | wait={w:.1f}s | "
                    "estimated_tokens={t}",
                    w=wait,
                    t=estimated_tokens,
                )
                # Release lock while sleeping so other coroutines
                # can check state
                pass

        # Sleep OUTSIDE lock so we don't block other coroutines
        if wait > 0:
            await asyncio.sleep(wait)

        # Re-acquire lock to record this request
        async with self._lock:
            now = time.time()
            self._cleanup(now)
            self._token_log.append((now, estimated_tokens))
            self._request_log.append(now)

        return wait

    def _calculate_wait(self, estimated_tokens: int) -> float:
        """
        Calculate required wait time in seconds.
        Checks both TPM and RPM windows.
        """
        now = time.time()
        self._cleanup(now)

        waits: list[float] = []

        # ── TPM check ─────────────────────────────────────────────
        used_tokens = sum(t for _, t in self._token_log)
        if used_tokens + estimated_tokens > self.tpm_limit:
            if self._token_log:
                # Wait until oldest entry falls out of window
                oldest_ts  = self._token_log[0][0]
                tpm_wait   = self.window_seconds - (now - oldest_ts) + 0.5
                waits.append(max(0.0, tpm_wait))

        # ── RPM check ─────────────────────────────────────────────
        if len(self._request_log) >= self.rpm_limit:
            oldest_req = self._request_log[0]
            rpm_wait   = self.window_seconds - (now - oldest_req) + 0.5
            waits.append(max(0.0, rpm_wait))

        return max(waits) if waits else 0.0

    def _cleanup(self, now: float) -> None:
        """Remove log entries older than the sliding window."""
        cutoff = now - self.window_seconds
        self._token_log    = [(ts, t) for ts, t in self._token_log if ts > cutoff]
        self._request_log  = [ts for ts in self._request_log if ts > cutoff]

    def record_actual_tokens(self, actual_tokens: int) -> None:
        """
        Correct the last log entry with the real token count.
        Prevents estimation drift from accumulating over time.
        """
        if self._token_log:
            ts, _ = self._token_log[-1]
            self._token_log[-1] = (ts, actual_tokens)

    def current_usage(self) -> dict[str, int]:
        """Returns current window usage stats."""
        now = time.time()
        self._cleanup(now)
        return {
            "tokens_used":    sum(t for _, t in self._token_log),
            "requests_made":  len(self._request_log),
            "tpm_limit":      self.tpm_limit,
            "rpm_limit":      self.rpm_limit,
            "tpm_remaining":  self.tpm_limit - sum(t for _, t in self._token_log),
            "rpm_remaining":  self.rpm_limit - len(self._request_log),
        }


# ══════════════════════════════════════════════════════════════════
# RETRY HANDLER
# ══════════════════════════════════════════════════════════════════

class RetryHandler:
    """
    Exponential backoff retry logic for 429 rate limit errors.

    Strategy:
    1. Parse Groq's 'try again in Xs' from error message (most accurate)
    2. Fall back to exponential backoff with jitter
    """

    def __init__(
        self,
        max_retries: int   = 4,
        base_delay:  float = 5.0,
        max_delay:   float = 65.0,
        jitter:      float = 2.0,
    ):
        self.max_retries = max_retries
        self.base_delay  = base_delay
        self.max_delay   = max_delay
        self.jitter      = jitter

    def is_rate_limit_error(self, error: Exception) -> bool:
        """Return True if this is a 429 / rate limit error."""
        msg = str(error).lower()
        return any(kw in msg for kw in [
            "429",
            "rate_limit_exceeded",
            "rate limit",
            "tokens per minute",
            "requests per minute",
            "too many requests",
        ])

    def get_wait_time(self, attempt: int, error_message: str = "") -> float:
        """
        Calculate wait time for retry attempt N.

        Prefers Groq's suggested wait time when available.
        """
        groq_suggestion = self._parse_groq_wait(error_message)
        if groq_suggestion:
            wait = groq_suggestion + random.uniform(0.5, self.jitter)
            logger.info(
                "Using Groq suggested wait | groq={g:.1f}s | total={t:.1f}s",
                g=groq_suggestion,
                t=wait,
            )
            return wait

        # Exponential: 5s → 10s → 20s → 40s (capped at max_delay)
        exponential = min(self.base_delay * (2 ** attempt), self.max_delay)
        return exponential + random.uniform(0, self.jitter)

    def _parse_groq_wait(self, error_message: str) -> float | None:
        """Extract 'try again in 24.44s' from Groq error message."""
        match = re.search(
            r"try again in (\d+(?:\.\d+)?)s",
            error_message,
            re.IGNORECASE,
        )
        return float(match.group(1)) if match else None


# ══════════════════════════════════════════════════════════════════
# TOKEN ESTIMATOR
# ══════════════════════════════════════════════════════════════════

class TokenEstimator:
    """
    Fast token estimation for pre-call rate limit checks.
    Uses tiktoken when available, otherwise chars/4 approximation.
    Adds a 20% buffer to account for underestimation.
    """

    _encoder = None  # Lazy-loaded tiktoken encoder

    @classmethod
    def estimate(cls, *texts: str) -> int:
        """
        Estimate total tokens across one or more text strings.
        Adds 20% buffer for safety.
        """
        combined = " ".join(texts)
        raw      = cls._count(combined)
        buffered = int(raw * 1.2)
        return buffered

    @classmethod
    def _count(cls, text: str) -> int:
        """Raw token count using tiktoken or fallback."""
        try:
            if cls._encoder is None:
                import tiktoken
                cls._encoder = tiktoken.get_encoding("cl100k_base")
            return len(cls._encoder.encode(text))
        except Exception:
            return max(1, len(text) // 4)


# ══════════════════════════════════════════════════════════════════
# LLM SERVICE  (your original class, upgraded)
# ══════════════════════════════════════════════════════════════════

class LLMService:
    """
    Unified LLM service supporting free and paid providers.

    Additions over original:
    - TokenBucketRateLimiter  → pre-emptive TPM/RPM gating
    - RetryHandler            → automatic backoff on 429
    - TokenEstimator          → pre-call token estimation
    - Usage stats             → get_stats() method
    """

    def __init__(self):
        self._client: BaseChatModel | None = None
        self._provider = settings.LLM_PROVIDER
        self._model    = settings.active_model

        # ── NEW: rate limiting + retry ─────────────────────────────
        self._rate_limiter = TokenBucketRateLimiter(
            provider=self._provider
        )
        self._retry_handler = RetryHandler(
            max_retries=4,
            base_delay=5.0,
            max_delay=65.0,
        )


        # ── Stats ──────────────────────────────────────────────────
        self._total_tokens   = 0
        self._total_requests = 0
        self._total_retries  = 0
        self._total_pre_wait = 0.0
        self.session_token_usage = {}

        logger.info(
            "LLMService created | provider={p} | model={m}",
            p=self._provider,
            m=self._model,
        )

    # ─── Your original client creation (unchanged) ─────────────────

    @property
    def client(self) -> BaseChatModel:
        """Lazy-initialized LLM client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> BaseChatModel:
        """
        Creates the appropriate LangChain client based on provider setting.
        """
        provider = self._provider
        logger.info(
            "Initializing LLM client | provider={p} | model={m}",
            p=provider,
            m=self._model,
        )
        if provider == "groq":
            return self._create_groq_client()
        elif provider == "gemini":
            return self._create_gemini_client()
        else:
            raise ValueError(
                f"Unsupported LLM provider: '{provider}'. "
                f"Choose from: groq, gemini"
            )

    def _create_groq_client(self) -> BaseChatModel:
        """Creates Groq client (free tier)."""
        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set.\n"
                "Get your free key at: https://console.groq.com\n"
                "Then add to .env: GROQ_API_KEY=gsk_..."
            )
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError(
                "langchain-groq not installed. "
                "Run: pip install langchain-groq"
            )
        logger.info(
            "✅ Groq client ready | model={m} | free_tier=True",
            m=settings.GROQ_MODEL,
        )
        return ChatGroq(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0.3,
            max_retries=0,   # ← We handle retries ourselves now
            timeout=60,
        )

    def _create_gemini_client(self) -> BaseChatModel:
        """Creates Google Gemini client (free tier)."""
        if not settings.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set.\n"
                "Get your free key at: https://aistudio.google.com\n"
                "Then add to .env: GEMINI_API_KEY=AIza..."
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai not installed. "
                "Run: pip install langchain-google-genai"
            )
        logger.info(
            "✅ Gemini client ready | model={m} | free_tier=True",
            m=settings.GEMINI_MODEL,
        )
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.3,
            max_retries=0,   # ← We handle retries ourselves now
            convert_system_message_to_human=True,
        )

    # ─── Core Invoke Method (upgraded with rate limit + retry) ─────

    async def invoke(
        self,
        system_prompt: str,
        user_message: str,
        temperature_override: float | None = None,
        expected_output_tokens: int = 800,
    ) -> dict[str, Any]:
        """
        Invokes the LLM with automatic rate limiting and retry.

        Args:
            system_prompt: Instructions/context for the LLM
            user_message: The actual task/query
            temperature_override: Optional per-call temperature
            expected_output_tokens: Hint for pre-call token estimation

        Returns:
            dict with 'content', 'tokens_used', 'latency_ms',
                       'retries', 'pre_wait_seconds'
        """
        start_time = time.time()

        # ── Estimate tokens before calling ────────────────────────
        estimated_tokens = TokenEstimator.estimate(
            system_prompt,
            user_message,
        ) + expected_output_tokens

        # ── Pre-emptive rate limit gate ────────────────────────────
        # Blocks here if we're too close to the TPM/RPM limit
        pre_wait = await self._rate_limiter.acquire(estimated_tokens)
        self._total_pre_wait += pre_wait

        # ── Build messages (your original logic, unchanged) ────────
        messages = self._build_messages(system_prompt, user_message)

        # ── Apply temperature override ─────────────────────────────
        client = self.client
        if temperature_override is not None:
            client = client.bind(temperature=temperature_override)

        logger.debug(
            "Invoking LLM | provider={p} | model={m} | "
            "sys_len={sl} | usr_len={ul} | est_tokens={et}",
            p=self._provider,
            m=self._model,
            sl=len(system_prompt),
            ul=len(user_message),
            et=estimated_tokens,
        )

        # ── Retry loop ─────────────────────────────────────────────
        last_error: Exception | None = None

        for attempt in range(self._retry_handler.max_retries + 1):
            try:
                response   = await client.ainvoke(messages)
                content    = self._extract_content(response)
                tokens     = self._extract_tokens(
                    response, system_prompt, user_message
                )
                latency_ms = round((time.time() - start_time) * 1000, 2)

                # Update rate limiter with actual token count
                self._rate_limiter.record_actual_tokens(tokens)

                # Update stats
                self._total_tokens   += tokens
                self._total_requests += 1

                logger.info(
                    "LLM response | tokens={t} | latency={l}ms | "
                    "provider={p} | attempt={a}",
                    t=tokens,
                    l=latency_ms,
                    p=self._provider,
                    a=attempt + 1,
                )

                return {
                    "content":          content,
                    "tokens_used":      tokens,
                    "latency_ms":       latency_ms,
                    "retries":          attempt,
                    "pre_wait_seconds": pre_wait,
                }

            except Exception as e:
                last_error  = e
                error_str   = str(e)

                if self._retry_handler.is_rate_limit_error(e):
                    # ── 429 handling ───────────────────────────────
                    if attempt >= self._retry_handler.max_retries:
                        # All retries exhausted
                        logger.error(
                            "Rate limit retries exhausted | "
                            "attempts={a} | provider={p}",
                            a=attempt + 1,
                            p=self._provider,
                        )
                        break

                    wait = self._retry_handler.get_wait_time(
                        attempt=attempt,
                        error_message=error_str,
                    )
                    self._total_retries += 1

                    logger.warning(
                        "Rate limit hit | attempt={a}/{m} | "
                        "waiting={w:.1f}s | provider={p}",
                        a=attempt + 1,
                        m=self._retry_handler.max_retries,
                        w=wait,
                        p=self._provider,
                    )

                    await asyncio.sleep(wait)

                    # Re-acquire rate limiter slot after waiting
                    await self._rate_limiter.acquire(estimated_tokens)

                else:
                    # ── Non-429 error — don't retry ────────────────
                    hint = self._get_error_hint(error_str)
                    logger.error(
                        "LLM invoke failed | provider={p} | error={e}",
                        p=self._provider,
                        e=error_str[:200],
                    )
                    raise RuntimeError(
                        f"LLM call failed [{self._provider}]: {error_str}\n{hint}"
                    ) from e

        # Exhausted all retries on rate limit errors
        hint = self._get_error_hint(str(last_error))
        raise RuntimeError(
            f"LLM call failed [{self._provider}]: {last_error}\n{hint}"
        ) from last_error

    # ─── Your original helpers (unchanged except _extract_content) ─

    def _extract_content(self, response: Any) -> str:
        """Extract string content from LangChain response."""
        content = response.content
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return content

    def _build_messages(
        self,
        system_prompt: str,
        user_message:  str,
    ) -> list:
        """
        Builds message list appropriate for each provider.
        Unchanged from your original implementation.
        """
        if self._provider == "gemini":
            combined = (
                f"[INSTRUCTIONS]\n{system_prompt}\n\n"
                f"[TASK]\n{user_message}"
            )
            return [HumanMessage(content=combined)]

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

    def _extract_tokens(
        self,
        response: Any,
        system_prompt: str,
        user_message: str,
    ) -> int:
        """
        Extracts token usage from response metadata.
        Unchanged from your original implementation.
        """
        try:
            meta  = response.response_metadata or {}
            usage = meta.get("token_usage") or meta.get("usage", {})
            if usage:
                total = (
                    usage.get("total_tokens")
                    or usage.get("input_tokens", 0)
                    + usage.get("output_tokens", 0)
                )
                if total:
                    return int(total)
        except AttributeError:
            pass

        return self._estimate_tokens(
            system_prompt + user_message + str(response.content)
        )

    def _estimate_tokens(self, text: str) -> int:
        """Unchanged from your original implementation."""
        return TokenEstimator._count(text)

    def _get_error_hint(self, error_msg: str) -> str:
        """Unchanged from your original implementation."""
        error_lower = error_msg.lower()

        if any(k in error_lower for k in ["authentication", "api key", "unauthorized"]):
            key_hints = {
                "groq":   "Check GROQ_API_KEY in .env → https://console.groq.com",
                "gemini": "Check GEMINI_API_KEY in .env → https://aistudio.google.com",
            }
            return f"💡 Hint: {key_hints.get(self._provider, 'Check your API key in .env')}"

        if any(k in error_lower for k in ["rate limit", "quota", "429"]):
            free_hints = {
                "groq":   "Groq free tier: 30 req/min. Wait a moment and retry.",
                "gemini": "Gemini free tier: 15 req/min. Wait a moment and retry.",
            }
            return f"💡 Hint: {free_hints.get(self._provider, 'Rate limit hit. Wait and retry.')}"

        if "connection" in error_lower or "connect" in error_lower:
            return "💡 Hint: Check your internet connection."

        if "model" in error_lower and "not found" in error_lower:
            return f"💡 Hint: Model '{self._model}' may not exist. Check model name in .env."

        return ""

    def count_tokens(self, text: str) -> int:
        """Public token counter — unchanged from original."""
        return self._estimate_tokens(text)

    def get_stats(self) -> dict[str, Any]:
        """
        Returns cumulative usage stats.
        NEW: added in this upgrade.
        """
        return {
            "total_tokens":    self._total_tokens,
            "total_requests":  self._total_requests,
            "total_retries":   self._total_retries,
            "total_pre_wait":  round(self._total_pre_wait, 2),
            "provider":        self._provider,
            "model":           self._model,
            "current_window":  self._rate_limiter.current_usage(),
        }

    def track_session_tokens(
            self,
            session_id: str,
            tokens: int,
    ):
        """
        Track cumulative token usage per session.
        """

        if not session_id:
            return 0

        current = self.session_token_usage.get(
            session_id,
            0,
        )

        updated = current + tokens

        self.session_token_usage[session_id] = updated

        return updated

# ══════════════════════════════════════════════════════════════════
# SINGLETON  (unchanged)
# ══════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    """Returns the LLMService singleton."""
    return LLMService()