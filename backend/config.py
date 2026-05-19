"""
Application Configuration - Updated for Free Model Support.

Supports providers:
- groq     (FREE - Llama 3, Mixtral via Groq API)
- gemini   (FREE - Google Gemini via AI Studio)
- ollama   (FREE - Local models, no API key)

"""

from functools import lru_cache
from pathlib import Path
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application settings loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ─── Provider Selection ───────────────────────────────────────
    LLM_PROVIDER: str = Field(
        default="groq",
        description="LLM provider: groq | gemini | ollama ",
    )

    # ─── Groq Settings (FREE) ────────────────────────────────────
    GROQ_API_KEY: str = Field(default="")
    GROQ_MODEL: str = Field(default="llama-3.1-8b-instant")

    # ─── Gemini Settings (FREE) ───────────────────────────────────
    GEMINI_API_KEY: str = Field(default="")
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash")


    # ─── FastAPI ──────────────────────────────────────────────────
    BACKEND_HOST: str = Field(default="localhost")
    BACKEND_PORT: int = Field(default=8000)
    API_PREFIX: str = Field(default="/api/v1")
    DEBUG: bool = Field(default=False)

    # ─── Frontend ─────────────────────────────────────────────────
    FRONTEND_PORT: int = Field(default=8501)

    # ─── Database ─────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./data/app.db"
    )

    # ─── ChromaDB ─────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = Field(default="./data/chroma_db")
    CHROMA_COLLECTION_NAME: str = Field(default="prompt_knowledge")

    # ─── Logging ──────────────────────────────────────────────────
    LOG_LEVEL: str = Field(default="INFO")
    LOG_DIR: str = Field(default="./logs")

    # ─── Validators ───────────────────────────────────────────────
    @field_validator("LLM_PROVIDER")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"groq", "gemini", "ollama"}
        v = v.lower().strip()
        if v not in allowed:
            raise ValueError(
                f"LLM_PROVIDER must be one of {allowed}, got: '{v}'"
            )
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper().strip()
        if v not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return v

    @model_validator(mode="after")
    def validate_api_keys(self) -> "Settings":
        """
        Warns if selected provider has no API key configured.
        Ollama is excluded since it runs locally with no key.
        """
        warnings = {
            "groq":      (self.GROQ_API_KEY,     "GROQ_API_KEY",     "https://console.groq.com"),
            "gemini":    (self.GEMINI_API_KEY,    "GEMINI_API_KEY",   "https://aistudio.google.com"),
        }

        if self.LLM_PROVIDER in warnings:
            key_value, key_name, signup_url = warnings[self.LLM_PROVIDER]
            if not key_value:
                print(
                    f"\n⚠️  WARNING: LLM_PROVIDER='{self.LLM_PROVIDER}' "
                    f"but {key_name} is not set.\n"
                    f"   Get your free key at: {signup_url}\n"
                    f"   Then add to .env: {key_name}=your_key_here\n"
                )

        if self.LLM_PROVIDER == "ollama":
            print(
                f"\nℹ️  Using Ollama (local) | model={self.OLLAMA_MODEL} | "
                f"url={self.OLLAMA_BASE_URL}\n"
                f"   Make sure Ollama is running: ollama serve\n"
                f"   Pull your model first:       ollama pull {self.OLLAMA_MODEL}\n"
            )

        return self

    # ─── Computed Properties ──────────────────────────────────────
    @property
    def active_model(self) -> str:
        """Returns the model name for the active provider."""
        model_map = {
            "groq":      self.GROQ_MODEL,
            "gemini":    self.GEMINI_MODEL,

        }
        return model_map.get(self.LLM_PROVIDER, "unknown")

    @property
    def active_api_key(self) -> str:
        """Returns the API key for the active provider."""
        key_map = {
            "groq":      self.GROQ_API_KEY,
            "gemini":    self.GEMINI_API_KEY,

        }
        return key_map.get(self.LLM_PROVIDER, "")

    @property
    def active_llm_model(self) -> str:
        """Alias for active_model (backward compatibility)."""
        return self.active_model

    @property
    def backend_url(self) -> str:
        return f"http://{self.BACKEND_HOST}:{self.BACKEND_PORT}"

    @property
    def api_base_url(self) -> str:
        return f"{self.backend_url}{self.API_PREFIX}"

    @property
    def is_free_provider(self) -> bool:
        """Returns True if using a free provider."""
        return self.LLM_PROVIDER in {"groq", "gemini", "ollama"}

    def ensure_directories(self) -> None:
        """Creates required directories if they don't exist."""
        for d in [
            Path(self.LOG_DIR),
            Path("./data"),
            Path("./data/knowledge"),
            Path(self.CHROMA_PERSIST_DIR),
        ]:
            d.mkdir(parents=True, exist_ok=True)

    # ─── Pipeline Optimization ──────────────────────────────────────
    ENABLE_LLM_SCORING: bool = Field(default=False)
    ENABLE_DRIFT_DETECTION: bool = Field(default=False)
    ENABLE_DEEP_REASONING: bool = Field(default=False)

    DEFAULT_FAST_MODE: bool = Field(default=True)

    MAX_PIPELINE_TOKENS: int = Field(default=12000)
    MAX_STAGE_TOKENS: int = Field(default=3500)

    DEFAULT_PROMPT_STYLES: list[str] = Field(
        default=[
            "chain_of_thought",
            "structured",
        ]
    )
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns cached Settings singleton."""
    return Settings()


# Module-level singleton
settings = get_settings()