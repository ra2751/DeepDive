"""
Typed application settings for DeepDive AI.

Settings are read from environment variables (and a local .env file) only when
`get_settings()` is called, never at import time, so importing project modules
needs no API keys and makes no network calls.
"""

import os
from functools import lru_cache
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _env(environ, *names, default=None):
    for name in names:
        value = environ.get(name)
        if value not in (None, ""):
            return value
    return default


def _opt_number(environ, name, default, cast):
    raw = environ.get(name)
    if raw in (None, ""):
        return default
    if raw.strip().lower() in ("none", "off", "unlimited"):
        return None
    try:
        return cast(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


class Settings(BaseModel):
    # --- Mesh API (all AI inference) ---
    mesh_api_key: Optional[str] = None
    mesh_base_url: str = "https://api.meshapi.ai/v1"
    # Default chat models (overridable per role in .env);
    # availability via Mesh is confirmed by the G1 catalogue probe.
    mesh_model_supervisor: str = "openai/gpt-4o-mini"
    mesh_model_generator: str = "openai/gpt-4o-mini"
    mesh_model_verifier: str = "google/gemini-2.5-flash"
    mesh_embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536
    mesh_call_timeout_s: float = 60.0
    mesh_max_attempts: int = 3
    # Mesh gateway response cache (documented: hits carry "X-Cache: HIT" and are not billed).
    # "default" leaves Mesh's caching on; "no-store" sends the documented "X-Mesh-Cache: no-store"
    # header so every request reaches the provider (e.g. for independent evaluation re-runs).
    mesh_cache_mode: Literal["default", "no-store"] = "default"

    # --- Web search (non-model provider) ---
    tavily_api_key: Optional[str] = None
    web_search_provider: Literal["tavily", "replay"] = "tavily"
    web_search_timeout_s: float = 15.0
    web_search_max_attempts: int = 2
    web_search_max_results: int = 3
    web_search_depth: Literal["basic", "advanced"] = "basic"
    web_search_replay_file: Optional[str] = None

    # --- Milvus ---
    milvus_endpoint: Optional[str] = None
    milvus_api_key: Optional[str] = None
    milvus_collection: str = "ai_research_assistant"
    # Result of the G2 / LT-4 compatibility check between Mesh embeddings and the
    # vectors already stored in the collection.
    milvus_embedding_compat: Literal["unverified", "verified", "incompatible"] = "unverified"

    # --- Research-run budget (approved provisional limits, 2026-10-16) ---
    max_llm_calls: Optional[int] = 25
    max_embedding_calls: Optional[int] = 10
    max_web_searches: Optional[int] = 8
    max_tool_calls: Optional[int] = 30
    max_retry_rounds: int = 2
    max_run_seconds: Optional[float] = 300.0
    max_mesh_cost_usd: Optional[float] = 0.50
    cost_estimate_safety_factor: float = 1.25
    mesh_pricing_file: str = "config/mesh_pricing.yaml"

    # Claims extracted per research question (provisional; lower = fewer verifier calls and lower
    # cost/latency, but possibly less coverage). User-supplied fact-check claims are never cut silently.
    max_research_claims: int = 8
    max_fact_check_claims: int = 15

    # --- Ingestion budget (approved 2026-10-16), separate from research runs ---
    max_ingest_embedding_calls: int = 50
    embedding_batch_size: int = 64
    max_upload_mb: float = 10.0

    @property
    def max_ingest_chunks(self) -> int:
        return self.max_ingest_embedding_calls * self.embedding_batch_size

    def model_for_role(self, role: str) -> str:
        models = {
            "supervisor": self.mesh_model_supervisor,
            "generator": self.mesh_model_generator,
            "verifier": self.mesh_model_verifier,
        }
        if role not in models:
            raise ConfigError(f"Unknown model role {role!r}; expected one of {sorted(models)}")
        return models[role]

    def require(self, *fields: str) -> None:
        """Raise ConfigError naming the environment variables that are missing."""
        env_names = {
            "mesh_api_key": "MESH_API_KEY",
            "tavily_api_key": "TAVILY_API_KEY",
            "milvus_endpoint": "MILVUS_ENDPOINT (or Milvus_ENDPOINT)",
            "milvus_api_key": "MILVUS_API_KEY (or Milvus_API_KEY)",
        }
        missing = [env_names.get(f, f.upper()) for f in fields if not getattr(self, f)]
        if missing:
            raise ConfigError("Missing required configuration: " + ", ".join(missing))


def load_settings(environ=None) -> Settings:
    if environ is None:
        load_dotenv()
        environ = os.environ
    e = environ
    default_model = _env(e, "MESH_MODEL")
    base = Settings()
    return Settings(
        mesh_api_key=_env(e, "MESH_API_KEY"),
        mesh_base_url=_env(e, "MESH_BASE_URL", default=base.mesh_base_url),
        mesh_model_supervisor=_env(e, "MESH_MODEL_SUPERVISOR", default=default_model or base.mesh_model_supervisor),
        mesh_model_generator=_env(e, "MESH_MODEL_GENERATOR", default=default_model or base.mesh_model_generator),
        mesh_model_verifier=_env(e, "MESH_MODEL_VERIFIER", default=base.mesh_model_verifier),
        mesh_embedding_model=_env(e, "MESH_EMBEDDING_MODEL", default=base.mesh_embedding_model),
        embedding_dim=_opt_number(e, "MESH_EMBEDDING_DIM", base.embedding_dim, int),
        mesh_call_timeout_s=_opt_number(e, "MESH_CALL_TIMEOUT_S", base.mesh_call_timeout_s, float),
        mesh_max_attempts=_opt_number(e, "MESH_MAX_ATTEMPTS", base.mesh_max_attempts, int),
        mesh_cache_mode=_env(e, "MESH_CACHE_MODE", default=base.mesh_cache_mode),
        tavily_api_key=_env(e, "TAVILY_API_KEY"),
        web_search_provider=_env(e, "WEB_SEARCH_PROVIDER", default=base.web_search_provider),
        web_search_timeout_s=_opt_number(e, "WEB_SEARCH_TIMEOUT_S", base.web_search_timeout_s, float),
        web_search_max_attempts=_opt_number(e, "WEB_SEARCH_MAX_ATTEMPTS", base.web_search_max_attempts, int),
        web_search_max_results=_opt_number(e, "WEB_SEARCH_MAX_RESULTS", base.web_search_max_results, int),
        web_search_depth=_env(e, "WEB_SEARCH_DEPTH", default=base.web_search_depth),
        web_search_replay_file=_env(e, "WEB_SEARCH_REPLAY_FILE"),
        milvus_endpoint=_env(e, "MILVUS_ENDPOINT", "Milvus_ENDPOINT"),
        milvus_api_key=_env(e, "MILVUS_API_KEY", "Milvus_API_KEY"),
        milvus_collection=_env(e, "MILVUS_COLLECTION", default=base.milvus_collection),
        milvus_embedding_compat=_env(e, "MILVUS_EMBEDDING_COMPAT", default=base.milvus_embedding_compat),
        max_llm_calls=_opt_number(e, "MAX_LLM_CALLS", base.max_llm_calls, int),
        max_embedding_calls=_opt_number(e, "MAX_EMBEDDING_CALLS", base.max_embedding_calls, int),
        max_web_searches=_opt_number(e, "MAX_WEB_SEARCHES", base.max_web_searches, int),
        max_tool_calls=_opt_number(e, "MAX_TOOL_CALLS", base.max_tool_calls, int),
        max_retry_rounds=_opt_number(e, "MAX_RETRY_ROUNDS", base.max_retry_rounds, int),
        max_run_seconds=_opt_number(e, "MAX_RUN_SECONDS", base.max_run_seconds, float),
        max_mesh_cost_usd=_opt_number(e, "MAX_MESH_COST_USD", base.max_mesh_cost_usd, float),
        cost_estimate_safety_factor=_opt_number(e, "COST_ESTIMATE_SAFETY_FACTOR", base.cost_estimate_safety_factor, float),
        mesh_pricing_file=_env(e, "MESH_PRICING_FILE", default=base.mesh_pricing_file),
        max_research_claims=_opt_number(e, "MAX_RESEARCH_CLAIMS", base.max_research_claims, int),
        max_fact_check_claims=_opt_number(e, "MAX_FACT_CHECK_CLAIMS", base.max_fact_check_claims, int),
        max_ingest_embedding_calls=_opt_number(e, "MAX_INGEST_EMBEDDING_CALLS", base.max_ingest_embedding_calls, int),
        embedding_batch_size=_opt_number(e, "EMBEDDING_BATCH_SIZE", base.embedding_batch_size, int),
        max_upload_mb=_opt_number(e, "MAX_UPLOAD_MB", base.max_upload_mb, float),
    )


_override: Optional[Settings] = None


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return load_settings()


def get_settings() -> Settings:
    return _override if _override is not None else _cached_settings()


def set_settings_override(settings: Optional[Settings]) -> None:
    """Use explicit settings (tests, scripts) instead of the environment."""
    global _override
    _override = settings


def reset_settings_cache() -> None:
    _cached_settings.cache_clear()
