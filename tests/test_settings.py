"""T04: settings are lazy, validated and default to the approved limits."""

import pytest

from config.settings import ConfigError, load_settings


def test_defaults_match_approved_limits():
    s = load_settings(environ={})
    assert (s.max_llm_calls, s.max_embedding_calls, s.max_web_searches, s.max_tool_calls) == (25, 10, 8, 30)
    assert (s.max_retry_rounds, s.max_run_seconds, s.max_mesh_cost_usd) == (2, 300.0, 0.50)
    assert (s.max_ingest_embedding_calls, s.embedding_batch_size) == (50, 64)
    assert s.max_ingest_chunks == 3200
    assert s.mesh_base_url == "https://api.meshapi.ai/v1"
    assert s.mesh_embedding_model == "openai/text-embedding-3-small"


def test_missing_key_is_reported_by_name_not_value():
    s = load_settings(environ={})
    with pytest.raises(ConfigError, match="MESH_API_KEY"):
        s.require("mesh_api_key")


def test_environment_overrides_and_legacy_milvus_names():
    s = load_settings(environ={"MAX_LLM_CALLS": "12", "MAX_MESH_COST_USD": "off", "MESH_MODEL": "x/y",
                               "Milvus_ENDPOINT": "https://legacy", "MILVUS_EMBEDDING_COMPAT": "verified"})
    assert s.max_llm_calls == 12 and s.max_mesh_cost_usd is None
    assert s.mesh_model_supervisor == s.mesh_model_generator == "x/y"
    assert s.milvus_endpoint == "https://legacy" and s.milvus_embedding_compat == "verified"


def test_invalid_number_is_rejected():
    with pytest.raises(ConfigError, match="MAX_LLM_CALLS"):
        load_settings(environ={"MAX_LLM_CALLS": "lots"})
