import json

from llamaindex_retrieval.config import get_settings


def test_explicit_json_preserves_null_and_does_not_read_dotenv(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"rwkvos_state_id": None,
        "rwkvos_input_token_limit": None, "rwkvos_stop_tokens": None}))
    (tmp_path / ".env").write_text("RWKVRAG_NATIVE_MODEL=stale-model\nRWKVRAG_RWKVOS_STATE_ID=stale-state\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RWKVRAG_SETTINGS_FILE", str(path))
    monkeypatch.setenv("RWKVRAG_RWKVOS_STATE_ID", "inherited-state")
    monkeypatch.delenv("RWKVRAG_NATIVE_MODEL", raising=False)
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.rwkvos_state_id is None
        assert settings.rwkvos_input_token_limit is None
        assert settings.rwkvos_stop_tokens is None
        assert settings.native_model != "stale-model"
    finally:
        get_settings.cache_clear()
