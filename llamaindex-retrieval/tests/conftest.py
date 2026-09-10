import os

import pytest

from llamaindex_retrieval.config import Settings


@pytest.fixture(autouse=True)
def isolate_runtime_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name in os.environ:
        if name.startswith("RWKVRAG_"):
            monkeypatch.delenv(name)
