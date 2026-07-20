from types import SimpleNamespace

import pytest
from qdrant_client.http.exceptions import ResponseHandlingException

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.qdrant_admin import QdrantAdmin, QdrantUnavailableError


class FakeQdrantClient:
    def __init__(self, *, available: bool) -> None:
        self.available = available
        self.closed = False

    def get_collections(self) -> SimpleNamespace:
        if not self.available:
            raise ResponseHandlingException(ConnectionRefusedError())
        return SimpleNamespace(collections=[SimpleNamespace(name="knowledge")])

    def close(self) -> None:
        self.closed = True


def test_qdrant_admin_recovers_by_creating_a_new_client(monkeypatch: pytest.MonkeyPatch) -> None:
    all_clients = [FakeQdrantClient(available=False), FakeQdrantClient(available=True)]
    pending_clients = list(all_clients)

    def create_client(**_: object) -> FakeQdrantClient:
        return pending_clients.pop(0)

    monkeypatch.setattr(
        "llamaindex_retrieval.qdrant_admin.qdrant_client.QdrantClient",
        create_client,
    )
    admin = QdrantAdmin(Settings())

    with pytest.raises(QdrantUnavailableError, match="Qdrant 暂时不可用"):
        admin.health()

    recovered = admin.health()

    assert recovered["ok"] is True
    assert recovered["collections"] == 1
    assert not pending_clients
    assert all(client.closed for client in all_clients)
