from typing import Any

import pytest

from llamaindex_retrieval.routers.admin import list_search_history


class FakeSearchHistoryRepository:
    def __init__(self) -> None:
        self.arguments: tuple[int, int] | None = None

    async def list_search_tests(self, *, page: int, page_size: int) -> dict[str, Any]:
        self.arguments = (page, page_size)
        return {"items": [], "total": 137, "page": page, "page_size": page_size}


@pytest.mark.asyncio
async def test_search_history_uses_server_side_pagination() -> None:
    repository = FakeSearchHistoryRepository()

    response = await list_search_history(
        page=3,
        page_size=50,
        repo=repository,  # type: ignore[arg-type]
    )

    assert repository.arguments == (3, 50)
    assert response == {"items": [], "total": 137, "page": 3, "page_size": 50}
