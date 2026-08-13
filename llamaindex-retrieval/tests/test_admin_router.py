from typing import Any

import pytest

from llamaindex_retrieval.routers.admin import list_search_history


class FakeSearchHistoryRepository:
    def __init__(self) -> None:
        self.arguments: tuple[int, int, str | None] | None = None

    async def list_search_tests(
        self,
        *,
        page: int,
        page_size: int,
        answer_status: str | None,
    ) -> dict[str, Any]:
        self.arguments = (page, page_size, answer_status)
        return {"items": [], "total": 137, "page": page, "page_size": page_size}


@pytest.mark.asyncio
async def test_search_history_uses_server_side_pagination() -> None:
    repository = FakeSearchHistoryRepository()

    response = await list_search_history(
        page=3,
        page_size=50,
        answer_status="refused",
        repo=repository,  # type: ignore[arg-type]
    )

    assert repository.arguments == (3, 50, "refused")
    assert response == {"items": [], "total": 137, "page": 3, "page_size": 50}
