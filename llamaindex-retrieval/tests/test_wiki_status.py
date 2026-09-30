from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from llamaindex_retrieval.wiki import WikiService


@pytest.mark.asyncio
async def test_status_reports_running_settings_and_missing_bindings_without_generating():
    files = SimpleNamespace(count_documents=AsyncMock(side_effect=[5019, 0]))
    repo = SimpleNamespace(files=files)
    settings = SimpleNamespace(wiki_auto_generate=False, wiki_max_source_characters=24000)
    service = WikiService(settings, repo, None, None, None)
    status = await service.status()
    assert status == {"auto_generate": False, "generator_configured": False,
                      "max_source_characters": 24000, "ready_files": 5019,
                      "version_bound_files": 0, "unversioned_files": 5019}
    bound = files.count_documents.call_args_list[1].args[0]
    assert bound["last_indexed_revision"] == {"$type": "object"}
    assert bound["last_indexed_index_version"] == {"$type": "string", "$ne": ""}
    assert bound["status"] == "ready"


@pytest.mark.asyncio
async def test_status_does_not_claim_bound_sources_or_generated_drafts_are_reviewed():
    repo = SimpleNamespace(files=SimpleNamespace(count_documents=AsyncMock(side_effect=[4, 3])))
    settings = SimpleNamespace(wiki_auto_generate=True, wiki_max_source_characters=500)
    service = WikiService(settings, repo, None, object(), None)
    status = await service.status()
    assert status["version_bound_files"] == 3
    assert status["unversioned_files"] == 1
    assert status["auto_generate"] and status["generator_configured"]
    assert "semantic_reviewed" not in status and "ready_to_publish" not in status
