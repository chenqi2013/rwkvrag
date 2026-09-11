import pytest
from llamaindex_retrieval.document_retrieval import document_groups
from llamaindex_retrieval.lexical_index import LexicalResult, LexicalIndex
from llamaindex_retrieval.config import Settings


def hit(node, doc):
    return LexicalResult(node_id=node, document_id=doc, text="原文", metadata={}, score=1)


@pytest.mark.asyncio
async def test_document_votes_deduplicate_then_reopen_only_winning_documents():
    class Index:
        calls = []
        def search_chunks(self, query, **kwargs):
            self.calls.append((query, kwargs))
            if kwargs.get("collapse_documents"):
                return ([hit("a1", "a"), hit("a2", "a"), hit("b1", "b")]
                        if query == "问题一" else [hit("b2", "b"), hit("c1", "c")])
            assert kwargs["document_ids"] == ["b"]
            return [hit("b-deep", "b")]
    index = Index()
    groups, trace = await document_groups(index, ["问题一", "问题二"], candidate_k=20,
        knowledge_base_id="kb", document_limit=1)
    assert trace["selected_document_ids"] == ["b"]
    assert sum(v["document_id"] == "a" for v in trace["rrf_votes"]) == 1
    assert [g[0].node_id for g in groups] == ["b-deep", "b-deep"]
    assert len(index.calls) == 4
    assert all(kwargs["knowledge_base_id"] == "kb" for _, kwargs in index.calls)


@pytest.mark.asyncio
async def test_empty_document_selection_does_not_query_all_documents():
    class Index:
        def search_chunks(self, query, **kwargs):
            assert kwargs.get("collapse_documents")
            return []
    groups, _ = await document_groups(Index(), ["问题"], candidate_k=20,
        knowledge_base_id=None, document_limit=5)
    assert groups == [[]]


@pytest.mark.asyncio
async def test_expansion_refuses_a_document_outside_filter():
    class Index:
        def search_chunks(self, query, **kwargs):
            return [hit("a", "a")] if kwargs.get("collapse_documents") else [hit("b", "b")]
    with pytest.raises(ValueError, match="unselected document"):
        await document_groups(Index(), ["问题"], candidate_k=20,
            knowledge_base_id=None, document_limit=1)


def test_opensearch_document_filter_and_collapse_do_not_change_query_fields():
    class Client:
        requests = []
        def search(self, **kwargs):
            self.requests.append(kwargs["body"])
            return {"hits": {"hits": []}}
    client = Client()
    class ExistingIndex(LexicalIndex):
        def ensure_index(self):
            pass
    index = ExistingIndex(Settings(), client=client)
    index.search_chunks("查询对象", candidate_k=20, knowledge_base_id="kb")
    index.search_chunks("查询对象", candidate_k=20, knowledge_base_id="kb", collapse_documents=True)
    index.search_chunks("查询对象", candidate_k=20, knowledge_base_id="kb", document_ids=["a", "b"])
    base, collapsed, scoped = client.requests
    assert collapsed == {**base, "collapse": {"field": "document_id"}}
    assert scoped["query"]["bool"]["must"] == base["query"]["bool"]["must"]
    assert scoped["query"]["bool"]["filter"] == [{"term": {"knowledge_base_id": "kb"}}, {"terms": {"document_id": ["a", "b"]}}]


@pytest.mark.parametrize("collapse", [False, True])
def test_long_queries_keep_tail_tokens_filters_and_unique_rank_votes(collapse):
    class Client:
        def __init__(self): self.requests = []
        def search(self, **kwargs):
            body = kwargs["body"]; self.requests.append(body)
            assert len(body["query"]["bool"]["must"][0]["multi_match"]["query"].split()) <= 128
            n = len(self.requests)
            def result(node, doc):
                return {"_source": {"node_id": node, "document_id": doc, "text": "完整证据", "metadata": {}}, "_score": 99}
            return {"hits": {"hits": [result("shared", "shared-doc"), result(f"tail-{n}", f"tail-doc-{n}")]}}
    class ExistingIndex(LexicalIndex):
        def ensure_index(self): pass
    client = Client(); index = ExistingIndex(Settings(), client=client)
    tokens = [f"term{i}" for i in range(300)]
    results = index.search_chunks(" ".join(tokens), candidate_k=10, knowledge_base_id="kb",
        document_ids=["shared-doc", "tail-doc-1", "tail-doc-2", "tail-doc-3"], collapse_documents=collapse)
    assert len(client.requests) == 3
    assert [t for b in client.requests for t in b["query"]["bool"]["must"][0]["multi_match"]["query"].split()] == tokens
    assert all(b["query"]["bool"]["filter"] == client.requests[0]["query"]["bool"]["filter"] for b in client.requests)
    assert all(bool(b.get("collapse")) == collapse for b in client.requests)
    assert results[0].node_id == "shared" and len(results) == 4
    assert results[0].score == pytest.approx(3 / 61)
    assert "tail-3" in [r.node_id for r in results]


@pytest.mark.parametrize('collapse', [False, True])
def test_long_query_rejects_conflicting_node_identity(collapse):
    class Client:
        calls = 0
        def search(self, **kwargs):
            self.calls += 1
            return {'hits': {'hits': [{'_source': {'node_id': 'same', 'document_id': 'doc',
                'text': f'version-{self.calls}', 'metadata': {}}, '_score': 1}]}}
    class ExistingIndex(LexicalIndex):
        def ensure_index(self): pass
    index = ExistingIndex(Settings(), client=Client())
    with pytest.raises(ValueError, match='conflicting source identity'):
        index.search_chunks(' '.join(f't{i}' for i in range(129)), candidate_k=5,
                            collapse_documents=collapse)
