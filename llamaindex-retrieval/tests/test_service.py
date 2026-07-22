from typing import Any, cast

from llama_index.core.schema import TextNode

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.lexical_index import LexicalIndex, LexicalResult
from llamaindex_retrieval.service import SearchService


def result(document_id: str, score: float) -> LexicalResult:
    return LexicalResult(
        node_id=f"node-{document_id}",
        document_id=document_id,
        text=document_id,
        metadata={"document_id": document_id},
        score=score,
    )


class FakeIndices:
    @staticmethod
    def exists(*, index: str) -> bool:
        return True


class FakeOpenSearch:
    indices = FakeIndices()

    def __init__(self) -> None:
        self.search_body: dict[str, Any] = {}

    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.search_body = body
        return {
            "hits": {
                "hits": [
                    {
                        "_score": 3.0,
                        "_source": {
                            "node_id": "capital-node",
                            "document_id": "capital",
                            "title": "首都",
                            "text": "中华人民共和国的首都是北京。",
                            "metadata": {
                                "document_id": "capital",
                                "title": "首都",
                                "source": "wiki",
                            },
                        },
                    }
                ]
            }
        }


def test_select_results_deduplicates_documents_and_filters_low_scores() -> None:
    service = SearchService(
        Settings(relative_score_threshold=0.55),
        cast(Any, None),
    )
    selected = service._select_results(
        [
            result("capital", 1.0),
            result("capital", 0.8),
            result("country", 0.7),
            result("irrelevant", 0.2),
        ],
        top_k=5,
        min_score=0,
    )
    assert [item.document_id for item in selected] == ["capital", "country"]


def test_lexical_index_searches_chinese_and_titles() -> None:
    client = FakeOpenSearch()
    index = LexicalIndex(Settings(), client=cast(Any, client))
    results = index.search("中国首都在哪里", candidate_k=5)
    assert results
    assert results[0].document_id == "capital"
    fields = client.search_body["query"]["bool"]["must"][0]["multi_match"]["fields"]
    assert fields == ["body_tokens", "title_tokens^2", "tags_tokens^1.5"]


def test_opensearch_record_contains_pretokenized_fields() -> None:
    index = LexicalIndex(Settings(), client=cast(Any, FakeOpenSearch()))
    record = index._record(
        TextNode(
            text="中华人民共和国的首都是北京。",
            metadata={"document_id": "capital", "title": "首都", "source": "wiki"},
        )
    )
    assert "北京" in record["body_tokens"]
    assert record["title_tokens"] == "首都"


def test_source_item_returns_complete_qa_answer() -> None:
    full_answer = "完整答案" * 500
    item = SearchService._source_item(
        LexicalResult(
            node_id="qa-node",
            document_id="qa-1",
            text="只命中了答案的一小部分",
            metadata={
                "document_id": "qa-1",
                "source": "uploaded-document",
                "title": "问题标题",
                "full_answer": full_answer,
                "question_id": "1",
            },
            score=0.9,
        )
    )

    assert item.snippet == full_answer
    assert item.metadata["question_id"] == "1"
    assert "full_answer" not in item.metadata
