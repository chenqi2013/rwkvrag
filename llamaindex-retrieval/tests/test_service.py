from llama_index.core.schema import NodeWithScore, TextNode

from llamaindex_retrieval.config import Settings
from llamaindex_retrieval.components import InstructedOpenAIEmbedding
from llamaindex_retrieval.service import SearchService


def result(document_id: str, score: float) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(text=document_id, metadata={"document_id": document_id}),
        score=score,
    )


def test_select_results_deduplicates_documents_and_filters_low_scores() -> None:
    service = SearchService(
        settings=Settings(relative_score_threshold=0.55, max_chunks_per_document=1),
        index=object(),  # type: ignore[arg-type]
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
    assert [node.node.metadata["document_id"] for node in selected] == [
        "capital",
        "country",
    ]


def test_qwen3_defaults_use_4096_dimensions() -> None:
    settings = Settings()
    assert settings.embedding_model == "qwen3-embedding:8b"
    assert settings.embedding_dimensions == 4096
    assert settings.qdrant_collection == "rwkvrag-knowledge-current"


def test_query_instruction_uses_qwen3_retrieval_format() -> None:
    model = InstructedOpenAIEmbedding.model_construct(query_instruction="检索相关资料。")
    assert model._query_text("中国的首都是哪里？") == (
        "Instruct: 检索相关资料。\nQuery: 中国的首都是哪里？"
    )
