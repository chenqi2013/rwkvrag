"""Application entry points for the model-authored RWKV pipeline only."""

from .config import Settings
from .lexical_index import LexicalIndex
from .rwkv_pipeline import RWKVPipeline, fuse_chunks
from .schemas import AskResponse, ConversationMessage, SearchRequest, SearchResponse, SourceItem


class SearchService:
    def __init__(self, settings: Settings, index: LexicalIndex | None,
                 *, native_recorder=None) -> None:
        # Also reject settings mutated after validation; never dispatch to a
        # rule-based answerer or retain an optional legacy fallback.
        if settings.rag_pipeline != "rwkv" or settings.generation_output_mode != "immutable":
            raise ValueError("Only the rwkv pipeline with immutable output is supported")
        self.settings = settings
        self.index = index
        if settings.native_unified is not None:
            from .unified_retrieval import UnifiedRetrievalPipeline
            self.native_pipeline = UnifiedRetrievalPipeline(settings, index, recorder=native_recorder)
        else:
            self.native_pipeline = RWKVPipeline(settings, index, recorder=native_recorder)

    async def aclose(self) -> None:
        await self.native_pipeline.aclose()

    async def search(self, request: SearchRequest) -> SearchResponse:
        if self.settings.native_unified is not None:
            return await self.native_pipeline.search(request)
        provider_trace = {}
        if request.retrieval_mode != "knowledge_base":
            groups, provider_trace = await self.native_pipeline.retrieve_groups(
                request, [request.question], request.candidate_k or self.settings.candidate_k)
            sources = fuse_chunks(groups, order=self.settings.native_candidate_order)
        else:
            sources = await self.native_pipeline.search(request)
        top_k = min(request.top_k or self.settings.default_top_k, self.settings.max_top_k)
        return SearchResponse(results=sources[:top_k], retrieval={
            "mode": "native-bm25-chunks" if request.retrieval_mode == "knowledge_base" else request.retrieval_mode,
            **provider_trace, "index": self.settings.opensearch_index,
            "candidate_count": len(sources), "returned": len(sources[:top_k]),
            "per_document_limit": None,
        })

    async def ask(self, request: SearchRequest) -> AskResponse:
        return await self.native_pipeline.ask(request)

    async def ask_materials(self, question: str, materials: list[SourceItem],
                            *, history: list[ConversationMessage] | None = None) -> AskResponse:
        return await self.native_pipeline.ask_materials(question, materials, history=history)
