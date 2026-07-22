import hashlib

from llama_index.core.node_parser import SentenceSplitter

from .config import Settings


def stable_node_id(index: int, document) -> str:
    value = f"{document.id_}:{index}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:32]


def create_splitter(settings: Settings) -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        id_func=stable_node_id,
    )
