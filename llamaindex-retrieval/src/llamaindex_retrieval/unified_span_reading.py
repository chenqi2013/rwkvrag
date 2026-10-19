"""ID-selected original reading and derived synthesis; binding is not support proof."""
from __future__ import annotations

from dataclasses import dataclass
import json

from llama_index.core import Document
from pydantic import Field

from .unified_contracts import Contract, Text, digest, dumps, unique
from .verbatim_chunking import packed_verbatim_nodes

PROTOCOL = "rwkv-local-span-ids-v1"


@dataclass(frozen=True)
class SpanCatalog:
    """Immutable per-call catalog, with evidence-local IDs (not global IDs).

    The caller must bind against the same catalog sent in this call, never a
    global/latest catalog. Full source identity and catalog hash are receipts.
    Adjacent structure is packed without trimming, not relevance selection.
    Nonblank materials cover all body characters, including attached whitespace.
    Whitespace-only materials retain their text but expose no selectable IDs.
    """

    material_json: str
    spans_json: str
    window: int
    overlap: int

    @classmethod
    def build(cls, material, *, window=640, overlap=80):
        text, span = material["text"], material["span"]
        start, end = span["start"], span["end"]
        if (type(start) is not int or type(end) is not int or start < 0
                or end - start != len(text) or span["unit"] != "unicode_code_points"
                or span["sha256"] != digest(text)):
            raise ValueError("invalid_original_span")
        for key in ("evidence_id", "source_id", "snapshot_id"):
            if not isinstance(material[key], str) or not material[key].strip():
                raise ValueError("invalid_source_identity")
        document = Document(text=text, id_=material["source_id"],
                            metadata={"kind": material.get("kind")})
        nodes = packed_verbatim_nodes(document, chunk_characters=window,
                                      overlap_characters=overlap, absorb_whitespace=True)
        spans = []
        for node in nodes:
            if not node.text.strip():
                continue
            index = len(spans)
            local = node.metadata["source_span"]
            contexts = []
            for context in node.metadata["context_spans"]:
                contexts.append({**context, "start": start + context["start"],
                                 "end": start + context["end"]})
            spans.append({"id": f"{material['evidence_id']}.S{index + 1}", "text": node.text,
                          "span": {**local, "start": start + local["start"],
                                   "end": start + local["end"]},
                          "kind": node.metadata["content_type"],
                          "atomic": node.metadata["atomic"], "context_spans": contexts})
        return cls(dumps(material), dumps(spans), window, overlap)

    @property
    def sha256(self):
        return digest(dumps([PROTOCOL, self.material_json, self.spans_json, self.window, self.overlap]))

    def check(self):
        if self != self.build(json.loads(self.material_json), window=self.window, overlap=self.overlap):
            raise ValueError("invalid_span_catalog")

    def model_material(self):
        """Exact body slices replace the duplicate body string; metadata stays."""
        self.check()
        material = json.loads(self.material_json)
        spans = json.loads(self.spans_json)
        if spans:
            del material["text"]
        return {**material, "body_spans": spans,
                "catalog_sha256": self.sha256,
                "representation": "verbatim_body_spans_not_model_summaries"}


class SpanFinding(Contract):
    span_id: Text
    statement: Text


class SpanReading(Contract):
    findings: list[SpanFinding] = Field(max_length=16)
    missing: list[Text] = Field(max_length=32)

    def bind(self, catalog: SpanCatalog):
        """Resolve only model-selected IDs; never add choices or rewrite raw."""
        catalog.check()
        material = json.loads(catalog.material_json)
        spans = {span["id"]: span for span in json.loads(catalog.spans_json)}
        bindings = []
        for index, finding in enumerate(self.findings):
            if finding.span_id not in spans:
                raise ValueError(f"unknown_span_id:{index}")
            selected = spans[finding.span_id]
            bindings.append({"finding_index": index, "selected_span_id": finding.span_id,
                "catalog_sha256": catalog.sha256, "evidence_id": material["evidence_id"],
                "source_id": material["source_id"], "snapshot_id": material["snapshot_id"],
                "span": selected["span"], "context_spans": selected["context_spans"],
                "selected_original_text": selected["text"],
                "selection_source": "model_span_id", "text_source": "original_material",
                "model_generated_quote": False, "semantic_support_verified": False})
        return bindings


class SpanAnswer(Contract):
    """A new model-authored synthesis, never a program projection of leaf findings."""
    answer: Text
    span_ids: list[Text] = Field(max_length=128)
    missing: list[Text] = Field(max_length=32)

    def check(self, allowed):
        unique(self.span_ids)
        if not set(self.span_ids) <= set(allowed):
            raise ValueError("span_outside_children")


def _reading_context(payload):
    # Routing/Task interpretations stay in the audit and control calls. Readers
    # use the original request; child raw and selected provenance pass unchanged.
    projected = {key: payload[key] for key in (
        "history", "latest_user_message", "children", "inherited_span_bindings"
    ) if key in payload}
    if "execution_scope" in payload:
        scope = payload["execution_scope"]
        projected["execution_scope"] = {key: scope[key] for key in (
            "stop_reason", "read_units", "unread_units", "failed_actions", "failed_nodes", "control_failures",
            "reading_attempts"
        ) if key in scope}
    return projected


def synthesis_prompt(payload, *, terminal, contract_schema=None):
    purpose = "final" if terminal else "merge"
    instruction = ("忠实汇总本组阅读结果，简洁回答当前用户的问题。" if terminal else
                   "忠实汇总本组阅读结果，保留本题所需的事实、具体名称和数值。")
    if terminal and payload.get("execution_scope", {}).get("stop_reason") == "retrieval_not_required":
        instruction = "根据原始对话完成当前用户请求。本次未检索外部资料，不要声称查过来源；缺少必要信息时如实说明。"
    return (
        f"协议：rwkv-unified-v3；用途：{purpose}\n{instruction}\n"
        "answer写正文，span_ids列依据的片段ID；missing列本题尚未回答的要点，已答全则为空数组。"
        "\n输出结构：" + dumps(SpanAnswer.model_json_schema() if contract_schema is None else contract_schema)
        + "\n输入数据：\n" + dumps(_reading_context(payload))
    )


def span_prompt(payload, catalog: SpanCatalog, *, rereading=False):
    if dumps(payload["original_material"]) != catalog.material_json:
        raise ValueError("catalog_material_mismatch")
    projected = {**_reading_context(payload), "original_material": catalog.model_material()}
    instruction = "重新阅读本次原文，提取本题所需事实。" if rereading else "阅读本次原文，提取回答当前用户问题所需的事实。"
    return (
        f"协议：{PROTOCOL}；用途：local_reading\n{instruction}\n"
        "输出JSON：findings中的每项用span_id选原文片段，statement忠实表述该片段内容；"
        "missing列原文尚未回答的本题要点。没有对应内容的数组为空。"
        "\n输入数据：\n" + dumps(projected)
    )
