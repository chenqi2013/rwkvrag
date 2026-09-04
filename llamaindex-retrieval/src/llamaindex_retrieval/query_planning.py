from dataclasses import dataclass
from typing import Literal

from .lexical_index import normalize_query_text
from .qa_analysis import QuestionAnalysis, analyze_question, clean_question_shell

MergeStrategy = Literal["rank_fusion", "document_interleave"]
ContextPolicy = Literal["none", "lead", "lead_append", "section", "structure"]
AnswerShape = Literal["single_fact", "list", "summary", "narrative"]
SetSemantics = Literal["latest", "all", "partial", "specific"]


@dataclass(frozen=True)
class TaskField:
    field_id: str
    question: str
    relations: tuple[str, ...]


@dataclass(frozen=True)
class QueryPlan:
    original_question: str
    normalized_question: str
    analysis: QuestionAnalysis
    queries: tuple[str, ...]
    subject: str
    relations: tuple[str, ...]
    merge_strategy: MergeStrategy
    context_policy: ContextPolicy
    fields: tuple[TaskField, ...] = ()
    answer_shape: AnswerShape = "single_fact"
    set_semantics: SetSemantics = "specific"

    @property
    def query_rewritten(self) -> bool:
        return self.queries != (self.original_question,)


def build_query_plan(question: str) -> QueryPlan:
    normalized = normalize_query_text(clean_question_shell(question))
    analysis = analyze_question(normalized)
    return QueryPlan(
        original_question=question,
        normalized_question=normalized,
        analysis=analysis,
        queries=(normalized,),
        subject="",
        relations=(),
        merge_strategy="rank_fusion",
        context_policy="none",
        fields=(TaskField("f1", normalized, ()),),
    )
