"""Generic evidence transport helpers.

Semantic decisions belong to the configured language model. This module
keeps only content-neutral cleanup and compatibility shims for integrations
that still import former extractive helper names. The shims deliberately
return ``None`` instead of inferring an answer from question words.
"""

import re


_HTML_TAG = re.compile(r"<[^>]{1,80}>")
_SPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_evidence_text(text: str) -> str:
    """Remove transport markup without changing factual content."""

    cleaned = _HTML_TAG.sub("", text)
    cleaned = cleaned.replace("\u200b", "").replace("\ufeff", "")
    cleaned = "\n".join(_SPACE.sub(" ", line).strip() for line in cleaned.splitlines())
    return _BLANK_LINES.sub("\n\n", cleaned).strip()


def structured_list_answer(question: str, sources: list[object]) -> None:
    return None


def direct_evidence_answer(question: str, sources: list[object]) -> None:
    return None


def list_evidence_answer(question: str, sources: list[object]) -> None:
    return None


def time_evidence_answer(question: str, sources: list[object]) -> None:
    return None


def coordinated_time_evidence_answer(
    sources: list[object], subjects: tuple[str, ...] | list[str]
) -> None:
    return None


def agent_evidence_answer(
    question: str, sources: list[object], relations: tuple[str, ...] = ()
) -> None:
    return None


def cause_evidence_answer(sources: list[object]) -> None:
    return None


def ordinal_evidence_answer(question: str, sources: list[object]) -> None:
    return None


def location_evidence_answer(question: str, sources: list[object]) -> None:
    return None


def quantitative_evidence_answer(question: str, sources: list[object]) -> None:
    return None


def birthplace_evidence_answer(question: str, sources: list[object]) -> None:
    return None


def definition_evidence_answer(question: str, sources: list[object]) -> None:
    return None
