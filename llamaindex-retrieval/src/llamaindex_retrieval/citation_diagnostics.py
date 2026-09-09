"""Non-destructive diagnostics for positional citations in model text."""

import re
from dataclasses import dataclass


_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})([^\r\n]*)[\r\n]*$")
_BACKTICKS = re.compile(r"`+")
_CITATION = re.compile(r"^\[\s*资料\s*([1-9][0-9]*)\s*\]$")
_CITATION_PREFIX = re.compile(r"^\[\s*资料")


def _escaped(text: str, position: int) -> bool:
    backslashes = 0
    position -= 1
    while position >= 0 and text[position] == "\\":
        backslashes += 1
        position -= 1
    return backslashes % 2 == 1


@dataclass(frozen=True)
class CitationDiagnostics:
    references: tuple[int, ...]
    unknown: tuple[int, ...]
    in_code: tuple[int, ...]
    malformed: tuple[str, ...]
    unclosed: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "references": list(self.references),
            "unknown": list(self.unknown),
            "in_code": list(self.in_code),
            "malformed": list(self.malformed),
            "unclosed": self.unclosed,
            "output_modified": False,
        }


def _fenced_code_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    opening: tuple[int, str, int] | None = None
    position = 0
    for line in text.splitlines(keepends=True):
        match = _FENCE.fullmatch(line)
        if opening is not None:
            start, character, length = opening
            if (
                match
                and match.group(1)[0] == character
                and len(match.group(1)) >= length
                and not match.group(2).strip()
            ):
                spans.append((start, position + len(line)))
                opening = None
        elif match and (match.group(1)[0] != "`" or "`" not in match.group(2)):
            opening = (position, match.group(1)[0], len(match.group(1)))
        position += len(line)
    if opening is not None:
        spans.append((opening[0], len(text)))
    return spans


def _inline_code_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    ticks = list(_BACKTICKS.finditer(text, start, end))
    following: dict[int, int] = {}
    spans: list[tuple[int, int]] = []
    for index in range(len(ticks) - 1, -1, -1):
        length = ticks[index].end() - ticks[index].start()
        closing = following.get(length)
        if closing is not None and not _escaped(text, ticks[index].start()):
            spans.append((ticks[index].start(), ticks[closing].end()))
            following.pop(length, None)
        else:
            following[length] = index
    return spans


def _code_mask(text: str) -> bytearray:
    mask = bytearray(len(text))
    fences = _fenced_code_spans(text)
    spans = list(fences)
    start = 0
    for fence_start, fence_end in [*fences, (len(text), len(text))]:
        paragraph_start = start
        position = start
        for line in text[start:fence_start].splitlines(keepends=True):
            if not line.strip():
                spans.extend(_inline_code_spans(text, paragraph_start, position))
                paragraph_start = position + len(line)
            position += len(line)
        spans.extend(_inline_code_spans(text, paragraph_start, fence_start))
        start = fence_end
    for start, end in spans:
        mask[start:end] = b"\1" * (end - start)
    return mask


def diagnose_citations(text: str, source_count: int) -> CitationDiagnostics:
    """Inspect citations while leaving ``text`` byte-for-byte untouched."""

    mask = _code_mask(text)
    references: list[int] = []
    unknown: list[int] = []
    in_code: list[int] = []
    malformed: list[str] = []
    unclosed = False

    # Scan one bracketed label at a time.  A regex spanning the complete
    # string can accidentally consume a later ``[`` when an earlier label is
    # unclosed, making the unclosed condition invisible to diagnostics.
    position = 0
    while position < len(text):
        if text[position] != "[" or _escaped(text, position):
            position += 1
            continue
        remainder = text[position:]
        if not _CITATION_PREFIX.match(remainder):
            position += 1
            continue
        end = position + 1
        while end < len(text) and text[end] not in "]\r\n[":
            end += 1
        if end < len(text) and text[end] == "]":
            token = text[position:end + 1]
            match = _CITATION.fullmatch(token)
            if match is not None:
                number = int(match.group(1))
                if mask[position]:
                    in_code.append(number)
                else:
                    if number not in references:
                        references.append(number)
                    if number > source_count and number not in unknown:
                        unknown.append(number)
            elif not mask[position]:
                malformed.append(token)
            position = end + 1
            continue
        # A line break or another opening bracket terminates the malformed
        # candidate.  The latter is left for the next scan iteration.
        elif not mask[position]:
            unclosed = True
        position += 1
    return CitationDiagnostics(
        tuple(references), tuple(unknown), tuple(in_code), tuple(malformed), unclosed
    )
