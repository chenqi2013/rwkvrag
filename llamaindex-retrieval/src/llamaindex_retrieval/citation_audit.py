"""Literal citation/quote checks only; never infer entailment or edit answers."""
import re

TAG = re.compile(r"\[资料[^\]\r\n]*(?:\]|$)", re.MULTILINE)
VALID_TAG = re.compile(r"\[资料\s*([1-9][0-9]*)\]")
QUOTE_LINE = re.compile(r"^[ \t]*原文[：:](.*)$", re.MULTILINE)


def audit_citations(text, sources, *, check_quotes=False):
    tags = list(dict.fromkeys(TAG.findall(text)))
    valid = [(tag, VALID_TAG.fullmatch(tag)) for tag in tags]
    labels = sorted({int(match[1]) for _, match in valid if match})
    result = {
        "label_ids": labels,
        "scope": "literal_labels_in_answer_span",
        "unknown_label_ids": [i for i in labels if i > len(sources)],
        "invalid_labels": [tag for tag, match in valid if match is None],
        "semantic_support_verified": False,
    }
    if check_quotes:
        quotes = []
        for line in QUOTE_LINE.finditer(text):
            body = line[1]
            line_tags = TAG.findall(body)
            matches = [VALID_TAG.fullmatch(tag) for tag in line_tags]
            ids = sorted({int(match[1]) for match in matches if match})
            quote = TAG.sub("", body).strip()
            for left, right in (("「", "」"), ("“", "”"), ('"', '"')):
                if quote.startswith(left) and quote.endswith(right) and len(quote) >= 2:
                    quote = quote[1:-1]
                    break
            found = []
            for label in ids:
                if 1 <= label <= len(sources):
                    source = sources[label - 1]
                    texts = [source["snippet"]]
                    spans = source.get("metadata", {}).get("context_spans", [])
                    if not isinstance(spans, list):
                        spans = []
                    texts += [span["text"] for span in spans
                              if isinstance(span, dict) and isinstance(span.get("text"), str)]
                    if quote and any(quote in original for original in texts):
                        found.append(label)
            quotes.append({"body_span": [line.start(), line.end()], "label_ids": ids,
                           "verbatim_label_ids": found,
                           "verbatim": bool(quote and ids and all(matches) and found == ids)})
        result["quote_audit"] = {
            "scope": "single_line_original_text_fields_in_evidence_checked_protocol",
            "offset_unit": "unicode_characters_in_answer_body",
            "checked": len(quotes), "failed": sum(not q["verbatim"] for q in quotes),
            "quotes": quotes, "semantic_support_verified": False,
        }
    return result
