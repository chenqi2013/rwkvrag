"""Rank exact windows from the full provider snapshot using lexical BM25.

Ranking chooses material, not answers. Selected text and offsets are unchanged;
the complete snapshot stays available independently of the Reader budget.
"""
from collections import Counter
import math
import re

from llama_index.core import Document

from .verbatim_chunking import verbatim_nodes


def terms(text):
    # Segment Latin identifiers and CJK bigrams without normalizing source text.
    tokens = []
    for token in re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]+", text.lower()):
        if "\u3400" <= token[0] <= "\u9fff" and len(token) > 1:
            tokens.extend(token[i:i + 2] for i in range(len(token) - 1))
        else:
            tokens.append(token)
    return tokens


def ranked_windows(text, query, document_id, *, window=2400, limit=4):
    nodes = [node for node in verbatim_nodes(Document(text=text, id_=document_id),
             chunk_characters=window, overlap_characters=min(180, window // 4))
             if node.text.strip()]
    if not nodes:
        return [], {"policy": "ranked_windows", "total_windows": 0, "selected_windows": 0}
    counters = [Counter(terms(node.text + "\n" + "".join(node.metadata["heading_path"])))
                for node in nodes]
    lengths = [sum(c.values()) for c in counters]
    average = max(1, sum(lengths) / len(lengths))
    query_terms = set(terms(query))
    frequency = {term: sum(term in c for c in counters) for term in query_terms}

    def score(index):
        count = counters[index]
        return sum(math.log(1 + (len(nodes) - frequency[t] + .5) / (frequency[t] + .5))
                   * count[t] * 2.2 / (count[t] + 1.2 * (.25 + .75 * lengths[index] / average))
                   for t in query_terms if count[t])

    order = sorted(range(len(nodes)), key=lambda i: (-score(i), i))[:limit]
    selected = [(nodes[i], score(i)) for i in order]
    return selected, {"policy": "ranked_windows", "total_windows": len(nodes),
                      "selected_windows": len(selected), "unselected_windows": len(nodes) - len(selected),
                      "full_snapshot_characters": len(text),
                      "selected_characters": sum(len(n.text) for n, _ in selected)}
