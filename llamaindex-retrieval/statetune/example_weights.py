"""Per-example weighting utility."""

def example_weights(rows, vocab, negative_weight):
    """Global mean-one weights, preserving the actual last-group denominator."""
    raw = {}
    for row in rows:
        target = b''.join(vocab.by_id[t] for t in row['input_ids'][row['prompt_tokens']:-1])
        raw[row['id']] = negative_weight if target == b'NONE' else 1.0
    if len(raw) != len(rows) or not rows:
        raise ValueError('empty/duplicate training identity')
    normalizer = len(rows)/sum(raw.values())
    return {name: value*normalizer for name, value in raw.items()}
