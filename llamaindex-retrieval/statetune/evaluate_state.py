"""Shared raw generation utility; no training or experiment entrypoint."""

def generate(prompt, vocab, decide, *, cap=32):
    """Recompute each full prefix from the same initial state; never repair output."""
    if not prompt or len(prompt) + cap - 1 > 4096:
        raise ValueError("complete generation would exceed context")
    generated = []
    for _ in range(cap):
        token = decide(prompt + generated)
        if type(token) is not int or (token != 0 and token not in vocab):
            raise ValueError("invalid generated token")
        generated.append(token)
        if token == 0:
            break
    raw = b"".join(vocab[t] for t in generated if t != 0)
    try:
        text = raw.decode("utf-8")
        utf8 = True
    except UnicodeDecodeError:
        text, utf8 = raw.decode("utf-8", errors="replace"), False
    return {"token_ids": generated, "raw_hex": raw.hex(), "raw_text": text,
            "utf8_valid": utf8, "eos_observed": generated[-1] == 0,
            "cap_reached": generated[-1] != 0, "gold_opened": False}
