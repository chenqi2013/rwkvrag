from llamaindex_retrieval.sparse import chinese_tokens, encode_sparse_texts


def test_chinese_tokens_include_characters_and_bigrams() -> None:
    tokens = chinese_tokens("中国人口")
    assert "中" in tokens
    assert "中国" in tokens
    assert "人口" in tokens


def test_sparse_encoding_is_deterministic() -> None:
    first = encode_sparse_texts(["中国人口"])
    second = encode_sparse_texts(["中国人口"])
    assert first == second
    assert first[0][0]
    assert len(first[0][0]) == len(first[1][0])
