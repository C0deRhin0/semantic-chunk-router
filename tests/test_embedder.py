from semantic_chunk_router import MockEmbedder


def test_mock_embedder_preserves_singular_words_ending_in_s():
    weights = MockEmbedder()._term_weights("analysis status campus")

    assert "analysis" in weights
    assert "status" in weights
    assert "campus" in weights
    assert "analysi" not in weights
    assert "statu" not in weights
    assert "campu" not in weights


def test_mock_embedder_normalizes_known_plural_aliases():
    weights = MockEmbedder()._term_weights("deployments components indexes queries")

    assert "deployment" in weights
    assert "component" in weights
    assert "index" in weights
    assert "query" in weights
