import pytest
from semantic_chunk_router import SemanticChunker

def test_empty_string():
    chunker = SemanticChunker()
    assert chunker.chunk_text("") == []

def test_single_sentence():
    chunker = SemanticChunker()
    text = "Only one sentence."
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text
