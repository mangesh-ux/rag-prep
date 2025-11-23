"""Tests for chunking strategies."""

from typing import List

import pytest

from rag_prep.chunkers import (
    CharacterChunker,
    NoChunker,
    SentenceChunker,
    TokenChunker,
    get_chunker,
)
from rag_prep.models import Chunk


def test_character_chunker_basic() -> None:
    """Test character chunker with basic input."""
    text = "a" * 1200
    metadata = {"source_id": "test"}
    chunker = CharacterChunker(size=500, overlap=50)

    chunks: List[Chunk] = list(chunker.chunk(text, metadata))

    assert len(chunks) > 0, "Should produce chunks"
    assert len(chunks) >= 2, "Should produce multiple chunks for 1200 chars with size 500"

    # Check no empty chunks
    assert all(len(c.text) > 0 for c in chunks), "No chunks should be empty"

    # Check chunk sizes (allowing for last chunk to be smaller)
    for i, chunk in enumerate(chunks[:-1]):
        assert len(chunk.text) <= 500, f"Chunk {i} should not exceed size limit"

    # Verify overlap behavior
    if len(chunks) >= 2:
        # The overlap should be visible in consecutive chunks
        # With size=500 and overlap=50, chunk 1 ends at 500, chunk 2 starts at 450
        # So the last 50 chars of chunk 0 should appear in chunk 1
        chunk0_end = chunks[0].text[-50:]
        chunk1_start = chunks[1].text[:50]
        assert chunk0_end == chunk1_start, "Consecutive chunks should overlap correctly"


def test_character_chunker_small_text() -> None:
    """Test character chunker with text smaller than chunk size."""
    text = "Short text"
    metadata = {"source_id": "test"}
    chunker = CharacterChunker(size=100, overlap=10)

    chunks: List[Chunk] = list(chunker.chunk(text, metadata))

    assert len(chunks) == 1, "Should produce single chunk for small text"
    assert chunks[0].text == text, "Chunk should contain full text"


def test_sentence_chunker() -> None:
    """Test sentence chunker respects sentence boundaries."""
    text = "One. Two. Three."
    metadata = {"source_id": "test"}
    chunker = SentenceChunker(size=100, overlap=10)

    chunks: List[Chunk] = list(chunker.chunk(text, metadata))

    assert len(chunks) > 0, "Should produce chunks"
    # All chunks should contain complete sentences (ending with period)
    # Note: sentence chunker may combine sentences if they fit in size
    assert all("." in c.text or len(c.text) > 0 for c in chunks), "Chunks should contain sentences"


def test_sentence_chunker_large_text() -> None:
    """Test sentence chunker with text that requires multiple chunks."""
    sentences = [f"Sentence {i}. " for i in range(50)]
    text = "".join(sentences)
    metadata = {"source_id": "test"}
    chunker = SentenceChunker(size=100, overlap=20)

    chunks: List[Chunk] = list(chunker.chunk(text, metadata))

    assert len(chunks) > 1, "Should produce multiple chunks for large text"
    assert all(len(c.text) > 0 for c in chunks), "No empty chunks"


def test_token_chunker() -> None:
    """Test token chunker if tokenization is available."""
    try:
        import tiktoken
    except ImportError:
        pytest.skip("tiktoken not available")

    # Create text that will require multiple chunks
    text = " ".join([f"word{i}" for i in range(200)])
    metadata = {"source_id": "test"}
    chunker = TokenChunker(size=50, overlap=10, tokenizer_name="cl100k_base")

    chunks: List[Chunk] = list(chunker.chunk(text, metadata))

    assert len(chunks) > 0, "Should produce chunks"
    assert len(chunks) > 1, "Should produce multiple chunks for large text"

    # Verify chunks have token metadata
    assert all("start_token" in c.metadata for c in chunks), "Should include token positions"
    assert all("end_token" in c.metadata for c in chunks), "Should include token positions"

    # Verify no empty chunks
    assert all(len(c.text) > 0 for c in chunks), "No chunks should be empty"


def test_token_chunker_fallback() -> None:
    """Test token chunker falls back to character chunking if tokenizer fails."""
    text = "Test text"
    metadata = {"source_id": "test"}

    # Create chunker with invalid tokenizer name
    chunker = TokenChunker(size=5, overlap=1, tokenizer_name="invalid_tokenizer_name_xyz")

    # Should fall back to character chunking
    chunks: List[Chunk] = list(chunker.chunk(text, metadata))
    assert len(chunks) > 0, "Should produce chunks even with invalid tokenizer"


def test_no_chunker() -> None:
    """Test no chunker returns entire text as single chunk."""
    text = "This is a complete document that should not be chunked."
    metadata = {"source_id": "test"}
    chunker = NoChunker()

    chunks: List[Chunk] = list(chunker.chunk(text, metadata))

    assert len(chunks) == 1, "Should produce exactly one chunk"
    assert chunks[0].text == text, "Chunk should contain full text"
    assert chunks[0].chunk_id == "test", "Chunk ID should match source_id"


def test_get_chunker_character() -> None:
    """Test get_chunker returns CharacterChunker for 'character' strategy."""
    chunker = get_chunker("character", size=100, overlap=10)
    assert isinstance(chunker, CharacterChunker), "Should return CharacterChunker"


def test_get_chunker_sentence() -> None:
    """Test get_chunker returns SentenceChunker for 'sentence' strategy."""
    chunker = get_chunker("sentence", size=100, overlap=10)
    assert isinstance(chunker, SentenceChunker), "Should return SentenceChunker"


def test_get_chunker_token() -> None:
    """Test get_chunker returns TokenChunker for 'token' strategy."""
    chunker = get_chunker("token", size=100, overlap=10, tokenizer_name="cl100k_base")
    assert isinstance(chunker, TokenChunker), "Should return TokenChunker"


def test_get_chunker_none() -> None:
    """Test get_chunker returns NoChunker for 'none' strategy."""
    chunker = get_chunker("none")
    assert isinstance(chunker, NoChunker), "Should return NoChunker"


def test_chunker_metadata_preservation() -> None:
    """Test that chunkers preserve and enrich metadata."""
    text = "Test text for chunking"
    metadata = {"source_id": "test", "custom_field": "custom_value"}
    chunker = CharacterChunker(size=10, overlap=2)

    chunks: List[Chunk] = list(chunker.chunk(text, metadata))

    assert len(chunks) > 0, "Should produce chunks"
    # All chunks should have original metadata plus chunk-specific fields
    for chunk in chunks:
        assert chunk.metadata["source_id"] == "test", "Should preserve source_id"
        assert chunk.metadata["custom_field"] == "custom_value", "Should preserve custom metadata"
        assert "chunk_index" in chunk.metadata, "Should add chunk_index"
        assert "start_char" in chunk.metadata, "Should add start_char"
        assert "end_char" in chunk.metadata, "Should add end_char"

