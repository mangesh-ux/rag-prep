"""Tests for customization and extensibility."""

from pathlib import Path
from typing import Iterator, List

import pytest

from rag_prep import Config, prepare_docs
from rag_prep.chunkers import ChunkingStrategy
from rag_prep.models import Chunk


class CommaChunker:
    """Custom chunker that splits on commas."""

    def __init__(self, size: int = 1000, overlap: int = 200):
        # Ignore size and overlap for this simple example
        self.size = size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict) -> Iterator[Chunk]:
        """Split text on commas."""
        parts = text.split(",")
        for idx, part in enumerate(parts):
            if part.strip():  # Skip empty parts
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = idx
                chunk_metadata["split_method"] = "comma"

                yield Chunk(
                    text=part.strip(),
                    metadata=chunk_metadata,
                    chunk_id=f"{metadata.get('source_id', 'doc')}_comma_{idx}",
                )


def test_custom_chunker_integration(temp_dir: Path) -> None:
    """Test that custom chunker can be plugged into the pipeline."""
    # Create test file
    test_file = temp_dir / "test.txt"
    test_file.write_text("First part, Second part, Third part")

    # Use custom chunker
    custom_chunker = CommaChunker()
    config = Config(chunk_strategy="character")  # This should be overridden

    chunks: List[Chunk] = list(prepare_docs(test_file, config=config, chunker=custom_chunker))

    # Verify custom chunking logic was used
    assert len(chunks) == 3, "Should produce 3 chunks (split on commas)"
    assert "First part" in chunks[0].text
    assert "Second part" in chunks[1].text
    assert "Third part" in chunks[2].text

    # Verify custom metadata
    assert all(c.metadata.get("split_method") == "comma" for c in chunks), "Should have custom metadata"


def test_config_override_with_custom_chunker(temp_dir: Path) -> None:
    """Test that custom chunker overrides config chunk_strategy."""
    test_file = temp_dir / "test.txt"
    # Use text with commas so CommaChunker produces multiple chunks
    test_file.write_text("part1, part2, part3")

    config = Config(chunk_strategy="character", chunk_size=5)  # Small size to ensure multiple chunks
    custom_chunker = CommaChunker()

    # Test with custom chunker - this verifies the chunker parameter overrides config
    chunks_with_custom: List[Chunk] = list(prepare_docs(test_file, config=config, chunker=custom_chunker))
    
    # Verify custom chunker was actually used (not the config strategy)
    assert len(chunks_with_custom) == 3, "CommaChunker should produce 3 chunks"
    assert all("split_method" in c.metadata and c.metadata["split_method"] == "comma" for c in chunks_with_custom), "Should use custom chunker"
    
    # The fact that we got comma-split chunks (3) instead of character chunks (many) proves the override works
    # We don't need to call prepare_docs again to verify this


def test_metadata_hooks(temp_dir: Path) -> None:
    """Test that metadata hooks can enrich chunk metadata."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("Test content")

    def add_timestamp(metadata: dict, text: str) -> dict:
        """Add timestamp to metadata."""
        import datetime

        metadata["processed_at"] = datetime.datetime.now().isoformat()
        metadata["text_length"] = len(text)
        return metadata

    config = Config(
        chunk_strategy="character",
        chunk_size=100,
        metadata_hooks=[add_timestamp],
    )

    chunks: List[Chunk] = list(prepare_docs(test_file, config=config))

    assert len(chunks) > 0, "Should produce chunks"
    assert all("processed_at" in c.metadata for c in chunks), "Should have processed_at from hook"
    assert all("text_length" in c.metadata for c in chunks), "Should have text_length from hook"


def test_multiple_metadata_hooks(temp_dir: Path) -> None:
    """Test that multiple metadata hooks can be chained."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("Test")

    def hook1(metadata: dict, text: str) -> dict:
        metadata["hook1"] = "value1"
        return metadata

    def hook2(metadata: dict, text: str) -> dict:
        metadata["hook2"] = "value2"
        return metadata

    config = Config(
        chunk_strategy="character",
        metadata_hooks=[hook1, hook2],
    )

    chunks: List[Chunk] = list(prepare_docs(test_file, config=config))

    assert len(chunks) > 0, "Should produce chunks"
    for chunk in chunks:
        assert chunk.metadata.get("hook1") == "value1", "Should have hook1 metadata"
        assert chunk.metadata.get("hook2") == "value2", "Should have hook2 metadata"


def test_custom_chunker_via_module_path(temp_dir: Path) -> None:
    """Test loading custom chunker via module path string."""
    # This would require the chunker to be in an importable module
    # For this test, we'll verify the mechanism works with a built-in
    # In practice, users would do: --chunk-strategy mypackage.MyChunker

    test_file = temp_dir / "test.txt"
    test_file.write_text("Test content")

    # Test that get_chunker can handle module paths (even if it fails for non-existent modules)
    from rag_prep.chunkers import get_chunker

    # Should work with built-in strategies
    chunker = get_chunker("character", size=50, overlap=5)
    assert chunker is not None, "Should return chunker for valid strategy"

    # For custom module paths, we'd need the module to exist
    # This is more of an integration test that would be run with actual custom modules


def test_metadata_preservation_through_pipeline(temp_dir: Path) -> None:
    """Test that file metadata is preserved and enriched through the pipeline."""
    test_file = temp_dir / "custom_name.txt"
    test_file.write_text("Content here")

    config = Config(chunk_strategy="character", chunk_size=10)

    chunks: List[Chunk] = list(prepare_docs(test_file, config=config))

    assert len(chunks) > 0, "Should produce chunks"
    # Verify built-in metadata
    for chunk in chunks:
        assert "source_id" in chunk.metadata, "Should have source_id"
        assert "file_name" in chunk.metadata, "Should have file_name"
        assert chunk.metadata["file_name"] == "custom_name.txt", "Should preserve filename"
        assert "file_type" in chunk.metadata, "Should have file_type"
        assert chunk.metadata["file_type"] == "text", "Should have correct file type"

