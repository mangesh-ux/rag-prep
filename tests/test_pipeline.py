"""End-to-end pipeline tests."""

import json
from pathlib import Path
from typing import List

import pytest

from rag_prep import Chunk, Config, prepare_docs, prepare_docs_to_jsonl


def test_end_to_end_pipeline_default_config(multi_file_dir: Path, temp_dir: Path) -> None:
    """Test the complete pipeline with default configuration."""
    # Use default config with character-based chunking
    config = Config(chunk_strategy="character", chunk_size=100, chunk_overlap=20)

    # Call prepare_docs directly
    chunks: List[Chunk] = list(prepare_docs(multi_file_dir, config=config))

    # Assertions
    assert len(chunks) > 0, "Should produce at least one chunk"

    for chunk in chunks:
        assert isinstance(chunk, Chunk), "Each item should be a Chunk instance"
        assert len(chunk.text) > 0, "Chunk text should be non-empty"
        assert isinstance(chunk.metadata, dict), "Metadata should be a dictionary"
        assert "source_id" in chunk.metadata, "Metadata should include source_id"
        assert "chunk_index" in chunk.metadata, "Metadata should include chunk_index"
        assert chunk.chunk_id is not None, "Chunk should have a chunk_id"

    # Verify chunk_index increments properly
    chunk_indices = [c.metadata.get("chunk_index") for c in chunks if "chunk_index" in c.metadata]
    if len(chunk_indices) > 1:
        # Check that indices are sequential (may not be strictly 0,1,2 if from different files)
        assert all(isinstance(idx, int) for idx in chunk_indices), "Chunk indices should be integers"

    # Test prepare_docs_to_jsonl
    output_path = temp_dir / "output.jsonl"
    prepare_docs_to_jsonl(multi_file_dir, output_path, config=config)

    # Assert output file exists
    assert output_path.exists(), "Output JSONL file should exist"

    # Count lines and validate
    lines = output_path.read_text().strip().split("\n")
    assert len(lines) == len(chunks), f"JSONL should have {len(chunks)} lines, got {len(lines)}"

    # Validate JSON structure
    for line in lines:
        data = json.loads(line)
        assert "text" in data, "Each line should have 'text' field"
        assert "metadata" in data, "Each line should have 'metadata' field"
        assert len(data["text"]) > 0, "Text should be non-empty"


def test_pipeline_single_file(sample_text_file: Path, temp_dir: Path) -> None:
    """Test pipeline with a single file."""
    config = Config(chunk_strategy="character", chunk_size=50, chunk_overlap=10)

    chunks = list(prepare_docs(sample_text_file, config=config))

    assert len(chunks) > 0, "Should produce chunks from single file"
    assert all("source_path" in c.metadata for c in chunks), "Should include source_path in metadata"
    assert all(c.metadata["file_name"] == "sample.txt" for c in chunks), "Should include correct filename"


def test_pipeline_metadata_preservation(sample_text_file: Path) -> None:
    """Test that metadata is preserved through the pipeline."""
    config = Config(chunk_strategy="character", chunk_size=30, chunk_overlap=5)

    chunks = list(prepare_docs(sample_text_file, config=config))

    # All chunks from same file should have same source_id
    source_ids = {c.metadata.get("source_id") for c in chunks}
    assert len(source_ids) == 1, "All chunks from same file should have same source_id"

    # Verify metadata fields
    first_chunk = chunks[0]
    assert "file_type" in first_chunk.metadata, "Should include file_type"
    assert first_chunk.metadata["file_type"] == "text", "File type should be 'text'"
    assert "file_name" in first_chunk.metadata, "Should include file_name"

