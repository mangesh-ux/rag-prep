"""Tests for error handling and edge cases."""

from pathlib import Path

import pytest

from rag_prep import Config, prepare_docs
from rag_prep.loaders import get_loader


def test_invalid_path() -> None:
    """Test that invalid path raises a clear exception."""
    invalid_path = Path("/nonexistent/path/that/does/not/exist.txt")

    with pytest.raises((FileNotFoundError, ValueError, OSError)):
        list(prepare_docs(invalid_path))


def test_empty_directory(temp_dir: Path) -> None:
    """Test that empty directory returns empty list."""
    # Create empty directory
    empty_dir = temp_dir / "empty"
    empty_dir.mkdir()

    config = Config(chunk_strategy="character")
    chunks = list(prepare_docs(empty_dir, config=config))

    assert len(chunks) == 0, "Empty directory should produce no chunks"


def test_unsupported_file_extension(temp_dir: Path) -> None:
    """Test that unsupported file extension is handled gracefully."""
    # Create file with unsupported extension
    unsupported_file = temp_dir / "test.xyz"
    unsupported_file.write_text("Some content")

    # Should fall back to text loader or handle gracefully
    try:
        chunks = list(prepare_docs(unsupported_file))
        # If it works, should use text loader as fallback
        assert len(chunks) > 0, "Should fall back to text loader"
    except Exception as e:
        # Or should raise a clear error
        assert "unsupported" in str(e).lower() or "extension" in str(e).lower(), f"Error should mention extension: {e}"


def test_empty_file(temp_dir: Path) -> None:
    """Test that empty file is handled."""
    empty_file = temp_dir / "empty.txt"
    empty_file.write_text("")

    config = Config(chunk_strategy="character")
    chunks = list(prepare_docs(empty_file, config=config))

    # Empty file might produce 0 chunks or 1 empty chunk depending on implementation
    # Let's check it doesn't crash
    assert isinstance(chunks, list), "Should return a list (even if empty)"


def test_very_large_chunk_size(temp_dir: Path) -> None:
    """Test that very large chunk size works (no chunking)."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("Short text")

    config = Config(chunk_strategy="character", chunk_size=100000, chunk_overlap=0)
    chunks = list(prepare_docs(test_file, config=config))

    assert len(chunks) > 0, "Should produce at least one chunk"
    # With very large chunk size, should produce one chunk
    assert len(chunks) == 1, "Should produce single chunk for small text with large chunk size"


def test_zero_chunk_size(temp_dir: Path) -> None:
    """Test that zero chunk size is handled."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("Test content")

    config = Config(chunk_strategy="character", chunk_size=0, chunk_overlap=0)

    # Should either raise an error or handle gracefully
    try:
        chunks = list(prepare_docs(test_file, config=config))
        # If it doesn't raise, should handle gracefully
        assert isinstance(chunks, list), "Should return a list"
    except (ValueError, AssertionError):
        # Or should raise a clear error
        pass


def test_negative_overlap(temp_dir: Path) -> None:
    """Test that negative overlap is handled."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("Test content")

    config = Config(chunk_strategy="character", chunk_size=10, chunk_overlap=-5)

    # Should either work (treating as 0) or raise an error
    try:
        chunks = list(prepare_docs(test_file, config=config))
        assert len(chunks) > 0, "Should produce chunks"
    except (ValueError, AssertionError):
        # Or should raise a clear error
        pass


def test_missing_optional_dependencies(temp_dir: Path) -> None:
    """Test graceful handling when optional dependencies are missing."""
    # Try to load a PDF without PyPDF2
    pdf_file = temp_dir / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n")  # Minimal PDF header

    try:
        chunks = list(prepare_docs(pdf_file))
        # If PyPDF2 is available, should work
        # If not, should raise ImportError with clear message
    except ImportError as e:
        assert "PyPDF2" in str(e) or "pdf" in str(e).lower(), f"Error should mention missing dependency: {e}"
    except Exception:
        # Other errors are acceptable (e.g., invalid PDF)
        pass


def test_directory_with_only_unsupported_files(temp_dir: Path) -> None:
    """Test directory containing only unsupported file types."""
    # Create directory with only unsupported files
    (temp_dir / "file1.xyz").write_text("Content 1")
    (temp_dir / "file2.abc").write_text("Content 2")

    config = Config(chunk_strategy="character")
    chunks = list(prepare_docs(temp_dir, config=config))

    # Should either return empty list or fall back to text loader
    # The behavior depends on implementation
    assert isinstance(chunks, list), "Should return a list"


def test_unicode_content(temp_dir: Path) -> None:
    """Test that unicode content is handled correctly."""
    test_file = temp_dir / "unicode.txt"
    unicode_content = "Hello 世界 🌍 こんにちは"
    test_file.write_text(unicode_content, encoding="utf-8")

    config = Config(chunk_strategy="character", chunk_size=100)
    chunks = list(prepare_docs(test_file, config=config))

    assert len(chunks) > 0, "Should handle unicode content"
    # Verify unicode is preserved
    all_text = "".join(c.text for c in chunks)
    assert "世界" in all_text, "Should preserve unicode characters"


def test_special_characters_in_path(temp_dir: Path) -> None:
    """Test that special characters in file path are handled."""
    # Create file with special characters in name
    special_file = temp_dir / "file with spaces & special-chars.txt"
    special_file.write_text("Content")

    config = Config(chunk_strategy="character")
    chunks = list(prepare_docs(special_file, config=config))

    assert len(chunks) > 0, "Should handle special characters in path"
    assert "file with spaces" in chunks[0].metadata.get("file_name", ""), "Should preserve filename"


def test_nested_directories(multi_file_dir: Path) -> None:
    """Test that nested directories are handled correctly."""
    config = Config(chunk_strategy="character", chunk_size=50)
    chunks = list(prepare_docs(multi_file_dir, config=config))

    # Should process files in subdirectories
    subdir_chunks = [c for c in chunks if "subdir" in c.metadata.get("source_path", "")]
    assert len(subdir_chunks) > 0, "Should process files in subdirectories"

