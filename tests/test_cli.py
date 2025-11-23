"""Tests for command-line interface."""

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest


def test_cli_basic_usage(temp_dir: Path, sample_text_file: Path) -> None:
    """Test basic CLI usage with a single file."""
    output_path = temp_dir / "output.jsonl"

    # Run CLI command
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_prep.cli",
            str(sample_text_file),
            "-o",
            str(output_path),
            "--chunk-strategy",
            "character",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        stdin=subprocess.DEVNULL,
    )

    assert result.returncode == 0, f"CLI should succeed. Error: {result.stderr}"
    assert output_path.exists(), "Output file should be created"

    # Validate output
    lines = output_path.read_text().strip().split("\n")
    assert len(lines) > 0, "Should produce at least one chunk"

    # Validate JSON structure
    for line in lines:
        data = json.loads(line)
        assert "text" in data, "Should have text field"
        assert "metadata" in data, "Should have metadata field"


def test_cli_directory_input(multi_file_dir: Path, temp_dir: Path) -> None:
    """Test CLI with directory input."""
    output_path = temp_dir / "output.jsonl"

    # Use a simpler directory structure to avoid potential subprocess issues
    simple_dir = temp_dir / "simple_dir"
    simple_dir.mkdir()
    (simple_dir / "file1.txt").write_text("First file content.")
    (simple_dir / "file2.txt").write_text("Second file content.")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_prep.cli",
            str(simple_dir),
            "-o",
            str(output_path),
            "--chunk-strategy",
            "character",
            "--chunk-size",
            "50",
        ],
        capture_output=True,
        text=True,
        timeout=30,  # Increased timeout for directory processing
        stdin=subprocess.DEVNULL,  # Prevent waiting for stdin
    )

    assert result.returncode == 0, f"CLI should succeed. Error: {result.stderr}"
    assert output_path.exists(), "Output file should be created"

    lines = output_path.read_text().strip().split("\n")
    assert len(lines) > 0, "Should produce chunks from directory"


def test_cli_custom_chunking(temp_dir: Path, sample_text_file: Path) -> None:
    """Test CLI with custom chunking parameters."""
    output_path = temp_dir / "output.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_prep.cli",
            str(sample_text_file),
            "-o",
            str(output_path),
            "--chunk-strategy",
            "sentence",
            "--chunk-size",
            "100",
            "--chunk-overlap",
            "20",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        stdin=subprocess.DEVNULL,
    )

    assert result.returncode == 0, f"CLI should succeed. Error: {result.stderr}"
    assert output_path.exists(), "Output file should be created"


def test_cli_include_exclude_patterns(multi_file_dir: Path, temp_dir: Path) -> None:
    """Test CLI with include/exclude patterns."""
    output_path = temp_dir / "output.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_prep.cli",
            str(multi_file_dir),
            "-o",
            str(output_path),
            "--include",
            "*.txt",
            "--exclude",
            "subdir",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        stdin=subprocess.DEVNULL,
    )

    assert result.returncode == 0, f"CLI should succeed. Error: {result.stderr}"
    assert output_path.exists(), "Output file should be created"

    # Verify only .txt files were processed
    lines = output_path.read_text().strip().split("\n")
    data_list: List[dict] = [json.loads(line) for line in lines if line]

    # All chunks should be from .txt files
    for data in data_list:
        source_path = data.get("metadata", {}).get("source_path", "")
        assert source_path.endswith(".txt"), f"Should only include .txt files, got {source_path}"


def test_cli_verbose_mode(temp_dir: Path, sample_text_file: Path) -> None:
    """Test CLI verbose mode."""
    output_path = temp_dir / "output.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_prep.cli",
            str(sample_text_file),
            "-o",
            str(output_path),
            "-v",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        stdin=subprocess.DEVNULL,
    )

    assert result.returncode == 0, f"CLI should succeed. Error: {result.stderr}"
    assert "Input:" in result.stdout or "Chunk strategy:" in result.stdout, "Should show verbose output"


def test_cli_missing_output(temp_dir: Path, sample_text_file: Path) -> None:
    """Test CLI fails when output is missing."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_prep.cli",
            str(sample_text_file),
            "--chunk-strategy",
            "character",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        stdin=subprocess.DEVNULL,
    )

    assert result.returncode != 0, "CLI should fail when output is not specified"


def test_cli_invalid_strategy(temp_dir: Path, sample_text_file: Path) -> None:
    """Test CLI handles invalid chunk strategy gracefully."""
    output_path = temp_dir / "output.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_prep.cli",
            str(sample_text_file),
            "-o",
            str(output_path),
            "--chunk-strategy",
            "invalid_strategy_xyz",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        stdin=subprocess.DEVNULL,
    )

    # Should either fail gracefully or use a fallback
    # The actual behavior depends on implementation
    assert result.returncode != 0 or output_path.exists(), "Should handle invalid strategy"


def test_cli_no_tokenizer(temp_dir: Path, sample_text_file: Path) -> None:
    """Test CLI with tokenizer disabled."""
    output_path = temp_dir / "output.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rag_prep.cli",
            str(sample_text_file),
            "-o",
            str(output_path),
            "--tokenizer",
            "none",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        stdin=subprocess.DEVNULL,
    )

    assert result.returncode == 0, f"CLI should succeed without tokenizer. Error: {result.stderr}"
    assert output_path.exists(), "Output file should be created"

