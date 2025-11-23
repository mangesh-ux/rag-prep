"""Pytest configuration and shared fixtures."""

import tempfile
from pathlib import Path
from typing import Iterator

import pytest


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text_file(temp_dir: Path) -> Path:
    """Create a sample text file."""
    file_path = temp_dir / "sample.txt"
    file_path.write_text("This is a sample text file.\nIt has multiple lines.\nFor testing purposes.")
    return file_path


@pytest.fixture
def sample_markdown_file(temp_dir: Path) -> Path:
    """Create a sample markdown file."""
    file_path = temp_dir / "sample.md"
    file_path.write_text("# Heading\n\nThis is markdown content.\n\nWith **bold** text.")
    return file_path


@pytest.fixture
def sample_csv_file(temp_dir: Path) -> Path:
    """Create a sample CSV file."""
    file_path = temp_dir / "sample.csv"
    file_path.write_text("name,age,city\nAlice,30,New York\nBob,25,London\nCharlie,35,Paris")
    return file_path


@pytest.fixture
def sample_html_file(temp_dir: Path) -> Path:
    """Create a sample HTML file."""
    file_path = temp_dir / "sample.html"
    file_path.write_text(
        "<html><body><h1>Title</h1><p>This is HTML content.</p><script>console.log('ignore');</script></body></html>"
    )
    return file_path


@pytest.fixture
def multi_file_dir(temp_dir: Path) -> Path:
    """Create a directory with multiple test files."""
    (temp_dir / "file1.txt").write_text("First file content.")
    (temp_dir / "file2.md").write_text("# Second file\n\nMarkdown content.")
    (temp_dir / "subdir").mkdir()
    (temp_dir / "subdir" / "file3.txt").write_text("Nested file content.")
    return temp_dir


@pytest.fixture
def long_text_file(temp_dir: Path) -> Path:
    """Create a long text file for chunking tests."""
    content = " ".join([f"Word{i}" for i in range(1000)])
    file_path = temp_dir / "long.txt"
    file_path.write_text(content)
    return file_path

