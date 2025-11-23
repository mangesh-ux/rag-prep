"""Document loaders for various file types and sources."""

import csv
import fnmatch
import os
from pathlib import Path
from typing import Dict, Iterator, Optional, Protocol, Union

from rag_prep.models import Chunk


class Loader(Protocol):
    """Protocol for document loaders."""

    def load(self, source: Union[str, Path]) -> Iterator[Chunk]:
        """
        Load documents from a source.

        Args:
            source: Path to file, directory, or other source identifier.

        Yields:
            Chunk objects (may be single chunk per document or pre-chunked).
        """
        ...


class TextLoader:
    """Loader for plain text files."""

    def load(self, source: Union[str, Path]) -> Iterator[Chunk]:
        """Load text file."""
        path = Path(source)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        metadata = {
            "source_id": str(path),
            "source_path": str(path),
            "file_type": "text",
            "file_name": path.name,
        }

        yield Chunk(text=text, metadata=metadata, chunk_id=str(path))


class MarkdownLoader:
    """Loader for Markdown files."""

    def load(self, source: Union[str, Path]) -> Iterator[Chunk]:
        """Load Markdown file."""
        path = Path(source)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        metadata = {
            "source_id": str(path),
            "source_path": str(path),
            "file_type": "markdown",
            "file_name": path.name,
        }

        yield Chunk(text=text, metadata=metadata, chunk_id=str(path))


class PDFLoader:
    """Loader for PDF files."""

    def load(self, source: Union[str, Path]) -> Iterator[Chunk]:
        """Load PDF file."""
        try:
            import PyPDF2
        except ImportError:
            raise ImportError(
                "PyPDF2 is required for PDF loading. Install with: pip install PyPDF2"
            )

        path = Path(source)
        text_parts = []

        with open(path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(pdf_reader.pages):
                text_parts.append(page.extract_text())

        text = "\n\n".join(text_parts)

        metadata = {
            "source_id": str(path),
            "source_path": str(path),
            "file_type": "pdf",
            "file_name": path.name,
            "num_pages": len(pdf_reader.pages),
        }

        yield Chunk(text=text, metadata=metadata, chunk_id=str(path))


class DocxLoader:
    """Loader for DOCX files."""

    def load(self, source: Union[str, Path]) -> Iterator[Chunk]:
        """Load DOCX file."""
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "python-docx is required for DOCX loading. Install with: pip install python-docx"
            )

        path = Path(source)
        doc = Document(path)
        text_parts = []

        for paragraph in doc.paragraphs:
            text_parts.append(paragraph.text)

        text = "\n\n".join(text_parts)

        metadata = {
            "source_id": str(path),
            "source_path": str(path),
            "file_type": "docx",
            "file_name": path.name,
        }

        yield Chunk(text=text, metadata=metadata, chunk_id=str(path))


class HTMLLoader:
    """Loader for HTML files."""

    def load(self, source: Union[str, Path]) -> Iterator[Chunk]:
        """Load HTML file."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "beautifulsoup4 is required for HTML loading. Install with: pip install beautifulsoup4"
            )

        path = Path(source)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        # Extract text, removing script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator="\n", strip=True)

        metadata = {
            "source_id": str(path),
            "source_path": str(path),
            "file_type": "html",
            "file_name": path.name,
        }

        yield Chunk(text=text, metadata=metadata, chunk_id=str(path))


class CSVLoader:
    """Loader for CSV files."""

    def load(self, source: Union[str, Path]) -> Iterator[Chunk]:
        """Load CSV file, treating each row as a document."""
        path = Path(source)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                # Convert row to text representation
                text_parts = [f"{k}: {v}" for k, v in row.items() if v]
                text = "\n".join(text_parts)

                metadata = {
                    "source_id": f"{path}_row_{row_idx}",
                    "source_path": str(path),
                    "file_type": "csv",
                    "file_name": path.name,
                    "row_index": row_idx,
                    "csv_columns": list(row.keys()),
                }

                yield Chunk(text=text, metadata=metadata, chunk_id=f"{path}_row_{row_idx}")


class StringLoader:
    """Loader for in-memory strings."""

    def load(self, source: Union[str, Path]) -> Iterator[Chunk]:
        """Load from string source."""
        text = str(source)
        metadata = {
            "source_id": "string_input",
            "source_type": "string",
        }

        yield Chunk(text=text, metadata=metadata, chunk_id="string_input")


class DirectoryLoader:
    """Loader for directories, recursively loading matching files."""

    def __init__(
        self,
        include_patterns: Optional[list] = None,
        exclude_patterns: Optional[list] = None,
        loader_registry: Optional[Dict[str, Loader]] = None,
    ):
        self.include_patterns = include_patterns
        self.exclude_patterns = exclude_patterns or []
        self.loader_registry = loader_registry or get_default_loader_registry()

    def _should_load(self, path: Path) -> bool:
        """Check if file should be loaded based on patterns."""
        path_str = str(path)

        # Check exclude patterns first
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(
                path.name, pattern
            ):
                return False

        # Check include patterns
        # If include_patterns is None or empty, include all files
        if self.include_patterns and len(self.include_patterns) > 0:
            for pattern in self.include_patterns:
                if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(
                    path.name, pattern
                ):
                    return True
            return False

        return True

    def _get_loader(self, path: Path) -> Optional[Loader]:
        """Get appropriate loader for file extension."""
        ext = path.suffix.lower()
        return self.loader_registry.get(ext)

    def load(self, source: Union[str, Path]) -> Iterator[Chunk]:
        """Load all matching files from directory."""
        path = Path(source)
        if not path.is_dir():
            raise ValueError(f"Source must be a directory: {source}")

        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = Path(root) / file
                if not self._should_load(file_path):
                    continue

                loader = self._get_loader(file_path)
                if loader:
                    try:
                        yield from loader.load(file_path)
                    except Exception as e:
                        # Log error but continue processing other files
                        # This prevents one bad file from stopping entire directory processing
                        import warnings
                        warnings.warn(
                            f"Failed to load {file_path}: {e}. Skipping this file.",
                            UserWarning
                        )
                        continue


# Loader registry
_LOADER_REGISTRY: Dict[str, Loader] = {
    ".txt": TextLoader(),
    ".md": MarkdownLoader(),
    ".markdown": MarkdownLoader(),
    ".pdf": PDFLoader(),
    ".docx": DocxLoader(),
    ".html": HTMLLoader(),
    ".htm": HTMLLoader(),
    ".csv": CSVLoader(),
}


def get_default_loader_registry() -> Dict[str, Loader]:
    """Get the default loader registry."""
    return _LOADER_REGISTRY.copy()


def register_loader(extension: str, loader: Loader):
    """
    Register a custom loader for a file extension.

    Args:
        extension: File extension (e.g., ".json").
        loader: Loader instance.
    """
    _LOADER_REGISTRY[extension.lower()] = loader


def get_loader(
    source: Union[str, Path],
    include_patterns: Optional[list] = None,
    exclude_patterns: Optional[list] = None,
) -> Loader:
    """
    Get appropriate loader for a source.

    Args:
        source: Path to file, directory, or string.
        include_patterns: Optional include patterns for directory loading.
        exclude_patterns: Optional exclude patterns for directory loading.

    Returns:
        Loader instance.
    """
    path = Path(source)

    # Directory
    if path.is_dir():
        return DirectoryLoader(
            include_patterns=include_patterns, exclude_patterns=exclude_patterns
        )

    # String (if not a file path)
    if not path.exists() and not path.suffix:
        return StringLoader()

    # File by extension
    ext = path.suffix.lower()
    loader = _LOADER_REGISTRY.get(ext)
    if loader:
        return loader

    # Fallback to text loader
    return TextLoader()

