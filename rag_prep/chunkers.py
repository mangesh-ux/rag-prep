"""Chunking strategies for document preparation."""

import re
from typing import Iterator, Optional, Protocol

from rag_prep.models import Chunk
from rag_prep.tokenizers import Tokenizer, get_tokenizer


class ChunkingStrategy(Protocol):
    """Protocol for chunking strategies."""

    def chunk(self, text: str, metadata: dict) -> Iterator[Chunk]:
        """
        Chunk text into smaller pieces.

        Args:
            text: The text to chunk.
            metadata: Base metadata to attach to each chunk.

        Yields:
            Chunk objects.
        """
        ...


class CharacterChunker:
    """Chunk text by character count."""

    def __init__(self, size: int = 1000, overlap: int = 200):
        self.size = size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict) -> Iterator[Chunk]:
        """Chunk text by character count with overlap."""
        if not text:
            return

        start = 0
        chunk_idx = 0
        while start < len(text):
            end = start + self.size
            chunk_text = text[start:end]

            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = chunk_idx
            chunk_metadata["start_char"] = start
            chunk_metadata["end_char"] = min(end, len(text))

            yield Chunk(
                text=chunk_text,
                metadata=chunk_metadata,
                chunk_id=f"{metadata.get('source_id', 'doc')}_chunk_{chunk_idx}",
            )

            chunk_idx += 1
            start = end - self.overlap
            if start >= len(text):
                break


class TokenChunker:
    """Chunk text by token count using a tokenizer."""

    def __init__(
        self,
        size: int = 1000,
        overlap: int = 200,
        tokenizer: Optional[Tokenizer] = None,
        tokenizer_name: Optional[str] = "cl100k_base",
    ):
        self.size = size
        self.overlap = overlap
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            try:
                self.tokenizer = get_tokenizer(tokenizer_name)
            except (ImportError, ValueError):
                # Fall back to None if tokenizer can't be loaded
                self.tokenizer = None

    def chunk(self, text: str, metadata: dict) -> Iterator[Chunk]:
        """Chunk text by token count with overlap."""
        if not text or self.tokenizer is None:
            # Fallback to character chunking if no tokenizer
            chunker = CharacterChunker(self.size, self.overlap)
            yield from chunker.chunk(text, metadata)
            return

        tokens = self.tokenizer.encode(text)
        if not tokens:
            return

        start = 0
        chunk_idx = 0
        while start < len(tokens):
            end = start + self.size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = chunk_idx
            chunk_metadata["start_token"] = start
            chunk_metadata["end_token"] = min(end, len(tokens))

            yield Chunk(
                text=chunk_text,
                metadata=chunk_metadata,
                chunk_id=f"{metadata.get('source_id', 'doc')}_chunk_{chunk_idx}",
            )

            chunk_idx += 1
            start = end - self.overlap
            if start >= len(tokens):
                break


class SentenceChunker:
    """Chunk text by sentences, respecting size limits."""

    def __init__(self, size: int = 1000, overlap: int = 200):
        self.size = size
        self.overlap = overlap

    def chunk(self, text: str, metadata: dict) -> Iterator[Chunk]:
        """Chunk text by sentences with size limits."""
        if not text:
            return

        # Simple sentence splitting (can be improved)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if not sentences:
            # Fallback to character chunking
            chunker = CharacterChunker(self.size, self.overlap)
            yield from chunker.chunk(text, metadata)
            return

        current_chunk = []
        current_size = 0
        chunk_idx = 0
        overlap_sentences = []

        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size <= self.size:
                current_chunk.append(sentence)
                current_size += sentence_size
            else:
                # Emit current chunk
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = chunk_idx

                    yield Chunk(
                        text=chunk_text,
                        metadata=chunk_metadata,
                        chunk_id=f"{metadata.get('source_id', 'doc')}_chunk_{chunk_idx}",
                    )

                    chunk_idx += 1

                    # Prepare overlap
                    overlap_sentences = []
                    overlap_size = 0
                    for s in reversed(current_chunk):
                        if overlap_size + len(s) <= self.overlap:
                            overlap_sentences.insert(0, s)
                            overlap_size += len(s)
                        else:
                            break

                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) for s in current_chunk)

        # Emit final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = chunk_idx

            yield Chunk(
                text=chunk_text,
                metadata=chunk_metadata,
                chunk_id=f"{metadata.get('source_id', 'doc')}_chunk_{chunk_idx}",
            )


class NoChunker:
    """Pass-through chunker that returns the entire text as a single chunk."""

    def chunk(self, text: str, metadata: dict) -> Iterator[Chunk]:
        """Return entire text as a single chunk."""
        yield Chunk(
            text=text,
            metadata=metadata.copy(),
            chunk_id=metadata.get("source_id", "doc"),
        )


def get_chunker(
    strategy: str,
    size: int = 1000,
    overlap: int = 200,
    tokenizer: Optional[Tokenizer] = None,
    tokenizer_name: Optional[str] = "cl100k_base",
) -> ChunkingStrategy:
    """
    Get a chunking strategy by name.

    Args:
        strategy: Strategy name ("character", "token", "sentence", "none")
                  or a fully qualified class path (e.g., "mypackage.CustomChunker").
        size: Maximum chunk size.
        overlap: Overlap between chunks.
        tokenizer: Optional tokenizer instance.
        tokenizer_name: Tokenizer name if tokenizer not provided.

    Returns:
        ChunkingStrategy instance.
    """
    strategy_lower = strategy.lower()

    if strategy_lower == "character":
        return CharacterChunker(size, overlap)
    elif strategy_lower == "token":
        return TokenChunker(size, overlap, tokenizer, tokenizer_name)
    elif strategy_lower == "sentence":
        return SentenceChunker(size, overlap)
    elif strategy_lower == "none":
        return NoChunker()
    else:
        # Try to load custom chunker from module path
        try:
            parts = strategy.split(".")
            module_name = ".".join(parts[:-1])
            class_name = parts[-1]

            import importlib

            module = importlib.import_module(module_name)
            chunker_class = getattr(module, class_name)

            # Try to instantiate with common parameters
            try:
                return chunker_class(size=size, overlap=overlap)
            except TypeError:
                # Try without parameters
                return chunker_class()
        except Exception as e:
            raise ValueError(
                f"Unknown chunking strategy '{strategy}'. "
                f"Supported: 'character', 'token', 'sentence', 'none', "
                f"or a module path like 'mypackage.CustomChunker'. Error: {e}"
            )

