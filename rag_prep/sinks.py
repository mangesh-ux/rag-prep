"""Output sinks for processed chunks."""

import json
from pathlib import Path
from typing import Iterator, List, Optional, Protocol, Union

from rag_prep.models import Chunk


class Sink(Protocol):
    """Protocol for output sinks."""

    def write(self, chunks: Iterator[Chunk]) -> None:
        """
        Write chunks to the sink.

        Args:
            chunks: Iterator of chunks to write.
        """
        ...


class JSONLSink:
    """Write chunks to a JSONL file."""

    def __init__(self, output_path: Union[str, Path]):
        self.output_path = Path(output_path)

    def write(self, chunks: Iterator[Chunk]) -> None:
        """Write chunks to JSONL file."""
        with open(self.output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                json.dump(chunk.to_dict(), f, ensure_ascii=False)
                f.write("\n")


class ListSink:
    """Collect chunks into a list (in-memory sink)."""

    def __init__(self):
        self.chunks = []

    def write(self, chunks: Iterator[Chunk]) -> None:
        """Collect chunks into list."""
        self.chunks.extend(chunks)

    def get_chunks(self) -> List[Chunk]:
        """Get collected chunks."""
        return self.chunks


def write_jsonl(chunks: Iterator[Chunk], output_path: Union[str, Path]) -> None:
    """
    Convenience function to write chunks to JSONL.

    Args:
        chunks: Iterator of chunks.
        output_path: Path to output JSONL file.
    """
    sink = JSONLSink(output_path)
    sink.write(chunks)

