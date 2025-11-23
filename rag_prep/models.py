"""Core data models for rag_prep."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Chunk:
    """A document chunk with text content and flexible metadata."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            "text": self.text,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Create chunk from dictionary."""
        return cls(
            text=data["text"],
            metadata=data.get("metadata", {}),
            chunk_id=data.get("chunk_id"),
        )


@dataclass
class Config:
    """Configuration for the document preparation pipeline."""

    chunk_strategy: str = "character"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    tokenizer_name: Optional[str] = "cl100k_base"  # tiktoken default
    include_patterns: Optional[list] = None
    exclude_patterns: Optional[list] = None
    metadata_hooks: Optional[list] = None  # List of callables to enrich metadata
    verbose: bool = False

    def __post_init__(self):
        """Normalize configuration values."""
        if self.include_patterns is None:
            self.include_patterns = []
        if self.exclude_patterns is None:
            self.exclude_patterns = []
        if self.metadata_hooks is None:
            self.metadata_hooks = []

