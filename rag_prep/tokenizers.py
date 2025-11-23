"""Tokenization backend abstraction."""

from typing import Optional, Protocol


class Tokenizer(Protocol):
    """Protocol for tokenization backends."""

    def encode(self, text: str) -> list:
        """Encode text into tokens."""
        ...

    def decode(self, tokens: list) -> str:
        """Decode tokens back to text."""
        ...


def get_tokenizer(name: Optional[str] = "cl100k_base") -> Optional[Tokenizer]:
    """
    Get a tokenizer instance by name.

    Args:
        name: Tokenizer name. If None, returns None (no tokenization).
              Supported: "cl100k_base" (tiktoken), or a custom tokenizer instance.

    Returns:
        Tokenizer instance or None.
    """
    if name is None:
        return None

    try:
        import tiktoken

        encoding = tiktoken.get_encoding(name)
        return TiktokenWrapper(encoding)
    except ImportError:
        raise ImportError(
            "tiktoken is required for tokenization. Install with: pip install tiktoken"
        )
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer '{name}': {e}")


class TiktokenWrapper:
    """Wrapper around tiktoken encoding to match Tokenizer protocol."""

    def __init__(self, encoding):
        self.encoding = encoding

    def encode(self, text: str) -> list:
        """Encode text into tokens."""
        return self.encoding.encode(text)

    def decode(self, tokens: list) -> str:
        """Decode tokens back to text."""
        return self.encoding.decode(tokens)

