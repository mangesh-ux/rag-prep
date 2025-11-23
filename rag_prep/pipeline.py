"""Main pipeline for document preparation."""

from pathlib import Path
from typing import Iterator, Optional, Union

from rag_prep.chunkers import ChunkingStrategy, get_chunker
from rag_prep.loaders import get_loader
from rag_prep.models import Chunk, Config
from rag_prep.sinks import write_jsonl
from rag_prep.tokenizers import Tokenizer, get_tokenizer


def prepare_docs(
    source: Union[str, Path],
    config: Optional[Config] = None,
    chunker: Optional[ChunkingStrategy] = None,
    tokenizer: Optional[Tokenizer] = None,
) -> Iterator[Chunk]:
    """
    Prepare documents for RAG/LLM workflows.

    Pipeline: load → normalize → chunk → emit

    Args:
        source: Path to file, directory, or string input.
        config: Configuration object. If None, uses defaults.
        chunker: Optional custom chunker. If None, uses config.chunk_strategy.
        tokenizer: Optional custom tokenizer. If None, uses config.tokenizer_name.

    Yields:
        Chunk objects ready for RAG/LLM workflows.
    """
    if config is None:
        config = Config()

    # Get loader
    loader = get_loader(
        source,
        include_patterns=config.include_patterns,
        exclude_patterns=config.exclude_patterns,
    )

    # Get tokenizer if needed
    if tokenizer is None and config.tokenizer_name:
        try:
            tokenizer = get_tokenizer(config.tokenizer_name)
        except (ImportError, ValueError):
            if config.verbose:
                print(f"Warning: Could not load tokenizer '{config.tokenizer_name}', proceeding without tokenization")
            tokenizer = None

    # Get chunker
    if chunker is None:
        chunker = get_chunker(
            strategy=config.chunk_strategy,
            size=config.chunk_size,
            overlap=config.chunk_overlap,
            tokenizer=tokenizer,
            tokenizer_name=config.tokenizer_name,
        )

    # Load documents
    for doc_chunk in loader.load(source):
        # Apply metadata hooks
        for hook in config.metadata_hooks:
            if callable(hook):
                doc_chunk.metadata = hook(doc_chunk.metadata, doc_chunk.text)

        # Chunk the document
        for chunk in chunker.chunk(doc_chunk.text, doc_chunk.metadata):
            yield chunk


def prepare_docs_to_jsonl(
    source: Union[str, Path],
    output_path: Union[str, Path],
    config: Optional[Config] = None,
    chunker: Optional[ChunkingStrategy] = None,
    tokenizer: Optional[Tokenizer] = None,
) -> None:
    """
    Prepare documents and write to JSONL file.

    Args:
        source: Path to file, directory, or string input.
        output_path: Path to output JSONL file.
        config: Configuration object. If None, uses defaults.
        chunker: Optional custom chunker.
        tokenizer: Optional custom tokenizer.
    """
    chunks = prepare_docs(source, config, chunker, tokenizer)
    write_jsonl(chunks, output_path)

