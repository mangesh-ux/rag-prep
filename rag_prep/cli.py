"""Command-line interface for rag_prep."""

import argparse
import sys
from pathlib import Path

from rag_prep.models import Config
from rag_prep.pipeline import prepare_docs_to_jsonl


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Prepare documents for RAG/LLM workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input path (file, directory, or string)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path",
    )

    parser.add_argument(
        "--chunk-strategy",
        type=str,
        default="character",
        help="Chunking strategy: 'character', 'token', 'sentence', 'none', "
        "or module path like 'mypackage.CustomChunker' (default: character)",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size (default: 1000)",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks (default: 200)",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="cl100k_base",
        help="Tokenizer name (e.g., 'cl100k_base' for tiktoken) or 'none' to disable (default: cl100k_base)",
    )

    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        help="Include file patterns (e.g., '*.txt', '*.md')",
    )

    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        help="Exclude file patterns (e.g., '*.tmp', '__pycache__')",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Build config
    config = Config(
        chunk_strategy=args.chunk_strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        tokenizer_name=None if args.tokenizer.lower() == "none" else args.tokenizer,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
        verbose=args.verbose,
    )

    if args.verbose:
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Chunk strategy: {config.chunk_strategy}")
        print(f"Chunk size: {config.chunk_size}")
        print(f"Chunk overlap: {config.chunk_overlap}")
        print(f"Tokenizer: {config.tokenizer_name}")

    try:
        prepare_docs_to_jsonl(
            source=args.input,
            output_path=args.output,
            config=config,
        )
        if args.verbose:
            print(f"Successfully wrote chunks to {args.output}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

