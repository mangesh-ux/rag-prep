"""
rag_prep: A minimal, extensible framework for preparing documents for RAG/LLM workflows.

Pipeline: load → normalize → chunk → emit
"""

from rag_prep.models import Chunk, Config
from rag_prep.pipeline import prepare_docs, prepare_docs_to_jsonl

__version__ = "0.1.0"
__all__ = [
    "Chunk",
    "Config",
    "prepare_docs",
    "prepare_docs_to_jsonl",
]

