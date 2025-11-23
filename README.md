# rag_prep

A minimal, extensible framework for preparing documents for RAG/LLM workflows.

**Pipeline:** `load → normalize → chunk → emit`

## Design Philosophy

- **Simple, orthogonal building blocks** - Each component has a clear, single responsibility
- **Clear interfaces for extensibility** - Protocols and ABCs make it easy to plug in custom implementations
- **Sensible defaults with freedom to override** - Works out of the box, but nothing is hardcoded
- **Python API first** - CLI is a thin wrapper over the Python API

## Installation

```bash
pip install rag-prep
```

For optional file format support:

```bash
# PDF support
pip install rag-prep[pdf]

# DOCX support
pip install rag-prep[docx]

# HTML support
pip install rag-prep[html]

# All optional dependencies
pip install rag-prep[all]
```

## Quick Start

### Python API

The source path can be a single file or a directory; directories are walked recursively by default, respecting include/exclude filters.

```python
from rag_prep import prepare_docs, prepare_docs_to_jsonl, Config

# Simple usage with defaults
prepare_docs_to_jsonl("document.txt", "output.jsonl")

# With custom configuration
config = Config(
    chunk_strategy="token",
    chunk_size=500,
    chunk_overlap=100,
    tokenizer_name="cl100k_base",
)

prepare_docs_to_jsonl("document.txt", "output.jsonl", config=config)

# Get chunks as iterator (for custom processing)
for chunk in prepare_docs("document.txt", config=config):
    print(chunk.text)
    print(chunk.metadata)
```

### CLI

```bash
# Basic usage
rag-prep document.txt -o output.jsonl

# With custom chunking
rag-prep document.txt -o output.jsonl \
    --chunk-strategy token \
    --chunk-size 500 \
    --chunk-overlap 100

# Process directory with filters
rag-prep ./documents -o output.jsonl \
    --include "*.txt" "*.md" \
    --exclude "*.tmp" \
    --chunk-strategy sentence

# Verbose output
rag-prep document.txt -o output.jsonl -v
```

## Extension Points

### 1. Custom Chunking Strategies

Create a custom chunker by implementing the `ChunkingStrategy` protocol:

```python
from typing import Iterator
from rag_prep.models import Chunk
from rag_prep.chunkers import ChunkingStrategy

class MyCustomChunker:
    def __init__(self, size: int = 1000, overlap: int = 200):
        self.size = size
        self.overlap = overlap
    
    def chunk(self, text: str, metadata: dict) -> Iterator[Chunk]:
        # Your custom chunking logic
        # ...
        yield Chunk(text=chunk_text, metadata=chunk_metadata, chunk_id="custom_id")
```

Use it in Python:

```python
from rag_prep import prepare_docs, Config

config = Config(chunk_strategy="character")  # Will use your chunker if registered
chunker = MyCustomChunker(size=500)
chunks = prepare_docs("doc.txt", config=config, chunker=chunker)
```

Or via CLI with module path:

```bash
rag-prep doc.txt -o out.jsonl --chunk-strategy mypackage.MyCustomChunker
```

### 2. Custom Loaders

Register a loader for a new file type:

```python
from rag_prep.loaders import register_loader
from rag_prep.models import Chunk
from pathlib import Path

class JSONLoader:
    def load(self, source):
        import json
        path = Path(source)
        with open(path, 'r') as f:
            data = json.load(f)
            # Convert to Chunk objects
            yield Chunk(
                text=str(data), 
                metadata={"source": str(path), "file_type": "json"},
                chunk_id=str(path)
            )

# Register for .json files
register_loader(".json", JSONLoader())
```

### 3. Custom Output Sinks

Implement the `Sink` protocol:

```python
from rag_prep.sinks import Sink
from rag_prep.models import Chunk
from typing import Iterator

class DatabaseSink:
    def write(self, chunks: Iterator[Chunk]):
        # Write chunks to your database
        for chunk in chunks:
            # ... insert into database
            pass
```

### 4. Metadata Enrichment

Add metadata hooks to enrich chunks:

```python
from rag_prep import Config, prepare_docs

def add_timestamp(metadata: dict, text: str) -> dict:
    import datetime
    metadata["processed_at"] = datetime.datetime.now().isoformat()
    return metadata

config = Config(metadata_hooks=[add_timestamp])
chunks = prepare_docs("doc.txt", config=config)
```

### 5. Custom Tokenizers

Inject your own tokenizer:

```python
from rag_prep import prepare_docs, Config
from rag_prep.tokenizers import Tokenizer

class MyTokenizer:
    def encode(self, text: str) -> list:
        # Your encoding logic
        return tokens
    
    def decode(self, tokens: list) -> str:
        # Your decoding logic
        return text

tokenizer = MyTokenizer()
config = Config(chunk_strategy="token")
chunks = prepare_docs("doc.txt", config=config, tokenizer=tokenizer)
```

## Built-in Features

### Chunking Strategies

- **`character`** - Character-based chunking (default)
- **`token`** - Token-based chunking (requires tokenizer). If you choose a token-based strategy without providing a tokenizer, rag_prep will use its default tokenizer if available, otherwise it will raise a clear error.
- **`sentence`** - Sentence-aware chunking
- **`none`** - No chunking (entire document as single chunk)

### Supported File Types

- `.txt` - Plain text
- `.md`, `.markdown` - Markdown
- `.pdf` - PDF (requires `rag-prep[pdf]`)
- `.docx` - Word documents (requires `rag-prep[docx]`)
- `.html`, `.htm` - HTML (requires `rag-prep[html]`)
- `.csv` - CSV files (each row becomes a document)

### Output Format

Default output is JSONL (JSON Lines), where each line is a chunk:

```json
{"text": "chunk text...", "metadata": {"source_id": "doc.txt", "chunk_index": 0}, "chunk_id": "doc.txt_chunk_0"}
{"text": "next chunk...", "metadata": {"source_id": "doc.txt", "chunk_index": 1}, "chunk_id": "doc.txt_chunk_1"}
```

## Use Cases

- **Prepare a docs/ folder for ingestion into a vector database** - Process entire directories of mixed file types into chunked JSONL format ready for embedding and indexing.
- **Convert mixed PDFs + DOCX + markdown into JSONL for RAG** - Handle diverse document formats and output a standardized chunked format for retrieval-augmented generation pipelines.
- **Generate chunked datasets suitable for fine-tuning or eval** - Create properly chunked datasets with metadata for training or evaluating language models.
- **Stream processing for large document collections** - Use iterator-based processing to handle large directories without loading everything into memory.

## Architecture

```
rag_prep/
├── models.py      # Chunk, Config data models
├── chunkers.py    # Chunking strategies (Protocol + implementations)
├── loaders.py     # Document loaders (registry pattern)
├── sinks.py       # Output sinks (Protocol + implementations)
├── tokenizers.py  # Tokenization backend (Protocol + tiktoken)
├── pipeline.py    # Main prepare_docs() API
└── cli.py         # Command-line interface
```

## License

MIT

