# Basic Example

This example demonstrates the simplest usage of rag_prep with a small collection of documents.

## Files

- `docs/example.txt` - A sample text file
- `docs/example.md` - A sample markdown file

## Usage

### Command Line

Process the documents directory and output to JSONL:

```bash
rag-prep ./examples/basic/docs -o ./examples/basic/output.jsonl --chunk-strategy character
```

Or with more options:

```bash
rag-prep ./examples/basic/docs -o ./examples/basic/output.jsonl \
    --chunk-strategy character \
    --chunk-size 500 \
    --chunk-overlap 50
```

### Python API

```python
from rag_prep import prepare_docs, Config

config = Config(chunk_strategy="character", chunk_size=500, chunk_overlap=50)
chunks = list(prepare_docs("./examples/basic/docs", config=config))
print(f"Created {len(chunks)} chunks")

# Process chunks
for chunk in chunks:
    print(f"Chunk from {chunk.metadata.get('file_name')}: {chunk.text[:100]}...")
```

## Expected Output

The output JSONL file will contain one chunk per line, with each chunk including:
- `text`: The chunk content
- `metadata`: File information and chunk index
- `chunk_id`: Unique identifier for the chunk

