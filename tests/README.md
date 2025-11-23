# Test Suite for rag_prep

This directory contains a comprehensive test suite for the `rag_prep` package.

## Test Organization

- `test_pipeline.py` - End-to-end pipeline tests
- `test_loaders.py` - Document loader tests (text, markdown, PDF, DOCX, HTML, CSV)
- `test_chunkers.py` - Chunking strategy tests (character, sentence, token, none)
- `test_customization.py` - Custom chunker integration and metadata tests
- `test_cli.py` - Command-line interface tests
- `test_error_handling.py` - Error handling and edge case tests
- `conftest.py` - Shared pytest fixtures

## Running Tests

### Install Dependencies

```bash
# Install the package in development mode
pip install -e .

# Install test dependencies
pip install -e ".[dev]"
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Files

```bash
# Run only pipeline tests
pytest tests/test_pipeline.py

# Run only loader tests
pytest tests/test_loaders.py

# Run only chunker tests
pytest tests/test_chunkers.py
```

### Run with Verbose Output

```bash
pytest -v
```

### Run with Coverage

```bash
pytest --cov=rag_prep --cov-report=html
```

## Test Categories

### 1. End-to-End Pipeline Tests
- Tests the complete pipeline with default configuration
- Validates chunk production, metadata, and JSONL output
- Tests single file and directory processing

### 2. Loader Tests
- Tests all built-in loaders (text, markdown, CSV, HTML)
- Tests optional loaders (PDF, DOCX) with graceful skipping
- Tests automatic loader selection
- Tests directory loading with patterns

### 3. Chunker Strategy Tests
- Character chunker: size limits, overlap behavior
- Sentence chunker: sentence boundary alignment
- Token chunker: token-based splitting (requires tiktoken)
- No chunker: pass-through behavior

### 4. Customization Tests
- Custom chunker integration
- Metadata hooks
- Config overrides

### 5. CLI Tests
- Basic usage
- Custom parameters
- Include/exclude patterns
- Error handling

### 6. Error Handling Tests
- Invalid paths
- Empty files/directories
- Unsupported file types
- Edge cases (unicode, special characters, etc.)

## Notes

- Tests use temporary files and directories created via pytest fixtures
- Tests are designed to be deterministic and isolated
- Optional dependencies (PDF, DOCX) are tested with graceful skipping if unavailable
- All tests use type hints for clarity

