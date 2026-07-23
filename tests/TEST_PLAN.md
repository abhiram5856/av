# Unit Test Plan

Our testing strategy follows the testing pyramid, prioritizing isolated unit tests for core logic, ensuring robust CI/CD integration.

## Tools
- `pytest`
- `pytest-mock`
- `pytest-cov`

## Module Test Breakdown

### 1. `tests/test_document_loader.py`
- Mock filesystem interactions.
- Assert correct extraction of text from known PDF and DOCX mocks.
- Test `DocumentLoadError` on corrupted files.

### 2. `tests/test_chunking.py`
- Test character count limits and overlap correctness.
- Assert edge cases (e.g., text smaller than chunk size).

### 3. `tests/test_embeddings.py`
- Mock `sentence-transformers`.
- Assert shape of generated embeddings matches expected dimension (e.g., 384).

### 4. `tests/test_vectorstore.py`
- Use a mock FAISS index in memory.
- Assert `add_embeddings` increases index size.
- Assert `search` returns `SearchResult` with correct scores.

### 5. `tests/test_translation.py`
- Mock `langdetect`.
- Test mapping of Hindi/Telugu inputs to expected English strings using mock translation functions.

### 6. `tests/test_prompt_builder.py`
- Pure function test. 
- Assert the final string contains the user query, context chunks, and history correctly formatted.

### 7. `tests/test_chat_engine.py`
- **Integration Test**: Mock all external dependencies (LLM, Embeddings).
- Assert the data flow: Input -> Translate -> Retrieve -> Prompt -> Output.
