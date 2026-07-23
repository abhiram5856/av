# Module Responsibilities

This document defines the exact scope, inputs, outputs, and dependencies of each backend package.

## `document_loader`
- **Purpose**: Ingest raw files (PDFs, DOCX, TXT, Markdown) and convert them into standard `Document` objects.
- **Inputs**: File paths or byte streams.
- **Outputs**: List of `Document` Pydantic models.
- **Dependencies**: `PyPDF`, `python-docx`.
- **Future**: Connect to web scrapers or cloud storage.

## `chunking`
- **Purpose**: Split large `Document` text into manageable semantic `Chunk` objects.
- **Inputs**: List of `Document`.
- **Outputs**: List of `Chunk` models.
- **Dependencies**: None strictly (regex/spacy potentially).

## `embeddings`
- **Purpose**: Convert text chunks into numerical vectors.
- **Inputs**: Text strings.
- **Outputs**: `EmbeddingRecord` models (List[float]).
- **Dependencies**: `sentence-transformers`.

## `vectorstore`
- **Purpose**: Store and rapidly retrieve high-dimensional vectors.
- **Inputs**: `EmbeddingRecord` for insertion; Vector for search.
- **Outputs**: `SearchResult` models.
- **Dependencies**: `FAISS`, `numpy`.

## `retrieval`
- **Purpose**: Coordinate the embedding of queries and fetching from the vector store.
- **Inputs**: String queries.
- **Outputs**: List of `SearchResult`.
- **Dependencies**: `embeddings`, `vectorstore`.

## `prompt_builder`
- **Purpose**: Construct robust prompts dynamically injecting context, instructions, and history.
- **Inputs**: Query, Context chunks, Chat History, System Prompts.
- **Outputs**: Formatted String Prompt.
- **Dependencies**: None.

## `llm`
- **Purpose**: Interface with local Large Language Models to generate text.
- **Inputs**: String prompt.
- **Outputs**: String response.
- **Dependencies**: `ollama`, `requests`.

## `translation`
- **Purpose**: Detect language and translate seamlessly between EN, TE, HI.
- **Inputs**: Text strings.
- **Outputs**: Text strings, Language code.
- **Dependencies**: `langdetect`, custom translation models.

## `memory`
- **Purpose**: Maintain conversational context across multiple turns.
- **Inputs**: Session IDs, `ChatMessage` models.
- **Outputs**: List of `ChatMessage`.
- **Dependencies**: SQLite or In-Memory stores.

## `chat_engine`
- **Purpose**: The orchestrator. Ties all the above modules together sequentially.
- **Inputs**: `UserQuery`.
- **Outputs**: `ChatResponse`.
- **Dependencies**: ALL other core modules.
