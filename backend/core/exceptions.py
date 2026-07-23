class ZenithAgriBotError(Exception):
    """Base exception for Zenith AgriBot."""
    pass

class DocumentLoadError(ZenithAgriBotError):
    """Raised when a document cannot be loaded."""
    pass

class ChunkingError(ZenithAgriBotError):
    """Raised when an error occurs during text chunking."""
    pass

class EmbeddingError(ZenithAgriBotError):
    """Raised when the embedding model fails to generate vectors."""
    pass

class VectorStoreError(ZenithAgriBotError):
    """Raised for issues communicating with the vector store (e.g., FAISS)."""
    pass

class RetrievalError(ZenithAgriBotError):
    """Raised when the retriever fails to fetch relevant documents."""
    pass

class TranslationError(ZenithAgriBotError):
    """Raised when language detection or translation fails."""
    pass

class LLMError(ZenithAgriBotError):
    """Raised when the local LLM fails to generate a response."""
    pass

class ConfigurationError(ZenithAgriBotError):
    """Raised when there is an invalid configuration setting."""
    pass
