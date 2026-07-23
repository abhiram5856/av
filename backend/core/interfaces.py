from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from backend.models.domain import Document, Chunk, EmbeddingRecord, SearchResult, ChatMessage, Conversation

class BaseDocumentLoader(ABC):
    @abstractmethod
    def load(self, source: str) -> List[Document]:
        pass

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Chunk]:
        pass

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        pass

class BaseVectorStore(ABC):
    @abstractmethod
    def add_embeddings(self, records: List[EmbeddingRecord]) -> None:
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        pass

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        pass

class BaseTranslator(ABC):
    @abstractmethod
    def detect_language(self, text: str) -> str:
        pass

    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        pass

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

class BaseMemory(ABC):
    @abstractmethod
    def add_message(self, session_id: str, message: ChatMessage) -> None:
        pass

    @abstractmethod
    def get_history(self, session_id: str, limit: int = 10) -> List[ChatMessage]:
        pass

class BasePromptBuilder(ABC):
    @abstractmethod
    def build_prompt(self, query: str, context: List[SearchResult], chat_history: List[ChatMessage]) -> str:
        pass
