from typing import List
from sentence_transformers import SentenceTransformer
from backend.core.interfaces import BaseEmbeddingModel
from backend.core_logging.logger import retrieve_logger

class LocalEmbedder(BaseEmbeddingModel):
    """
    Generates embeddings locally using a fast HuggingFace sentence-transformer model.
    Default model: 'all-MiniLM-L6-v2' (384 dimensions).
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        retrieve_logger.info(f"Loading local embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single string into a vector.
        """
        embedding = self.model.encode(text)
        return embedding.tolist()
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a batch of strings into vectors efficiently.
        """
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
