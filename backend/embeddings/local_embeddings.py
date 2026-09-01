from typing import List
import torch
from sentence_transformers import SentenceTransformer
from backend.core.interfaces import BaseEmbeddingModel
from backend.core.exceptions import EmbeddingError
from backend.config.settings import settings
from backend.core_logging.logger import embed_logger

class SentenceTransformerModel(BaseEmbeddingModel):
    """
    Local embedding model using sentence-transformers, optimized for GPU if available.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        
        # Check for GPU (since user has RTX 4060)
        import os
        if os.environ.get("NOVA_FORCE_CPU") == "true":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        embed_logger.info(f"Initializing SentenceTransformer: {self.model_name} on {self.device}")
        
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            embed_logger.error(f"Failed to load embedding model: {str(e)}")
            raise EmbeddingError(f"Could not load {self.model_name}: {str(e)}")

    def embed_text(self, text: str) -> List[float]:
        try:
            # We convert to a list of python floats for strict Pydantic compliance later
            vector = self.model.encode(text, convert_to_numpy=True).tolist()
            return vector
        except Exception as e:
            embed_logger.error(f"Failed to embed single text: {str(e)}")
            raise EmbeddingError(f"Embedding failed: {str(e)}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
            
        try:
            embed_logger.debug(f"Embedding batch of {len(texts)} texts...")
            vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return [vec.tolist() for vec in vectors]
        except Exception as e:
            embed_logger.error(f"Failed to embed batch: {str(e)}")
            raise EmbeddingError(f"Batch embedding failed: {str(e)}")
