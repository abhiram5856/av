from typing import List
from backend.core.interfaces import BaseRetriever, BaseEmbeddingModel, BaseVectorStore
from backend.models.domain import SearchResult
from backend.core.exceptions import RetrievalError
from backend.core_logging.logger import retrieve_logger
from backend.config.settings import settings

class VectorRetriever(BaseRetriever):
    """
    Combines embedding generation and vector store searching into a single retrieval step.
    """

    def __init__(self, embedder: BaseEmbeddingModel, vector_store: BaseVectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = None) -> List[SearchResult]:
        k = top_k or settings.RETRIEVAL_TOP_K
        retrieve_logger.info(f"Retrieving top {k} chunks for query: {query}")
        
        try:
            # Step 1: Embed the query
            query_vector = self.embedder.embed_text(query)
            
            # Step 2: Search the vector store
            results = self.vector_store.search(query_vector, top_k=k)
            
            retrieve_logger.debug(f"Retrieved {len(results)} results.")
            return results
            
        except Exception as e:
            retrieve_logger.error(f"Retrieval pipeline failed: {str(e)}")
            raise RetrievalError(f"Failed to retrieve contexts: {str(e)}")
