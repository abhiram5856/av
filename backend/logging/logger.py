import logging
import sys
from backend.config.settings import settings

def setup_logger(name: str) -> logging.Logger:
    """Configures and returns a logger for a specific module."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.DEBUG if settings.DEBUG_MODE else logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Optionally add a File handler in the future for production
        
    return logger

# Common loggers
doc_logger = setup_logger("zenith.document_loader")
chunk_logger = setup_logger("zenith.chunking")
embed_logger = setup_logger("zenith.embeddings")
retrieve_logger = setup_logger("zenith.retrieval")
llm_logger = setup_logger("zenith.llm")
translate_logger = setup_logger("zenith.translation")
api_logger = setup_logger("zenith.api")
