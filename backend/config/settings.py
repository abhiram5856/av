import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Config
    APP_NAME: str = "Zenith AgriBot API"
    APP_VERSION: str = "0.1.0"
    DEBUG_MODE: bool = False

    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    KNOWLEDGE_BASE_DIR: str = os.path.join(BASE_DIR, "..", "knowledge_base")
    FAISS_INDEX_PATH: str = os.path.join(BASE_DIR, "..", "data", "faiss_index")

    # Embedding Config
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Retrieval Config
    RETRIEVAL_TOP_K: int = 5

    # LLM Config
    LLM_MODEL_NAME: str = "llama3" # for Ollama
    LLM_TEMPERATURE: float = 0.2

    # Language Options
    SUPPORTED_LANGUAGES: List[str] = ["en", "te", "hi"]
    DEFAULT_LANGUAGE: str = "en"

    # Future CNN Integration
    CNN_MODEL_PATH: str = os.path.join(BASE_DIR, "..", "models", "disease_cnn.pt")
    ENABLE_CNN: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
