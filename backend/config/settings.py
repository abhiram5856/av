import os
from typing import List
from pydantic_settings import BaseSettings

_BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH: str = os.path.join(_BASE_DIR, 'data', 'agrivision.db').replace('\\', '/')
_DEFAULT_DB: str = f"sqlite+aiosqlite:///{_DB_PATH}"

class Settings(BaseSettings):
    model_config = {
        'extra': 'ignore', 
        'env_file': (os.path.join(_BASE_DIR, '.env'), '.env'), 
        'env_file_encoding': 'utf-8'
    }

    # App Config
    APP_NAME: str = "Zenith AgriBot API"
    APP_VERSION: str = "0.1.0"
    DEBUG_MODE: bool = False
    
    # API Keys
    OPENWEATHER_API_KEY: str = ""

    # Database Config - defaults to resilient local SQLite; override with PostgreSQL in production
    DATABASE_URL: str = _DEFAULT_DB

    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    KNOWLEDGE_BASE_DIR: str = os.path.join(BASE_DIR, "..", "knowledge_base")
    FAISS_INDEX_PATH: str = os.path.join(BASE_DIR, "data", "faiss_index")

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
    CNN_MODEL_PATH: str = os.path.join(BASE_DIR, "models", "weights", "nova_mobilenet_v3_34_classes.pth")
    ENABLE_CNN: bool = False

settings = Settings()

# Automatically patch standard postgres URLs from Render/Neon to use asyncpg
if settings.DATABASE_URL.startswith("postgres://"):
    settings.DATABASE_URL = settings.DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif settings.DATABASE_URL.startswith("postgresql://") and not settings.DATABASE_URL.startswith("postgresql+asyncpg://"):
    settings.DATABASE_URL = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# asyncpg does not support 'sslmode' or 'channel_binding' in the connection arguments directly.
settings.DATABASE_URL = settings.DATABASE_URL.replace("sslmode=", "ssl=")
settings.DATABASE_URL = settings.DATABASE_URL.replace("&channel_binding=require", "")
settings.DATABASE_URL = settings.DATABASE_URL.replace("?channel_binding=require", "")




