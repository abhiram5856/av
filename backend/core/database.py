from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from backend.config.settings import settings

# Create the async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG_MODE,
    future=True,
    # connection pool settings to handle production load
    pool_size=5,
    max_overflow=10
)

# Create an async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

# Declarative base for ORM models
Base = declarative_base()

# FastAPI Dependency for Database Sessions
async def get_db():
    """
    Dependency function that yields a database session.
    Automatically handles session cleanup.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
