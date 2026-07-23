from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from backend.config.settings import settings
from backend.api import chat, history, health, config, diagnose

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Zenith AgriBot - Enterprise AI Chatbot Architecture"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers (placeholders)
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(history.router, prefix="/api/history", tags=["History"])
app.include_router(health.router, prefix="/api/health", tags=["Health"])
app.include_router(config.router, prefix="/api/config", tags=["Config"])
app.include_router(diagnose.router, prefix="/api/diagnose", tags=["Diagnosis"])
