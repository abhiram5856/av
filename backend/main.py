import sys
import importlib

# Standard library logging override to prevent backend/logging package shadowing standard logging
original_path = sys.path.copy()
sys.path = [p for p in sys.path if not p.endswith('backend') and not p.endswith('backend\\')]
stdlib_logging = importlib.import_module('logging')
sys.path = original_path
sys.modules['logging'] = stdlib_logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from backend.config.settings import settings
from backend.api import chat, history, health, config, diagnose, report, iot, predict_risk
from backend.core.middleware import CorrelationIdMiddleware
from backend.core.exceptions import ZenithAgriBotError
from backend.logging.logger import api_logger


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Zenith AgriBot - Enterprise AI Chatbot Architecture"
)
app.state.limiter = limiter

# Rate limiter exception handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Custom Global Exception Handler for ZenithAgriBotError hierarchy
@app.exception_handler(ZenithAgriBotError)
async def custom_error_handler(request: Request, exc: ZenithAgriBotError):
    request_id = getattr(request.state, "request_id", "N/A")
    api_logger.error(f"Request {request_id} encountered custom domain error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={
            "status": "error",
            "request_id": request_id,
            "error_type": exc.__class__.__name__,
            "message": str(exc)
        }
    )

# Fallback General Exception Handler
@app.exception_handler(Exception)
async def fallback_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "N/A")
    api_logger.error(f"Request {request_id} encountered unhandled system error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "request_id": request_id,
            "error_type": "InternalServerError",
            "message": "An unexpected system error occurred. Please contact the administrator."
        }
    )

# CORS middleware config
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://agrivision-ai.vercel.app",  # Example Vercel URL
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom correlation ID middleware
app.add_middleware(CorrelationIdMiddleware)

# Include routers
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(history.router, prefix="/api/v1/history", tags=["History"])
app.include_router(health.router, prefix="", tags=["Health"])
app.include_router(config.router, prefix="/api/v1/config", tags=["Config"])
app.include_router(diagnose.router, prefix="/api/v1/diagnose", tags=["Diagnosis"])
app.include_router(report.router, prefix="/api/v1", tags=["Report"])
app.include_router(iot.router, prefix="/api/v1/iot", tags=["IoT"])
app.include_router(predict_risk.router, prefix="/api/v1/predict-risk", tags=["Predictive Risk"])

