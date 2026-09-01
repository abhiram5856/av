import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from backend.core_logging.logger import api_logger

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware that assigns a unique X-Request-ID to every request.
    This correlates log messages across various services and requests.
    """
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Attach the request ID to the request state
        request.state.request_id = request_id
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Inject request ID into response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"
        
        # Structured log entry
        api_logger.info(
            f"CorrelationID: {request_id} | Path: {request.url.path} | "
            f"Method: {request.method} | Status: {response.status_code} | "
            f"Latency: {process_time:.4f}s"
        )
        
        return response
