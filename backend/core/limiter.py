from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request
import jwt

def rate_limit_key_func(request: Request) -> str:
    """
    Extracts the user ID from the Authorization header if present.
    Falls back to the remote IP address if unauthenticated.
    This prevents NAT/Carrier-grade NAT from blocking entire villages of legitimate farmers.
    """
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            # We don't verify signature here, just decode payload to get sub (user_id) for rate limit bucket.
            # Security verification is still handled safely by the actual verify_token dependency.
            payload = jwt.decode(token, options={"verify_signature": False})
            user_id = payload.get("sub")
            if user_id:
                return f"user:{user_id}"
        except Exception:
            pass
            
    # Fallback to IP address
    return f"ip:{get_remote_address(request)}"

limiter = Limiter(key_func=rate_limit_key_func)
