import os
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

security = HTTPBearer(auto_error=False)

# In production, these should be securely injected via environment variables
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "your-super-secret-jwt-token-with-at-least-32-characters-long")
ALGORITHM = "HS256"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # 1. If valid credentials provided, decode and verify JWT
    if credentials:
        token = credentials.credentials
        try:
            payload = jwt.decode(
                token, 
                SUPABASE_JWT_SECRET, 
                algorithms=[ALGORITHM], 
                options={"verify_aud": False}
            )
            user_id: str = payload.get("sub")
            if user_id:
                return user_id
        except JWTError:
            # If token was supplied but invalid/expired, fall through to demo check or raise
            pass

    # 2. In local development / demo mode, allow smooth access with demo user
    if os.getenv("DEMO_MODE", "true").lower() == "true":
        return "usr_demo"
        
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
