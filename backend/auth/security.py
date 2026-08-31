"""
AgriVision AI — Authentication & JWT Security
=============================================

Production behaviour:
  - Missing JWT secret → application startup failure (startup_check() enforced in main.py)
  - Invalid / expired / missing token → HTTP 401

Development behaviour:
  - Set environment variable NOVA_DEV_MODE=true to enable mock authentication
  - Without NOVA_DEV_MODE=true, even in local runs, authentication is enforced

NEVER bypass authentication silently in any environment.
"""

import os
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

security = HTTPBearer(auto_error=False)

SUPABASE_JWT_SECRET: str = os.getenv("SUPABASE_JWT_SECRET", "")
ALGORITHM: str           = "HS256"
_DEV_MODE: bool          = os.getenv("NOVA_DEV_MODE", "false").lower() == "true"

# ─────────────────────────────────────────────────────────────────────────────
# Startup validation (RF12)
# ─────────────────────────────────────────────────────────────────────────────

def startup_check() -> None:
    """
    Called from main.py on startup.
    Fails fast if production credentials are missing.
    """
    if not _DEV_MODE and not SUPABASE_JWT_SECRET:
        raise RuntimeError(
            "SUPABASE_JWT_SECRET environment variable is not set and NOVA_DEV_MODE is not enabled. "
            "The application cannot start without authentication credentials in production mode. "
            "Set SUPABASE_JWT_SECRET or set NOVA_DEV_MODE=true for local development."
        )
    if _DEV_MODE:
        import logging
        logging.getLogger("nova.auth").warning(
            "[DEV MODE ACTIVE] Mock authentication is enabled (NOVA_DEV_MODE=true). "
            "Do NOT run with this setting in production."
        )


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Validates the Supabase JWT token from the Authorization header.

    Returns the authenticated user_id (sub claim).

    Raises HTTP 401 for:
      - Missing token (unless NOVA_DEV_MODE=true)
      - Invalid token format
      - Expired token
      - Missing 'sub' claim
      - Signature verification failure
    """
    # ── Development mock bypass ────────────────────────────────────────────
    # Only active when NOVA_DEV_MODE=true is explicitly set.
    if _DEV_MODE and credentials is None:
        return "local-dev-mock-user-123"

    # ── Production: require token ──────────────────────────────────────────
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        if _DEV_MODE and not SUPABASE_JWT_SECRET:
            # Dev mode with no secret: read claims without verifying signature
            # This is explicitly opt-in and logs a warning on startup.
            payload = jwt.get_unverified_claims(token)
        else:
            payload = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=[ALGORITHM],
                options={"verify_aud": False},
            )

        user_id: str = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token is missing the 'sub' (user ID) claim.",
            )
        return user_id

    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(exc)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
