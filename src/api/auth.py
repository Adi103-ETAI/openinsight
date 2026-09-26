"""
Shared Supabase JWT auth — one identity for website, UI, and backend.

Backend is stateless: it verifies the JWT signature (RS256 via Supabase
JWKS, HS256 legacy-secret fallback) and returns the `sub` claim as user_id.
No session rows. Redis holds cache/rate-limits only.

If SUPABASE_URL and SUPABASE_JWT_SECRET are both empty, auth is disabled
and routes fall back to the X-User-ID header (local dev behavior).
"""
from __future__ import annotations

import json
import time
import urllib.request
from functools import lru_cache
from typing import Any

import jwt
from fastapi import HTTPException, Request
from loguru import logger

from src.config.settings import get_settings

_JWKS_CACHE: dict[str, Any] = {"keys": {}, "fetched_at": 0.0}
_JWKS_TTL = 600.0


def auth_configured() -> bool:
    s = get_settings()
    return bool(s.supabase_url or s.supabase_jwt_secret)


def _fetch_jwks(supabase_url: str) -> dict[str, Any]:
    now = time.time()
    if _JWKS_CACHE["keys"] and now - _JWKS_CACHE["fetched_at"] < _JWKS_TTL:
        return _JWKS_CACHE["keys"]
    url = supabase_url.rstrip("/") + "/auth/v1/.well-known/jwks.json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 (https URL)
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning(f"JWKS fetch failed ({url}): {exc}")
        return _JWKS_CACHE["keys"]
    keys = {k.get("kid"): k for k in data.get("keys", []) if k.get("kid")}
    _JWKS_CACHE.update({"keys": keys, "fetched_at": now})
    return keys


def verify_supabase_jwt(token: str) -> dict[str, Any]:
    """Verify a Supabase access token. Returns claims. Raises ValueError."""
    s = get_settings()
    if not token:
        raise ValueError("empty token")
    # 1. RS256 via JWKS (current Supabase default)
    if s.supabase_url:
        keys = _fetch_jwks(s.supabase_url)
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        jwk = keys.get(kid) if kid else None
        if jwk is not None:
            key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
            return jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                audience=s.supabase_jwt_audience,
                options={"require": ["exp", "sub"]},
            )
    # 2. HS256 legacy secret fallback
    if s.supabase_jwt_secret:
        return jwt.decode(
            token,
            s.supabase_jwt_secret,
            algorithms=["HS256"],
            audience=s.supabase_jwt_audience,
            options={"require": ["exp", "sub"]},
        )
    raise ValueError("no key material: set SUPABASE_URL or SUPABASE_JWT_SECRET")


def _bearer_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return None


def get_user_id(request: Request) -> str:
    """Verified Supabase user.id, or dev fallback. Raises 401 when configured but invalid."""
    if not auth_configured():
        return request.headers.get("X-User-ID", "default_user")
    token = _bearer_token(request)
    if token is None:
        # Allow service-to-service header override only when explicitly set?
        # No — require auth when configured.
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        claims = verify_supabase_jwt(token)
    except Exception as exc:
        logger.warning(f"JWT verification failed: {exc}")
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc
    sub = str(claims.get("sub", ""))
    if not sub:
        raise HTTPException(status_code=401, detail="Invalid token claims")
    return sub


def get_optional_user_id(request: Request) -> str | None:
    """Verified user or None (for public endpoints like /search). Never raises."""
    if not auth_configured():
        return request.headers.get("X-User-ID")
    token = _bearer_token(request)
    if token is None:
        return None
    try:
        return str(verify_supabase_jwt(token).get("sub"))
    except Exception:
        return None


@lru_cache(maxsize=1)
def _noop() -> None:
    return None
