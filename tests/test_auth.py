"""Auth tests — Supabase JWT verifier with dev fallback (no network)."""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request


def _req(headers: dict | None = None) -> Request:
    scope = {"type": "http", "headers": [
        (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
    ]}
    return Request(scope)


@pytest.mark.unit
def test_dev_fallback_no_config(monkeypatch):
    from src.api import auth as auth_mod
    monkeypatch.setattr(auth_mod, "auth_configured", lambda: False)
    assert auth_mod.get_user_id(_req({"X-User-ID": "doc1"})) == "doc1"
    assert auth_mod.get_user_id(_req()) == "default_user"


@pytest.mark.unit
def test_configured_requires_bearer(monkeypatch):
    from src.api import auth as auth_mod
    monkeypatch.setattr(auth_mod, "auth_configured", lambda: True)
    with pytest.raises(HTTPException) as exc:
        auth_mod.get_user_id(_req())
    assert exc.value.status_code == 401


@pytest.mark.unit
def test_configured_rejects_bad_token(monkeypatch):
    from src.api import auth as auth_mod
    monkeypatch.setattr(auth_mod, "auth_configured", lambda: True)
    with pytest.raises(HTTPException) as exc:
        auth_mod.get_user_id(_req({"Authorization": "Bearer garbage"}))
    assert exc.value.status_code == 401
    assert auth_mod.get_optional_user_id(_req({"Authorization": "Bearer garbage"})) is None
