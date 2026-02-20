"""Tests for the bot's HTTP API endpoints."""

from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


async def _handle_positions(request: web.Request):
    """Mirror of discord_bot._handle_positions for testing."""
    import json
    import os

    api_key = os.environ.get("RALLY_API_KEY")
    if api_key:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {api_key}":
            return web.json_response({"error": "unauthorized"}, status=401)
    get_merged = request.app["get_merged_positions"]
    state = await get_merged()
    return web.Response(text=json.dumps(state), content_type="application/json")


async def _handle_health(request: web.Request):
    return web.json_response({"status": "ok"})


def _make_app(get_merged_positions=None):
    app = web.Application()
    app.router.add_get("/api/positions", _handle_positions)
    app.router.add_get("/health", _handle_health)
    app["get_merged_positions"] = get_merged_positions or AsyncMock(
        return_value={"positions": [], "closed_today": []}
    )
    return app


@pytest.mark.asyncio
async def test_health_endpoint():
    app = _make_app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_positions_endpoint_no_auth(monkeypatch):
    """Without RALLY_API_KEY set, endpoint is open."""
    monkeypatch.delenv("RALLY_API_KEY", raising=False)

    mock_fn = AsyncMock(return_value={
        "positions": [{"ticker": "AAPL", "qty": 10}],
        "closed_today": [],
        "last_updated": "2024-01-01T00:00:00",
    })
    app = _make_app(mock_fn)
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/positions")
        assert resp.status == 200
        data = await resp.json()
        assert len(data["positions"]) == 1
        assert data["positions"][0]["ticker"] == "AAPL"


@pytest.mark.asyncio
async def test_positions_endpoint_valid_auth(monkeypatch):
    """With RALLY_API_KEY set, valid bearer token succeeds."""
    monkeypatch.setenv("RALLY_API_KEY", "test-secret")

    app = _make_app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.get(
            "/api/positions",
            headers={"Authorization": "Bearer test-secret"},
        )
        assert resp.status == 200


@pytest.mark.asyncio
async def test_positions_endpoint_invalid_auth(monkeypatch):
    """With RALLY_API_KEY set, wrong bearer token returns 401."""
    monkeypatch.setenv("RALLY_API_KEY", "test-secret")

    app = _make_app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.get(
            "/api/positions",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status == 401
        data = await resp.json()
        assert data["error"] == "unauthorized"


@pytest.mark.asyncio
async def test_positions_endpoint_missing_auth(monkeypatch):
    """With RALLY_API_KEY set, missing Authorization header returns 401."""
    monkeypatch.setenv("RALLY_API_KEY", "test-secret")

    app = _make_app()
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/api/positions")
        assert resp.status == 401
