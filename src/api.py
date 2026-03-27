"""HTTP API server for the rally-scanner service.

Exposes position data and a health endpoint.  Started once per process by
app.py before the Discord bot connects.
"""

import json as _json
import logging
import os

from aiohttp import web

from trading.positions import get_merged_positions

logger = logging.getLogger(__name__)


async def _handle_positions(request: web.Request) -> web.Response:
    """Return merged Alpaca + DB positions as JSON."""
    api_key = os.environ.get("RALLY_API_KEY")
    if api_key:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {api_key}":
            return web.json_response({"error": "unauthorized"}, status=401)
    state = await get_merged_positions()
    return web.Response(
        text=_json.dumps(state), content_type="application/json",
    )


async def _handle_health(request: web.Request) -> web.Response:
    """Simple health check."""
    return web.json_response({"status": "ok"})


async def start_api_server() -> None:
    """Start the aiohttp API server."""
    app = web.Application()
    app.router.add_get("/api/positions", _handle_positions)
    app.router.add_get("/health", _handle_health)
    port = int(os.environ.get("PORT", 8080))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info("API server listening on port %d", port)
