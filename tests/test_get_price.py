"""Tests for the get_price tool wrapper — unique wrapper behavior only.

Core fetch_quotes tests live in test_data.py.
"""

import sys
from unittest.mock import MagicMock, patch

# claude_agent imports anthropic at module level; stub it before importing.
sys.modules.setdefault("anthropic", MagicMock())

from integrations.discord.agent import _get_price  # noqa: E402, I001


@patch("integrations.discord.agent.fetch_quotes")
def test_get_price_caps_at_10(mock_fetch):
    mock_fetch.return_value = {}
    _get_price([f"T{i}" for i in range(20)])
    assert len(mock_fetch.call_args[0][0]) == 10


@patch("integrations.discord.agent.fetch_quotes")
def test_get_price_network_error(mock_fetch):
    mock_fetch.side_effect = ConnectionError("network down")
    result = _get_price(["HD"])
    assert "error" in result
