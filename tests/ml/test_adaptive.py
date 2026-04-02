"""Tests for adaptive scan frequency and watchlist scanning."""

from rally_ml.config import PARAMS


def test_scan_watchlist_returns_results(monkeypatch):
    """scan_watchlist returns results for valid tickers."""
    from pipeline import scanner

    # Mock load_manifest to return known tickers
    monkeypatch.setattr(scanner, "load_manifest", lambda: {
        "AAPL": {}, "MSFT": {},
    })

    # Mock apply_config
    cons = scanner.CONFIGS_BY_NAME["conservative"]
    monkeypatch.setattr(scanner, "resolve_config", lambda _: cons)

    # Mock data fetching
    monkeypatch.setattr(scanner, "fetch_vix_safe", lambda **kw: None)
    monkeypatch.setattr(scanner, "fetch_daily_batch", lambda t, **kw: {})

    # Mock _scan_one to return test data
    def mock_scan_one(args):
        ticker = args[0]
        return {"ticker": ticker, "status": "ok", "signal": False, "p_rally": 0.40}

    monkeypatch.setattr(scanner, "_scan_one", mock_scan_one)

    # Mock ProcessPoolExecutor to run synchronously
    from concurrent.futures import Future
    from unittest.mock import MagicMock, patch

    def mock_submit(fn, item):
        f = Future()
        f.set_result(fn(item))
        return f

    with patch("pipeline.scanner.ProcessPoolExecutor") as MockPool:
        mock_pool = MagicMock()
        mock_pool.__enter__ = lambda s: s
        mock_pool.__exit__ = lambda s, *a: None
        mock_pool.submit = mock_submit
        MockPool.return_value = mock_pool

        results = scanner.scan_watchlist(["AAPL", "MSFT"])

    assert len(results) == 2
    assert all(r["status"] == "ok" for r in results)


def test_scan_watchlist_empty_manifest(monkeypatch):
    """scan_watchlist returns empty for no manifest."""
    from pipeline import scanner
    monkeypatch.setattr(scanner, "load_manifest", lambda: {})
    results = scanner.scan_watchlist(["AAPL"])
    assert results == []


def test_scan_watchlist_unknown_tickers(monkeypatch):
    """scan_watchlist skips tickers not in manifest."""
    from pipeline import scanner
    monkeypatch.setattr(scanner, "load_manifest", lambda: {"AAPL": {}})
    cons = scanner.CONFIGS_BY_NAME["conservative"]
    monkeypatch.setattr(scanner, "resolve_config", lambda _: cons)
    monkeypatch.setattr(scanner, "fetch_vix_safe", lambda **kw: None)
    monkeypatch.setattr(scanner, "fetch_daily_batch", lambda t, **kw: {})

    def mock_scan_one(args):
        return {"ticker": args[0], "status": "ok", "signal": False, "p_rally": 0.40}

    monkeypatch.setattr(scanner, "_scan_one", mock_scan_one)

    from concurrent.futures import Future
    from unittest.mock import MagicMock, patch

    def mock_submit(fn, item):
        f = Future()
        f.set_result(fn(item))
        return f

    with patch("pipeline.scanner.ProcessPoolExecutor") as MockPool:
        mock_pool = MagicMock()
        mock_pool.__enter__ = lambda s: s
        mock_pool.__exit__ = lambda s, *a: None
        mock_pool.submit = mock_submit
        MockPool.return_value = mock_pool

        results = scanner.scan_watchlist(["AAPL", "UNKNOWN"])

    assert len(results) == 1
    assert results[0]["ticker"] == "AAPL"


def test_watchlist_p_rally_threshold():
    """Verify the watchlist threshold config param exists and is sensible."""
    assert 0.0 < PARAMS.watchlist_p_rally_min < PARAMS.p_rally_threshold


def test_adaptive_alert_intervals():
    """Verify fast interval < base interval."""
    assert PARAMS.fast_alert_interval < PARAMS.base_alert_interval


def test_vix_fast_threshold():
    """VIX fast threshold should be a reasonable value."""
    assert 20 <= PARAMS.vix_fast_threshold <= 50
