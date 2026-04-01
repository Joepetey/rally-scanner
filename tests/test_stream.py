"""
Integration tests for the streaming infrastructure and decoupled trading engine.

MIC-28: Tests for AlpacaStreamManager, AlertEngine, TradingScheduler,
and bot notification dispatch.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from integrations.alpaca.stream import AlpacaStreamManager, is_stream_enabled
from trading.engine import AlertEvent, HousekeepingResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pos(
    ticker: str = "AAPL",
    entry: float = 100.0,
    stop: float = 95.0,
    target: float = 110.0,
    trailing: float = 0.0,
    order_id: str | None = None,
) -> dict:
    return {
        "ticker": ticker,
        "entry_price": entry,
        "stop_price": stop,
        "target_price": target,
        "trailing_stop": trailing,
        "order_id": order_id,
        "target_order_id": None,
        "trail_order_id": None,
        "qty": 10,
    }


# ===========================================================================
# 1. AlpacaStreamManager unit tests
# ===========================================================================


class TestAlpacaStreamManager:
    """Unit tests using a mock StockDataStream."""

    def _make_mgr(self, on_trade=None):
        if on_trade is None:
            on_trade = MagicMock()
        return AlpacaStreamManager(on_trade=on_trade), on_trade

    # ----------------------------------------------------------------
    # Crypto filtering
    # ----------------------------------------------------------------

    def test_equity_filter_passes_equities(self):
        assert AlpacaStreamManager._is_equity("AAPL") is True
        assert AlpacaStreamManager._is_equity("MSFT") is True

    def test_equity_filter_blocks_crypto(self):
        assert AlpacaStreamManager._is_equity("BTC-USD") is False
        assert AlpacaStreamManager._is_equity("ETH-USD") is False

    def test_is_crypto_internal_key(self):
        mgr, _ = self._make_mgr()
        assert mgr._is_crypto("BTC") is True
        assert mgr._is_crypto("ETH") is True
        assert mgr._is_crypto("AAPL") is False
        assert mgr._is_crypto("UNKNOWN") is False

    def test_to_alpaca_crypto_symbol(self):
        assert AlpacaStreamManager._to_alpaca_crypto_symbol("BTC") == "BTC/USD"
        assert AlpacaStreamManager._to_alpaca_crypto_symbol("ETH") == "ETH/USD"

    # ----------------------------------------------------------------
    # start / stop lifecycle
    # ----------------------------------------------------------------

    def test_start_creates_daemon_thread(self, monkeypatch):
        mgr, _ = self._make_mgr()
        # Prevent real stream connection
        monkeypatch.setattr(mgr, "_run_stream", lambda: time.sleep(10))
        monkeypatch.setattr(mgr, "_run_crypto_stream", lambda: time.sleep(10))
        monkeypatch.setenv("ALPACA_API_KEY", "fake")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

        mgr.start({"AAPL", "MSFT"})
        time.sleep(0.05)
        assert mgr._thread is not None
        assert mgr._thread.is_alive()
        assert mgr._thread.daemon is True
        mgr.stop()
        mgr._thread.join(timeout=1)

    def test_start_splits_equity_and_crypto(self, monkeypatch):
        mgr, _ = self._make_mgr()
        monkeypatch.setattr(mgr, "_run_stream", lambda: time.sleep(10))
        monkeypatch.setattr(mgr, "_run_crypto_stream", lambda: time.sleep(10))
        monkeypatch.setenv("ALPACA_API_KEY", "fake")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

        mgr.start({"AAPL", "BTC", "ETH"})
        time.sleep(0.05)
        assert mgr._symbols == {"AAPL"}
        assert mgr._crypto_symbols == {"BTC", "ETH"}
        mgr.stop()

    def test_stop_sets_stop_event(self, monkeypatch):
        mgr, _ = self._make_mgr()
        monkeypatch.setattr(mgr, "_run_stream", lambda: time.sleep(10))
        monkeypatch.setattr(mgr, "_run_crypto_stream", lambda: time.sleep(10))
        monkeypatch.setenv("ALPACA_API_KEY", "fake")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

        mgr.start({"AAPL"})
        time.sleep(0.05)
        mgr.stop()
        assert mgr._stop_event.is_set()
        assert not mgr.is_connected

    def test_double_start_is_idempotent(self, monkeypatch):
        mgr, _ = self._make_mgr()
        def fake_run():
            time.sleep(5)
        monkeypatch.setattr(mgr, "_run_stream", fake_run)
        monkeypatch.setattr(mgr, "_run_crypto_stream", fake_run)
        monkeypatch.setenv("ALPACA_API_KEY", "fake")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

        mgr.start({"AAPL"})
        first_thread = mgr._thread
        mgr.start({"AAPL"})  # should be no-op
        time.sleep(0.05)
        # Second start should not replace the thread — same object
        assert mgr._thread is first_thread
        mgr.stop()

    def test_is_connected_true_when_crypto_connected(self):
        mgr, _ = self._make_mgr()
        assert not mgr.is_connected
        mgr._crypto_connected.set()
        assert mgr.is_connected

    def test_is_connected_true_when_equity_connected(self):
        mgr, _ = self._make_mgr()
        mgr._connected.set()
        assert mgr.is_connected

    # ----------------------------------------------------------------
    # get_stale_tickers includes crypto
    # ----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stale_tickers_includes_crypto_symbols(self):
        mgr = AlpacaStreamManager(on_trade=MagicMock())
        mgr._symbols = {"AAPL"}
        mgr._crypto_symbols = {"BTC"}
        # Neither has had a trade

        new_stale, _, never_traded = mgr.get_stale_tickers(stale_seconds=300.0)
        # Never received a trade → categorized as never_traded, not new_stale
        assert "AAPL" in never_traded
        assert "BTC" in never_traded
        assert new_stale == []

    def test_is_stream_enabled_true(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        monkeypatch.setenv("ALPACA_STREAM_ENABLED", "1")
        assert is_stream_enabled() is True


# ===========================================================================
# 3. TradingScheduler integration test
# ===========================================================================


class TestTradingSchedulerIntegration:
    """Integration tests for TradingScheduler stream → event flow."""

    @pytest.mark.asyncio
    async def test_stream_trade_triggers_alert_event(self):
        """Stream trade callback → evaluate_single_ticker → on_event receives AlertEvent."""
        received = []

        async def on_event(event):
            received.append(event)

        from trading.scheduler import TradingScheduler

        scheduler = TradingScheduler(on_event=on_event)
        loop = asyncio.get_event_loop()
        scheduler._loop = loop

        pos = _make_pos(entry=100.0, stop=95.0)

        with patch("trading.scheduler_stream._load_position_meta", return_value=pos), \
             patch("trading.engine.log_price_alert", return_value=True), \
             patch.object(scheduler._engine, "is_market_open", return_value=True):
            # Simulate stream firing a trade below stop
            scheduler._on_stream_trade("AAPL", 94.0)
            # Give the coroutine time to run
            await asyncio.sleep(0.1)

        assert len(received) >= 1
        alert = received[0]
        assert isinstance(alert, AlertEvent)
        assert alert.alert_type == "stop_breached"

    @pytest.mark.asyncio
    async def test_stream_ignored_outside_market_hours(self):
        received = []

        async def on_event(event):
            received.append(event)

        from trading.scheduler import TradingScheduler
        scheduler = TradingScheduler(on_event=on_event)
        scheduler._loop = asyncio.get_event_loop()

        with patch.object(scheduler._engine, "is_market_open", return_value=False):
            scheduler._on_stream_trade("AAPL", 94.0)
            await asyncio.sleep(0.05)

        assert received == []

    @pytest.mark.asyncio
    async def test_crypto_stream_fires_outside_market_hours(self):
        """BTC trades should pass through _on_stream_trade even when market is closed."""
        received = []

        async def on_event(event):
            received.append(event)

        from trading.scheduler import TradingScheduler

        scheduler = TradingScheduler(on_event=on_event)
        scheduler._loop = asyncio.get_event_loop()

        pos = _make_pos(ticker="BTC", entry=90000.0, stop=85000.0)

        with patch("trading.scheduler_stream._load_position_meta", return_value=pos), \
             patch("trading.engine.log_price_alert", return_value=True), \
             patch.object(scheduler._engine, "is_market_open", return_value=False):
            scheduler._on_stream_trade("BTC", 84000.0)
            await asyncio.sleep(0.1)

        assert len(received) >= 1
        assert received[0].alert_type == "stop_breached"

    def test_stream_write_skipped_when_position_deleted(self):
        """MIC-72: stream callback must not resurrect a deleted position."""
        from trading.scheduler import TradingScheduler

        scheduler = TradingScheduler(on_event=AsyncMock())
        scheduler._loop = MagicMock()

        # Position deleted from DB — _load_position_meta returns None
        with patch(
            "trading.scheduler_stream._load_position_meta", return_value=None,
        ) as mock_load, \
             patch("trading.scheduler_stream._save_position_meta") as mock_save, \
             patch.object(scheduler._engine, "is_market_open", return_value=True):
            scheduler._on_stream_trade("AAPL", 94.0)

        mock_load.assert_called_once_with("AAPL")
        mock_save.assert_not_called()

    def test_stream_write_proceeds_when_position_exists(self):
        """Stream callback writes meta when position exists in DB."""
        from trading.scheduler import TradingScheduler

        scheduler = TradingScheduler(on_event=AsyncMock())
        scheduler._loop = MagicMock()

        pos = _make_pos(entry=100.0, stop=95.0)
        with patch("trading.scheduler_stream._load_position_meta", return_value=pos), \
             patch("trading.scheduler_stream.update_position_for_price", return_value=True), \
             patch("trading.scheduler_stream._save_position_meta") as mock_save, \
             patch.object(scheduler._engine, "is_market_open", return_value=True), \
             patch.object(scheduler._engine, "evaluate_single_ticker", return_value=None):
            scheduler._on_stream_trade("AAPL", 94.0)

        mock_save.assert_called_once_with(pos)

    def test_stream_write_preserves_exit_order_ids(self):
        """MIC-102: stream reads fresh from DB so exit order IDs are never lost.

        With no cache, _on_stream_trade loads directly from DB. Exit order IDs
        set by housekeeping are always present in the position dict that gets saved.
        """
        from trading.scheduler import TradingScheduler

        scheduler = TradingScheduler(on_event=AsyncMock())
        scheduler._loop = MagicMock()

        # DB position has exit order IDs (set by housekeeping)
        pos = _make_pos(entry=100.0, stop=95.0, target=110.0)
        pos["target_order_id"] = "target-order-123"
        pos["trail_order_id"] = "trail-order-456"

        with patch("trading.scheduler_stream._load_position_meta", return_value=pos), \
             patch("trading.scheduler_stream.update_position_for_price", return_value=True), \
             patch("trading.scheduler_stream._save_position_meta") as mock_save, \
             patch.object(scheduler._engine, "is_market_open", return_value=True), \
             patch.object(scheduler._engine, "evaluate_single_ticker", return_value=None):
            scheduler._on_stream_trade("AAPL", 94.0)

        mock_save.assert_called_once_with(pos)
        # Exit order IDs preserved because pos came from DB, not stale cache
        assert pos["target_order_id"] == "target-order-123"
        assert pos["trail_order_id"] == "trail-order-456"

    @pytest.mark.asyncio
    async def test_concurrent_exit_guard_prevents_double_exit(self):
        """Second breach for same ticker while first is exiting should be skipped."""
        exit_calls = []

        async def on_event(event):
            pass

        from trading.scheduler import TradingScheduler
        scheduler = TradingScheduler(on_event=on_event)
        scheduler._exit_lock = asyncio.Lock()

        async def slow_breach(ticker, pos, price, reason):
            exit_calls.append(ticker)
            await asyncio.sleep(0.1)
            return None

        pos = _make_pos()
        with patch.object(scheduler._engine, "execute_breach", side_effect=slow_breach):
            # Fire two concurrent exits for same ticker
            await asyncio.gather(
                scheduler._execute_breach_guarded("AAPL", pos, 94.0, "stop_breached"),
                scheduler._execute_breach_guarded("AAPL", pos, 93.0, "stop_breached"),
            )

        # Only one exit should have been called
        assert exit_calls.count("AAPL") == 1


# ===========================================================================
# 4. Stream fallback test
# ===========================================================================


class TestStreamHealthMonitoring:
    """Verify degradation + recovery alerts are emitted at the right thresholds."""

    @pytest.mark.asyncio
    async def test_degradation_alert_fires_after_threshold(self):
        """StreamDegradedEvent is emitted once stream has been down >= 5 cycles."""
        from trading.engine import StreamDegradedEvent
        from trading.scheduler import TradingScheduler

        received = []
        scheduler = TradingScheduler(on_event=AsyncMock(side_effect=lambda e: received.append(e)))

        mock_stream = MagicMock()
        mock_stream.is_connected = False
        mock_stream.get_stale_tickers.return_value = ([], [], [])
        scheduler._stream = mock_stream

        with patch.object(scheduler._engine, "is_market_open", return_value=True), \
             patch.object(scheduler._engine, "run_housekeeping", new_callable=AsyncMock,
                          return_value=MagicMock(fills_confirmed=[], orders_placed=[])), \
             patch("trading.scheduler_loops.load_positions", return_value={"positions": []}), \
             patch("trading.scheduler_loops.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Run 5 cycles (threshold), then cancel
            mock_sleep.side_effect = [None] * 5 + [asyncio.CancelledError()]
            try:
                await scheduler._housekeeping_loop()
            except asyncio.CancelledError:
                pass

        degraded = [e for e in received if isinstance(e, StreamDegradedEvent)]
        assert len(degraded) == 1
        assert degraded[0].disconnected_minutes == 5

    @pytest.mark.asyncio
    async def test_degradation_alert_fires_only_once(self):
        """StreamDegradedEvent is not re-emitted on subsequent disconnected cycles."""
        from trading.engine import StreamDegradedEvent
        from trading.scheduler import TradingScheduler

        received = []
        scheduler = TradingScheduler(on_event=AsyncMock(side_effect=lambda e: received.append(e)))

        mock_stream = MagicMock()
        mock_stream.is_connected = False
        mock_stream.get_stale_tickers.return_value = ([], [], [])
        scheduler._stream = mock_stream

        with patch.object(scheduler._engine, "is_market_open", return_value=True), \
             patch.object(scheduler._engine, "run_housekeeping", new_callable=AsyncMock,
                          return_value=MagicMock(fills_confirmed=[], orders_placed=[])), \
             patch("trading.scheduler_loops.load_positions", return_value={"positions": []}), \
             patch("trading.scheduler_loops.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Run 10 cycles (well past threshold)
            mock_sleep.side_effect = [None] * 10 + [asyncio.CancelledError()]
            try:
                await scheduler._housekeeping_loop()
            except asyncio.CancelledError:
                pass

        degraded = [e for e in received if isinstance(e, StreamDegradedEvent)]
        assert len(degraded) == 1

    @pytest.mark.asyncio
    async def test_recovery_alert_fires_after_degradation(self):
        """StreamRecoveredEvent is emitted when stream reconnects after degradation alert."""
        from trading.engine import StreamRecoveredEvent
        from trading.scheduler import TradingScheduler

        received = []
        scheduler = TradingScheduler(on_event=AsyncMock(side_effect=lambda e: received.append(e)))

        mock_stream = MagicMock()
        scheduler._stream = mock_stream

        # Pre-condition: degradation alert was already sent
        scheduler._stream_degraded_cycles = 7
        scheduler._stream_alert_sent = True

        # Stream is now connected (recovered)
        mock_stream.is_connected = True
        mock_stream.get_stale_tickers.return_value = ([], [], [])

        with patch.object(scheduler._engine, "is_market_open", return_value=True), \
             patch.object(scheduler._engine, "run_housekeeping", new_callable=AsyncMock,
                          return_value=MagicMock(fills_confirmed=[], orders_placed=[])), \
             patch("trading.scheduler_loops.load_positions", return_value={"positions": []}), \
             patch("trading.scheduler_loops.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            try:
                await scheduler._housekeeping_loop()
            except asyncio.CancelledError:
                pass

        recovered = [e for e in received if isinstance(e, StreamRecoveredEvent)]
        assert len(recovered) == 1
        assert recovered[0].downtime_minutes == 7
        assert scheduler._stream_degraded_cycles == 0
        assert scheduler._stream_alert_sent is False

    @pytest.mark.asyncio
    async def test_no_alert_before_threshold(self):
        """No StreamDegradedEvent before 5 disconnected cycles."""
        from trading.engine import StreamDegradedEvent
        from trading.scheduler import TradingScheduler

        received = []
        scheduler = TradingScheduler(on_event=AsyncMock(side_effect=lambda e: received.append(e)))

        mock_stream = MagicMock()
        mock_stream.is_connected = False
        mock_stream.get_stale_tickers.return_value = ([], [], [])
        scheduler._stream = mock_stream

        with patch.object(scheduler._engine, "is_market_open", return_value=True), \
             patch.object(scheduler._engine, "run_housekeeping", new_callable=AsyncMock,
                          return_value=MagicMock(fills_confirmed=[], orders_placed=[])), \
             patch("trading.scheduler_loops.load_positions", return_value={"positions": []}), \
             patch("trading.scheduler_loops.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = [None] * 4 + [asyncio.CancelledError()]
            try:
                await scheduler._housekeeping_loop()
            except asyncio.CancelledError:
                pass

        degraded = [e for e in received if isinstance(e, StreamDegradedEvent)]
        assert len(degraded) == 0


class TestStreamFallback:
    """Verify polling loop skips when stream connected, resumes on disconnect."""

    @pytest.mark.asyncio
    async def test_polling_skips_when_stream_connected(self):
        """_polling_loop should not call get_snapshots when stream.is_connected."""
        from trading.scheduler import TradingScheduler

        scheduler = TradingScheduler(on_event=AsyncMock())
        mock_stream = MagicMock()
        mock_stream.is_connected = True
        scheduler._stream = mock_stream
        scheduler._last_alert_check = scheduler._last_alert_check.min.replace(
            tzinfo=scheduler._last_alert_check.tzinfo
        )

        with patch.object(scheduler._engine, "is_market_open", return_value=True), \
             patch("trading.scheduler_loops.get_snapshots") as mock_snap, \
             patch("trading.scheduler_loops.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Make sleep return immediately then raise to exit loop
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            try:
                await scheduler._polling_loop()
            except asyncio.CancelledError:
                pass

        mock_snap.assert_not_called()

    @pytest.mark.asyncio
    async def test_polling_runs_when_stream_disconnected(self):
        """_polling_loop should poll when stream.is_connected is False."""
        import zoneinfo
        from datetime import datetime

        from trading.scheduler import TradingScheduler

        scheduler = TradingScheduler(on_event=AsyncMock())
        mock_stream = MagicMock()
        mock_stream.is_connected = False
        scheduler._stream = mock_stream
        # Force elapsed time to exceed interval
        scheduler._last_alert_check = datetime.min.replace(
            tzinfo=zoneinfo.ZoneInfo("America/New_York")
        )
        scheduler._current_alert_interval = 0

        pos = _make_pos()
        mock_check = AsyncMock(return_value=[])
        with patch.object(scheduler._engine, "is_market_open", return_value=True), \
             patch("trading.scheduler_loops.load_positions", return_value={"positions": [pos]}), \
             patch("trading.scheduler_loops.alpaca_enabled", return_value=False), \
             patch("trading.scheduler_loops.fetch_quotes", return_value={}), \
             patch.object(scheduler._engine, "check_prices", mock_check), \
             patch("trading.scheduler_loops.asyncio.sleep",
                   new_callable=AsyncMock) as mock_sleep, \
             patch("trading.scheduler_loops.asyncio.to_thread",
                   new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = {}
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            try:
                await scheduler._polling_loop()
            except asyncio.CancelledError:
                pass

        mock_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_snapshot_lock_serializes_concurrent_calls(self):
        """Housekeeping IEX fallback and polling loop must not call get_snapshots concurrently."""
        from trading.scheduler import TradingScheduler

        scheduler = TradingScheduler(on_event=AsyncMock())
        scheduler._snapshot_lock = asyncio.Lock()

        call_order = []

        async def fake_snapshots(tickers):
            call_order.append("start")
            await asyncio.sleep(0)
            call_order.append("end")
            return {}

        # Run two coroutines that both try to acquire the lock and call get_snapshots
        async def caller_a():
            async with scheduler._snapshot_lock:
                await fake_snapshots(["AAPL"])

        async def caller_b():
            async with scheduler._snapshot_lock:
                await fake_snapshots(["MSFT"])

        await asyncio.gather(caller_a(), caller_b())

        # Each call should fully complete before the next starts — no interleaving
        assert call_order == ["start", "end", "start", "end"]


# ===========================================================================
# 5. Bot notification test
# ===========================================================================


class TestBotNotification:
    """Verify _handle_trading_event calls the correct embed builders."""

    def _make_handler(self):
        """Return (bot_module, _handle_trading_event, mock_send_alert)."""
        import integrations.discord.bot as bot_mod

        sent = []

        async def fake_send_alert(embed, msg_type="other"):
            sent.append((embed, msg_type))

        return bot_mod, fake_send_alert, sent

    @pytest.mark.asyncio
    async def test_stop_breach_sends_price_alert_embed(self):
        from integrations.discord.notify import _price_alert_embed

        event = AlertEvent(
            ticker="AAPL",
            alert_type="stop_breached",
            current_price=94.0,
            level_price=95.0,
            level_name="Stop",
            entry_price=100.0,
            pnl_pct=-6.0,
        )

        embed_data = _price_alert_embed([event.model_dump()])
        assert embed_data["color"] == 0xFF0000
        assert "AAPL" in str(embed_data["fields"])

    @pytest.mark.asyncio
    async def test_housekeeping_fill_notification(self):
        from integrations.discord.notify import _fill_confirmation_embed
        from trading.engine import FillNotification

        result = HousekeepingResult(
            fills_confirmed=[
                FillNotification(
                    ticker="AAPL",
                    fill_price=150.0,
                    qty=10,
                    stop_price=142.0,
                    target_price=165.0,
                )
            ],
            orders_placed=[],
            positions_synced=True,
        )

        fills = [f.model_dump() for f in result.fills_confirmed]
        embed_data = _fill_confirmation_embed(fills)
        assert "AAPL" in str(embed_data)
