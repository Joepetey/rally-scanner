"""
Integration tests for the streaming infrastructure and decoupled trading engine.

MIC-28: Tests for AlpacaStreamManager, AlertEngine, TradingScheduler,
and bot notification dispatch.
"""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from integrations.alpaca.stream import AlpacaStreamManager, is_stream_enabled
from trading.engine import AlertEngine, AlertEvent, ExitResult, HousekeepingResult


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

    # ----------------------------------------------------------------
    # start / stop lifecycle
    # ----------------------------------------------------------------

    def test_start_creates_daemon_thread(self, monkeypatch):
        mgr, _ = self._make_mgr()
        # Prevent real stream connection
        monkeypatch.setattr(mgr, "_run_stream", lambda: time.sleep(10))
        monkeypatch.setenv("ALPACA_API_KEY", "fake")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

        mgr.start({"AAPL", "MSFT"})
        time.sleep(0.05)
        assert mgr._thread is not None
        assert mgr._thread.is_alive()
        assert mgr._thread.daemon is True
        mgr.stop()
        mgr._thread.join(timeout=1)

    def test_start_filters_crypto(self, monkeypatch):
        mgr, _ = self._make_mgr()
        monkeypatch.setattr(mgr, "_run_stream", lambda: time.sleep(10))
        monkeypatch.setenv("ALPACA_API_KEY", "fake")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

        mgr.start({"AAPL", "BTC-USD", "ETH-USD"})
        time.sleep(0.05)
        assert mgr._symbols == {"AAPL"}
        mgr.stop()

    def test_stop_sets_stop_event(self, monkeypatch):
        mgr, _ = self._make_mgr()
        monkeypatch.setattr(mgr, "_run_stream", lambda: time.sleep(10))
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
        monkeypatch.setenv("ALPACA_API_KEY", "fake")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "fake")

        mgr.start({"AAPL"})
        first_thread = mgr._thread
        mgr.start({"AAPL"})  # should be no-op
        time.sleep(0.05)
        # Second start should not replace the thread — same object
        assert mgr._thread is first_thread
        mgr.stop()

    # ----------------------------------------------------------------
    # Per-ticker throttle
    # ----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_throttle_suppresses_rapid_callbacks(self, monkeypatch):
        fired = []
        mgr, _ = self._make_mgr(on_trade=lambda t, p: fired.append((t, p)))
        mgr._throttle = 1.0  # 1s throttle for test

        trade = MagicMock()
        trade.symbol = "AAPL"
        trade.price = 150.0

        await mgr._handle_trade(trade)
        await mgr._handle_trade(trade)  # should be suppressed
        assert len(fired) == 1

    @pytest.mark.asyncio
    async def test_throttle_fires_after_interval(self, monkeypatch):
        fired = []
        mgr, _ = self._make_mgr(on_trade=lambda t, p: fired.append((t, p)))
        mgr._throttle = 0.01  # 10ms for test

        trade = MagicMock()
        trade.symbol = "AAPL"
        trade.price = 150.0

        await mgr._handle_trade(trade)
        time.sleep(0.02)
        await mgr._handle_trade(trade)  # should fire again
        assert len(fired) == 2

    @pytest.mark.asyncio
    async def test_throttle_is_per_ticker(self):
        fired = []
        mgr, _ = self._make_mgr(on_trade=lambda t, p: fired.append(t))
        mgr._throttle = 1.0

        t1 = MagicMock(); t1.symbol = "AAPL"; t1.price = 150.0
        t2 = MagicMock(); t2.symbol = "MSFT"; t2.price = 400.0

        await mgr._handle_trade(t1)
        await mgr._handle_trade(t2)  # different ticker — should fire
        assert fired == ["AAPL", "MSFT"]

    # ----------------------------------------------------------------
    # update_subscriptions diff logic
    # ----------------------------------------------------------------

    def test_update_subscriptions_adds_new(self, monkeypatch):
        mgr, _ = self._make_mgr()
        mgr._symbols = {"AAPL"}
        mgr._connected.set()

        mock_stream = MagicMock()
        mgr._stream = mock_stream

        mgr.update_subscriptions({"AAPL", "MSFT"})

        mock_stream.subscribe_trades.assert_called_once_with(mgr._handle_trade, "MSFT")
        assert "MSFT" in mgr._symbols

    def test_update_subscriptions_removes_old(self, monkeypatch):
        mgr, _ = self._make_mgr()
        mgr._symbols = {"AAPL", "NVDA"}
        mgr._connected.set()

        mock_stream = MagicMock()
        mgr._stream = mock_stream

        mgr.update_subscriptions({"AAPL"})

        mock_stream.unsubscribe_trades.assert_called_once_with("NVDA")
        assert "NVDA" not in mgr._symbols

    def test_update_subscriptions_no_op_when_same(self):
        mgr, _ = self._make_mgr()
        mgr._symbols = {"AAPL", "MSFT"}
        mgr._connected.set()

        mock_stream = MagicMock()
        mgr._stream = mock_stream

        mgr.update_subscriptions({"AAPL", "MSFT"})
        mock_stream.subscribe_trades.assert_not_called()
        mock_stream.unsubscribe_trades.assert_not_called()

    def test_update_subscriptions_filters_crypto(self):
        mgr, _ = self._make_mgr()
        mgr._symbols = {"AAPL"}
        mgr._connected.set()
        mock_stream = MagicMock()
        mgr._stream = mock_stream

        mgr.update_subscriptions({"AAPL", "BTC-USD"})
        # BTC-USD filtered; no change to AAPL
        mock_stream.subscribe_trades.assert_not_called()

    # ----------------------------------------------------------------
    # Stale tickers
    # ----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stale_tickers_reported(self):
        on_trade = []
        mgr = AlpacaStreamManager(on_trade=lambda t, p: on_trade.append(t))
        mgr._symbols = {"AAPL", "MSFT"}
        # MSFT: simulate last trade was 400s ago
        mgr._last_trade_time["MSFT"] = time.monotonic() - 400

        stale = mgr.get_stale_tickers(stale_seconds=300.0)
        assert "MSFT" in stale
        assert "AAPL" in stale  # never had a trade → stale too

    @pytest.mark.asyncio
    async def test_recently_seen_ticker_not_stale(self):
        mgr = AlpacaStreamManager(on_trade=MagicMock())
        mgr._symbols = {"AAPL"}
        mgr._last_trade_time["AAPL"] = time.monotonic() - 10  # 10s ago

        stale = mgr.get_stale_tickers(stale_seconds=300.0)
        assert "AAPL" not in stale

    # ----------------------------------------------------------------
    # is_stream_enabled
    # ----------------------------------------------------------------

    def test_is_stream_enabled_requires_keys(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
        monkeypatch.delenv("ALPACA_STREAM_ENABLED", raising=False)
        assert is_stream_enabled() is False

    def test_is_stream_enabled_can_be_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        monkeypatch.setenv("ALPACA_STREAM_ENABLED", "0")
        assert is_stream_enabled() is False

    def test_is_stream_enabled_true(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        monkeypatch.setenv("ALPACA_STREAM_ENABLED", "1")
        assert is_stream_enabled() is True


# ===========================================================================
# 2. AlertEngine.evaluate_single_ticker unit tests
# ===========================================================================


class TestAlertEngineEvaluateSingleTicker:
    """Unit tests for AlertEngine threshold evaluation. Mocks the DB dedup call."""

    def setup_method(self):
        self.engine = AlertEngine()

    def _eval(self, pos: dict, price: float, log_returns: bool = True):
        """Helper: evaluate with mocked log_price_alert returning log_returns."""
        with patch("trading.engine.log_price_alert", return_value=log_returns) as mock_log:
            result = self.engine.evaluate_single_ticker(pos["ticker"], price, pos)
        return result, mock_log

    # Stop breach
    def test_stop_breach_returns_event(self):
        pos = _make_pos(entry=100.0, stop=95.0)
        result, _ = self._eval(pos, price=94.0)
        assert result is not None
        assert result.alert_type == "stop_breached"
        assert result.ticker == "AAPL"
        assert result.current_price == 94.0
        assert result.level_price == 95.0

    def test_stop_breach_uses_trailing_stop_when_higher(self):
        pos = _make_pos(entry=100.0, stop=95.0, trailing=97.0)
        result, _ = self._eval(pos, price=96.0)
        assert result is not None
        assert result.alert_type == "stop_breached"
        assert result.level_name == "Trailing Stop"
        assert result.level_price == 97.0

    def test_stop_breach_dedup_returns_none(self):
        pos = _make_pos(entry=100.0, stop=95.0)
        result, _ = self._eval(pos, price=94.0, log_returns=False)
        assert result is None

    # Target breach
    def test_target_breach_returns_event(self):
        pos = _make_pos(entry=100.0, target=110.0)
        result, _ = self._eval(pos, price=111.0)
        assert result is not None
        assert result.alert_type == "target_breached"
        assert result.level_name == "Target"
        assert result.pnl_pct == pytest.approx(11.0, abs=0.1)

    def test_target_breach_dedup_returns_none(self):
        pos = _make_pos(entry=100.0, target=110.0)
        result, _ = self._eval(pos, price=111.0, log_returns=False)
        assert result is None

    # Near stop
    def test_near_stop_within_proximity(self):
        self.engine._proximity_pct = 1.5
        pos = _make_pos(entry=100.0, stop=95.0)
        # 1% above stop
        result, _ = self._eval(pos, price=95.95)
        assert result is not None
        assert result.alert_type == "near_stop"

    def test_near_stop_outside_proximity_returns_none(self):
        self.engine._proximity_pct = 1.5
        pos = _make_pos(entry=100.0, stop=95.0)
        # 3% above stop — outside proximity
        result, _ = self._eval(pos, price=97.85)
        assert result is None

    # Near target
    def test_near_target_within_proximity(self):
        self.engine._proximity_pct = 1.5
        pos = _make_pos(entry=100.0, stop=90.0, target=110.0)
        # 1% below target
        result, _ = self._eval(pos, price=108.9)
        assert result is not None
        assert result.alert_type == "near_target"

    # Profit lock (check_prices)
    @pytest.mark.asyncio
    async def test_check_prices_applies_profit_lock(self):
        pos = _make_pos(entry=100.0, stop=95.0, target=110.0)
        quotes = {"AAPL": {"price": 102.5}}  # above 2% lock level

        with patch("trading.engine.save_position_meta") as mock_save, \
             patch("trading.engine.log_price_alert", return_value=False):
            events = await self.engine.check_prices([pos], quotes)

        # stop_price should have been raised to lock level
        assert pos["stop_price"] == pytest.approx(100.0 * 1.02, rel=1e-4)
        mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_prices_skips_missing_quote(self):
        pos = _make_pos()
        with patch("trading.engine.log_price_alert", return_value=False):
            events = await self.engine.check_prices([pos], {})
        assert events == []


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

        with patch("trading.scheduler.load_positions", return_value={"positions": [pos]}), \
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
    async def test_concurrent_exit_guard_prevents_double_exit(self):
        """Second breach for same ticker while first is exiting should be skipped."""
        exit_calls = []

        async def on_event(event):
            pass

        from trading.scheduler import TradingScheduler
        scheduler = TradingScheduler(on_event=on_event)

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
        from trading.scheduler import TradingScheduler
        from trading.engine import StreamDegradedEvent

        received = []
        scheduler = TradingScheduler(on_event=AsyncMock(side_effect=lambda e: received.append(e)))

        mock_stream = MagicMock()
        mock_stream.is_connected = False
        mock_stream.get_stale_tickers.return_value = []
        scheduler._stream = mock_stream

        with patch.object(scheduler._engine, "is_market_open", return_value=True), \
             patch.object(scheduler._engine, "run_housekeeping", new_callable=AsyncMock,
                          return_value=MagicMock(fills_confirmed=[], orders_placed=[])), \
             patch("trading.scheduler.load_positions", return_value={"positions": []}), \
             patch("trading.scheduler.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
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
        from trading.scheduler import TradingScheduler
        from trading.engine import StreamDegradedEvent

        received = []
        scheduler = TradingScheduler(on_event=AsyncMock(side_effect=lambda e: received.append(e)))

        mock_stream = MagicMock()
        mock_stream.is_connected = False
        mock_stream.get_stale_tickers.return_value = []
        scheduler._stream = mock_stream

        with patch.object(scheduler._engine, "is_market_open", return_value=True), \
             patch.object(scheduler._engine, "run_housekeeping", new_callable=AsyncMock,
                          return_value=MagicMock(fills_confirmed=[], orders_placed=[])), \
             patch("trading.scheduler.load_positions", return_value={"positions": []}), \
             patch("trading.scheduler.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
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
        from trading.scheduler import TradingScheduler
        from trading.engine import StreamDegradedEvent, StreamRecoveredEvent

        received = []
        scheduler = TradingScheduler(on_event=AsyncMock(side_effect=lambda e: received.append(e)))

        mock_stream = MagicMock()
        scheduler._stream = mock_stream

        # Pre-condition: degradation alert was already sent
        scheduler._stream_degraded_cycles = 7
        scheduler._stream_alert_sent = True

        # Stream is now connected (recovered)
        mock_stream.is_connected = True
        mock_stream.get_stale_tickers.return_value = []

        with patch.object(scheduler._engine, "is_market_open", return_value=True), \
             patch.object(scheduler._engine, "run_housekeeping", new_callable=AsyncMock,
                          return_value=MagicMock(fills_confirmed=[], orders_placed=[])), \
             patch("trading.scheduler.load_positions", return_value={"positions": []}), \
             patch("trading.scheduler.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
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
        from trading.scheduler import TradingScheduler
        from trading.engine import StreamDegradedEvent

        received = []
        scheduler = TradingScheduler(on_event=AsyncMock(side_effect=lambda e: received.append(e)))

        mock_stream = MagicMock()
        mock_stream.is_connected = False
        mock_stream.get_stale_tickers.return_value = []
        scheduler._stream = mock_stream

        with patch.object(scheduler._engine, "is_market_open", return_value=True), \
             patch.object(scheduler._engine, "run_housekeeping", new_callable=AsyncMock,
                          return_value=MagicMock(fills_confirmed=[], orders_placed=[])), \
             patch("trading.scheduler.load_positions", return_value={"positions": []}), \
             patch("trading.scheduler.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
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

        received = []
        scheduler = TradingScheduler(on_event=AsyncMock())
        mock_stream = MagicMock()
        mock_stream.is_connected = True
        scheduler._stream = mock_stream
        scheduler._last_alert_check = scheduler._last_alert_check.min.replace(
            tzinfo=scheduler._last_alert_check.tzinfo
        )

        with patch.object(scheduler._engine, "is_market_open", return_value=True), \
             patch("trading.scheduler.get_snapshots") as mock_snap, \
             patch("trading.scheduler.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
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
        from trading.scheduler import TradingScheduler
        import zoneinfo
        from datetime import datetime

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
             patch("trading.scheduler.load_positions", return_value={"positions": [pos]}), \
             patch("trading.scheduler.alpaca_enabled", return_value=False), \
             patch("trading.scheduler.fetch_quotes", return_value={}), \
             patch.object(scheduler._engine, "check_prices", mock_check), \
             patch("trading.scheduler.asyncio.sleep", new_callable=AsyncMock) as mock_sleep, \
             patch("trading.scheduler.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = {}
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            try:
                await scheduler._polling_loop()
            except asyncio.CancelledError:
                pass

        mock_check.assert_called_once()


# ===========================================================================
# 5. Bot notification test
# ===========================================================================


class TestBotNotification:
    """Verify _handle_trading_event calls the correct embed builders."""

    def _make_handler(self):
        """Return (bot_module, _handle_trading_event, mock_send_alert)."""
        import importlib
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
    async def test_near_stop_sends_approaching_embed(self):
        from integrations.discord.notify import _approaching_alert_embed

        event = AlertEvent(
            ticker="MSFT",
            alert_type="near_stop",
            current_price=96.0,
            level_price=95.0,
            level_name="Stop",
            entry_price=100.0,
            pnl_pct=-4.0,
        )

        embed_data = _approaching_alert_embed([event.model_dump()])
        assert embed_data["color"] == 0xFF8C00  # orange
        assert "MSFT" in str(embed_data["fields"])

    @pytest.mark.asyncio
    async def test_exit_result_sends_exit_embed(self):
        from integrations.discord.notify import _exit_embed

        event = ExitResult(
            ticker="AAPL",
            exit_reason="stop",
            fill_price=94.5,
            order_id="abc123",
        )

        embed_data = _exit_embed([event.model_dump()])
        assert "AAPL" in str(embed_data)

    @pytest.mark.asyncio
    async def test_scan_result_signals_uses_signal_embed(self):
        from trading.engine import ScanResult
        from integrations.discord.notify import _signal_embed

        event = ScanResult(
            signals=[{
                "ticker": "NVDA",
                "p_rally": 0.72,
                "close": 850.0,
                "signal": True,
                "size": 0.05,
            }],
            exits=[],
            orders=[],
            positions_summary={"positions": []},
            scan_type="daily",
        )

        embed_data = _signal_embed(event.signals)
        assert "NVDA" in str(embed_data)

    @pytest.mark.asyncio
    async def test_stream_degraded_embed(self):
        from trading.engine import StreamDegradedEvent
        from integrations.discord.notify import _stream_degraded_embed

        event = StreamDegradedEvent(disconnected_minutes=7)
        embed_data = _stream_degraded_embed(event.disconnected_minutes)
        assert embed_data["color"] == 0xFF8C00
        assert "7" in embed_data["description"]
        assert "15 minutes" in embed_data["description"]

    @pytest.mark.asyncio
    async def test_stream_recovered_embed(self):
        from trading.engine import StreamRecoveredEvent
        from integrations.discord.notify import _stream_recovered_embed

        event = StreamRecoveredEvent(downtime_minutes=12)
        embed_data = _stream_recovered_embed(event.downtime_minutes)
        assert embed_data["color"] == 0x00FF00
        assert "12" in embed_data["description"]

    @pytest.mark.asyncio
    async def test_housekeeping_fill_notification(self):
        from trading.engine import FillNotification
        from integrations.discord.notify import _fill_confirmation_embed

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
