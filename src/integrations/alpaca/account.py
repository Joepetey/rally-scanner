# stdlib
import asyncio
import logging

# third-party
try:
    from alpaca.data.requests import StockSnapshotRequest

    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

# local
import rally_ml.config as config
from rally_ml.core.data import fetch_quotes

from integrations.alpaca.broker import _data_client, _trading_client

logger = logging.getLogger(__name__)


async def get_account_equity() -> float:
    def _sync():
        client = _trading_client()
        account = client.get_account()
        return float(account.equity)

    return await asyncio.to_thread(_sync)


async def get_all_positions() -> list[dict]:
    def _sync():
        client = _trading_client()
        positions = client.get_all_positions()
        return [
            {
                "ticker": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
            }
            for p in positions
        ]

    return await asyncio.to_thread(_sync)


async def get_snapshots(tickers: list[str]) -> dict[str, dict]:
    equity_tickers = [
        t for t in tickers
        if t in config.ASSETS and config.ASSETS[t].asset_class == "equity"
    ]
    crypto_tickers = [
        t for t in tickers
        if t in config.ASSETS and config.ASSETS[t].asset_class == "crypto"
    ]

    result: dict[str, dict] = {}

    if equity_tickers:
        def _sync():
            client = _data_client()
            snapshots = client.get_stock_snapshot(
                StockSnapshotRequest(symbol_or_symbols=equity_tickers)
            )
            equity_result: dict[str, dict] = {}
            for ticker in equity_tickers:
                snap = snapshots.get(ticker)
                if not snap:
                    continue
                trade = snap.latest_trade
                quote = snap.latest_quote
                equity_result[ticker] = {
                    "price": float(trade.price) if trade else 0,
                    "bid": float(quote.bid_price) if quote else 0,
                    "ask": float(quote.ask_price) if quote else 0,
                    "bid_size": float(quote.bid_size) if quote else 0,
                    "ask_size": float(quote.ask_size) if quote else 0,
                }
            return equity_result

        result.update(await asyncio.to_thread(_sync))

    if crypto_tickers:
        # Use yfinance for crypto (BTC-USD format)
        yf_tickers = [config.ASSETS[t].ticker for t in crypto_tickers]
        yf_data = await asyncio.to_thread(fetch_quotes, yf_tickers)
        for t in crypto_tickers:
            yf_ticker = config.ASSETS[t].ticker.upper()
            data = yf_data.get(yf_ticker)
            if data and "error" not in data:
                result[t] = {
                    "price": data["price"],
                    "bid": 0,
                    "ask": 0,
                    "bid_size": 0,
                    "ask_size": 0,
                }

    return result
