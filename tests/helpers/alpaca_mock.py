"""In-memory fake of the Alpaca trading client.

Provides AlpacaMock — a shared test double for all order execution tests.
Methods are wrapped in MagicMock so call assertions (assert_called_once_with,
call_args, etc.) work exactly as they would against a real MagicMock, while
the underlying logic simulates realistic Alpaca broker behaviour.

Typical usage (via conftest.py fixture):

    async def test_something(alpaca_mock):
        alpaca_mock.set_fill_behavior("immediate", fill_price=150.0)
        result = await execute_exit("AAPL")
        alpaca_mock.close_position.assert_called_once()

For complex sequential behaviour, override side_effect directly:

    alpaca_mock.close_position.side_effect = [
        Exception('{"code":40410000,"message":"position not found"}'),
        MockAlpacaOrder(id="x", symbol="MSFT", ...),
    ]
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from unittest.mock import MagicMock

from alpaca.trading.enums import OrderSide, OrderStatus

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class MockAlpacaOrder:
    id: str
    symbol: str
    qty: str
    side: OrderSide
    status: OrderStatus
    filled_avg_price: str | None = None
    filled_qty: str | None = None
    filled_at: datetime | None = None
    legs: list[MockAlpacaOrder] = field(default_factory=list)


@dataclass
class MockAlpacaPosition:
    symbol: str
    qty: str
    avg_entry_price: str
    market_value: str
    unrealized_pl: str


@dataclass
class MockAlpacaAccount:
    equity: str


# ---------------------------------------------------------------------------
# Main mock class
# ---------------------------------------------------------------------------


class AlpacaMock:
    """In-memory fake of the Alpaca TradingClient.

    All public methods are MagicMock instances wrapping internal _impl methods.
    This means you can:
      - Use alpaca_mock.submit_order.assert_called_once_with(...)
      - Override alpaca_mock.close_position.side_effect = [exc, value, ...]
      - Inspect alpaca_mock.cancel_order_by_id.call_args_list
    """

    def __init__(self) -> None:
        self._orders: dict[str, MockAlpacaOrder] = {}
        self._positions: list[MockAlpacaPosition] = []
        self._fill_behavior: str = "immediate"
        self._fill_price: float = 150.0
        self._close_behavior: str = "immediate"
        self._oco_unlock_after: int = 0
        self._oco_attempts: int = 0
        self._submit_error: str | None = None
        self._cancel_error: str | None = None
        self._account_equity: float = 100_000.0

        # Public API — MagicMock wrappers around _impl methods
        self.submit_order = MagicMock(side_effect=self._submit_order_impl)
        self.close_position = MagicMock(side_effect=self._close_position_impl)
        self.get_orders = MagicMock(side_effect=self._get_orders_impl)
        self.get_order_by_id = MagicMock(side_effect=self._get_order_by_id_impl)
        self.cancel_order_by_id = MagicMock(side_effect=self._cancel_order_by_id_impl)
        self.get_all_positions = MagicMock(side_effect=self._get_all_positions_impl)
        self.get_account = MagicMock(side_effect=self._get_account_impl)

    # ── Configuration helpers ──────────────────────────────────────────────

    def set_fill_behavior(self, mode: str, fill_price: float | None = None) -> None:
        """Configure how submit_order behaves.

        mode:
          "immediate"  — order fills instantly (default)
          "pending"    — order submitted but not yet filled (filled_avg_price=None)
          "reject"     — raises Exception with code 42210000 (asset not tradable)
        """
        self._fill_behavior = mode
        if fill_price is not None:
            self._fill_price = fill_price

    def set_submit_error(self, error_str: str) -> None:
        """Cause the next submit_order call to raise Exception(error_str).

        Clears after one use.
        """
        self._submit_error = error_str

    def set_close_behavior(self, mode: str, unlock_after: int = 1) -> None:
        """Configure how close_position behaves.

        mode:
          "immediate"     — closes successfully (default)
          "oco_locked"    — raises 40310000 for first `unlock_after` attempts,
                            then succeeds
          "already_closed" — always raises 40410000 (position not found)
        unlock_after: number of locked failures before succeeding (oco_locked only)
        """
        self._close_behavior = mode
        self._oco_unlock_after = unlock_after
        self._oco_attempts = 0

    def set_cancel_error(self, error_str: str) -> None:
        """Cause the next cancel_order_by_id call to raise Exception(error_str).

        Clears after one use.
        """
        self._cancel_error = error_str

    def set_account_equity(self, equity: float) -> None:
        self._account_equity = equity

    def add_open_position(
        self,
        symbol: str,
        qty: float,
        avg_entry_price: float,
        market_value: float | None = None,
        unrealized_pl: float = 0.0,
    ) -> None:
        """Seed an open broker position (for get_all_positions tests)."""
        if market_value is None:
            market_value = qty * avg_entry_price
        self._positions.append(MockAlpacaPosition(
            symbol=symbol,
            qty=str(qty),
            avg_entry_price=str(avg_entry_price),
            market_value=str(market_value),
            unrealized_pl=str(unrealized_pl),
        ))

    def add_filled_order(
        self,
        order_id: str,
        fill_price: float,
        symbol: str = "AAPL",
        qty: str = "10",
        side: OrderSide = OrderSide.BUY,
    ) -> MockAlpacaOrder:
        """Pre-populate a filled order so get_orders / check_pending_fills can find it."""
        order = MockAlpacaOrder(
            id=order_id,
            symbol=symbol,
            qty=qty,
            side=side,
            status=OrderStatus.FILLED,
            filled_avg_price=str(fill_price),
            filled_qty=qty,
            filled_at=datetime.now(tz=UTC),
        )
        self._orders[order_id] = order
        return order

    # ── TradingClient method implementations ──────────────────────────────

    def _submit_order_impl(self, request) -> MockAlpacaOrder:
        if self._submit_error:
            err, self._submit_error = self._submit_error, None
            raise Exception(err)

        if self._fill_behavior == "reject":
            raise Exception('{"code":42210000,"message":"asset is not tradable"}')

        symbol = str(request.symbol)
        qty = str(request.qty)
        side = request.side

        if self._fill_behavior == "pending":
            status = OrderStatus.PENDING_NEW
            filled_avg_price = None
            filled_qty = None
        else:  # "immediate" (default)
            status = OrderStatus.FILLED
            filled_avg_price = str(self._fill_price)
            filled_qty = qty

        order = MockAlpacaOrder(
            id=str(uuid.uuid4()),
            symbol=symbol,
            qty=qty,
            side=side,
            status=status,
            filled_avg_price=filled_avg_price,
            filled_qty=filled_qty,
        )
        self._orders[order.id] = order
        return order

    def _close_position_impl(self, symbol_or_asset_id: str) -> MockAlpacaOrder:
        if self._close_behavior == "already_closed":
            raise Exception(
                f'{{"code":40410000,"message":"position not found: {symbol_or_asset_id}"}}'
            )

        if self._close_behavior == "oco_locked":
            self._oco_attempts += 1
            if self._oco_attempts <= self._oco_unlock_after:
                raise Exception(
                    '{"code":40310000,"message":"shares held by OCO order"}'
                )

        order = MockAlpacaOrder(
            id=str(uuid.uuid4()),
            symbol=symbol_or_asset_id,
            qty="10",
            side=OrderSide.SELL,
            status=OrderStatus.FILLED,
            filled_avg_price=str(self._fill_price),
            filled_qty="10",
        )
        self._orders[order.id] = order
        self._positions = [
            p for p in self._positions if p.symbol != symbol_or_asset_id
        ]
        return order

    def _get_orders_impl(self, filter=None) -> list[MockAlpacaOrder]:
        return list(self._orders.values())

    def _get_order_by_id_impl(self, order_id: str) -> MockAlpacaOrder:
        if order_id not in self._orders:
            raise Exception(f"404: order {order_id!r} not found")
        return self._orders[order_id]

    def _cancel_order_by_id_impl(self, order_id: str) -> None:
        if self._cancel_error:
            err, self._cancel_error = self._cancel_error, None
            raise Exception(err)
        # Silently succeed for unknown IDs — we're testing executor logic, not
        # Alpaca's order-existence validation.
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELED

    def _get_all_positions_impl(self) -> list[MockAlpacaPosition]:
        return list(self._positions)

    def _get_account_impl(self) -> MockAlpacaAccount:
        return MockAlpacaAccount(equity=str(self._account_equity))
