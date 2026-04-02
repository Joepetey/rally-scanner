"""Shared test utilities for manual integration test scripts.

Provides a TestRunner that tracks pass/fail counts and common builders
for fake positions and price results used across test_all, test_exits,
and test_trade.
"""

from datetime import date

from rally_ml.config import PARAMS

from db.trading.positions import delete_position_meta, save_position_meta


class TestRunner:
    """Tracks pass/fail counts and provides assertion + output helpers."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def divider(self, title: str, char: str = "=") -> None:
        line = char * 64
        print(f"\n{line}")
        print(f"  {title}")
        print(line)

    def ok(self, msg: str = "") -> None:
        self.passed += 1
        print(f"  [PASS] {msg}")

    def fail(self, msg: str = "") -> None:
        self.failed += 1
        print(f"  [FAIL] {msg}")

    def check(self, cond: bool, msg: str) -> bool:
        if cond:
            self.ok(msg)
        else:
            self.fail(msg)
        return cond

    def summary(self) -> bool:
        total = self.passed + self.failed
        print(f"\n{'=' * 64}")
        if self.failed == 0:
            print(f"  ALL TESTS PASSED — {self.passed}/{total}")
        else:
            print(f"  {self.failed} FAILED — {self.passed}/{total} passed")
        print("=" * 64)
        return self.failed == 0


def cleanup(ticker: str) -> None:
    """Remove any test artifacts from DB."""
    delete_position_meta(ticker)


def fake_position(ticker: str, price: float, **overrides) -> dict:
    """Build a minimal position dict and save it to DB."""
    atr_val = price * PARAMS.default_atr_pct
    pos = {
        "ticker": ticker,
        "entry_price": price,
        "entry_date": str(date.today()),
        "stop_price": round(price * 0.97, 2),
        "target_price": round(price + PARAMS.profit_atr_mult * atr_val, 2),
        "trailing_stop": round(price - PARAMS.trailing_stop_atr_mult * atr_val, 2),
        "highest_close": price,
        "atr": round(atr_val, 4),
        "bars_held": 0,
        "size": 0.02,
        "qty": 1,
        "order_id": None,
        "trail_order_id": None,
        "p_rally": 0.65,
        "status": "open",
    }
    pos.update(overrides)
    save_position_meta(pos)
    return pos


def price_result(ticker: str, close: float, **extras) -> dict:
    """Minimal scanner result dict for update_existing_positions."""
    return {
        "ticker": ticker,
        "status": "ok",
        "close": close,
        "date": str(date.today()),
        "atr": close * PARAMS.default_atr_pct,
        "rv_pctile": 0.0,
        **extras,
    }
