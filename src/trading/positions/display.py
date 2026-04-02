"""Position display utilities."""


def print_positions(state: dict) -> None:
    """Print current positions and today's closes."""
    positions = state.get("positions", [])
    closed = state.get("closed_today", [])

    if positions:
        print(f"\n  {'='*72}")
        print(f"  OPEN POSITIONS ({len(positions)})")
        print(f"  {'='*72}")
        header = (
            f"  {'Ticker':<7} {'Entry':>8} {'Current':>8} {'PnL%':>7} "
            f"{'Stop':>8} {'Target':>8} {'Bars':>5} {'Size':>6}"
        )
        print(header)
        print(f"  {'-'*68}")
        for p in sorted(
            positions,
            key=lambda x: x.get("unrealized_pnl_pct", 0),
            reverse=True,
        ):
            pnl = p.get("unrealized_pnl_pct", 0)
            sign = "+" if pnl >= 0 else ""
            print(
                f"  {p['ticker']:<7} {p['entry_price']:>8.2f} "
                f"{p.get('current_price', 0):>8.2f} {sign}{pnl:>6.2f}% "
                f"{p.get('stop_price', 0):>8.2f} "
                f"{p.get('target_price', 0):>8.2f} "
                f"{p.get('bars_held', 0):>5d} {p.get('size', 0):>5.1%}"
            )

    if closed:
        print(f"\n  {'='*72}")
        print(f"  CLOSED TODAY ({len(closed)})")
        print(f"  {'='*72}")
        for c in closed:
            pnl = c.get("realized_pnl_pct", 0)
            sign = "+" if pnl >= 0 else ""
            print(
                f"  {c['ticker']:<7} {c.get('exit_reason', '?'):<15} "
                f"PnL: {sign}{pnl:.2f}%  (held {c.get('bars_held', 0)} bars)"
            )
