"""Discord embed builders for market-rally alerts.

All functions are pure — they take data dicts and return embed dicts.
No I/O, no side effects.
"""

from rally_ml.config import PARAMS as _P

from integrations.discord import colors

_MAX_FIELDS = 25  # Discord embed field limit


def _pnl_sign(pnl: float) -> str:
    return "+" if pnl >= 0 else ""


def _build_embed(
    title: str,
    color: int,
    *,
    fields: list[dict] | None = None,
    description: str | None = None,
    footer: str | None = None,
) -> dict:
    """Shared embed constructor with Discord field limit enforcement."""
    embed: dict = {"title": title, "color": color}
    if fields is not None:
        embed["fields"] = fields[:_MAX_FIELDS]
    if description is not None:
        embed["description"] = description
    if footer is not None:
        embed["footer"] = {"text": footer}
    return embed


def _signal_embed(signals: list[dict]) -> dict:
    fields = []
    for s in sorted(signals, key=lambda x: x.get("p_rally", 0), reverse=True):
        close = s.get("close", 0)
        atr_pct = s.get("atr_pct", _P.default_atr_pct)
        target = close * (1 + 2.0 * atr_pct)
        fields.append({
            "name": s.get("ticker", "?"),
            "value": (
                f"P(rally): **{s.get('p_rally', 0):.0%}**\n"
                f"Price: ${close:.2f}\n"
                f"Stop: ${s.get('range_low', 0):.2f}\n"
                f"Target: ${target:.2f}\n"
                f"Size: {s.get('size', 0):.0%}"
            ),
            "inline": True,
        })
    return _build_embed(f"New Signals ({len(signals)})", colors.GREEN, fields=fields)


def _exit_embed(closed: list[dict]) -> dict:
    fields = []
    for c in closed:
        pnl = c.get("realized_pnl_pct") or 0
        fields.append({
            "name": c.get("ticker", "?"),
            "value": (
                f"Reason: {c.get('exit_reason', '?')}\n"
                f"PnL: **{_pnl_sign(pnl)}{pnl:.2f}%**\n"
                f"Bars held: {c.get('bars_held') or 0}"
            ),
            "inline": True,
        })
    any_loss = any((c.get("realized_pnl_pct") or 0) < 0 for c in closed)
    return _build_embed(
        f"Position Exits ({len(closed)})",
        colors.RED if any_loss else colors.GREEN,
        fields=fields,
    )


def _retrain_embed(health: dict, elapsed: float) -> dict:
    fresh = health.get("fresh_count", 0)
    total = health.get("total_count", 0)
    return _build_embed("Retrain Complete", colors.BLUE, fields=[
        {"name": "Models", "value": f"{fresh}/{total} fresh", "inline": True},
        {"name": "Stale", "value": str(health.get("stale_count", 0)), "inline": True},
        {"name": "Elapsed", "value": f"{elapsed:.0f}s", "inline": True},
    ])


def _alert_fields(alerts: list[dict], *, breached: bool) -> list[dict]:
    """Shared field builder for price alert and approaching alert embeds."""
    fields = []
    for a in alerts:
        pnl = a.get("pnl_pct", 0)
        if breached:
            emoji = colors.EMOJI_STOP if "stop" in a["alert_type"] else colors.EMOJI_TARGET
            name = f"{emoji} {a['ticker']}"
            lines = [
                f"**{a['level_name']} BREACHED**",
                f"Level: ${a['level_price']:.2f}",
                f"Current: ${a['current_price']:.2f}",
                f"Entry: ${a['entry_price']:.2f}",
                f"PnL: {_pnl_sign(pnl)}{pnl:.2f}%",
            ]
        else:
            name = a["ticker"]
            lines = [
                f"Approaching **{a['level_name']}**",
                f"Level: ${a['level_price']:.2f} ({a['distance_pct']:.1f}% away)",
                f"Current: ${a['current_price']:.2f}",
                f"PnL: {_pnl_sign(pnl)}{pnl:.2f}%",
            ]
        fields.append({"name": name, "value": "\n".join(lines), "inline": True})
    return fields


def _price_alert_embed(alerts: list[dict]) -> dict:
    fields = _alert_fields(alerts, breached=True)
    any_stop = any("stop" in a["alert_type"] for a in alerts)
    return _build_embed(
        f"Price Alert ({len(alerts)})",
        colors.RED if any_stop else colors.GREEN,
        fields=fields,
        footer=colors.FOOTER_INTRADAY,
    )


def _approaching_alert_embed(alerts: list[dict]) -> dict:
    fields = _alert_fields(alerts, breached=False)
    return _build_embed(
        f"Price Warning ({len(alerts)})",
        colors.ORANGE,
        fields=fields,
        footer=colors.FOOTER_INTRADAY_WARN,
    )


def _order_embed(results: list, equity: float) -> dict:
    fields = []
    for r in results:
        parts = [f"Side: **{r['side'].upper()}**", f"Qty: {r['qty']}"]
        if r.get("fill_price"):
            parts.append(f"Fill: ${r['fill_price']:.2f}")
        if r.get("order_id"):
            parts.append(f"Order: `{r['order_id'][:8]}`")
        fields.append({"name": r["ticker"], "value": "\n".join(parts), "inline": True})
    return _build_embed(
        f"Alpaca Orders ({len(results)})", colors.SKY,
        fields=fields, footer=f"Account equity: ${equity:,.0f}",
    )


def _fill_confirmation_embed(fills: list[dict]) -> dict:
    fields = []
    for f in fills:
        fields.append({
            "name": f["ticker"],
            "value": (
                f"Fill confirmed: **${f['fill_price']:.2f}**\n"
                f"Qty: {f.get('qty', '?')}\n"
                f"Stop: ${f.get('stop_price', 0):.2f}  Target: ${f.get('target_price', 0):.2f}"
            ),
            "inline": True,
        })
    return _build_embed(
        f"Entry Fills Confirmed ({len(fills)})", colors.SKY,
        fields=fields, footer="Intraday fill confirmation",
    )


def _order_failure_embed(results: list) -> dict:
    fields = []
    for r in results:
        fields.append({
            "name": r["ticker"],
            "value": f"Side: {r['side'].upper()}\nError: {r.get('error', '?')}",
            "inline": True,
        })
    return _build_embed(
        f"Alpaca Order Failures ({len(results)})", colors.RED_ORANGE,
        fields=fields, footer="System positions unaffected",
    )


def _regime_shift_embed(transitions: list[dict]) -> dict:
    fields = []
    for t in transitions:
        fields.append({
            "name": t["ticker"],
            "value": (
                f"**{t['prev_regime']}** \u2192 **{t['new_regime']}**\n"
                f"P(exp): {t['p_expanding']:.0%}\n"
                f"P(comp): {t['p_compressed']:.0%}"
            ),
            "inline": True,
        })
    any_expanding = any(t["new_regime"] == "expanding" for t in transitions)
    return _build_embed(
        f"Regime Shift ({len(transitions)})",
        colors.RED if any_expanding else colors.ORANGE,
        fields=fields, footer="HMM volatility regime transition detected",
    )


def _risk_action_embed(results: list[dict]) -> dict:
    fields = []
    for r in results:
        if r.get("skipped"):
            continue
        if r["action"] == "close":
            pnl = r.get("pnl_pct", 0)
            fields.append({
                "name": f"{colors.EMOJI_CLOSE} {r['ticker']}",
                "value": (
                    f"**CLOSED** (risk reduction)\n"
                    f"PnL: {_pnl_sign(pnl)}{pnl:.2f}%\n"
                    f"Reason: {r['reason']}"
                ),
                "inline": True,
            })
        elif r["action"] == "tighten":
            fields.append({
                "name": f"{colors.EMOJI_WARN} {r['ticker']}",
                "value": (
                    f"**Stop tightened**\n"
                    f"${r.get('old_stop', 0):.2f} \u2192 ${r.get('new_stop', 0):.2f}\n"
                    f"Reason: {r['reason']}"
                ),
                "inline": True,
            })
    if not fields:
        return _build_embed("Risk Actions", colors.GRAY, description="No actions needed")
    any_close = any(r["action"] == "close" for r in results if not r.get("skipped"))
    return _build_embed(
        f"Proactive Risk Actions ({len(fields)})",
        colors.RED if any_close else colors.ORANGE,
        fields=fields, footer="Automated risk management",
    )


def _positions_embed(positions: list[dict]) -> dict:
    if not positions:
        return _build_embed("Open Positions (0)", colors.GRAY, description="No open positions")

    sorted_pos = sorted(positions, key=lambda x: x.get("unrealized_pnl_pct", 0), reverse=True)
    lines = []
    for p in sorted_pos:
        pnl = p.get("unrealized_pnl_pct", 0)
        current = p.get("current_price", p.get("entry_price", 0))
        lines.append(
            f"`{p['ticker']:<6s}` "
            f"${current:>7.2f}  "
            f"**{_pnl_sign(pnl)}{pnl:.2f}%**  "
            f"Stop ${p['stop_price']:.2f}  "
            f"Target ${p['target_price']:.2f}  "
            f"{p.get('bars_held', 0)}d  "
            f"{p.get('size', 0):.0%}"
        )
    any_loss = any(p.get("unrealized_pnl_pct", 0) < 0 for p in positions)
    return _build_embed(
        f"Open Positions ({len(positions)})",
        colors.ORANGE if any_loss else colors.GREEN,
        description="\n".join(lines),
    )


def _error_embed(title: str, details: str) -> dict:
    return _build_embed(f"Error: {title}", colors.RED, description=details[:4096])


def _stream_degraded_embed(disconnected_minutes: int) -> dict:
    return _build_embed(
        "Stream Degraded \u2014 Polling-Only Mode", colors.ORANGE,
        description=(
            f"The Alpaca WebSocket stream has been disconnected for "
            f"**{disconnected_minutes} min**.\n"
            "Price alert latency has degraded from seconds to ~15 minutes.\n"
            "Reconnect attempts are ongoing \u2014 check Railway logs for details."
        ),
    )


def _stream_recovered_embed(downtime_minutes: int) -> dict:
    return _build_embed(
        "Stream Recovered", colors.GREEN,
        description=(
            f"The Alpaca WebSocket stream has reconnected after "
            f"**{downtime_minutes} min** of downtime.\n"
            "Real-time price alerts are active again."
        ),
    )
