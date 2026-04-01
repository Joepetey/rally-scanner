"""Scan result display — console tables and breadth statistics."""

import numpy as np

from config import PARAMS


def _compute_breadth(ok_results: list[dict]) -> None:
    """Print market breadth statistics to stdout."""
    above_ma50 = sum(
        1 for r in ok_results
        if r["close"] > r.get("ma50", 0) and r.get("ma50", 0) > 0
    )
    above_ma200 = sum(1 for r in ok_results if r.get("trend", 0))
    golden_crosses = sum(1 for r in ok_results if r.get("golden_cross", 0))
    avg_vix_pctile = np.mean([r.get("vix_pctile", 0.5) for r in ok_results])
    breadth_50 = above_ma50 / len(ok_results)
    breadth_200 = above_ma200 / len(ok_results)
    n = len(ok_results)

    print(f"\n  {'='*86}")
    print("  MARKET BREADTH")
    print(f"  {'='*86}")
    b50_bar = "#" * int(breadth_50 * 40) + "." * (40 - int(breadth_50 * 40))
    b200_bar = "#" * int(breadth_200 * 40) + "." * (40 - int(breadth_200 * 40))
    print(f"  Above 50-day MA:  {above_ma50:>3}/{n} ({breadth_50:.0%})  [{b50_bar}]")
    print(f"  Above 200-day MA: {above_ma200:>3}/{n} ({breadth_200:.0%})  [{b200_bar}]")
    gc_pct = golden_crosses / n
    print(f"  Golden crosses:   {golden_crosses:>3}/{n} ({gc_pct:.0%})")
    if avg_vix_pctile > 0.80:
        vix_label = "EXTREME FEAR"
    elif avg_vix_pctile > 0.60:
        vix_label = "FEAR"
    elif avg_vix_pctile > 0.40:
        vix_label = "NEUTRAL"
    elif avg_vix_pctile > 0.20:
        vix_label = "GREED"
    else:
        vix_label = "EXTREME GREED"
    print(f"  VIX percentile:   {avg_vix_pctile:.0%} ({vix_label})")


def _print_signal_table(signals: list[dict]) -> None:
    header = (f"  {'Ticker':<7} {'Close':>8} {'P(rally)':>9} {'Comp':>6} "
              f"{'GC':>3} {'MACD':>7} {'Vol':>5} {'VIX':>4} "
              f"{'Size':>6} {'Stop':>8} {'Target':>8}")
    print(header)
    print(f"  {'-'*78}")
    for r in sorted(signals, key=lambda x: x["p_rally"], reverse=True):
        atr_val = r["close"] * r["atr_pct"]
        target = r["close"] + PARAMS.profit_atr_mult * atr_val
        print(f"  {r['ticker']:<7} {r['close']:>8.2f} {r['p_rally']:>8.1%} "
              f"{r['comp_score']:>6.3f} "
              f"{'Y' if r.get('golden_cross') else 'N':>3} "
              f"{r.get('macd_hist', 0):>+6.4f} "
              f"{r.get('vol_ratio', 1):>5.1f} "
              f"{r.get('vix_pctile', 0.5):>3.0%} "
              f"{r['size']:>5.1%} "
              f"{r['range_low']:>8.2f} {target:>8.2f}")


def _print_watchlist_table(watchlist: list[dict]) -> None:
    header = (f"  {'Ticker':<7} {'Close':>8} {'P(rally)':>9} {'Comp':>6} "
              f"{'Trend':>5} {'HMM_C':>6} {'RV%':>5} {'RSI':>5} {'Missing':>12}")
    print(header)
    print(f"  {'-'*67}")
    for r in sorted(watchlist, key=lambda x: x["p_rally"], reverse=True)[:20]:
        missing = []
        if r["p_rally"] <= PARAMS.p_rally_threshold:
            missing.append("P<thr")
        if r["comp_score"] <= PARAMS.comp_score_threshold:
            missing.append("Comp")
        if not r["trend"]:
            missing.append("Trend")
        missing_str = ",".join(missing) if missing else "?"

        print(f"  {r['ticker']:<7} {r['close']:>8.2f} {r['p_rally']:>8.1%} "
              f"{r['comp_score']:>6.3f} {'Y' if r['trend'] else 'N':>5} "
              f"{r['hmm_compressed']:>6.3f} {r['rv_pctile']:>5.1%} "
              f"{r['rsi']:>5.1f} {missing_str:>12}")


def _print_probability_table(results: list[dict]) -> None:
    """Print every asset ranked by P(rally) with visual probability bar."""
    p = PARAMS
    sorted_r = sorted(results, key=lambda x: x["p_rally"], reverse=True)

    header = (f"  {'#':>3} {'Ticker':<7} {'P(rally)':>9} {'':20} "
              f"{'Comp':>6} {'Trend':>5} {'HMM_C':>6} {'RV%':>5} {'RSI':>5} {'Close':>9}")
    print(header)
    print(f"  {'-'*80}")

    for i, r in enumerate(sorted_r, 1):
        prob = r["p_rally"]

        # Visual bar: 20 chars wide, filled proportionally
        bar_len = 20
        filled = int(prob * bar_len)
        # Color coding via symbols: above signal threshold vs below
        if r.get("signal"):
            bar = ">" * filled + "." * (bar_len - filled)
            tag = " <<< SIGNAL"
        elif prob > p.p_rally_threshold:
            bar = "#" * filled + "." * (bar_len - filled)
            tag = ""
        else:
            bar = "|" * filled + "." * (bar_len - filled)
            tag = ""

        print(f"  {i:>3} {r['ticker']:<7} {prob:>8.1%} [{bar}] "
              f"{r['comp_score']:>6.3f} {'Y' if r['trend'] else 'N':>5} "
              f"{r['hmm_compressed']:>6.3f} {r['rv_pctile']:>5.1%} "
              f"{r['rsi']:>5.1f} {r['close']:>9.2f}{tag}")
