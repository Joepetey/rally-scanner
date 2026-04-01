"""Asset definitions and group mappings."""

from pydantic import BaseModel


class AssetConfig(BaseModel):
    ticker: str
    asset_class: str  # "equity" or "crypto"
    r_up: float       # min rally magnitude
    d_dn: float       # max adverse excursion allowed


ASSETS: dict[str, AssetConfig] = {
    # --- Indices ---
    "SPY": AssetConfig(ticker="SPY", asset_class="equity", r_up=0.02, d_dn=0.01),
    "QQQ": AssetConfig(ticker="QQQ", asset_class="equity", r_up=0.025, d_dn=0.012),
    # --- Mega-cap tech ---
    "AAPL": AssetConfig(ticker="AAPL", asset_class="equity", r_up=0.03, d_dn=0.015),
    "MSFT": AssetConfig(ticker="MSFT", asset_class="equity", r_up=0.03, d_dn=0.015),
    "NVDA": AssetConfig(ticker="NVDA", asset_class="equity", r_up=0.05, d_dn=0.025),
    "AMZN": AssetConfig(ticker="AMZN", asset_class="equity", r_up=0.035, d_dn=0.018),
    "GOOG": AssetConfig(ticker="GOOG", asset_class="equity", r_up=0.03, d_dn=0.015),
    "META": AssetConfig(ticker="META", asset_class="equity", r_up=0.04, d_dn=0.02),
    "TSLA": AssetConfig(ticker="TSLA", asset_class="equity", r_up=0.06, d_dn=0.03),
    # --- Other popular ---
    "AMD": AssetConfig(ticker="AMD", asset_class="equity", r_up=0.05, d_dn=0.025),
    "NFLX": AssetConfig(ticker="NFLX", asset_class="equity", r_up=0.04, d_dn=0.02),
    "JPM": AssetConfig(ticker="JPM", asset_class="equity", r_up=0.03, d_dn=0.015),
    # --- Crypto ---
    "BTC": AssetConfig(ticker="BTC-USD", asset_class="crypto", r_up=0.04, d_dn=0.02),
    "ETH": AssetConfig(ticker="ETH-USD", asset_class="crypto", r_up=0.05, d_dn=0.025),
}

# Asset groups for concentration limits
ASSET_GROUPS: dict[str, list[str]] = {
    "mega_tech": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "META"],
    "growth": ["TSLA", "AMD", "NFLX"],
    "index": ["SPY", "QQQ"],
    "finance": ["JPM"],
    "crypto": ["BTC", "ETH"],
}

# Reverse lookup: ticker -> group
TICKER_TO_GROUP: dict[str, str] = {
    ticker: group
    for group, tickers in ASSET_GROUPS.items()
    for ticker in tickers
}
