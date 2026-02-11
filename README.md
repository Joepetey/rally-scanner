# Rally Phase-Transition Detector

A quantitative system that detects the transition from volatility compression to directional rally across equities and crypto. Automated pipeline: weekly model retraining, daily scanning with position tracking, Discord/Telegram/email alerts, interactive Discord bot for per-user trade tracking, and a terminal dashboard.

For strategy theory, feature engineering, model architecture, and trading rules, see [THEORY.md](THEORY.md).

---

## Quick Start

```bash
# Setup
make setup                    # creates .venv, installs package + dev deps

# Or manually:
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Core workflow
make retrain                  # train models for all ~843 tickers
make scan                     # daily scan + position tracking + alerts
make dashboard                # terminal portfolio view
make health                   # model freshness report
make discord                  # start Discord bot
make test                     # run test suite (84 tests)
```

For alerts, copy `.env.example` to `.env` and fill in your Discord bot token (or Telegram/SMTP/webhook credentials).

---

## Project Structure

```
market-rally/
    src/rally/                      # Python package (all core code)
        config.py                   # Parameters, asset definitions, thresholds
        data.py                     # yfinance OHLCV + VIX fetcher with batch + caching
        features.py                 # 10 base features: compression, range, traps, trend, volume, MACD, VIX
        hmm.py                      # 3-state Gaussian HMM for regime detection (+3 features)
        labels.py                   # RALLY_ST binary label (forward-looking)
        model.py                    # Walk-forward logistic regression + isotonic calibration
        calibrate.py                # Auto-calibrate r_up/d_dn from historical volatility
        trading.py                  # Signal generation, vol-targeted sizing, trade simulation
        backtest.py                 # Equity curve computation, metrics, diagnostic reports
        persistence.py              # Save/load trained models with joblib
        positions.py                # Track open positions in JSON across scanner runs
        universe.py                 # S&P 500 + Nasdaq Top 500 universe (~843 tickers)
        scanner.py                  # Daily scan — load models, fetch prices, generate alerts
        retrain.py                  # Weekly model retraining (parallel, batch fetch)
        notify.py                   # Discord, Telegram, email, webhook notification backends
        portfolio.py                # Daily snapshots, trade journal (CSV)
        discord_db.py               # SQLite persistence for per-user trade tracking
        discord_bot.py              # Interactive Discord bot (slash commands)
        log.py                      # Centralized logging (console + rotating file)
        main.py                     # Single-asset or 14-asset backtest
        optimize.py                 # Parameter sweep across named configurations
        backtest_universe.py        # Full universe 20-year backtest with config sweeps
    scripts/
        orchestrator.py             # Cron entry point: scan, retrain, auto, health
        dashboard.py                # Terminal dashboard for portfolio state
        run_discord.py              # Discord bot entry point
    tests/                          # pytest test suite (84 tests)
        conftest.py                 # Synthetic OHLCV fixtures
        test_config.py
        test_features.py
        test_labels.py
        test_trading.py
        test_positions.py
        test_notify.py
        test_portfolio.py
        test_discord_db.py
        test_discord_embeds.py
    scanner.py                      # Root shim → rally.scanner
    retrain.py                      # Root shim → rally.retrain
    main.py                         # Root shim → rally.main
    optimize.py                     # Root shim → rally.optimize
    backtest_universe.py            # Root shim → rally.backtest_universe
    models/                         # Trained model artifacts (gitignored)
    data_cache/                     # OHLCV parquet cache (gitignored)
    plots/                          # Generated charts (gitignored)
    logs/                           # Run logs (gitignored)
    Makefile
    pyproject.toml
    requirements.txt
    .env.example                    # Notification config template
    THEORY.md                       # Strategy theory & design docs
    STANDARDS.md                    # Project coding standards
```

Root-level `.py` files are 3-line shims that forward to the package, so `python scanner.py` keeps working.

### Data Flow

```
                 orchestrator.py retrain (weekly, via cron)
                      |
  yfinance ──> data.py ──> features.py ──> labels.py
                                |               |
                            hmm.py          model.py
                                |               |
                                └──> models/*.joblib
                                         |
                 orchestrator.py scan (daily, via cron)
                      |
              load model + fetch latest data
                      |
              predict P(rally) ──> generate signals
                      |
              positions.json ──> portfolio.py (snapshots, journal)
                      |
              notify.py ──> Discord / Telegram / email / webhook alerts
                      |
              discord_bot.py ──> interactive Discord commands
                      |
              dashboard.py ──> terminal display
```

---

## Usage

### Setup

```bash
# Option 1: Makefile (recommended)
make setup

# Option 2: Manual
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Makefile Targets

| Target | Command | Description |
|--------|---------|-------------|
| `make setup` | Create venv + install deps | First-time setup |
| `make scan` | `orchestrator.py scan` | Daily scan + alerts + portfolio snapshot |
| `make retrain` | `orchestrator.py retrain` | Weekly model retraining |
| `make dashboard` | `dashboard.py` | Terminal portfolio view |
| `make health` | `orchestrator.py health` | Model freshness report |
| `make discord` | `run_discord.py` | Start Discord bot |
| `make test` | `pytest -v` | Run 84 tests |
| `make lint` | `ruff check` | Lint source and tests |
| `make clean` | rm caches | Remove `__pycache__` and build artifacts |

### Weekly Retraining

```bash
python retrain.py                              # all tickers (parallel, batch fetch)
python retrain.py --tickers AAPL MSFT SPY      # specific tickers only
python retrain.py --validate                   # compare auto-cal vs hand-tuned
```

### Daily Scanning

```bash
python scanner.py                              # scan all trained assets
python scanner.py --positions                  # include open position tracking
python scanner.py --config conservative        # use conservative thresholds
python scanner.py --tickers AAPL NVDA MSFT     # scan specific tickers
```

Scanner output includes:
- **Market breadth**: % above 50/200-day MA, golden crosses, VIX sentiment
- **New signals**: tickers meeting all entry criteria with sizing and levels
- **Watchlist**: near-miss tickers (P_rally > 35%) approaching signal threshold
- **Full probability ranking**: all scanned assets sorted by rally probability
- **Open positions**: current holdings with P&L, stops, targets (with `--positions`)

### Backtesting

```bash
# Single asset
python main.py --asset SPY --plot

# Original 14 assets with portfolio aggregation
python main.py --run-all --plot

# Full universe backtest (800+ assets, ~2-4 hours)
python backtest_universe.py --plot

# Parameter optimization sweep
python optimize.py
```

All plots are saved to the `plots/` directory.

---

## Automation & Alerts

### Orchestrator

The orchestrator (`scripts/orchestrator.py`) is the cron entry point:

```bash
python scripts/orchestrator.py scan      # daily scan + position updates + alerts
python scripts/orchestrator.py retrain   # weekly retrain all models
python scripts/orchestrator.py auto      # auto-detect: Sun=retrain, Mon-Fri=scan
python scripts/orchestrator.py health    # model freshness report
```

**Cron setup** (add to `crontab -e`):

```cron
# Daily scan at 4:30 PM ET on weekdays
30 16 * * 1-5 cd /path/to/market-rally && .venv/bin/python scripts/orchestrator.py scan

# Weekly retrain on Sunday at 6 PM ET
0 18 * * 0 cd /path/to/market-rally && .venv/bin/python scripts/orchestrator.py retrain
```

### Notifications

Four backends — each silently no-ops if not configured:

| Backend | Config (`.env`) | Use case |
|---------|----------------|----------|
| **Discord** | `DISCORD_BOT_TOKEN`, `DISCORD_CHANNEL_ID` | Primary — rich embeds in your server |
| **Telegram** | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Instant mobile alerts |
| **Email** | `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `NOTIFY_EMAIL` | Backup / audit trail |
| **Webhook** | `WEBHOOK_URL` | Integration with Slack, custom systems |

Copy `.env.example` to `.env` and fill in credentials for your preferred backend(s).

### Dashboard

```bash
python scripts/dashboard.py              # cached state
python scripts/dashboard.py --live       # fetch live prices for open positions
python scripts/dashboard.py --journal 20 # show last 20 closed trades
python scripts/dashboard.py --equity 90  # show 90-day equity history
```

### Portfolio Tracking

Automatically maintained by the orchestrator:
- **Equity history** (`models/equity_history.csv`): daily snapshots of exposure, unrealized P&L, signal counts
- **Trade journal** (`models/trade_journal.csv`): every closed trade with entry/exit dates, P&L, exit reason

---

## Discord Bot

Interactive Discord bot for per-user trade tracking and system monitoring.

### Setup

1. Create a bot at [discord.com/developers/applications](https://discord.com/developers/applications)
2. Enable **MESSAGE CONTENT** intent in Bot settings
3. Generate an invite URL with `bot` + `applications.commands` scopes
4. Invite the bot to your server
5. Add to `.env`:
   ```
   DISCORD_BOT_TOKEN=your-token-here
   DISCORD_CHANNEL_ID=your-alerts-channel-id
   ```
6. Install with Discord support: `pip install -e ".[discord]"`
7. Start the bot: `make discord`

### Slash Commands

| Command | Description |
|---------|-------------|
| `/signals` | Show recent entry signals with sizing (dollar amounts if capital is set) |
| `/positions` | Show system's open positions with P&L and dollar exposure |
| `/setcapital <amount>` | Set your portfolio capital for dollar-based sizing |
| `/enter <ticker> <price> [size] [stop] [target]` | Record a trade entry — auto-fills size/stop/target from system signal |
| `/exit <ticker> <price> [notes]` | Close your oldest open trade for that ticker (FIFO) |
| `/history [ticker] [limit]` | Show your trade history with stop/target levels and dollar P&L |
| `/pnl [period]` | P&L summary with dollar amounts: all-time, 30d, or 7d |
| `/health` | Model freshness + system status |
| `/portfolio` | System equity history (last 30 days) |

### Auto-Alerts

The orchestrator pushes rich embeds to your Discord channel automatically (no bot process needed for alerts):

| Event | Embed Color | Content |
|-------|-------------|---------|
| New signals | Green | Ticker, P(rally), price, stop, target, size |
| Position exits | Green/Red | Ticker, exit reason, PnL%, bars held |
| Retrain complete | Blue | Models fresh/stale count, elapsed time |
| Errors | Red | Error title + details |

### Per-User Trade Tracking

Each Discord user gets their own trade journal stored in SQLite (`models/rally_discord.db`):

1. **Set your capital**: `/setcapital 100000` — tells the bot your portfolio size
2. **Enter trades**: `/enter AAPL 185.50` — auto-fills the system's recommended position size (e.g., 15%), stop price, and profit target from the active signal. Shows dollar allocation ("$15,000 of $100,000") and max risk.
3. **Exit trades**: `/exit AAPL 192.00` — closes the oldest open AAPL trade (FIFO), shows both percentage and dollar P&L ("+3.51% / +$527")
4. **Track performance**: `/pnl` — cumulative P&L in both % and $, win rate, best/worst trades
5. **Review history**: `/history` — full trade log with entry/exit dates, stop/target levels, and dollar P&L

Users are auto-registered on their first command — no setup needed. Capital can be updated anytime with `/setcapital`.

The bot is **capital-aware**: when you set your portfolio size, every command shows dollar amounts alongside percentages — position allocations, risk per trade, unrealized P&L on open positions, and cumulative returns.

### Cloud Deployment (Railway)

The bot includes a built-in scheduler so you don't need cron. One process handles everything: slash commands + daily scans + weekly retrains.

**Deploy to Railway:**

1. Push your repo to GitHub
2. Go to [railway.app](https://railway.app), create a new project from your repo
3. Add environment variables in Railway dashboard:
   ```
   DISCORD_BOT_TOKEN=your-token
   DISCORD_CHANNEL_ID=your-channel-id
   ENABLE_SCHEDULER=1
   ```
4. Railway auto-detects the `Procfile` and deploys

The scheduler runs:
- **Daily scan**: Mon-Fri at 4:30 PM ET (21:30 UTC)
- **Weekly retrain**: Sunday at 6:00 PM ET (23:00 UTC)

Set `ENABLE_SCHEDULER=1` to activate. When disabled (default), the bot only responds to slash commands.

Railway provides a persistent volume for `models/` and the SQLite database. Add it in the Railway dashboard under your service's Settings > Volumes, mounted at `/app/models`.

---

## Configuration

### Named Configs (scanner.py)

| Config | P(rally) | Comp | MaxSize | ProfitATR | TimeStop |
|--------|----------|------|---------|-----------|----------|
| `baseline` | 50% | 0.55 | 25% | 2.0x | 10 bars |
| `conservative` | 55% | 0.60 | 15% | 1.5x | 8 bars |
| `aggressive` | 40% | 0.45 | 30% | 2.5x | 12 bars |
| `concentrated` | 55% | 0.60 | 40% | 2.0x | 10 bars |

### Key Parameters (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `forward_horizon` | 10 | Label lookforward window (bars) |
| `atr_period` | 20 | ATR calculation period |
| `bb_period` | 20 | Bollinger Band period |
| `rv_period` | 20 | Realized volatility period |
| `percentile_window` | 252 | Rolling percentile lookback (~1 year) |
| `range_period` | 40 | Main range calculation period |
| `trap_range_period` | 20 | Shorter range for breakdown detection |
| `trap_lookforward` | 5 | Bars to confirm failed breakdown |
| `rv_trap_percentile` | 0.70 | Max RV percentile to qualify trap |
| `walk_forward_train_years` | 5 | Training window per fold |
| `walk_forward_test_years` | 1 | Test window per fold |
| `vol_target_k` | 0.10 | Position sizing risk scalar |
| `profit_atr_mult` | 2.0 | Take-profit in ATR multiples |
| `time_stop_bars` | 10 | Maximum holding period |
| `rv_exit_pct` | 0.80 | RV percentile for exhaustion exit |

### Pipeline Config (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_workers` | 8 | Max parallel training workers |
| `cache_dir` | `data_cache` | Parquet OHLCV cache directory |
| `cache_enabled` | True | Use disk cache for OHLCV data |
| `hmm_n_iter` | 100 | HMM EM iterations (converges in ~50-80) |
| `skip_fresh_days` | 7 | Skip retrain if model < N days old |

---

## Dependencies

- Python 3.11+
- numpy, pandas, scikit-learn, matplotlib
- yfinance (Yahoo Finance API)
- hmmlearn (Hidden Markov Models)
- joblib (model persistence)
- lxml (Wikipedia universe scraping)
- python-dotenv (environment variable management)
- pyarrow (parquet OHLCV cache)

Optional: discord.py (for Discord bot — install with `pip install -e ".[discord]"`)

Dev dependencies: pytest, ruff

Install everything with `pip install -e ".[dev]"` or `make setup`.
