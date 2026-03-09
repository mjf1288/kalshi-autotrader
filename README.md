# Kalshi Mean Reversion Autotrader

Automated trading bot for [Kalshi](https://kalshi.com) prediction markets using a Mean Levels confluence strategy adapted from equity S/R trading.

## Strategy Overview

### v1 — Simple Mean Reversion (`kalshi_trader.py` + `kalshi_autotrader.py`)
- Rolling mean + standard deviation bands on trade history
- Entry when price deviates 2σ from the mean (oversold/overbought)
- Exit at 50% reversion toward mean or stop-loss
- Position sizing: 1% of balance per trade

### v2 — Mean Levels Confluence (`kalshi_autotrader_v2.py`)
Adapts the equity Mean Levels S/R strategy for binary prediction markets:

| Equity Level | Prediction Market Equivalent | Weight |
|---|---|---|
| CDM (Current Day Mean) | CVWAP — Current session VWAP | 1 |
| PDM (Previous Day Mean) | PVWAP — Previous session VWAP | 2 |
| CMM (Current Month Mean) | LWMA — Long-window moving average | 3 |
| PMM (Previous Month Mean) | EWMA — Exponential weighted MA | 4 |

**Key features:**
- **Confluence scoring** — Multiple levels stacking within 3% = high conviction
- **Break-only entries** — The dominant edge from backtesting (69.1% WR, 4.07 PF on equities)
- **Time-to-expiry decay** — Signal strength diminishes as market approaches resolution
- **Cross-market confluence** — Related markets in same event boost conviction
- **Probability boundary awareness** — Mean reversion weakens near 0/100

## Files

| File | Description |
|---|---|
| `kalshi_client.py` | Authenticated REST client with RSA-PSS signing |
| `kalshi_trader.py` | v1 mean reversion trading engine |
| `kalshi_autotrader.py` | v1 automated scanner + executor |
| `kalshi_autotrader_v2.py` | v2 Mean Levels confluence autotrader |

## Setup

### Prerequisites
```bash
pip install requests cryptography
```

### API Keys
1. Log in to [Kalshi](https://kalshi.com) → Account & Security → API Keys → Create Key
2. Save the API Key ID (UUID) and download the `.key` private key file

### Environment Variables
```bash
export KALSHI_API_KEY_ID="your-api-key-uuid"
export KALSHI_PRIVATE_KEY_PATH="/path/to/kalshi-key.key"
```

## Usage

### v2 Autotrader (Recommended)
```bash
# Dry run — scan and simulate
python3 kalshi_autotrader_v2.py --dry-run

# Live trading
python3 kalshi_autotrader_v2.py --live

# Check state
python3 kalshi_autotrader_v2.py --status
```

### v1 Trader
```bash
# Scan markets
python3 kalshi_trader.py --scan

# Execute with dry run
python3 kalshi_trader.py --execute --dry-run

# Monitor positions
python3 kalshi_trader.py --monitor

# Portfolio status
python3 kalshi_trader.py --status
```

## Configuration (v2)

Key parameters in `kalshi_autotrader_v2.py`:

| Parameter | Default | Description |
|---|---|---|
| `risk_pct` | 0.02 | 2% of balance per trade |
| `min_confluence_score` | 5 | Minimum score to trade |
| `max_positions` | 5 | Max concurrent positions |
| `stop_loss_cents` | 8 | 8¢ stop per contract |
| `take_profit_multiple` | 2.0 | 2x risk target (R:R = 1:2) |
| `min_volume` | 300 | Minimum market volume |
| `max_spread_cents` | 5 | Max bid-ask spread |
| `min_hours_to_expiry` | 2 | Don't trade markets expiring in < 2h |

## Risk Disclaimer

This software is for educational and informational purposes only. Trading prediction markets involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.
