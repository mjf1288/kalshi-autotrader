"""
Kalshi Autotrader — Mean Reversion Bot for High-Volume Prediction Markets.

Dynamically discovers the most actively traded markets on Kalshi,
calculates mean reversion signals from recent trade history, and
executes trades when price deviates significantly from its rolling average.

Designed to run on a schedule (e.g., every hour) via cron.

Strategy:
  1. Fetch recent public trades to find the most active markets
  2. For each active market with sufficient trade history:
     - Calculate rolling mean and standard deviation of yes_price
     - If price is >N std devs below mean → BUY YES (oversold)
     - If price is >N std devs above mean → BUY NO (overbought)
  3. Position size based on % of account balance
  4. Monitor existing positions for stop-loss / take-profit exits

Risk Controls:
  - Max 1% of balance per trade
  - Max 5 concurrent positions
  - 10¢ stop-loss per contract
  - Exit when price reverts 50%+ toward mean
  - Only trade markets with bid-ask spread ≤ 5¢
  - Skip markets near settlement (yes_price < 5 or > 95)
"""

import os
import sys
import json
import math
import uuid
import time
import logging
import datetime
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kalshi_client import KalshiClient, KalshiAPIError

# ── Logging ──────────────────────────────────────────────────────────────

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotrader_log.json")
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotrader_state.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kalshi_autotrader")


# ── Configuration ────────────────────────────────────────────────────────

CONFIG = {
    # Strategy
    "lookback_trades": 50,          # Number of recent trades for mean calculation
    "entry_z_threshold": 2.0,       # Std devs from mean to trigger entry
    "exit_reversion_pct": 0.50,     # Exit when 50% of deviation has reverted
    "min_std_dev": 2.0,             # Minimum std dev to consider (avoid flat markets)

    # Risk management
    "risk_pct": 0.01,               # 1% of balance per trade
    "max_positions": 5,             # Max concurrent open positions
    "stop_loss_cents": 10,          # Max loss per contract in cents
    "max_spread_cents": 5,          # Max bid-ask spread to enter
    "min_price": 10,                # Skip markets with yes_price < 10
    "max_price": 90,                # Skip markets with yes_price > 90
    "min_volume": 500,              # Minimum market volume
    "min_trade_history": 20,        # Minimum trades needed for signal calculation

    # Execution
    "time_in_force": "good_till_canceled",
    "cancel_on_pause": True,
    "order_offset_cents": 1,        # Place limit 1¢ better than current price for fills

    # Discovery
    "recent_trades_fetch": 500,     # How many recent trades to scan for market discovery
    "max_markets_to_analyze": 20,   # Max markets to deep-analyze per run
    "rate_limit_delay": 0.12,       # Seconds between API calls (Basic: 20/sec)
}


# ── Data Classes ─────────────────────────────────────────────────────────

@dataclass
class Signal:
    ticker: str
    title: str
    side: str           # "yes" or "no"
    action: str         # "buy"
    entry_price: int    # limit price in cents
    current_yes: int    # current yes_price
    mean_price: float
    std_dev: float
    z_score: float
    volume: int
    bid: int
    ask: int
    spread: int
    contracts: int
    max_loss_cents: int
    score: float
    reason: str


# ── Helpers ──────────────────────────────────────────────────────────────

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"open_positions": {}, "trade_log": [], "run_count": 0}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def append_log(entry: dict):
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    logs.append(entry)
    # Keep last 500 entries
    if len(logs) > 500:
        logs = logs[-500:]
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2, default=str)


# ── Core Functions ───────────────────────────────────────────────────────

def discover_active_markets(client: KalshiClient) -> List[str]:
    """
    Find the most actively traded non-parlay markets by scanning recent trades.
    Returns list of tickers sorted by trade frequency.
    """
    all_trades = []
    cursor = None
    remaining = CONFIG["recent_trades_fetch"]

    while remaining > 0:
        batch = min(remaining, 200)
        try:
            resp = client.get_trades(limit=batch, cursor=cursor)
            trades = resp.get("trades", [])
            if not trades:
                break
            all_trades.extend(trades)
            cursor = resp.get("cursor", "")
            remaining -= len(trades)
            if not cursor:
                break
            time.sleep(CONFIG["rate_limit_delay"])
        except KalshiAPIError as e:
            if e.status_code == 429:
                logger.warning("Rate limited during discovery, waiting 5s...")
                time.sleep(5)
                continue
            raise

    # Aggregate by ticker — exclude parlay markets (KXMVE*)
    ticker_activity = defaultdict(lambda: {"count": 0, "volume": 0, "prices": []})
    for t in all_trades:
        tk = t.get("ticker", "")
        if tk.startswith("KXMVE"):
            continue
        ticker_activity[tk]["count"] += 1
        ticker_activity[tk]["volume"] += t.get("count", 0)
        ticker_activity[tk]["prices"].append(t.get("yes_price", 50))

    # Sort by trade count
    sorted_tickers = sorted(
        ticker_activity.items(),
        key=lambda x: x[1]["count"],
        reverse=True,
    )

    # Return top N tickers
    top = [tk for tk, _ in sorted_tickers[: CONFIG["max_markets_to_analyze"]]]
    logger.info(f"Discovered {len(ticker_activity)} active markets, analyzing top {len(top)}")
    return top


def get_trade_history(client: KalshiClient, ticker: str) -> List[int]:
    """Fetch recent trade prices for a market (oldest first)."""
    prices = []
    cursor = None
    remaining = CONFIG["lookback_trades"]

    while remaining > 0:
        batch = min(remaining, 200)
        try:
            resp = client.get_trades(ticker=ticker, limit=batch, cursor=cursor)
            trades = resp.get("trades", [])
            if not trades:
                break
            for t in trades:
                p = t.get("yes_price", 0)
                if p > 0:
                    prices.append(p)
            cursor = resp.get("cursor", "")
            remaining -= len(trades)
            if not cursor:
                break
            time.sleep(CONFIG["rate_limit_delay"])
        except KalshiAPIError as e:
            if e.status_code == 429:
                time.sleep(5)
                continue
            break

    prices.reverse()  # oldest first
    return prices


def calculate_signal(
    prices: List[int], current_yes: int, market: dict, balance_cents: int
) -> Optional[Signal]:
    """Calculate mean reversion signal for a market."""
    lookback = min(len(prices), CONFIG["lookback_trades"])
    if lookback < CONFIG["min_trade_history"]:
        return None

    window = prices[-lookback:]
    mean = sum(window) / len(window)
    variance = sum((p - mean) ** 2 for p in window) / len(window)
    std_dev = math.sqrt(variance) if variance > 0 else 0

    if std_dev < CONFIG["min_std_dev"]:
        return None  # Market is too flat for mean reversion

    z_score = (current_yes - mean) / std_dev

    # Check thresholds
    threshold = CONFIG["entry_z_threshold"]
    if abs(z_score) < threshold:
        return None  # Not enough deviation

    ticker = market.get("ticker", "")
    title = market.get("title", ticker)
    bid = market.get("yes_bid", 0)
    ask = market.get("yes_ask", 0)
    volume = market.get("volume", 0)
    spread = ask - bid if ask > bid else 0

    # Filters
    if spread > CONFIG["max_spread_cents"]:
        return None
    if current_yes < CONFIG["min_price"] or current_yes > CONFIG["max_price"]:
        return None
    if volume < CONFIG["min_volume"]:
        return None

    # Determine direction
    if z_score < -threshold:
        # Oversold → Buy YES
        side = "yes"
        # Place limit at ask (or 1¢ below ask for better fill)
        entry_price = max(bid, ask - CONFIG["order_offset_cents"])
        max_loss = min(entry_price, CONFIG["stop_loss_cents"])
    else:
        # Overbought → Buy NO (which means selling YES exposure)
        side = "no"
        no_ask = 100 - bid  # no_ask = 100 - yes_bid
        no_bid = 100 - ask  # no_bid = 100 - yes_ask
        entry_price = max(no_bid, no_ask - CONFIG["order_offset_cents"])
        max_loss = min(entry_price, CONFIG["stop_loss_cents"])

    if max_loss <= 0:
        return None

    # Position sizing
    risk_amount = balance_cents * CONFIG["risk_pct"]
    contracts = max(1, int(risk_amount / max_loss))

    # Score: |z| * log(volume) * (1 / spread) — strong signal + liquid + tight
    spread_factor = 1.0 / max(spread, 1)
    score = abs(z_score) * math.log1p(volume) * spread_factor

    direction = "Oversold" if side == "yes" else "Overbought"
    return Signal(
        ticker=ticker,
        title=title[:80],
        side=side,
        action="buy",
        entry_price=entry_price,
        current_yes=current_yes,
        mean_price=round(mean, 2),
        std_dev=round(std_dev, 2),
        z_score=round(z_score, 3),
        volume=volume,
        bid=bid,
        ask=ask,
        spread=spread,
        contracts=contracts,
        max_loss_cents=max_loss,
        score=round(score, 2),
        reason=f"{direction}: yes={current_yes}¢ vs mean={mean:.1f}¢ ({abs(z_score):.1f}σ), spread={spread}¢, vol={volume:,}",
    )


def monitor_positions(client: KalshiClient, state: dict) -> List[dict]:
    """Check open positions for exit conditions."""
    actions = []
    positions = state.get("open_positions", {})

    if not positions:
        return actions

    for ticker, pos in list(positions.items()):
        try:
            market = client.get_market(ticker)
            m = market.get("market", market)
            current_yes = m.get("yes_bid", 0)  # use bid for conservative valuation
            market_status = m.get("status", "")
            time.sleep(CONFIG["rate_limit_delay"])

            side = pos["side"]
            entry = pos["entry_price"]
            mean_target = pos["mean_price"]

            if side == "yes":
                current_val = current_yes
            else:
                current_val = 100 - current_yes

            pnl_per = current_val - entry
            total_pnl = pnl_per * pos["contracts"]

            action = {"ticker": ticker, "side": side, "pnl_cents": total_pnl, "action": "hold"}

            should_exit = False
            exit_reason = ""

            # Market settled/closed
            if market_status in ("settled", "closed", "finalized"):
                should_exit = True
                exit_reason = f"Market {market_status}"

            # Stop-loss
            elif pnl_per <= -CONFIG["stop_loss_cents"]:
                should_exit = True
                exit_reason = f"Stop-loss ({pnl_per}¢/contract)"

            # Mean reversion exit
            else:
                original_dev = abs(entry - mean_target)
                current_dev = abs(current_val - mean_target)
                if original_dev > 0:
                    reversion = 1 - (current_dev / original_dev)
                    if reversion >= CONFIG["exit_reversion_pct"]:
                        should_exit = True
                        exit_reason = f"Mean reverted {reversion:.0%}"

            if should_exit:
                action["action"] = "exit"
                action["reason"] = exit_reason
                logger.info(f"EXIT: {ticker} {side} — {exit_reason} (P&L: {total_pnl}¢)")

                if market_status not in ("settled", "closed", "finalized"):
                    try:
                        client.create_order(
                            ticker=ticker,
                            side=side,
                            action="sell",
                            count=pos["contracts"],
                            reduce_only=True,
                            time_in_force="immediate_or_cancel",
                        )
                        action["action"] = "exit_placed"
                        time.sleep(CONFIG["rate_limit_delay"])
                    except KalshiAPIError as e:
                        logger.error(f"Exit order failed: {e}")
                        action["action"] = "exit_failed"

                del positions[ticker]
            else:
                logger.info(f"HOLD: {ticker} {side} entry={entry}¢ current={current_val}¢ P&L={pnl_per}¢/ct")

            actions.append(action)

        except KalshiAPIError as e:
            if e.status_code == 429:
                time.sleep(5)
            logger.error(f"Error checking {ticker}: {e}")

    return actions


def execute_signals(
    client: KalshiClient, signals: List[Signal], state: dict, dry_run: bool = False
) -> List[dict]:
    """Place orders for top signals."""
    results = []
    open_count = len(state.get("open_positions", {}))
    max_new = CONFIG["max_positions"] - open_count

    if max_new <= 0:
        logger.info(f"At max positions ({CONFIG['max_positions']}), skipping entries.")
        return results

    for signal in signals[:max_new]:
        # Skip if we already have a position in this market
        if signal.ticker in state.get("open_positions", {}):
            continue

        result = {
            "ticker": signal.ticker,
            "side": signal.side,
            "contracts": signal.contracts,
            "entry_price": signal.entry_price,
            "score": signal.score,
        }

        if dry_run:
            result["status"] = "dry_run"
            logger.info(
                f"[DRY RUN] {signal.action} {signal.contracts}x {signal.ticker} "
                f"{signal.side} @ {signal.entry_price}¢ — {signal.reason}"
            )
        else:
            try:
                cid = str(uuid.uuid4())
                order_kwargs = {
                    "ticker": signal.ticker,
                    "side": signal.side,
                    "action": "buy",
                    "count": signal.contracts,
                    "client_order_id": cid,
                    "time_in_force": CONFIG["time_in_force"],
                    "cancel_order_on_pause": CONFIG["cancel_on_pause"],
                }
                if signal.side == "yes":
                    order_kwargs["yes_price"] = signal.entry_price
                else:
                    order_kwargs["no_price"] = signal.entry_price

                resp = client.create_order(**order_kwargs)
                order = resp.get("order", {})
                oid = order.get("order_id", "")
                status = order.get("status", "submitted")

                result["status"] = status
                result["order_id"] = oid

                state.setdefault("open_positions", {})[signal.ticker] = {
                    "order_id": oid,
                    "client_order_id": cid,
                    "side": signal.side,
                    "entry_price": signal.entry_price,
                    "mean_price": signal.mean_price,
                    "contracts": signal.contracts,
                    "z_score": signal.z_score,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }

                logger.info(
                    f"ORDER: {signal.contracts}x {signal.ticker} {signal.side} "
                    f"@ {signal.entry_price}¢ → {status} (id={oid})"
                )
                time.sleep(CONFIG["rate_limit_delay"])

            except KalshiAPIError as e:
                logger.error(f"Order failed: {e}")
                result["status"] = "error"
                result["error"] = str(e)

        results.append(result)

    return results


# ── Main Run ─────────────────────────────────────────────────────────────

def run(dry_run: bool = False) -> dict:
    """
    Execute one full cycle of the autotrader:
    1. Check balance
    2. Monitor existing positions (exits)
    3. Discover active markets
    4. Calculate signals
    5. Execute new trades

    Returns summary dict.
    """
    client = KalshiClient(
        api_key_id=os.environ.get("KALSHI_API_KEY_ID", "7d3dd187-0c77-4650-b100-3ae589ef8098"),
        private_key_path=os.environ.get("KALSHI_PRIVATE_KEY_PATH",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "kalshi-key.key")),
    )

    state = load_state()
    state["run_count"] = state.get("run_count", 0) + 1
    run_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

    summary = {
        "run_time": run_time,
        "run_number": state["run_count"],
        "dry_run": dry_run,
    }

    # 1. Check balance
    try:
        bal = client.get_balance()
        balance_cents = bal["balance"]
        portfolio_cents = bal["portfolio_value"]
        summary["balance"] = f"${balance_cents / 100:.2f}"
        summary["portfolio_value"] = f"${portfolio_cents / 100:.2f}"
        logger.info(f"Balance: ${balance_cents / 100:.2f} | Portfolio: ${portfolio_cents / 100:.2f}")

        if balance_cents < 100 and not dry_run:  # Less than $1
            summary["status"] = "insufficient_balance"
            summary["message"] = f"Balance too low (${balance_cents/100:.2f}). Need at least $1.00."
            logger.warning(summary["message"])
            save_state(state)
            append_log(summary)
            return summary
    except KalshiAPIError as e:
        summary["status"] = "error"
        summary["error"] = f"Balance check failed: {e}"
        logger.error(summary["error"])
        return summary

    # 2. Monitor existing positions
    logger.info("--- Monitoring positions ---")
    exit_actions = monitor_positions(client, state)
    summary["position_actions"] = exit_actions

    # 3. Discover active markets
    logger.info("--- Discovering active markets ---")
    active_tickers = discover_active_markets(client)

    # 4. Analyze and generate signals
    logger.info("--- Analyzing signals ---")
    signals = []
    for ticker in active_tickers:
        # Skip markets we already hold
        if ticker in state.get("open_positions", {}):
            continue

        try:
            # Get market details
            m = client.get_market(ticker)
            market = m.get("market", m)
            current_yes = market.get("yes_bid", 0)
            time.sleep(CONFIG["rate_limit_delay"])

            if current_yes <= 0:
                continue

            # Get trade history
            prices = get_trade_history(client, ticker)

            # Calculate signal
            signal = calculate_signal(prices, current_yes, market, balance_cents)
            if signal:
                signals.append(signal)
                logger.info(f"  SIGNAL: {signal.reason}")

        except KalshiAPIError as e:
            if e.status_code == 429:
                time.sleep(5)
            continue

    # Sort by score
    signals.sort(key=lambda s: s.score, reverse=True)
    summary["signals_found"] = len(signals)
    summary["signals"] = [
        {"ticker": s.ticker, "side": s.side, "price": s.entry_price,
         "z_score": s.z_score, "score": s.score, "reason": s.reason}
        for s in signals
    ]

    # 5. Execute
    if signals:
        logger.info(f"--- Executing {len(signals)} signals ---")
        trade_results = execute_signals(client, signals, state, dry_run=dry_run)
        summary["trades"] = trade_results
    else:
        logger.info("No signals found this run.")
        summary["trades"] = []

    summary["open_positions"] = len(state.get("open_positions", {}))
    summary["status"] = "completed"

    save_state(state)
    append_log(summary)

    return summary


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kalshi Autotrader")
    parser.add_argument("--dry-run", action="store_true", help="Simulate only")
    parser.add_argument("--live", action="store_true", help="Execute real orders")
    parser.add_argument("--status", action="store_true", help="Show current state")
    args = parser.parse_args()

    if args.status:
        state = load_state()
        print(json.dumps(state, indent=2, default=str))
    else:
        dry_run = not args.live
        if dry_run:
            print("Running in DRY RUN mode (use --live for real orders)")
        result = run(dry_run=dry_run)
        print("\n" + "=" * 60)
        print(json.dumps(result, indent=2, default=str))
