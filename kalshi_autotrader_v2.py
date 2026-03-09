"""
Kalshi Mean Levels v2 Autotrader
=================================
Adapts the Mean Levels S/R strategy (CDM/PDM/CMM/PMM confluence + breaks-only)
for prediction market binary contracts.

Equity Mean Levels → Prediction Market Adaptation:
  CDM (Current Day Mean)    → CVWAP  — Volume-weighted avg price of trades in the current session window
  PDM (Previous Day Mean)   → PVWAP  — Volume-weighted avg price of trades in the prior session window
  CMM (Current Month Mean)  → LWMA   — Longer-window moving average (e.g., 100 trades)
  PMM (Previous Month Mean) → EWMA   — Exponential weighted moving average (α=0.05, all history)

Confluence = multiple anchor levels stacking within a tight band, just like
CDM+PDM+CMM stacking creates high-conviction zones on equities.

Key adaptations:
  1. VWAP-anchored levels instead of OHLC means (prediction markets have no sessions)
  2. Break-only entries (the dominant edge from backtest: 69.1% WR, 4.07 PF)
  3. Confluence scoring with weighted levels (same PMM>CMM>PDM>CDM weighting)
  4. Time-to-expiry decay — signals weaken as market approaches resolution
  5. Cross-market confluence — related markets in same event boost conviction
  6. Probability boundary awareness — mean reversion weakens near 0/100

Usage:
  python3 kalshi_autotrader_v2.py --dry-run     # Scan + simulate
  python3 kalshi_autotrader_v2.py --live         # Scan + execute real orders
  python3 kalshi_autotrader_v2.py --status       # Show state
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
from dataclasses import dataclass, field, asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kalshi_client import KalshiClient, KalshiAPIError

# ── Logging ──────────────────────────────────────────────────────────────

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotrader_v2_log.json")
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotrader_v2_state.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kalshi_v2")


# ── Configuration ────────────────────────────────────────────────────────

CONFIG = {
    # ── Mean Levels Adaptation ──
    # Session windows (in number of trades) for computing VWAP anchors
    "current_session_trades": 30,     # CVWAP: last 30 trades (≈ CDM)
    "previous_session_trades": 60,    # PVWAP: trades 31-90 (≈ PDM)
    "long_window_trades": 100,        # LWMA: last 100 trades (≈ CMM)
    "ewma_alpha": 0.05,              # EWMA decay factor (≈ PMM — slow anchor)

    # Level weights (mirrors equity strategy: PMM=4, CMM=3, PDM=2, CDM=1)
    "level_weights": {
        "EWMA": 4,    # Slowest anchor — equivalent to PMM
        "LWMA": 3,    # Long window — equivalent to CMM
        "PVWAP": 2,   # Prior session — equivalent to PDM
        "CVWAP": 1,   # Current session — equivalent to CDM
    },

    # Confluence zone detection
    "confluence_proximity_pct": 0.03,  # 3% grouping radius (wider than equities due to prediction market volatility)
    "min_confluence_score": 5,         # Minimum score to trade (matches equity strategy's high-conviction filter)

    # ── Break-Only Entries (the dominant edge) ──
    # A "break" = price closed through a confluence zone after N trades on the other side
    "break_lookback_trades": 10,       # How many recent trades to check for prior-side residency
    "min_sessions_on_other_side": 3,   # At least 3 of last 10 trades must have been on the other side
    # No bounce trades — backtest showed they're a net drag

    # ── Time-to-Expiry Decay ──
    "max_hours_to_expiry": 720,        # 30 days — beyond this, no decay applied
    "min_hours_to_expiry": 2,          # Don't trade markets expiring in < 2 hours
    "expiry_decay_power": 0.5,         # sqrt decay — signal strength * sqrt(hours_remaining / max_hours)

    # ── Risk Management ──
    "risk_pct": 0.02,                  # 2% of balance per trade (matches v2 equity backtest)
    "half_risk_below_score": 5,        # Half-size positions with score < 5 (same as equity strategy)
    "max_positions": 5,                # Max concurrent positions
    "stop_loss_cents": 8,              # 8¢ stop per contract
    "take_profit_multiple": 2.0,       # Exit at 2x risk (R:R = 1:2)
    "mean_reversion_exit": True,       # Also exit when price reverts to CVWAP

    # ── Market Filters ──
    "min_volume": 300,                 # Minimum market volume
    "max_spread_cents": 5,             # Max bid-ask spread
    "min_price": 12,                   # Skip near-zero probability markets
    "max_price": 88,                   # Skip near-certain markets
    "min_trade_count": 30,             # Need at least 30 trades for signal calculation

    # ── Execution ──
    "time_in_force": "good_till_canceled",
    "cancel_on_pause": True,
    "order_offset_cents": 1,           # Limit 1¢ inside the spread for fills

    # ── Discovery ──
    "recent_trades_fetch": 500,
    "max_markets_to_analyze": 20,
    "rate_limit_delay": 0.12,

    # ── Cross-Market Confluence ──
    "check_sibling_markets": True,     # Check other markets in the same event
    "sibling_alignment_bonus": 2,      # Add 2 to score if sibling markets confirm direction
}


# ── Data Classes ─────────────────────────────────────────────────────────

@dataclass
class MeanLevels:
    """The 4 anchor levels adapted for prediction markets."""
    CVWAP: float   # Current session VWAP (≈ CDM)
    PVWAP: float   # Previous session VWAP (≈ PDM)
    LWMA: float    # Long-window moving average (≈ CMM)
    EWMA: float    # Exponential weighted moving average (≈ PMM)

    def as_dict(self) -> Dict[str, float]:
        return {"CVWAP": self.CVWAP, "PVWAP": self.PVWAP, "LWMA": self.LWMA, "EWMA": self.EWMA}


@dataclass
class ConfluenceZone:
    """A zone where multiple mean levels stack."""
    center: float
    score: int
    levels: List[str]
    level_prices: Dict[str, float]


@dataclass
class Signal:
    ticker: str
    title: str
    side: str              # "yes" or "no"
    action: str            # "buy"
    setup_type: str        # "BREAK_LONG" or "BREAK_SHORT"
    entry_price: int       # cents
    stop_price: int        # cents
    target_price: int      # cents
    current_yes: int
    zone: Dict
    mean_levels: Dict
    confluence_score: int
    expiry_decay: float
    final_score: float
    contracts: int
    risk_per_contract: int
    reason: str


# ── State Management ─────────────────────────────────────────────────────

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"open_positions": {}, "trade_log": [], "run_count": 0, "stats": {"total_trades": 0, "wins": 0, "losses": 0}}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def append_log(entry: dict):
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    logs.append(entry)
    if len(logs) > 500:
        logs = logs[-500:]
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2, default=str)


# ── Mean Level Computation ───────────────────────────────────────────────

def compute_mean_levels(trades: List[Dict]) -> Optional[MeanLevels]:
    """
    Compute the 4 VWAP-anchored mean levels from trade history.

    trades: list of trade dicts with 'yes_price' and 'count' fields, oldest first.
    """
    if len(trades) < CONFIG["min_trade_count"]:
        return None

    # Extract price/volume pairs
    prices = [(t.get("yes_price", 50), max(t.get("count", 1), 1)) for t in trades]

    # ── CVWAP: Volume-weighted average of current session (last N trades) ──
    current_window = prices[-CONFIG["current_session_trades"]:]
    cvwap_num = sum(p * v for p, v in current_window)
    cvwap_den = sum(v for _, v in current_window)
    cvwap = cvwap_num / cvwap_den if cvwap_den > 0 else prices[-1][0]

    # ── PVWAP: Volume-weighted average of previous session ──
    prev_start = max(0, len(prices) - CONFIG["current_session_trades"] - CONFIG["previous_session_trades"])
    prev_end = len(prices) - CONFIG["current_session_trades"]
    prev_window = prices[prev_start:prev_end]
    if prev_window:
        pvwap_num = sum(p * v for p, v in prev_window)
        pvwap_den = sum(v for _, v in prev_window)
        pvwap = pvwap_num / pvwap_den if pvwap_den > 0 else cvwap
    else:
        pvwap = cvwap

    # ── LWMA: Simple moving average of last N trades ──
    long_window = prices[-CONFIG["long_window_trades"]:]
    lwma = sum(p for p, _ in long_window) / len(long_window)

    # ── EWMA: Exponential weighted moving average over all history ──
    alpha = CONFIG["ewma_alpha"]
    ewma = prices[0][0]
    for p, _ in prices[1:]:
        ewma = alpha * p + (1 - alpha) * ewma

    return MeanLevels(
        CVWAP=round(cvwap, 2),
        PVWAP=round(pvwap, 2),
        LWMA=round(lwma, 2),
        EWMA=round(ewma, 2),
    )


# ── Confluence Zone Detection ────────────────────────────────────────────

def compute_confluence_zones(levels: MeanLevels) -> List[ConfluenceZone]:
    """
    Group mean levels that stack within proximity_pct of each other.
    Exactly mirrors the equity strategy's confluence zone logic.
    """
    weights = CONFIG["level_weights"]
    items = [(getattr(levels, name), name, weights[name]) for name in weights]

    assigned = [False] * len(items)
    zones = []

    for i, (price_i, name_i, weight_i) in enumerate(items):
        if assigned[i]:
            continue

        group_prices = [price_i]
        group_names = [name_i]
        group_weights = [weight_i]
        group_map = {name_i: price_i}
        assigned[i] = True

        for j, (price_j, name_j, weight_j) in enumerate(items):
            if i == j or assigned[j]:
                continue
            if price_i > 0 and abs(price_i - price_j) / max(price_i, 1) <= CONFIG["confluence_proximity_pct"]:
                group_prices.append(price_j)
                group_names.append(name_j)
                group_weights.append(weight_j)
                group_map[name_j] = price_j
                assigned[j] = True

        zones.append(ConfluenceZone(
            center=round(sum(group_prices) / len(group_prices), 2),
            score=sum(group_weights),
            levels=group_names,
            level_prices=group_map,
        ))

    zones.sort(key=lambda z: z.score, reverse=True)
    return zones


# ── Break Detection ──────────────────────────────────────────────────────

def detect_breaks(
    trades: List[Dict],
    current_yes: int,
    zones: List[ConfluenceZone],
) -> List[Dict]:
    """
    Detect BREAK setups: price has crossed through a confluence zone after
    spending multiple trades on the other side.

    BREAK_LONG:  price is now ABOVE zone, and N of last M trades were BELOW it
                 → momentum carrying price higher, buy YES
    BREAK_SHORT: price is now BELOW zone, and N of last M trades were ABOVE it
                 → momentum carrying price lower, buy NO
    """
    if len(trades) < CONFIG["break_lookback_trades"]:
        return []

    recent_prices = [t.get("yes_price", 50) for t in trades[-CONFIG["break_lookback_trades"]:]]
    setups = []

    for zone in zones:
        if zone.score < CONFIG["min_confluence_score"]:
            continue

        center = zone.center

        # ── BREAK_LONG: current price above zone, prior trades below ──
        if current_yes > center:
            trades_below = sum(1 for p in recent_prices[:-1] if p < center)
            if trades_below >= CONFIG["min_sessions_on_other_side"]:
                risk = CONFIG["stop_loss_cents"]
                stop = max(1, int(center - risk * 0.7))  # Stop below the broken zone
                target = min(99, int(current_yes + risk * CONFIG["take_profit_multiple"]))

                setups.append({
                    "type": "BREAK_LONG",
                    "zone": {"center": center, "score": zone.score, "levels": zone.levels, "level_prices": zone.level_prices},
                    "side": "yes",
                    "entry": current_yes,
                    "stop": stop,
                    "target": target,
                    "risk_per_contract": current_yes - stop,
                    "score": zone.score,
                    "trades_on_other_side": trades_below,
                })

        # ── BREAK_SHORT: current price below zone, prior trades above ──
        elif current_yes < center:
            trades_above = sum(1 for p in recent_prices[:-1] if p > center)
            if trades_above >= CONFIG["min_sessions_on_other_side"]:
                no_price = 100 - current_yes
                risk = CONFIG["stop_loss_cents"]
                stop = max(1, int(no_price - risk * 0.7))
                target = min(99, int(no_price + risk * CONFIG["take_profit_multiple"]))

                setups.append({
                    "type": "BREAK_SHORT",
                    "zone": {"center": center, "score": zone.score, "levels": zone.levels, "level_prices": zone.level_prices},
                    "side": "no",
                    "entry": no_price,
                    "stop": stop,
                    "target": target,
                    "risk_per_contract": no_price - stop,
                    "score": zone.score,
                    "trades_on_other_side": trades_above,
                })

    # Sort by score descending
    setups.sort(key=lambda s: s["score"], reverse=True)
    return setups


# ── Expiry Decay ─────────────────────────────────────────────────────────

def compute_expiry_decay(market: Dict) -> float:
    """
    Calculate time-to-expiry decay factor (0 to 1).
    Markets near expiry get weaker signals — price is converging to 0 or 100.
    """
    close_time_str = market.get("close_time") or market.get("expiration_time", "")
    if not close_time_str:
        return 1.0  # No expiry info — full signal

    try:
        # Parse ISO timestamp
        if close_time_str.endswith("Z"):
            close_time_str = close_time_str[:-1] + "+00:00"
        close_time = datetime.datetime.fromisoformat(close_time_str)
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=datetime.timezone.utc)

        now = datetime.datetime.now(datetime.timezone.utc)
        hours_remaining = (close_time - now).total_seconds() / 3600

        if hours_remaining < CONFIG["min_hours_to_expiry"]:
            return 0.0  # Too close to expiry — don't trade

        max_hours = CONFIG["max_hours_to_expiry"]
        if hours_remaining >= max_hours:
            return 1.0  # Full signal

        # Sqrt decay: preserves most signal until final hours
        return min(1.0, (hours_remaining / max_hours) ** CONFIG["expiry_decay_power"])

    except (ValueError, TypeError):
        return 1.0


# ── Cross-Market Confluence ──────────────────────────────────────────────

def check_sibling_confluence(
    client: KalshiClient, market: Dict, direction: str
) -> int:
    """
    Check if sibling markets in the same event confirm the direction.
    Returns bonus score points.
    """
    if not CONFIG["check_sibling_markets"]:
        return 0

    event_ticker = market.get("event_ticker", "")
    current_ticker = market.get("ticker", "")
    if not event_ticker:
        return 0

    try:
        resp = client.get_markets(event_ticker=event_ticker, status="open", limit=10)
        siblings = [m for m in resp.get("markets", []) if m.get("ticker") != current_ticker]
        time.sleep(CONFIG["rate_limit_delay"])

        if not siblings:
            return 0

        # Check if sibling markets are moving in the same direction
        aligned = 0
        for sib in siblings[:5]:
            sib_yes = sib.get("yes_bid", 50)
            # For BREAK_LONG (bullish), siblings with yes > 50 confirm
            # For BREAK_SHORT (bearish), siblings with yes < 50 confirm
            if direction == "BREAK_LONG" and sib_yes > 55:
                aligned += 1
            elif direction == "BREAK_SHORT" and sib_yes < 45:
                aligned += 1

        if aligned >= 2:
            return CONFIG["sibling_alignment_bonus"]

    except KalshiAPIError:
        pass

    return 0


# ── Position Sizing ──────────────────────────────────────────────────────

def calculate_contracts(
    balance_cents: int, entry_price: int, stop_price: int, confluence_score: int
) -> int:
    """
    Fixed-fractional risk sizing, with half-size for lower confluence.
    Mirrors equity strategy: full size at score >= 5, half at score < 5.
    """
    risk_pct = CONFIG["risk_pct"]
    if confluence_score < CONFIG["half_risk_below_score"]:
        risk_pct /= 2  # Half-size for marginal confluence

    risk_amount = balance_cents * risk_pct
    risk_per_contract = abs(entry_price - stop_price)

    if risk_per_contract <= 0:
        return 0

    contracts = max(1, int(risk_amount / risk_per_contract))
    return contracts


# ── Market Discovery ─────────────────────────────────────────────────────

def discover_active_markets(client: KalshiClient) -> List[str]:
    """Find most actively traded non-parlay markets."""
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
                time.sleep(5)
                continue
            raise

    ticker_activity = defaultdict(int)
    for t in all_trades:
        tk = t.get("ticker", "")
        if not tk.startswith("KXMVE"):  # Exclude parlays
            ticker_activity[tk] += 1

    sorted_tickers = sorted(ticker_activity.items(), key=lambda x: x[1], reverse=True)
    top = [tk for tk, _ in sorted_tickers[:CONFIG["max_markets_to_analyze"]]]
    logger.info(f"Discovered {len(ticker_activity)} active markets, analyzing top {len(top)}")
    return top


def get_trade_history(client: KalshiClient, ticker: str, limit: int = 150) -> List[Dict]:
    """Fetch recent trades with full detail (price + volume)."""
    all_trades = []
    cursor = None
    remaining = limit

    while remaining > 0:
        batch = min(remaining, 200)
        try:
            resp = client.get_trades(ticker=ticker, limit=batch, cursor=cursor)
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
                time.sleep(5)
                continue
            break

    all_trades.reverse()  # Oldest first
    return all_trades


# ── Main Analysis Pipeline ───────────────────────────────────────────────

def analyze_market(
    client: KalshiClient, ticker: str, balance_cents: int
) -> Optional[Signal]:
    """
    Full Mean Levels v2 analysis pipeline for a single market:
    1. Fetch market details + trade history
    2. Compute 4 VWAP-anchored mean levels
    3. Find confluence zones
    4. Detect break setups
    5. Apply expiry decay
    6. Check cross-market confluence
    7. Score and return signal
    """
    # Get market details
    try:
        resp = client.get_market(ticker)
        market = resp.get("market", resp)
        time.sleep(CONFIG["rate_limit_delay"])
    except KalshiAPIError:
        return None

    yes_bid = market.get("yes_bid", 0)
    yes_ask = market.get("yes_ask", 0)
    volume = market.get("volume", 0)
    spread = yes_ask - yes_bid if yes_ask > yes_bid else 0
    title = market.get("title", ticker)[:80]

    # Filters
    if yes_bid < CONFIG["min_price"] or yes_bid > CONFIG["max_price"]:
        return None
    if spread > CONFIG["max_spread_cents"]:
        return None
    if volume < CONFIG["min_volume"]:
        return None

    # Get trade history
    trades = get_trade_history(client, ticker)
    if len(trades) < CONFIG["min_trade_count"]:
        return None

    # Step 1: Compute mean levels
    levels = compute_mean_levels(trades)
    if levels is None:
        return None

    logger.info(f"  {ticker}: CVWAP={levels.CVWAP} PVWAP={levels.PVWAP} LWMA={levels.LWMA} EWMA={levels.EWMA} | yes={yes_bid}¢")

    # Step 2: Find confluence zones
    zones = compute_confluence_zones(levels)
    if not zones or zones[0].score < CONFIG["min_confluence_score"]:
        return None

    # Step 3: Detect break setups
    setups = detect_breaks(trades, yes_bid, zones)
    if not setups:
        return None

    best_setup = setups[0]

    # Step 4: Expiry decay
    decay = compute_expiry_decay(market)
    if decay <= 0:
        return None

    # Step 5: Cross-market confluence bonus
    sibling_bonus = check_sibling_confluence(client, market, best_setup["type"])
    total_score = best_setup["score"] + sibling_bonus

    # Step 6: Final scoring
    # Score = confluence_score * log(volume) * expiry_decay * (1/spread)
    spread_factor = 1.0 / max(spread, 1)
    final_score = total_score * math.log1p(volume) * decay * spread_factor

    # Step 7: Position sizing
    entry = best_setup["entry"]
    stop = best_setup["stop"]
    target = best_setup["target"]
    contracts = calculate_contracts(balance_cents, entry, stop, total_score)

    direction = "Bullish break" if best_setup["type"] == "BREAK_LONG" else "Bearish break"
    zone_levels = "+".join(best_setup["zone"]["levels"])

    return Signal(
        ticker=ticker,
        title=title,
        side=best_setup["side"],
        action="buy",
        setup_type=best_setup["type"],
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        current_yes=yes_bid,
        zone=best_setup["zone"],
        mean_levels=levels.as_dict(),
        confluence_score=total_score,
        expiry_decay=round(decay, 3),
        final_score=round(final_score, 2),
        contracts=contracts,
        risk_per_contract=abs(entry - stop),
        reason=(
            f"{direction} through {zone_levels} zone @ {best_setup['zone']['center']:.0f}¢ "
            f"(score={total_score}, decay={decay:.2f}), "
            f"entry={entry}¢ stop={stop}¢ target={target}¢, "
            f"spread={spread}¢, vol={volume:,}"
        ),
    )


# ── Position Monitoring ──────────────────────────────────────────────────

def monitor_positions(client: KalshiClient, state: dict) -> List[dict]:
    """Monitor open positions for stop-loss, take-profit, or mean reversion exit."""
    actions = []
    positions = state.get("open_positions", {})

    if not positions:
        return actions

    for ticker, pos in list(positions.items()):
        try:
            m_resp = client.get_market(ticker)
            market = m_resp.get("market", m_resp)
            current_yes = market.get("yes_bid", 0)
            market_status = market.get("status", "")
            time.sleep(CONFIG["rate_limit_delay"])

            side = pos["side"]
            entry = pos["entry_price"]
            stop = pos["stop_price"]
            target = pos["target_price"]

            if side == "yes":
                current_val = current_yes
            else:
                current_val = 100 - current_yes

            pnl_per = current_val - entry
            total_pnl = pnl_per * pos["contracts"]

            action = {
                "ticker": ticker, "side": side,
                "entry": entry, "current": current_val,
                "pnl_per_contract": pnl_per,
                "total_pnl_cents": total_pnl,
                "action": "hold",
            }

            should_exit = False
            exit_reason = ""

            # Market resolved
            if market_status in ("settled", "closed", "finalized"):
                should_exit = True
                exit_reason = f"Market {market_status}"

            # Stop-loss
            elif current_val <= stop:
                should_exit = True
                exit_reason = f"Stop-loss hit ({current_val}¢ <= {stop}¢, P&L: {pnl_per}¢)"
                state["stats"]["losses"] = state["stats"].get("losses", 0) + 1

            # Take-profit target
            elif current_val >= target:
                should_exit = True
                exit_reason = f"Target hit ({current_val}¢ >= {target}¢, P&L: +{pnl_per}¢)"
                state["stats"]["wins"] = state["stats"].get("wins", 0) + 1

            # Mean reversion exit: price reverted back to CVWAP zone center
            elif CONFIG["mean_reversion_exit"]:
                zone_center = pos.get("zone_center", entry)
                if side == "yes" and current_val >= zone_center:
                    should_exit = True
                    exit_reason = f"Mean reverted to zone ({current_val}¢ >= {zone_center}¢)"
                    if pnl_per > 0:
                        state["stats"]["wins"] = state["stats"].get("wins", 0) + 1
                    else:
                        state["stats"]["losses"] = state["stats"].get("losses", 0) + 1
                elif side == "no" and (100 - current_yes) >= zone_center:
                    should_exit = True
                    exit_reason = f"Mean reverted to zone"
                    if pnl_per > 0:
                        state["stats"]["wins"] = state["stats"].get("wins", 0) + 1
                    else:
                        state["stats"]["losses"] = state["stats"].get("losses", 0) + 1

            if should_exit:
                action["action"] = "exit"
                action["reason"] = exit_reason
                logger.info(f"EXIT: {ticker} {side} — {exit_reason} (total P&L: {total_pnl}¢)")

                if market_status not in ("settled", "closed", "finalized"):
                    try:
                        sell_kwargs = {
                            "ticker": ticker,
                            "side": side,
                            "action": "sell",
                            "count": pos["contracts"],
                            "time_in_force": "immediate_or_cancel",
                        }
                        # IOC sell orders require a price — use 1¢ to fill at any available price
                        if side == "yes":
                            sell_kwargs["yes_price"] = 1
                        else:
                            sell_kwargs["no_price"] = 1
                        client.create_order(**sell_kwargs)
                        action["action"] = "exit_placed"
                        time.sleep(CONFIG["rate_limit_delay"])
                    except KalshiAPIError as e:
                        logger.error(f"Exit order failed: {e}")
                        action["action"] = "exit_failed"

                del positions[ticker]
            else:
                logger.info(f"HOLD: {ticker} {side} entry={entry}¢ current={current_val}¢ stop={stop}¢ target={target}¢ P&L={pnl_per}¢")

            actions.append(action)

        except KalshiAPIError as e:
            if e.status_code == 429:
                time.sleep(5)
            logger.error(f"Monitor error for {ticker}: {e}")

    return actions


# ── Trade Execution ──────────────────────────────────────────────────────

def execute_signals(
    client: KalshiClient, signals: List[Signal], state: dict, dry_run: bool = False
) -> List[dict]:
    """Place orders for break signals."""
    results = []
    open_count = len(state.get("open_positions", {}))
    max_new = CONFIG["max_positions"] - open_count

    if max_new <= 0:
        logger.info(f"At max positions ({CONFIG['max_positions']}), skipping.")
        return results

    for signal in signals[:max_new]:
        if signal.ticker in state.get("open_positions", {}):
            continue

        result = {
            "ticker": signal.ticker,
            "type": signal.setup_type,
            "side": signal.side,
            "contracts": signal.contracts,
            "entry": signal.entry_price,
            "stop": signal.stop_price,
            "target": signal.target_price,
            "score": signal.confluence_score,
            "final_score": signal.final_score,
        }

        if dry_run:
            result["status"] = "dry_run"
            logger.info(
                f"[DRY RUN] {signal.setup_type}: {signal.contracts}x {signal.ticker} "
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
                    "setup_type": signal.setup_type,
                    "entry_price": signal.entry_price,
                    "stop_price": signal.stop_price,
                    "target_price": signal.target_price,
                    "zone_center": signal.zone["center"],
                    "mean_levels": signal.mean_levels,
                    "confluence_score": signal.confluence_score,
                    "contracts": signal.contracts,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }

                state["stats"]["total_trades"] = state["stats"].get("total_trades", 0) + 1

                logger.info(
                    f"ORDER: {signal.setup_type} {signal.contracts}x {signal.ticker} "
                    f"{signal.side} @ {signal.entry_price}¢ → {status}"
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
    Full cycle: check balance → monitor exits → discover markets →
    compute mean levels → find confluence → detect breaks → execute.
    """
    client = KalshiClient(
        api_key_id=os.environ["KALSHI_API_KEY_ID"],
        private_key_path=os.environ.get("KALSHI_PRIVATE_KEY_PATH",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "kalshi-key.key")),
    )

    state = load_state()
    state["run_count"] = state.get("run_count", 0) + 1
    run_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

    summary = {"run_time": run_time, "run_number": state["run_count"], "dry_run": dry_run, "strategy": "mean_levels_v2"}

    # 1. Check balance
    try:
        bal = client.get_balance()
        balance_cents = bal["balance"]
        portfolio_cents = bal["portfolio_value"]
        summary["balance"] = f"${balance_cents / 100:.2f}"
        summary["portfolio_value"] = f"${portfolio_cents / 100:.2f}"
        logger.info(f"Balance: ${balance_cents / 100:.2f} | Portfolio: ${portfolio_cents / 100:.2f}")

        if balance_cents < 100 and not dry_run:
            summary["status"] = "insufficient_balance"
            save_state(state)
            append_log(summary)
            return summary
    except KalshiAPIError as e:
        summary["status"] = "error"
        summary["error"] = str(e)
        return summary

    # 2. Monitor positions
    logger.info("--- Monitoring positions ---")
    exit_actions = monitor_positions(client, state)
    summary["position_actions"] = exit_actions

    # 3. Discover active markets
    logger.info("--- Discovering active markets ---")
    tickers = discover_active_markets(client)

    # 4. Analyze each market through the Mean Levels pipeline
    logger.info("--- Mean Levels v2 Analysis ---")
    signals = []
    for ticker in tickers:
        if ticker in state.get("open_positions", {}):
            continue
        signal = analyze_market(client, ticker, balance_cents)
        if signal:
            signals.append(signal)

    signals.sort(key=lambda s: s.final_score, reverse=True)
    summary["signals_found"] = len(signals)
    summary["signals"] = [
        {
            "ticker": s.ticker, "type": s.setup_type, "side": s.side,
            "entry": s.entry_price, "stop": s.stop_price, "target": s.target_price,
            "score": s.confluence_score, "final_score": s.final_score,
            "reason": s.reason,
        }
        for s in signals
    ]

    # 5. Execute
    if signals:
        logger.info(f"--- Executing {len(signals)} signals ---")
        results = execute_signals(client, signals, state, dry_run=dry_run)
        summary["trades"] = results
    else:
        logger.info("No break signals found this run.")
        summary["trades"] = []

    summary["open_positions"] = len(state.get("open_positions", {}))
    summary["stats"] = state.get("stats", {})
    summary["status"] = "completed"

    save_state(state)
    append_log(summary)
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kalshi Mean Levels v2 Autotrader")
    parser.add_argument("--dry-run", action="store_true", help="Simulate only")
    parser.add_argument("--live", action="store_true", help="Execute real orders")
    parser.add_argument("--status", action="store_true", help="Show state")
    args = parser.parse_args()

    if args.status:
        s = load_state()
        print(json.dumps(s, indent=2, default=str))
    else:
        dry = not args.live
        if dry:
            print("Running in DRY RUN mode (use --live for real orders)\n")
        result = run(dry_run=dry)
        print("\n" + "=" * 60)
        print(json.dumps(result, indent=2, default=str))
