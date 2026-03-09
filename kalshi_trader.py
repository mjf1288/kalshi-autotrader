"""
Kalshi Mean Reversion Trader — Automated trading engine for prediction markets.

Adapts the IBS + RSI-2 mean reversion concept for binary event contracts.
In prediction markets, price = implied probability (1-99 cents).
Mean reversion signals fire when price deviates from its rolling average,
suggesting the market has overreacted and will revert.

Usage:
    from kalshi_trader import KalshiTrader

    trader = KalshiTrader(
        api_key_id="your-api-key-id",
        private_key_path="/path/to/kalshi-key.key",
        config={
            "risk_pct": 0.01,          # 1% of balance per trade
            "lookback": 20,            # 20-period moving average
            "entry_threshold": 2.0,    # 2 std devs from mean to enter
            "exit_threshold": 0.5,     # exit when deviation closes by 50%
            "stop_loss_cents": 10,     # max loss per contract in cents
            "max_positions": 5,        # max concurrent positions
            "min_volume": 100,         # minimum market volume to consider
            "min_spread_cents": 2,     # minimum bid-ask spread
        }
    )

    # Scan for opportunities
    signals = trader.scan_markets(series_ticker="KXHIGHNY")

    # Execute trades (dry run by default)
    results = trader.execute_signals(signals, dry_run=True)

    # Monitor positions
    trader.monitor_positions()

Command-line usage:
    python kalshi_trader.py --scan                          # Scan all open markets
    python kalshi_trader.py --scan --series KXHIGHNY        # Scan specific series
    python kalshi_trader.py --execute                       # Execute with live orders
    python kalshi_trader.py --execute --dry-run             # Dry run (no real orders)
    python kalshi_trader.py --monitor                       # Monitor open positions
    python kalshi_trader.py --status                        # Portfolio status
    python kalshi_trader.py --close-all                     # Close all positions
"""

import os
import sys
import json
import math
import uuid
import logging
import argparse
import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict

# Ensure kalshi_client is importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kalshi_client import KalshiClient, KalshiAPIError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kalshi_trader")

# ── Configuration ────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "risk_pct": 0.01,           # Risk 1% of account balance per trade
    "lookback": 20,             # Rolling mean lookback periods
    "entry_threshold": 2.0,     # Std dev multiplier for entry signal
    "exit_threshold": 0.5,      # Close when deviation retracts by this fraction
    "stop_loss_cents": 10,      # Max loss per contract before stop-out (cents)
    "take_profit_cents": None,  # Optional take-profit in cents (None = revert to mean)
    "max_positions": 5,         # Max concurrent open positions
    "min_volume": 100,          # Min traded volume to consider a market
    "min_liquidity_depth": 5,   # Min contracts on best bid/ask
    "time_in_force": "good_till_canceled",  # Default TIF for orders
    "cancel_on_pause": True,    # Cancel orders if exchange pauses
}


@dataclass
class Signal:
    """A trading signal for a specific market."""
    ticker: str
    title: str
    side: str           # "yes" or "no"
    action: str         # "buy" or "sell"
    entry_price: int    # cents (1-99)
    mean_price: float   # rolling mean price
    std_dev: float      # price standard deviation
    deviation: float    # how many std devs from mean
    volume: int         # market volume
    contracts: int      # suggested contract count
    max_loss_cents: int # max loss per contract
    score: float        # signal strength score (higher = stronger)
    reason: str         # human-readable signal description


@dataclass
class TradeRecord:
    """Record of an executed or planned trade."""
    signal: Signal
    order_id: Optional[str] = None
    status: str = "planned"    # planned, submitted, filled, canceled, stopped
    fill_price: Optional[int] = None
    fill_count: Optional[int] = None
    pnl_cents: Optional[int] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()


class KalshiTrader:
    """Mean reversion trading engine for Kalshi prediction markets."""

    def __init__(
        self,
        api_key_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        config: Optional[Dict] = None,
        state_file: str = "kalshi_trader_state.json",
    ):
        self.client = KalshiClient(
            api_key_id=api_key_id,
            private_key_path=private_key_path,
        )
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.state_file = state_file
        self.state = self._load_state()

    # ── State Management ─────────────────────────────────────────────────

    def _load_state(self) -> Dict:
        """Load persisted trader state."""
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {"trades": [], "open_positions": {}, "stats": {}}

    def _save_state(self):
        """Persist trader state."""
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    # ── Market Analysis ──────────────────────────────────────────────────

    def _get_trade_history(self, ticker: str, limit: int = 200) -> List[int]:
        """
        Fetch recent trade prices for a market.

        Returns list of yes_price values (cents) from oldest to newest.
        """
        prices = []
        cursor = None
        remaining = limit

        while remaining > 0:
            batch_size = min(remaining, 200)
            resp = self.client.get_trades(
                ticker=ticker, limit=batch_size, cursor=cursor
            )
            trades = resp.get("trades", [])
            if not trades:
                break
            for t in trades:
                prices.append(t.get("yes_price", 50))
            cursor = resp.get("cursor", "")
            remaining -= len(trades)
            if not cursor:
                break

        # Reverse so oldest is first
        prices.reverse()
        return prices

    def _calculate_mean_reversion_signal(
        self, prices: List[int], current_price: int
    ) -> Tuple[Optional[str], float, float, float]:
        """
        Calculate mean reversion signal from price history.

        Returns:
            (direction, mean, std_dev, deviation_z_score)
            direction: "buy_yes" if oversold, "buy_no" if overbought, None if no signal
        """
        lookback = self.config["lookback"]
        threshold = self.config["entry_threshold"]

        if len(prices) < lookback:
            return None, 0.0, 0.0, 0.0

        # Use the last `lookback` prices
        window = prices[-lookback:]
        mean = sum(window) / len(window)
        variance = sum((p - mean) ** 2 for p in window) / len(window)
        std_dev = math.sqrt(variance) if variance > 0 else 1.0

        # Z-score: how far current price is from mean in std dev units
        z_score = (current_price - mean) / std_dev if std_dev > 0 else 0.0

        if z_score < -threshold:
            return "buy_yes", mean, std_dev, z_score   # Price too low → buy YES
        elif z_score > threshold:
            return "buy_no", mean, std_dev, z_score    # Price too high → buy NO
        else:
            return None, mean, std_dev, z_score

    def _check_orderbook_liquidity(self, ticker: str) -> Tuple[bool, int, int]:
        """
        Check if market has sufficient liquidity.

        Returns:
            (is_liquid, best_yes_bid_size, best_no_bid_size)
        """
        try:
            ob = self.client.get_orderbook(ticker)
            orderbook = ob.get("orderbook", {})
            yes_bids = orderbook.get("yes", [])
            no_bids = orderbook.get("no", [])

            best_yes_size = yes_bids[0][1] if yes_bids else 0
            best_no_size = no_bids[0][1] if no_bids else 0

            min_depth = self.config["min_liquidity_depth"]
            is_liquid = best_yes_size >= min_depth or best_no_size >= min_depth

            return is_liquid, best_yes_size, best_no_size
        except KalshiAPIError:
            return False, 0, 0

    def _calculate_position_size(
        self, entry_price: int, side: str, balance_cents: int
    ) -> int:
        """
        Calculate position size based on risk percentage.

        For buying YES at price P: max loss per contract = P cents
        For buying NO at price (100-P): max loss per contract = (100-P) cents
        """
        risk_amount = balance_cents * self.config["risk_pct"]
        stop_loss = self.config["stop_loss_cents"]

        if side == "yes":
            max_loss_per_contract = min(entry_price, stop_loss)
        else:
            max_loss_per_contract = min(100 - entry_price, stop_loss)

        if max_loss_per_contract <= 0:
            return 0

        contracts = int(risk_amount / max_loss_per_contract)
        return max(1, contracts)  # At least 1 contract

    # ── Market Scanning ──────────────────────────────────────────────────

    def scan_markets(
        self,
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        tickers: Optional[List[str]] = None,
    ) -> List[Signal]:
        """
        Scan markets for mean reversion trading signals.

        Args:
            series_ticker: Scan all open markets in this series
            event_ticker: Scan all markets in this event
            tickers: Scan specific market tickers

        Returns:
            List of Signal objects sorted by score (highest first)
        """
        signals = []

        # Get markets to scan
        if tickers:
            markets = []
            for t in tickers:
                try:
                    m = self.client.get_market(t)
                    markets.append(m.get("market", m))
                except KalshiAPIError as e:
                    logger.warning(f"Could not fetch market {t}: {e}")
        else:
            markets = self.client.get_all_markets_paginated(
                series_ticker=series_ticker,
                event_ticker=event_ticker,
                status="open",
            )

        logger.info(f"Scanning {len(markets)} markets...")

        # Get balance for position sizing
        try:
            balance = self.client.get_balance()
            balance_cents = balance["balance"]
        except KalshiAPIError:
            logger.error("Could not fetch balance. Using 0 for position sizing.")
            balance_cents = 0

        for market in markets:
            ticker = market.get("ticker", "")
            title = market.get("title", ticker)
            volume = market.get("volume", 0)
            yes_price = market.get("yes_price", 50)

            # Filter: minimum volume
            if volume < self.config["min_volume"]:
                continue

            # Filter: skip extreme prices (< 5 or > 95 — too close to settlement)
            if yes_price < 5 or yes_price > 95:
                continue

            # Get trade history
            prices = self._get_trade_history(ticker, limit=self.config["lookback"] * 2)
            if len(prices) < self.config["lookback"]:
                logger.debug(f"  {ticker}: insufficient trade history ({len(prices)} trades)")
                continue

            # Calculate signal
            direction, mean, std_dev, z_score = self._calculate_mean_reversion_signal(
                prices, yes_price
            )

            if direction is None:
                continue

            # Check liquidity
            is_liquid, yes_depth, no_depth = self._check_orderbook_liquidity(ticker)
            if not is_liquid:
                logger.debug(f"  {ticker}: insufficient liquidity")
                continue

            # Build signal
            if direction == "buy_yes":
                side = "yes"
                entry_price = yes_price
                max_loss = min(entry_price, self.config["stop_loss_cents"])
            else:
                side = "no"
                entry_price = 100 - yes_price
                max_loss = min(entry_price, self.config["stop_loss_cents"])

            contracts = self._calculate_position_size(
                entry_price, side, balance_cents
            )

            # Score: |z_score| * log(volume) — strong deviation in liquid markets
            score = abs(z_score) * math.log1p(volume)

            signal = Signal(
                ticker=ticker,
                title=title,
                side=side,
                action="buy",
                entry_price=entry_price,
                mean_price=round(mean, 2),
                std_dev=round(std_dev, 2),
                deviation=round(z_score, 3),
                volume=volume,
                contracts=contracts,
                max_loss_cents=max_loss,
                score=round(score, 2),
                reason=(
                    f"{'Oversold' if side == 'yes' else 'Overbought'}: "
                    f"price={yes_price}¢ vs mean={mean:.1f}¢ "
                    f"({abs(z_score):.1f}σ deviation), "
                    f"volume={volume}"
                ),
            )
            signals.append(signal)
            logger.info(f"  SIGNAL: {ticker} — {signal.reason}")

        # Sort by score descending
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals

    # ── Trade Execution ──────────────────────────────────────────────────

    def execute_signals(
        self, signals: List[Signal], dry_run: bool = True
    ) -> List[TradeRecord]:
        """
        Execute trading signals by placing orders.

        Args:
            signals: List of Signal objects to execute
            dry_run: If True, only simulate (no real orders placed)

        Returns:
            List of TradeRecord objects
        """
        records = []

        # Check how many positions we already have
        existing_positions = len(self.state.get("open_positions", {}))
        max_new = self.config["max_positions"] - existing_positions

        if max_new <= 0:
            logger.warning(
                f"Already at max positions ({self.config['max_positions']}). "
                "No new trades."
            )
            return records

        # Only take top N signals
        signals_to_execute = signals[:max_new]

        for signal in signals_to_execute:
            record = TradeRecord(signal=signal)

            if dry_run:
                logger.info(
                    f"[DRY RUN] Would place: {signal.action} {signal.contracts}x "
                    f"{signal.ticker} {signal.side} @ {signal.entry_price}¢ "
                    f"(score={signal.score})"
                )
                record.status = "dry_run"
                records.append(record)
                continue

            # Place real order
            try:
                client_order_id = str(uuid.uuid4())

                order_kwargs = {
                    "ticker": signal.ticker,
                    "side": signal.side,
                    "action": signal.action,
                    "count": signal.contracts,
                    "client_order_id": client_order_id,
                    "time_in_force": self.config["time_in_force"],
                    "cancel_order_on_pause": self.config["cancel_on_pause"],
                }

                if signal.side == "yes":
                    order_kwargs["yes_price"] = signal.entry_price
                else:
                    order_kwargs["no_price"] = signal.entry_price

                resp = self.client.create_order(**order_kwargs)
                order = resp.get("order", {})
                record.order_id = order.get("order_id", "")
                record.status = order.get("status", "submitted")
                record.fill_count = order.get("fill_count", 0)

                # Track open position
                self.state["open_positions"][signal.ticker] = {
                    "order_id": record.order_id,
                    "client_order_id": client_order_id,
                    "side": signal.side,
                    "entry_price": signal.entry_price,
                    "mean_price": signal.mean_price,
                    "contracts": signal.contracts,
                    "stop_loss": signal.entry_price - self.config["stop_loss_cents"]
                    if signal.side == "yes"
                    else signal.entry_price - self.config["stop_loss_cents"],
                    "take_profit_price": round(signal.mean_price),
                    "status": record.status,
                    "timestamp": record.timestamp,
                }

                logger.info(
                    f"ORDER PLACED: {signal.action} {signal.contracts}x "
                    f"{signal.ticker} {signal.side} @ {signal.entry_price}¢ "
                    f"→ order_id={record.order_id}, status={record.status}"
                )

            except KalshiAPIError as e:
                logger.error(f"Order failed for {signal.ticker}: {e}")
                record.status = "error"

            records.append(record)

        # Save state
        self.state["trades"].extend(
            [
                {
                    "ticker": r.signal.ticker,
                    "side": r.signal.side,
                    "action": r.signal.action,
                    "entry_price": r.signal.entry_price,
                    "contracts": r.signal.contracts,
                    "order_id": r.order_id,
                    "status": r.status,
                    "score": r.signal.score,
                    "timestamp": r.timestamp,
                }
                for r in records
            ]
        )
        self._save_state()

        return records

    # ── Position Monitoring ──────────────────────────────────────────────

    def monitor_positions(self) -> List[Dict]:
        """
        Check open positions and manage exits (stop-loss / take-profit / mean reversion).

        Returns:
            List of position status dicts with any actions taken.
        """
        results = []
        positions = self.state.get("open_positions", {})

        if not positions:
            logger.info("No open positions to monitor.")
            return results

        for ticker, pos in list(positions.items()):
            try:
                market = self.client.get_market(ticker)
                market_data = market.get("market", market)
                current_yes_price = market_data.get("yes_price", 50)
                status_val = market_data.get("status", "")

                side = pos["side"]
                entry = pos["entry_price"]
                mean_target = pos["take_profit_price"]
                stop = pos.get("stop_loss", entry - self.config["stop_loss_cents"])

                if side == "yes":
                    current_price = current_yes_price
                    pnl_per_contract = current_price - entry
                else:
                    current_price = 100 - current_yes_price
                    pnl_per_contract = current_price - entry

                result = {
                    "ticker": ticker,
                    "side": side,
                    "entry": entry,
                    "current": current_price,
                    "mean_target": mean_target,
                    "pnl_per_contract": pnl_per_contract,
                    "contracts": pos["contracts"],
                    "total_pnl_cents": pnl_per_contract * pos["contracts"],
                    "action": "hold",
                }

                # Check exit conditions
                should_exit = False
                exit_reason = ""

                # Market settled
                if status_val in ("settled", "closed"):
                    should_exit = True
                    exit_reason = f"Market {status_val}"

                # Stop-loss
                elif pnl_per_contract <= -self.config["stop_loss_cents"]:
                    should_exit = True
                    exit_reason = f"Stop-loss hit (P&L: {pnl_per_contract}¢/contract)"

                # Take-profit: price reverted to mean
                elif side == "yes" and current_price >= mean_target:
                    should_exit = True
                    exit_reason = f"Mean reversion target hit ({current_price}¢ >= {mean_target}¢)"
                elif side == "no" and current_price >= mean_target:
                    should_exit = True
                    exit_reason = f"Mean reversion target hit ({current_price}¢ >= {mean_target}¢)"

                # Exit threshold: deviation closed by configured fraction
                elif self.config["exit_threshold"]:
                    original_deviation = abs(entry - pos["mean_price"])
                    current_deviation = abs(current_price - pos["mean_price"])
                    if original_deviation > 0:
                        reversion_pct = 1 - (current_deviation / original_deviation)
                        if reversion_pct >= self.config["exit_threshold"]:
                            should_exit = True
                            exit_reason = (
                                f"Deviation closed by {reversion_pct:.0%} "
                                f"(threshold: {self.config['exit_threshold']:.0%})"
                            )

                if should_exit:
                    result["action"] = "exit"
                    result["reason"] = exit_reason
                    logger.info(
                        f"EXIT SIGNAL: {ticker} — {exit_reason} "
                        f"(P&L: {pnl_per_contract}¢ x {pos['contracts']} = "
                        f"{pnl_per_contract * pos['contracts']}¢)"
                    )

                    # Place sell order to close position
                    if status_val not in ("settled", "closed"):
                        try:
                            self.client.create_order(
                                ticker=ticker,
                                side=side,
                                action="sell",
                                count=pos["contracts"],
                                reduce_only=True,
                                time_in_force="immediate_or_cancel",
                            )
                            result["action"] = "exit_order_placed"
                        except KalshiAPIError as e:
                            logger.error(f"Exit order failed for {ticker}: {e}")
                            result["action"] = "exit_failed"

                    # Remove from open positions
                    del positions[ticker]
                else:
                    logger.info(
                        f"  HOLD: {ticker} {side} — entry={entry}¢, "
                        f"current={current_price}¢, target={mean_target}¢, "
                        f"P&L={pnl_per_contract}¢/contract"
                    )

                results.append(result)

            except KalshiAPIError as e:
                logger.error(f"Error monitoring {ticker}: {e}")
                results.append({"ticker": ticker, "error": str(e)})

        self._save_state()
        return results

    def close_all_positions(self, dry_run: bool = True) -> List[Dict]:
        """Close all open positions."""
        results = []
        positions = self.state.get("open_positions", {})

        if not positions:
            logger.info("No open positions to close.")
            return results

        for ticker, pos in list(positions.items()):
            if dry_run:
                logger.info(
                    f"[DRY RUN] Would close: {pos['contracts']}x "
                    f"{ticker} {pos['side']}"
                )
                results.append({"ticker": ticker, "action": "dry_run_close"})
                continue

            try:
                self.client.create_order(
                    ticker=ticker,
                    side=pos["side"],
                    action="sell",
                    count=pos["contracts"],
                    reduce_only=True,
                    time_in_force="immediate_or_cancel",
                )
                del positions[ticker]
                logger.info(f"Closed position: {ticker}")
                results.append({"ticker": ticker, "action": "closed"})
            except KalshiAPIError as e:
                logger.error(f"Failed to close {ticker}: {e}")
                results.append({"ticker": ticker, "action": "error", "error": str(e)})

        self._save_state()
        return results

    # ── Reporting ────────────────────────────────────────────────────────

    def portfolio_status(self) -> Dict:
        """Get current portfolio status with all positions and P&L."""
        summary = self.client.portfolio_summary()
        open_pos = self.state.get("open_positions", {})
        trade_history = self.state.get("trades", [])

        return {
            "account": {
                "balance": f"${summary['balance_dollars']:.2f}",
                "portfolio_value": f"${summary['portfolio_value_dollars']:.2f}",
                "total_value": f"${summary['total_value_dollars']:.2f}",
            },
            "tracked_positions": len(open_pos),
            "api_positions": summary["open_positions"],
            "resting_orders": summary["resting_orders"],
            "trade_history_count": len(trade_history),
            "positions": open_pos,
        }


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kalshi Mean Reversion Trader")
    parser.add_argument("--scan", action="store_true", help="Scan for signals")
    parser.add_argument("--execute", action="store_true", help="Execute signals")
    parser.add_argument("--monitor", action="store_true", help="Monitor positions")
    parser.add_argument("--status", action="store_true", help="Portfolio status")
    parser.add_argument("--close-all", action="store_true", help="Close all positions")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no real orders)")
    parser.add_argument("--series", type=str, help="Series ticker to scan")
    parser.add_argument("--event", type=str, help="Event ticker to scan")
    parser.add_argument("--tickers", type=str, nargs="+", help="Specific tickers to scan")
    parser.add_argument("--risk-pct", type=float, help="Risk percentage per trade")
    parser.add_argument("--lookback", type=int, help="Moving average lookback period")
    parser.add_argument("--threshold", type=float, help="Entry threshold (std devs)")
    parser.add_argument("--max-positions", type=int, help="Max concurrent positions")

    args = parser.parse_args()

    # Build config overrides
    config = {}
    if args.risk_pct:
        config["risk_pct"] = args.risk_pct
    if args.lookback:
        config["lookback"] = args.lookback
    if args.threshold:
        config["entry_threshold"] = args.threshold
    if args.max_positions:
        config["max_positions"] = args.max_positions

    trader = KalshiTrader(config=config)

    if args.status:
        status = trader.portfolio_status()
        print(json.dumps(status, indent=2, default=str))

    elif args.scan or args.execute:
        signals = trader.scan_markets(
            series_ticker=args.series,
            event_ticker=args.event,
            tickers=args.tickers,
        )
        print(f"\nFound {len(signals)} signals:")
        for s in signals:
            print(f"  [{s.score:6.1f}] {s.ticker}: {s.reason}")
            print(f"          → {s.action} {s.contracts}x {s.side} @ {s.entry_price}¢")

        if args.execute and signals:
            dry_run = args.dry_run
            print(f"\n{'[DRY RUN] ' if dry_run else ''}Executing {len(signals)} signals...")
            records = trader.execute_signals(signals, dry_run=dry_run)
            for r in records:
                print(
                    f"  {r.signal.ticker}: {r.status}"
                    + (f" → order_id={r.order_id}" if r.order_id else "")
                )

    elif args.monitor:
        results = trader.monitor_positions()
        for r in results:
            print(
                f"  {r['ticker']}: {r.get('action', 'unknown')} "
                f"(P&L: {r.get('total_pnl_cents', 0)}¢)"
            )

    elif args.close_all:
        dry_run = args.dry_run
        results = trader.close_all_positions(dry_run=dry_run)
        for r in results:
            print(f"  {r['ticker']}: {r['action']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
