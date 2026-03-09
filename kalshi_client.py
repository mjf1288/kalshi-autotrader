"""
Kalshi API Client — Authenticated REST client with RSA-PSS signing.

Usage:
    from kalshi_client import KalshiClient

    client = KalshiClient(
        api_key_id="your-api-key-id",
        private_key_path="/path/to/kalshi-key.key"
    )

    # Check balance
    balance = client.get_balance()
    print(f"Balance: ${balance['balance'] / 100:.2f}")

    # Browse markets
    markets = client.get_markets(status="open", limit=20)

    # Place an order
    order = client.create_order(
        ticker="KXHIGHNY-26MAR08-T50",
        side="yes",
        action="buy",
        count=10,
        yes_price=45
    )

Environment variables (alternative to constructor args):
    KALSHI_API_KEY_ID       — Your API key UUID
    KALSHI_PRIVATE_KEY_PATH — Path to your .key file
    KALSHI_BASE_URL         — Override base URL (default: production)
"""

import os
import json
import time
import base64
import logging
import datetime
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse, urlencode

import requests
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger("kalshi_client")

# ── URLs ────────────────────────────────────────────────────────────────
PROD_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEMO_BASE_URL = "https://demo-api.kalshi.co/trade-api/v2"


class KalshiAPIError(Exception):
    """Raised when the Kalshi API returns an error response."""

    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self.body = body
        self.code = body.get("code", "")
        self.message = body.get("message", str(body))
        super().__init__(f"Kalshi API {status_code}: {self.message}")


class KalshiClient:
    """Authenticated client for the Kalshi Exchange REST API."""

    def __init__(
        self,
        api_key_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        private_key_str: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30,
    ):
        self.api_key_id = api_key_id or os.environ.get("KALSHI_API_KEY_ID", "")
        self.base_url = (
            base_url
            or os.environ.get("KALSHI_BASE_URL", PROD_BASE_URL)
        ).rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

        # Load private key
        if private_key_str:
            self._private_key = serialization.load_pem_private_key(
                private_key_str.encode(), password=None, backend=default_backend()
            )
        else:
            key_path = private_key_path or os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
            if key_path:
                with open(key_path, "rb") as f:
                    self._private_key = serialization.load_pem_private_key(
                        f.read(), password=None, backend=default_backend()
                    )
            else:
                self._private_key = None  # unauthenticated mode

    # ── Signing ──────────────────────────────────────────────────────────

    def _sign(self, timestamp: str, method: str, path: str) -> str:
        """Create RSA-PSS SHA-256 signature: timestamp + METHOD + path (no query)."""
        path_clean = path.split("?")[0]
        message = f"{timestamp}{method}{path_clean}".encode("utf-8")
        sig = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode("utf-8")

    def _auth_headers(self, method: str, path: str) -> Dict[str, str]:
        ts = str(int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000))
        full_path = urlparse(self.base_url + path).path
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, full_path),
        }

    # ── HTTP helpers ─────────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        json_body: Optional[Dict] = None,
        auth: bool = True,
    ) -> Dict[str, Any]:
        url = self.base_url + path
        headers = {"Content-Type": "application/json"}
        if auth and self._private_key:
            headers.update(self._auth_headers(method, path))

        resp = self.session.request(
            method,
            url,
            params=params,
            json=json_body,
            headers=headers,
            timeout=self.timeout,
        )
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = {"message": resp.text}
            raise KalshiAPIError(resp.status_code, body)

        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()

    def get(self, path: str, params: Optional[Dict] = None, auth: bool = True):
        return self._request("GET", path, params=params, auth=auth)

    def post(self, path: str, body: Optional[Dict] = None, auth: bool = True):
        return self._request("POST", path, json_body=body, auth=auth)

    def delete(self, path: str, params: Optional[Dict] = None, auth: bool = True):
        return self._request("DELETE", path, params=params, auth=auth)

    def put(self, path: str, body: Optional[Dict] = None, auth: bool = True):
        return self._request("PUT", path, json_body=body, auth=auth)

    # ── Market Data (public — no auth needed) ────────────────────────────

    def get_markets(
        self,
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        status: Optional[str] = None,
        ticker: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Dict:
        """Get list of markets with optional filters."""
        params = {"limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status:
            params["status"] = status
        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor
        return self.get("/markets", params=params, auth=False)

    def get_market(self, ticker: str) -> Dict:
        """Get details for a single market."""
        return self.get(f"/markets/{ticker}", auth=False)

    def get_orderbook(self, ticker: str, depth: Optional[int] = None) -> Dict:
        """Get orderbook for a market."""
        params = {}
        if depth:
            params["depth"] = depth
        return self.get(f"/markets/{ticker}/orderbook", params=params, auth=False)

    def get_trades(
        self,
        ticker: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
    ) -> Dict:
        """Get public trade history."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts
        return self.get("/markets/trades", params=params, auth=False)

    def get_candlesticks(
        self,
        series_ticker: str,
        ticker: Optional[str] = None,
        period_interval: str = "1h",
        limit: int = 100,
    ) -> Dict:
        """Get OHLC candlestick data for a market."""
        params = {
            "series_ticker": series_ticker,
            "period_interval": period_interval,
            "limit": limit,
        }
        if ticker:
            params["ticker"] = ticker
        return self.get(f"/markets/{ticker or series_ticker}/candlesticks", params=params, auth=False)

    def get_series(self, series_ticker: str) -> Dict:
        """Get series information."""
        return self.get(f"/series/{series_ticker}", auth=False)

    def get_series_list(self, limit: int = 100, cursor: Optional[str] = None) -> Dict:
        """List all series."""
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self.get("/series", params=params, auth=False)

    def get_events(
        self,
        series_ticker: Optional[str] = None,
        status: Optional[str] = None,
        with_nested_markets: bool = False,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> Dict:
        """List events."""
        params = {"limit": limit, "with_nested_markets": with_nested_markets}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        return self.get("/events", params=params, auth=False)

    def get_event(self, event_ticker: str) -> Dict:
        """Get event details."""
        return self.get(f"/events/{event_ticker}", auth=False)

    # ── Portfolio (auth required) ────────────────────────────────────────

    def get_balance(self, subaccount: int = 0) -> Dict:
        """Get account balance and portfolio value (in cents)."""
        return self.get("/portfolio/balance", params={"subaccount": subaccount})

    def get_positions(
        self,
        ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        count_filter: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        subaccount: int = 0,
    ) -> Dict:
        """Get portfolio positions."""
        params = {"limit": limit, "subaccount": subaccount}
        if ticker:
            params["ticker"] = ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if count_filter:
            params["count_filter"] = count_filter
        if cursor:
            params["cursor"] = cursor
        return self.get("/portfolio/positions", params=params)

    def get_fills(
        self,
        ticker: Optional[str] = None,
        order_id: Optional[str] = None,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        subaccount: Optional[int] = None,
    ) -> Dict:
        """Get fill history."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if order_id:
            params["order_id"] = order_id
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts
        if cursor:
            params["cursor"] = cursor
        if subaccount is not None:
            params["subaccount"] = subaccount
        return self.get("/portfolio/fills", params=params)

    def get_orders(
        self,
        ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        status: Optional[str] = None,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
        subaccount: Optional[int] = None,
    ) -> Dict:
        """Get orders."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status:
            params["status"] = status
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts
        if cursor:
            params["cursor"] = cursor
        if subaccount is not None:
            params["subaccount"] = subaccount
        return self.get("/portfolio/orders", params=params)

    def get_order(self, order_id: str, subaccount: int = 0) -> Dict:
        """Get a single order by ID."""
        return self.get(
            f"/portfolio/orders/{order_id}", params={"subaccount": subaccount}
        )

    def get_settlements(
        self, limit: int = 100, cursor: Optional[str] = None
    ) -> Dict:
        """Get settlement history."""
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self.get("/portfolio/settlements", params=params)

    def get_total_resting_order_value(self, subaccount: int = 0) -> Dict:
        """Get total value of resting orders."""
        return self.get(
            "/portfolio/total_resting_order_value",
            params={"subaccount": subaccount},
        )

    # ── Order Management (auth required) ─────────────────────────────────

    def create_order(
        self,
        ticker: str,
        side: str,
        action: str,
        count: Optional[int] = None,
        count_fp: Optional[str] = None,
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        yes_price_dollars: Optional[str] = None,
        no_price_dollars: Optional[str] = None,
        time_in_force: Optional[str] = None,
        client_order_id: Optional[str] = None,
        expiration_ts: Optional[int] = None,
        buy_max_cost: Optional[int] = None,
        post_only: Optional[bool] = None,
        reduce_only: Optional[bool] = None,
        self_trade_prevention_type: Optional[str] = None,
        order_group_id: Optional[str] = None,
        cancel_order_on_pause: Optional[bool] = None,
        subaccount: int = 0,
    ) -> Dict:
        """
        Place an order on a market.

        Args:
            ticker: Market ticker (e.g., "KXHIGHNY-26MAR08-T50")
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts (integer)
            count_fp: Contract count as string (e.g., "10.00")
            yes_price: Price in cents (1-99) for yes side
            no_price: Price in cents (1-99) for no side
            yes_price_dollars: Price as dollar string (e.g., "0.5600")
            no_price_dollars: Price as dollar string (e.g., "0.5600")
            time_in_force: "good_till_canceled", "fill_or_kill", "immediate_or_cancel"
            client_order_id: Idempotency key
            post_only: If True, order will only rest (maker only)
            reduce_only: If True, order can only reduce existing position
            cancel_order_on_pause: Cancel if exchange pauses trading
            subaccount: Subaccount number (0 = primary)

        Returns:
            Order response dict with order details.
        """
        body: Dict[str, Any] = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "subaccount": subaccount,
        }
        if count is not None:
            body["count"] = count
        if count_fp is not None:
            body["count_fp"] = count_fp
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price
        if yes_price_dollars is not None:
            body["yes_price_dollars"] = yes_price_dollars
        if no_price_dollars is not None:
            body["no_price_dollars"] = no_price_dollars
        if time_in_force is not None:
            body["time_in_force"] = time_in_force
        if client_order_id is not None:
            body["client_order_id"] = client_order_id
        if expiration_ts is not None:
            body["expiration_ts"] = expiration_ts
        if buy_max_cost is not None:
            body["buy_max_cost"] = buy_max_cost
        if post_only is not None:
            body["post_only"] = post_only
        if reduce_only is not None:
            body["reduce_only"] = reduce_only
        if self_trade_prevention_type is not None:
            body["self_trade_prevention_type"] = self_trade_prevention_type
        if order_group_id is not None:
            body["order_group_id"] = order_group_id
        if cancel_order_on_pause is not None:
            body["cancel_order_on_pause"] = cancel_order_on_pause

        return self.post("/portfolio/orders", body=body)

    def cancel_order(self, order_id: str, subaccount: int = 0) -> Dict:
        """Cancel a resting order."""
        return self.delete(
            f"/portfolio/orders/{order_id}", params={"subaccount": subaccount}
        )

    def amend_order(
        self,
        order_id: str,
        count: Optional[int] = None,
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        subaccount: int = 0,
    ) -> Dict:
        """Amend an existing order (change price or increase quantity)."""
        body: Dict[str, Any] = {"subaccount": subaccount}
        if count is not None:
            body["count"] = count
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price
        return self.post(f"/portfolio/orders/{order_id}/amend", body=body)

    def decrease_order(
        self, order_id: str, reduce_by: int, subaccount: int = 0
    ) -> Dict:
        """Decrease quantity on a resting order."""
        return self.post(
            f"/portfolio/orders/{order_id}/decrease",
            body={"reduce_by": reduce_by, "subaccount": subaccount},
        )

    def batch_create_orders(self, orders: List[Dict], subaccount: int = 0) -> Dict:
        """
        Create multiple orders in a single request (max 20).

        Args:
            orders: List of order dicts (same fields as create_order body).
            subaccount: Subaccount number.

        Returns:
            Batch response with per-order results.
        """
        for o in orders:
            o.setdefault("subaccount", subaccount)
        return self.post("/portfolio/orders/batched", body={"orders": orders})

    def batch_cancel_orders(
        self, order_ids: List[str], subaccount: int = 0
    ) -> Dict:
        """Cancel multiple orders."""
        return self.delete(
            "/portfolio/orders/batched",
            params={"order_ids": ",".join(order_ids), "subaccount": subaccount},
        )

    # ── Convenience helpers ──────────────────────────────────────────────

    def get_all_markets_paginated(
        self,
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        status: Optional[str] = None,
        max_pages: int = 10,
    ) -> List[Dict]:
        """Fetch all markets across pages."""
        all_markets = []
        cursor = None
        for _ in range(max_pages):
            resp = self.get_markets(
                series_ticker=series_ticker,
                event_ticker=event_ticker,
                status=status,
                limit=200,
                cursor=cursor,
            )
            all_markets.extend(resp.get("markets", []))
            cursor = resp.get("cursor", "")
            if not cursor:
                break
        return all_markets

    def get_all_positions_paginated(self, max_pages: int = 10) -> List[Dict]:
        """Fetch all positions across pages."""
        all_positions = []
        cursor = None
        for _ in range(max_pages):
            resp = self.get_positions(
                count_filter="position", limit=1000, cursor=cursor
            )
            all_positions.extend(resp.get("market_positions", []))
            cursor = resp.get("cursor", "")
            if not cursor:
                break
        return all_positions

    def portfolio_summary(self) -> Dict:
        """Get a complete portfolio summary."""
        balance = self.get_balance()
        positions = self.get_all_positions_paginated()
        resting = self.get_orders(status="resting")
        return {
            "balance_dollars": balance["balance"] / 100,
            "portfolio_value_dollars": balance["portfolio_value"] / 100,
            "total_value_dollars": (balance["balance"] + balance["portfolio_value"]) / 100,
            "open_positions": len(positions),
            "positions": positions,
            "resting_orders": len(resting.get("orders", [])),
            "orders": resting.get("orders", []),
        }
