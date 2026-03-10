#!/usr/bin/env python3
"""
ingestion_engine.py - Polymarket Market Making Bot | Ingestion Engine
======================================================================
Integrates directly with the local Nautilus repository for low-latency
data ingestion from the Polymarket CLOB.

Architecture:
  - Token Discovery  : Polls Gamma Markets API for new tokenIDs
  - Market Feed      : WebSocket subscription to Polymarket CLOB
  - Message Parsing  : Nautilus msgspec tagged-union decoder (raw JSON fallback)
  - Data Mapping     : Maps raw WS messages to TradingTick format

pip dependencies (outside Nautilus):
  pip install msgspec aiohttp websockets

Usage:
  python ingestion_engine.py
"""

import sys
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOCAL NAUTILUS PATH SETUP
# ─────────────────────────────────────────────────────────────────────────────
# Adjust this path to wherever you cloned / unzipped the Nautilus source.
NAUTILUS_LOCAL_PATH = "./nautilus_trader-develop"
sys.path.insert(0, NAUTILUS_LOCAL_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  NAUTILUS SCHEMA IMPORTS  (pure-Python / msgspec — always works)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import msgspec

    from nautilus_trader.adapters.polymarket.schemas.book import (
        PolymarketBookSnapshot,
        PolymarketBookLevel,
        PolymarketQuote,
        PolymarketQuotes,
        PolymarketTrade,
        PolymarketTickSizeChange,
    )
    from nautilus_trader.adapters.polymarket.websocket.client import (
        PolymarketWebSocketChannel,
    )
    from nautilus_trader.adapters.polymarket.websocket.types import MARKET_WS_MESSAGE
    from nautilus_trader.adapters.polymarket.common.gamma_markets import (
        normalize_gamma_market_to_clob_format,
    )

    NAUTILUS_SCHEMAS = True
    print("[OK] Nautilus schemas loaded from local repo")

except ImportError as _e:
    NAUTILUS_SCHEMAS = False
    print(f"[WARN] Nautilus schema import failed ({_e}). Falling back to raw JSON.")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  NAUTILUS WS CLIENT  (requires compiled pyo3 Rust extension)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from nautilus_trader.adapters.polymarket.websocket.client import (
        PolymarketWebSocketClient,
    )
    from nautilus_trader.common.component import LiveClock

    NAUTILUS_WS = True
    print("[OK] Nautilus WebSocketClient (pyo3) available")

except ImportError:
    NAUTILUS_WS = False
    print("[INFO] Nautilus pyo3 WS client unavailable — using 'websockets' fallback")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  THIRD-PARTY FALLBACK LIBRARIES
# ─────────────────────────────────────────────────────────────────────────────
try:
    import websockets  # type: ignore
except ImportError as _e:
    raise RuntimeError("pip install websockets") from _e

try:
    import aiohttp  # type: ignore
except ImportError as _e:
    raise RuntimeError("pip install aiohttp") from _e

# ─────────────────────────────────────────────────────────────────────────────
# 5.  DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class TradingTick:
    """
    Normalised market tick — the canonical output of the ingestion engine,
    consumed by the MM strategy layer.
    """
    token_id: str
    condition_id: str
    bid: float          # Best bid price  (0–1 probability scale)
    ask: float          # Best ask price
    bid_size: float     # Size at best bid (USDC)
    ask_size: float     # Size at best ask (USDC)
    timestamp_ms: float # Exchange timestamp in milliseconds
    source: str         # "snapshot" | "quote_update" | "trade"


@dataclass
class TokenInfo:
    """Metadata about a discovered Polymarket token."""
    token_id: str
    condition_id: str
    question: str
    outcome: str
    first_seen_ms: float = field(default_factory=lambda: time.time() * 1000)


@dataclass
class MarketState:
    """
    Hot-cache of the current best bid / ask for a single token,
    plus full order-book depth (all price levels).
    Updated in-place on every incoming WS message.
    """
    token_id: str
    condition_id: str
    best_bid: float = 0.001
    best_ask: float = 0.999
    bid_size: float = 0.0
    ask_size: float = 0.0
    last_update_ms: float = 0.0
    # Full book: price → size (size == 0 means level removed)
    bids: dict = field(default_factory=dict)
    asks: dict = field(default_factory=dict)

    def top_bids(self, n: int = 8) -> list[list[float]]:
        """Return top-N bids sorted by price descending (best bid first)."""
        return sorted(
            [[p, s] for p, s in self.bids.items() if s > 0],
            key=lambda x: -x[0],
        )[:n]

    def top_asks(self, n: int = 8) -> list[list[float]]:
        """Return top-N asks sorted by price ascending (best ask first)."""
        return sorted(
            [[p, s] for p, s in self.asks.items() if s > 0],
            key=lambda x: x[0],
        )[:n]

# ─────────────────────────────────────────────────────────────────────────────
# 6.  MSGSPEC DECODER  (Nautilus tagged-union, created once at module level)
# ─────────────────────────────────────────────────────────────────────────────
# When Nautilus schemas are available, this decoder handles all four
# event_type values ("book", "price_change", "last_trade_price",
# "tick_size_change") via msgspec's discriminated-union machinery.
_market_decoder: "msgspec.json.Decoder | None" = (
    msgspec.json.Decoder(MARKET_WS_MESSAGE) if NAUTILUS_SCHEMAS else None
)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  MESSAGE PARSER
# ─────────────────────────────────────────────────────────────────────────────

class MessageParser:
    """
    Decodes raw WebSocket bytes from the Polymarket CLOB.

    When Nautilus schemas are available the decoder uses the
    PolymarketBookSnapshot / PolymarketQuotes / PolymarketTrade structs
    directly via msgspec's tagged-union decoder (zero-copy, minimal overhead).

    Falls back to plain json.loads() otherwise.
    """

    def __init__(
        self,
        market_states: dict[str, MarketState],
        token_map: dict[str, TokenInfo],
        on_tick: Callable[[TradingTick], Awaitable[None]],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._states = market_states
        self._token_map = token_map
        self._on_tick = on_tick
        self._loop = loop
        self._log = logging.getLogger(self.__class__.__name__)

    # ── Public callback (synchronous, registered with the WS client) ─────────

    def handle_bytes(self, raw: bytes) -> None:
        """
        Registered as the synchronous handler for the Nautilus WS client.
        Schedules async processing to avoid blocking the event loop.
        """
        self._loop.create_task(self._dispatch(raw))

    # ── Dispatch ─────────────────────────────────────────────────────────────

    async def _dispatch(self, raw: bytes) -> None:
        try:
            if NAUTILUS_SCHEMAS and _market_decoder is not None:
                await self._handle_nautilus(raw)
            else:
                await self._handle_raw_json(raw)
        except Exception as exc:
            self._log.error(f"[Parser] Error: {exc} | payload={raw[:120]}")

    # ── Path A: Nautilus msgspec decoder ─────────────────────────────────────

    async def _handle_nautilus(self, raw: bytes) -> None:
        """Decode via Nautilus tagged-union decoder (fastest path)."""
        msg = _market_decoder.decode(raw)  # type: ignore[union-attr]

        if isinstance(msg, list):
            # Initial batch: list[PolymarketBookSnapshot]
            for snap in msg:
                await self._on_snapshot(snap)
        elif isinstance(msg, PolymarketBookSnapshot):
            await self._on_snapshot(msg)
        elif isinstance(msg, PolymarketQuotes):
            await self._on_quotes(msg)
        elif isinstance(msg, PolymarketTrade):
            await self._on_trade(msg)
        elif isinstance(msg, PolymarketTickSizeChange):
            self._log.info(
                f"[TickSize] {msg.asset_id[:20]}.. "
                f"{msg.old_tick_size} → {msg.new_tick_size}"
            )

    async def _on_snapshot(self, msg: "PolymarketBookSnapshot") -> None:
        """Full order-book snapshot (event_type='book')."""
        state = self._state(msg.asset_id, msg.market)
        ts = float(msg.timestamp)

        # Populate full book depth
        state.bids = {float(b.price): float(b.size) for b in msg.bids}
        state.asks = {float(a.price): float(a.size) for a in msg.asks}

        # Nautilus stores bids sorted ascending: best bid is bids[-1]
        if msg.bids:
            state.best_bid = float(msg.bids[-1].price)
            state.bid_size = float(msg.bids[-1].size)
        if msg.asks:
            state.best_ask = float(msg.asks[-1].price)
            state.ask_size = float(msg.asks[-1].size)
        state.last_update_ms = ts

        await self._emit(state, ts, "snapshot")

    async def _on_quotes(self, msg: "PolymarketQuotes") -> None:
        """Incremental price-level update (event_type='price_change')."""
        ts = float(msg.timestamp)

        for change in msg.price_changes:
            state = self._state(change.asset_id, msg.market)

            # Use the authoritative best_bid / best_ask fields provided by
            # Polymarket in each price_change entry — most reliable source.
            # Also prune any ghost levels that contradict this authoritative top.
            if change.best_bid:
                bb = float(change.best_bid)
                state.best_bid = bb
                for stale in [p for p in state.bids if p > bb]:
                    del state.bids[stale]
            if change.best_ask:
                ba = float(change.best_ask)
                state.best_ask = ba
                for stale in [p for p in state.asks if p < ba]:
                    del state.asks[stale]

            # Update full book at the changed level
            try:
                px = float(change.price)
                sz = float(change.size) if change.size else 0.0
                if change.side.value == "BUY":
                    if sz == 0:
                        state.bids.pop(px, None)
                    else:
                        state.bids[px] = sz
                    state.bid_size = sz
                else:
                    if sz == 0:
                        state.asks.pop(px, None)
                    else:
                        state.asks[px] = sz
                    state.ask_size = sz
            except Exception:
                pass

            state.last_update_ms = ts
            await self._emit(state, ts, "quote_update")

    async def _on_trade(self, msg: "PolymarketTrade") -> None:
        """Last-trade-price event (event_type='last_trade_price')."""
        state = self._state(msg.asset_id, msg.market)
        ts = float(msg.timestamp)
        state.last_update_ms = ts

        self._log.debug(
            f"[Trade] {msg.asset_id[:20]}.. "
            f"side={msg.side.value} px={msg.price} qty={msg.size}"
        )
        await self._emit(state, ts, "trade")

    # ── Path B: Raw JSON fallback ─────────────────────────────────────────────

    async def _handle_raw_json(self, raw: bytes) -> None:
        payload = json.loads(raw)
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            await self._handle_raw_item(item)

    async def _handle_raw_item(self, item: dict) -> None:
        ev = item.get("event_type", "")
        ts = float(item.get("timestamp", time.time() * 1000))

        if ev == "book":
            token_id = item.get("asset_id", "")
            state = self._state(token_id, item.get("market", ""))
            bids_raw = item.get("bids", [])
            asks_raw = item.get("asks", [])
            # Populate full book depth
            state.bids = {float(b["price"]): float(b["size"]) for b in bids_raw}
            state.asks = {float(a["price"]): float(a["size"]) for a in asks_raw}
            if bids_raw:
                state.best_bid = float(bids_raw[-1]["price"])
                state.bid_size = float(bids_raw[-1]["size"])
            if asks_raw:
                state.best_ask = float(asks_raw[-1]["price"])
                state.ask_size = float(asks_raw[-1]["size"])
            state.last_update_ms = ts
            await self._emit(state, ts, "snapshot")

        elif ev == "price_change":
            for change in item.get("price_changes", []):
                token_id = change.get("asset_id", "")
                state = self._state(token_id, item.get("market", ""))
                bb = change.get("best_bid")
                ba = change.get("best_ask")
                if bb:
                    state.best_bid = float(bb)
                if ba:
                    state.best_ask = float(ba)
                side = change.get("side", "")
                try:
                    px = float(change.get("price", 0))
                    sz = float(change.get("size", 0))
                    if px > 0:
                        if side == "BUY":
                            if sz == 0:
                                state.bids.pop(px, None)
                            else:
                                state.bids[px] = sz
                            state.bid_size = sz
                        else:
                            if sz == 0:
                                state.asks.pop(px, None)
                            else:
                                state.asks[px] = sz
                            state.ask_size = sz
                except Exception:
                    pass
                state.last_update_ms = ts
                await self._emit(state, ts, "quote_update")

        elif ev == "last_trade_price":
            token_id = item.get("asset_id", "")
            state = self._state(token_id, item.get("market", ""))
            state.last_update_ms = ts
            await self._emit(state, ts, "trade")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _state(self, token_id: str, condition_id: str) -> MarketState:
        if token_id not in self._states:
            info = self._token_map.get(token_id)
            cid = info.condition_id if info else condition_id
            self._states[token_id] = MarketState(
                token_id=token_id, condition_id=cid
            )
        return self._states[token_id]

    async def _emit(self, state: MarketState, ts_ms: float, source: str) -> None:
        tick = TradingTick(
            token_id=state.token_id,
            condition_id=state.condition_id,
            bid=state.best_bid,
            ask=state.best_ask,
            bid_size=state.bid_size,
            ask_size=state.ask_size,
            timestamp_ms=ts_ms,
            source=source,
        )
        await self._on_tick(tick)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  TOKEN DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

class TokenDiscovery:
    """
    Polls the Gamma Markets REST API to discover new tokenIDs.

    Uses Nautilus normalize_gamma_market_to_clob_format() when available;
    otherwise applies a minimal manual normalisation.

    Parameters
    ----------
    on_new_token : async callable
        Invoked with a TokenInfo whenever a previously-unseen token is found.
    poll_interval_secs : float
        How often to poll the API.
    filters : dict
        Gamma API query params (active, closed, limit, …).
    """

    GAMMA_URL = "https://gamma-api.polymarket.com/markets"

    def __init__(
        self,
        on_new_token: Callable[[TokenInfo], Awaitable[None]],
        poll_interval_secs: float = 30.0,
        filters: dict[str, Any] | None = None,
    ) -> None:
        self._on_new_token = on_new_token
        self._interval = poll_interval_secs
        self._filters = filters or {
            "active": "true",
            "closed": "false",
            "archived": "false",
            "limit": "500",
        }
        self._seen: set[str] = set()
        self._log = logging.getLogger(self.__class__.__name__)

    async def run(self) -> None:
        """Continuous discovery loop — run as an asyncio Task."""
        self._log.info("[Discovery] Token discovery loop started")
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    await self._poll(session)
                except Exception as exc:
                    self._log.error(f"[Discovery] Poll error: {exc}")
                await asyncio.sleep(self._interval)

    async def _poll(self, session: aiohttp.ClientSession) -> None:
        async with session.get(self.GAMMA_URL, params=self._filters) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Gamma API returned {resp.status}")
            raw = await resp.json(content_type=None)

        markets = raw if isinstance(raw, list) else raw.get("data", [])

        for market in markets:
            normalized = (
                normalize_gamma_market_to_clob_format(market)
                if NAUTILUS_SCHEMAS
                else self._minimal_normalize(market)
            )
            condition_id = normalized.get("condition_id", "")
            question = normalized.get("question", "")

            for token in normalized.get("tokens", []):
                token_id = token.get("token_id", "")
                if not token_id or token_id in self._seen:
                    continue

                self._seen.add(token_id)
                info = TokenInfo(
                    token_id=token_id,
                    condition_id=condition_id,
                    question=question,
                    outcome=token.get("outcome", ""),
                    first_seen_ms=time.time() * 1000,
                )
                self._log.info(
                    f"[Discovery] NEW token={token_id[:22]}.. "
                    f"cond={condition_id[:18]}.. "
                    f"outcome={info.outcome} | "
                    f"{question[:60]}"
                )
                await self._on_new_token(info)

    @staticmethod
    def _minimal_normalize(market: dict) -> dict:
        clob_ids = market.get("clobTokenIds", "[]")
        outcomes = market.get("outcomes", "[]")
        if isinstance(clob_ids, str):
            clob_ids = json.loads(clob_ids)
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        return {
            "condition_id": market.get("conditionId", ""),
            "question": market.get("question", ""),
            "tokens": [
                {"token_id": tid, "outcome": out}
                for tid, out in zip(clob_ids, outcomes)
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MARKET FEED  (Nautilus WS client or websockets fallback)
# ─────────────────────────────────────────────────────────────────────────────

class MarketFeed:
    """
    Manages the Polymarket CLOB WebSocket connection.

    Priority:
      1. Nautilus PolymarketWebSocketClient  (requires compiled pyo3)
      2. websockets library                  (pure-Python fallback)

    The Nautilus client handles:
      - Multi-connection pooling (max 200 subs / connection by default)
      - Reference-counted subscribe/unsubscribe
      - Automatic reconnection with re-subscription
    """

    _WS_BASE = "wss://ws-subscriptions-clob.polymarket.com/ws/"

    def __init__(
        self,
        handler: Callable[[bytes], None],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._handler = handler
        self._loop = loop
        self._subs: set[str] = set()
        self._pending_q: asyncio.Queue[str] = asyncio.Queue()
        self._ws_conn = None          # fallback websockets connection
        self._nautilus_client: "PolymarketWebSocketClient | None" = None
        self._last_recv_ts: float = 0.0   # updated on every incoming WS message
        self._log = logging.getLogger(self.__class__.__name__)

        if NAUTILUS_WS:
            self._init_nautilus_client()

    def _init_nautilus_client(self) -> None:
        """
        Instantiate PolymarketWebSocketClient from the local Nautilus repo.

        LiveClock provides nanosecond-precision timestamps used internally
        by the Nautilus message bus.
        """
        clock = LiveClock()
        self._nautilus_client = PolymarketWebSocketClient(
            clock=clock,
            base_url=self._WS_BASE,
            channel=PolymarketWebSocketChannel.MARKET,
            handler=self._handler,
            handler_reconnect=self._on_reconnect,
            loop=self._loop,
            auth=None,               # MARKET channel requires no auth
            max_subscriptions_per_connection=200,
        )
        self._log.info("[Feed] Nautilus PolymarketWebSocketClient initialised")

    async def _on_reconnect(self) -> None:
        self._log.warning("[Feed] Reconnected to Polymarket CLOB WS")

    # ── Public API ────────────────────────────────────────────────────────────

    def queue_subscription(self, token_id: str) -> None:
        """
        Queue a token_id before calling connect().
        Maps to PolymarketWebSocketClient.add_subscription() on the Nautilus path.
        """
        if token_id in self._subs:
            return
        self._subs.add(token_id)
        if self._nautilus_client:
            self._nautilus_client.add_subscription(token_id)

    async def subscribe(self, token_id: str) -> None:
        """
        Subscribe to a token's order book while the feed is running.
        Thread-safe, reference-counted on the Nautilus path.
        """
        if token_id in self._subs:
            return
        self._subs.add(token_id)
        if self._nautilus_client:
            await self._nautilus_client.subscribe(token_id)
        else:
            await self._pending_q.put(token_id)

    async def connect(self) -> None:
        """Connect the feed. Must be called after queue_subscription()."""
        if self._nautilus_client:
            # Nautilus client connects all queued subscriptions at once
            await self._nautilus_client.connect()
            self._log.info("[Feed] Connected via Nautilus WebSocketClient")
        else:
            # Start fallback WS loop as a background task
            self._loop.create_task(self._fallback_loop())

    async def disconnect(self) -> None:
        if self._nautilus_client:
            await self._nautilus_client.disconnect()
        elif self._ws_conn:
            await self._ws_conn.close()
        self._log.info("[Feed] Disconnected")

    async def force_reconnect(self) -> None:
        """
        Close the current WS connection so _fallback_loop reconnects immediately.
        Used after market rollover to ensure new token subscriptions are sent
        on a fresh connection (Polymarket WS does not reliably support
        dynamic subscribe on existing connections).
        """
        if self._nautilus_client:
            return   # Nautilus handles this internally
        if self._ws_conn:
            self._log.info("[Feed] Force-reconnect for rollover")
            try:
                await self._ws_conn.close()
            except Exception:
                pass

    async def unsubscribe(self, token_id: str) -> None:
        """
        Unsubscribe from a token's order book feed.
        Used during market rollover to drop stale token streams.
        """
        self._subs.discard(token_id)
        if self._nautilus_client:
            await self._nautilus_client.unsubscribe(token_id)
        elif self._ws_conn:
            try:
                msg = json.dumps({
                    "assets_ids": [token_id],
                    "operation": "unsubscribe",
                })
                await self._ws_conn.send(msg)
                self._log.info(f"[Feed] Unsubscribed from {token_id[:22]}..")
            except Exception as exc:
                self._log.warning(f"[Feed] Unsubscribe error: {exc}")

    # ── Fallback WebSocket (websockets library) ───────────────────────────────

    _WS_SILENCE_TIMEOUT = 45   # seconds without a message → force reconnect

    async def _fallback_loop(self) -> None:
        """Reconnecting WS loop using the websockets library."""
        while True:
            try:
                async with websockets.connect(
                    self._WS_BASE + "market",
                    ping_interval=20,
                    ping_timeout=30,
                    close_timeout=5,
                ) as ws:
                    self._ws_conn = ws
                    self._last_recv_ts = time.time()
                    self._log.info("[Feed] Connected (websockets fallback)")

                    # Send initial subscription for all queued tokens
                    if self._subs:
                        init_msg = json.dumps({
                            "assets_ids": list(self._subs),
                            "type": "market",
                        })
                        await ws.send(init_msg)
                        self._log.info(
                            f"[Feed] Subscribed to {len(self._subs)} tokens"
                        )

                    # Run recv, dynamic-subscribe, and watchdog concurrently.
                    # Any one raising cancels the others → outer except reconnects.
                    await asyncio.gather(
                        self._fb_recv_loop(ws),
                        self._fb_subscribe_loop(ws),
                        self._fb_watchdog(ws),
                    )

            except Exception as exc:
                self._log.error(f"[Feed] WS error: {exc} — reconnecting in 3 s")
                self._ws_conn = None
                await asyncio.sleep(3)

    async def _fb_recv_loop(self, ws) -> None:
        async for raw in ws:
            self._last_recv_ts = time.time()
            payload = raw.encode() if isinstance(raw, str) else raw
            self._handler(payload)

    async def _fb_subscribe_loop(self, ws) -> None:
        """Picks up subscriptions added after connect()."""
        while True:
            token_id = await self._pending_q.get()
            try:
                msg = json.dumps({
                    "assets_ids": [token_id],
                    "type": "market",
                })
                await ws.send(msg)
                self._log.info(f"[Feed] Dynamic subscribe → {token_id[:22]}..")
            except Exception as exc:
                self._log.warning(f"[Feed] Dynamic subscribe error: {exc}")
                # Re-queue so the token is retried after reconnect
                await self._pending_q.put(token_id)
                raise   # propagate to trigger reconnect

    async def _fb_watchdog(self, ws) -> None:
        """Force-close the WS if no messages received for _WS_SILENCE_TIMEOUT seconds."""
        while True:
            await asyncio.sleep(10)
            silence = time.time() - self._last_recv_ts
            if silence > self._WS_SILENCE_TIMEOUT:
                self._log.warning(
                    f"[Feed] No WS messages for {silence:.0f}s — forcing reconnect"
                )
                await ws.close()


# ─────────────────────────────────────────────────────────────────────────────
# 10. DATA MAPPING UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def map_tick_for_trading(tick: TradingTick) -> dict[str, Any]:
    """
    Map a TradingTick to a flat dict ready for strategy / order logic.

    Fields
    ------
    token_id, condition_id, bid, ask, mid, spread,
    bid_size, ask_size, timestamp_ms, source
    """
    spread = round(tick.ask - tick.bid, 5)
    mid    = round((tick.bid + tick.ask) / 2, 5)
    return {
        "token_id":     tick.token_id,
        "condition_id": tick.condition_id,
        "bid":          tick.bid,
        "ask":          tick.ask,
        "mid":          mid,
        "spread":       spread,
        "bid_size":     tick.bid_size,
        "ask_size":     tick.ask_size,
        "timestamp_ms": tick.timestamp_ms,
        "source":       tick.source,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 11. INGESTION ENGINE  (top-level coordinator)
# ─────────────────────────────────────────────────────────────────────────────

class IngestionEngine:
    """
    Top-level coordinator for the Polymarket ingestion pipeline.

    Wires together:
      TokenDiscovery  →  MarketFeed (subscribe)
      MarketFeed      →  MessageParser (bytes callback)
      MessageParser   →  on_tick (TradingTick)

    Parameters
    ----------
    on_tick : async callable
        Your strategy entry point.  Called with every TradingTick.
    token_ids : list[str], optional
        Pre-configured token IDs to subscribe to on startup.
    discovery_enabled : bool
        Poll Gamma API for new tokens automatically.
    discovery_interval_secs : float
        How often to poll Gamma API.
    discovery_filters : dict, optional
        Gamma API query params.
    """

    def __init__(
        self,
        on_tick: Callable[[TradingTick], Awaitable[None]],
        token_ids: list[str] | None = None,
        discovery_enabled: bool = True,
        discovery_interval_secs: float = 30.0,
        discovery_filters: dict[str, Any] | None = None,
    ) -> None:
        self._loop = asyncio.get_event_loop()
        self._log  = logging.getLogger(self.__class__.__name__)

        # Shared state
        self._market_states: dict[str, MarketState] = {}
        self._token_map: dict[str, TokenInfo]       = {}

        # Components
        self._parser = MessageParser(
            market_states=self._market_states,
            token_map=self._token_map,
            on_tick=on_tick,
            loop=self._loop,
        )
        self._feed = MarketFeed(
            handler=self._parser.handle_bytes,
            loop=self._loop,
        )
        self._discovery = (
            TokenDiscovery(
                on_new_token=self._on_new_token,
                poll_interval_secs=discovery_interval_secs,
                filters=discovery_filters,
            )
            if discovery_enabled
            else None
        )

        self._initial_tokens: list[str] = token_ids or []

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._log.info(
            "[Engine] Starting — "
            f"nautilus_schemas={NAUTILUS_SCHEMAS}  "
            f"nautilus_ws={NAUTILUS_WS}"
        )

        # Pre-queue known tokens so they are included in the first WS frame
        for token_id in self._initial_tokens:
            self._feed.queue_subscription(token_id)

        # Connect WebSocket feed
        await self._feed.connect()

        # Launch background tasks
        tasks: list[asyncio.Task] = []
        if self._discovery:
            tasks.append(asyncio.create_task(self._discovery.run()))

        self._log.info("[Engine] Running — awaiting market data...")

        if tasks:
            await asyncio.gather(*tasks)
        else:
            await asyncio.Event().wait()   # block forever until KeyboardInterrupt

    async def stop(self) -> None:
        await self._feed.disconnect()
        self._log.info("[Engine] Stopped.")

    # ── Internal callbacks ────────────────────────────────────────────────────

    async def _on_new_token(self, info: TokenInfo) -> None:
        """Invoked by TokenDiscovery for each new token found."""
        self._token_map[info.token_id] = info
        await self._feed.subscribe(info.token_id)

    async def unsubscribe(self, token_id: str) -> None:
        """
        Unsubscribe from a token and remove its state.
        Called by RolloverManager when a market expires.
        """
        self._token_map.pop(token_id, None)
        self._market_states.pop(token_id, None)
        await self._feed.unsubscribe(token_id)

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def market_states(self) -> dict[str, MarketState]:
        """Read-only snapshot of current bid/ask state for all tracked tokens."""
        return dict(self._market_states)

    @property
    def token_map(self) -> dict[str, TokenInfo]:
        return dict(self._token_map)


# ─────────────────────────────────────────────────────────────────────────────
# 12. EXAMPLE ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

async def on_tick(tick: TradingTick) -> None:
    """
    Replace / extend this with your MM strategy logic.
    `mapped` is the flat dict fed into quote / order generation.
    """
    mapped = map_tick_for_trading(tick)
    logging.info(
        "[Tick] %-22s  bid=%.4f (%.1f)  ask=%.4f (%.1f)  "
        "spread=%.4f  mid=%.4f  src=%s",
        mapped["token_id"][:22] + "..",
        mapped["bid"], mapped["bid_size"],
        mapped["ask"], mapped["ask_size"],
        mapped["spread"],
        mapped["mid"],
        mapped["source"],
    )


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Optional: hard-code specific token IDs to track immediately ──────────
    # Leave empty to rely purely on discovery.
    KNOWN_TOKEN_IDS: list[str] = [
        # Example:
        # "21742633143463906290569050155826241533067272736897614950488156847949938836455",
    ]

    engine = IngestionEngine(
        on_tick=on_tick,
        token_ids=KNOWN_TOKEN_IDS,
        discovery_enabled=True,
        discovery_interval_secs=30.0,
        discovery_filters={
            "active":   "true",
            "closed":   "false",
            "archived": "false",
            "limit":    "500",
        },
    )

    try:
        await engine.start()
    except KeyboardInterrupt:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
