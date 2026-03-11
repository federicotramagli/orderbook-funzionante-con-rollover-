#!/usr/bin/env python3
"""
rollover_manager.py - Polymarket Auto-Rollover for BTC-UPDOWN-15M series
=========================================================================
Monitors an active BTC-UPDOWN-15M market, detects its expiry, and seamlessly
switches the ingestion engine to the next market the moment it appears on the
Gamma API — zero gap in market-making.

Architecture:
  RolloverWatcher          — polls Gamma API, detects new conditionId
  PolymarketRolloverManager — orchestrates engine subscribe/unsubscribe +
                              emits YesNoMarketTick to the strategy layer

pip extras (beyond ingestion_engine deps):
  pip install aiohttp   (already required)

Usage (standalone demo):
  python rollover_manager.py

Integration:
  from rollover_manager import PolymarketRolloverManager, MarketSeriesConfig
"""

import asyncio
import dataclasses
import json
import logging
import re
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

import aiohttp

# SSL context that skips certificate verification (macOS cert bundle fix)
_SSL_NO_VERIFY = ssl.create_default_context()
_SSL_NO_VERIFY.check_hostname = False
_SSL_NO_VERIFY.verify_mode = ssl.CERT_NONE

# ── Import from local ingestion engine ───────────────────────────────────────
from ingestion_engine import (
    IngestionEngine,
    TradingTick,
    TokenInfo,
    map_tick_for_trading,
    NAUTILUS_SCHEMAS,
)

try:
    from nautilus_trader.adapters.polymarket.common.gamma_markets import (
        normalize_gamma_market_to_clob_format,
    )
except ImportError:
    normalize_gamma_market_to_clob_format = None  # type: ignore[assignment]

_log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MarketSeriesConfig:
    """
    Describes a recurring Polymarket market series.

    slug_prefix : str
        Prefix of the Gamma API slug, e.g. "btc-updown-15m".
        Markets are matched client-side: slug.startswith(slug_prefix).
    question_keywords : list[str]
        Fallback question-text filter (case-insensitive AND logic).
    fast_poll_threshold_secs : float
        When TTL (seconds to expiry) drops below this value, switch to
        fast polling. Default: 120 s (2 minutes).
    fast_poll_interval_secs : float
        Poll interval during the fast-poll window. Default: 3 s.
    normal_poll_interval_secs : float
        Poll interval during normal operation. Default: 30 s.
    post_expiry_poll_interval_secs : float
        Poll interval after current market expires, while waiting for the
        next one to appear. Default: 2 s.
    """
    slug_prefix: str = "btc-updown-15m"
    question_keywords: list[str] = field(
        default_factory=lambda: ["btc", "15"]
    )
    fast_poll_threshold_secs: float = 120.0
    fast_poll_interval_secs: float = 3.0
    normal_poll_interval_secs: float = 30.0
    post_expiry_poll_interval_secs: float = 2.0
    gamma_base_url: str = "https://gamma-api.polymarket.com/markets"
    force_slug: str = ""   # if set, this slug is always tried first (no filters)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActiveMarket:
    """
    Represents a single active market in the series.
    Both YES and NO token IDs are stored so the rollover manager can
    subscribe/unsubscribe both at once.
    """
    condition_id: str
    question: str
    yes_token_id: str
    yes_outcome: str        # "Yes" / "Up" / etc.
    no_token_id: str
    no_outcome: str         # "No" / "Down" / etc.
    end_timestamp_ms: float # UTC milliseconds
    start_timestamp_ms: float
    accepting_orders: bool = True
    # Gamma API state — authoritative source of truth for liveness.
    # Polymarket keeps markets active=True / closed=False for several minutes
    # AFTER the slug timestamp, while the resolution is pending.
    api_active: bool = True
    api_closed: bool = False
    price_to_beat: float = 0.0   # BTC price at market open (from eventMetadata)

    @property
    def seconds_to_expiry(self) -> float:
        return (self.end_timestamp_ms - time.time() * 1000) / 1_000

    @property
    def is_truly_done(self) -> bool:
        """
        True when the market has expired.
        Primary: api_closed from Polymarket.
        Fallback: TTL <= 0 (slug timestamp elapsed) with a 15s grace period
        to absorb minor clock skew.
        """
        if self.api_closed:
            return True
        return self.seconds_to_expiry < -15

    @property
    def is_expired(self) -> bool:
        return self.seconds_to_expiry <= 0

    @property
    def ttl_str(self) -> str:
        tte = self.seconds_to_expiry
        if tte < 0:
            return f"CLOSING({abs(tte):.0f}s)" if not self.api_closed else "CLOSED"
        m, s = divmod(int(tte), 60)
        return f"{m:02d}:{s:02d}"

    def __repr__(self) -> str:
        return (
            f"ActiveMarket(cond={self.condition_id[:12]}.. "
            f"yes={self.yes_token_id[:12]}.. "
            f"no={self.no_token_id[:12]}.. "
            f"ttl={self.ttl_str})"
        )


@dataclass
class MarketBookState:
    """
    Real-time combined order-book state for YES and NO legs of a binary market.
    The key arbitrage invariant is:  yes_mid + no_mid ≈ 1.00
    """
    condition_id: str
    yes_token_id: str
    no_token_id: str

    # YES leg
    yes_bid: float = 0.001
    yes_ask: float = 0.999
    yes_bid_size: float = 0.0
    yes_ask_size: float = 0.0
    yes_ts_ms: float = 0.0

    # NO leg
    no_bid: float = 0.001
    no_ask: float = 0.999
    no_bid_size: float = 0.0
    no_ask_size: float = 0.0
    no_ts_ms: float = 0.0

    # ── Derived metrics ───────────────────────────────────────────────────────

    @property
    def yes_mid(self) -> float:
        return round((self.yes_bid + self.yes_ask) / 2, 5)

    @property
    def no_mid(self) -> float:
        return round((self.no_bid + self.no_ask) / 2, 5)

    @property
    def yes_spread(self) -> float:
        return round(self.yes_ask - self.yes_bid, 5)

    @property
    def no_spread(self) -> float:
        return round(self.no_ask - self.no_bid, 5)

    @property
    def price_sum(self) -> float:
        """yes_mid + no_mid  (target ≈ 1.00)"""
        return round(self.yes_mid + self.no_mid, 5)

    @property
    def price_deviation(self) -> float:
        """|price_sum - 1.0|  (0 = perfectly priced, > 0.02 = arbitrage)"""
        return round(abs(self.price_sum - 1.0), 5)

    @property
    def ts_spread_ms(self) -> float:
        """Time delta between the last YES and NO updates (sync quality)."""
        return abs(self.yes_ts_ms - self.no_ts_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id":   self.condition_id,
            "yes_token_id":   self.yes_token_id,
            "no_token_id":    self.no_token_id,
            "yes_bid":        self.yes_bid,
            "yes_ask":        self.yes_ask,
            "yes_mid":        self.yes_mid,
            "yes_spread":     self.yes_spread,
            "yes_bid_size":   self.yes_bid_size,
            "yes_ask_size":   self.yes_ask_size,
            "yes_ts_ms":      self.yes_ts_ms,
            "no_bid":         self.no_bid,
            "no_ask":         self.no_ask,
            "no_mid":         self.no_mid,
            "no_spread":      self.no_spread,
            "no_bid_size":    self.no_bid_size,
            "no_ask_size":    self.no_ask_size,
            "no_ts_ms":       self.no_ts_ms,
            "price_sum":      self.price_sum,
            "price_deviation": self.price_deviation,
            "ts_spread_ms":   self.ts_spread_ms,
        }


@dataclass(frozen=True, slots=True)
class YesNoMarketTick:
    """
    Combined YES/NO tick emitted by PolymarketRolloverManager.
    This is the canonical input to the market-making strategy.
    """
    condition_id: str
    yes_token_id: str
    no_token_id: str
    # YES
    yes_bid: float
    yes_ask: float
    yes_mid: float
    yes_spread: float
    yes_bid_size: float
    yes_ask_size: float
    yes_ts_ms: float
    # NO
    no_bid: float
    no_ask: float
    no_mid: float
    no_spread: float
    no_bid_size: float
    no_ask_size: float
    no_ts_ms: float
    # Arbitrage metrics
    price_sum: float        # should be ≈ 1.0
    price_deviation: float  # |price_sum - 1.0|
    ts_spread_ms: float     # sync quality
    source: str             # "snapshot" | "quote_update" | "trade"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GAMMA API HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_iso_to_ms(iso_str: str | None) -> float:
    """Convert ISO-8601 datetime string → Unix milliseconds (UTC)."""
    if not iso_str:
        return 0.0
    try:
        s = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp() * 1_000
    except Exception:
        return 0.0


def _extract_slug_timestamp_ms(slug: str, period_s: int = 900) -> float:
    """
    Extract the Unix START-timestamp embedded in Polymarket slugs and convert
    it to the END-timestamp (ms) by adding one period.

    Examples
    --------
    "btc-updown-15m-1773106200"  →  (1773106200 + 900) * 1000 = 1773107100000.0
    "some-market"                →  0.0  (no timestamp found)
    """
    m = re.search(r"-(\d{9,11})$", slug)
    if m:
        return (float(m.group(1)) + period_s) * 1_000
    return 0.0


def _minimal_normalize(market: dict) -> dict:
    clob_ids = market.get("clobTokenIds", "[]")
    outcomes  = market.get("outcomes", "[]")
    op = market.get("outcomePrices", "[]")
    for field_val, name in [(clob_ids, "clobTokenIds"), (outcomes, "outcomes"), (op, "op")]:
        pass  # just using the variable
    if isinstance(clob_ids, str):
        clob_ids = json.loads(clob_ids)
    if isinstance(outcomes, str):
        outcomes = json.loads(outcomes)
    if isinstance(op, str):
        op = json.loads(op)

    tokens = [
        {"token_id": tid, "outcome": out,
         "price": float(op[i]) if i < len(op) else 0.5}
        for i, (tid, out) in enumerate(zip(clob_ids, outcomes))
    ]
    return {
        "condition_id": market.get("conditionId", ""),
        "question":     market.get("question", ""),
        "end_date_iso": market.get("endDateIso") or market.get("end_date_iso"),
        "start_date_iso": market.get("startDateIso") or market.get("start_date_iso"),
        "accepting_orders": market.get("acceptingOrders", True),
        "active":         market.get("active", False),
        "closed":         market.get("closed", False),
        "market_slug":    market.get("slug", ""),
        "event_metadata": _extract_event_metadata(market),
        "tokens": tokens,
    }


def _extract_ptb_from_description(market: dict) -> float:
    """
    Extract priceToBeat from the market question or description text.
    Polymarket formats it as: 'Is Bitcoin above $71,384.09?'
    Falls back to description field if question doesn't contain a price.
    """
    for field in ("question", "description"):
        text = market.get(field) or ""
        m = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", text)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                pass
    return 0.0


def _extract_event_metadata(market: dict) -> dict:
    """Extract eventMetadata from market dict.
    Polymarket embeds it at market.events[0].eventMetadata."""
    meta = market.get("eventMetadata") or market.get("event_metadata")
    if meta and isinstance(meta, dict) and meta.get("priceToBeat"):
        return meta
    events = market.get("events") or []
    if events and isinstance(events, list):
        return events[0].get("eventMetadata") or {}
    return {}


def _normalize(market: dict) -> dict:
    if normalize_gamma_market_to_clob_format is not None:
        try:
            n = normalize_gamma_market_to_clob_format(market)
            # Gamma normaliser doesn't expose end_date_iso in all versions; patch it
            if not n.get("end_date_iso"):
                n["end_date_iso"] = (
                    market.get("endDateIso") or market.get("end_date_iso")
                )
            if not n.get("start_date_iso"):
                n["start_date_iso"] = (
                    market.get("startDateIso") or market.get("start_date_iso")
                )
            if not n.get("market_slug"):
                n["market_slug"] = market.get("slug", "")
            # Always merge raw eventMetadata — Nautilus may set event_metadata
            # without priceToBeat, so the guard `if not n.get(...)` would skip it.
            extracted = _extract_event_metadata(market)
            existing  = n.get("event_metadata") or {}
            n["event_metadata"] = {**existing, **extracted}
            return n
        except Exception:
            pass
    return _minimal_normalize(market)


def _build_active_market(n: dict) -> ActiveMarket | None:
    """
    Build an ActiveMarket from a normalised Gamma market dict.
    Returns None if the market has < 2 tokens or no usable end time.

    TTL resolution priority
    -----------------------
    1. Unix timestamp embedded in the slug  (e.g. "btc-updown-15m-1773106200")
       → most accurate for short-duration recurring markets
    2. endDateIso with full datetime        (e.g. "2026-03-10T02:45:00Z")
    3. endDateIso as date-only              (e.g. "2026-03-10")
       → Gamma often returns date-only; we add 24 h so the market is not
         prematurely discarded when the actual end time is unknown.
    """
    tokens = n.get("tokens", [])
    if len(tokens) < 2:
        return None

    # ── End-timestamp resolution ──────────────────────────────────────────────
    slug  = n.get("market_slug", "")
    end_ms = _extract_slug_timestamp_ms(slug)          # method 1

    if end_ms == 0.0:
        end_date_str = n.get("end_date_iso") or ""
        if end_date_str:
            if len(end_date_str) <= 10:                # date-only → add 24 h
                end_ms = _parse_iso_to_ms(end_date_str) + 86_400_000
            else:
                end_ms = _parse_iso_to_ms(end_date_str)

    if end_ms == 0.0:
        return None

    start_ms = _parse_iso_to_ms(n.get("start_date_iso"))

    # Identify YES vs NO by outcome label
    # YES-equivalent: "Yes", "UP", "Up", "YES" (index 0 by convention)
    # NO-equivalent:  "No",  "DOWN", "Down", "NO" (index 1)
    yes_tok = tokens[0]
    no_tok  = tokens[1]

    yes_label = yes_tok.get("outcome", "Yes")
    no_label  = no_tok.get("outcome", "No")

    # If somehow reversed, swap
    no_keywords = {"no", "down", "lower", "fall"}
    if yes_label.lower() in no_keywords:
        yes_tok, no_tok = no_tok, yes_tok
        yes_label, no_label = no_label, yes_label

    ptb = float((n.get("event_metadata") or {}).get("priceToBeat", 0) or 0)
    return ActiveMarket(
        condition_id=n.get("condition_id", ""),
        question=n.get("question", ""),
        yes_token_id=yes_tok.get("token_id", ""),
        yes_outcome=yes_label,
        no_token_id=no_tok.get("token_id", ""),
        no_outcome=no_label,
        end_timestamp_ms=end_ms,
        start_timestamp_ms=start_ms,
        accepting_orders=bool(n.get("accepting_orders", True)),
        api_active=bool(n.get("active", True)),
        api_closed=bool(n.get("closed", False)),
        price_to_beat=ptb,
    )


_GAMMA_EVENTS_URL    = "https://gamma-api.polymarket.com/events"
_PYTH_BENCHMARKS_URL = (
    "https://benchmarks.pyth.network/v1/shims/tradingview/history"
    "?symbol=Crypto.BTC%2FUSD&resolution=1&from={from_ts}&to={to_ts}"
)


async def _fetch_ptb_from_pyth_benchmarks(
    session: "aiohttp.ClientSession", start_ts_secs: float
) -> float:
    """
    Query Pyth Benchmarks for the BTC/USD 1-min close at the exact market
    open timestamp.  This is the most accurate source for priceToBeat on
    BTC 15-min markets (Gamma eventMetadata is empty for these markets).

    Returns 0.0 on any error so the caller can fall through to other sources.
    """
    if start_ts_secs <= 0:
        return 0.0
    ts = int(start_ts_secs)
    url = _PYTH_BENCHMARKS_URL.format(from_ts=ts - 90, to_ts=ts + 30)
    try:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=6)
        ) as resp:
            if resp.status != 200:
                return 0.0
            data = await resp.json(content_type=None)
            if data.get("s") != "ok":
                return 0.0
            timestamps = data.get("t", [])
            closes     = data.get("c", [])
            if not timestamps or not closes:
                return 0.0
            # Find the bar whose close time is closest to (and ≤) start_ts
            best_price = 0.0
            best_diff  = float("inf")
            for t, c in zip(timestamps, closes):
                diff = abs(t - ts)
                if diff < best_diff:
                    best_diff  = diff
                    best_price = float(c)
            if best_price > 0:
                _log.info(
                    "[ptb_pyth] Pyth Benchmarks BTC @ ts=%d → $%.2f  (Δt=%ds)",
                    ts, best_price, int(best_diff),
                )
            return best_price
    except Exception as exc:
        _log.debug("[ptb_pyth] Benchmarks error: %s", exc)
        return 0.0


async def _fetch_ptb_from_polymarket_api(
    session: "aiohttp.ClientSession", start_ts_secs: float
) -> float:
    """
    Fetch priceToBeat from Polymarket's internal crypto-price API.
    This is the most accurate source: openPrice = BTC at market open,
    fixed for the entire 15-min window.

    start_ts_secs : Unix timestamp of market START (slug timestamp).
    """
    if start_ts_secs <= 0:
        return 0.0
    start_iso = datetime.fromtimestamp(start_ts_secs, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso   = datetime.fromtimestamp(start_ts_secs + 900, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = (
        "https://polymarket.com/api/crypto/crypto-price"
        f"?symbol=BTC&eventStartTime={start_iso}&variant=fifteen&endDate={end_iso}"
    )
    try:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=6), ssl=_SSL_NO_VERIFY
        ) as resp:
            if resp.status != 200:
                return 0.0
            data = await resp.json(content_type=None)
            price = data.get("openPrice")
            if price and float(price) > 0:
                _log.info(
                    "[ptb_poly] Polymarket API BTC @ %s → $%.2f",
                    start_iso, float(price),
                )
                return float(price)
    except Exception as exc:
        _log.debug("[ptb_poly] error: %s", exc)
    return 0.0


async def _fetch_ptb_for_event(
    session: aiohttp.ClientSession, event_slug: str
) -> float:
    """
    Fetch priceToBeat from the Gamma Events API.
    The /events endpoint returns eventMetadata even for active markets,
    whereas /markets only includes it after the market is resolved.
    """
    if not event_slug:
        return 0.0
    try:
        async with session.get(
            _GAMMA_EVENTS_URL,
            params={"slug": event_slug},
            timeout=aiohttp.ClientTimeout(total=6),
        ) as resp:
            if resp.status != 200:
                return 0.0
            data = await resp.json(content_type=None)
            events = data if isinstance(data, list) else data.get("data", [])
            for ev in events:
                meta = ev.get("eventMetadata") or {}
                ptb = meta.get("priceToBeat")
                if ptb:
                    _log.info("[ptb_fetch] slug=%s  priceToBeat=%.2f", event_slug, float(ptb))
                    return float(ptb)
    except Exception as exc:
        _log.debug("[ptb_fetch] slug=%s  error=%s", event_slug, exc)
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ROLLOVER WATCHER
# ─────────────────────────────────────────────────────────────────────────────

class RolloverWatcher:
    """
    Polls the Gamma Markets API for a specific recurring series and fires
    callbacks when a new market appears (rollover).

    Poll-interval logic
    -------------------
    - Normal:        config.normal_poll_interval_secs  (default 30 s)
    - Fast window:   config.fast_poll_interval_secs    (default 3 s)
                     when TTL < fast_poll_threshold_secs
    - Post-expiry:   config.post_expiry_poll_interval_secs (default 2 s)
                     while waiting for the next market to appear

    Parameters
    ----------
    config : MarketSeriesConfig
    on_market_found : async (ActiveMarket) -> None
        Called when the initial (or any new) market is first identified.
    on_rollover : async (old: ActiveMarket, new: ActiveMarket) -> None
        Called when a NEW conditionId matching the series appears.
    """

    def __init__(
        self,
        config: MarketSeriesConfig,
        on_market_found: Callable[[ActiveMarket], Awaitable[None]],
        on_rollover: Callable[[ActiveMarket, ActiveMarket], Awaitable[None]],
    ) -> None:
        self._cfg             = config
        self._on_market_found = on_market_found
        self._on_rollover     = on_rollover
        self._current: ActiveMarket | None = None
        self._seen_conditions: set[str] = set()
        self._log = logging.getLogger(self.__class__.__name__)

    # ── Public ────────────────────────────────────────────────────────────────

    @property
    def current_market(self) -> ActiveMarket | None:
        return self._current

    @property
    def poll_interval(self) -> float:
        if self._current is None:
            return self._cfg.normal_poll_interval_secs
        tte = self._current.seconds_to_expiry
        if tte <= 0:
            return self._cfg.post_expiry_poll_interval_secs
        if tte <= self._cfg.fast_poll_threshold_secs:
            return self._cfg.fast_poll_interval_secs
        # Poll quickly while priceToBeat is not yet available from the API
        if self._current.price_to_beat == 0.0:
            return 5.0
        return self._cfg.normal_poll_interval_secs

    async def run(self) -> None:
        """Main watcher loop. Run as an asyncio Task."""
        self._log.info(
            f"[Watcher] Starting — series='{self._cfg.slug_prefix}' "
            f"fast_thresh={self._cfg.fast_poll_threshold_secs}s"
        )
        connector = aiohttp.TCPConnector(ssl=_SSL_NO_VERIFY)
        async with aiohttp.ClientSession(connector=connector) as session:
            while True:
                try:
                    await self._poll(session)
                except Exception as exc:
                    self._log.error(f"[Watcher] Poll error: {exc}")

                interval = self.poll_interval
                if self._current:
                    self._log.debug(
                        f"[Watcher] TTL={self._current.ttl_str}  "
                        f"next_poll={interval:.1f}s"
                    )
                await asyncio.sleep(interval)

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _poll(self, session: aiohttp.ClientSession) -> None:
        markets = await self._fetch_series(session)
        if not markets:
            self._log.debug("[Watcher] No matching markets in poll")
            return

        # Mark markets that are truly done (api_closed=True or TTL < -900s).
        # Do NOT mark just-expired markets as done — Polymarket keeps
        # active=True / closed=False for several minutes past the slug timestamp
        # while the result is pending. We must keep quoting until it's closed.
        for m in markets:
            if m.condition_id and m.is_truly_done:
                self._seen_conditions.add(m.condition_id)

        # "Live" = not truly done AND unseen.
        # Sort by seconds_to_expiry ascending so the most-current market comes
        # first (negative TTL but active beats far-future markets).
        new_live = sorted(
            [m for m in markets
             if m.condition_id
             and not m.is_truly_done
             and m.condition_id not in self._seen_conditions],
            key=lambda m: m.seconds_to_expiry,
        )

        if self._current is None:
            if not new_live:
                self._log.debug("[Watcher] No live markets found yet")
                return
            candidate = new_live[0]
            self._seen_conditions.add(candidate.condition_id)
            self._current = candidate
            self._log.info(f"[Watcher] INITIAL market found: {candidate}")
            await self._on_market_found(candidate)

        else:
            # Refresh current market state from latest poll data.
            fresh = next(
                (m for m in markets
                 if m.condition_id == self._current.condition_id),
                None,
            )
            if fresh:
                self._current = fresh   # update api_active / api_closed

            # Rollover only when current is truly done.
            if not self._current.is_truly_done:
                return

            if not new_live:
                self._log.debug("[Watcher] Current closed — waiting for successor")
                return

            candidate = new_live[0]
            self._seen_conditions.add(candidate.condition_id)
            old = self._current
            self._current = candidate
            self._log.warning(
                f"[Watcher] *** ROLLOVER ***\n"
                f"  OLD: {old}\n"
                f"  NEW: {candidate}"
            )
            await self._on_rollover(old, candidate)

    async def _fetch_series(
        self, session: aiohttp.ClientSession
    ) -> list[ActiveMarket]:
        """
        Fetch active markets from Gamma API for the configured series.

        Strategy
        --------
        1. Direct slug lookup using clock-aligned candidate timestamps.
           BTC-15m slugs follow the pattern ``{prefix}-{unix_end_timestamp}``.
           We generate candidates around the current 15-minute boundary and
           query each one directly — bypassing Gamma's unreliable filters.

        2. Broad generic search (fallback) — fetches 200 markets with no
           active filter and matches client-side by slug prefix / question.
        """
        results: list[ActiveMarket] = []

        # ── Strategy 0: forced slug (env override) ────────────────────────────
        # Always falls through to Strategy 1 so the successor market is fetched
        # before the forced slug expires (enabling seamless rollover).
        if self._cfg.force_slug:
            try:
                async with session.get(
                    self._cfg.gamma_base_url,
                    params={"slug": self._cfg.force_slug},
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    if resp.status == 200:
                        raw = await resp.json(content_type=None)
                        markets_raw = raw if isinstance(raw, list) else raw.get("data", [])
                        for m in markets_raw:
                            normalized = _normalize(m)
                            am = _build_active_market(normalized)
                            if am and not am.is_truly_done:
                                if am not in results:
                                    results.append(am)
                                    self._log.info(
                                        f"[Watcher] Force-slug '{self._cfg.force_slug}': {am}"
                                    )
            except Exception as exc:
                self._log.warning(f"[Watcher] Force-slug fetch error: {exc}")

        # ── Strategy 1: clock-based direct slug lookup ────────────────────────
        period = 900   # 15 minutes in seconds
        now    = int(time.time())
        floor  = (now // period) * period   # current 15-min boundary

        # Candidate end-timestamps: current window + next 3, previous 1
        candidates = [floor + i * period for i in range(-1, 4)]

        for ts in candidates:
            slug = f"{self._cfg.slug_prefix}-{ts}"
            try:
                async with session.get(
                    self._cfg.gamma_base_url,
                    params={"slug": slug},
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    if resp.status != 200:
                        continue
                    raw = await resp.json(content_type=None)
            except Exception:
                continue

            markets_raw = raw if isinstance(raw, list) else raw.get("data", [])
            for m in markets_raw:
                # Only skip archived (permanently removed) markets.
                # Do NOT filter on `closed` — Polymarket sets closed=True when
                # it stops accepting new orders (a few minutes before nominal
                # expiry), but the market is still live and tradeable.
                if m.get("archived"):
                    continue
                normalized = _normalize(m)
                am = _build_active_market(normalized)
                if am is None or am.is_truly_done:
                    continue
                # If priceToBeat is missing, try multiple sources
                if am.price_to_beat == 0.0:
                    ptb = 0.0
                    # Source 0: Polymarket internal crypto-price API (most accurate)
                    ptb = await _fetch_ptb_from_polymarket_api(session, float(ts))
                    # Source 1: market.line (Gamma field for strike price)
                    if not ptb and m.get("line"):
                        try:
                            ptb = float(m["line"])
                            _log.info("[ptb_line] slug=%s  priceToBeat=%.2f (from line)", slug, ptb)
                        except (ValueError, TypeError):
                            ptb = 0.0
                    # Source 2: events API
                    if not ptb:
                        event_slug = (m.get("events") or [{}])[0].get("slug", "")
                        ptb = await _fetch_ptb_for_event(session, event_slug)
                    # Source 3: regex from question/description
                    if not ptb:
                        ptb = _extract_ptb_from_description(m)
                        if ptb > 0:
                            _log.info("[ptb_regex] slug=%s  priceToBeat=%.2f (from description)", slug, ptb)
                    # Source 4: live Pyth price at rollover moment (set in dashboard.publish_rollover)
                    if ptb > 0:
                        am = dataclasses.replace(am, price_to_beat=ptb)
                if am not in results:
                    results.append(am)
                    self._log.debug(
                        f"[Watcher] Found via slug '{slug}': {am}"
                    )

        # Return immediately if Strategy 1 found any live market.
        if any(not am.is_truly_done for am in results):
            return results

        # No live market from slug lookup — try broad search as supplement.
        self._log.debug("[Watcher] No live market in slug lookup — trying broad search")
        try:
            params: dict[str, str] = {
                "archived":  "false",
                "limit":     "200",
                "order":     "endDateIso",
                "ascending": "true",
            }
            async with session.get(
                self._cfg.gamma_base_url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Gamma API {resp.status}")
                raw = await resp.json(content_type=None)

            seen_cids = {am.condition_id for am in results}
            markets_raw = raw if isinstance(raw, list) else raw.get("data", [])
            for m in markets_raw:
                if m.get("archived"):
                    continue
                normalized = _normalize(m)
                slug     = normalized.get("market_slug", "").lower()
                question = normalized.get("question", "").lower()
                slug_match = slug.startswith(self._cfg.slug_prefix.lower())
                kw_match   = all(kw.lower() in question
                                 for kw in self._cfg.question_keywords)
                if not (slug_match or kw_match):
                    continue
                am = _build_active_market(normalized)
                if am is None or am.is_truly_done:
                    continue
                if am.condition_id not in seen_cids:
                    seen_cids.add(am.condition_id)
                    results.append(am)

        except Exception as exc:
            # Non-fatal — return whatever Strategy 1 already found.
            self._log.warning(f"[Watcher] Broad search failed (non-fatal): {exc}")

        return results


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ROLLOVER MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class PolymarketRolloverManager:
    """
    Orchestrates the full lifecycle of a recurring market series.

    Responsibilities
    ----------------
    1. Runs a RolloverWatcher to detect new markets.
    2. On market found/rollover:
       - Subscribes new YES + NO token IDs via the IngestionEngine.
       - Unsubscribes expired token IDs.
       - Resets MarketBookState and notifies the strategy.
    3. Routes raw TradingTicks from the engine to the correct YES/NO leg
       and builds YesNoMarketTick objects for the strategy.

    Parameters
    ----------
    engine : IngestionEngine
        The running ingestion engine (already connected or about to be started).
    config : MarketSeriesConfig
        Series configuration (slug, fast-poll thresholds, …).
    on_yesno_tick : async (YesNoMarketTick) -> None
        Strategy callback. Called on every YES or NO update, with full
        combined state including price_sum and ts_spread_ms.
    on_rollover_signal : async (old, new) -> None, optional
        Strategy callback fired when a rollover is detected (before the new
        book state arrives). Use to cancel open orders / reset inventory.
    """

    def __init__(
        self,
        engine: IngestionEngine,
        config: MarketSeriesConfig,
        on_yesno_tick: Callable[[YesNoMarketTick], Awaitable[None]],
        on_rollover_signal: (
            Callable[[ActiveMarket, ActiveMarket], Awaitable[None]] | None
        ) = None,
    ) -> None:
        self._engine = engine
        self._cfg    = config
        self._on_yesno_tick      = on_yesno_tick
        self._on_rollover_signal = on_rollover_signal
        self._log = logging.getLogger(self.__class__.__name__)

        # Active state
        self._current_market: ActiveMarket | None   = None
        self._book_state: MarketBookState | None    = None

        # token_id → "yes" | "no" routing table
        self._token_role: dict[str, str] = {}

        # Watcher
        self._watcher = RolloverWatcher(
            config=config,
            on_market_found=self._on_market_found,
            on_rollover=self._on_rollover,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def current_market(self) -> ActiveMarket | None:
        return self._current_market

    @property
    def book_state(self) -> MarketBookState | None:
        return self._book_state

    async def run(self) -> None:
        """
        Start the rollover watcher loop.
        Must be run as a Task alongside IngestionEngine.start().

        The engine must have been started (or have connect() called) before
        this coroutine makes WS subscriptions.
        """
        # Register our tick router into the engine's parser
        # We monkey-patch the engine's on_tick to route through us first.
        # A cleaner alternative is to pass this manager's handler as
        # the engine's on_tick when constructing the engine.
        self._log.info("[RolloverMgr] Watcher started")
        await self._watcher.run()   # blocks until cancelled

    def build_tick_handler(
        self,
    ) -> Callable[[TradingTick], Awaitable[None]]:
        """
        Returns an async on_tick callable to pass to IngestionEngine.

        Example
        -------
        manager = PolymarketRolloverManager(...)
        engine  = IngestionEngine(on_tick=manager.build_tick_handler(), ...)
        """
        async def _handler(tick: TradingTick) -> None:
            await self._route_tick(tick)
        return _handler

    # ── Rollover callbacks ────────────────────────────────────────────────────

    async def _on_market_found(self, market: ActiveMarket) -> None:
        """Called by RolloverWatcher when the first (or initial) market appears."""
        self._log.info(f"[RolloverMgr] Initial market → {market}")
        await self._activate_market(market)

    async def _on_rollover(
        self, old: ActiveMarket, new: ActiveMarket
    ) -> None:
        """Called by RolloverWatcher on every new conditionId."""
        self._log.warning(
            f"[RolloverMgr] ROLLOVER  "
            f"{old.condition_id[:14]}.. → {new.condition_id[:14]}.."
        )

        # 1. Notify strategy FIRST — it should cancel orders / zero inventory
        if self._on_rollover_signal:
            try:
                await self._on_rollover_signal(old, new)
            except Exception as exc:
                self._log.error(f"[RolloverMgr] rollover_signal callback error: {exc}")

        # 2. Unsubscribe expired tokens
        for token_id in (old.yes_token_id, old.no_token_id):
            if token_id:
                try:
                    await self._engine.unsubscribe(token_id)
                except Exception as exc:
                    self._log.warning(
                        f"[RolloverMgr] Unsubscribe {token_id[:14]}.. failed: {exc}"
                    )

        self._token_role.pop(old.yes_token_id, None)
        self._token_role.pop(old.no_token_id,  None)

        # 3. Subscribe new tokens
        await self._activate_market(new)

    async def _activate_market(self, market: ActiveMarket) -> None:
        """Subscribe to YES + NO tokens and reset book state."""
        self._current_market = market
        self._book_state = MarketBookState(
            condition_id=market.condition_id,
            yes_token_id=market.yes_token_id,
            no_token_id=market.no_token_id,
        )

        # Register routing table
        self._token_role[market.yes_token_id] = "yes"
        self._token_role[market.no_token_id]  = "no"

        # Register token metadata in engine so MarketState is keyed correctly
        now_ms = time.time() * 1_000
        for token_id, outcome in (
            (market.yes_token_id, market.yes_outcome),
            (market.no_token_id,  market.no_outcome),
        ):
            self._engine._token_map[token_id] = TokenInfo(
                token_id=token_id,
                condition_id=market.condition_id,
                question=market.question,
                outcome=outcome,
                first_seen_ms=now_ms,
            )
            await self._engine._feed.subscribe(token_id)

        # Force WS reconnect so the new tokens are subscribed on a fresh
        # connection — Polymarket WS does not reliably accept dynamic
        # subscribe messages on existing connections.
        await self._engine._feed.force_reconnect()

        self._log.info(
            f"[RolloverMgr] Activated  "
            f"YES={market.yes_token_id[:14]}..  "
            f"NO={market.no_token_id[:14]}..  "
            f"TTL={market.ttl_str}"
        )

    # ── Tick routing ──────────────────────────────────────────────────────────

    async def _route_tick(self, tick: TradingTick) -> None:
        """
        Route an incoming TradingTick to the YES or NO leg of the book state
        and emit a YesNoMarketTick if both sides have been seen at least once.
        """
        role = self._token_role.get(tick.token_id)
        if role is None or self._book_state is None:
            return  # tick for an untracked token (other series, expired)

        bs = self._book_state

        if role == "yes":
            bs.yes_bid      = tick.bid
            bs.yes_ask      = tick.ask
            bs.yes_bid_size = tick.bid_size
            bs.yes_ask_size = tick.ask_size
            bs.yes_ts_ms    = tick.timestamp_ms
        else:  # "no"
            bs.no_bid      = tick.bid
            bs.no_ask      = tick.ask
            bs.no_bid_size = tick.bid_size
            bs.no_ask_size = tick.ask_size
            bs.no_ts_ms    = tick.timestamp_ms

        # Emit as soon as at least one side has data.
        # The other side keeps its default values (bid=0.001, ask=0.999)
        # until its first update arrives (usually within milliseconds
        # since Polymarket sends both snapshots simultaneously on subscribe).
        if bs.yes_ts_ms == 0.0 and bs.no_ts_ms == 0.0:
            return

        ynt = YesNoMarketTick(
            condition_id=bs.condition_id,
            yes_token_id=bs.yes_token_id,
            no_token_id=bs.no_token_id,
            yes_bid=bs.yes_bid,
            yes_ask=bs.yes_ask,
            yes_mid=bs.yes_mid,
            yes_spread=bs.yes_spread,
            yes_bid_size=bs.yes_bid_size,
            yes_ask_size=bs.yes_ask_size,
            yes_ts_ms=bs.yes_ts_ms,
            no_bid=bs.no_bid,
            no_ask=bs.no_ask,
            no_mid=bs.no_mid,
            no_spread=bs.no_spread,
            no_bid_size=bs.no_bid_size,
            no_ask_size=bs.no_ask_size,
            no_ts_ms=bs.no_ts_ms,
            price_sum=bs.price_sum,
            price_deviation=bs.price_deviation,
            ts_spread_ms=bs.ts_spread_ms,
            source=tick.source,
        )

        try:
            await self._on_yesno_tick(ynt)
        except Exception as exc:
            _log.error(f"[RolloverMgr] on_yesno_tick callback error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  UTILITY: map YesNoMarketTick → flat dict for strategy/dashboard
# ─────────────────────────────────────────────────────────────────────────────

def map_yesno_tick(ynt: YesNoMarketTick) -> dict[str, Any]:
    """
    Flat representation of a YesNoMarketTick for logging, UI, or order logic.
    """
    return {
        "condition_id":    ynt.condition_id,
        "yes_token_id":    ynt.yes_token_id,
        "no_token_id":     ynt.no_token_id,
        # YES quotes
        "yes_bid":         ynt.yes_bid,
        "yes_ask":         ynt.yes_ask,
        "yes_mid":         ynt.yes_mid,
        "yes_spread":      ynt.yes_spread,
        "yes_bid_size":    ynt.yes_bid_size,
        "yes_ask_size":    ynt.yes_ask_size,
        "yes_ts_ms":       ynt.yes_ts_ms,
        # NO quotes
        "no_bid":          ynt.no_bid,
        "no_ask":          ynt.no_ask,
        "no_mid":          ynt.no_mid,
        "no_spread":       ynt.no_spread,
        "no_bid_size":     ynt.no_bid_size,
        "no_ask_size":     ynt.no_ask_size,
        "no_ts_ms":        ynt.no_ts_ms,
        # Arbitrage metrics
        "price_sum":       ynt.price_sum,
        "price_deviation": ynt.price_deviation,
        "ts_spread_ms":    ynt.ts_spread_ms,
        "source":          ynt.source,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  DEMO ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def on_yesno_tick(ynt: YesNoMarketTick) -> None:
    dev_flag = "⚠ ARB" if ynt.price_deviation > 0.02 else "   ok"
    _log.info(
        "[YN] %s | "
        "YES bid=%.4f ask=%.4f spd=%.4f | "
        "NO  bid=%.4f ask=%.4f spd=%.4f | "
        "sum=%.4f dev=%.5f sync=%.0fms | src=%s",
        dev_flag,
        ynt.yes_bid, ynt.yes_ask, ynt.yes_spread,
        ynt.no_bid,  ynt.no_ask,  ynt.no_spread,
        ynt.price_sum, ynt.price_deviation, ynt.ts_spread_ms,
        ynt.source,
    )


async def on_rollover_signal(old: ActiveMarket, new: ActiveMarket) -> None:
    _log.warning(
        "[ROLLOVER SIGNAL] Cancel all orders! "
        "old_cond=%s..  new_cond=%s..",
        old.condition_id[:14], new.condition_id[:14],
    )


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = MarketSeriesConfig(
        slug_prefix="btc-updown-15m",
        question_keywords=["btc", "15"],
        fast_poll_threshold_secs=120.0,
        fast_poll_interval_secs=3.0,
        normal_poll_interval_secs=30.0,
        post_expiry_poll_interval_secs=2.0,
    )

    # 1. Build the rollover manager first so we can pass its tick handler
    #    to the engine (chicken-and-egg: use a temporary async queue)
    _tick_q: asyncio.Queue[TradingTick] = asyncio.Queue()

    async def _engine_on_tick(tick: TradingTick) -> None:
        await _tick_q.put(tick)

    engine = IngestionEngine(
        on_tick=_engine_on_tick,
        token_ids=[],
        discovery_enabled=False,   # RolloverWatcher handles discovery
    )

    manager = PolymarketRolloverManager(
        engine=engine,
        config=cfg,
        on_yesno_tick=on_yesno_tick,
        on_rollover_signal=on_rollover_signal,
    )

    # Route engine ticks through the manager
    async def _dispatch_to_manager(tick: TradingTick) -> None:
        await manager._route_tick(tick)

    # Patch on_tick to both queue and dispatch
    engine._parser._on_tick = _dispatch_to_manager  # type: ignore[assignment]

    try:
        await asyncio.gather(
            engine.start(),         # connects WS feed, blocks
            manager.run(),          # watcher loop, blocks
        )
    except KeyboardInterrupt:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
