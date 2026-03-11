#!/usr/bin/env python3
"""
dashboard.py - Localhost Live Dashboard for the Polymarket MM Bot
=================================================================
FastAPI backend that:
  - Serves the HTML dashboard from static/index.html
  - Exposes a WebSocket endpoint /ws/quotes that streams YesNoMarketTick
    data to all connected browser clients
  - Bridges the asyncio ingestion pipeline to the dashboard via a Queue

Architecture:
  IngestionEngine (on_tick) ──► asyncio.Queue ──► WebSocket /ws/quotes
                                                         │
                                                   browser clients

pip deps:
  pip install fastapi uvicorn[standard] websockets aiohttp msgspec

Usage (standalone — launches engine + watcher + dashboard):
  python dashboard.py

Then open:  http://localhost:8080
"""

import asyncio
import json
import logging
import os
import ssl
import sys
import time
from pathlib import Path
from typing import Any

_MARKET_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "market_state.json")


def _save_market_state(strike_k: float) -> None:
    """Persist strike_k so it survives bot restarts within the same market session."""
    try:
        with open(_MARKET_STATE_PATH, "w") as fh:
            json.dump({"strike_k": strike_k, "saved_at": time.time()}, fh)
    except Exception:
        pass


def _load_market_state() -> float:
    """
    Return strike_k from state file if saved within the last 15 minutes
    (a single Polymarket 15-min session).  Returns 0.0 if stale or missing.
    """
    try:
        if not os.path.isfile(_MARKET_STATE_PATH):
            return 0.0
        with open(_MARKET_STATE_PATH) as fh:
            data = json.load(fh)
        age = time.time() - data.get("saved_at", 0)
        if age > 900:          # older than one full market session → discard
            return 0.0
        k = float(data.get("strike_k", 0.0))
        if k > 0:
            logging.getLogger("System").info(
                "[SYSTEM] Restored strike_k=%.2f from state file (age=%.0fs)", k, age
            )
        return k
    except Exception:
        return 0.0

from strategy_engine import PredictionMarketPricer
from analytics_engine import AnalyticsEngine
from paper_trading_engine import PaperTradingEngine
from telegram_notifier import TelegramNotifier

# SSL context that skips certificate verification (macOS cert bundle fix)
_SSL_NO_VERIFY = ssl.create_default_context()
_SSL_NO_VERIFY.check_hostname = False
_SSL_NO_VERIFY.verify_mode = ssl.CERT_NONE

# ── FastAPI / Uvicorn ─────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError as _e:
    raise RuntimeError("pip install fastapi 'uvicorn[standard]'") from _e

# ── Local modules ─────────────────────────────────────────────────────────────
from ingestion_engine import IngestionEngine, TradingTick
from rollover_manager import (
    PolymarketRolloverManager,
    MarketSeriesConfig,
    YesNoMarketTick,
    ActiveMarket,
    map_yesno_tick,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DASHBOARD_HOST    = os.getenv("DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT    = int(os.getenv("DASHBOARD_PORT", "8080"))
STATIC_DIR        = Path(__file__).parent / "static"
LOG_LEVEL         = os.getenv("LOG_LEVEL", "INFO").upper()
FORCE_MARKET_SLUG = os.getenv("FORCE_MARKET_SLUG", "")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE  (shared between FastAPI and the asyncio pipeline)
# ─────────────────────────────────────────────────────────────────────────────

# One queue per connected WS client (fan-out pattern)
_ws_clients: set[asyncio.Queue] = set()

# Latest tick cached for clients that connect mid-stream
_latest_tick_dict: dict[str, Any] | None = None
_latest_rollover_event: dict[str, Any] | None = None

# Pipeline references for /api/status
_manager_ref: "PolymarketRolloverManager | None" = None
_tick_count: int = 0

# Throttle flag: publish_yesno_tick marks dirty, broadcaster_loop flushes every 100ms
_book_dirty: bool = False

# BTC price state — updated by Pyth price feed loop
_btc_current: float = 0.0

# priceToBeat captured at rollover moment (Pyth price = BTC at market boundary)
# Pre-seeded from state file so a restart mid-session doesn't lose the strike K.
_ptb_at_rollover: float = _load_market_state()

# ── Strategy / Kernel layer ───────────────────────────────────────────────────
# Single pricer instance shared across all async tasks (all run on same loop)
_pricer: PredictionMarketPricer = PredictionMarketPricer()
_strategy_log = logging.getLogger("Strategy")

# ── Analytics layer ────────────────────────────────────────────────────────────
_analytics: AnalyticsEngine = AnalyticsEngine()
_analytics_log = logging.getLogger("Analytics")

# ── Telegram notifier ──────────────────────────────────────────────────────────
_tg_notifier = TelegramNotifier(
    token       = "8603640008:AAEh8FmkjgOQhNQH1WnhHkKjO815Yz8ko-Y",
    chat_id     = ["172349632", "971957662"],
    asset_label = "BTC 15m",
)

# ── Paper trading layer ────────────────────────────────────────────────────────
_pte: PaperTradingEngine = PaperTradingEngine(notifier=_tg_notifier)

# ─────────────────────────────────────────────────────────────────────────────
# BRIDGE FUNCTION: called by the MM pipeline on every tick
# ─────────────────────────────────────────────────────────────────────────────

async def publish_yesno_tick(ynt: YesNoMarketTick) -> None:
    """
    Called by PolymarketRolloverManager.on_yesno_tick on every WS message.
    Only updates internal state — does NOT broadcast directly.
    The broadcaster_loop flushes to clients every 100ms.
    """
    global _latest_tick_dict, _tick_count, _book_dirty
    _tick_count += 1

    d = map_yesno_tick(ynt)
    d["type"]         = "tick"
    d["server_ts_ms"] = time.time() * 1_000

    # Attach full order book depth + market metadata
    if _manager_ref is not None:
        mkt = _manager_ref.current_market
        if mkt:
            d["end_timestamp_ms"] = mkt.end_timestamp_ms
            d["question"]         = mkt.question
            d["price_to_beat"]    = mkt.price_to_beat
        engine = _manager_ref._engine
        yes_st = engine._market_states.get(ynt.yes_token_id)
        no_st  = engine._market_states.get(ynt.no_token_id)
        d["yes_bids"] = yes_st.top_bids(8) if yes_st else []
        d["yes_asks"] = yes_st.top_asks(8) if yes_st else []
        d["no_bids"]  = no_st.top_bids(8)  if no_st  else []
        d["no_asks"]  = no_st.top_asks(8)  if no_st  else []

    d["btc_current"] = _btc_current

    # ── Kernel / Strategy layer ───────────────────────────────────────────────
    # Seed kernel with market data when strike_k is not yet set.
    # Priority: API price_to_beat → rollover-captured Pyth price → no K (market_mid fallback)
    if _pricer.strike_k == 0.0 and _manager_ref is not None:
        mkt = _manager_ref.current_market
        if mkt:
            k = mkt.price_to_beat or _ptb_at_rollover   # use UI-captured value as fallback
            if k > 0:
                _pricer.reset_market(
                    strike_k=k,
                    seconds_to_expiry=mkt.seconds_to_expiry,
                )
                _save_market_state(k)
            else:
                # No K available yet — at least seed tau so A-S spreads are correct
                _pricer.tau_secs = max(0.0, mkt.seconds_to_expiry)

    # Ensure BTC spot is always in sync (in case pyth_price_loop hasn't fired yet)
    if _pricer.btc_spot == 0.0 and _btc_current > 0:
        _pricer.btc_spot = _btc_current

    # Tick filter: only run kernel when YES price changes (not size-only)
    price_changed = _pricer.process_tick(ynt.yes_bid, ynt.yes_ask)
    quote = _pricer.decide(ynt.yes_bid, ynt.yes_ask)

    d["bs_fair"]         = quote.fair_price
    d["bs_bid"]          = quote.theoretical_bid
    d["bs_ask"]          = quote.theoretical_ask
    d["bs_action"]       = quote.action
    d["bs_action_price"] = quote.action_price
    d["bs_sigma_b"]      = quote.sigma_b
    d["bs_tau_secs"]     = quote.tau_secs
    d["bs_inventory"]    = quote.inventory
    d["bs_fair_source"]  = quote.fair_source   # "lognormal" | "market_mid" | "prior"
    d["bs_btc_spot"]     = quote.btc_spot
    d["bs_strike_k"]     = quote.strike_k
    d["bs_cutoff"]       = _pricer.in_cutoff   # True when T ≤ HARD_CUTOFF_SECONDS
    # NO/DOWN kernel (derived by complement)
    d["bs_no_fair"]         = quote.no_fair_price
    d["bs_no_bid"]          = quote.no_theoretical_bid
    d["bs_no_ask"]          = quote.no_theoretical_ask
    d["bs_no_action"]       = quote.no_action
    d["bs_no_action_price"] = quote.no_action_price

    if price_changed:
        _strategy_log.info(_pricer.format_log_line(ynt.yes_bid, ynt.yes_ask, quote))

    # ── Analytics layer ───────────────────────────────────────────────────────
    # Always seed these keys so the UI panel always gets valid JSON fields.
    d["an_rv"]      = None
    d["an_iv"]      = None
    d["an_vol_gap"] = None
    d["an_vega"]    = None
    d["an_delta"]   = None
    d["an_gamma"]   = None
    d["an_theta"]   = None
    d["an_rv_n"]    = len(_analytics._price_log)

    # Full compute only when K, S, and T are all known.
    snap = None
    if _pricer.strike_k > 0 and _pricer.btc_spot > 0 and _pricer.tau_secs >= 1.0:
        snap = _analytics.compute(
            btc_spot  = _pricer.btc_spot,
            strike_k  = _pricer.strike_k,
            tau_secs  = _pricer.tau_secs,
            yes_bid   = ynt.yes_bid,
            yes_ask   = ynt.yes_ask,
        )
        if snap is not None:
            # Update pricer's sigma_btc with live RV (replaces hardcoded 0.80)
            if snap.rv is not None and snap.rv > 0:
                _pricer.sigma_btc = snap.rv
            if price_changed:
                _analytics_log.info(_analytics.format_log_line(snap))
            d["an_rv"]      = round(snap.rv * 100, 2)      if snap.rv      is not None else None
            d["an_iv"]      = round(snap.iv * 100, 2)      if snap.iv      is not None else None
            d["an_vol_gap"] = round(snap.vol_gap * 100, 2) if snap.vol_gap is not None else None
            d["an_vega"]    = round(snap.vega,  4)         if snap.vega    is not None else None
            d["an_delta"]   = round(snap.delta, 6)         if snap.delta   is not None else None
            d["an_gamma"]   = snap.gamma                   if snap.gamma   is not None else None
            d["an_theta"]   = round(snap.theta, 6)         if snap.theta   is not None else None
            d["an_rv_n"]    = snap.rv_samples

    # ── Paper trading layer (PaperTradingEngine) ─────────────────────────────
    _bs_fair_yes_safe = quote.fair_price if quote.fair_price is not None else 0.5
    _mkt_bid_yes_safe = ynt.yes_bid      if ynt.yes_bid      is not None else 0.5
    _mkt_ask_yes_safe = ynt.yes_ask      if ynt.yes_ask      is not None else 0.5

    _pte.update_tick(
        bs_fair_yes    = _bs_fair_yes_safe,
        mkt_bid_yes    = _mkt_bid_yes_safe,
        mkt_ask_yes    = _mkt_ask_yes_safe,
        time_to_expiry = _pricer.tau_secs,
        vega           = d["an_vega"],   # None when tau < 5s or IV unavailable
        iv             = snap.iv if (snap is not None) else None,  # raw 0-1 for notifier
    )

    # Actual taker edges the engine computes (for UI transparency)
    _fair_r = round(_bs_fair_yes_safe, 2)  # matches _round_tick inside PTE
    _long_yes_edge = round(_fair_r - _mkt_ask_yes_safe, 4)
    _long_no_edge  = round(_mkt_bid_yes_safe - _fair_r, 4)
    d["pt_long_yes_edge"] = _long_yes_edge
    d["pt_long_no_edge"]  = _long_no_edge
    d["pt_min_edge"]      = _pte.min_edge   # current effective threshold

    pt_stats = _pte.stats(
        mkt_bid_yes = _mkt_bid_yes_safe,
        mkt_ask_yes = _mkt_ask_yes_safe,
    )
    d["pt_balance"]     = pt_stats.balance
    d["pt_equity"]      = pt_stats.equity
    d["pt_inventory"]   = pt_stats.inventory.value
    d["pt_shares"]      = pt_stats.shares_open
    d["pt_avg_entry"]   = pt_stats.entry_price
    d["pt_exit_target"] = pt_stats.exit_target
    d["pt_current_bid"] = pt_stats.current_bid
    d["pt_upnl"]        = pt_stats.unrealized_pnl
    d["pt_uroi"]        = pt_stats.uroi_pct
    d["pt_rpnl"]        = pt_stats.realized_pnl
    d["pt_total_pnl"]   = pt_stats.total_pnl
    d["pt_trades"]      = pt_stats.trade_count
    d["pt_wins"]        = pt_stats.win_count
    d["pt_losses"]      = pt_stats.loss_count
    d["pt_last_close"]  = pt_stats.last_close
    # Performance analytics
    p = pt_stats.perf
    if p:
        d["pt_win_rate"]  = p.win_rate_pct
        d["pt_pf"]        = p.profit_factor if p.profit_factor != float("inf") else 999.0
        d["pt_avg_win"]   = p.avg_win
        d["pt_avg_loss"]  = p.avg_loss
        d["pt_max_dd"]    = p.max_drawdown_pct
        d["pt_roi"]       = p.roi_pct
        d["pt_sharpe"]    = p.sharpe_ratio
    else:
        for k in ("pt_win_rate", "pt_pf", "pt_avg_win", "pt_avg_loss", "pt_max_dd", "pt_roi", "pt_sharpe"):
            d[k] = None
    # Trade log (last 20 trades for UI table)
    d["pt_trade_log"]      = pt_stats.trade_log[:20]
    # Live open positions for UI portfolio panel
    d["pt_open_positions"] = pt_stats.open_positions_list
    # Full equity curve (all closed trades)
    d["pt_equity_curve"]   = pt_stats.equity_curve

    _latest_tick_dict = d
    _book_dirty = True


def _broadcast_payload(payload: str) -> None:
    """Fan-out a serialised JSON string to all connected WS clients."""
    dead: list[asyncio.Queue] = []
    for q in _ws_clients:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _ws_clients.discard(q)


async def broadcaster_loop() -> None:
    """
    Runs every 100ms. Flushes the latest tick to all WS clients if new data
    arrived since the last broadcast. Sends a keepalive heartbeat every 20s
    to prevent the WS from timing out during quiet periods (e.g. post-rollover).
    """
    global _book_dirty
    _last_broadcast = time.time()
    _HEARTBEAT_INTERVAL = 20.0

    while True:
        await asyncio.sleep(0.1)   # 100ms
        now = time.time()
        if _book_dirty and _latest_tick_dict is not None:
            _book_dirty = False
            _broadcast_payload(json.dumps(_latest_tick_dict))
            _last_broadcast = now
        elif now - _last_broadcast >= _HEARTBEAT_INTERVAL:
            _broadcast_payload(json.dumps({"type": "heartbeat", "ts": now}))
            _last_broadcast = now


# ─────────────────────────────────────────────────────────────────────────────
# PYTH PRICE FEED  (real-time BTC/USD from Pyth Network WS)
# ─────────────────────────────────────────────────────────────────────────────

_PYTH_WS_URL  = "wss://hermes.pyth.network/ws"
_PYTH_BTC_ID  = "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43"

async def pyth_price_loop() -> None:
    """
    Connects to Pyth Network WebSocket and streams real-time BTC/USD price.
    Updates _btc_current and marks the latest tick dirty so the UI refreshes.
    Reconnects automatically on any error.
    """
    global _btc_current, _book_dirty
    log = logging.getLogger("PythFeed")

    import websockets as _ws

    while True:
        try:
            async with _ws.connect(_PYTH_WS_URL, ping_interval=20, ssl=_SSL_NO_VERIFY) as ws:
                await ws.send(json.dumps({
                    "ids":  [_PYTH_BTC_ID],
                    "type": "subscribe",
                }))
                log.info("[Pyth] Connected — streaming BTC/USD")

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        if msg.get("type") != "price_update":
                            continue
                        feed = msg.get("price_feed", {})
                        if feed.get("id", "").lstrip("0") != _PYTH_BTC_ID:
                            continue
                        p    = feed["price"]
                        price = float(p["price"]) * (10 ** int(p["expo"]))
                        if price > 0:
                            _btc_current = price
                            _book_dirty  = True   # piggyback on next tick broadcast
                            _pricer.update_btc(price)  # recompute fair value
                            _analytics.add_pyth_price(price)  # feed RV rolling window
                    except Exception:
                        pass

        except Exception as exc:
            log.warning(f"[Pyth] WS error: {exc} — reconnecting in 5s")
            await asyncio.sleep(5)


async def theta_decay_loop() -> None:
    """
    Runs every second. Pushes time-to-expiry decay into the pricer (theta effect).
    Marks tick dirty if theoretical quotes changed, so the UI refreshes even
    during quiet periods (no new WS ticks).
    """
    global _book_dirty
    while True:
        await asyncio.sleep(1.0)
        if _manager_ref is not None:
            mkt = _manager_ref.current_market
            if mkt and not mkt.is_expired:
                changed = _pricer.update_tau(mkt.seconds_to_expiry)
                if changed:
                    _book_dirty = True


async def publish_rollover(old: ActiveMarket, new: ActiveMarket) -> None:
    """
    Called on rollover events — broadcasts a rollover notification to the UI.
    """
    global _latest_rollover_event, _ptb_at_rollover
    _log = logging.getLogger("publish_rollover")
    # Capture Pyth price at rollover boundary — this IS the priceToBeat
    # for the new market (BTC price at the exact market-open moment).
    # Only use if API didn't provide it already.
    if new.price_to_beat == 0.0 and _btc_current > 0:
        new.price_to_beat = _btc_current
        _ptb_at_rollover  = _btc_current
        _log.info("[rollover] priceToBeat captured at rollover: %.2f", _btc_current)

    # Reset kernel: new strike K = price_to_beat, new T = seconds to expiry
    _pricer.reset_market(
        strike_k=new.price_to_beat,
        seconds_to_expiry=new.seconds_to_expiry,
    )
    _save_market_state(new.price_to_beat)
    # Close paper position at last known mid and start fresh for new market
    _pte.reset_market()
    # Reset KDE microstructure buffer and start 60s entry warm-up
    _pte.trigger_kde_rollover(warmup_secs=60.0)
    logging.getLogger("System").warning(
        "[SYSTEM] Rollover detected. KDE Reset. Waiting for warm-up…"
    )
    d = {
        "type": "rollover",
        "old_condition_id": old.condition_id,
        "old_yes_token":    old.yes_token_id,
        "old_no_token":     old.no_token_id,
        "new_condition_id": new.condition_id,
        "new_yes_token":    new.yes_token_id,
        "new_no_token":     new.no_token_id,
        "new_question":     new.question,
        "new_ttl_str":      new.ttl_str,
        "server_ts_ms":     time.time() * 1_000,
    }
    _latest_rollover_event = d
    # Clear stale tick so reconnecting clients don't receive the old market's book
    global _latest_tick_dict
    _latest_tick_dict = None

    payload = json.dumps(d)
    for q in list(_ws_clients):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Polymarket MM Dashboard", version="1.0")

# Serve static files (index.html, etc.)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root() -> FileResponse:
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return JSONResponse({"error": "index.html not found in static/"}, status_code=404)
    return FileResponse(str(html_path))


@app.get("/api/latest")
async def api_latest() -> JSONResponse:
    """REST fallback: returns the last received tick as JSON."""
    if _latest_tick_dict is None:
        return JSONResponse({"status": "no_data_yet"})
    return JSONResponse(_latest_tick_dict)


@app.post("/api/reset-stats")
async def api_reset_stats() -> JSONResponse:
    """Reset paper trading stats: clear all trades, restore $10,000 balance, wipe CSV."""
    _pte.reset_all_stats()
    logging.getLogger("System").warning(
        "[SYSTEM] Paper trading stats manually reset via /api/reset-stats"
    )
    return JSONResponse({"status": "ok", "message": "Stats reset. Balance restored to $10,000."})


@app.get("/api/status")
async def api_status() -> JSONResponse:
    """Debug endpoint: shows the current pipeline state."""
    m = _manager_ref
    current = m.current_market if m else None
    book    = m.book_state    if m else None
    token_role = dict(m._token_role) if m else {}

    return JSONResponse({
        "ws_clients_connected": len(_ws_clients),
        "tick_count":           _tick_count,
        "has_tick_data":        _latest_tick_dict is not None,
        "current_market": {
            "condition_id":  current.condition_id  if current else None,
            "question":      current.question       if current else None,
            "yes_token_id":  current.yes_token_id   if current else None,
            "no_token_id":   current.no_token_id    if current else None,
            "ttl_str":       current.ttl_str         if current else None,
            "is_expired":    current.is_expired      if current else None,
        },
        "book_state": book.to_dict() if book else None,
        "token_roles": {k[:20]+"..": v for k, v in token_role.items()},
        "last_tick_preview": {
            k: v for k, v in (_latest_tick_dict or {}).items()
            if k in ("yes_bid","yes_ask","no_bid","no_ask","price_sum","price_deviation","source")
        },
    })


@app.websocket("/ws/quotes")
async def ws_quotes(websocket: WebSocket) -> None:
    """
    WebSocket endpoint.
    Each connected client gets its own asyncio.Queue.
    The pipeline pushes serialised JSON strings into every queue.
    """
    await websocket.accept()
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=50)
    _ws_clients.add(q)

    log = logging.getLogger("ws_quotes")
    log.info(f"[WS] Client connected — total={len(_ws_clients)}")

    # Send current snapshot immediately so the UI isn't blank
    if _latest_tick_dict is not None:
        try:
            await websocket.send_text(json.dumps(_latest_tick_dict))
        except Exception:
            pass

    try:
        while True:
            msg = await asyncio.wait_for(q.get(), timeout=60.0)
            await websocket.send_text(msg)
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    except Exception as exc:
        log.warning(f"[WS] Client error: {exc}")
    finally:
        _ws_clients.discard(q)
        log.info(f"[WS] Client disconnected — total={len(_ws_clients)}")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE SETUP
# ─────────────────────────────────────────────────────────────────────────────

async def build_and_run_pipeline() -> None:
    """
    Builds the IngestionEngine + PolymarketRolloverManager pipeline and
    runs it as a long-lived asyncio coroutine alongside Uvicorn.
    """
    log = logging.getLogger("pipeline")

    cfg = MarketSeriesConfig(
        slug_prefix="btc-updown-15m",
        question_keywords=["btc", "15"],
        fast_poll_threshold_secs=120.0,
        fast_poll_interval_secs=3.0,
        normal_poll_interval_secs=30.0,
        post_expiry_poll_interval_secs=1.0,
        force_slug=FORCE_MARKET_SLUG,
    )

    # Placeholder on_tick — will be replaced below
    async def _noop_tick(tick: TradingTick) -> None:
        pass

    engine = IngestionEngine(
        on_tick=_noop_tick,
        token_ids=[],
        discovery_enabled=False,   # RolloverWatcher owns discovery
    )

    manager = PolymarketRolloverManager(
        engine=engine,
        config=cfg,
        on_yesno_tick=publish_yesno_tick,     # ← feeds the dashboard
        on_rollover_signal=publish_rollover,   # ← notifies UI of rollover
    )

    # Expose manager for /api/status
    global _manager_ref
    _manager_ref = manager

    # Wire engine ticks → rollover manager router
    engine._parser._on_tick = manager._route_tick  # type: ignore[assignment]

    log.info("[Pipeline] Starting engine + rollover watcher …")
    try:
        await asyncio.gather(
            engine.start(),
            manager.run(),
        )
    except asyncio.CancelledError:
        log.info("[Pipeline] Cancelled — shutting down engine")
        await engine.stop()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def _run_all() -> None:
    """Run Uvicorn + the MM pipeline on the same event loop."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    uv_config = uvicorn.Config(
        app=app,
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        log_level=LOG_LEVEL.lower(),
        loop="none",   # use the already-running asyncio loop
    )
    server = uvicorn.Server(uv_config)

    pipeline_task    = asyncio.create_task(build_and_run_pipeline())
    server_task      = asyncio.create_task(server.serve())
    broadcaster_task = asyncio.create_task(broadcaster_loop())
    pyth_task        = asyncio.create_task(pyth_price_loop())
    theta_task       = asyncio.create_task(theta_decay_loop())

    logging.info(f"Dashboard live at http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")

    try:
        await asyncio.gather(pipeline_task, server_task, broadcaster_task, pyth_task, theta_task)
    except (KeyboardInterrupt, asyncio.CancelledError):
        server.should_exit = True
        pipeline_task.cancel()
        broadcaster_task.cancel()
        pyth_task.cancel()
        theta_task.cancel()
        await asyncio.gather(
            pipeline_task, server_task, broadcaster_task, pyth_task, theta_task,
            return_exceptions=True,
        )


if __name__ == "__main__":
    asyncio.run(_run_all())
