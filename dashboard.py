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
import sys
import time
from pathlib import Path
from typing import Any

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
        engine = _manager_ref._engine
        yes_st = engine._market_states.get(ynt.yes_token_id)
        no_st  = engine._market_states.get(ynt.no_token_id)
        d["yes_bids"] = yes_st.top_bids(8) if yes_st else []
        d["yes_asks"] = yes_st.top_asks(8) if yes_st else []
        d["no_bids"]  = no_st.top_bids(8)  if no_st  else []
        d["no_asks"]  = no_st.top_asks(8)  if no_st  else []

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


async def publish_rollover(old: ActiveMarket, new: ActiveMarket) -> None:
    """
    Called on rollover events — broadcasts a rollover notification to the UI.
    """
    global _latest_rollover_event
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

    logging.info(f"Dashboard live at http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")

    try:
        await asyncio.gather(pipeline_task, server_task, broadcaster_task)
    except (KeyboardInterrupt, asyncio.CancelledError):
        server.should_exit = True
        pipeline_task.cancel()
        broadcaster_task.cancel()
        await asyncio.gather(pipeline_task, server_task, broadcaster_task, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(_run_all())
