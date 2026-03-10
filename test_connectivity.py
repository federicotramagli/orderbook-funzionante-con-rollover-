#!/usr/bin/env python3
"""
test_connectivity.py - Diagnostica rapida per il bot Polymarket MM
==================================================================
Testa ogni layer del pipeline in ordine:
  1. Gamma API  → trova mercati BTC-UPDOWN-15M attivi
  2. WebSocket  → connessione al CLOB e ricezione raw bytes
  3. Decoder    → parsing Nautilus (se disponibile) o raw JSON
  4. Pipeline   → integrazione IngestionEngine + RolloverManager

Uso:
  python test_connectivity.py
  python test_connectivity.py --ws-only    # solo test WS con token hardcoded
  python test_connectivity.py --duration 60  # ascolta per 60s (default 20s)
"""

import asyncio
import json
import sys
import time
import argparse
import logging
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Gamma API
# ─────────────────────────────────────────────────────────────────────────────

async def test_gamma_api(slug_prefix: str = "btc-updown-15m") -> list[dict]:
    """Fetch active BTC-15m markets and return token info."""
    import aiohttp

    GAMMA_URL = "https://gamma-api.polymarket.com/markets"

    print("\n" + "═"*60)
    print("STEP 1 — Gamma API connectivity")
    print("═"*60)

    found_tokens: list[dict] = []
    btc_markets: list[dict] = []

    # ── Strategy 1: search with tag_slug / slug filter (most specific) ────────
    # Polymarket Gamma API supports ?slug= for exact/prefix match on some versions
    search_attempts = [
        ("slug prefix",      {"slug": slug_prefix, "active": "true", "closed": "false", "limit": "50"}),
        ("question keyword", {"active": "true", "closed": "false", "limit": "500",
                              "order": "volume24hr", "ascending": "false"}),
    ]

    async with aiohttp.ClientSession() as sess:
        for strategy_name, params in search_attempts:
            print(f"  Searching via {strategy_name}: {params}")
            try:
                async with sess.get(
                    GAMMA_URL, params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as r:
                    status = r.status
                    raw = await r.json(content_type=None)
                    print(f"  HTTP {status}")
            except Exception as e:
                print(f"  [FAIL] Gamma API error: {e}")
                continue

            markets = raw if isinstance(raw, list) else raw.get("data", [])
            print(f"  Markets returned: {len(markets)}")

            for m in markets:
                slug     = m.get("slug", "").lower()
                question = m.get("question", "").lower()
                if slug.startswith(slug_prefix.lower()) or (
                    "btc" in question and "15" in question
                ):
                    btc_markets.append(m)

            if btc_markets:
                print(f"  BTC-15m markets matched: {len(btc_markets)}")
                break
            else:
                print(f"  No match — trying next strategy…\n")

    if not btc_markets:
        # ── Strategy 3: direct slug lookup for known active markets ──────────
        print("\n  All generic searches empty. Trying direct slug lookups…")
        known_slugs = [
            "btc-updown-15m-1773106200",
            "btc-updown-15m-1773105300",
        ]
        async with aiohttp.ClientSession() as sess:
            for slug in known_slugs:
                try:
                    async with sess.get(
                        GAMMA_URL, params={"slug": slug},
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as r:
                        raw2 = await r.json(content_type=None)
                    ms = raw2 if isinstance(raw2, list) else raw2.get("data", [])
                    if ms:
                        btc_markets.extend(ms)
                        print(f"  Found via direct slug '{slug}': {len(ms)} market(s)")
                        break
                except Exception as e:
                    print(f"  Direct slug '{slug}' failed: {e}")

    if not btc_markets:
        print("  [WARN] No BTC-15m markets found via Gamma API.")
        print("         The round may be between intervals. Returning hardcoded fallback.")
        return [{
            "token_id":     "69111612932906493798572456624932775016707311628693666746500824335434097111444",
            "condition_id": "0xee7f037e41b0930eb57716787f3d55aab27335a4dc21672ec46392ee11542fff",
            "outcome":      "Up (hardcoded fallback)",
            "slug":         "btc-updown-15m-1773106200",
            "tte_str":      "?",
        }, {
            "token_id":     "57215281997091589825809125960899268231098984838888436885931892744917956921407",
            "condition_id": "0xee7f037e41b0930eb57716787f3d55aab27335a4dc21672ec46392ee11542fff",
            "outcome":      "Down (hardcoded fallback)",
            "slug":         "btc-updown-15m-1773106200",
            "tte_str":      "?",
        }]

    for m in btc_markets:
        slug     = m.get("slug", "")
        question = m.get("question", "")
        cond_id  = m.get("conditionId", "")
        end_date = m.get("endDateIso", "—")
        clob_ids = m.get("clobTokenIds", "[]")
        outcomes = m.get("outcomes", "[]")

        if isinstance(clob_ids, str):
            clob_ids = json.loads(clob_ids)
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)

        # TTL
        try:
            dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            tte_s = (dt.timestamp() - time.time())
            tte_str = f"{int(tte_s//60):02d}:{int(tte_s%60):02d}" if tte_s > 0 else "EXPIRED"
        except Exception:
            tte_str = "?"

        print(f"\n  ✓ Market found:")
        print(f"    slug:     {slug}")
        print(f"    question: {question}")
        print(f"    cond_id:  {cond_id}")
        print(f"    end_date: {end_date}  (TTL: {tte_str})")
        for i, (tok, out) in enumerate(zip(clob_ids, outcomes)):
            print(f"    token[{i}]: {tok}  → outcome={out}")
            found_tokens.append({
                "token_id":     tok,
                "condition_id": cond_id,
                "outcome":      out,
                "slug":         slug,
                "tte_str":      tte_str,
            })

    return found_tokens


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — WebSocket raw connection
# ─────────────────────────────────────────────────────────────────────────────

async def test_websocket(token_ids: list[str], duration_secs: float = 20.0) -> int:
    """Connect to Polymarket CLOB WS and print incoming messages for N seconds."""
    import websockets

    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    print("\n" + "═"*60)
    print("STEP 2 — WebSocket raw connection")
    print("═"*60)
    print(f"  URL:      {WS_URL}")
    print(f"  Tokens:   {len(token_ids)}")
    for t in token_ids:
        print(f"    {t}")
    print(f"  Listening: {duration_secs}s\n")

    if not token_ids:
        print("  [SKIP] No tokens to subscribe to")
        return 0

    msg_count = 0
    t_start   = time.time()

    try:
        async with websockets.connect(
            WS_URL,
            ping_interval=10,
            ping_timeout=30,
            close_timeout=5,
        ) as ws:
            print(f"  [OK] Connected to {WS_URL}")

            sub_msg = json.dumps({"type": "market", "assets_ids": token_ids})
            await ws.send(sub_msg)
            print(f"  [OK] Subscription sent for {len(token_ids)} token(s)")

            deadline = t_start + duration_secs
            while time.time() < deadline:
                remaining = deadline - time.time()
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 5.0))
                except asyncio.TimeoutError:
                    continue

                msg_count += 1
                payload = raw if isinstance(raw, str) else raw.decode()

                try:
                    parsed = json.loads(payload)
                except Exception:
                    parsed = {"raw": payload[:200]}

                elapsed = time.time() - t_start

                # Pretty-print each message type
                if isinstance(parsed, list):
                    for item in parsed:
                        _print_msg(item, elapsed, msg_count)
                else:
                    _print_msg(parsed, elapsed, msg_count)

    except Exception as e:
        print(f"\n  [FAIL] WebSocket error: {e}")

    print(f"\n  ── Received {msg_count} messages in {time.time()-t_start:.1f}s ──")
    return msg_count


def _print_msg(msg: dict, elapsed: float, n: int) -> None:
    ev   = msg.get("event_type", "?")
    aid  = msg.get("asset_id", msg.get("market", "?"))[:16]
    ts   = msg.get("timestamp", "")

    if ev == "book":
        bids = msg.get("bids", [])
        asks = msg.get("asks", [])
        best_bid = f"{float(bids[-1]['price']):.4f}" if bids else "—"
        best_ask = f"{float(asks[-1]['price']):.4f}" if asks else "—"
        print(
            f"  [{elapsed:6.2f}s] #{n:04d}  "
            f"SNAPSHOT  token={aid}..  "
            f"bid={best_bid}  ask={best_ask}  "
            f"({len(bids)} bids, {len(asks)} asks)"
        )

    elif ev == "price_change":
        changes = msg.get("price_changes", [])
        for c in changes:
            print(
                f"  [{elapsed:6.2f}s] #{n:04d}  "
                f"QUOTE_UPD token={c.get('asset_id','?')[:16]}..  "
                f"side={c.get('side','?')}  "
                f"px={c.get('price','?')}  sz={c.get('size','?')}  "
                f"best_bid={c.get('best_bid','?')}  best_ask={c.get('best_ask','?')}"
            )

    elif ev == "last_trade_price":
        print(
            f"  [{elapsed:6.2f}s] #{n:04d}  "
            f"TRADE     token={aid}..  "
            f"px={msg.get('price','?')}  sz={msg.get('size','?')}  "
            f"side={msg.get('side','?')}"
        )

    else:
        print(
            f"  [{elapsed:6.2f}s] #{n:04d}  "
            f"OTHER[{ev}] {str(msg)[:120]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Decoder check
# ─────────────────────────────────────────────────────────────────────────────

def test_decoder() -> None:
    print("\n" + "═"*60)
    print("STEP 3 — Nautilus decoder check")
    print("═"*60)

    sys.path.insert(0, "./nautilus_trader-develop")

    try:
        import msgspec
        print("  [OK] msgspec available")
    except ImportError:
        print("  [FAIL] msgspec missing — run: pip install msgspec")
        return

    try:
        from nautilus_trader.adapters.polymarket.schemas.book import (
            PolymarketBookSnapshot, PolymarketQuotes, PolymarketTrade,
        )
        from nautilus_trader.adapters.polymarket.websocket.types import MARKET_WS_MESSAGE
        decoder = msgspec.json.Decoder(MARKET_WS_MESSAGE)
        print("  [OK] Nautilus schemas loaded from local repo")
        print("  [OK] msgspec tagged-union decoder created for MARKET_WS_MESSAGE")

        # Smoke test with a fake snapshot
        fake = json.dumps([{
            "event_type": "book",
            "market":     "0xTESTCOND",
            "asset_id":   "0xTESTTOKEN",
            "bids":       [{"price": "0.55", "size": "100"}],
            "asks":       [{"price": "0.57", "size": "80"}],
            "timestamp":  str(int(time.time() * 1000)),
        }]).encode()

        result = decoder.decode(fake)
        snap = result[0]
        print(f"  [OK] Smoke-test decode: bid={snap.bids[-1].price}  ask={snap.asks[-1].price}")

    except ImportError as e:
        print(f"  [WARN] Nautilus schemas not available: {e}")
        print("         Raw JSON fallback will be used (fully functional)")
    except Exception as e:
        print(f"  [FAIL] Decoder error: {e}")

    try:
        from nautilus_trader.adapters.polymarket.websocket.client import (
            PolymarketWebSocketClient,
        )
        print("  [OK] Nautilus pyo3 WebSocketClient available (fastest path)")
    except ImportError:
        print("  [INFO] Nautilus pyo3 WS client not compiled — websockets fallback active")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Quick pipeline smoke test
# ─────────────────────────────────────────────────────────────────────────────

async def test_pipeline(token_ids: list[str], duration_secs: float = 15.0) -> None:
    print("\n" + "═"*60)
    print("STEP 4 — IngestionEngine + RolloverManager pipeline test")
    print("═"*60)

    if not token_ids:
        print("  [SKIP] No tokens — skipping pipeline test")
        return

    received: list = []

    try:
        from ingestion_engine import IngestionEngine, TradingTick
        from rollover_manager import (
            PolymarketRolloverManager, MarketSeriesConfig,
            YesNoMarketTick, ActiveMarket, TokenInfo,
        )
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return

    async def on_tick(tick: TradingTick) -> None:
        received.append(tick)
        if len(received) <= 5:
            log.info(
                "[Tick #%d] token=%s..  bid=%.4f  ask=%.4f  src=%s",
                len(received), tick.token_id[:16],
                tick.bid, tick.ask, tick.source,
            )

    engine = IngestionEngine(
        on_tick=on_tick,
        token_ids=token_ids,
        discovery_enabled=False,
    )

    print(f"  Subscribing to {len(token_ids)} token(s), listening {duration_secs}s…\n")

    try:
        async with asyncio.timeout(duration_secs + 5):
            engine_task = asyncio.create_task(engine.start())
            await asyncio.sleep(duration_secs)
            engine_task.cancel()
            try:
                await engine_task
            except (asyncio.CancelledError, Exception):
                pass
    except Exception as e:
        print(f"  Engine error: {e}")
    finally:
        await engine.stop()

    print(f"\n  ── Pipeline received {len(received)} TradingTick(s) in {duration_secs}s ──")
    if received:
        last = received[-1]
        print(f"  Last tick: bid={last.bid:.4f}  ask={last.ask:.4f}  "
              f"src={last.source}  token={last.token_id[:20]}..")
        print("  [PASS] Pipeline is working correctly ✓")
    else:
        print("  [WARN] No ticks received — check WS connectivity above")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def main(ws_only: bool, duration: float) -> None:
    print("\n" + "█"*60)
    print("  POLYMARKET MM BOT — CONNECTIVITY DIAGNOSTICS")
    print("█"*60)
    print(f"  Date/time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python:    {sys.version.split()[0]}")

    # Step 3 is sync, run first
    test_decoder()

    if ws_only:
        # Use a fallback token ID for quick WS testing
        FALLBACK_TOKENS = [
            # paste a known live token ID here for quick testing
        ]
        if not FALLBACK_TOKENS:
            print("\n[ws-only mode] Paste a token ID in FALLBACK_TOKENS in this file.")
            return
        await test_websocket(FALLBACK_TOKENS, duration)
        return

    # Full test flow
    tokens = await test_gamma_api()

    token_ids = [t["token_id"] for t in tokens]
    await test_websocket(token_ids, duration)
    await test_pipeline(token_ids[:2], min(duration, 15.0))

    print("\n" + "█"*60)
    print("  DIAGNOSTICS COMPLETE")
    print("█"*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws-only",  action="store_true",
                        help="Skip Gamma API, connect WS directly with hardcoded tokens")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Seconds to listen on the WebSocket (default: 20)")
    args = parser.parse_args()
    asyncio.run(main(ws_only=args.ws_only, duration=args.duration))
