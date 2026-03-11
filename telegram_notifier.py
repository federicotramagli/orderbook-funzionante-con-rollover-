"""
telegram_notifier.py — Telegram push notifications for the Polymarket bot.
==========================================================================
Fire-and-forget: each send runs in a daemon thread so it never blocks the
trading loop.  All network errors are caught and logged — the bot continues
regardless of Telegram availability.

Usage (from paper_trading_engine or dashboard):
    notifier = TelegramNotifier(TOKEN, CHAT_ID, asset_label="BTC 15m")
    notifier.notify_trade_open(pos_id=1, asset="YES", entry=0.74,
                               shares=67.57, cost=50.0, iv_pct=69.3,
                               expiry_secs=750)
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

log = logging.getLogger("TelegramNotifier")

_TG_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:

    def __init__(
        self,
        token:       str,
        chat_id:     "str | list[str]",
        asset_label: str = "BTC 15m",
    ) -> None:
        self._token      = token
        self._chat_ids   = [chat_id] if isinstance(chat_id, str) else list(chat_id)
        self.asset_label = asset_label

    # ── Public notification methods ───────────────────────────────────────────

    def notify_trade_open(
        self,
        *,
        pos_id:      int,
        asset:       str,           # "YES" or "NO"
        entry:       float,
        shares:      float,
        cost:        float,
        iv_pct:      Optional[float],   # already in % (e.g. 69.3)
        expiry_secs: float,
    ) -> None:
        iv_str = f"{iv_pct:.1f}%" if iv_pct is not None else "n/a"
        exp_str = _fmt_duration(expiry_secs)
        text = (
            "🔵 <b>NUOVO TRADE ESEGUITO</b>\n"
            "\n"
            f"📌 Asset: {self.asset_label}\n"
            f"↕️ Posizione: <b>{asset}</b> (#{pos_id})\n"
            f"💰 Prezzo di Entrata: <code>${entry:.4f}</code>\n"
            f"📦 Quantità: <code>{shares:.4f} shares</code>\n"
            f"💵 Capitale Investito: <code>${cost:.2f}</code>\n"
            f"📈 IV al momento: <code>{iv_str}</code>\n"
            f"⏰ Scadenza: {exp_str}"
        )
        self._send(text)

    def notify_trade_close(
        self,
        *,
        trade_id:    int,
        asset:       str,
        pnl:         float,
        pnl_pct:     float,
        exit_price:  float,
        session_roi: float,
    ) -> None:
        won     = pnl > 0
        icon    = "✅" if won else "❌"
        esito   = "VINTO" if won else "PERSO"
        sign    = "+" if pnl >= 0 else ""
        text = (
            "🏁 <b>TRADE CHIUSO</b>\n"
            "\n"
            f"📌 Asset: {self.asset_label}\n"
            f"{icon} Esito: <b>{esito}</b> (#{trade_id})\n"
            f"💸 Profit/Loss Netto: <code>{sign}${pnl:.2f} ({sign}{pnl_pct:.1f}%)</code>\n"
            f"🏷️ Prezzo Finale Asset: <code>${exit_price:.4f}</code>\n"
            f"📊 ROI cumulativo sessione: <code>{session_roi:+.2f}%</code>"
        )
        self._send(text)

    def notify_stats(
        self,
        *,
        win_rate:       float,
        total_trades:   int,
        total_profit:   float,
        max_dd_pct:     float,
        last_iv:        Optional[float],   # raw 0-1 form
        kernel_ok:      bool  = True,
        # live portfolio fields
        balance:        float = 0.0,
        equity:         float = 0.0,
        unrealized_pnl: float = 0.0,
        yes_shares:     float = 0.0,
        no_shares:      float = 0.0,
        avg_price_yes:  Optional[float] = None,
        avg_price_no:   Optional[float] = None,
    ) -> None:
        iv_str     = f"{last_iv * 100:.1f}%" if last_iv is not None else "n/a"
        status_str = "OK" if kernel_ok else "Error"
        pnl_sign   = "+" if total_profit >= 0 else ""
        upnl_sign  = "+" if unrealized_pnl >= 0 else ""

        net_shares = yes_shares - no_shares
        net_sign   = "+" if net_shares >= 0 else ""

        avg_yes_str = f"${avg_price_yes:.4f}" if avg_price_yes is not None else "—"
        avg_no_str  = f"${avg_price_no:.4f}"  if avg_price_no  is not None else "—"

        text = (
            "📊 <b>AGGIORNAMENTO STATISTICHE BOT</b>\n"
            "\n"
            f"💵 Balance: <code>${balance:,.2f}</code>  |  Equity: <code>${equity:,.2f}</code>\n"
            f"📈 uPnL: <code>{upnl_sign}${unrealized_pnl:.2f}</code>\n"
            "\n"
            f"📦 Esposizione YES: <code>{yes_shares:.3f} sh</code>  @  {avg_yes_str}\n"
            f"📦 Esposizione NO:  <code>{no_shares:.3f} sh</code>  @  {avg_no_str}\n"
            f"⚖️ Net (YES−NO): <code>{net_sign}{net_shares:.3f} sh</code>\n"
            "\n"
            f"🎯 Win Rate: <code>{win_rate:.1f}%</code>  |  Trade: <code>{total_trades}</code>\n"
            f"💰 rPnL Totale: <code>{pnl_sign}${total_profit:.2f}</code>  |  MaxDD: <code>{max_dd_pct:.2f}%</code>\n"
            f"🔧 Kernel: {status_str}  (IV: {iv_str})"
        )
        self._send(text)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _send(self, text: str) -> None:
        """Fire-and-forget: send in a daemon thread."""
        t = threading.Thread(target=self._post, args=(text,), daemon=True)
        t.start()

    def _post(self, text: str) -> None:
        url = _TG_API.format(token=self._token)
        for cid in self._chat_ids:
            payload = json.dumps({
                "chat_id":    cid,
                "text":       text,
                "parse_mode": "HTML",
            }).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status != 200:
                        log.warning("[TG] HTTP %d for chat_id=%s", resp.status, cid)
            except urllib.error.URLError as exc:
                log.warning("[TG] Network error for chat_id=%s: %s", cid, exc)
            except Exception as exc:
                log.warning("[TG] Unexpected error for chat_id=%s: %s", cid, exc)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_duration(secs: float) -> str:
    secs = max(0.0, secs)
    m = int(secs // 60)
    s = int(secs % 60)
    return f"{m}m {s:02d}s"
