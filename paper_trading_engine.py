"""
paper_trading_engine.py — Aggressive Taker / Multi-Position Model + KDE Filter
================================================================================
10,000 USDC paper trading simulation with $50 fixed-USD sizing per trade.

Polymarket constraints:
  - No margin shorting: to bet against YES, buy NO at (1.00 - mkt_bid_yes).
  - Tick size: 0.01. Valid limit prices: [0.01, 0.99].
  - YES price + NO price = 1.00 (no-arbitrage complement).

Sizing: shares = TRADE_SIZE_USD / entry_price  (dynamic per-trade)

Entry — Aggressive Taker:
  Crosses the spread when edge >= effective_min_edge, after KDE filters pass.
  LONG YES : bs_fair_yes - mkt_ask_yes >= edge  → buy YES at mkt_ask_yes
  LONG NO  : mkt_bid_yes - bs_fair_yes >= edge  → buy NO  at (1 - mkt_bid_yes)

Exit — Multi-position model (no pegged maker, no stop loss):
  Each open position is closed independently when:
    1. mkt_bid >= 0.99  → sell-limit hit
    2. mkt_bid <= 0.01  → contract worthless
    3. Kill switch (T <= 90s) → close all positions at market bid

Risk management:
  Kill switch : T <= 90s → block entries, force-liquidate all open positions at market bid.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

try:
    from telegram_notifier import TelegramNotifier as _TelegramNotifier
    _TG_AVAILABLE = True
except ImportError:
    _TelegramNotifier = None  # type: ignore[assignment,misc]
    _TG_AVAILABLE = False

log = logging.getLogger("PaperTradingEngine")


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL RUST MODULE IMPORT
# ─────────────────────────────────────────────────────────────────────────────

try:
    import kde_polymarket as _kde
    _KDE_AVAILABLE = True
    log.info("[PTE] kde_polymarket loaded — microstructure filters ACTIVE.")
except ImportError:
    _kde = None  # type: ignore[assignment]
    _KDE_AVAILABLE = False
    log.warning("[PTE] kde_polymarket not found — running BS-only mode.")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

AVAILABLE_BALANCE:      float = 10_000.00
TRADE_SIZE_USD:         float = 100.0       # fixed dollar notional per trade
MIN_EDGE:               float = 0.08   # lowered: KDE Hurst filter handles trend risk
MIN_EDGE_DIVERGENCE:    float = 0.05   # divergence alpha boost (div > 0.30)
KILL_SWITCH_SECONDS:    float = 90.0
TICK_SIZE:              float = 0.01

HURST_TREND_THRESHOLD:  float = 0.55
DIVERGENCE_ALPHA_FLOOR: float = 0.30
ENTRY_BUFFER_SECS:      float = 10.0   # minimum seconds between consecutive entries
MAX_SIDE_EXPOSURE_PCT:  float = 0.06   # max 6% of initial capital per side (YES/NO independent)
                                        # e.g. $10,000 → max $300 YES + max $300 NO

TRADE_LOG_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "trades_history.csv"
)
OPEN_POS_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "open_positions.json"
)
_CSV_COLUMNS = [
    "Trade_ID", "Timestamp", "Asset", "Entry_Price", "Exit_Price",
    "Shares", "Gross_PnL", "PnL_Pct", "Duration_Seconds", "Exit_Type",
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _round_tick(price: float) -> float:
    snapped = round(round(price / TICK_SIZE) * TICK_SIZE, 2)
    return max(TICK_SIZE, min(1.0 - TICK_SIZE, snapped))


def _append_csv_row(row: dict) -> None:
    file_exists = os.path.isfile(TRADE_LOG_PATH)
    try:
        with open(TRADE_LOG_PATH, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except OSError as exc:
        log.error("[PTE] Failed to write trade log: %s", exc)


def _load_csv_history() -> list[dict]:
    if not os.path.isfile(TRADE_LOG_PATH):
        return []
    try:
        with open(TRADE_LOG_PATH, newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError as exc:
        log.error("[PTE] Failed to read trade log: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

class InventoryType(str, Enum):
    NONE  = "NONE"
    YES   = "YES"
    NO    = "NO"
    MIXED = "MIXED"


class CloseReason(str, Enum):
    TARGET     = "TARGET"
    KILLSWITCH = "KILL_SWITCH_EXPIRY"


@dataclass
class MicroStats:
    hurst:      float = 0.5
    entropy:    float = 1.0
    divergence: float = 0.0

    @property
    def is_stub(self) -> bool:
        return self.hurst == 0.5 and self.entropy == 1.0 and self.divergence == 0.0


@dataclass
class Trade:
    trade_id:    int
    ts_open:     float
    ts_close:    float
    asset:       InventoryType
    entry_price: float
    exit_price:  float
    shares:      float
    pnl:         float
    pnl_pct:     float        # % of position cost
    duration:    float        # seconds
    close_reason: CloseReason
    balance_after: float
    hurst_at_entry:      float = 0.5
    divergence_at_entry: float = 0.0
    edge_used:           float = MIN_EDGE


@dataclass
class OpenPosition:
    pos_id:      int
    ts_open:     float
    asset:       InventoryType
    entry_price: float
    shares:      float
    cost:        float
    edge_used:   float
    micro:       Optional[MicroStats]


@dataclass
class PerfStats:
    initial_capital:  float
    current_equity:   float
    roi_pct:          float
    total_trades:     int
    win_rate_pct:     float
    avg_win:          float
    avg_loss:         float
    profit_factor:    float
    max_drawdown_pct: float
    sharpe_ratio:     float = 0.0


@dataclass
class EngineStats:
    balance:             float
    equity:              float
    inventory:           InventoryType
    shares_open:         float
    entry_price:         Optional[float]
    exit_target:         Optional[float]
    cost_locked:         float
    current_bid:         float
    unrealized_pnl:      float
    uroi_pct:            float
    realized_pnl:        float
    total_pnl:           float
    trade_count:         int
    win_count:           int
    loss_count:          int
    in_kill_switch:      bool
    last_close:          Optional[str]
    last_ts:             Optional[float]
    kde_available:       bool
    last_micro:          Optional[MicroStats]
    perf:                Optional[PerfStats]
    trade_log:            list[dict]   # last 50 trades for UI table
    open_positions_list:  list[dict]  = field(default_factory=list)  # live positions for UI
    open_positions_count: int = 0
    equity_curve:         list[dict]  = field(default_factory=list)  # cumulative rPnL per trade


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class PaperTradingEngine:

    def __init__(
        self,
        balance:          float = AVAILABLE_BALANCE,
        trade_size_usd:   float = TRADE_SIZE_USD,
        min_edge:         float = MIN_EDGE,
        kill_switch_secs: float = KILL_SWITCH_SECONDS,
        notifier:         object = None,   # Optional[TelegramNotifier]
    ) -> None:
        self._initial_capital: float = balance
        self.trade_size_usd:   float = trade_size_usd
        self.min_edge:         float = min_edge
        self.kill_switch_secs: float = kill_switch_secs
        self._notifier                = notifier

        # Multi-position state
        self._open_positions: list[OpenPosition] = []
        self._pos_counter:    int                = 0

        # Accounting
        self._realized_pnl:   float        = 0.0
        self._trades:         list[Trade]  = []
        self._trade_counter:  int          = 0
        self._last_close_str: Optional[str] = None
        self._last_ts:        Optional[float] = None
        self._peak_equity:    float        = balance

        # KDE state
        self._last_micro:    Optional[MicroStats] = None
        self._kill_logged:   bool  = False
        self._warmup_until:  float = 0.0    # epoch secs; entries blocked until this time
        self._restore_grace_until: float = 0.0  # epoch secs; bid≤0.01 closes suppressed until this time

        # Entry buffer: minimum 10s between consecutive entries
        self._last_entry_ts: float = 0.0

        # Periodic active-positions log (every 30s when positions are open)
        self._last_pos_log_ts: float = 0.0

        # Last known IV and expiry (updated on every update_tick call)
        self._last_iv:          Optional[float] = None
        self._last_expiry_secs: float           = 0.0

        # Periodic Telegram stats notification (every 30 minutes)
        self._last_stats_tg_ts: float = 0.0

        # Persistence
        self.balance = balance
        self._restore_from_csv()
        self._restore_open_positions()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _restore_from_csv(self) -> None:
        rows = _load_csv_history()
        if not rows:
            return
        for row in rows:
            try:
                pnl      = float(row["Gross_PnL"])
                pnl_pct  = float(row.get("PnL_Pct", 0.0))
                shares   = float(row.get("Shares", 0.0))
                duration = float(row.get("Duration_Seconds", 0.0))
                tid      = int(row.get("Trade_ID", self._trade_counter + 1))
                self._trade_counter = max(self._trade_counter, tid)
                self._realized_pnl += pnl
                self._trades.append(Trade(
                    trade_id      = tid,
                    ts_open       = 0.0,
                    ts_close      = 0.0,
                    asset         = InventoryType(row["Asset"]),
                    entry_price   = float(row["Entry_Price"]),
                    exit_price    = float(row["Exit_Price"]),
                    shares        = shares,
                    pnl           = pnl,
                    pnl_pct       = pnl_pct,
                    duration      = duration,
                    close_reason  = CloseReason(row["Exit_Type"]),
                    balance_after = 0.0,
                ))
            except (KeyError, ValueError):
                continue

        self.balance      += self._realized_pnl
        self._peak_equity  = self.balance
        log.info(
            "[PTE] Restored %d trades from CSV — rPnL=%+.2f  bal=$%.2f",
            len(self._trades), self._realized_pnl, self.balance,
        )

    # ── Open positions persistence ────────────────────────────────────────────

    def _save_open_positions(self) -> None:
        """Write current open positions to JSON so they survive restarts."""
        data = {
            "pos_counter": self._pos_counter,
            "positions": [
                {
                    "pos_id":      p.pos_id,
                    "ts_open":     p.ts_open,
                    "asset":       p.asset.value,
                    "entry_price": p.entry_price,
                    "shares":      p.shares,
                    "cost":        p.cost,
                    "edge_used":   p.edge_used,
                }
                for p in self._open_positions
            ],
        }
        try:
            with open(OPEN_POS_PATH, "w") as fh:
                json.dump(data, fh)
        except OSError as exc:
            log.error("[PTE] Failed to save open positions: %s", exc)

    def _restore_open_positions(self) -> None:
        """
        Restore open positions from JSON on startup.

        Balance adjustment: when positions were opened, their cost was already
        deducted from self.balance. _restore_from_csv() resets balance to
        initial + realized_pnl (without knowing about locked capital), so we
        re-deduct each position's cost here to keep accounting consistent.
        """
        if not os.path.isfile(OPEN_POS_PATH):
            return
        try:
            with open(OPEN_POS_PATH) as fh:
                data = json.load(fh)
            positions = data.get("positions", [])
            if not positions:
                return
            restored   = 0
            total_cost = 0.0
            for p in positions:
                pos = OpenPosition(
                    pos_id      = int(p["pos_id"]),
                    ts_open     = float(p["ts_open"]),
                    asset       = InventoryType(p["asset"]),
                    entry_price = float(p["entry_price"]),
                    shares      = float(p["shares"]),
                    cost        = float(p["cost"]),
                    edge_used   = float(p.get("edge_used", MIN_EDGE)),
                    micro       = None,
                )
                self._open_positions.append(pos)
                total_cost += pos.cost
                restored   += 1
            self._pos_counter = max(self._pos_counter, int(data.get("pos_counter", 0)))
            # Re-deduct locked capital that _restore_from_csv doesn't know about
            self.balance -= total_cost
            self._peak_equity = max(self._peak_equity, self.balance)
            self._restore_grace_until = time.time() + 10.0
            log.warning(
                "[PTE] Restored %d open position(s) from JSON — "
                "locked=$%.2f  bal=$%.2f  grace=10s",
                restored, total_cost, self.balance,
            )
        except Exception as exc:
            log.error("[PTE] Failed to restore open positions: %s", exc)

    # ── Public interface ──────────────────────────────────────────────────────

    def reset_market(self) -> None:
        self._open_positions = []
        self._kill_logged    = False
        # Clear the JSON so stale positions don't survive the next session
        try:
            with open(OPEN_POS_PATH, "w") as fh:
                json.dump({"pos_counter": self._pos_counter, "positions": []}, fh)
        except OSError:
            pass
        log.info("[PTE] Market reset — all positions cleared.")

    def reset_all_stats(self) -> None:
        """Full accounting reset: clear all trades, restore initial balance, wipe CSV."""
        self.reset_market()
        self.balance           = self._initial_capital
        self._realized_pnl     = 0.0
        self._trades           = []
        self._trade_counter    = 0
        self._pos_counter      = 0
        self._open_positions   = []
        self._last_close_str   = None
        self._last_ts          = None
        self._peak_equity      = self._initial_capital
        self._last_entry_ts    = 0.0
        # Wipe the CSV and the open positions JSON
        try:
            with open(TRADE_LOG_PATH, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
                writer.writeheader()
            with open(OPEN_POS_PATH, "w") as fh:
                json.dump({"pos_counter": 0, "positions": []}, fh)
            log.warning("[PTE] Full stats reset — balance restored to $%.2f, CSV + JSON cleared.",
                        self._initial_capital)
        except Exception as exc:
            log.error("[PTE] reset_all_stats: failed to clear files: %s", exc)

    def trigger_kde_rollover(self, warmup_secs: float = 60.0) -> None:
        """
        Called on market rollover.
        Resets the KDE internal buffer (if Rust module is loaded) and starts
        a warm-up window during which new entries are blocked.
        """
        if _KDE_AVAILABLE and _kde is not None:
            try:
                _kde.reset_buffer()
                log.info("[SYSTEM] KDE buffer reset.")
            except Exception as exc:
                log.warning("[SYSTEM] KDE reset_buffer() failed: %s", exc)
        self._warmup_until = time.time() + warmup_secs
        log.warning(
            "[SYSTEM] Rollover detected. KDE Reset. Waiting for warm-up (%.0fs)…",
            warmup_secs,
        )

    def update_tick(
        self,
        bs_fair_yes:    float,
        mkt_bid_yes:    float,
        mkt_ask_yes:    float,
        time_to_expiry: float,
        micro_stats:    Optional[MicroStats] = None,
        vega:           Optional[float]      = None,
        iv:             Optional[float]      = None,   # raw 0-1 form from analytics
    ) -> None:
        if micro_stats is not None:
            self._last_micro = micro_stats
        if iv is not None:
            self._last_iv = iv
        self._last_expiry_secs = time_to_expiry

        # Periodic Telegram stats: every 2 minutes
        _TG_STATS_INTERVAL = 120.0
        if (self._notifier is not None
                and time.time() - self._last_stats_tg_ts >= _TG_STATS_INTERVAL):
            self._last_stats_tg_ts = time.time()
            self._send_tg_stats(mkt_bid_yes, _round_tick(1.0 - mkt_ask_yes))

        mkt_bid_no:    float = _round_tick(1.0 - mkt_ask_yes)
        bs_fair_yes_r: float = _round_tick(bs_fair_yes)
        in_kill:       bool  = time_to_expiry <= self.kill_switch_secs

        # ── Exit management — processes all open positions ────────────────────
        # (runs regardless of kill switch — positions stay open until 0.99/0.01)
        self._manage_exits(bs_fair_yes_r, mkt_bid_yes, mkt_bid_no)

        # ── Kill switch: block NEW entries only, don't close existing positions ─
        if in_kill:
            if not self._kill_logged:
                self._kill_logged = True
                log.warning(
                    "[PTE] KILL_SWITCH — T=%.0fs ≤ %.0fs — "
                    "new entries blocked. Existing %d position(s) held until 0.99/0.01.",
                    time_to_expiry, self.kill_switch_secs, len(self._open_positions),
                )
            return

        # ── Periodic active-positions log (every 30s) ─────────────────────────
        if self._open_positions and time.time() - self._last_pos_log_ts >= 30.0:
            self._last_pos_log_ts = time.time()
            self._log_active_positions(mkt_bid_yes, mkt_bid_no)

        # ── KDE warm-up guard ─────────────────────────────────────────────────
        # After a rollover the KDE buffer is empty; block entries until the
        # buffer has accumulated enough microstructure data (60s by default).
        if time.time() < self._warmup_until:
            remaining = self._warmup_until - time.time()
            log.debug("[PTE] KDE warm-up in progress — %.0fs remaining.", remaining)
            return

        # ── Vega / Gamma risk guard ───────────────────────────────────────────
        # High binary vega near expiry means the position P&L is hyper-sensitive
        # to a single BTC tick.  Block new entries when:
        #   |vega| > 1.5  AND  time_to_expiry < 180s
        _VEGA_BLOCK_THRESHOLD = 1.5
        _VEGA_BLOCK_WINDOW    = 180.0
        if (vega is not None
                and abs(vega) > _VEGA_BLOCK_THRESHOLD
                and time_to_expiry < _VEGA_BLOCK_WINDOW):
            log.debug(
                "[PTE] VEGA BLOCK — |vega|=%.4f > %.1f with T=%.0fs < %.0fs",
                abs(vega), _VEGA_BLOCK_THRESHOLD,
                time_to_expiry, _VEGA_BLOCK_WINDOW,
            )
            return

        # ── Entry signals ─────────────────────────────────────────────────────
        effective_edge = self._compute_effective_edge(micro_stats)
        long_yes_edge  = bs_fair_yes_r - mkt_ask_yes
        long_no_edge   = mkt_bid_yes - bs_fair_yes_r

        # Determine dominant side for logging
        if long_yes_edge >= long_no_edge and long_yes_edge > 0:
            self._log_strategy_signal("YES", long_yes_edge, effective_edge,
                                      long_yes_edge >= effective_edge)
        elif long_no_edge > 0:
            self._log_strategy_signal("NO", long_no_edge, effective_edge,
                                      long_no_edge >= effective_edge)

        # Ask price floor: skip trades where the purchase price is below 0.15
        # (near-zero probability contracts — illiquid, high slippage, binary risk)
        ask_no = _round_tick(1.0 - mkt_bid_yes)
        if long_yes_edge >= effective_edge:
            if mkt_ask_yes < 0.15:
                log.debug("[STRATEGY] YES ask=%.2f < 0.15 — skipping.", mkt_ask_yes)
            elif self._entry_buffer_ok() and \
               self._kde_entry_allowed("YES", mkt_bid_yes, mkt_ask_yes, micro_stats):
                self._open_long_yes(mkt_ask_yes, mkt_bid_yes, bs_fair_yes_r, effective_edge, micro_stats)
        elif long_no_edge >= effective_edge:
            if ask_no < 0.15:
                log.debug("[STRATEGY] NO ask=%.2f < 0.15 — skipping.", ask_no)
            elif self._entry_buffer_ok() and \
               self._kde_entry_allowed("NO", mkt_bid_yes, mkt_ask_yes, micro_stats):
                self._open_long_no(mkt_bid_yes, mkt_ask_yes, bs_fair_yes_r, effective_edge, micro_stats)

    # ── Analytics ─────────────────────────────────────────────────────────────

    def get_performance_stats(self) -> Optional[PerfStats]:
        if not self._trades:
            return None

        pnls   = [t.pnl for t in self._trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_profit  = sum(wins)
        total_loss    = abs(sum(losses)) if losses else 0.0
        win_rate      = len(wins) / len(pnls) * 100
        avg_win       = sum(wins)   / len(wins)   if wins   else 0.0
        avg_loss      = sum(losses) / len(losses) if losses else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        # Peak-to-Valley drawdown on equity curve
        eq = self._initial_capital
        peak = eq
        max_dd = 0.0
        for t in self._trades:
            eq    += t.pnl
            peak   = max(peak, eq)
            dd     = (peak - eq) / peak * 100 if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        roi = (self.balance / self._initial_capital - 1.0) * 100

        # Sharpe Ratio (per-trade, risk-free = 0):
        #   Sharpe = mean(pnl_pct) / std(pnl_pct)
        # Uses per-trade PnL% so it's scale-invariant.
        pnl_pcts = [t.pnl_pct for t in self._trades]
        if len(pnl_pcts) >= 2:
            mean_r = sum(pnl_pcts) / len(pnl_pcts)
            variance = sum((r - mean_r) ** 2 for r in pnl_pcts) / (len(pnl_pcts) - 1)
            std_r = math.sqrt(variance) if variance > 0 else 0.0
            sharpe = mean_r / std_r if std_r > 0 else 0.0
        else:
            sharpe = 0.0

        return PerfStats(
            initial_capital  = self._initial_capital,
            current_equity   = round(self.balance, 2),
            roi_pct          = round(roi, 2),
            total_trades     = len(pnls),
            win_rate_pct     = round(win_rate, 1),
            avg_win          = round(avg_win, 2),
            avg_loss         = round(avg_loss, 2),
            profit_factor    = round(profit_factor, 3),
            max_drawdown_pct = round(max_dd, 2),
            sharpe_ratio     = round(sharpe, 3),
        )

    def get_equity_curve(self) -> list[dict]:
        """Return cumulative realized PnL after each closed trade (anchor at 0)."""
        cum = 0.0
        result = [{"x": 0, "r": 0.0}]
        for t in self._trades:
            cum += t.pnl
            result.append({"x": t.trade_id, "r": round(cum, 2)})
        return result

    def get_trade_log(self, limit: int = 50) -> list[dict]:
        """Return the last `limit` trades as dicts for the UI table."""
        out = []
        for t in self._trades[-limit:]:
            out.append({
                "id":       t.trade_id,
                "asset":    t.asset.value,
                "entry":    round(t.entry_price, 4),
                "exit":     round(t.exit_price, 4),
                "shares":   round(t.shares, 4),
                "pnl":      round(t.pnl, 2),
                "pnl_pct":  round(t.pnl_pct, 2),
                "duration": round(t.duration, 0),
                "reason":   t.close_reason.value,
            })
        return list(reversed(out))   # newest first

    def stats(
        self,
        mkt_bid_yes: float = 0.5,
        mkt_ask_yes: float = 0.5,
    ) -> EngineStats:
        unrealized  = 0.0
        total_cost  = 0.0
        total_shares = 0.0
        weighted_bid = 0.0
        weighted_entry = 0.0

        try:
            mkt_bid_no = _round_tick(1.0 - mkt_ask_yes)
            for pos in self._open_positions:
                bid = mkt_bid_yes if pos.asset == InventoryType.YES else mkt_bid_no
                unrealized    += pos.shares * (bid - pos.entry_price)
                total_cost    += pos.cost
                total_shares  += pos.shares
                weighted_bid  += pos.shares * bid
                weighted_entry += pos.shares * pos.entry_price
        except Exception as exc:
            log.error("[PTE] stats() unrealized compute error: %s", exc)

        # Weighted averages (per-share weighted)
        avg_entry = (weighted_entry / total_shares) if total_shares > 0 else None
        avg_bid   = (weighted_bid   / total_shares) if total_shares > 0 else 0.0

        uroi_pct = (unrealized / total_cost * 100) if total_cost > 0 else 0.0

        open_val = total_shares * avg_bid if self._open_positions else 0.0
        equity   = self.balance + open_val
        self._peak_equity = max(self._peak_equity, equity)

        # Determine aggregate inventory type
        if not self._open_positions:
            inventory = InventoryType.NONE
        else:
            asset_types = {pos.asset for pos in self._open_positions}
            if asset_types == {InventoryType.YES}:
                inventory = InventoryType.YES
            elif asset_types == {InventoryType.NO}:
                inventory = InventoryType.NO
            else:
                inventory = InventoryType.MIXED

        try:
            wins   = sum(1 for t in self._trades if isinstance(t.pnl, (int, float)) and t.pnl > 0)
            losses = sum(1 for t in self._trades if isinstance(t.pnl, (int, float)) and t.pnl <= 0)
        except Exception:
            wins = losses = 0

        try:
            perf = self.get_performance_stats()
        except Exception as exc:
            log.error("[PTE] get_performance_stats() error: %s", exc)
            perf = None

        try:
            trade_log = self.get_trade_log(50)
        except Exception as exc:
            log.error("[PTE] get_trade_log() error: %s", exc)
            trade_log = []

        try:
            equity_curve = self.get_equity_curve()
        except Exception as exc:
            log.error("[PTE] get_equity_curve() error: %s", exc)
            equity_curve = []

        # Snapshot of all live open positions for the UI
        now = time.time()
        mkt_bid_no_snap = _round_tick(1.0 - mkt_ask_yes)
        open_pos_list: list[dict] = []
        for pos in self._open_positions:
            bid    = mkt_bid_yes if pos.asset == InventoryType.YES else mkt_bid_no_snap
            upnl   = round(pos.shares * (bid - pos.entry_price), 2)
            upnl_p = round(upnl / pos.cost * 100, 1) if pos.cost else 0.0
            held   = round(now - pos.ts_open)
            open_pos_list.append({
                "id":      pos.pos_id,
                "asset":   pos.asset.value,
                "entry":   round(pos.entry_price, 4),
                "bid":     round(bid, 4),
                "shares":  round(pos.shares, 3),
                "cost":    round(pos.cost, 2),
                "upnl":    upnl,
                "upnl_p":  upnl_p,
                "held":    held,
            })

        return EngineStats(
            balance              = round(self.balance, 2),
            equity               = round(equity, 2),
            inventory            = inventory,
            shares_open          = round(total_shares, 4),
            entry_price          = avg_entry,
            exit_target          = None,
            cost_locked          = round(total_cost, 2),
            current_bid          = round(avg_bid, 4),
            unrealized_pnl       = round(unrealized, 2),
            uroi_pct             = round(uroi_pct, 2),
            realized_pnl         = round(self._realized_pnl, 2),
            total_pnl            = round(self._realized_pnl + unrealized, 2),
            trade_count          = len(self._trades),
            win_count            = wins,
            loss_count           = losses,
            in_kill_switch       = False,
            last_close           = self._last_close_str,
            last_ts              = self._last_ts,
            kde_available        = _KDE_AVAILABLE,
            last_micro           = self._last_micro,
            perf                 = perf,
            trade_log            = trade_log,
            open_positions_list  = open_pos_list,
            open_positions_count = len(self._open_positions),
            equity_curve         = equity_curve,
        )

    def print_dashboard(
        self,
        mkt_bid_yes: float = 0.5,
        mkt_ask_yes: float = 0.5,
    ) -> None:
        s   = self.stats(mkt_bid_yes, mkt_ask_yes)
        sep = "-" * 50
        print(sep)
        print("[PORTFOLIO STATUS]")
        print(f"Balance: ${s.balance:,.2f} | Equity: ${s.equity:,.2f}")

        if s.inventory != InventoryType.NONE:
            print(
                f"Open Positions: {s.open_positions_count} ({s.inventory.value}) | "
                f"Total Shares: {s.shares_open:.4f} | "
                f"Avg Entry: {s.entry_price:.4f} | Avg Bid: {s.current_bid:.4f}"
            )
            print(f"Unrealized PnL: {s.unrealized_pnl:+.2f} (uROI: {s.uroi_pct:+.1f}%)")
        else:
            print("Active Position: FLAT")

        print(sep)
        print("[PERFORMANCE SUMMARY]")
        p = s.perf
        if p:
            pf = f"{p.profit_factor:.2f}" if math.isfinite(p.profit_factor) else "∞"
            print(
                f"Total Trades: {p.total_trades} | Win Rate: {p.win_rate_pct:.0f}% | "
                f"Max Drawdown: {p.max_drawdown_pct:.1f}%"
            )
            print(
                f"Avg Win: ${p.avg_win:+.2f} | Avg Loss: ${p.avg_loss:+.2f} | "
                f"Profit Factor: {pf} | ROI: {p.roi_pct:+.2f}%"
            )
        else:
            print("No closed trades yet.")
        print(sep)

    def format_log_line(
        self,
        mkt_bid_yes: float = 0.5,
        mkt_ask_yes: float = 0.5,
    ) -> str:
        s = self.stats(mkt_bid_yes, mkt_ask_yes)
        if s.inventory != InventoryType.NONE:
            inv_str = (
                f"{s.inventory.value} {s.open_positions_count}pos "
                f"{s.shares_open:.3f}sh avg_entry={s.entry_price:.2f}"
            )
        else:
            inv_str = "FLAT"
        return (
            f"[PTE] {inv_str}  uPnL={s.unrealized_pnl:+.2f}  "
            f"rPnL={s.realized_pnl:+.2f}  tot={s.total_pnl:+.2f}  "
            f"bal=${s.balance:,.2f}  trades={s.trade_count} "
            f"(W{s.win_count}/L{s.loss_count})"
        )

    # ── KDE filter layer ──────────────────────────────────────────────────────

    def _compute_effective_edge(self, micro: Optional[MicroStats]) -> float:
        """
        Returns the effective edge threshold after applying the Divergence Alpha boost.
        MIN_EDGE (0.08) is the base; if book pressure (divergence > 0.30) is detected,
        it drops to MIN_EDGE_DIVERGENCE (0.05) to be more aggressive.
        """
        if micro is not None and not micro.is_stub and micro.divergence > DIVERGENCE_ALPHA_FLOOR:
            log.debug(
                "[STRATEGY] Divergence alpha boost — div=%.3f > %.2f → edge %.2f → %.2f",
                micro.divergence, DIVERGENCE_ALPHA_FLOOR,
                self.min_edge, MIN_EDGE_DIVERGENCE,
            )
            return MIN_EDGE_DIVERGENCE
        return self.min_edge

    def _log_active_positions(self, mkt_bid_yes: float, mkt_bid_no: float) -> None:
        """Log a snapshot of all currently open positions."""
        now = time.time()
        lines = [
            f"[ACTIVE POSITIONS] {len(self._open_positions)} open  |  "
            f"Balance: ${self.balance:,.2f}"
        ]
        total_upnl = 0.0
        for pos in self._open_positions:
            bid    = mkt_bid_yes if pos.asset == InventoryType.YES else mkt_bid_no
            upnl   = pos.shares * (bid - pos.entry_price)
            upnl_p = upnl / pos.cost * 100 if pos.cost else 0.0
            held   = now - pos.ts_open
            total_upnl += upnl
            lines.append(
                f"  #{pos.pos_id:>3} {pos.asset.value:<3}  "
                f"entry={pos.entry_price:.4f}  bid={bid:.4f}  "
                f"shares={pos.shares:.3f}  cost=${pos.cost:.2f}  "
                f"uPnL={upnl:+.2f} ({upnl_p:+.1f}%)  "
                f"held={held:.0f}s  "
                f"exit: bid≥0.99 | bid≤0.01"
            )
        lines.append(
            f"  ── Total uPnL: {total_upnl:+.2f}  |  "
            f"Equity: ${self.balance + total_upnl:,.2f}"
        )
        log.info("\n".join(lines))

    def _send_tg_stats(self, mkt_bid_yes: float = 0.5, mkt_bid_no: float = 0.5) -> None:
        """Send a periodic stats snapshot to Telegram with live portfolio data."""
        if self._notifier is None:
            return
        perf = self.get_performance_stats()

        # Compute live portfolio breakdown
        unrealized   = 0.0
        yes_shares   = 0.0
        no_shares    = 0.0
        yes_cost     = 0.0
        no_cost      = 0.0
        for pos in self._open_positions:
            bid = mkt_bid_yes if pos.asset == InventoryType.YES else mkt_bid_no
            unrealized += pos.shares * (bid - pos.entry_price)
            if pos.asset == InventoryType.YES:
                yes_shares += pos.shares
                yes_cost   += pos.cost
            else:
                no_shares  += pos.shares
                no_cost    += pos.cost

        avg_yes = (yes_cost / yes_shares) if yes_shares > 0 else None
        avg_no  = (no_cost  / no_shares)  if no_shares  > 0 else None
        equity  = self.balance + (yes_shares * mkt_bid_yes) + (no_shares * mkt_bid_no)

        self._notifier.notify_stats(
            win_rate       = perf.win_rate_pct     if perf else 0.0,
            total_trades   = perf.total_trades     if perf else 0,
            total_profit   = self._realized_pnl,
            max_dd_pct     = perf.max_drawdown_pct if perf else 0.0,
            last_iv        = self._last_iv,
            kernel_ok      = True,
            balance        = self.balance,
            equity         = round(equity, 2),
            unrealized_pnl = round(unrealized, 2),
            yes_shares     = round(yes_shares, 3),
            no_shares      = round(no_shares, 3),
            avg_price_yes  = avg_yes,
            avg_price_no   = avg_no,
        )

    def _side_exposure(self, asset: InventoryType) -> float:
        """Total USD cost locked in all open positions on the given side."""
        return sum(p.cost for p in self._open_positions if p.asset == asset)

    def _exposure_ok(self, asset: InventoryType, new_cost: float) -> bool:
        """Returns False if adding new_cost would exceed MAX_SIDE_EXPOSURE_PCT of initial capital."""
        limit     = self._initial_capital * MAX_SIDE_EXPOSURE_PCT
        exposure  = self._side_exposure(asset)
        if exposure + new_cost > limit:
            log.debug(
                "[PTE] EXPOSURE LIMIT %s — current=$%.2f + new=$%.2f > cap=$%.2f (%.0f%% of $%.0f)",
                asset.value, exposure, new_cost, limit,
                MAX_SIDE_EXPOSURE_PCT * 100, self._initial_capital,
            )
            return False
        return True

    def _entry_buffer_ok(self) -> bool:
        """Returns True if at least ENTRY_BUFFER_SECS have passed since the last entry."""
        elapsed = time.time() - self._last_entry_ts
        if elapsed < ENTRY_BUFFER_SECS:
            log.debug(
                "[PTE] Entry buffer — %.1fs / %.0fs elapsed.",
                elapsed, ENTRY_BUFFER_SECS,
            )
            return False
        return True

    def _kde_entry_allowed(
        self, side: str, mkt_bid_yes: float, mkt_ask_yes: float, micro: Optional[MicroStats]
    ) -> bool:
        """
        Hurst guard: hard block when H > 0.55 AND trend opposes the trade.
        This veto is unconditional — no edge threshold overrides it.
        """
        if micro is None or micro.is_stub:
            return True
        if micro.hurst <= HURST_TREND_THRESHOLD:
            return True
        trend_up = micro.divergence >= 0.0
        if side == "YES" and not trend_up:
            log.warning(
                "[STRATEGY] Hurst BLOCK %s — H=%.3f > %.2f, trend=DOWN — "
                "entry VETOED regardless of edge.",
                side, micro.hurst, HURST_TREND_THRESHOLD,
            )
            return False
        if side == "NO" and trend_up:
            log.warning(
                "[STRATEGY] Hurst BLOCK %s — H=%.3f > %.2f, trend=UP — "
                "entry VETOED regardless of edge.",
                side, micro.hurst, HURST_TREND_THRESHOLD,
            )
            return False
        return True

    def _log_strategy_signal(
        self,
        side: str,
        detected_edge: float,
        required_edge: float,
        triggered: bool,
    ) -> None:
        if triggered:
            log.info(
                "[STRATEGY] Edge detected: %.4f | Required: %.4f | Side: %s | STATUS: TRIGGERED",
                detected_edge, required_edge, side,
            )
        else:
            log.debug(
                "[STRATEGY] Edge detected: %.4f | Required: %.4f | Side: %s | STATUS: BELOW THRESHOLD",
                detected_edge, required_edge, side,
            )

    # ── Internal: open positions ──────────────────────────────────────────────

    def _open_long_yes(
        self,
        mkt_ask_yes: float,
        mkt_bid_yes: float,
        bs_fair_r:   float,
        edge_used:   float,
        micro:       Optional[MicroStats],
    ) -> None:
        shares = self.trade_size_usd / mkt_ask_yes
        cost   = shares * mkt_ask_yes           # == TRADE_SIZE_USD (sanity)
        if cost > self.balance:
            log.warning("[PTE] Insufficient balance for LONG YES (need %.2f, have %.2f)",
                        cost, self.balance)
            return
        if not self._exposure_ok(InventoryType.YES, cost):
            log.warning(
                "[PTE] YES exposure cap hit — side=$%.2f / limit=$%.2f — skipping.",
                self._side_exposure(InventoryType.YES),
                self._initial_capital * MAX_SIDE_EXPOSURE_PCT,
            )
            return
        self.balance -= cost
        self._last_entry_ts = time.time()
        self._pos_counter += 1
        pos = OpenPosition(
            pos_id      = self._pos_counter,
            ts_open     = self._last_entry_ts,
            asset       = InventoryType.YES,
            entry_price = mkt_ask_yes,
            shares      = shares,
            cost        = cost,
            edge_used   = edge_used,
            micro       = micro,
        )
        self._open_positions.append(pos)
        self._save_open_positions()
        upnl_now = shares * (mkt_bid_yes - mkt_ask_yes)
        log.info(
            "[POSITION OPEN] LONG YES #%d\n"
            "  Entry: $%.4f  |  Current bid: $%.4f  |  Shares: %.4f  |  Cost: $%.2f\n"
            "  Exit: sell-limit $0.9900 | worthless $0.0100\n"
            "  uPnL: $%.2f  |  Balance: $%.2f  |  Open positions: %d",
            pos.pos_id,
            mkt_ask_yes, mkt_bid_yes, shares, cost,
            upnl_now, self.balance, len(self._open_positions),
        )
        if self._notifier is not None:
            iv_pct = self._last_iv * 100 if self._last_iv is not None else None
            self._notifier.notify_trade_open(
                pos_id      = pos.pos_id,
                asset       = "YES",
                entry       = mkt_ask_yes,
                shares      = shares,
                cost        = cost,
                iv_pct      = iv_pct,
                expiry_secs = self._last_expiry_secs,
            )

    def _open_long_no(
        self,
        mkt_bid_yes: float,
        mkt_ask_yes: float,
        bs_fair_r:   float,
        edge_used:   float,
        micro:       Optional[MicroStats],
    ) -> None:
        entry_price  = _round_tick(1.0 - mkt_bid_yes)
        current_bid  = _round_tick(1.0 - mkt_ask_yes)   # best bid for NO right now
        shares       = self.trade_size_usd / entry_price
        cost         = shares * entry_price
        if cost > self.balance:
            log.warning("[PTE] Insufficient balance for LONG NO (need %.2f, have %.2f)",
                        cost, self.balance)
            return
        if not self._exposure_ok(InventoryType.NO, cost):
            log.warning(
                "[PTE] NO exposure cap hit — side=$%.2f / limit=$%.2f — skipping.",
                self._side_exposure(InventoryType.NO),
                self._initial_capital * MAX_SIDE_EXPOSURE_PCT,
            )
            return
        self.balance -= cost
        self._last_entry_ts = time.time()
        self._pos_counter += 1
        pos = OpenPosition(
            pos_id      = self._pos_counter,
            ts_open     = self._last_entry_ts,
            asset       = InventoryType.NO,
            entry_price = entry_price,
            shares      = shares,
            cost        = cost,
            edge_used   = edge_used,
            micro       = micro,
        )
        self._open_positions.append(pos)
        self._save_open_positions()
        upnl_now = shares * (current_bid - entry_price)
        log.info(
            "[POSITION OPEN] LONG NO #%d\n"
            "  Entry: $%.4f  |  Current bid: $%.4f  |  Shares: %.4f  |  Cost: $%.2f\n"
            "  Exit: sell-limit $0.9900 | worthless $0.0100\n"
            "  uPnL: $%.2f  |  Balance: $%.2f  |  Open positions: %d",
            pos.pos_id,
            entry_price, current_bid, shares, cost,
            upnl_now, self.balance, len(self._open_positions),
        )
        if self._notifier is not None:
            iv_pct = self._last_iv * 100 if self._last_iv is not None else None
            self._notifier.notify_trade_open(
                pos_id      = pos.pos_id,
                asset       = "NO",
                entry       = entry_price,
                shares      = shares,
                cost        = cost,
                iv_pct      = iv_pct,
                expiry_secs = self._last_expiry_secs,
            )

    # ── Internal: exit management ─────────────────────────────────────────────

    def _manage_exits(
        self,
        bs_fair_r:   float,
        mkt_bid_yes: float,
        mkt_bid_no:  float,
    ) -> None:
        for pos in self._open_positions[:]:  # copy to allow mutation
            if pos.asset == InventoryType.YES:
                bid = mkt_bid_yes
                if bid >= 0.99:
                    log.info("[PTE] SELL-LIMIT YES @ 0.99 hit — pos #%d", pos.pos_id)
                    self._close_position(pos, bid, CloseReason.TARGET)
                elif bid <= 0.01:
                    if time.time() < self._restore_grace_until:
                        log.debug("[PTE] Grace period active — skipping worthless-YES close for pos #%d", pos.pos_id)
                    else:
                        log.warning("[PTE] YES worthless (bid=%.2f) — pos #%d", bid, pos.pos_id)
                        self._close_position(pos, bid, CloseReason.TARGET)
            elif pos.asset == InventoryType.NO:
                bid = mkt_bid_no
                if bid >= 0.99:
                    log.info("[PTE] SELL-LIMIT NO @ 0.99 hit — pos #%d", pos.pos_id)
                    self._close_position(pos, bid, CloseReason.TARGET)
                elif bid <= 0.01:
                    if time.time() < self._restore_grace_until:
                        log.debug("[PTE] Grace period active — skipping worthless-NO close for pos #%d", pos.pos_id)
                    else:
                        log.warning("[PTE] NO worthless (bid=%.2f) — pos #%d", bid, pos.pos_id)
                        self._close_position(pos, bid, CloseReason.TARGET)

    # ── Internal: close position ──────────────────────────────────────────────

    def _close_position(self, pos: OpenPosition, exit_price: float, reason: CloseReason) -> None:
        ts_close  = time.time()
        duration  = round(ts_close - pos.ts_open, 1)
        pnl       = pos.shares * (exit_price - pos.entry_price)
        pnl_pct   = (pnl / pos.cost * 100) if pos.cost else 0.0
        proceeds  = pos.cost + pnl
        self.balance       += proceeds
        self._realized_pnl += pnl
        self._peak_equity   = max(self._peak_equity, self.balance)

        # Remove from open positions list and persist updated state
        self._open_positions.remove(pos)
        self._save_open_positions()

        self._trade_counter += 1
        m = pos.micro if pos.micro is not None else self._last_micro
        trade = Trade(
            trade_id            = self._trade_counter,
            ts_open             = pos.ts_open,
            ts_close            = ts_close,
            asset               = pos.asset,
            entry_price         = pos.entry_price,
            exit_price          = exit_price,
            shares              = pos.shares,
            pnl                 = round(pnl, 4),
            pnl_pct             = round(pnl_pct, 2),
            duration            = duration,
            close_reason        = reason,
            balance_after       = round(self.balance, 4),
            hurst_at_entry      = m.hurst      if m else 0.5,
            divergence_at_entry = m.divergence if m else 0.0,
            edge_used           = pos.edge_used,
        )
        self._trades.append(trade)

        _append_csv_row({
            "Trade_ID":         self._trade_counter,
            "Timestamp":        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_close)),
            "Asset":            pos.asset.value,
            "Entry_Price":      f"{pos.entry_price:.4f}",
            "Exit_Price":       f"{exit_price:.4f}",
            "Shares":           f"{pos.shares:.4f}",
            "Gross_PnL":        f"{pnl:.4f}",
            "PnL_Pct":          f"{pnl_pct:.2f}",
            "Duration_Seconds": f"{duration:.1f}",
            "Exit_Type":        reason.value,
        })

        self._last_close_str = (
            f"#{pos.pos_id} {pos.asset.value} | "
            f"entry={pos.entry_price:.4f} exit={exit_price:.4f} "
            f"pnl={pnl:+.2f} ({pnl_pct:+.1f}%) | "
            f"open_positions: {len(self._open_positions)}"
        )
        self._last_ts = ts_close

        print(
            f"[TRADE CLOSED] #{pos.pos_id} {pos.asset.value} | "
            f"Entry: {pos.entry_price:.4f} | Exit: {exit_price:.4f} | "
            f"Shares: {pos.shares:.4f} | PnL: ${pnl:+.2f} ({pnl_pct:+.1f}%) | "
            f"Dur: {duration:.0f}s | Reason: {reason.value} | "
            f"Balance: ${self.balance:,.2f} | Open positions: {len(self._open_positions)}"
        )
        if self._notifier is not None:
            session_roi = (self.balance / self._initial_capital - 1.0) * 100
            self._notifier.notify_trade_close(
                trade_id    = self._trade_counter,
                asset       = pos.asset.value,
                pnl         = pnl,
                pnl_pct     = pnl_pct,
                exit_price  = exit_price,
                session_roi = session_roi,
            )
