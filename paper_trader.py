"""
paper_trader.py — Simulated Order Execution for Polymarket MM Bot
=================================================================
Tracks paper trades driven by the strategy kernel's BID/ASK signals.

Rules:
  - One trade per unique action_price (avoids re-entering at the same level)
  - Position limit: ±MAX_ABS_POS shares (prevents runaway accumulation)
  - Unit size: 1.0 share per signal (configurable via UNIT_SIZE)
  - BUY  signal  → long  YES at action_price  (pays cash)
  - SELL signal  → short YES at action_price  (receives cash)

P&L accounting (mark-to-market):
  Portfolio value  = cash_flow + position * current_mid
  Unrealized P&L   = cash_flow + position * current_mid
    (cash_flow starts at 0 for each market; negative = net paid)

At rollover / resolution:
  Call resolve(outcome) where outcome = 1.0 (YES wins) or 0.0 (NO wins).
  Settled P&L = cash_flow + position * outcome
  Then position and cash_flow reset for the next market.

No order execution. Read-only analytics — just tracks what WOULD have happened.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List

log = logging.getLogger("PaperTrader")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

UNIT_SIZE:    float = 1.0    # Shares per paper trade
MAX_ABS_POS:  float = 10.0   # Max long or short position (shares)


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PaperTrade:
    """Record of a single simulated fill."""
    ts:      float   # Unix timestamp
    action:  str     # "BUY" | "SELL"
    price:   float   # Fill price
    qty:     float   # Quantity (always UNIT_SIZE)
    mkt_id:  str     # condition_id of the market


@dataclass
class PaperStats:
    """Snapshot of paper trading state for the UI."""
    position:       float          # Net YES position (+ long, − short)
    avg_entry:      Optional[float]  # Weighted average entry price
    unrealized_pnl: float          # Mark-to-market P&L (current market)
    realized_pnl:   float          # Cumulative settled P&L (all markets)
    total_pnl:      float          # unrealized + realized
    trade_count:    int
    last_action:    Optional[str]  # e.g. "BUY @ 0.41"
    last_ts:        Optional[float]


# ─────────────────────────────────────────────────────────────────────────────
# PAPER TRADER
# ─────────────────────────────────────────────────────────────────────────────

class PaperTrader:
    """
    Stateful paper trading engine.

    Usage:
        pt = PaperTrader()

        # On each strategy tick (call only when price changed):
        executed = pt.try_execute(
            action       = quote.action,
            action_price = quote.action_price,
            current_mid  = (yes_bid + yes_ask) / 2,
            mkt_id       = condition_id,
        )

        # Get UI snapshot:
        stats = pt.stats(current_mid=(yes_bid + yes_ask) / 2)

        # At market resolution:
        pt.resolve(outcome=1.0)   # 1.0 = YES won, 0.0 = NO won
    """

    def __init__(self) -> None:
        self.position:     float = 0.0    # net YES shares
        self.cash_flow:    float = 0.0    # cumulative cash paid/received this market
        self.realized_pnl: float = 0.0    # settled P&L across all markets
        self.trades:       List[PaperTrade] = []
        self._current_mkt_id:    str            = ""
        self._last_action_price: Optional[float] = None
        self._last_action:       Optional[str]   = None
        self._last_ts:           Optional[float] = None
        # For avg entry calculation
        self._total_buy_cost:    float = 0.0
        self._total_buy_qty:     float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset_market(self, mkt_id: str, close_at_mid: Optional[float] = None) -> None:
        """
        Called at rollover. If close_at_mid is provided, marks current position
        as realized at that price (approximate settlement). Otherwise position
        carries over as realized_pnl loss/gain at last known mid.
        """
        if close_at_mid is not None and self.position != 0.0:
            settled = self.cash_flow + self.position * close_at_mid
            self.realized_pnl += settled
            log.info(
                "[Paper] Market closed at mid %.4f — settled P&L: %+.4f  "
                "(cumulative: %+.4f)",
                close_at_mid, settled, self.realized_pnl,
            )
        self.position      = 0.0
        self.cash_flow     = 0.0
        self._last_action_price   = None
        self._total_buy_cost      = 0.0
        self._total_buy_qty       = 0.0
        self._current_mkt_id      = mkt_id
        log.info("[Paper] Market reset → %s", mkt_id[:16] if mkt_id else "?")

    def resolve(self, outcome: float, mkt_id: str = "") -> float:
        """
        Settle the current position at the known outcome (1.0 or 0.0).
        Returns the settled P&L for this market.
        """
        settled = self.cash_flow + self.position * outcome
        self.realized_pnl += settled
        log.info(
            "[Paper] Resolved outcome=%.0f  position=%.1f  settled=%+.4f  "
            "cumulative=%+.4f",
            outcome, self.position, settled, self.realized_pnl,
        )
        self.position    = 0.0
        self.cash_flow   = 0.0
        self._last_action_price = None
        self._total_buy_cost    = 0.0
        self._total_buy_qty     = 0.0
        return settled

    # ── Trade execution ───────────────────────────────────────────────────────

    def try_execute(
        self,
        action:       str,
        action_price: Optional[float],
        current_mid:  float,
        mkt_id:       str = "",
    ) -> bool:
        """
        Attempt a paper trade based on the strategy signal.
        Returns True if a trade was executed.

        Skips if:
          - action is WAIT or action_price is None
          - Same price as last executed trade (dedup)
          - Position limit would be breached
        """
        if action == "WAIT" or action_price is None:
            return False

        # Dedup: skip if same action + price as last trade
        if action_price == self._last_action_price:
            return False

        side = "BUY" if action == "BID" else "SELL"

        # Position limits
        if side == "BUY"  and self.position >= MAX_ABS_POS:
            log.debug("[Paper] BUY skipped — max long position reached (%.1f)", self.position)
            return False
        if side == "SELL" and self.position <= -MAX_ABS_POS:
            log.debug("[Paper] SELL skipped — max short position reached (%.1f)", self.position)
            return False

        qty = UNIT_SIZE
        if side == "BUY":
            self.position  += qty
            self.cash_flow -= action_price * qty
            self._total_buy_cost += action_price * qty
            self._total_buy_qty  += qty
        else:
            self.position  -= qty
            self.cash_flow += action_price * qty

        self._last_action_price = action_price
        self._last_action       = f"{side} @ {action_price:.2f}"
        self._last_ts           = time.time()

        trade = PaperTrade(
            ts     = self._last_ts,
            action = side,
            price  = action_price,
            qty    = qty,
            mkt_id = mkt_id or self._current_mkt_id,
        )
        self.trades.append(trade)

        log.info(
            "[Paper] %s %.1f YES @ %.4f  pos=%.1f  cash=%+.4f  uPnL=%+.4f",
            side, qty, action_price,
            self.position, self.cash_flow,
            self.cash_flow + self.position * current_mid,
        )
        return True

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def stats(self, current_mid: float) -> PaperStats:
        """
        Build a PaperStats snapshot for the UI / logging.
        current_mid should be (yes_bid + yes_ask) / 2 from the live order book.
        """
        unrealized = self.cash_flow + self.position * current_mid
        total      = unrealized + self.realized_pnl
        avg_entry  = (
            self._total_buy_cost / self._total_buy_qty
            if self._total_buy_qty > 0 else None
        )
        return PaperStats(
            position       = round(self.position, 2),
            avg_entry      = round(avg_entry, 4) if avg_entry is not None else None,
            unrealized_pnl = round(unrealized, 4),
            realized_pnl   = round(self.realized_pnl, 4),
            total_pnl      = round(total, 4),
            trade_count    = len(self.trades),
            last_action    = self._last_action,
            last_ts        = self._last_ts,
        )

    def format_log_line(self, current_mid: float) -> str:
        s = self.stats(current_mid)
        pos_str = (
            f"+{s.position:.1f} YES" if s.position > 0
            else f"{s.position:.1f} YES" if s.position < 0
            else "FLAT"
        )
        entry_str = f"{s.avg_entry:.4f}" if s.avg_entry is not None else "—"
        return (
            f"[Paper] pos={pos_str}  entry={entry_str}  "
            f"uPnL={s.unrealized_pnl:+.4f}  rPnL={s.realized_pnl:+.4f}  "
            f"tot={s.total_pnl:+.4f}  trades={s.trade_count}"
        )
