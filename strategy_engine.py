"""
strategy_engine.py — Prediction Market Pricing & Decision Layer
===============================================================
Two-layer architecture:

  Layer 1 – Fair Value (Theoretical):
    P(BTC_T > K) under lognormal model.
    Inputs: BTC spot S (Pyth), strike K (price_to_beat), T (seconds to expiry),
            sigma_btc (BTC annual vol).

  Layer 2 – Avellaneda-Stoikov Quoting (Logit Space):
    Spreads around fair value using calibrated belief volatility sigma_b.
    Follows Shaw & Dalen (2025) "Toward Black-Scholes for Prediction Markets".
    https://arxiv.org/pdf/2510.15205

  Tick Filter:
    Kernel recalculation is triggered ONLY when YES bid/ask PRICE changes,
    not on size-only updates → saves CPU, reduces latency.

  Decision Logic:
    If theoretical_bid > real_bid → beat the market at real_bid + 0.001
    (capped at theoretical_bid as safety upper limit).
    If theoretical_ask < real_ask → beat the market at real_ask - 0.001
    (floored at theoretical_ask as safety lower limit).
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("StrategyEngine")

# ─────────────────────────────────────────────────────────────────────────────
# MATH HELPERS — no scipy required (pure Python fallback)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from scipy.stats import norm as _scipy_norm
    def _norm_cdf(x: float) -> float:
        return float(_scipy_norm.cdf(x))
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    def _erf_approx(x: float) -> float:
        """Abramowitz & Stegun erf approximation (max error: 1.5e-7)."""
        sign = 1 if x >= 0 else -1
        x = abs(x)
        t = 1.0 / (1.0 + 0.3275911 * x)
        poly = t * (0.254829592 + t * (-0.284496736 + t * (
            1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        return sign * (1.0 - poly * math.exp(-x * x))

    def _norm_cdf(x: float) -> float:
        """Standard normal CDF via erf approximation."""
        return 0.5 * (1.0 + _erf_approx(x / math.sqrt(2)))


def sigmoid(x: float) -> float:
    """Logit → probability (logistic function)."""
    if x > 500:  return 1.0
    if x < -500: return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    """Probability → logit (log-odds). Clamps to avoid ±inf."""
    p = max(1e-9, min(1.0 - 1e-9, p))
    return math.log(p / (1.0 - p))


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KernelQuote:
    """Full output of one PredictionMarketPricer.decide() call."""
    fair_price:       float         # P(BTC > K at expiry) — YES theoretical probability
    theoretical_bid:  float         # YES A-S spread bid
    theoretical_ask:  float         # YES A-S spread ask
    real_bid:         float         # YES market best bid (from Polymarket CLOB)
    real_ask:         float         # YES market best ask
    action:           str           # "BID" | "ASK" | "WAIT" | "CUTOFF"
    action_price:     Optional[float] = None  # Proposed limit price (YES side)
    sigma_b:          float = 0.0   # Calibrated belief volatility (logit space)
    sigma_btc:        float = 0.0   # BTC annual vol used for fair value
    tau_secs:         float = 0.0   # Seconds to expiry used
    inventory:        float = 0.0   # Net YES position (shares)
    fair_source:      str   = "?"   # "lognormal" | "market_mid" | "prior"
    btc_spot:         float = 0.0   # BTC spot used for fair value
    strike_k:         float = 0.0   # Strike K used
    # NO/DOWN kernel (derived from YES by complement)
    no_fair_price:    float = 0.0   # P(BTC ≤ K) = 1 - fair_price
    no_theoretical_bid: float = 0.0 # NO bid = 1 - YES ask
    no_theoretical_ask: float = 0.0 # NO ask = 1 - YES bid
    no_action:        str   = "WAIT"
    no_action_price:  Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# PRICER
# ─────────────────────────────────────────────────────────────────────────────

_SECS_PER_YEAR: float = 365.25 * 24.0 * 3600.0

# Polymarket minimum tradable tick size
TICK_SIZE: float = 0.01   # $0.01 — minimum price increment on Polymarket CLOB

# Safety guard: don't let A-S spreads be tighter than one tick each side
_MIN_HALF_SPREAD: float = TICK_SIZE   # 1 tick = 0.01 minimum each side

# ── Risk Manager — Hard Cutoff ────────────────────────────────────────────────
# When T ≤ HARD_CUTOFF_SECONDS the bot enters FLAT/OBSERVE mode:
#   • No new BID or ASK signals are emitted
#   • All pending quotes are considered cancelled (paper + real)
#   • Action output is forced to "CUTOFF"
# Rationale: gamma / pin risk near expiry makes fills unprofitable.
HARD_CUTOFF_SECONDS: float = 90.0   # 1 min 30 sec before expiry → stop quoting


class PredictionMarketPricer:
    """
    Market-making pricer for Polymarket binary prediction markets.

    Usage:
        pricer = PredictionMarketPricer()

        # On rollover (new market starts):
        pricer.reset_market(strike_k=price_to_beat, seconds_to_expiry=900.0)

        # When Pyth BTC price updates:
        changed = pricer.update_btc(btc_spot=83000.0)

        # Every ~1 second (theta decay):
        pricer.update_tau(seconds_to_expiry=market.seconds_to_expiry)

        # On every WS tick (tick-filtered):
        if pricer.process_tick(yes_bid, yes_ask):
            quote = pricer.decide(yes_bid, yes_ask)
            print(pricer.format_log_line(yes_bid, yes_ask, quote))
    """

    # Defaults
    DEFAULT_SIGMA_BTC: float = 0.80   # BTC annual vol (80%) — wide prior
    DEFAULT_SIGMA_B:   float = 1.50   # Belief vol in logit space (calibrated from spread)
    DEFAULT_GAMMA:     float = 0.10   # Inventory aversion (A-S risk parameter)
    DEFAULT_K:         float = 1.50   # Liquidity depth parameter

    def __init__(
        self,
        gamma:     float = DEFAULT_GAMMA,
        k:         float = DEFAULT_K,
        sigma_btc: float = DEFAULT_SIGMA_BTC,
    ) -> None:
        # Risk parameters
        self.gamma:     float = gamma
        self.k:         float = k
        self.sigma_btc: float = sigma_btc
        self.sigma_b:   float = self.DEFAULT_SIGMA_B

        # Market state
        self.strike_k:    float = 0.0   # BTC price to beat (from ActiveMarket)
        self.btc_spot:    float = 0.0   # Current BTC price (Pyth feed)
        self.tau_secs:    float = 0.0   # Seconds to expiry
        self.inventory:   float = 0.0   # Net YES position (USDC shares); updated externally
        self.market_mid:  float = 0.5   # Latest market mid-price (yes_bid+yes_ask)/2
                                        # Used as fallback fair value when K is unknown

        # Cached kernel output
        self.fair_price:       float = 0.5
        self.fair_source:      str   = "prior"  # tracks which layer produced fair_price
        self.theoretical_bid:  float = 0.49
        self.theoretical_ask:  float = 0.51

        # Tick filter — skip recalc when only size changes, not price
        self._last_yes_bid: float = -1.0
        self._last_yes_ask: float = -1.0

        # Risk Manager state
        self._cutoff_logged: bool = False   # emit the cutoff log line only once per market

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset_market(self, strike_k: float, seconds_to_expiry: float) -> None:
        """
        Called at rollover boundary. Resets strike K and time T instantly.
        Inventory carries over (positions in old market may still be live).
        Clears tick filter so the first tick of the new market triggers recalc.
        """
        old_k = self.strike_k
        self.strike_k      = strike_k
        self.tau_secs      = max(0.0, seconds_to_expiry)
        self.sigma_b       = self.DEFAULT_SIGMA_B   # recalibrate fresh each market
        self._last_yes_bid = -1.0
        self._last_yes_ask = -1.0
        self._cutoff_logged = False   # reset so cutoff fires once in new market
        self._recalculate()
        log.info(
            "[Kernel] Market reset: K %.2f→%.2f  tau=%.0fs  fair_price=%.4f",
            old_k, strike_k, seconds_to_expiry, self.fair_price,
        )

    # ── External update triggers ──────────────────────────────────────────────

    def update_btc(self, btc_spot: float) -> bool:
        """
        Called when Pyth BTC price updates.
        Returns True if the fair price changed materially (>0.01%).
        Ignores sub-$1 movements to reduce pointless recalculations.
        """
        if abs(btc_spot - self.btc_spot) < 1.0:
            return False
        self.btc_spot = btc_spot
        old_fp = self.fair_price
        self._recalculate()
        return abs(self.fair_price - old_fp) > 0.0001

    def update_tau(self, seconds_to_expiry: float) -> bool:
        """
        Called every ~1 second for theta decay.
        Returns True if theoretical quotes changed materially (>0.01%).
        Ignores sub-1-second changes.
        """
        new_tau = max(0.0, seconds_to_expiry)
        if abs(new_tau - self.tau_secs) < 1.0:
            return False
        self.tau_secs = new_tau
        old_bid = self.theoretical_bid
        self._recalculate()
        return abs(self.theoretical_bid - old_bid) > 0.0001

    def process_tick(self, yes_bid: float, yes_ask: float) -> bool:
        """
        TICK FILTER: returns True only if yes_bid or yes_ask PRICE changed.
        On True, calibrates sigma_b from the new spread and recalculates quotes.

        Size-only updates return False immediately — kernel is NOT called.
        This is the primary CPU-saver for high-frequency feeds.
        """
        if yes_bid == self._last_yes_bid and yes_ask == self._last_yes_ask:
            return False  # size-only update → skip
        self._last_yes_bid = yes_bid
        self._last_yes_ask = yes_ask
        # Update market mid — used as fallback fair value when strike_k is unknown
        if yes_bid > 0 and yes_ask > 0:
            self.market_mid = (yes_bid + yes_ask) / 2.0
        self._calibrate_sigma_b(yes_bid, yes_ask)
        self._recalculate()
        return True

    @property
    def in_cutoff(self) -> bool:
        """True when T ≤ HARD_CUTOFF_SECONDS — no quoting allowed."""
        return self.tau_secs <= HARD_CUTOFF_SECONDS

    def decide(self, real_bid: float, real_ask: float) -> KernelQuote:
        """
        Compare kernel quotes with real market and output a trading action.

        Hard Cutoff (T ≤ HARD_CUTOFF_SECONDS):
          Returns action="CUTOFF" — no BID/ASK emitted.
          All pending quotes must be cancelled. System observes only.

        Bid logic:
          If theoretical_bid > real_bid: quote at real_bid + 0.01
          Safety cap: proposed bid must not exceed theoretical_bid + 1 tick

        Ask logic:
          If theoretical_ask < real_ask: quote at real_ask - 0.01
          Safety floor: proposed ask must not go below theoretical_ask - 1 tick

        Returns KernelQuote with action ∈ {"BID", "ASK", "WAIT", "CUTOFF"}.
        """
        # ── Hard Cutoff — pin risk protection ───────────────────────────────
        if self.in_cutoff:
            if not self._cutoff_logged:
                self._cutoff_logged = True
                t_min = int(self.tau_secs // 60)
                t_sec = int(self.tau_secs % 60)
                log.warning(
                    "[RiskManager] HARD CUTOFF at T=%dm %02ds (≤%.0fs) — "
                    "cancelling all quotes, entering OBSERVE mode.",
                    t_min, t_sec, HARD_CUTOFF_SECONDS,
                )
            _no_bid_co = round(round((1.0 - self.theoretical_ask) / TICK_SIZE) * TICK_SIZE, 2)
            _no_ask_co = round(round((1.0 - self.theoretical_bid) / TICK_SIZE) * TICK_SIZE, 2)
            _no_bid_co = max(TICK_SIZE, min(1.0 - TICK_SIZE, _no_bid_co))
            _no_ask_co = max(TICK_SIZE, min(1.0 - TICK_SIZE, _no_ask_co))
            return KernelQuote(
                fair_price          = round(self.fair_price, 4),
                theoretical_bid     = self.theoretical_bid,
                theoretical_ask     = self.theoretical_ask,
                real_bid            = real_bid,
                real_ask            = real_ask,
                action              = "CUTOFF",
                action_price        = None,
                sigma_b             = round(self.sigma_b,   4),
                sigma_btc           = round(self.sigma_btc, 4),
                tau_secs            = round(self.tau_secs,  1),
                inventory           = round(self.inventory, 4),
                fair_source         = self.fair_source,
                btc_spot            = round(self.btc_spot,  2),
                strike_k            = round(self.strike_k,  2),
                no_fair_price       = round(1.0 - self.fair_price, 4),
                no_theoretical_bid  = _no_bid_co,
                no_theoretical_ask  = _no_ask_co,
                no_action           = "CUTOFF",
                no_action_price     = None,
            )

        action = "WAIT"
        action_price = None

        # ── YES BID signal ───────────────────────────────────────────────────
        if self.theoretical_bid > real_bid:
            candidate = round(real_bid + TICK_SIZE, 2)
            if candidate <= self.theoretical_bid + TICK_SIZE:
                action       = "BID"
                action_price = candidate

        # ── YES ASK signal ───────────────────────────────────────────────────
        if action == "WAIT" and self.theoretical_ask < real_ask:
            candidate = round(real_ask - TICK_SIZE, 2)
            if candidate >= self.theoretical_ask - TICK_SIZE:
                action       = "ASK"
                action_price = candidate

        # ── NO/DOWN kernel (complement of YES) ──────────────────────────────
        # P_NO = 1 - P_YES.  Under no-arbitrage:
        #   NO_bid = 1 - YES_ask  (best price to buy NO = best price to sell YES)
        #   NO_ask = 1 - YES_bid
        # Snapped to 0.01 grid.
        no_fair  = round(1.0 - self.fair_price, 4)
        no_bid   = round(round((1.0 - self.theoretical_ask) / TICK_SIZE) * TICK_SIZE, 2)
        no_ask   = round(round((1.0 - self.theoretical_bid) / TICK_SIZE) * TICK_SIZE, 2)
        # Clamp to valid range
        no_bid   = max(TICK_SIZE, min(1.0 - TICK_SIZE, no_bid))
        no_ask   = max(TICK_SIZE, min(1.0 - TICK_SIZE, no_ask))

        # NO action: mirror of YES action (BID YES ↔ ASK NO, ASK YES ↔ BID NO)
        no_action = "WAIT"
        no_action_price = None
        if action == "BID" and action_price is not None:
            # Buying YES at X ≡ Selling NO at (1-X)
            no_action = "ASK"
            no_action_price = round(round((1.0 - action_price) / TICK_SIZE) * TICK_SIZE, 2)
        elif action == "ASK" and action_price is not None:
            # Selling YES at X ≡ Buying NO at (1-X)
            no_action = "BID"
            no_action_price = round(round((1.0 - action_price) / TICK_SIZE) * TICK_SIZE, 2)

        return KernelQuote(
            fair_price          = round(self.fair_price, 4),
            theoretical_bid     = self.theoretical_bid,
            theoretical_ask     = self.theoretical_ask,
            real_bid            = real_bid,
            real_ask            = real_ask,
            action              = action,
            action_price        = action_price,
            sigma_b             = round(self.sigma_b,   4),
            sigma_btc           = round(self.sigma_btc, 4),
            tau_secs            = round(self.tau_secs,  1),
            inventory           = round(self.inventory, 4),
            fair_source         = self.fair_source,
            btc_spot            = round(self.btc_spot,  2),
            strike_k            = round(self.strike_k,  2),
            no_fair_price       = no_fair,
            no_theoretical_bid  = no_bid,
            no_theoretical_ask  = no_ask,
            no_action           = no_action,
            no_action_price     = no_action_price,
        )

    def format_log_line(
        self,
        real_bid: float,
        real_ask: float,
        quote: KernelQuote,
    ) -> str:
        """
        Returns a rich single-line strategy status string:
          [BTC: $83,242] [K: $83,000] [T: 3m 27s] [MKT: 0.510/0.520] |
          [BS-Fair: 0.5553 (lognormal)] | [BS: 0.55/0.56] | [ACTION: BID @ 0.52]
        """
        action_str = (
            f"{quote.action} @ {quote.action_price:.2f}"
            if quote.action_price is not None
            else quote.action
        )
        t_min   = int(quote.tau_secs // 60)
        t_sec   = int(quote.tau_secs % 60)
        t_str   = f"{t_min}m {t_sec:02d}s"
        btc_str = f"${quote.btc_spot:,.0f}" if quote.btc_spot > 0 else "—"
        k_str   = f"${quote.strike_k:,.0f}" if quote.strike_k  > 0 else "—"
        return (
            f"[BTC: {btc_str}] [K: {k_str}] [T: {t_str}] "
            f"[MKT: {real_bid:.2f}/{real_ask:.2f}] | "
            f"[BS-Fair: {quote.fair_price:.4f} ({quote.fair_source})] | "
            f"[BS: {quote.theoretical_bid:.2f}/{quote.theoretical_ask:.2f}] | "
            f"[ACTION: {action_str}]"
        )

    # ── Internal calculations ─────────────────────────────────────────────────

    # Volatility floor: BTC fat tails mean RV can never be treated as <50% annualised.
    # Without this, a calm 15-min window would produce near-binary 0/1 prices and
    # make the kernel dangerously over-confident about small spot/strike gaps.
    _VOL_FLOOR: float = 0.50   # 50% annualised minimum effective sigma

    def _fair_value(self) -> float:
        """
        Layer 1 — Lognormal fair value P(BTC_T > K)  [Cash-or-Nothing Binary Call].

        Formula (r = 0):
            effective_sigma = max(sigma_btc, VOL_FLOOR)   ← fat-tail guard
            T_years         = max(tau_secs, 0.5) / 31_536_000
            d2              = (ln(S/K) − 0.5·σ²·T) / (σ·√T)
            P               = N(d2)   rounded to 4dp, clamped [0.01, 0.99]

        Priority:
          1. Lognormal  — when strike_k > 0, btc_spot > 0, tau_secs >= 1s
          2. Market mid — K or S unavailable; use (yes_bid + yes_ask)/2
          3. Prior 0.5  — no market data yet

        Source is stored in self.fair_source for logging/UI transparency.
        """
        # ── 1. Full lognormal model ───────────────────────────────────────────
        if self.strike_k > 0 and self.btc_spot > 0 and self.tau_secs >= 1.0:
            # Volatility floor: prevents over-confidence when RV is low
            effective_sigma: float = max(self.sigma_btc, self._VOL_FLOOR)
            # Time: floor at 0.5s to avoid asymptotic d2 in the final tick
            T_years: float = max(self.tau_secs, 0.5) / 31_536_000.0
            try:
                d2 = (
                    math.log(self.btc_spot / self.strike_k)
                    - 0.5 * effective_sigma ** 2 * T_years
                ) / (effective_sigma * math.sqrt(T_years))
                self.fair_source = "lognormal"
                return round(max(0.01, min(0.99, _norm_cdf(d2))), 4)
            except (ValueError, ZeroDivisionError):
                pass   # fall through to market_mid

        # ── 2. Market-implied fair value (when K or S unknown) ───────────────
        if 0.001 < self.market_mid < 0.999:
            self.fair_source = "market_mid"
            return self.market_mid

        # ── 3. Uninformative prior ────────────────────────────────────────────
        self.fair_source = "prior"
        return 0.5

    def _calibrate_sigma_b(self, market_bid: float, market_ask: float) -> None:
        """
        Infer belief volatility sigma_b from observed market spread.

        From A-S theory: half_spread ≈ gamma * sigma_b² * tau_y / 2
        Solving: sigma_b = sqrt(2 * half_spread / (gamma * tau_y))

        This calibration runs only when a price change is detected (tick filter).
        """
        if self.tau_secs < 1.0:
            return
        half_spread = max(0.001, (market_ask - market_bid) / 2.0)
        tau_y = self.tau_secs / _SECS_PER_YEAR
        try:
            sigma_b = math.sqrt(2.0 * half_spread / (self.gamma * tau_y))
            # Clamp to a reasonable range for prediction markets
            self.sigma_b = max(0.10, min(50.0, sigma_b))
        except (ValueError, ZeroDivisionError):
            pass

    def _recalculate(self) -> None:
        """
        Layer 2 — Avellaneda-Stoikov quoting in logit space.

        Following Shaw & Dalen (2025):
          fair_x          = logit(fair_price)
          reservation_x   = fair_x - q * gamma * sigma_b² * tau_y
          half_spread_x   = gamma * sigma_b² * tau_y / 2
                          + ln(1 + gamma / k) / 2
          bid_x = reservation_x - half_spread_x
          ask_x = reservation_x + half_spread_x

        Convert back to probability space:
          bid_p = sigmoid(bid_x)
          ask_p = sigmoid(ask_x)

        The logit transform ensures quotes stay in (0, 1) by construction.
        """
        fp = self._fair_value()
        self.fair_price = fp

        if self.tau_secs < 1.0:
            # At/past expiry: collapse to fair price ± one tick (already on grid)
            raw_bid = max(TICK_SIZE, fp - _MIN_HALF_SPREAD)
            raw_ask = min(1.0 - TICK_SIZE, fp + _MIN_HALF_SPREAD)
            self.theoretical_bid = round(round(raw_bid / TICK_SIZE) * TICK_SIZE, 2)
            self.theoretical_ask = round(round(raw_ask / TICK_SIZE) * TICK_SIZE, 2)
            return

        tau_y  = self.tau_secs / _SECS_PER_YEAR
        fair_x = logit(fp)

        # Reservation price shifts away from fair value by inventory * risk
        q = self.inventory
        reservation_x = fair_x - q * self.gamma * (self.sigma_b ** 2) * tau_y

        # Half spread: risk term + liquidity term
        try:
            spread_x = (
                self.gamma * (self.sigma_b ** 2) * tau_y / 2.0
                + math.log(1.0 + self.gamma / self.k) / 2.0
            )
        except (ValueError, ZeroDivisionError):
            spread_x = 0.1   # fallback to ~logit-space 0.1 (≈ 2.5¢ at mid=0.5)

        bid_x = reservation_x - spread_x
        ask_x = reservation_x + spread_x

        # Snap to Polymarket tick grid (0.01) at the kernel level
        raw_bid = max(TICK_SIZE, min(1.0 - TICK_SIZE, sigmoid(bid_x)))
        raw_ask = max(TICK_SIZE, min(1.0 - TICK_SIZE, sigmoid(ask_x)))
        self.theoretical_bid = round(round(raw_bid / TICK_SIZE) * TICK_SIZE, 2)
        self.theoretical_ask = round(round(raw_ask / TICK_SIZE) * TICK_SIZE, 2)
        # At extremes (fair ≈ 0 or ≈ 1) both bid/ask can snap to the same grid level.
        # Enforce minimum 1-tick spread so the NO complement (1-bid, 1-ask) is meaningful.
        if self.theoretical_ask <= self.theoretical_bid:
            self.theoretical_bid = max(TICK_SIZE,        self.theoretical_ask - TICK_SIZE)
            self.theoretical_ask = min(1.0 - TICK_SIZE,  self.theoretical_bid  + TICK_SIZE)
