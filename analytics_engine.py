"""
analytics_engine.py — Volatility Analytics for Binary Prediction Markets
=========================================================================
Provides three analytics computed on live market data:

  1. Realized Volatility (RV)
       Rolling 5-min window of Pyth BTC prices.
       Annualized std-dev of log returns: σ_RV = std(log(S_t / S_{t-1})) * sqrt(252 * T_per_day)

  2. Implied Volatility (IV)
       Root-finds the σ that prices the binary call at the observed market mid.
       Model: P = N(d2),  d2 = (ln(S/K) - 0.5σ²T) / (σ√T)
       Solver: scipy.optimize.brentq when available, else manual bisection fallback.

  3. Binary Vega
       ∂P/∂σ = -φ(d2) * d1 / (σ * √T)
       where d1 = d2 + σ√T

Structured log (emitted on every analytics update):
  [S: 71250] | [T: 0.0014y] | [Mkt_Price: 0.42] | [RV: 55.2%] | [IV: 41.5%] | [Vol_Gap: +13.7%] | [Vega: -0.15]

All input prices are snapped to the 0.01 Polymarket tick grid before use.
No order-execution logic — read-only analytics module.
"""

import math
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Deque, Tuple

log = logging.getLogger("AnalyticsEngine")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_SECS_PER_YEAR: float = 365.25 * 24.0 * 3600.0
_TICK_SIZE:     float = 0.01        # Polymarket minimum price increment
_RV_WINDOW_SECS: int  = 300         # 5-minute rolling window for Pyth prices
_RV_MIN_SAMPLES: int  = 5           # Minimum data points to compute RV
_IV_SIGMA_LO:   float = 0.001       # Brentq lower bound: 0.1% annualized vol
_IV_SIGMA_HI:   float = 5.0         # Brentq upper bound: 500% annualized vol
_IV_TOLERANCE:  float = 1e-6        # Root-finding convergence tolerance
_IV_MAX_ITER:   int   = 100         # Max brentq/bisection iterations
# IV is meaningless when market price is near 0/1 or T is tiny
_IV_PRICE_MIN:  float = 0.02        # Skip IV solve below 2¢
_IV_PRICE_MAX:  float = 0.98        # Skip IV solve above 98¢
_IV_T_MIN:      float = 1e-6        # Skip IV solve below ~31 sec in years (T < 0.000001y)


# ─────────────────────────────────────────────────────────────────────────────
# MATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

try:
    from scipy.stats import norm as _scipy_norm
    from scipy.optimize import brentq as _scipy_brentq

    def _norm_cdf(x: float) -> float:
        return float(_scipy_norm.cdf(x))

    def _norm_pdf(x: float) -> float:
        return float(_scipy_norm.pdf(x))

    def _brentq(f, lo, hi) -> float:
        return float(_scipy_brentq(f, lo, hi, xtol=_IV_TOLERANCE, maxiter=_IV_MAX_ITER))

    _HAS_SCIPY = True

except ImportError:
    _HAS_SCIPY = False

    def _erf_approx(x: float) -> float:
        sign = 1 if x >= 0 else -1
        x = abs(x)
        t = 1.0 / (1.0 + 0.3275911 * x)
        poly = t * (0.254829592 + t * (-0.284496736 + t * (
            1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        return sign * (1.0 - poly * math.exp(-x * x))

    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + _erf_approx(x / math.sqrt(2)))

    def _norm_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    def _brentq(f, lo, hi) -> float:
        """
        Manual bisection fallback — monotonicity-agnostic.

        Works correctly for both ITM binaries (f decreasing in σ) and OTM
        binaries (f increasing in σ).  The key fix vs the naïve version:
        instead of always moving `hi` when f(mid) > 0, we compare the sign
        of f(mid) against f(lo) — same sign means the root is in [mid, hi],
        opposite sign means it is in [lo, mid].  This is correct regardless
        of whether f is increasing or decreasing.
        """
        f_lo = f(lo)
        for _ in range(_IV_MAX_ITER):
            mid = (lo + hi) / 2.0
            f_mid = f(mid)
            if abs(f_mid) < _IV_TOLERANCE:
                return mid
            if f_lo * f_mid > 0:
                # f(lo) and f(mid) have the same sign → root is in (mid, hi)
                lo = mid
                f_lo = f_mid
            else:
                # root is in (lo, mid)
                hi = mid
            if (hi - lo) < _IV_TOLERANCE:
                break
        return (lo + hi) / 2.0


def _snap(p: float) -> float:
    """Snap a price to the 0.01 Polymarket tick grid."""
    return round(round(p / _TICK_SIZE) * _TICK_SIZE, 2)


# ─────────────────────────────────────────────────────────────────────────────
# BINARY OPTION MATH
# ─────────────────────────────────────────────────────────────────────────────

def _binary_price(S: float, K: float, T: float, sigma: float) -> float:
    """
    Binary call price (cash-or-nothing, r=0):
      P = N(d2)
      d2 = (ln(S/K) - 0.5*sigma^2*T) / (sigma * sqrt(T))

    Returns probability in (0, 1).
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    sqrt_T = math.sqrt(T)
    d2 = (math.log(S / K) - 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    return _norm_cdf(d2)


def _d1_d2(S: float, K: float, T: float, sigma: float):
    """Shared helper: returns (d1, d2, sqrt_T)."""
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2, sqrt_T


def _binary_delta(S: float, K: float, T: float, sigma: float) -> float:
    """
    Binary call Delta — ∂P/∂S:
      Δ = φ(d2) / (S · σ · √T)

    Derivation: ∂d2/∂S = 1/(S·σ·√T)
    → Δ = φ(d2) · 1/(S·σ·√T)

    Explodes near expiry at-the-money (denominator → 0).
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    _, d2, sqrt_T = _d1_d2(S, K, T, sigma)
    return _norm_pdf(d2) / (S * sigma * sqrt_T)


def _binary_gamma(S: float, K: float, T: float, sigma: float) -> float:
    """
    Binary call Gamma — ∂²P/∂S²:
      Γ = −φ(d2) · d1 / (S² · σ² · T)

    Derivation: ∂Δ/∂S = ∂/∂S [φ(d2)/(S·σ·√T)]
      = [φ'(d2)·∂d2/∂S·(S·σ·√T) − φ(d2)·σ·√T] / (S·σ·√T)²
      = [−d2·φ(d2)/(S·σ·√T) − φ(d2)/(S)] / (S·σ·√T)
      = −φ(d2)·(d2 + σ√T) / (S²·σ²·T)
      = −φ(d2)·d1 / (S²·σ²·T)   ∎
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1, d2, _ = _d1_d2(S, K, T, sigma)
    return -_norm_pdf(d2) * d1 / (S ** 2 * sigma ** 2 * T)


def _binary_theta(S: float, K: float, T: float, sigma: float) -> float:
    """
    Binary call Theta — ∂P/∂t (t = real time, T = time remaining).
    Since ∂T/∂t = −1:

      ∂d2/∂T = −d1/(2T)        [derivation: d2 = a/√T − 0.5σ√T,
                                  ∂d2/∂T = −a/(2σT√T) − σ/(4√T)
                                         = −[ln(S/K)+0.5σ²T]/(2σT√T)
                                         = −d1·σ√T/(2σT√T) = −d1/(2T)]
      ∂P/∂T = φ(d2)·(−d1/(2T))
      Θ = −∂P/∂T = φ(d2)·d1/(2T)   [per year]

    Note: user formula used d2 in place of d1 (typo — d1 is correct).
    Returned in per-minute units (divide by 525,960 min/year).
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1, d2, _ = _d1_d2(S, K, T, sigma)
    theta_per_year = _norm_pdf(d2) * d1 / (2.0 * T)
    return theta_per_year / (_SECS_PER_YEAR / 60.0)   # → per minute


def _binary_vega(S: float, K: float, T: float, sigma: float) -> float:
    """
    Binary call Vega — ∂P/∂σ:
      ν = −φ(d2) · d1 / σ

    Derivation: ∂d2/∂σ = −d1/σ
    → ν = φ(d2) · (−d1/σ) = −φ(d2)·d1/σ
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1, d2, _ = _d1_d2(S, K, T, sigma)
    return -_norm_pdf(d2) * d1 / sigma


def _solve_iv(
    observed_price: float,
    S: float,
    K: float,
    T: float,
) -> Optional[float]:
    """
    Implied volatility solver: find σ ∈ [0.001, 5.0] such that
      _binary_price(S, K, T, σ) = observed_price

    Uses scipy.optimize.brentq (or bisection fallback).
    Returns None — not a cap value — for all degenerate / out-of-range cases.

    Edge-case bailouts (IV mathematically undefined):
      • mkt_price ≤ _IV_PRICE_MIN or ≥ _IV_PRICE_MAX  (too close to 0/1)
      • T < _IV_T_MIN                                   (< ~5 min to expiry)
      • S, K, T ≤ 0                                     (bad inputs)
    """
    # ── Edge-case bailouts ────────────────────────────────────────────────────
    if S <= 0 or K <= 0 or T <= 0:
        return None
    if T < _IV_T_MIN:
        return None   # too close to expiry — IV meaningless
    if observed_price <= _IV_PRICE_MIN or observed_price >= _IV_PRICE_MAX:
        return None   # extreme price — IV mathematically undefined / unstable

    target = observed_price  # already in valid range

    # ── Evaluate bounds ───────────────────────────────────────────────────────
    try:
        f_lo = _binary_price(S, K, T, _IV_SIGMA_LO) - target
        f_hi = _binary_price(S, K, T, _IV_SIGMA_HI) - target
    except Exception:
        return None

    if not math.isfinite(f_lo) or not math.isfinite(f_hi):
        return None

    # brentq requires f(lo) and f(hi) to have opposite signs.
    # If same sign → target is outside achievable range for this model → IV undefined.
    if f_lo * f_hi >= 0:
        return None

    # ── Root find ─────────────────────────────────────────────────────────────
    try:
        iv = _brentq(
            lambda s: _binary_price(S, K, T, s) - target,
            _IV_SIGMA_LO,
            _IV_SIGMA_HI,
        )
        if not math.isfinite(iv) or iv <= 0:
            return None
        return iv
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnalyticsSnapshot:
    """Output of one AnalyticsEngine.compute() call."""
    btc_spot:    float          # BTC/USD used (Pyth)
    strike_k:    float          # Market strike (price_to_beat)
    tau_years:   float          # Time to expiry in years
    mkt_price:   float          # Observed market mid (snapped to 0.01)
    rv:          Optional[float] = None   # Realized vol (annualized, e.g. 0.552 = 55.2%)
    iv:          Optional[float] = None   # Implied vol (annualized)
    vol_gap:     Optional[float] = None   # RV - IV (positive = market under-pricing vol)
    vega:        Optional[float] = None   # ν = −φ(d2)·d1/σ
    delta:       Optional[float] = None   # Δ = φ(d2)/(S·σ·√T)
    gamma:       Optional[float] = None   # Γ = −φ(d2)·d1/(S²·σ²·T)
    theta:       Optional[float] = None   # Θ per minute = φ(d2)·d1/(2T) ÷ 525960
    rv_samples:  int = 0                  # Number of price samples in RV window


# ─────────────────────────────────────────────────────────────────────────────
# ANALYTICS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AnalyticsEngine:
    """
    Stateful analytics engine — attach one instance to the live dashboard.

    Usage:
        analytics = AnalyticsEngine()

        # Feed every Pyth BTC price tick:
        analytics.add_pyth_price(price, timestamp=time.time())

        # Compute on every strategy tick (pass in live market state):
        snap = analytics.compute(
            btc_spot   = 71250.0,
            strike_k   = 71000.0,
            tau_secs   = 120.0,
            yes_bid    = 0.40,
            yes_ask    = 0.44,
        )
        if snap:
            print(analytics.format_log_line(snap))

        # Share RV with the pricer (replaces hardcoded sigma_btc):
        if snap and snap.rv:
            pricer.sigma_btc = snap.rv
    """

    def __init__(self, rv_window_secs: int = _RV_WINDOW_SECS) -> None:
        self._rv_window_secs = rv_window_secs
        # Deque of (timestamp, log_price) pairs
        self._price_log: Deque[Tuple[float, float]] = deque()
        self._last_snap: Optional[AnalyticsSnapshot] = None

    # ── Data Ingestion ────────────────────────────────────────────────────────

    def add_pyth_price(self, price: float, timestamp: Optional[float] = None) -> None:
        """
        Feed a new BTC/USD price from the Pyth stream.
        Automatically evicts samples older than rv_window_secs.
        """
        if price <= 0:
            return
        ts = timestamp if timestamp is not None else time.time()
        self._price_log.append((ts, math.log(price)))
        cutoff = ts - self._rv_window_secs
        while self._price_log and self._price_log[0][0] < cutoff:
            self._price_log.popleft()

    # ── Realized Volatility ───────────────────────────────────────────────────

    def realized_vol(self) -> Optional[float]:
        """
        Compute annualized realized volatility from the rolling price window.

        Method: std-dev of log returns, annualized by the actual window length.
        Returns None if fewer than _RV_MIN_SAMPLES data points are available.
        """
        pts = list(self._price_log)
        n = len(pts)
        if n < _RV_MIN_SAMPLES:
            return None

        # Log returns
        log_rets = [pts[i + 1][1] - pts[i][1] for i in range(n - 1)]
        if len(log_rets) < 2:
            return None

        mean = sum(log_rets) / len(log_rets)
        variance = sum((r - mean) ** 2 for r in log_rets) / (len(log_rets) - 1)
        std_per_sample = math.sqrt(variance)

        # Annualize: estimate how many samples there are per year
        # Window duration in seconds
        window_secs = pts[-1][0] - pts[0][0]
        if window_secs <= 0:
            return None
        samples_per_sec = (n - 1) / window_secs
        samples_per_year = samples_per_sec * _SECS_PER_YEAR

        return std_per_sample * math.sqrt(samples_per_year)

    # ── Main Compute ──────────────────────────────────────────────────────────

    def compute(
        self,
        btc_spot: float,
        strike_k: float,
        tau_secs: float,
        yes_bid:  float,
        yes_ask:  float,
    ) -> Optional[AnalyticsSnapshot]:
        """
        Compute full analytics snapshot given current market state.

        Returns None if inputs are degenerate (zero spot, zero K, etc.).
        mkt_price is snapped to 0.01 Polymarket tick before use.
        """
        if btc_spot <= 0 or strike_k <= 0 or tau_secs < 1.0:
            return None
        if yes_bid <= 0 or yes_ask <= 0 or yes_ask <= yes_bid:
            return None

        tau_years = tau_secs / _SECS_PER_YEAR
        mkt_price = _snap((yes_bid + yes_ask) / 2.0)

        # 1. Realized Volatility
        rv = self.realized_vol()
        rv_samples = len(self._price_log)

        # 2. Implied Volatility (solve for σ given observed price)
        iv = _solve_iv(mkt_price, btc_spot, strike_k, tau_years)

        # 3. Vol gap
        vol_gap = (rv - iv) if (rv is not None and iv is not None) else None

        # 4. Greeks — IV preferred, RV fallback, VOL_FLOOR=0.50 last resort.
        # Require tau_secs >= 5s to avoid degenerate near-expiry values.
        # Safety clamp: at price extremes (≤ 0.01 or ≥ 0.99), analytical Greeks
        # are numerically unstable (d2 → ±∞). Return 0.0 instead.
        _VOL_FLOOR = 0.50
        vega = delta = gamma = theta = None
        if tau_secs >= 5.0 and tau_years > 0:
            if mkt_price <= 0.01 or mkt_price >= 0.99:
                # Safety clamp: d2 → ±∞ at price extremes → analytical Greeks
                # blow up. Return 0.0 — position is essentially resolved.
                vega = delta = gamma = theta = 0.0
            else:
                sigma_g = iv if iv is not None else (rv if rv is not None else _VOL_FLOOR)

                try:
                    _v = _binary_vega(btc_spot, strike_k, tau_years, sigma_g)
                    if math.isfinite(_v) and abs(_v) <= 5.0:
                        vega = _v
                except Exception:
                    vega = 0.0

                try:
                    _d = _binary_delta(btc_spot, strike_k, tau_years, sigma_g)
                    if math.isfinite(_d) and abs(_d) <= 1.0:   # Δ ∈ [0,1] for binary
                        delta = _d
                except Exception:
                    delta = 0.0

                try:
                    _g = _binary_gamma(btc_spot, strike_k, tau_years, sigma_g)
                    if math.isfinite(_g):
                        gamma = _g
                except Exception:
                    gamma = 0.0

                try:
                    _t = _binary_theta(btc_spot, strike_k, tau_years, sigma_g)
                    if math.isfinite(_t):
                        theta = _t
                except Exception:
                    theta = 0.0

        snap = AnalyticsSnapshot(
            btc_spot   = btc_spot,
            strike_k   = strike_k,
            tau_years  = tau_years,
            mkt_price  = mkt_price,
            rv         = rv,
            iv         = iv,
            vol_gap    = vol_gap,
            vega       = vega,
            delta      = delta,
            gamma      = gamma,
            theta      = theta,
            rv_samples = rv_samples,
        )
        self._last_snap = snap
        return snap

    # ── Logging ───────────────────────────────────────────────────────────────

    @staticmethod
    def format_log_line(snap: AnalyticsSnapshot) -> str:
        """
        Returns a structured single-line analytics status string:
          [S: 71,250] | [T: 12m 04s] | [Mkt_Price: 0.42] | [RV: 55.2%] | [IV: 41.5%] | [Vol_Gap: +13.7%] | [Vega: -0.1500]
        """
        tau_secs  = snap.tau_years * _SECS_PER_YEAR
        t_min     = int(tau_secs // 60)
        t_sec     = int(tau_secs % 60)
        t_str     = f"{t_min}m {t_sec:02d}s"
        rv_str    = f"{snap.rv * 100:.1f}%"       if snap.rv      is not None else "n/a"
        iv_str    = f"{snap.iv * 100:.1f}%"       if snap.iv      is not None else "n/a"
        gap_str   = f"{snap.vol_gap * 100:+.1f}%" if snap.vol_gap is not None else "n/a"
        vega_str  = f"{snap.vega:.4f}"            if snap.vega    is not None else "n/a"
        delta_str = f"{snap.delta:.6f}"           if snap.delta   is not None else "n/a"
        gamma_str = f"{snap.gamma:.2e}"           if snap.gamma   is not None else "n/a"
        theta_str = f"{snap.theta:.6f}/min"       if snap.theta   is not None else "n/a"
        return (
            f"[S: {snap.btc_spot:,.0f}] | "
            f"[T: {t_str}] | "
            f"[Mkt_Price: {snap.mkt_price:.2f}] | "
            f"[RV: {rv_str}] | "
            f"[IV: {iv_str}] | "
            f"[Vol_Gap: {gap_str}] | "
            f"[Vega: {vega_str}] | "
            f"[Δ: {delta_str}] | "
            f"[Γ: {gamma_str}] | "
            f"[Θ/min: {theta_str}]"
        )

    def log_snapshot(self, snap: AnalyticsSnapshot) -> None:
        """Emit the formatted log line at INFO level."""
        log.info(self.format_log_line(snap))

    @property
    def last_snapshot(self) -> Optional[AnalyticsSnapshot]:
        """Last successfully computed snapshot (useful for polling)."""
        return self._last_snap
