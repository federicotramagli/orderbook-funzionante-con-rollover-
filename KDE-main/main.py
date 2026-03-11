"""
Polymarket BTC 15-Min Microstructure KDE Pipeline
──────────────────────────────────────────────────
Orchestrates the Rust PyO3 engine for data recording,
segments by time phase, and generates dark-themed KDE plots.
"""

import time
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

import requests
import polars as pl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ─── Market Discovery ──────────────────────────────────────────────────────

def discover_market_python() -> dict:
    """Discover the next active BTC 15-min market via Gamma API."""
    now = int(time.time())
    next_slot = ((now // 900) + 1) * 900
    slug = f"btc-updown-15m-{next_slot}"

    url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    print(f"[*] Querying Gamma API: {slug}")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    events = resp.json()

    if not events:
        # Try current slot instead of next
        current_slot = (now // 900) * 900
        slug = f"btc-updown-15m-{current_slot}"
        url = f"https://gamma-api.polymarket.com/events?slug={slug}"
        print(f"[*] Retrying with current slot: {slug}")
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        events = resp.json()

    if not events:
        raise RuntimeError(f"No BTC 15-min market found for slug: {slug}")

    event = events[0]
    market = event["markets"][0]
    condition_id = market["conditionId"]
    tokens = json.loads(market["clobTokenIds"])
    question = market.get("question", slug)

    print(f"[+] Found: {question}")
    print(f"    Condition ID: {condition_id}")
    print(f"    Up token:  {tokens[0][:30]}...")
    print(f"    Down token: {tokens[1][:30]}...")

    return {
        "slug": slug,
        "condition_id": condition_id,
        "up_token_id": tokens[0],
        "down_token_id": tokens[1],
        "question": question,
    }


# ─── Timing ─────────────────────────────────────────────────────────────────

def wait_for_boundary():
    """Wait until the next 15-minute clock boundary (XX:00, XX:15, XX:30, XX:45)."""
    now = time.time()
    current_slot_end = ((int(now) // 900) + 1) * 900
    wait_secs = current_slot_end - now

    if wait_secs < 5:
        # Too close, skip to next
        current_slot_end += 900
        wait_secs = current_slot_end - now

    boundary = datetime.fromtimestamp(current_slot_end, tz=timezone.utc)
    print(f"[*] Waiting {wait_secs:.1f}s until next boundary: {boundary.strftime('%H:%M:%S')} UTC")

    # Countdown
    while True:
        remaining = current_slot_end - time.time()
        if remaining <= 0:
            break
        if remaining > 60:
            print(f"    {remaining:.0f}s remaining...", end="\r")
            time.sleep(30)
        elif remaining > 5:
            print(f"    {remaining:.0f}s remaining...", end="\r")
            time.sleep(1)
        else:
            time.sleep(0.01)

    print(f"\n[+] Boundary reached at {datetime.now(timezone.utc).strftime('%H:%M:%S.%f')} UTC")
    return current_slot_end


# ─── Data Segmentation ──────────────────────────────────────────────────────

def segment_data(df: pl.DataFrame, boundary_ts_s: int) -> dict:
    """
    Segment data into three time phases:
    - First 12 Min:       [boundary, boundary + 720s)
    - Last 3 Min excl 30s: [boundary + 720s, boundary + 870s)
    - Last 30 Sec:        [boundary + 870s, boundary + 900s]
    """
    b = boundary_ts_s * 1000  # convert to ms

    first_12 = df.filter(
        (pl.col("timestamp_ms") >= b) & (pl.col("timestamp_ms") < b + 720_000)
    )
    last_3 = df.filter(
        (pl.col("timestamp_ms") >= b + 720_000) & (pl.col("timestamp_ms") < b + 870_000)
    )
    last_30s = df.filter(
        (pl.col("timestamp_ms") >= b + 870_000) & (pl.col("timestamp_ms") <= b + 900_000)
    )

    print(f"[+] Segments: first_12={len(first_12)}, last_3={len(last_3)}, last_30s={len(last_30s)}")

    return {
        "First 12 Min": first_12,
        "Last 3 Min (excl 30s)": last_3,
        "Last 30 Sec": last_30s,
    }


# ─── KDE Plotting ───────────────────────────────────────────────────────────

FEATURES = [
    "tick_intensity",
    "sample_entropy",
    "permutation_entropy",
    "contract_price_entropy",
    "hurst_exponent",
    "microprice_divergence",
]

FEATURE_LABELS = [
    "Tick Intensity (Updates/Sec)",
    "Sample Entropy (Predictability)",
    "Permutation Entropy",
    "Contract Price Entropy (Price concentration)",
    "Hurst Exponent (Trending > 0.5)",
    "Microprice Divergence (vwap - mid)",
]

SEGMENT_COLORS = {
    "First 12 Min": "#00ffff",            # cyan
    "Last 3 Min (excl 30s)": "#ff00ff",   # magenta
    "Last 30 Sec": "#ffff00",             # yellow
}

SEGMENT_LINEWIDTHS = {
    "First 12 Min": 1.8,
    "Last 3 Min (excl 30s)": 1.8,
    "Last 30 Sec": 2.2,
}


def _clean_data(col_data: np.ndarray) -> np.ndarray:
    """Remove NaN/inf and clip outliers beyond 3 sigma."""
    col_data = col_data[np.isfinite(col_data)]
    if len(col_data) < 3:
        return col_data
    mu, sigma = np.mean(col_data), np.std(col_data)
    if sigma > 0:
        col_data = col_data[np.abs(col_data - mu) < 3.5 * sigma]
    return col_data


def plot_kde(segments: dict, out_path: str, title_suffix: str = ""):
    """Generate a 3x2 dark-themed KDE grid matching reference style."""

    plt.rcParams.update({
        "figure.facecolor": "#000000",
        "axes.facecolor": "#000000",
        "axes.edgecolor": "#333333",
        "text.color": "#cccccc",
        "axes.labelcolor": "#aaaaaa",
        "xtick.color": "#777777",
        "ytick.color": "#777777",
        "font.family": "monospace",
        "font.size": 9,
    })

    # 3 rows x 2 cols like the reference
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.patch.set_facecolor("#000000")

    for idx, (feat, label) in enumerate(zip(FEATURES, FEATURE_LABELS)):
        row, col = idx // 2, idx % 2
        ax = axes[row][col]

        for seg_name, seg_df in segments.items():
            col_data = seg_df[feat].drop_nulls().to_numpy()
            col_data = _clean_data(col_data)
            if len(col_data) < 3:
                continue

            # Adaptive bandwidth: narrow data needs more smoothing
            data_range = np.ptp(col_data)
            data_std = np.std(col_data)
            # If data is very concentrated (cv < 0.3), smooth more
            cv = data_std / (abs(np.mean(col_data)) + 1e-12)
            bw = 1.2 if cv > 0.3 else 2.0

            sns.kdeplot(
                col_data,
                ax=ax,
                color=SEGMENT_COLORS[seg_name],
                linewidth=SEGMENT_LINEWIDTHS[seg_name],
                label=seg_name,
                fill=False,
                bw_adjust=bw,
                warn_singular=False,
            )

            # Dashed median vertical line
            median = np.median(col_data)
            ax.axvline(
                median,
                color=SEGMENT_COLORS[seg_name],
                linestyle="--",
                alpha=0.45,
                linewidth=0.9,
            )

        # Per-subplot legend (like reference)
        leg = ax.legend(
            loc="upper right",
            fontsize=7,
            frameon=True,
            framealpha=0.3,
            edgecolor="#333333",
            facecolor="#111111",
            labelcolor="#cccccc",
        )

        # Styling
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        ax.set_ylabel("Density", color="#999999", fontsize=9)
        ax.set_title(label, color="#ffffff", fontsize=11, fontweight="bold", pad=10)

    # Watermark
    fig.text(
        0.5, 0.5,
        "www.dreambound.org  x  borkiss",
        fontsize=48,
        fontweight="bold",
        fontfamily="monospace",
        color="#ffffff",
        alpha=0.06,
        ha="center",
        va="center",
        rotation=25,
        transform=fig.transFigure,
        zorder=999,
    )

    suptitle = "BTC 15-Min Microstructure KDE"
    if title_suffix:
        suptitle += f"  ·  {title_suffix}"
    fig.suptitle(suptitle, color="#ffffff", fontsize=14, fontweight="bold", y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#000000")
    plt.close(fig)
    print(f"[+] KDE plot saved to {out_path}")


# ─── Main Pipeline ──────────────────────────────────────────────────────────

def run_pipeline(skip_wait: bool = False, duration: int = 900):
    """
    Full pipeline:
    1. Discover the next BTC 15-min market
    2. Wait for the 15-min boundary
    3. Record via Rust engine
    4. Segment and plot
    """
    # Try importing the Rust module
    try:
        import polymarket_kde as engine
        use_rust = True
        print("[+] Rust PyO3 engine loaded.")
    except ImportError:
        use_rust = False
        print("[!] Rust module not found. Using Python-only market discovery.")

    # Step 1: Discover market
    market = discover_market_python()
    asset_id = market["up_token_id"]  # Record the "Up" side order book

    # Step 2: Wait for boundary
    if skip_wait:
        boundary_ts = (int(time.time()) // 900) * 900
        print(f"[*] Skipping wait, using current boundary: {boundary_ts}")
    else:
        boundary_ts = wait_for_boundary()

    # Step 3: Record data
    parquet_path = f"data_{market['slug']}.parquet"
    print(f"[*] Recording {duration}s of data for {market['slug']}...")

    if use_rust:
        result = engine.record_and_process_market(asset_id, duration, parquet_path)
        print(f"[+] Rust engine: {result}")
    else:
        print("[!] No Rust engine — run `maturin develop --release` first.")
        print(f"    Then re-run: python main.py")
        sys.exit(1)

    # Step 4: Load and segment
    print(f"[*] Loading {parquet_path}...")
    df = pl.read_parquet(parquet_path)
    print(f"[+] Loaded {len(df)} rows, columns: {df.columns}")

    segments = segment_data(df, boundary_ts)

    # Step 5: Plot
    plot_path = f"kde_{market['slug']}.png"
    plot_kde(segments, plot_path, title_suffix=market["question"])

    print(f"\n{'='*60}")
    print(f"  Pipeline complete.")
    print(f"  Data:  {parquet_path}")
    print(f"  Plot:  {plot_path}")
    print(f"{'='*60}")


def plot_existing(parquet_path: str, boundary_ts: int = 0):
    """Plot from an existing parquet file (for development/testing)."""
    df = pl.read_parquet(parquet_path)
    print(f"[+] Loaded {len(df)} rows from {parquet_path}")

    if boundary_ts == 0:
        # Infer from first timestamp
        first_ts_ms = df["timestamp_ms"][0]
        boundary_ts = (first_ts_ms // 900_000) * 900

    segments = segment_data(df, boundary_ts)
    plot_path = parquet_path.replace(".parquet", "_kde.png")
    plot_kde(segments, plot_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BTC 15-Min Microstructure KDE Pipeline")
    parser.add_argument("--skip-wait", action="store_true",
                        help="Skip waiting for 15-min boundary (start recording immediately)")
    parser.add_argument("--duration", type=int, default=900,
                        help="Recording duration in seconds (default: 900)")
    parser.add_argument("--plot-only", type=str, default=None,
                        help="Path to existing .parquet file to plot without recording")
    parser.add_argument("--boundary", type=int, default=0,
                        help="Unix timestamp of period boundary (for --plot-only)")

    args = parser.parse_args()

    if args.plot_only:
        plot_existing(args.plot_only, args.boundary)
    else:
        run_pipeline(skip_wait=args.skip_wait, duration=args.duration)
