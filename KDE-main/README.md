# BTC 15-Min Microstructure KDE Pipeline

Real-time microstructure analysis of Polymarket BTC 15-minute prediction markets. Records L2 order book data via WebSocket, computes rolling statistical features in Rust, and visualizes their distributions as dark-themed KDE plots segmented by time phase.

![KDE Example](kde_btc-updown-15m-1771688700.png)

## Architecture

**Rust core** (`src/lib.rs`) — PyO3 native module:
- Connects to Polymarket CLOB WebSocket, maintains a live L2 order book
- Computes 6 rolling features over a 500-tick window per book update
- Writes feature snapshots to Parquet (throttled to 1 per 200ms)

**Python orchestrator** (`main.py`):
- Discovers the next active BTC 15-min market via Gamma API
- Waits for the 15-minute clock boundary, then triggers Rust recording
- Segments data into 3 time phases and generates KDE plots

## Features Computed

| Feature | Description |
|---|---|
| **Tick Intensity** | Order book updates per second (activity proxy) |
| **Sample Entropy** | Predictability of mid-price returns (m=2, r=0.2σ) |
| **Permutation Entropy** | Ordinal pattern complexity of raw prices (order=3, delay=5) |
| **Contract Price Entropy** | Shannon entropy of L2 volume distribution |
| **Hurst Exponent** | Trend persistence via R/S analysis (>0.5 = trending) |
| **Microprice Divergence** | Top-10 level VWAP minus mid-price (directional pressure) |

## Time Segments

Each 15-minute window is split into three phases for comparative analysis:

- **First 12 Min** (cyan) — baseline regime
- **Last 3 Min excl 30s** (magenta) — transition zone
- **Last 30 Sec** (yellow) — settlement convergence

## Setup

### Prerequisites

- Python >= 3.9
- Rust toolchain (for building the PyO3 module)
- [maturin](https://github.com/PyO3/maturin) >= 1.0

### Install

```bash
# Create venv and install Python deps
python -m venv .venv
source .venv/bin/activate
pip install polars seaborn matplotlib numpy requests maturin

# Build the Rust extension
maturin develop --release
```

## Usage

### Live recording (full pipeline)

```bash
# Wait for the next 15-min boundary, record 900s, then plot
python main.py

# Skip the wait (start immediately from current boundary)
python main.py --skip-wait

# Custom recording duration
python main.py --duration 600
```

### Plot from existing data

```bash
python main.py --plot-only data_btc-updown-15m-1771688700.parquet
python main.py --plot-only data.parquet --boundary 1771688700
```

## Tech Stack

- **Rust** — async WebSocket client (tokio + tungstenite), order book engine, feature math, Parquet I/O (polars)
- **PyO3 / maturin** — Rust → Python bridge
- **Python** — market discovery (requests), data segmentation (polars), KDE visualization (seaborn + matplotlib)

## License

MIT
