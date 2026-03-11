#!/usr/bin/env python3
"""
analyze_delta.py — Delta Mid Analysis for Polymarket MM Bot
============================================================
Reads bot.log and extracts every (BS-Fair, Market Mid) pair from Strategy lines.

Usage:
    python analyze_delta.py              # reads bot.log in same directory
    python analyze_delta.py path/to/bot.log
"""

import re
import sys
import os
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot.log")
EDGE_THRESHOLDS = [0.05, 0.08, 0.10, 0.13, 0.15]

# Regex:  [MKT: 0.17/0.19]  and  [BS-Fair: 0.1800 (source)]
_RE_MKT    = re.compile(r'\[MKT:\s*([\d.]+)/([\d.]+)\]')
_RE_BSFAIR = re.compile(r'\[BS-Fair:\s*([\d.]+)')

# ── Parse ─────────────────────────────────────────────────────────────────────

def parse_log(path: str) -> list[dict]:
    samples = []
    skipped = 0
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if "Strategy" not in line:
                continue
            m_mkt    = _RE_MKT.search(line)
            m_bsfair = _RE_BSFAIR.search(line)
            if not m_mkt or not m_bsfair:
                skipped += 1
                continue
            bid  = float(m_mkt.group(1))
            ask  = float(m_mkt.group(2))
            fair = float(m_bsfair.group(1))
            mid  = (bid + ask) / 2.0
            delta = abs(fair - mid)
            samples.append({
                "fair":  fair,
                "bid":   bid,
                "ask":   ask,
                "mid":   mid,
                "delta": delta,
                "side":  "YES" if fair > mid else "NO" if fair < mid else "FLAT",
            })
    return samples

# ── Stats helpers ─────────────────────────────────────────────────────────────

def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    data = sorted(data)
    idx  = (len(data) - 1) * p / 100.0
    lo   = int(idx)
    hi   = lo + 1
    if hi >= len(data):
        return data[lo]
    frac = idx - lo
    return data[lo] + frac * (data[hi] - data[lo])

def mean(data: list[float]) -> float:
    return sum(data) / len(data) if data else 0.0

def stdev(data: list[float]) -> float:
    if len(data) < 2:
        return 0.0
    m = mean(data)
    return (sum((x - m) ** 2 for x in data) / (len(data) - 1)) ** 0.5

# ── Report ────────────────────────────────────────────────────────────────────

def print_report(samples: list[dict]) -> None:
    n = len(samples)
    if n == 0:
        print("❌ Nessun campione trovato. Verifica il percorso del log.")
        return

    deltas = [s["delta"] for s in samples]
    sides  = Counter(s["side"] for s in samples)

    p25  = percentile(deltas, 25)
    p50  = percentile(deltas, 50)
    p75  = percentile(deltas, 75)
    p90  = percentile(deltas, 90)
    p95  = percentile(deltas, 95)
    avg  = mean(deltas)
    std  = stdev(deltas)
    mn   = min(deltas)
    mx   = max(deltas)

    sep  = "─" * 56

    print(f"\n{'═'*56}")
    print(f"  DELTA MID ANALYSIS — {n:,} campioni")
    print(f"{'═'*56}")

    print(f"\n📊 STATISTICHE DESCRITTIVE")
    print(sep)
    print(f"  Campioni totali      : {n:>10,}")
    print(f"  Delta Medio (mean)   : {avg:>10.4f}")
    print(f"  Delta Mediano (p50)  : {p50:>10.4f}")
    print(f"  Std Dev              : {std:>10.4f}")
    print(f"  Min / Max            : {mn:>10.4f} / {mx:.4f}")

    print(f"\n📈 DISTRIBUZIONE PERCENTILE")
    print(sep)
    print(f"  P25  (25° percentile): {p25:>10.4f}")
    print(f"  P50  (50° percentile): {p50:>10.4f}  ← mediana")
    print(f"  P75  (75° percentile): {p75:>10.4f}")
    print(f"  P90  (90° percentile): {p90:>10.4f}")
    print(f"  P95  (95° percentile): {p95:>10.4f}")

    print(f"\n⚡ FREQUENZA GAP PER SOGLIA")
    print(sep)
    for thr in EDGE_THRESHOLDS:
        cnt  = sum(1 for d in deltas if d >= thr)
        pct  = cnt / n * 100
        bar  = "█" * int(pct / 2)
        print(f"  Delta ≥ {thr:.2f}  : {cnt:>7,}  ({pct:5.1f}%)  {bar}")

    print(f"\n📐 DIREZIONE SEGNALE")
    print(sep)
    for side, cnt in sides.most_common():
        print(f"  {side:<6}: {cnt:>7,}  ({cnt/n*100:.1f}%)")

    # ── Interpretazione bracket ────────────────────────────────────────────────
    print(f"\n🎯 INTERPRETAZIONE BRACKET DI SIZING")
    print(sep)

    freq_008 = sum(1 for d in deltas if d >= 0.08) / n * 100
    freq_010 = sum(1 for d in deltas if d >= 0.10) / n * 100
    freq_013 = sum(1 for d in deltas if d >= 0.13) / n * 100

    if p75 < 0.08:
        print(f"  ⚠️  P75={p75:.4f} < 0.08 → MERCATO TROPPO EFFICIENTE")
        print(f"      Solo il {freq_008:.1f}% dei tick raggiunge edge ≥ 0.08.")
        print(f"      Suggerimento: abbassa MIN_EDGE a {p50:.2f} (mediana)")
        print(f"      oppure usa sizing dinamico:")
        print(f"        edge ≥ {p50:.2f} → $25  (half-size)")
        print(f"        edge ≥ {p75:.2f} → $50  (full-size)")
        print(f"        edge ≥ {p90:.2f} → $75  (oversize)")
    elif p75 < 0.10:
        print(f"  ✅  P75={p75:.4f} — bracket 0.08 allineato, 0.10 è il limite superiore.")
        print(f"      Frequenze: ≥0.08={freq_008:.1f}%  ≥0.10={freq_010:.1f}%  ≥0.13={freq_013:.1f}%")
        print(f"      Sizing consigliato:")
        print(f"        edge 0.08–0.10 → $50 (standard)")
        print(f"        edge ≥ 0.10    → $75 (boost)")
    else:
        print(f"  🚀  P75={p75:.4f} ≥ 0.10 — buona inefficienza strutturale.")
        print(f"      Tutti e tre i bracket (0.08/0.10/0.13) sono nel range operativo.")
        print(f"      Frequenze: ≥0.08={freq_008:.1f}%  ≥0.10={freq_010:.1f}%  ≥0.13={freq_013:.1f}%")
        print(f"      Sizing dinamico pieno:")
        print(f"        edge 0.08–0.10 → $50")
        print(f"        edge 0.10–0.13 → $75")
        print(f"        edge ≥ 0.13    → $100")

    print(f"\n{'═'*56}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else LOG_PATH
    if not os.path.isfile(path):
        print(f"❌ File non trovato: {path}")
        sys.exit(1)
    print(f"📂 Lettura log: {path}")
    samples = parse_log(path)
    print_report(samples)
