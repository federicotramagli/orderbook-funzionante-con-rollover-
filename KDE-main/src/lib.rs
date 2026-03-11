#![allow(dead_code)]

use pyo3::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{Duration, Instant};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio_tungstenite::tungstenite::Message;

// ─── WebSocket Message Types ───────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(tag = "event_type")]
#[serde(rename_all = "snake_case")]
enum WsEvent {
    Book {
        asset_id: String,
        bids: Vec<PriceLevel>,
        asks: Vec<PriceLevel>,
        timestamp: String,
    },
    PriceChange {
        price_changes: Vec<PriceChange>,
        timestamp: String,
    },
    LastTradePrice {
        asset_id: String,
        price: String,
        size: String,
        side: String,
        timestamp: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize, Clone)]
struct PriceLevel {
    price: String,
    size: String,
}

#[derive(Debug, Deserialize)]
struct PriceChange {
    asset_id: String,
    price: String,
    size: String,
    side: String,
}

#[derive(Serialize)]
struct SubMessage {
    assets_ids: Vec<String>,
    r#type: String,
}

// ─── L2 Order Book ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct L2Book {
    bids: BTreeMap<OrderedFloat, f64>,
    asks: BTreeMap<OrderedFloat, f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedFloat(f64);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl std::hash::Hash for OrderedFloat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl L2Book {
    fn new() -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
        }
    }

    fn apply_snapshot(&mut self, bids: &[PriceLevel], asks: &[PriceLevel]) {
        self.bids.clear();
        self.asks.clear();
        for b in bids {
            let price = b.price.parse::<f64>().unwrap_or(0.0);
            let size = b.size.parse::<f64>().unwrap_or(0.0);
            if size > 0.0 {
                self.bids.insert(OrderedFloat(price), size);
            }
        }
        for a in asks {
            let price = a.price.parse::<f64>().unwrap_or(0.0);
            let size = a.size.parse::<f64>().unwrap_or(0.0);
            if size > 0.0 {
                self.asks.insert(OrderedFloat(price), size);
            }
        }
    }

    fn apply_delta(&mut self, target_asset: &str, change: &PriceChange) {
        if change.asset_id != target_asset {
            return;
        }
        let price = change.price.parse::<f64>().unwrap_or(0.0);
        let size = change.size.parse::<f64>().unwrap_or(0.0);
        let book_side = match change.side.as_str() {
            "BUY" => &mut self.bids,
            "SELL" => &mut self.asks,
            _ => return,
        };
        if size == 0.0 {
            book_side.remove(&OrderedFloat(price));
        } else {
            book_side.insert(OrderedFloat(price), size);
        }
    }

    fn best_bid(&self) -> Option<f64> {
        self.bids.keys().next_back().map(|k| k.0)
    }

    fn best_ask(&self) -> Option<f64> {
        self.asks.keys().next().map(|k| k.0)
    }

    fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some((b + a) / 2.0),
            (Some(b), None) => Some(b),
            (None, Some(a)) => Some(a),
            _ => None,
        }
    }

    /// Top-N level VWAP near the touch (bids descending, asks ascending)
    fn near_touch_vwap(&self, depth: usize) -> Option<f64> {
        let mut total_pv = 0.0;
        let mut total_v = 0.0;
        // Top N bids (highest prices = closest to mid)
        for (price, size) in self.bids.iter().rev().take(depth) {
            total_pv += price.0 * size;
            total_v += size;
        }
        // Top N asks (lowest prices = closest to mid)
        for (price, size) in self.asks.iter().take(depth) {
            total_pv += price.0 * size;
            total_v += size;
        }
        if total_v == 0.0 { None } else { Some(total_pv / total_v) }
    }
}

// ─── Feature Snapshot ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct FeatureSnapshot {
    timestamp_ms: i64,
    tick_intensity: f64,
    sample_entropy: f64,
    permutation_entropy: f64,
    contract_price_entropy: f64,
    hurst_exponent: f64,
    microprice_divergence: f64,
}

// ─── Rolling Window Math Engine ────────────────────────────────────────────
//
// Window = 500 ticks. This balances responsiveness with having enough price
// changes for statistical computation on Polymarket's sparse tick stream.

const WINDOW_SIZE: usize = 500;
const SAMPEN_MAX_WINDOW: usize = 150; // Cap for O(n^2) sample entropy

struct MathEngine {
    mid_prices: Vec<f64>,
    timestamps_ms: Vec<i64>,
    book_snapshots: Vec<L2Book>,
}

impl MathEngine {
    fn new() -> Self {
        Self {
            mid_prices: Vec::with_capacity(WINDOW_SIZE * 4),
            timestamps_ms: Vec::with_capacity(WINDOW_SIZE * 4),
            book_snapshots: Vec::with_capacity(WINDOW_SIZE * 4),
        }
    }

    fn push(&mut self, ts_ms: i64, mid: f64, book: &L2Book) {
        self.mid_prices.push(mid);
        self.timestamps_ms.push(ts_ms);
        self.book_snapshots.push(book.clone());
    }

    fn window_start(&self) -> usize {
        self.mid_prices.len().saturating_sub(WINDOW_SIZE)
    }

    fn window_prices(&self) -> &[f64] {
        &self.mid_prices[self.window_start()..]
    }

    fn window_timestamps(&self) -> &[i64] {
        &self.timestamps_ms[self.window_start()..]
    }

    /// Log returns over the window (ALL returns, including zeros)
    fn log_returns_all(&self) -> Vec<f64> {
        self.window_prices()
            .windows(2)
            .map(|w| {
                if w[0] > 0.0 && w[1] > 0.0 {
                    (w[1] / w[0]).ln()
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Non-zero log returns only (for entropy/hurst where zeros are degenerate)
    fn log_returns_nonzero(&self) -> Vec<f64> {
        self.window_prices()
            .windows(2)
            .filter_map(|w| {
                if w[0] > 0.0 && w[1] > 0.0 && (w[1] - w[0]).abs() > 1e-12 {
                    Some((w[1] / w[0]).ln())
                } else {
                    None
                }
            })
            .collect()
    }

    // ── Feature 1: Tick Intensity (updates/sec) ────────────────────────

    fn tick_intensity(&self) -> f64 {
        let ts = self.window_timestamps();
        let n = ts.len();
        if n < 2 { return 0.0; }
        let dt_secs = (ts[n - 1] - ts[0]) as f64 / 1000.0;
        if dt_secs <= 0.0 { return 0.0; }
        n as f64 / dt_secs
    }

    // ── Feature 2: Sample Entropy (m=2, r=0.2*std) ────────────────────
    // Uses ALL returns (zeros are meaningful — they indicate no price movement).
    // Subsampled to SAMPEN_MAX_WINDOW for O(n^2) feasibility.

    fn sample_entropy(&self) -> f64 {
        let all_returns = self.log_returns_all();
        let n_full = all_returns.len();
        if n_full < 10 { return 0.0; }

        // Subsample: take evenly-spaced elements if too large
        let returns: Vec<f64> = if n_full > SAMPEN_MAX_WINDOW {
            let step = n_full as f64 / SAMPEN_MAX_WINDOW as f64;
            (0..SAMPEN_MAX_WINDOW)
                .map(|i| all_returns[(i as f64 * step) as usize])
                .collect()
        } else {
            all_returns
        };
        let n = returns.len();

        let mean = returns.iter().sum::<f64>() / n as f64;
        let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64;
        let std = var.sqrt();
        if std < 1e-15 { return 0.0; }

        let r = 0.2 * std;
        let m = 2usize;

        let count_matches = |length: usize| -> usize {
            let max_i = n.saturating_sub(length);
            let mut count = 0usize;
            for i in 0..max_i {
                for j in (i + 1)..max_i {
                    let matched = (0..length).all(|k| (returns[i + k] - returns[j + k]).abs() <= r);
                    if matched { count += 1; }
                }
            }
            count
        };

        let b = count_matches(m) as f64;
        let a = count_matches(m + 1) as f64;

        if b == 0.0 || a == 0.0 { return 0.0; }
        -(a / b).ln()
    }

    // ── Feature 3: Permutation Entropy (order=3, delay=5, raw prices) ──
    // Delay=5 skips over runs of identical prices to capture real patterns.

    fn permutation_entropy(&self) -> f64 {
        let prices = self.window_prices();
        let order = 3usize;
        let delay = 5usize;
        let n = prices.len();
        let needed = (order - 1) * delay + 1;
        if n < needed { return 0.0; }

        let mut pattern_counts: std::collections::HashMap<Vec<u8>, usize> =
            std::collections::HashMap::new();
        let total = n - needed + 1;

        for i in 0..total {
            let mut embedding: Vec<(f64, usize)> = (0..order)
                .map(|k| (prices[i + k * delay], k))
                .collect();
            embedding.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1)));
            let mut pattern = vec![0u8; order];
            for (rank, &(_, orig_idx)) in embedding.iter().enumerate() {
                pattern[orig_idx] = rank as u8;
            }
            *pattern_counts.entry(pattern).or_insert(0) += 1;
        }

        let total_f = total as f64;
        let mut entropy = 0.0;
        for count in pattern_counts.values() {
            let p = *count as f64 / total_f;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        let max_entropy = (1..=order).map(|i| i as f64).product::<f64>().ln();
        if max_entropy > 0.0 { entropy / max_entropy } else { 0.0 }
    }

    // ── Feature 4: Contract Price Entropy (Shannon of L2 volume dist) ──

    fn contract_price_entropy(&self) -> f64 {
        if self.book_snapshots.is_empty() { return 0.0; }
        let book = self.book_snapshots.last().unwrap();

        let mut volumes: Vec<f64> = Vec::new();
        for size in book.bids.values() { volumes.push(*size); }
        for size in book.asks.values() { volumes.push(*size); }

        if volumes.is_empty() { return 0.0; }
        let total: f64 = volumes.iter().sum();
        if total == 0.0 { return 0.0; }

        let mut entropy = 0.0;
        for v in &volumes {
            let p = v / total;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    // ── Feature 5: Hurst Exponent (R/S on all returns, small chunks) ──
    // Uses ALL returns with small chunk sizes. Chunks with zero std are
    // skipped (they represent periods of no price movement).

    fn hurst_exponent(&self) -> f64 {
        let returns = self.log_returns_all();
        let n = returns.len();
        if n < 16 { return 0.5; }

        let mut rs_values: Vec<(f64, f64)> = Vec::new();

        for &chunk_size in &[4, 8, 16, 32, 64, 128] {
            if chunk_size > n { break; }
            let num_chunks = n / chunk_size;
            if num_chunks == 0 { continue; }

            let mut rs_sum = 0.0;
            let mut valid_chunks = 0;

            for c in 0..num_chunks {
                let start = c * chunk_size;
                let chunk = &returns[start..start + chunk_size];

                let mean = chunk.iter().sum::<f64>() / chunk_size as f64;
                let std = (chunk.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / chunk_size as f64).sqrt();
                if std < 1e-15 { continue; }

                let mut cum_dev = Vec::with_capacity(chunk_size);
                let mut running = 0.0;
                for x in chunk {
                    running += x - mean;
                    cum_dev.push(running);
                }

                let range = cum_dev.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                    - cum_dev.iter().cloned().fold(f64::INFINITY, f64::min);
                rs_sum += range / std;
                valid_chunks += 1;
            }

            if valid_chunks > 0 {
                let rs_avg = rs_sum / valid_chunks as f64;
                if rs_avg > 0.0 {
                    rs_values.push(((chunk_size as f64).ln(), rs_avg.ln()));
                }
            }
        }

        if rs_values.len() < 2 { return 0.5; }

        let n_pts = rs_values.len() as f64;
        let sum_x: f64 = rs_values.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = rs_values.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = rs_values.iter().map(|(x, y)| x * y).sum();
        let sum_xx: f64 = rs_values.iter().map(|(x, _)| x * x).sum();

        let denom = n_pts * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-12 { return 0.5; }

        let h = (n_pts * sum_xy - sum_x * sum_y) / denom;
        h.clamp(0.0, 1.0)
    }

    // ── Feature 6: Microprice Divergence (full-depth VWAP - mid) ───────

    fn microprice_divergence(&self) -> f64 {
        if self.book_snapshots.is_empty() { return 0.0; }
        let book = self.book_snapshots.last().unwrap();
        let mid = match book.mid_price() {
            Some(m) => m,
            None => return 0.0,
        };
        // Top-10 levels near the touch, not the entire 0-1 range
        let vwap = match book.near_touch_vwap(10) {
            Some(v) => v,
            None => return 0.0,
        };
        vwap - mid
    }

    // ── Compute all features ───────────────────────────────────────────

    fn compute_snapshot(&self, ts_ms: i64) -> FeatureSnapshot {
        FeatureSnapshot {
            timestamp_ms: ts_ms,
            tick_intensity: self.tick_intensity(),
            sample_entropy: self.sample_entropy(),
            permutation_entropy: self.permutation_entropy(),
            contract_price_entropy: self.contract_price_entropy(),
            hurst_exponent: self.hurst_exponent(),
            microprice_divergence: self.microprice_divergence(),
        }
    }
}

// ─── WebSocket Client & Recording ──────────────────────────────────────────

async fn run_recorder(
    asset_id: String,
    duration_secs: u64,
    results: Arc<Mutex<Vec<FeatureSnapshot>>>,
) -> Result<(), String> {
    let ws_url = "wss://ws-subscriptions-clob.polymarket.com/ws/market";

    tracing::info!("Connecting to Polymarket WebSocket...");
    let (ws_stream, _) = tokio_tungstenite::connect_async(ws_url)
        .await
        .map_err(|e| format!("WebSocket connect failed: {e}"))?;

    let (mut write, mut read) = ws_stream.split();

    // Subscribe to asset
    let sub = serde_json::json!({
        "assets_ids": [&asset_id],
        "type": "market"
    });
    write
        .send(Message::Text(sub.to_string().into()))
        .await
        .map_err(|e| format!("Subscribe send failed: {e}"))?;

    tracing::info!("Subscribed to asset {}", &asset_id[..20.min(asset_id.len())]);

    let book = Arc::new(Mutex::new(L2Book::new()));
    let engine = Arc::new(Mutex::new(MathEngine::new()));
    let start = Instant::now();
    let deadline = Duration::from_secs(duration_secs);
    let mut tick_count: u64 = 0;
    let mut last_emit_ms: i64 = 0;
    let mut last_mid: f64 = 0.0;
    const EMIT_INTERVAL_MS: i64 = 200; // Throttle: max 1 snapshot per 200ms

    // Ping task
    let write_arc = Arc::new(Mutex::new(write));
    let ping_writer = write_arc.clone();
    let ping_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(10));
        loop {
            interval.tick().await;
            let mut w = ping_writer.lock().await;
            if w.send(Message::Text("PING".into())).await.is_err() {
                break;
            }
        }
    });

    // Main message loop — emit per-tick, no timer
    let recv_asset = asset_id.clone();

    loop {
        if start.elapsed() >= deadline {
            tracing::info!("Recording duration reached. Total ticks: {tick_count}");
            break;
        }

        let msg = tokio::select! {
            msg = read.next() => msg,
            _ = tokio::time::sleep(deadline.saturating_sub(start.elapsed())) => {
                tracing::info!("Deadline timeout. Total ticks: {tick_count}");
                break;
            }
        };

        let msg = match msg {
            Some(Ok(m)) => m,
            Some(Err(e)) => {
                tracing::warn!("WebSocket error: {e}");
                break;
            }
            None => {
                tracing::warn!("WebSocket stream ended.");
                break;
            }
        };

        let text = match msg {
            Message::Text(t) => t.to_string(),
            Message::Ping(d) => {
                let mut w = write_arc.lock().await;
                let _ = w.send(Message::Pong(d)).await;
                continue;
            }
            Message::Close(_) => {
                tracing::warn!("WebSocket closed by server.");
                break;
            }
            _ => continue,
        };

        if text == "PONG" || text == "pong" {
            continue;
        }

        let event: WsEvent = match serde_json::from_str(&text) {
            Ok(e) => e,
            Err(_) => continue,
        };

        // Process event and emit feature snapshot per-tick
        let mut book_changed = false;

        match event {
            WsEvent::Book { asset_id: ref aid, bids, asks, .. } => {
                if aid == &recv_asset {
                    let mut b = book.lock().await;
                    b.apply_snapshot(&bids, &asks);
                    book_changed = true;
                    tracing::info!("Book snapshot: {} bids, {} asks", b.bids.len(), b.asks.len());
                }
            }
            WsEvent::PriceChange { price_changes, .. } => {
                let mut b = book.lock().await;
                let mut relevant = false;
                for pc in &price_changes {
                    if pc.asset_id == recv_asset {
                        relevant = true;
                    }
                    b.apply_delta(&recv_asset, pc);
                }
                if relevant {
                    book_changed = true;
                }
            }
            WsEvent::LastTradePrice { ref asset_id, .. } => {
                if asset_id == &recv_asset {
                    book_changed = true;
                }
            }
            WsEvent::Other => {}
        }

        if book_changed {
            let b = book.lock().await;
            if let Some(mid) = b.mid_price() {
                let now_ms = chrono::Utc::now().timestamp_millis();
                let mut eng = engine.lock().await;

                // Always push to engine (full tick-level granularity for calculations)
                eng.push(now_ms, mid, &b);
                tick_count += 1;

                // Throttled snapshot emission: only when mid changed OR enough time passed
                let mid_changed = (mid - last_mid).abs() > 1e-12;
                let time_elapsed = now_ms - last_emit_ms >= EMIT_INTERVAL_MS;

                if eng.mid_prices.len() >= 20 && (mid_changed || time_elapsed) {
                    let snap = eng.compute_snapshot(now_ms);
                    results.lock().await.push(snap);
                    last_emit_ms = now_ms;
                    last_mid = mid;
                }
            }
        }
    }

    ping_handle.abort();

    let count = results.lock().await.len();
    tracing::info!("Recording complete: {count} feature snapshots from {tick_count} ticks");
    Ok(())
}

/// Write collected feature snapshots to parquet
fn write_parquet(snapshots: &[FeatureSnapshot], path: &str) -> Result<(), String> {
    use polars::prelude::*;

    if snapshots.is_empty() {
        return Err("No data collected".to_string());
    }

    let timestamps: Vec<i64> = snapshots.iter().map(|s| s.timestamp_ms).collect();
    let tick_int: Vec<f64> = snapshots.iter().map(|s| s.tick_intensity).collect();
    let samp_ent: Vec<f64> = snapshots.iter().map(|s| s.sample_entropy).collect();
    let perm_ent: Vec<f64> = snapshots.iter().map(|s| s.permutation_entropy).collect();
    let cont_ent: Vec<f64> = snapshots.iter().map(|s| s.contract_price_entropy).collect();
    let hurst: Vec<f64> = snapshots.iter().map(|s| s.hurst_exponent).collect();
    let micro_div: Vec<f64> = snapshots.iter().map(|s| s.microprice_divergence).collect();

    let mut df = DataFrame::new(vec![
        Column::new("timestamp_ms".into(), &timestamps),
        Column::new("tick_intensity".into(), &tick_int),
        Column::new("sample_entropy".into(), &samp_ent),
        Column::new("permutation_entropy".into(), &perm_ent),
        Column::new("contract_price_entropy".into(), &cont_ent),
        Column::new("hurst_exponent".into(), &hurst),
        Column::new("microprice_divergence".into(), &micro_div),
    ]).map_err(|e| format!("DataFrame creation failed: {e}"))?;

    let file = std::fs::File::create(path)
        .map_err(|e| format!("File create failed: {e}"))?;
    ParquetWriter::new(file)
        .finish(&mut df)
        .map_err(|e| format!("Parquet write failed: {e}"))?;

    tracing::info!("Wrote {} rows to {}", snapshots.len(), path);
    Ok(())
}

// ─── PyO3 Module ───────────────────────────────────────────────────────────

#[pyfunction]
fn record_and_process_market(
    py: Python<'_>,
    asset_id: &str,
    duration_secs: u64,
    out_path: &str,
) -> PyResult<String> {
    let asset = asset_id.to_string();
    let path = out_path.to_string();

    py.allow_threads(move || {
        tracing_subscriber::fmt()
            .with_target(false)
            .with_env_filter("polymarket_kde=info")
            .try_init()
            .ok();

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Tokio init: {e}")))?;

        let results: Arc<Mutex<Vec<FeatureSnapshot>>> = Arc::new(Mutex::new(Vec::new()));

        rt.block_on(async {
            let res = run_recorder(asset.clone(), duration_secs, results.clone()).await;
            if let Err(e) = &res {
                tracing::error!("Recorder error: {e}");
            }
            res
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.clone()))?;

        let snapshots = rt.block_on(async { results.lock().await.clone() });
        let count = snapshots.len();

        write_parquet(&snapshots, &path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        Ok(format!("Recorded {count} snapshots to {path}"))
    })
}

#[pyfunction]
fn discover_active_market(py: Python<'_>) -> PyResult<(String, String, String, String)> {
    py.allow_threads(move || {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

        rt.block_on(async {
            let now = chrono::Utc::now().timestamp();
            let next_slot = ((now / 900) + 1) * 900;
            let slug = format!("btc-updown-15m-{next_slot}");

            let url = format!(
                "https://gamma-api.polymarket.com/events?slug={}",
                slug
            );

            let resp = reqwest::get(&url).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("HTTP request failed: {e}"))
            })?;

            let body: serde_json::Value = resp.json().await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("JSON parse failed: {e}"))
            })?;

            let events = body.as_array().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Expected array response")
            })?;

            if events.is_empty() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("No market found for slug: {slug}")
                ));
            }

            let event = &events[0];
            let markets = event["markets"].as_array().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("No markets array")
            })?;
            let market = &markets[0];
            let condition_id = market["conditionId"]
                .as_str()
                .unwrap_or("")
                .to_string();

            let tokens_str = market["clobTokenIds"]
                .as_str()
                .unwrap_or("[]");
            let tokens: Vec<String> = serde_json::from_str(tokens_str).unwrap_or_default();

            if tokens.len() < 2 {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Not enough token IDs"
                ));
            }

            Ok((slug, condition_id, tokens[0].clone(), tokens[1].clone()))
        })
    })
}

#[pymodule]
fn polymarket_kde(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(record_and_process_market, m)?)?;
    m.add_function(wrap_pyfunction!(discover_active_market, m)?)?;
    Ok(())
}
