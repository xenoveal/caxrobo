"""
Build the standalone interactive review chart from exported bar annotations.

Packs every 15m bar (OHLCV + regime + live-candidate count) into compressed
typed-array blobs, embeds them with the sparse trigger/trade lists in a
single self-contained HTML file, and writes it out. Nothing is fetched at
runtime, so the file works from the filesystem with no server.

Per-bar numeric series are delta-encoded, zlib-compressed and base64'd; the
page inflates them with DecompressionStream. Timestamps are not stored: the
15m grid is regular, so bar i is t0 + i * step.

Usage:
    python scripts/export_bar_annotations.py --out annotations.json
    python scripts/build_review_chart.py --annotations annotations.json \
        --out .claude/PRPs/reports/phase5-bar-by-bar-review.chart.html
"""

import argparse
import base64
import json
import logging
import sys
import zlib
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger("build_review_chart")

PRICE_SCALE = 100  # prices stored as integer hundredths
STEP_S = 900  # 15m grid, seconds
PRICE_KEYS = ("level", "entry", "stop", "target", "exit_price")


def _pack_symbol(s: dict) -> tuple[str, dict]:
    """Delta-encode and compress one symbol's per-bar series.

    Returns:
        Tuple of (base64 zlib blob, metadata describing the bar grid).
    """
    ts = np.asarray(s["ts"], dtype=np.int64)
    n = len(ts)
    steps = np.unique(np.diff(ts)) if n > 1 else np.array([STEP_S])
    if len(steps) != 1 or int(steps[0]) != STEP_S:
        raise ValueError(f"irregular 15m grid: steps={steps[:5]}")

    chunks: list[bytes] = []
    for key in "ohlc":
        q = np.round(np.asarray(s[key], dtype=np.float64) * PRICE_SCALE).astype(np.int64)
        # prepend=0 so the first delta IS the first price; the page cumsums.
        chunks.append(np.diff(q, prepend=0).astype("<i4").tobytes())

    # Volume stays full float32: it is cross-checked against the store, so the
    # bytes zlib would save by truncating the mantissa are not worth the drift.
    chunks.append(np.asarray(s["v"], dtype="<f4").tobytes())

    chunks.append(np.asarray(s["regime"], dtype=np.uint8).tobytes())
    chunks.append(np.asarray(s["ncand"], dtype=np.uint8).tobytes())

    blob = zlib.compress(b"".join(chunks), 9)
    meta = {"n": n, "t0": int(ts[0]), "step": STEP_S, "scale": PRICE_SCALE}
    return base64.b64encode(blob).decode("ascii"), meta


def _shrink(rows: list[dict]) -> list[dict]:
    """Trim the sparse lists: drop nulls, round prices and fractions."""
    out = []
    for r in rows:
        e = {}
        for k, v in r.items():
            if v is None:
                continue
            if isinstance(v, float):
                v = round(v, 2) if k in PRICE_KEYS else round(v, 6)
            e[k] = v
        out.append(e)
    return out


def build(annotations: dict) -> str:
    """Render the full HTML document for an annotations payload."""
    symbols: dict[str, dict] = {}
    for sym, s in annotations["symbols"].items():
        blob, meta = _pack_symbol(s)
        symbols[sym] = {
            **meta,
            "blob": blob,
            "events": _shrink(s["events"]),
            "trades": _shrink(s["trades"]),
        }
        logger.info("%s: packed %d bars, blob %.2f MB", sym, meta["n"], len(blob) / 1e6)

    payload = {
        "generated_ms": annotations["generated_ms"],
        "trigger_timeframe": annotations["trigger_timeframe"],
        "pattern_timeframe": annotations["pattern_timeframe"],
        "regime_timeframe": annotations["regime_timeframe"],
        "params": annotations["params"],
        "symbols": symbols,
    }
    gen = datetime.fromtimestamp(
        annotations["generated_ms"] / 1000, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")
    html = TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    return html.replace("__GENERATED__", gen)


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phase 5 — Bar-by-bar signal review (15m)</title>
<style>
  :root {
    --bg: #0e1117; --panel: #161b22; --line: #262d38; --fg: #e6edf3;
    --dim: #8b949e; --up: #26a69a; --down: #ef5350;
    --taken: #ffd166; --rejected: #7aa2f7; --busy: #b980f0; --superseded: #f78c6c;
    --trend: rgba(38,166,154,.09); --range: rgba(122,162,247,.09);
    --extreme: rgba(239,83,80,.10); --uncertain: rgba(139,148,158,.05);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 13px/1.45 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  header { padding: 14px 18px 10px; border-bottom: 1px solid var(--line); }
  h1 { margin: 0 0 4px; font-size: 16px; font-weight: 650; letter-spacing: -.01em; }
  .sub { color: var(--dim); font-size: 12px; }
  .sub code { color: var(--fg); background: #1f2530; padding: 1px 5px; border-radius: 4px; }
  .wrap { padding: 12px 18px 28px; }
  .bar {
    display: flex; flex-wrap: wrap; gap: 14px; align-items: flex-end;
    padding: 10px 12px; background: var(--panel); border: 1px solid var(--line);
    border-radius: 8px; margin-bottom: 10px;
  }
  .fld { display: flex; flex-direction: column; gap: 4px; }
  .fld label { font-size: 10.5px; text-transform: uppercase; letter-spacing: .07em; color: var(--dim); }
  select, input, button {
    background: #1c2230; color: var(--fg); border: 1px solid var(--line);
    border-radius: 6px; padding: 5px 8px; font: inherit; font-size: 12px;
  }
  button { cursor: pointer; }
  button:hover { border-color: #3d4757; background: #232b3a; }
  .presets { display: flex; gap: 5px; }
  .presets button { padding: 5px 9px; }
  .toggles { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip {
    display: inline-flex; align-items: center; gap: 6px; padding: 4px 9px;
    border: 1px solid var(--line); border-radius: 999px; cursor: pointer;
    user-select: none; font-size: 12px; background: #1c2230;
  }
  .chip.off { opacity: .38; }
  .dot { width: 9px; height: 9px; border-radius: 50%; }
  .kpis { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }
  .kpi {
    background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
    padding: 8px 12px; min-width: 104px;
  }
  .kpi .k { font-size: 10.5px; text-transform: uppercase; letter-spacing: .07em; color: var(--dim); }
  .kpi .v { font-size: 17px; font-weight: 650; font-variant-numeric: tabular-nums; }
  .kpi .n { font-size: 10.5px; color: var(--dim); }
  .chartbox {
    position: relative; background: var(--panel); border: 1px solid var(--line);
    border-radius: 8px; overflow: hidden;
  }
  canvas { display: block; width: 100%; }
  #tip {
    position: absolute; pointer-events: none; z-index: 5; display: none;
    background: rgba(13,17,23,.97); border: 1px solid #3d4757; border-radius: 7px;
    padding: 8px 10px; font-size: 11.5px; max-width: 340px; line-height: 1.5;
    box-shadow: 0 8px 24px rgba(0,0,0,.5);
  }
  #tip b { font-weight: 650; }
  #tip .row { display: flex; justify-content: space-between; gap: 14px; }
  #tip .row span:first-child { color: var(--dim); }
  #tip hr { border: 0; border-top: 1px solid var(--line); margin: 6px 0; }
  .hint { color: var(--dim); font-size: 11.5px; margin: 8px 0 14px; }
  h2 { font-size: 13.5px; margin: 18px 0 8px; font-weight: 650; }
  .tablebox { max-height: 430px; overflow: auto; border: 1px solid var(--line); border-radius: 8px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { padding: 6px 9px; text-align: right; white-space: nowrap; }
  th {
    position: sticky; top: 0; background: #1c2230; text-align: right;
    font-weight: 600; font-size: 10.5px; text-transform: uppercase;
    letter-spacing: .06em; color: var(--dim); border-bottom: 1px solid var(--line);
    cursor: pointer;
  }
  th.l, td.l { text-align: left; }
  tbody tr { border-bottom: 1px solid #1d2430; cursor: pointer; }
  tbody tr:hover { background: #1b2231; }
  .tag { display: inline-block; padding: 1px 7px; border-radius: 999px; font-size: 10.5px; font-weight: 600; }
  .pos { color: var(--up); } .neg { color: var(--down); }
  .loading { padding: 60px; text-align: center; color: var(--dim); }
</style>
</head>
<body>
<header>
  <h1>Phase 5 — bar-by-bar signal review</h1>
  <div class="sub">
    Every <code>15m</code> trigger bar in the store, annotated with what the strategy saw:
    the <code>4h</code> regime, live <code>1h</code> pattern candidates, and every breakout
    trigger — taken, rejected, or skipped. Generated __GENERATED__.
  </div>
</header>

<div class="wrap">
<div id="app" class="loading">Inflating bar data…</div>

<template id="tpl">
  <div class="bar">
    <div class="fld">
      <label for="sym">Symbol</label>
      <select id="sym"></select>
    </div>
    <div class="fld">
      <label for="from">From (UTC)</label>
      <input type="datetime-local" id="from" step="900">
    </div>
    <div class="fld">
      <label for="to">To (UTC)</label>
      <input type="datetime-local" id="to" step="900">
    </div>
    <div class="fld">
      <label>Range</label>
      <div class="presets">
        <button data-days="2">2d</button>
        <button data-days="7">1w</button>
        <button data-days="30">1m</button>
        <button data-days="90">3m</button>
        <button data-days="365">1y</button>
        <button data-days="0">All</button>
      </div>
    </div>
    <div class="fld">
      <label>Jump</label>
      <div class="presets">
        <button id="prevEv">‹ prev trigger</button>
        <button id="nextEv">next trigger ›</button>
      </div>
    </div>
  </div>

  <div class="bar">
    <div class="fld" style="flex:1">
      <label>Show markers</label>
      <div class="toggles" id="layers"></div>
    </div>
    <div class="fld">
      <label>Overlays</label>
      <div class="toggles">
        <span class="chip" data-ov="regime">Regime shading</span>
        <span class="chip" data-ov="cand">Candidate ribbon</span>
        <span class="chip" data-ov="levels">SL/TP of trades</span>
        <span class="chip" data-ov="hold">Holding periods</span>
      </div>
    </div>
  </div>

  <div class="kpis" id="kpis"></div>

  <div class="chartbox">
    <canvas id="cv"></canvas>
    <div id="tip"></div>
  </div>
  <div class="hint">
    Scroll to zoom at the cursor · drag the chart to pan · drag across the bottom
    overview to select a window · double-click the overview to reset · arrow keys pan,
    <b>+</b>/<b>−</b> zoom. Hollow markers are triggers that fired but produced no
    position; filled markers are executed entries.
  </div>

  <h2>Triggers in the visible window <span class="sub" id="evcount"></span></h2>
  <div class="tablebox">
    <table id="evtable">
      <thead><tr>
        <th class="l" data-s="time">Time (UTC)</th>
        <th class="l" data-s="type">Verdict</th>
        <th class="l" data-s="pattern">Pattern</th>
        <th class="l" data-s="direction">Side</th>
        <th data-s="entry">Entry</th>
        <th data-s="level">Level</th>
        <th data-s="stop">Stop</th>
        <th data-s="target">Target</th>
        <th data-s="risk_pct">Risk %</th>
        <th data-s="reward_pct">Reward %</th>
        <th data-s="rr">R:R</th>
        <th data-s="volume_ratio">Vol ×</th>
        <th class="l" data-s="reason">Remark</th>
        <th data-s="pnl_pct">Net P&amp;L %</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
</template>
</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
"use strict";

const DATA = JSON.parse(document.getElementById("payload").textContent);
const REGIMES = ["uncertain", "trending", "ranging", "extreme-volatility"];

function getVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const REGIME_FILL = {
  0: getVar("--uncertain"), 1: getVar("--trend"),
  2: getVar("--range"), 3: getVar("--extreme"),
};
const TYPES = {
  taken:      { label: "Position taken", color: getVar("--taken") },
  rejected:   { label: "Rejected",       color: getVar("--rejected") },
  busy:       { label: "Skipped (busy)", color: getVar("--busy") },
  superseded: { label: "Superseded",     color: getVar("--superseded") },
};
const REASON_TEXT = {
  "rr-below-floor": "reward:risk under RR_FLOOR",
  "reward-below-cost": "reward under round-trip cost — no winning outcome",
  "geometry": "unsizeable stop or entry past target",
  "trade-already-open": "another position was still open",
  "another-candidate-filled-this-bar": "an earlier candidate filled this bar",
};

/* ---------- payload inflation ---------- */

async function inflate(b64) {
  const bin = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const stream = new Blob([bin]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function cumsum(deltas, scale) {
  const out = new Float64Array(deltas.length);
  let acc = 0;
  for (let i = 0; i < deltas.length; i++) { acc += deltas[i]; out[i] = acc / scale; }
  return out;
}

async function unpack(sym) {
  const s = DATA.symbols[sym];
  if (s.o) return s;
  const bytes = await inflate(s.blob);
  const buf = bytes.buffer;
  const n = s.n;
  let off = 0;
  const i32 = () => { const a = new Int32Array(buf, off, n); off += n * 4; return a; };
  s.o = cumsum(i32(), s.scale);
  s.h = cumsum(i32(), s.scale);
  s.l = cumsum(i32(), s.scale);
  s.c = cumsum(i32(), s.scale);
  s.v = new Float32Array(buf, off, n); off += n * 4;
  s.regime = new Uint8Array(buf, off, n); off += n;
  s.ncand = new Uint8Array(buf, off, n); off += n;
  s.blob = null;

  // Index the sparse lists by bar for O(1) lookup while drawing and hovering.
  s.evByBar = new Map();
  for (const e of s.events) {
    if (!s.evByBar.has(e.i)) s.evByBar.set(e.i, []);
    s.evByBar.get(e.i).push(e);
  }
  s.evBars = [...s.evByBar.keys()].sort((a, b) => a - b);
  return s;
}

/* ---------- helpers (all times UTC) ---------- */

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const pad2 = (x) => String(x).padStart(2, "0");
const barTime = (s, i) => (s.t0 + i * s.step) * 1000;
const barOfTime = (s, ms) => clamp(Math.round((ms / 1000 - s.t0) / s.step), 0, s.n - 1);
const pct = (x) => (x * 100).toFixed(3) + "%";

function fmtUTC(ms, withTime = true) {
  const d = new Date(ms);
  const day = `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())}`;
  return withTime ? `${day} ${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}` : day;
}
function toInputValue(ms) {
  const d = new Date(ms);
  return `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())}` +
         `T${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}`;
}
const fromInputValue = (val) => Date.parse(val.length === 16 ? val + ":00Z" : val + "Z");

function fmtPrice(p) {
  const a = Math.abs(p);
  return a >= 1000 ? p.toFixed(0) : a >= 10 ? p.toFixed(2) : p.toFixed(4);
}

/* ---------- state ---------- */

const ST = {
  sym: Object.keys(DATA.symbols)[0],
  s: null,
  i0: 0, i1: 0,
  show: { taken: true, rejected: true, busy: true, superseded: true },
  ov: { regime: true, cand: true, levels: true, hold: true },
  sort: { key: "time", dir: 1 },
  hover: null,
  focus: null,
};

const PAD = { l: 66, r: 62, t: 12, b: 20 };
const H = { ribbon: 14, gap: 8, overview: 58 };
const MIN_BARS = 30;

let cv, ctx, tip, geom = {};

/* ---------- geometry ---------- */

function layout() {
  const w = cv.clientWidth, h = cv.clientHeight;
  const free = h - PAD.t - PAD.b - H.ribbon - H.overview - H.gap * 3 - 16;
  const priceH = Math.round(free * 0.8), volH = free - priceH;
  geom = {
    w, h, x0: PAD.l, plotW: w - PAD.l - PAD.r,
    price: { y: PAD.t, h: priceH },
    ribbon: { y: PAD.t + priceH + H.gap, h: H.ribbon },
    vol: { y: PAD.t + priceH + H.gap + H.ribbon + H.gap, h: volH },
    ov: { y: h - PAD.b - H.overview, h: H.overview },
  };
}

const nVis = () => ST.i1 - ST.i0 + 1;
const xOf = (i) => geom.x0 + ((i - ST.i0 + 0.5) / nVis()) * geom.plotW;
const barAtX = (x) => clamp(Math.floor(((x - geom.x0) / geom.plotW) * nVis()) + ST.i0, ST.i0, ST.i1);
const colWidth = () => Math.max(1, geom.plotW / nVis());
const stride = () => Math.max(1, Math.ceil(nVis() / geom.plotW));

function visRange() {
  const s = ST.s;
  let lo = Infinity, hi = -Infinity;
  for (let i = ST.i0; i <= ST.i1; i++) {
    if (s.l[i] < lo) lo = s.l[i];
    if (s.h[i] > hi) hi = s.h[i];
  }
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  const pad = (hi - lo) * 0.06 || hi * 0.01 || 1;
  return [lo - pad, hi + pad];
}

/* ---------- drawing ---------- */

function draw() {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth;
  const h = Math.max(560, Math.round(window.innerHeight * 0.68));
  cv.style.height = h + "px";
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  layout();

  const [lo, hi] = visRange();
  const P = geom.price;
  const yOf = (p) => P.y + P.h - ((p - lo) / (hi - lo)) * P.h;

  if (ST.ov.regime) drawRegime();
  drawAxes(lo, hi, yOf);
  if (ST.ov.hold) drawHolds();
  drawCandles(yOf);
  if (ST.ov.levels) drawLevels(yOf);
  if (ST.ov.cand) drawRibbon();
  drawVolume();
  drawMarkers(yOf);
  drawOverview();
  drawCrosshair(lo, hi);
}

function drawRegime() {
  const s = ST.s, st = stride();
  const yTop = geom.price.y, hh = geom.vol.y + geom.vol.h - yTop;
  let i = ST.i0;
  while (i <= ST.i1) {
    const r = s.regime[i];
    let j = i;
    while (j + st <= ST.i1 && s.regime[j + st] === r) j += st;
    ctx.fillStyle = REGIME_FILL[r];
    const xa = xOf(i) - colWidth() / 2, xb = xOf(j) + colWidth() / 2;
    ctx.fillRect(xa, yTop, Math.max(1, xb - xa), hh);
    i = j + st;
  }
}

function drawAxes(lo, hi, yOf) {
  const s = ST.s;
  ctx.strokeStyle = getVar("--line");
  ctx.fillStyle = getVar("--dim");
  ctx.lineWidth = 1;
  ctx.font = "11px ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const ticks = 6;
  for (let k = 0; k <= ticks; k++) {
    const p = lo + ((hi - lo) * k) / ticks, y = Math.round(yOf(p)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(geom.x0, y);
    ctx.lineTo(geom.x0 + geom.plotW, y);
    ctx.stroke();
    ctx.fillText(fmtPrice(p), geom.x0 - 8, y);
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const labels = 8, spanDays = (nVis() * s.step) / 86400;
  for (let k = 0; k <= labels; k++) {
    const i = Math.round(ST.i0 + ((nVis() - 1) * k) / labels);
    ctx.fillText(fmtUTC(barTime(s, i), spanDays < 6), xOf(i), geom.ov.y - 15);
  }
}

function drawCandles(yOf) {
  const s = ST.s, st = stride();
  const cw = colWidth() * st, body = Math.max(1, cw * 0.72), thin = body < 2.2;
  for (let i = ST.i0; i <= ST.i1; i += st) {
    const end = Math.min(i + st - 1, ST.i1);
    let hiP = -Infinity, loP = Infinity;
    for (let k = i; k <= end; k++) {
      if (s.h[k] > hiP) hiP = s.h[k];
      if (s.l[k] < loP) loP = s.l[k];
    }
    const op = s.o[i], cp = s.c[end], x = xOf(i + (end - i) / 2);
    ctx.strokeStyle = ctx.fillStyle = cp >= op ? getVar("--up") : getVar("--down");
    ctx.beginPath();
    ctx.moveTo(Math.round(x) + 0.5, yOf(hiP));
    ctx.lineTo(Math.round(x) + 0.5, yOf(loP));
    ctx.stroke();
    if (!thin) {
      const ya = yOf(Math.max(op, cp)), yb = yOf(Math.min(op, cp));
      ctx.fillRect(x - body / 2, ya, body, Math.max(1, yb - ya));
    }
  }
}

function drawHolds() {
  const s = ST.s;
  for (const t of s.trades) {
    const a = t.i, b = t.exit_i == null ? s.n - 1 : t.exit_i;
    if (b < ST.i0 || a > ST.i1) continue;
    const xa = xOf(Math.max(a, ST.i0)), xb = xOf(Math.min(b, ST.i1));
    ctx.fillStyle = t.pnl_pct >= 0 ? "rgba(38,166,154,.10)" : "rgba(239,83,80,.10)";
    ctx.fillRect(xa, geom.price.y, Math.max(1, xb - xa), geom.price.h);
  }
}

function drawLevels(yOf) {
  const s = ST.s;
  ctx.save();
  ctx.lineWidth = 1;
  for (const t of s.trades) {
    const a = t.i, b = t.exit_i == null ? s.n - 1 : t.exit_i;
    if (b < ST.i0 || a > ST.i1) continue;
    const xa = xOf(Math.max(a, ST.i0)), xb = xOf(Math.min(b, ST.i1));
    if (xb - xa < 2) continue;
    ctx.setLineDash([3, 3]);
    for (const [p, col] of [[t.stop, getVar("--down")], [t.target, getVar("--up")]]) {
      ctx.strokeStyle = col;
      const y = Math.round(yOf(p)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(xa, y);
      ctx.lineTo(xb, y);
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  ctx.restore();
}

function drawRibbon() {
  const s = ST.s, R = geom.ribbon, st = stride(), cw = colWidth() * st;
  ctx.fillStyle = "#11161f";
  ctx.fillRect(geom.x0, R.y, geom.plotW, R.h);
  for (let i = ST.i0; i <= ST.i1; i += st) {
    let m = 0;
    for (let k = i; k <= Math.min(i + st - 1, ST.i1); k++) m = Math.max(m, s.ncand[k]);
    if (!m) continue;
    ctx.fillStyle = `rgba(122,162,247,${Math.min(0.18 + m * 0.2, 0.95)})`;
    ctx.fillRect(xOf(i) - cw / 2, R.y + 2, Math.max(1, cw), R.h - 4);
  }
  ctx.fillStyle = getVar("--dim");
  ctx.font = "9.5px ui-sans-serif, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText("live candidates", geom.x0 + 4, R.y + R.h / 2);
}

function drawVolume() {
  const s = ST.s, V = geom.vol, st = stride(), cw = colWidth() * st;
  let vmax = 0;
  for (let i = ST.i0; i <= ST.i1; i++) if (s.v[i] > vmax) vmax = s.v[i];
  if (vmax > 0) {
    for (let i = ST.i0; i <= ST.i1; i += st) {
      const end = Math.min(i + st - 1, ST.i1);
      let vs = 0;
      for (let k = i; k <= end; k++) vs = Math.max(vs, s.v[k]);
      const hh = (vs / vmax) * V.h;
      ctx.fillStyle = s.c[end] >= s.o[i] ? "rgba(38,166,154,.5)" : "rgba(239,83,80,.5)";
      ctx.fillRect(xOf(i + (end - i) / 2) - cw * 0.36, V.y + V.h - hh, Math.max(1, cw * 0.72), hh);
    }
  }
  ctx.strokeStyle = getVar("--line");
  ctx.beginPath();
  ctx.moveTo(geom.x0, V.y + V.h + 0.5);
  ctx.lineTo(geom.x0 + geom.plotW, V.y + V.h + 0.5);
  ctx.stroke();
  ctx.fillStyle = getVar("--dim");
  ctx.font = "9.5px ui-sans-serif, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText("volume", geom.x0 + 4, V.y + 2);
}

function visibleEvents() {
  const s = ST.s, out = [];
  for (const bar of s.evBars) {
    if (bar < ST.i0) continue;
    if (bar > ST.i1) break;
    for (const e of s.evByBar.get(bar)) if (ST.show[e.type]) out.push(e);
  }
  return out;
}

function marker(x, y, type, dir, focused) {
  const c = TYPES[type].color, r = focused ? 7 : 5;
  ctx.strokeStyle = c;
  ctx.fillStyle = type === "taken" ? c : "rgba(14,17,23,.85)";
  ctx.lineWidth = focused ? 2 : 1.4;
  ctx.beginPath();
  if (type === "taken" || type === "busy") {
    const sgn = dir === "long" ? -1 : 1;  // triangle points the way of the side
    ctx.moveTo(x, y + sgn * r);
    ctx.lineTo(x - r, y - sgn * r * 0.75);
    ctx.lineTo(x + r, y - sgn * r * 0.75);
    ctx.closePath();
  } else if (type === "rejected") {
    ctx.arc(x, y, r * 0.85, 0, Math.PI * 2);
  } else {
    ctx.rect(x - r * 0.8, y - r * 0.8, r * 1.6, r * 1.6);
  }
  ctx.fill();
  ctx.stroke();
}

function drawMarkers(yOf) {
  const s = ST.s;
  for (const e of visibleEvents()) {
    const long = e.direction === "long";
    const y = clamp(
      yOf(long ? s.h[e.i] : s.l[e.i]) + (long ? -13 : 13),
      geom.price.y + 8, geom.price.y + geom.price.h - 8
    );
    marker(xOf(e.i), y, e.type, e.direction, ST.focus === e);
    if (e.type === "taken" && e.trade != null) {
      const t = s.trades[e.trade];
      if (t.exit_i != null && t.exit_i >= ST.i0 && t.exit_i <= ST.i1) {
        const ex = xOf(t.exit_i), ey = yOf(t.exit_price);
        ctx.strokeStyle = t.pnl_pct >= 0 ? getVar("--up") : getVar("--down");
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.moveTo(ex - 4, ey - 4); ctx.lineTo(ex + 4, ey + 4);
        ctx.moveTo(ex + 4, ey - 4); ctx.lineTo(ex - 4, ey + 4);
        ctx.stroke();
      }
    }
  }
}

function drawOverview() {
  const s = ST.s, O = geom.ov;
  ctx.fillStyle = "#11161f";
  ctx.fillRect(geom.x0, O.y, geom.plotW, O.h);
  const cols = Math.max(1, Math.floor(geom.plotW)), per = s.n / cols;
  let lo = Infinity, hi = -Infinity;
  for (let cx = 0; cx < cols; cx++) {
    const i = Math.min(s.n - 1, Math.floor(cx * per));
    if (s.c[i] < lo) lo = s.c[i];
    if (s.c[i] > hi) hi = s.c[i];
  }
  ctx.strokeStyle = "#4b566b";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let cx = 0; cx < cols; cx++) {
    const i = Math.min(s.n - 1, Math.floor(cx * per));
    const y = O.y + O.h - ((s.c[i] - lo) / (hi - lo || 1)) * O.h;
    cx ? ctx.lineTo(geom.x0 + cx, y) : ctx.moveTo(geom.x0 + cx, y);
  }
  ctx.stroke();

  // Trigger density strip, so coverage gaps are visible at full zoom-out.
  ctx.globalAlpha = 0.55;
  for (const bar of s.evBars) {
    for (const e of s.evByBar.get(bar)) {
      if (!ST.show[e.type]) continue;
      ctx.fillStyle = TYPES[e.type].color;
      ctx.fillRect(geom.x0 + (bar / s.n) * geom.plotW, O.y + O.h - 6, 1, 6);
    }
  }
  ctx.globalAlpha = 1;

  const xa = geom.x0 + (ST.i0 / s.n) * geom.plotW;
  const xb = geom.x0 + (ST.i1 / s.n) * geom.plotW;
  ctx.fillStyle = "rgba(90,140,220,.20)";
  ctx.fillRect(xa, O.y, Math.max(2, xb - xa), O.h);
  ctx.strokeStyle = "#5a8cdc";
  ctx.strokeRect(Math.round(xa) + 0.5, O.y + 0.5, Math.max(2, xb - xa), O.h - 1);
}

function drawCrosshair(lo, hi) {
  if (!ST.hover) return;
  const { i, y } = ST.hover;
  if (i < ST.i0 || i > ST.i1) return;
  const x = Math.round(xOf(i)) + 0.5;
  ctx.save();
  ctx.strokeStyle = "#5b6779";
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, geom.price.y);
  ctx.lineTo(x, geom.vol.y + geom.vol.h);
  ctx.stroke();
  if (y >= geom.price.y && y <= geom.price.y + geom.price.h) {
    const yy = Math.round(y) + 0.5;
    ctx.beginPath();
    ctx.moveTo(geom.x0, yy);
    ctx.lineTo(geom.x0 + geom.plotW, yy);
    ctx.stroke();
    const p = lo + ((geom.price.y + geom.price.h - y) / geom.price.h) * (hi - lo);
    ctx.setLineDash([]);
    ctx.fillStyle = "#5b6779";
    ctx.fillRect(geom.x0 + geom.plotW, yy - 8, PAD.r, 16);
    ctx.fillStyle = "#0e1117";
    ctx.font = "10.5px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(fmtPrice(p), geom.x0 + geom.plotW + PAD.r / 2, yy);
  }
  ctx.restore();
}

/* ---------- tooltip ---------- */

const rowsHtml = (rows) => rows
  .map((r) => `<div class="row"><span>${r[0]}</span><span><b>${r[1]}</b></span></div>`)
  .join("");

function showTip(px, py, i) {
  const s = ST.s;
  let html = rowsHtml([
    ["Bar (15m)", fmtUTC(barTime(s, i))],
    ["Open", fmtPrice(s.o[i])], ["High", fmtPrice(s.h[i])],
    ["Low", fmtPrice(s.l[i])], ["Close", fmtPrice(s.c[i])],
    ["Volume", s.v[i].toLocaleString(undefined, { maximumFractionDigits: 2 })],
    ["Regime (4h)", REGIMES[s.regime[i]]],
    ["Live candidates", String(s.ncand[i])],
  ]);

  const evs = (s.evByBar.get(i) || []).filter((e) => ST.show[e.type]);
  for (const e of evs) {
    const t = TYPES[e.type];
    html += `<hr><div class="row"><span style="color:${t.color}"><b>${t.label}</b></span>` +
            `<span>${e.direction} · ${e.pattern}</span></div>`;
    const det = [["Level", fmtPrice(e.level)], ["Entry", fmtPrice(e.entry)]];
    if (e.stop != null) det.push(["Stop", fmtPrice(e.stop)], ["Target", fmtPrice(e.target)]);
    if (e.risk_pct != null)
      det.push(["Risk / reward", `${pct(e.risk_pct)} / ${pct(e.reward_pct)} · ${e.rr.toFixed(2)}R`]);
    if (e.volume_ratio != null)
      det.push(["Trigger volume", `${e.volume_ratio.toFixed(2)}× avg${e.volume_high ? " (high)" : ""}`]);
    if (e.reason) det.push(["Remark", REASON_TEXT[e.reason] || e.reason]);
    if (e.type === "taken" && e.trade != null) {
      const tr = s.trades[e.trade];
      det.push(["Exit", `${fmtUTC(barTime(s, tr.exit_i))} @ ${fmtPrice(tr.exit_price)}`]);
      det.push(["Outcome", `${tr.outcome} · ${(tr.pnl_pct * 100).toFixed(3)}% net`]);
    }
    html += rowsHtml(det);
  }
  if (!evs.length && s.ncand[i]) {
    html += `<hr>` + rowsHtml([["No trigger", `${s.ncand[i]} candidate(s) waiting`]]);
  }

  tip.innerHTML = html;
  tip.style.display = "block";
  const box = cv.getBoundingClientRect();
  tip.style.left = Math.min(px + 16, box.width - tip.offsetWidth - 8) + "px";
  tip.style.top = clamp(py - tip.offsetHeight / 2, 6, box.height - tip.offsetHeight - 6) + "px";
}

/* ---------- KPIs and table ---------- */

function renderKpis() {
  const s = ST.s;
  const evs = [];
  for (const bar of s.evBars) {
    if (bar < ST.i0) continue;
    if (bar > ST.i1) break;
    evs.push(...s.evByBar.get(bar));
  }
  const by = (t) => evs.filter((e) => e.type === t).length;
  const taken = evs
    .filter((e) => e.type === "taken" && e.trade != null)
    .map((e) => s.trades[e.trade]);
  const wins = taken.filter((t) => t.pnl_pct > 0).length;
  const net = taken.reduce((a, t) => a + t.pnl_pct, 0);
  const days = (nVis() * s.step) / 86400;
  let barsWithCand = 0;
  for (let i = ST.i0; i <= ST.i1; i++) if (s.ncand[i]) barsWithCand++;

  const kpis = [
    ["Bars shown", nVis().toLocaleString(), `${days.toFixed(days < 10 ? 1 : 0)} days of 15m bars`],
    ["Bars with a candidate", `${((barsWithCand / nVis()) * 100).toFixed(1)}%`,
     `${barsWithCand.toLocaleString()} bars armed`],
    ["Triggers fired", String(evs.length), "level crossings"],
    ["Positions taken", String(by("taken")), "filled entries"],
    ["Rejected by R:R", String(by("rejected")), "screened out"],
    ["Skipped — busy", String(by("busy")), "position already open"],
    ["Win rate", taken.length ? `${((wins / taken.length) * 100).toFixed(2)}%` : "—",
     `${wins}/${taken.length} closed`],
    ["Net P&L (sum)", taken.length ? `${(net * 100).toFixed(2)}%` : "—", "equal size, after costs"],
  ];
  document.getElementById("kpis").innerHTML = kpis
    .map((k) => `<div class="kpi"><div class="k">${k[0]}</div><div class="v">${k[1]}</div>` +
                `<div class="n">${k[2]}</div></div>`)
    .join("");
}

function renderTable() {
  const s = ST.s;
  const { key, dir } = ST.sort;
  const rows = visibleEvents().slice().sort((a, b) => {
    const av = key === "time" ? a.i : a[key], bv = key === "time" ? b.i : b[key];
    if (av == null && bv == null) return a.i - b.i;
    if (av == null) return 1;
    if (bv == null) return -1;
    return (av > bv ? 1 : av < bv ? -1 : 0) * dir || a.i - b.i;
  });

  const CAP = 600;
  document.getElementById("evcount").textContent =
    `— ${rows.length.toLocaleString()} trigger(s)` +
    (rows.length > CAP ? `, showing first ${CAP}` : "");

  const cell = (v, cls) => `<td class="${cls || ""}">${v}</td>`;
  document.querySelector("#evtable tbody").innerHTML = rows.slice(0, CAP)
    .map((e) => {
      const t = TYPES[e.type];
      const tr = e.type === "taken" && e.trade != null ? s.trades[e.trade] : null;
      const pnl = tr
        ? `<span class="${tr.pnl_pct >= 0 ? "pos" : "neg"}">${(tr.pnl_pct * 100).toFixed(3)}%</span>`
        : "—";
      const remark = tr
        ? `exit ${tr.outcome} @ ${fmtPrice(tr.exit_price)}`
        : REASON_TEXT[e.reason] || e.reason || "—";
      return `<tr data-bar="${e.i}">` +
        cell(fmtUTC(barTime(s, e.i)), "l") +
        cell(`<span class="tag" style="background:${t.color}22;color:${t.color}">${t.label}</span>`, "l") +
        cell(e.pattern, "l") + cell(e.direction, "l") +
        cell(fmtPrice(e.entry)) + cell(fmtPrice(e.level)) +
        cell(e.stop != null ? fmtPrice(e.stop) : "—") +
        cell(e.target != null ? fmtPrice(e.target) : "—") +
        cell(e.risk_pct != null ? pct(e.risk_pct) : "—") +
        cell(e.reward_pct != null ? pct(e.reward_pct) : "—") +
        cell(e.rr != null ? e.rr.toFixed(2) : "—") +
        cell(e.volume_ratio != null ? e.volume_ratio.toFixed(2) : "—") +
        cell(remark, "l") + cell(pnl) + "</tr>";
    })
    .join("");
}

function refresh() {
  draw();
  renderKpis();
  renderTable();
  document.getElementById("from").value = toInputValue(barTime(ST.s, ST.i0));
  document.getElementById("to").value = toInputValue(barTime(ST.s, ST.i1));
}

/* ---------- view control ---------- */

function setView(i0, i1) {
  const n = ST.s.n;
  let a = Math.round(i0), b = Math.round(i1);
  const span = Math.max(MIN_BARS, b - a + 1);
  if (a < 0) { a = 0; b = a + span - 1; }
  if (b > n - 1) { b = n - 1; a = Math.max(0, b - span + 1); }
  ST.i0 = clamp(a, 0, n - 1);
  ST.i1 = clamp(Math.max(b, ST.i0 + MIN_BARS - 1), ST.i0, n - 1);
  refresh();
}

function zoomAt(i, factor) {
  const span = nVis();
  const next = clamp(Math.round(span * factor), MIN_BARS, ST.s.n);
  const frac = (i - ST.i0) / span;
  const a = i - frac * next;
  setView(a, a + next - 1);
}

function jump(dir) {
  const s = ST.s, mid = Math.round((ST.i0 + ST.i1) / 2), span = nVis();
  const bars = s.evBars.filter((b) => s.evByBar.get(b).some((e) => ST.show[e.type]));
  const next = dir > 0
    ? bars.find((b) => b > mid + 1)
    : [...bars].reverse().find((b) => b < mid - 1);
  if (next == null) return;
  ST.focus = s.evByBar.get(next).find((e) => ST.show[e.type]) || null;
  setView(next - span / 2, next + span / 2);
}

/* ---------- wiring ---------- */

async function selectSymbol(sym) {
  ST.sym = sym;
  ST.s = await unpack(sym);
  ST.focus = null;
  const bars30d = Math.round((30 * 86400) / ST.s.step);
  setView(Math.max(0, ST.s.n - bars30d), ST.s.n - 1);
}

function buildLayerChips() {
  const box = document.getElementById("layers");
  box.innerHTML = Object.entries(TYPES)
    .map(([k, t]) => `<span class="chip" data-layer="${k}">` +
                     `<span class="dot" style="background:${t.color}"></span>${t.label}</span>`)
    .join("");
  box.querySelectorAll("[data-layer]").forEach((el) => {
    el.addEventListener("click", () => {
      const k = el.dataset.layer;
      ST.show[k] = !ST.show[k];
      el.classList.toggle("off", !ST.show[k]);
      refresh();
    });
  });
}

function wire() {
  const symSel = document.getElementById("sym");
  symSel.innerHTML = Object.keys(DATA.symbols).map((s) => `<option>${s}</option>`).join("");
  symSel.value = ST.sym;
  symSel.addEventListener("change", () => selectSymbol(symSel.value));

  buildLayerChips();
  document.querySelectorAll("[data-ov]").forEach((el) => {
    el.addEventListener("click", () => {
      const k = el.dataset.ov;
      ST.ov[k] = !ST.ov[k];
      el.classList.toggle("off", !ST.ov[k]);
      draw();
    });
  });

  for (const id of ["from", "to"]) {
    document.getElementById(id).addEventListener("change", () => {
      const a = fromInputValue(document.getElementById("from").value);
      const b = fromInputValue(document.getElementById("to").value);
      if (isNaN(a) || isNaN(b) || b <= a) { refresh(); return; }
      setView(barOfTime(ST.s, a), barOfTime(ST.s, b));
    });
  }

  document.querySelectorAll("[data-days]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const d = +btn.dataset.days;
      if (!d) { setView(0, ST.s.n - 1); return; }
      const bars = Math.round((d * 86400) / ST.s.step);
      setView(ST.i1 - bars + 1, ST.i1);
    });
  });
  document.getElementById("nextEv").addEventListener("click", () => jump(1));
  document.getElementById("prevEv").addEventListener("click", () => jump(-1));

  document.querySelectorAll("#evtable th").forEach((th) => {
    th.addEventListener("click", () => {
      const k = th.dataset.s;
      ST.sort = { key: k, dir: ST.sort.key === k ? -ST.sort.dir : 1 };
      renderTable();
    });
  });
  document.querySelector("#evtable tbody").addEventListener("click", (ev) => {
    const tr = ev.target.closest("tr");
    if (!tr) return;
    const bar = +tr.dataset.bar, span = nVis();
    ST.focus = (ST.s.evByBar.get(bar) || [])[0] || null;
    setView(bar - span / 2, bar + span / 2);
  });

  let drag = null;
  cv.addEventListener("mousedown", (e) => {
    const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    drag = y >= geom.ov.y
      ? { mode: "brush", x0: x, x1: x }
      : { mode: "pan", x, i0: ST.i0, i1: ST.i1 };
  });
  window.addEventListener("mouseup", () => {
    if (drag && drag.mode === "brush" && Math.abs(drag.x1 - drag.x0) > 3) {
      const n = ST.s.n;
      const f = (x) => clamp(Math.round(((x - geom.x0) / geom.plotW) * n), 0, n - 1);
      const a = f(Math.min(drag.x0, drag.x1)), b = f(Math.max(drag.x0, drag.x1));
      drag = null;
      setView(a, b);
      return;
    }
    drag = null;
  });
  cv.addEventListener("mousemove", (e) => {
    const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    if (drag) {
      if (drag.mode === "brush") {
        drag.x1 = x;
        draw();
        ctx.fillStyle = "rgba(90,140,220,.25)";
        ctx.fillRect(Math.min(drag.x0, drag.x1), geom.ov.y, Math.abs(drag.x1 - drag.x0), geom.ov.h);
        return;
      }
      const shift = Math.round(((drag.x - x) / geom.plotW) * (drag.i1 - drag.i0 + 1));
      setView(drag.i0 + shift, drag.i1 + shift);
      return;
    }
    if (y > geom.vol.y + geom.vol.h || x < geom.x0 || x > geom.x0 + geom.plotW) {
      tip.style.display = "none";
      if (ST.hover) { ST.hover = null; draw(); }
      return;
    }
    ST.hover = { i: barAtX(x), y };
    draw();
    showTip(x, y, ST.hover.i);
  });
  cv.addEventListener("mouseleave", () => {
    tip.style.display = "none";
    ST.hover = null;
    draw();
  });
  cv.addEventListener("wheel", (e) => {
    const r = cv.getBoundingClientRect(), x = e.clientX - r.left;
    if (x < geom.x0 || x > geom.x0 + geom.plotW) return;
    e.preventDefault();
    zoomAt(barAtX(x), e.deltaY > 0 ? 1.25 : 0.8);
  }, { passive: false });
  cv.addEventListener("dblclick", (e) => {
    const r = cv.getBoundingClientRect();
    if (e.clientY - r.top >= geom.ov.y) setView(0, ST.s.n - 1);
  });
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    const step = Math.max(1, Math.round(nVis() * 0.2));
    const mid = Math.round((ST.i0 + ST.i1) / 2);
    if (e.key === "ArrowLeft") setView(ST.i0 - step, ST.i1 - step);
    else if (e.key === "ArrowRight") setView(ST.i0 + step, ST.i1 + step);
    else if (e.key === "+" || e.key === "=") zoomAt(mid, 0.8);
    else if (e.key === "-") zoomAt(mid, 1.25);
    else return;
    e.preventDefault();
  });
  window.addEventListener("resize", () => draw());
}

async function main() {
  const app = document.getElementById("app");
  if (typeof DecompressionStream === "undefined") {
    app.textContent = "This browser lacks DecompressionStream; open in a current Chrome, Safari or Firefox.";
    return;
  }
  app.classList.remove("loading");
  app.innerHTML = "";
  app.appendChild(document.getElementById("tpl").content.cloneNode(true));
  cv = document.getElementById("cv");
  ctx = cv.getContext("2d");
  tip = document.getElementById("tip");
  wire();
  await selectSymbol(ST.sym);
}

main();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", required=True, help="JSON from export_bar_annotations.py")
    ap.add_argument("--out", required=True, help="Output HTML path")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    with open(args.annotations) as fh:
        annotations = json.load(fh)
    html = build(annotations)
    with open(args.out, "w") as fh:
        fh.write(html)
    logger.info("wrote %s (%.2f MB)", args.out, len(html.encode()) / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
