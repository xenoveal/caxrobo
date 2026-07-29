"""
Build the standalone performance chart for the walk-forward result.

Runs the pooled walk-forward, then renders a single self-contained HTML file:
the gate scorecard, pooled and per-symbol equity curves with the one-shot OOS
window marked, the pooled underwater drawdown, and per-fold train/test
expectancy. Nothing is fetched at runtime and no script is loaded from a CDN, so
the file works from the filesystem.

Companion to build_review_chart.py, which draws per-BAR signal geometry. This
one draws PERFORMANCE, which is what the gate actually scores.

Usage:
    python scripts/build_performance_chart.py \
        --out .claude/PRPs/reports/phase7-walkforward-result.chart.html
"""

import argparse
import html
import json
import logging
import math
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "src")

from trading_bot import config  # noqa: E402
from trading_bot.backtest import walkforward  # noqa: E402
from trading_bot.backtest.engine import run_backtest  # noqa: E402
from trading_bot.backtest.equity import daily_returns  # noqa: E402
from trading_bot.backtest.metrics import compute_metrics  # noqa: E402
from trading_bot.backtest.walkforward import DAY_MS, walk_forward_pooled  # noqa: E402

logger = logging.getLogger("build_performance_chart")

# Categorical slots 1-4 from the validated reference palette (adjacent pairlist:
# lines). Light/dark are the same four hues re-stepped for each surface, not a
# flipped palette. Validated with the skill's validator: all checks PASS in both
# modes; light-mode aqua/yellow fall below 3:1 on the surface, so the relief rule
# applies and every series carries a direct label plus a table view.
SERIES = [
    ("Pooled", "#2a78d6", "#3987e5"),
    ("BTCUSDT", "#eb6834", "#d95926"),
    ("ETHUSDT", "#1baf7a", "#199e70"),
    ("SOLUSDT", "#eda100", "#c98500"),
]


def _fmt_pct(v, nd=2, signed=False):
    if v is None:
        return "--"
    return f"{v * 100:+.{nd}f}%" if signed else f"{v * 100:.{nd}f}%"


def _fmt_num(v, nd=2):
    return "--" if v is None else f"{v:.{nd}f}"


def _day_label(day_index, first_day_ms):
    d = datetime.fromtimestamp(first_day_ms / 1000, tz=timezone.utc) + timedelta(
        days=day_index
    )
    return d.strftime("%Y-%m-%d")


def _equity(rets):
    """Compounded equity curve starting at 1.0, one point per day."""
    eq, out = 1.0, []
    for r in rets:
        eq *= 1.0 + r
        out.append(eq)
    return out


def _underwater(eq):
    peak, out = 1.0, []
    for v in eq:
        peak = max(peak, v)
        out.append(0.0 if peak <= 0 else 1.0 - v / peak)
    return out


def _path(values, x_of, y_of):
    """SVG polyline path through (index, value) pairs."""
    return "M" + " L".join(f"{x_of(i):.2f},{y_of(v):.2f}" for i, v in enumerate(values))


def _area_path(values, x_of, y_of, y0):
    if not values:
        return ""
    body = " L".join(f"{x_of(i):.2f},{y_of(v):.2f}" for i, v in enumerate(values))
    return f"M{x_of(0):.2f},{y0:.2f} L{body} L{x_of(len(values) - 1):.2f},{y0:.2f} Z"


def _nice_ticks(lo, hi, count=5):
    """Roughly `count` ticks on a 1/2/2.5/5/10 x power-of-ten ladder.

    The magnitude must come from log10, not from the digit count of int(raw):
    for a range like 0.70-2.20 that floored the step at 0.01 and the ladder
    never caught up, so the axis rendered 16 gridlines instead of 5.
    """
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / max(1, count)
    mag = 10.0 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    step = mag * 10
    for mult in (1, 2, 2.5, 5, 10):
        if (hi - lo) / (mag * mult) <= count + 1:
            step = mag * mult
            break
    first = math.ceil(lo / step) * step
    ticks, t = [], first
    while t <= hi + 1e-9:
        ticks.append(round(t, 10))
        t += step
    return ticks or [lo, hi]


def _spread(labels, min_gap=13.0, top=0.0, bottom=1e9):
    """Nudge overlapping direct labels apart, preserving order.

    Three of the four series ended within 0.07x of each other, so their
    end-of-line labels overlapped into an unreadable stack.
    """
    items = sorted(labels, key=lambda t: t[1])
    for i in range(1, len(items)):
        text, y = items[i]
        prev_y = items[i - 1][1]
        if y - prev_y < min_gap:
            items[i] = (text, prev_y + min_gap)
    # If the stack ran past the bottom, push the whole run back up.
    if items and items[-1][1] > bottom:
        shift = items[-1][1] - bottom
        items = [(t, max(top, y - shift)) for t, y in items]
    return items


def collect(conn, symbols, start_ms, end_ms):
    """Run the walk-forward and build every series the page needs."""
    logger.info("running pooled walk-forward (this is the slow part)")
    result = walk_forward_pooled(conn, list(symbols), start_ms=start_ms, end_ms=end_ms)

    params = result.final_params
    extra = {}
    if result.final_max_hold_bars is not None:
        extra["max_hold_bars"] = result.final_max_hold_bars

    # Equity over the FULL span at the parameters the gate selected, so the
    # curve depicts the configuration the verdict is about.
    per_symbol_trades = {}
    for sym in symbols:
        per_symbol_trades[sym] = run_backtest(
            conn, sym, start_ms=start_ms, end_ms=end_ms, params=params, **extra
        )
    pooled_trades = [t for ts in per_symbol_trades.values() for t in ts]

    curves = {"Pooled": _equity(daily_returns(pooled_trades, start_ms, end_ms))}
    for sym, trades in per_symbol_trades.items():
        curves[sym] = _equity(daily_returns(trades, start_ms, end_ms))

    return {
        "result": result,
        "curves": curves,
        "underwater": _underwater(curves["Pooled"]),
        "pooled_metrics": compute_metrics(pooled_trades),
        "per_symbol_trades": {s: len(t) for s, t in per_symbol_trades.items()},
        "start_ms": start_ms,
        "end_ms": end_ms,
    }


def _scorecard(result):
    """The five gate conditions as (label, value, threshold, ok) rows.

    Every verdict ships with a mark and a word, never colour alone.
    """
    m, eq = result.oos_metrics, result.oos_equity
    per_sym = result.per_symbol_expectancy
    n_pos = sum(1 for v in per_sym.values() if v is not None and v > 0)
    return [
        (
            "Sample adequacy",
            f"{m['n_trades']} trades",
            f"≥ {config.WF_MIN_TRADES}",
            m["n_trades"] >= config.WF_MIN_TRADES,
        ),
        (
            "Sharpe (annualised)",
            _fmt_num(eq["sharpe"]),
            f"≥ {walkforward.GATE_MIN_SHARPE}",
            eq["sharpe"] is not None and eq["sharpe"] >= walkforward.GATE_MIN_SHARPE,
        ),
        (
            "Deflated Sharpe (DSR)",
            _fmt_num(eq["dsr"], 4),
            f"> {walkforward.GATE_MIN_DSR}",
            eq["dsr"] is not None and eq["dsr"] > walkforward.GATE_MIN_DSR,
        ),
        (
            "Equity max drawdown",
            _fmt_pct(eq["max_drawdown_pct"]),
            f"≤ {walkforward.GATE_MAX_DRAWDOWN * 100:.0f}%",
            eq["max_drawdown_pct"] is not None
            and eq["max_drawdown_pct"] <= walkforward.GATE_MAX_DRAWDOWN,
        ),
        (
            "Per-symbol expectancy",
            f"{n_pos} of {len(per_sym)} positive",
            "all > 0",
            n_pos == len(per_sym) and len(per_sym) > 0,
        ),
    ]


def build(data) -> str:
    result = data["result"]
    curves = data["curves"]
    uw = data["underwater"]
    folds = result.folds
    n_days = len(curves["Pooled"])
    first_day_ms = (data["start_ms"] // DAY_MS) * DAY_MS
    oos_day0 = max(0, result.oos_start // DAY_MS - data["start_ms"] // DAY_MS)
    live = [n for n, _l, _d in SERIES if n in curves]

    # ---- equity geometry ---------------------------------------------------
    W, H = 960, 340
    ML, MR, MT, MB = 58, 100, 18, 34
    PW, PH = W - ML - MR, H - MT - MB

    all_vals = [v for c in curves.values() for v in c] + [1.0]
    lo, hi = min(all_vals), max(all_vals)
    pad = (hi - lo) * 0.08 or 0.1
    lo, hi = lo - pad, hi + pad

    def x_of(i):
        return ML + (PW * i / max(1, n_days - 1))

    def y_of(v):
        return MT + PH * (1 - (v - lo) / (hi - lo))

    y_ticks = _nice_ticks(lo, hi, 5)
    grid = "".join(
        f'<line class="grid" x1="{ML}" y1="{y_of(t):.1f}" x2="{ML + PW}" y2="{y_of(t):.1f}"/>'
        for t in y_ticks
    )
    ylab = "".join(
        f'<text class="tick" x="{ML - 10}" y="{y_of(t) + 4:.1f}" text-anchor="end">'
        f"{t:.2f}×</text>"
        for t in y_ticks
    )

    x_ticks = []
    for i in range(n_days):
        d = datetime.fromtimestamp(first_day_ms / 1000, tz=timezone.utc) + timedelta(days=i)
        if d.day == 1 and d.month in (1, 4, 7, 10):
            x_ticks.append((i, d.strftime("%b %Y")))
    xlab = "".join(
        f'<line class="grid" x1="{x_of(i):.1f}" y1="{MT}" x2="{x_of(i):.1f}" y2="{MT + PH}"/>'
        f'<text class="tick" x="{x_of(i):.1f}" y="{MT + PH + 22}" text-anchor="middle">{lab}</text>'
        for i, lab in x_ticks
    )

    oos_band = ""
    if oos_day0 < n_days - 1:
        bx, bw = x_of(oos_day0), x_of(n_days - 1) - x_of(oos_day0)
        oos_band = (
            f'<rect class="oos" x="{bx:.1f}" y="{MT}" width="{bw:.1f}" height="{PH}"/>'
            f'<line class="oos-edge" x1="{bx:.1f}" y1="{MT}" x2="{bx:.1f}" y2="{MT + PH}"/>'
            # Right-anchored inside the band: left-anchoring ran the text off
            # the plot, since the band is only 90 of ~1100 days wide.
            f'<text class="oos-label" x="{ML + PW - 6}" y="{MT + 15}" '
            f'text-anchor="end">one-shot OOS (90d)</text>'
        )

    lines = []
    label_specs = []
    for idx, name in enumerate(live):
        c = curves[name]
        lines.append(f'<path class="series s{idx + 1}" d="{_path(c, x_of, y_of)}"/>')
        # Direct label at the series end: the documented relief for the
        # light-mode sub-3:1 slots, and it keeps identity off colour alone.
        label_specs.append(
            (
                (idx + 1, f"{html.escape(name)} {c[-1]:.2f}×"),
                y_of(c[-1]) + 4,
            )
        )
    direct = [
        f'<text class="dlabel s{slot}t" x="{ML + PW + 8}" y="{y:.1f}">{txt}</text>'
        for (slot, txt), y in _spread(label_specs, 13.0, MT + 8, MT + PH)
    ]
    baseline = (
        f'<line class="baseline" x1="{ML}" y1="{y_of(1.0):.1f}" '
        f'x2="{ML + PW}" y2="{y_of(1.0):.1f}"/>'
    )

    # ---- drawdown geometry -------------------------------------------------
    DH = 176
    dPH = DH - MT - MB
    dmax = max(uw + [0.05])

    def dy_of(v):
        return MT + dPH * (v / dmax)

    d_grid = "".join(
        f'<line class="grid" x1="{ML}" y1="{dy_of(t):.1f}" x2="{ML + PW}" y2="{dy_of(t):.1f}"/>'
        f'<text class="tick" x="{ML - 10}" y="{dy_of(t) + 4:.1f}" text-anchor="end">'
        f'{"0%" if t == 0 else f"-{t * 100:.0f}%"}</text>'
        for t in _nice_ticks(0, dmax, 3)
    )
    d_oos = ""
    if oos_day0 < n_days - 1:
        bx, bw = x_of(oos_day0), x_of(n_days - 1) - x_of(oos_day0)
        d_oos = f'<rect class="oos" x="{bx:.1f}" y="{MT}" width="{bw:.1f}" height="{dPH}"/>'

    # ---- fold geometry -----------------------------------------------------
    FW, FH = 960, 268
    fML, fMR, fMT, fMB = 58, 20, 22, 60
    fPW, fPH = FW - fML - fMR, FH - fMT - fMB
    vals = [
        v
        for f in folds
        for v in (f.train_expectancy, f.test_metrics["expectancy_pct"])
        if v is not None
    ]
    fhi = max(vals + [0.0])
    flo = min(vals + [0.0])
    fpad = (fhi - flo) * 0.12 or 0.01
    fhi, flo = fhi + fpad, flo - fpad

    def fy_of(v):
        return fMT + fPH * (1 - (v - flo) / (fhi - flo))

    zero_y = fy_of(0.0)
    group_w = fPW / max(1, len(folds))
    bar_w = min(16.0, group_w * 0.32)
    fold_bars, fold_labels = [], []
    for i, f in enumerate(folds):
        cx = fML + group_w * (i + 0.5)
        for k, (v, cls, what) in enumerate(
            (
                (f.train_expectancy, "s1", "train"),
                (f.test_metrics["expectancy_pct"], "s2", "test"),
            )
        ):
            if v is None:
                continue
            # 2px surface gap between adjacent bars; 4px rounded data-end.
            bx = cx - bar_w - 1 + k * (bar_w + 2)
            top = min(fy_of(v), zero_y)
            h = max(1.0, abs(fy_of(v) - zero_y))
            fold_bars.append(
                f'<rect class="bar {cls}" x="{bx:.1f}" y="{top:.1f}" '
                f'width="{bar_w:.1f}" height="{h:.1f}" rx="4">'
                f"<title>fold {i} {what} expectancy {_fmt_pct(v, 4, True)}</title></rect>"
            )
        fold_labels.append(
            f'<text class="tick" x="{cx:.1f}" y="{fMT + fPH + 18}" text-anchor="middle">'
            f'{i}{"*" if f.train_expectancy is None else ""}</text>'
        )
    f_grid = "".join(
        f'<line class="grid" x1="{fML}" y1="{fy_of(t):.1f}" x2="{fML + fPW}" y2="{fy_of(t):.1f}"/>'
        f'<text class="tick" x="{fML - 10}" y="{fy_of(t) + 4:.1f}" text-anchor="end">'
        f"{t * 100:+.1f}%</text>"
        for t in _nice_ticks(flo, fhi, 4)
    )

    # ---- tiles, tables, prose ---------------------------------------------
    eq_m = result.oos_equity
    rows = _scorecard(result)
    n_fail = sum(1 for r in rows if not r[3])
    passed = result.passed

    tiles = [
        ("OOS annualised return", _fmt_pct(eq_m["ann_return_pct"], 1, True), "reported, not gated"),
        ("OOS Sharpe", _fmt_num(eq_m["sharpe"]), f"gate ≥ {walkforward.GATE_MIN_SHARPE}"),
        ("OOS trades", str(result.oos_metrics["n_trades"]), f"gate ≥ {config.WF_MIN_TRADES}"),
        ("Deflated Sharpe", _fmt_num(eq_m["dsr"], 3), f"gate > {walkforward.GATE_MIN_DSR}"),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="tl">{html.escape(t)}</div>'
        f'<div class="tv">{html.escape(v)}</div>'
        f'<div class="ts">{html.escape(s)}</div></div>'
        for t, v, s in tiles
    )
    sc_html = "".join(
        f"<tr><td>{html.escape(lab)}</td><td class=num>{html.escape(val)}</td>"
        f"<td class=num>{html.escape(thr)}</td>"
        f'<td><span class="chip {"good" if ok else "crit"}">'
        f'{"✓" if ok else "✗"} {"PASS" if ok else "FAIL"}</span></td></tr>'
        for lab, val, thr, ok in rows
    )
    ps_html = "".join(
        f"<tr><td>{html.escape(s)}</td>"
        f"<td class=num>{data['per_symbol_trades'].get(s, 0)}</td>"
        f"<td class=num>{_fmt_pct(v, 4, True)}</td>"
        f'<td><span class="chip {"good" if (v is not None and v > 0) else "crit"}">'
        f'{"✓ OK" if (v is not None and v > 0) else "✗ FAIL"}</span></td></tr>'
        for s, v in result.per_symbol_expectancy.items()
    )
    tv_html = "".join(
        f"<tr><td>{_day_label(i, first_day_ms)}</td>"
        + "".join(f"<td class=num>{curves[n][i]:.3f}×</td>" for n in live)
        + f"<td class=num>-{uw[i] * 100:.1f}%</td></tr>"
        for i in range(0, n_days, 91)
    )
    fold_rows = "".join(
        f"<tr><td>{i}{'*' if f.train_expectancy is None else ''}</td>"
        f"<td>{_day_label(f.train_start // DAY_MS - first_day_ms // DAY_MS, first_day_ms)}</td>"
        f"<td class=num>{_fmt_pct(f.train_expectancy, 4, True)}</td>"
        f"<td class=num>{_fmt_pct(f.test_metrics['expectancy_pct'], 4, True)}</td>"
        f"<td class=num>{f.test_metrics['n_trades']}</td>"
        f"<td class=num>{_fmt_num(f.positive_neighbour_fraction)}</td></tr>"
        for i, f in enumerate(folds)
    )
    p = result.final_params
    sel = (
        f"trail_enabled={p.trail_enabled} · target_enabled={p.target_enabled} "
        f"· max_hold_bars={result.final_max_hold_bars} · rr_floor={p.rr_floor}"
    )
    pooled = data["pooled_metrics"]
    neg_folds = sum(1 for f in folds if (f.test_metrics["expectancy_pct"] or 0) < 0)
    fallbacks = sum(1 for f in folds if f.train_expectancy is None)
    legend_series = "".join(
        f'<span class="lg"><span class="sw" style="background:var(--s{i + 1})"></span>'
        f"{html.escape(n)}</span>"
        for i, n in enumerate(live)
    )
    head_series = "".join(f"<th class=num>{html.escape(n)}</th>" for n in live)
    series_json = json.dumps(
        {
            "names": live,
            "curves": [[round(v, 5) for v in curves[n]] for n in live],
            "uw": [round(v, 5) for v in uw],
            "first": first_day_ms,
            "ml": ML,
            "pw": PW,
            "n": n_days,
        }
    )
    verdict_cls = "good" if passed else "crit"
    verdict_mark = "✓ PASS" if passed else "✗ FAIL"

    # The doctype is not optional: without it browsers render in Quirks Mode,
    # which changes box sizing and line layout out from under the CSS.
    return f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 7 walk-forward result</title>
<style>
:root {{ color-scheme: light dark; }}
.viz-root {{
  --surface-1:#fcfcfb; --plane:#f9f9f7;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --good:#0ca30c; --crit:#d03b3b; --oos:rgba(42,120,214,0.07);
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) .viz-root {{
    --surface-1:#1a1a19; --plane:#0d0d0d;
    --text-primary:#fff; --text-secondary:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
    --oos:rgba(57,135,229,0.10);
  }}
}}
:root[data-theme="dark"] .viz-root {{
  --surface-1:#1a1a19; --plane:#0d0d0d;
  --text-primary:#fff; --text-secondary:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --oos:rgba(57,135,229,0.10);
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--plane); }}
.viz-root {{
  background:var(--plane); color:var(--text-primary);
  font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
  padding:28px 20px 56px; max-width:1040px; margin:0 auto;
}}
h1 {{ font-size:21px; margin:0 0 4px; letter-spacing:-0.01em; }}
h2 {{ font-size:15px; margin:34px 0 2px; }}
.sub {{ color:var(--text-secondary); font-size:13px; margin:0 0 6px; }}
.cap {{ color:var(--muted); font-size:12px; margin:0 0 10px; }}
.card {{
  background:var(--surface-1); border:1px solid var(--border);
  border-radius:10px; padding:14px 16px; margin-top:10px; overflow-x:auto;
}}
.verdict {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap;
  margin:10px 0 2px; font-size:15px; font-weight:600; }}
.chip {{ display:inline-block; padding:1px 8px; border-radius:999px;
  font-size:12px; font-weight:600; border:1px solid var(--border); white-space:nowrap; }}
.chip.good {{ color:var(--good); }}
.chip.crit {{ color:var(--crit); }}
.verdict .chip {{ font-size:14px; }}
.tiles {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:12px; }}
.tile {{ flex:1 1 190px; background:var(--surface-1); border:1px solid var(--border);
  border-radius:10px; padding:12px 14px; }}
.tl {{ color:var(--text-secondary); font-size:12px; }}
.tv {{ font-size:26px; font-weight:650; letter-spacing:-0.02em; margin:2px 0; }}
.ts {{ color:var(--muted); font-size:11px; }}
svg {{ display:block; width:100%; height:auto; }}
.grid {{ stroke:var(--grid); stroke-width:1; }}
.baseline {{ stroke:var(--axis); stroke-width:1; stroke-dasharray:3 3; }}
.tick {{ fill:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }}
.series {{ fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }}
.s1 {{ stroke:var(--s1); }} .s2 {{ stroke:var(--s2); }}
.s3 {{ stroke:var(--s3); }} .s4 {{ stroke:var(--s4); }}
rect.bar.s1 {{ fill:var(--s1); }} rect.bar.s2 {{ fill:var(--s2); }}
.dlabel {{ font-size:11px; font-weight:600; }}
.s1t {{ fill:var(--s1); }} .s2t {{ fill:var(--s2); }}
.s3t {{ fill:var(--s3); }} .s4t {{ fill:var(--s4); }}
.uw {{ fill:var(--s2); fill-opacity:0.22; stroke:var(--s2); stroke-width:2; }}
.oos {{ fill:var(--oos); }}
.oos-edge {{ stroke:var(--s1); stroke-width:1; stroke-dasharray:4 3; }}
.oos-label {{ fill:var(--text-secondary); font-size:11px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; margin:8px 0 0; }}
.lg {{ display:inline-flex; align-items:center; gap:6px;
  color:var(--text-secondary); font-size:12px; }}
.sw {{ width:11px; height:11px; border-radius:3px; flex:none; }}
table {{ border-collapse:collapse; width:100%; font-size:12.5px; }}
th, td {{ text-align:left; padding:5px 10px 5px 0; border-bottom:1px solid var(--grid); }}
th {{ color:var(--text-secondary); font-weight:600; }}
td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.cross {{ stroke:var(--axis); stroke-width:1; }}
#tip {{ position:fixed; pointer-events:none; opacity:0; transition:opacity .08s;
  background:var(--surface-1); border:1px solid var(--border); border-radius:8px;
  padding:7px 10px; font:12px/1.5 system-ui,sans-serif; color:var(--text-primary);
  box-shadow:0 4px 14px rgba(0,0,0,.16); z-index:9; white-space:nowrap; }}
.note {{ color:var(--text-secondary); font-size:12.5px; margin:8px 0 0; }}
</style>
<div class="viz-root" data-palette="#2a78d6,#eb6834,#1baf7a,#eda100">

<h1>Phase 7 walk-forward result</h1>
<p class="sub">Pooled across {len(result.per_symbol_expectancy)} symbols &middot;
{_day_label(0, first_day_ms)} to {_day_label(n_days - 1, first_day_ms)} &middot;
{n_days} days &middot; {len(folds)} folds &middot; fees, slippage and funding charged on every run</p>
<div class="verdict">THE GATE: <span class="chip {verdict_cls}">{verdict_mark}</span>
<span style="font-weight:400;color:var(--text-secondary)">&mdash;
{n_fail} of {len(rows)} conditions unmet</span></div>
<p class="cap">Selected parameters (median-low of fold winners): {html.escape(sel)}</p>

<div class="tiles">{tile_html}</div>

<h2>Gate scorecard</h2>
<p class="cap">All conditions are mandatory (AND, not average).</p>
<div class="card"><table>
<thead><tr><th>Condition</th><th class=num>Measured</th>
<th class=num>Threshold</th><th>Verdict</th></tr></thead>
<tbody>{sc_html}</tbody></table></div>

<h2>Equity curves at the selected parameters</h2>
<p class="cap">Compounded, equal notional per trade, starting at 1.00&times;. The
shaded band is the one-shot out-of-sample holdout, excluded from all tuning.
Hover for values.</p>
<div class="card">
<svg id="eqsvg" viewBox="0 0 {W} {H}" role="img"
     aria-label="Compounded equity curves, pooled and per symbol">
{grid}{oos_band}{baseline}{xlab}{ylab}
{"".join(lines)}
{"".join(direct)}
<g id="eqcross"></g>
<rect id="eqhit" x="{ML}" y="{MT}" width="{PW}" height="{PH}" fill="transparent"/>
</svg>
<div class="legend">{legend_series}
<span class="lg"><span class="sw" style="background:var(--oos);border:1px solid var(--s1)"></span>
one-shot OOS</span></div>
</div>

<h2>Pooled drawdown</h2>
<p class="cap">Underwater plot on the compounded pooled curve. Deeper is worse.</p>
<div class="card">
<svg id="dsvg" viewBox="0 0 {W} {DH}" role="img" aria-label="Pooled underwater drawdown">
{d_oos}{d_grid}
<path class="uw" d="{_area_path(uw, x_of, dy_of, MT)}"/>
<g id="dcross"></g>
<rect id="dhit" x="{ML}" y="{MT}" width="{PW}" height="{dPH}" fill="transparent"/>
</svg>
</div>

<h2>Per-fold expectancy: train versus test</h2>
<p class="cap">What the tuning saw (train) against what happened next (test),
as per-trade expectancy. Folds marked * had no combo reach WF_MIN_TRADES and fell
back to config defaults. Hover a bar for its value.</p>
<div class="card">
<svg viewBox="0 0 {FW} {FH}" role="img" aria-label="Per-fold train and test expectancy">
{f_grid}
<line class="baseline" x1="{fML}" y1="{zero_y:.1f}" x2="{fML + fPW}" y2="{zero_y:.1f}"/>
{"".join(fold_bars)}{"".join(fold_labels)}
<text class="tick" x="{fML + fPW / 2:.0f}" y="{FH - 16}" text-anchor="middle">fold</text>
</svg>
<div class="legend">
<span class="lg"><span class="sw" style="background:var(--s1)"></span>train expectancy (what tuning saw)</span>
<span class="lg"><span class="sw" style="background:var(--s2)"></span>test expectancy (what followed)</span>
</div>
<p class="note"><strong>{neg_folds} of {len(folds)}</strong> folds had negative test
expectancy and <strong>{fallbacks} of {len(folds)}</strong> fell back to defaults for
want of trades. The fold machinery is noisy; the favourable holdout is one 90-day
window that went well, not a demonstrated edge.</p>
</div>

<h2>Per-symbol out-of-sample</h2>
<div class="card"><table>
<thead><tr><th>Symbol</th><th class=num>Full-span trades</th>
<th class=num>OOS expectancy / trade</th><th>Verdict</th></tr></thead>
<tbody>{ps_html}</tbody></table></div>

<h2>Fold detail</h2>
<div class="card"><table>
<thead><tr><th>Fold</th><th>Train start</th><th class=num>Train exp.</th>
<th class=num>Test exp.</th><th class=num>Test trades</th>
<th class=num>Pos. neighbours</th></tr></thead>
<tbody>{fold_rows}</tbody></table></div>

<h2>Table view &mdash; equity, quarterly samples</h2>
<p class="cap">The same curves as numbers, for the non-visual path and for readers
whom the lighter series colours fail.</p>
<div class="card"><table>
<thead><tr><th>Date</th>{head_series}<th class=num>Pooled DD</th></tr></thead>
<tbody>{tv_html}</tbody></table></div>

<h2>Full-span totals at the selected parameters</h2>
<div class="card"><table><tbody>
<tr><td>Trades (pooled, full span)</td><td class=num>{pooled['n_trades']}</td></tr>
<tr><td>Win rate</td><td class=num>{_fmt_pct(pooled['win_rate'])}</td></tr>
<tr><td>Expectancy / trade</td><td class=num>{_fmt_pct(pooled['expectancy_pct'], 4, True)}</td></tr>
<tr><td>Profit factor</td><td class=num>{_fmt_num(pooled['profit_factor'])}</td></tr>
<tr><td>Final pooled equity</td><td class=num>{curves['Pooled'][-1]:.3f}&times;</td></tr>
<tr><td>Worst pooled drawdown</td><td class=num>-{max(uw) * 100:.2f}%</td></tr>
</tbody></table></div>

<p class="note">Generated by <code>scripts/build_performance_chart.py</code>.
Self-contained: no network requests, no external assets. Colours are the validated
reference categorical palette, slots 1&ndash;4, with direct labels and a table view
as the documented relief for the two light-mode slots below 3:1 against the surface.</p>
</div>
<div id="tip" role="status" aria-live="polite"></div>
<script>
const D = {series_json};
const tip = document.getElementById('tip');
function dayLabel(i) {{
  return new Date(D.first + i * 86400000).toISOString().slice(0, 10);
}}
function wire(svgId, hitId, crossId, rows) {{
  const svg = document.getElementById(svgId);
  const hit = document.getElementById(hitId);
  const cross = document.getElementById(crossId);
  hit.addEventListener('mousemove', ev => {{
    const box = svg.getBoundingClientRect();
    const vb = svg.viewBox.baseVal;
    const vx = (ev.clientX - box.left) / box.width * vb.width;
    let i = Math.round((vx - D.ml) / D.pw * (D.n - 1));
    i = Math.max(0, Math.min(D.n - 1, i));
    const hb = hit.getBBox();
    const x = D.ml + D.pw * i / (D.n - 1);
    cross.innerHTML = '<line class="cross" x1="' + x + '" y1="' + hb.y +
      '" x2="' + x + '" y2="' + (hb.y + hb.height) + '"/>';
    tip.innerHTML = '<strong>' + dayLabel(i) + '</strong><br>' + rows(i);
    tip.style.opacity = 1;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let left = ev.clientX + 14;
    if (left + tw > window.innerWidth - 8) left = ev.clientX - tw - 14;
    let top = ev.clientY - th - 12;
    if (top < 8) top = ev.clientY + 16;
    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
  }});
  hit.addEventListener('mouseleave', () => {{
    tip.style.opacity = 0;
    cross.innerHTML = '';
  }});
}}
wire('eqsvg', 'eqhit', 'eqcross', i => D.names.map((n, k) =>
  '<span style="color:var(--s' + (k + 1) + ')">\\u25a0</span> ' + n + ' ' +
  D.curves[k][i].toFixed(3) + '\\u00d7').join('<br>'));
wire('dsvg', 'dhit', 'dcross', i =>
  'Pooled drawdown -' + (D.uw[i] * 100).toFixed(2) + '%');
</script>
</html>
"""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="Output HTML path")
    ap.add_argument("--db", default=config.DB_PATH)
    ap.add_argument("--symbols", default=",".join(config.SYMBOLS))
    ap.add_argument(
        "--start", default="2023-07-27",
        help="UTC start YYYY-MM-DD; default is after the 1D regime warmup",
    )
    ap.add_argument("--end", default=None, help="UTC end YYYY-MM-DD (default: last stored day)")
    args = ap.parse_args()

    symbols = tuple(s for s in args.symbols.split(",") if s)
    conn = sqlite3.connect(args.db)
    start_ms = config.date_to_ms(args.start)
    if args.end:
        end_ms = config.date_to_ms(args.end)
    else:
        row = conn.execute(
            "select max(ts) from ohlcv where timeframe = ?", (config.REGIME_TIMEFRAME,)
        ).fetchone()
        if not row or row[0] is None:
            logger.error("no %s candles stored", config.REGIME_TIMEFRAME)
            return 1
        end_ms = int(row[0]) + DAY_MS

    data = collect(conn, symbols, start_ms, end_ms)
    conn.close()

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(build(data))
    r = data["result"]
    logger.info(
        "wrote %s — gate %s, OOS trades %d, sharpe %s, dsr %s",
        args.out, "PASS" if r.passed else "FAIL", r.oos_metrics["n_trades"],
        _fmt_num(r.oos_equity["sharpe"]), _fmt_num(r.oos_equity["dsr"], 4),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
