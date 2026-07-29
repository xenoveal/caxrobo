/* trading_bot dashboard client.
 *
 * No dependencies, no build step, no external request — a constraint inherited
 * from contract §10 Q6 (stdlib server, zero new deps). Charts are hand-drawn
 * inline SVG.
 *
 * HONESTY RULES THIS FILE MUST NOT BREAK (contract §4):
 *  - The gate verdict is READ from `payload.gate`, never recomputed here. A page
 *    that re-derived PASS/FAIL could disagree with the engine.
 *  - A benchmark comparison always shows the benchmark's OWN absolute number
 *    beside it. "Beats benchmark" against a basket that lost money means only
 *    "lost less than holding", and the UI must say so.
 *  - DSR is never shown without the trial count it was charged.
 *  - An in-sample diagnostic is labelled as not evidence.
 */
(function () {
  "use strict";

  var STAGES = [];          // composer pipeline
  var META = {};            // graph.meta round-trip (currently: evo.eligible_detectors)
  var PLUGINS = {};         // kind -> [plugin]
  var CURRENT_EDITABLE = true;
  var RUNS = [];            // run summaries, newest first
  var ACTIVE_RUN = null;    // run_id
  var WATCHED = {};         // run_id -> EventSource
  var LOGS = {};            // run_id -> accumulated stdout (per run, never shared)
  var LOG_DONE = {};        // run_id -> true once its stream reached a terminal state
  var LAST_PAYLOAD = null;  // active run's result payload
  var CONFIG = {};          // /api/config, cached so the evolution symbol/timeframe
                             // selects never hardcode a value
  var KINDS = ["data", "detector", "confirmation", "policy", "filter"];

  // ---------------------------------------------------------------- helpers
  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }
  function clear(node) { while (node && node.firstChild) node.removeChild(node.firstChild); }

  function api(method, path, body) {
    var opts = { method: method, headers: {} };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    return fetch(path, opts).then(function (resp) {
      return resp.text().then(function (txt) {
        var data = null;
        try { data = txt ? JSON.parse(txt) : null; } catch (e) { data = null; }
        if (!resp.ok) {
          var msg = (data && (data.error || data.message)) || (resp.status + " " + resp.statusText);
          throw new Error(msg);
        }
        return data;
      });
    });
  }

  function showError(msg) {
    var box = $("error");
    clear(box);
    box.appendChild(el("span", null, msg));
    var b = el("button", "sm", "dismiss");
    b.onclick = clearError;
    box.appendChild(b);
    box.hidden = false;
  }
  function clearError() { $("error").hidden = true; }

  // number formatting — tabular, explicit sign where the sign is the point
  function fmt(v, dp) {
    if (v === null || v === undefined || v === "") return "—";
    var n = Number(v);
    if (isNaN(n) || !isFinite(n)) return "—";
    return n.toFixed(dp === undefined ? 2 : dp);
  }
  function pct(v, dp) {
    if (v === null || v === undefined || !isFinite(Number(v))) return "—";
    return fmt(v, dp === undefined ? 2 : dp) + "%";
  }
  function signed(v, dp) {
    if (v === null || v === undefined || !isFinite(Number(v))) return "—";
    var n = Number(v);
    return (n > 0 ? "+" : "") + fmt(n, dp === undefined ? 2 : dp);
  }
  function signedPct(v, dp) {
    if (v === null || v === undefined || !isFinite(Number(v))) return "—";
    return signed(v, dp) + "%";
  }
  function day(ms) {
    if (!ms) return "—";
    var d = new Date(Number(ms));
    if (isNaN(d.getTime())) return "—";
    return d.toISOString().slice(0, 10);
  }
  function polarity(v) {
    if (v === null || v === undefined || !isFinite(Number(v))) return "";
    return Number(v) > 0 ? "pos" : (Number(v) < 0 ? "neg" : "");
  }

  /* THE `_pct` FIELDS ARE FRACTIONS, NOT PERCENTAGES, despite the suffix.
   * metrics.py stores `win_rate = len(wins)/n` and `expectancy_pct = sum/n`,
   * and cli.py renders them with `%` format specs (".2%", ".4%") which multiply
   * by 100. So 0.3498 means 34.98% and 0.922 means 92.2%. Treating them as
   * already-percent understates every magnitude by 100x — a UI that lies.
   * Fractional: win_rate, expectancy_pct, avg_win_pct, avg_loss_pct,
   *             max_drawdown_pct, ann_return_pct, pnl_pct, per-symbol expectancy.
   * NOT fractional: sharpe, sortino, dsr, profit_factor, skew, kurtosis,
   *             planned_rr, total_return, n_trades, n_days.
   */
  function fp(v, dp) {   // fraction -> signed percent
    if (v === null || v === undefined || !isFinite(Number(v))) return "—";
    return signed(Number(v) * 100, dp === undefined ? 2 : dp) + "%";
  }
  function fpa(v, dp) {  // fraction -> unsigned percent
    if (v === null || v === undefined || !isFinite(Number(v))) return "—";
    return fmt(Number(v) * 100, dp === undefined ? 2 : dp) + "%";
  }
  function firstFinite() {
    for (var i = 0; i < arguments.length; i++) {
      var v = arguments[i];
      if (v !== undefined && v !== null && v !== "" && isFinite(Number(v))) return Number(v);
    }
    return null;
  }

  function table(cols, rows) {
    var t = el("table");
    var thead = el("thead"), tr = el("tr");
    cols.forEach(function (c) { tr.appendChild(el("th", c.num ? "num" : null, c.label)); });
    thead.appendChild(tr);
    t.appendChild(thead);
    var tb = el("tbody");
    rows.forEach(function (r) {
      var row = el("tr");
      cols.forEach(function (c) {
        var v = r[c.key];
        var td = el("td", (c.num ? "num " : "") + (c.cls || ""));
        if (v instanceof Node) td.appendChild(v);
        else td.textContent = (v === undefined || v === null || v === "") ? "—" : String(v);
        row.appendChild(td);
      });
      tb.appendChild(row);
    });
    t.appendChild(tb);
    return t;
  }

  // --------------------------------------------------------------- tooltip
  function tipShow(html, x, y) {
    var TIP = $("tip");
    TIP.innerHTML = html;
    TIP.hidden = false;
    TIP.style.opacity = "1";
    var r = TIP.getBoundingClientRect();
    var left = x + 14, top = y - r.height - 10;
    if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
    if (top < 8) top = y + 16;
    TIP.style.left = left + "px";
    TIP.style.top = top + "px";
  }
  function tipHide() {
    var TIP = $("tip");
    TIP.style.opacity = "0";
    TIP.hidden = true;
  }

  // ------------------------------------------------------------ SVG charts
  var SVGNS = "http://www.w3.org/2000/svg";
  function svgEl(tag, attrs) {
    var n = document.createElementNS(SVGNS, tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    return n;
  }
  function svgText(x, y, str, cls, anchor) {
    var t = svgEl("text", { x: x, y: y, class: cls || "tick" });
    if (anchor) t.setAttribute("text-anchor", anchor);
    t.textContent = str;
    return t;
  }
  function niceTicks(lo, hi, count) {
    if (!isFinite(lo) || !isFinite(hi) || lo === hi) return [lo];
    var span = hi - lo, raw = span / (count || 4);
    var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var norm = raw / mag, step;
    if (norm < 1.5) step = 1; else if (norm < 3) step = 2; else if (norm < 7) step = 5; else step = 10;
    step *= mag;
    var out = [], v = Math.ceil(lo / step) * step;
    for (var guard = 0; v <= hi + step * 0.001 && guard < 200; v += step, guard++) out.push(v);
    return out;
  }

  /* Line chart: 1-2 series over a daily index. 2px strokes, recessive grid, a
     dashed 1.0 baseline, crosshair + tooltip, and DIRECT END LABELS so identity
     is never colour-alone (the light-mode contrast WARN requires that relief). */
  function lineChart(host, series, startMs, opts) {
    clear(host);
    opts = opts || {};
    var live = series.filter(function (s) { return s.data && s.data.length > 1; });
    if (!live.length) { host.appendChild(el("p", "empty", "No series to plot.")); return; }

    var W = 900, H = opts.height || 260;
    var m = { t: 14, r: 62, b: 26, l: 54 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var n = Math.max.apply(null, live.map(function (s) { return s.data.length; }));
    var lo = Infinity, hi = -Infinity;
    live.forEach(function (s) {
      s.data.forEach(function (v) { if (v < lo) lo = v; if (v > hi) hi = v; });
    });
    if (opts.includeOne) { lo = Math.min(lo, 1); hi = Math.max(hi, 1); }
    if (!isFinite(lo) || !isFinite(hi)) {
      host.appendChild(el("p", "empty", "Series has no finite values."));
      return;
    }
    var padv = (hi - lo) * 0.08 || 0.05;
    lo -= padv; hi += padv;

    var X = function (i) { return m.l + (n <= 1 ? 0 : (i / (n - 1)) * iw); };
    var Y = function (v) { return m.t + ih - ((v - lo) / (hi - lo)) * ih; };

    var svg = svgEl("svg", { class: "plot", viewBox: "0 0 " + W + " " + H,
                             preserveAspectRatio: "none", role: "img" });
    niceTicks(lo, hi, 4).forEach(function (v) {
      if (v < lo || v > hi) return;
      svg.appendChild(svgEl("line", { class: "grid-line", x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v) }));
      svg.appendChild(svgText(m.l - 8, Y(v) + 3.5,
        opts.asPct ? signed((v - 1) * 100, 0) + "%" : fmt(v, 2), "tick", "end"));
    });
    if (lo <= 1 && hi >= 1) {
      svg.appendChild(svgEl("line", { class: "baseline", x1: m.l, x2: m.l + iw, y1: Y(1), y2: Y(1) }));
    }
    [0, Math.floor((n - 1) / 2), n - 1].forEach(function (i, k) {
      if (!startMs) return;
      svg.appendChild(svgText(X(i), H - 8, day(Number(startMs) + i * 86400000), "tick",
        k === 0 ? "start" : (k === 2 ? "end" : "middle")));
    });
    svg.appendChild(svgEl("line", { class: "axis-line", x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }));

    live.forEach(function (s) {
      svg.appendChild(svgEl("path", { class: "series " + s.cls,
        d: s.data.map(function (v, i) {
          return (i ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1);
        }).join(" ") }));
      var lastI = s.data.length - 1, lastV = s.data[lastI];
      svg.appendChild(svgText(X(lastI) + 7, Y(lastV) + 3.5,
        signed((lastV - 1) * 100, 1) + "%", "dlabel " + s.cls + "t", "start"));
    });

    var cross = svgEl("line", { class: "crosshair", y1: m.t, y2: m.t + ih, x1: m.l, x2: m.l });
    cross.style.opacity = "0";
    svg.appendChild(cross);
    var hit = svgEl("rect", { class: "hit", x: m.l, y: m.t, width: iw, height: ih });
    svg.appendChild(hit);
    hit.addEventListener("mousemove", function (ev) {
      var box = svg.getBoundingClientRect();
      var i = Math.round((((ev.clientX - box.left) / box.width * W) - m.l) / iw * (n - 1));
      if (!isFinite(i) || i < 0) i = 0;
      if (i > n - 1) i = n - 1;
      cross.setAttribute("x1", X(i));
      cross.setAttribute("x2", X(i));
      cross.style.opacity = "1";
      var rows = live.map(function (s) {
        var v = s.data[Math.min(i, s.data.length - 1)];
        return '<div><span style="color:var(--' + s.cls + ')">&#9632;</span> ' + s.label +
               " <b>" + signed((v - 1) * 100, 2) + "%</b></div>";
      }).join("");
      var lab = startMs ? day(Number(startMs) + i * 86400000) : ("day " + i);
      tipShow('<div style="color:var(--muted);margin-bottom:3px">' + lab + "</div>" + rows,
              ev.clientX, ev.clientY);
    });
    hit.addEventListener("mouseleave", function () { cross.style.opacity = "0"; tipHide(); });
    host.appendChild(svg);
  }

  /* Underwater: drawdown-from-peak as a filled area. ONE series, so no legend
     box — the panel title names it (dataviz: a single series needs none). */
  function underwaterChart(host, equity, startMs) {
    clear(host);
    if (!equity || equity.length < 2) { host.appendChild(el("p", "empty", "No equity series.")); return; }
    var peak = -Infinity;
    var dd = equity.map(function (v) { peak = Math.max(peak, v); return (v / peak - 1) * 100; });
    // NARROW viewBox on purpose: this panel is half-width, and a 900-wide box
    // scaled into ~500px shrinks every tick label below legibility.
    var W = 470, H = 210, m = { t: 14, r: 14, b: 26, l: 42 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var lo = Math.min.apply(null, dd);
    if (!isFinite(lo) || lo > -1) lo = -1;
    var n = dd.length;
    var X = function (i) { return m.l + (i / (n - 1)) * iw; };
    var Y = function (v) { return m.t + (v / lo) * ih; };

    var svg = svgEl("svg", { class: "plot", viewBox: "0 0 " + W + " " + H,
                             preserveAspectRatio: "none", role: "img" });
    niceTicks(lo, 0, 4).forEach(function (v) {
      if (v > 0 || v < lo) return;
      svg.appendChild(svgEl("line", { class: "grid-line", x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v) }));
      svg.appendChild(svgText(m.l - 8, Y(v) + 3.5, fmt(v, 0) + "%", "tick", "end"));
    });
    var area = "M" + X(0) + " " + Y(0);
    dd.forEach(function (v, i) { area += " L" + X(i).toFixed(1) + " " + Y(v).toFixed(1); });
    area += " L" + X(n - 1) + " " + Y(0) + " Z";
    svg.appendChild(svgEl("path", { class: "fill-crit", d: area }));
    svg.appendChild(svgEl("path", { class: "series stroke-crit",
      d: dd.map(function (v, i) {
        return (i ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1);
      }).join(" ") }));
    svg.appendChild(svgEl("line", { class: "axis-line", x1: m.l, x2: m.l + iw, y1: Y(0), y2: Y(0) }));

    // Label ABOVE the trough: below it collided with the series stroke and the
    // date axis. The fill is only 16% opacity, so primary ink reads over it.
    var worstV = Math.min.apply(null, dd), worst = dd.indexOf(worstV);
    svg.appendChild(svgText(X(worst), Math.max(m.t + 11, Y(worstV) - 7),
      fmt(worstV, 1) + "%", "dlabel", "middle"));
    [0, n - 1].forEach(function (i, k) {
      if (!startMs) return;
      svg.appendChild(svgText(X(i), H - 8, day(Number(startMs) + i * 86400000), "tick", k ? "end" : "start"));
    });
    host.appendChild(svg);
  }

  /* Trade P&L histogram. Sign is carried positionally (a zero rule) as well as
     by status colour, so it is never colour-alone. */
  function pnlHistogram(host, pnls) {
    clear(host);
    pnls = (pnls || []).filter(function (v) { return isFinite(v); });
    if (!pnls.length) { host.appendChild(el("p", "empty", "No closed trades.")); return; }
    var lo = Math.min.apply(null, pnls), hi = Math.max.apply(null, pnls);
    if (lo === hi) { lo -= 1; hi += 1; }
    var BINS = Math.min(24, Math.max(8, Math.round(Math.sqrt(pnls.length) * 1.5)));
    var w = (hi - lo) / BINS;
    var counts = [];
    for (var z = 0; z < BINS; z++) counts.push(0);
    pnls.forEach(function (v) {
      var b = Math.floor((v - lo) / w);
      if (b >= BINS) b = BINS - 1;
      if (b < 0) b = 0;
      counts[b]++;
    });
    var maxC = Math.max.apply(null, counts);
    var W = 460, H = 200, m = { t: 12, r: 12, b: 30, l: 40 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var svg = svgEl("svg", { class: "plot", viewBox: "0 0 " + W + " " + H,
                             preserveAspectRatio: "none", role: "img" });

    niceTicks(0, maxC, 3).forEach(function (c) {
      if (c < 0 || c > maxC) return;
      var y = m.t + ih - (c / maxC) * ih;
      svg.appendChild(svgEl("line", { class: "grid-line", x1: m.l, x2: m.l + iw, y1: y, y2: y }));
      svg.appendChild(svgText(m.l - 7, y + 3.5, fmt(c, 0), "tick", "end"));
    });

    var bw = iw / BINS;
    counts.forEach(function (c, i) {
      if (!c) return;
      var x = m.l + i * bw, h = (c / maxC) * ih;
      var mid = lo + (i + 0.5) * w;
      var r = svgEl("rect", { class: mid >= 0 ? "bar-pos" : "bar-neg",
        x: (x + 1).toFixed(1), y: (m.t + ih - h).toFixed(1),
        width: Math.max(1, bw - 2).toFixed(1), height: h.toFixed(1),
        rx: Math.min(4, Math.max(0, bw / 2 - 1)) });
      r.addEventListener("mousemove", function (ev) {
        tipShow("<b>" + c + "</b> trade" + (c === 1 ? "" : "s") + "<br>" +
                '<span style="color:var(--muted)">' + signed(lo + i * w, 2) + "% &hellip; " +
                signed(lo + (i + 1) * w, 2) + "%</span>", ev.clientX, ev.clientY);
      });
      r.addEventListener("mouseleave", tipHide);
      svg.appendChild(r);
    });

    if (lo < 0 && hi > 0) {
      var zx = m.l + ((0 - lo) / (hi - lo)) * iw;
      svg.appendChild(svgEl("line", { class: "baseline", x1: zx, x2: zx, y1: m.t, y2: m.t + ih }));
    }
    svg.appendChild(svgEl("line", { class: "axis-line", x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }));
    svg.appendChild(svgText(m.l, H - 8, signed(lo, 1) + "%", "tick", "start"));
    svg.appendChild(svgText(m.l + iw, H - 8, signed(hi, 1) + "%", "tick", "end"));
    host.appendChild(svg);

    var wins = pnls.filter(function (v) { return v > 0; }).length;
    var cap = el("p", "note", pnls.length + " trades · " + wins + " up / " +
      (pnls.length - wins) + " down · median " + signed(median(pnls), 2) + "%");
    cap.style.margin = "6px 0 0";
    host.appendChild(cap);
  }

  function median(a) {
    var s = a.slice().sort(function (x, y) { return x - y; });
    var h = Math.floor(s.length / 2);
    return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
  }

  /* Monthly returns heatmap. DIVERGING: two hues + a NEUTRAL GRAY midpoint,
     never a rainbow. Every cell also carries its number, which is the relief
     the light-mode contrast WARN requires. */
  function monthlyHeatmap(host, equity, startMs) {
    clear(host);
    if (!equity || equity.length < 2 || !startMs) {
      host.appendChild(el("p", "empty", "Not enough history for a monthly view."));
      return;
    }
    var buckets = {};
    for (var i = 0; i < equity.length; i++) {
      var d = new Date(Number(startMs) + i * 86400000);
      var k = d.getUTCFullYear() + "-" + d.getUTCMonth();
      if (!buckets[k]) buckets[k] = { first: equity[i], last: equity[i] };
      buckets[k].last = equity[i];
    }
    var years = {}, maxAbs = 0;
    Object.keys(buckets).forEach(function (k) {
      var p = k.split("-"), y = +p[0], mo = +p[1];
      var b = buckets[k];
      if (!b.first) return;
      var r = (b.last / b.first - 1) * 100;
      if (!isFinite(r)) return;
      if (!years[y]) years[y] = {};
      years[y][mo] = r;
      maxAbs = Math.max(maxAbs, Math.abs(r));
    });
    var ys = Object.keys(years).sort();
    if (!ys.length) { host.appendChild(el("p", "empty", "No monthly buckets.")); return; }
    if (!maxAbs) maxAbs = 1;

    var MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    var t = el("table", "heat");
    var thead = el("thead"), hr = el("tr");
    hr.appendChild(el("th", null, ""));
    MON.forEach(function (mn) { hr.appendChild(el("th", null, mn)); });
    hr.appendChild(el("th", null, "Year"));
    thead.appendChild(hr);
    t.appendChild(thead);

    var tb = el("tbody");
    ys.forEach(function (y) {
      var tr = el("tr");
      tr.appendChild(el("th", null, y));
      var compound = 1, any = false;
      for (var mo = 0; mo < 12; mo++) {
        var v = years[y][mo];
        if (v === undefined) { tr.appendChild(el("td", "void", "·")); continue; }
        any = true;
        compound *= 1 + v / 100;
        var td = el("td", null, signed(v, 1));
        var mag = Math.min(1, Math.abs(v) / maxAbs);
        var mix = Math.round((0.14 + 0.66 * mag) * 100);
        td.style.background = Math.abs(v) < 1e-9
          ? "var(--grid)"
          : "color-mix(in srgb, var(--" + (v > 0 ? "good" : "crit") + ") " + mix + "%, var(--surface-2))";
        td.title = MON[mo] + " " + y + ": " + signed(v, 2) + "%";
        tr.appendChild(td);
      }
      var yt = el("td", null, any ? signed((compound - 1) * 100, 1) : "—");
      yt.style.fontWeight = "700";
      yt.style.background = "var(--surface-2)";
      tr.appendChild(yt);
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    host.appendChild(t);
    host.appendChild(el("p", "note",
      "Compounded within each calendar month. Colour scale is symmetric to ±" +
      fmt(maxAbs, 1) + "%; the neutral cell is zero."));
  }

  // ------------------------------------------------------------- composer
  function pluginsOfKind(kind) { return PLUGINS[kind] || []; }

  function renderPipe() {
    // Called first, unconditionally: the eligibility panel's force-checked set
    // depends on STAGES, and renderPipe has an early return below (empty
    // pipeline) that must not skip this recompute.
    renderEligibility();
    var host = $("pipe");
    clear(host);
    var chip = $("editable-chip");
    clear(chip);
    if (!CURRENT_EDITABLE) {
      var c = el("span", "chip warn", "read-only");
      c.title = "This graph has more than one branch, which the linear composer cannot represent. " +
                "It is shown read-only rather than silently flattened and re-saved.";
      chip.appendChild(c);
    }
    if (!STAGES.length) {
      host.appendChild(el("p", "empty", CURRENT_EDITABLE
        ? "Empty pipeline. Add a detector to begin."
        : "This strategy has multiple branches and cannot be shown in the linear composer."));
      return;
    }

    // "at least one" bookkeeping, not "exactly one" -- need[k] only tracks
    // whether ANY stage of that kind is present, so it already tolerates
    // multiple detector stages (the server enforces >=1, never ==1). Nothing
    // here assumed a single detector; the actual single-detector gap was the
    // insertion index in addStage (see below), not this presence check.
    var need = { data: 1, detector: 1, policy: 1 };
    STAGES.forEach(function (st) { if (need[st.kind]) need[st.kind] = 0; });

    STAGES.forEach(function (st, idx) {
      if (idx) host.appendChild(el("div", "arrow", "↓"));
      var box = el("div", "stage k-" + st.kind);
      var head = el("div", "head");
      head.appendChild(el("span", "kind", st.kind));

      var sel = el("select");
      var opts = pluginsOfKind(st.kind);
      if (!opts.length) {
        var only = el("option", null, st.key);
        only.value = st.key;
        sel.appendChild(only);
      }
      opts.forEach(function (p) {
        var o = el("option", null, p.key.replace(/^[a-z]+\./, ""));
        o.value = p.key;
        if (p.key === st.key) o.selected = true;
        sel.appendChild(o);
      });
      sel.disabled = !CURRENT_EDITABLE;
      sel.onchange = function () { st.key = sel.value; st.params = {}; renderPipe(); };
      head.appendChild(sel);
      head.appendChild(el("span", "spacer"));

      if (CURRENT_EDITABLE && st.kind !== "data" && st.kind !== "policy") {
        var rm = el("button", "sm ghost", "remove");
        rm.onclick = function () { STAGES.splice(idx, 1); renderPipe(); };
        head.appendChild(rm);
      }
      box.appendChild(head);

      var meta = opts.filter(function (p) { return p.key === st.key; })[0];
      if (meta && meta.rationale) box.appendChild(el("p", "rationale", meta.rationale));
      // `params` is a DICT keyed by name -> {kind, default, doc, min, max, step}.
      var pnames = (meta && meta.params) ? Object.keys(meta.params) : [];
      if (pnames.length) {
        var pg = el("div", "params");
        pnames.forEach(function (pname) {
          var ps = meta.params[pname] || {};
          var f = el("div", "field");
          var iid = "p-" + idx + "-" + pname;
          var lbl = el("label", null, pname);
          lbl.setAttribute("for", iid);
          // The registry declares FOUR param kinds — int, float, choice, bool.
          // Rendering them all as type=number blanks the non-numeric ones.
          var cur = st.params[pname];
          var dflt = (ps.default === undefined || ps.default === null) ? "" : ps.default;
          var inp;
          if (ps.kind === "choice") {
            inp = el("select");
            (ps.choices || []).forEach(function (ch) {
              var o = el("option", null, ch);
              o.value = ch;
              if (String(cur !== undefined ? cur : dflt) === String(ch)) o.selected = true;
              inp.appendChild(o);
            });
            inp.onchange = function () { st.params[pname] = inp.value; };
          } else if (ps.kind === "bool") {
            inp = el("select");
            ["true", "false"].forEach(function (ch) {
              var o = el("option", null, ch);
              o.value = ch;
              if (String(cur !== undefined ? cur : dflt) === ch) o.selected = true;
              inp.appendChild(o);
            });
            inp.onchange = function () { st.params[pname] = inp.value === "true"; };
          } else {
            inp = el("input");
            inp.type = "number";
            if (ps.step !== undefined && ps.step !== null) inp.step = ps.step;
            if (ps.min !== undefined && ps.min !== null) inp.min = ps.min;
            if (ps.max !== undefined && ps.max !== null) inp.max = ps.max;
            inp.value = cur !== undefined ? cur : dflt;
            inp.oninput = function () {
              if (inp.value === "") delete st.params[pname];
              else st.params[pname] = Number(inp.value);
            };
          }
          inp.id = iid;
          inp.disabled = !CURRENT_EDITABLE;
          // The declared bound is an inventory of degrees of freedom, so surface it.
          var bits = [];
          if (ps.doc) bits.push(ps.doc);
          if (ps.min !== undefined && ps.min !== null) {
            bits.push("declared range " + ps.min + " … " + ps.max);
          }
          if (bits.length) inp.title = bits.join(" — ");
          f.appendChild(lbl);
          f.appendChild(inp);
          pg.appendChild(f);
        });
        box.appendChild(pg);
      }
      host.appendChild(box);
    });

    Object.keys(need).forEach(function (k) {
      if (!need[k]) return;
      var miss = el("div", "stage missing");
      miss.appendChild(el("span", "kind", k));
      miss.appendChild(el("span", null, " required, not present"));
      host.appendChild(miss);
    });
  }

  function addStage(kind) {
    var opts = pluginsOfKind(kind);
    if (!opts.length) { showError("no " + kind + " plug-ins are registered"); return; }
    var at = STAGES.length;
    if (kind === "detector") {
      // Detector stages are contiguous right after `data` (mirrors
      // _graph_to_stages' emission order: data, detector(s), confirmations,
      // policy, filters). Insert after the LAST existing detector/data stage.
      // Inserting at the policy boundary like confirmations do would land a
      // new detector AFTER any confirmations already present, which
      // _stages_to_graph would then read back as a non-contiguous detector
      // run on the next save->reload -- an order that Task 4's uniform-branch
      // check on the server would no longer recognise as editable.
      at = 0;
      for (var i = 0; i < STAGES.length; i++) {
        if (STAGES[i].kind === "data" || STAGES[i].kind === "detector") at = i + 1;
      }
    } else if (kind === "confirmation") {
      for (var j = 0; j < STAGES.length; j++) {
        if (STAGES[j].kind === "policy") { at = j; break; }
      }
    }
    STAGES.splice(at, 0, { kind: kind, key: opts[0].key, params: {} });
    renderPipe();
  }

  // --------------------------------------------------- evolution eligibility
  // Which detectors evolution's graph-edit mutator may swap in or add, stored
  // in graph.meta.evo.eligible_detectors (a list of "detector.<key>" strings).
  // ALL checked => the key is omitted entirely (no constraint), so a strategy
  // that never touches this panel saves byte-identical meta to before this
  // feature existed.
  function usedDetectorKeys() {
    return STAGES.filter(function (s) { return s.kind === "detector"; })
                 .map(function (s) { return s.key; });
  }

  function renderEligibility() {
    var host = $("evo-eligibility-body");
    if (!host) return;
    clear(host);
    var opts = pluginsOfKind("detector");
    if (!opts.length) {
      host.appendChild(el("p", "empty", "No detector plug-ins registered."));
      return;
    }
    var used = usedDetectorKeys();
    var cur = (META.evo && META.evo.eligible_detectors) || null;  // null = all eligible
    opts.forEach(function (p) {
      var forced = used.indexOf(p.key) !== -1;
      var checked = forced || !cur || cur.indexOf(p.key) !== -1;
      var row = el("label", "elig-item");
      var cb = el("input");
      cb.type = "checkbox";
      cb.checked = checked;
      cb.disabled = forced || !CURRENT_EDITABLE;
      if (forced) row.title = "in the strategy — always eligible to keep";
      cb.onchange = applyEligibility;
      row.appendChild(cb);
      row.appendChild(el("span", null, p.key.replace(/^[a-z]+\./, "")));
      host.appendChild(row);
    });
  }

  function applyEligibility() {
    var host = $("evo-eligibility-body");
    if (!host) return;
    var opts = pluginsOfKind("detector");
    var used = usedDetectorKeys();
    var boxes = host.querySelectorAll("input[type=checkbox]");
    var eligible = [];
    opts.forEach(function (p, i) {
      var forced = used.indexOf(p.key) !== -1;
      if (forced || (boxes[i] && boxes[i].checked)) eligible.push(p.key);
    });
    if (eligible.length >= opts.length) {
      // Every registered detector is eligible: store no constraint at all.
      if (META.evo) delete META.evo.eligible_detectors;
      if (META.evo && !Object.keys(META.evo).length) delete META.evo;
    } else {
      META.evo = META.evo || {};
      META.evo.eligible_detectors = eligible;
    }
  }

  // ------------------------------------------------------------- campaigns
  // A campaign is one strategy's evolution history (one strategy, many
  // campaigns). The Run card only ever offers campaigns belonging to the
  // strategy currently in the composer, because the server refuses a
  // mismatched pair and an unofferable choice is just a hidden 400.
  var CAMPAIGNS = [];
  var NEW_CAMPAIGN = "__new__";

  function campaignValue() {
    var sel = $("run-campaign-select");
    return sel.value === NEW_CAMPAIGN
      ? $("run-campaign").value.trim()
      : sel.value;
  }

  function runEvoFieldsSet(isNew) {
    // Population is fixed at campaign creation -- run_campaign silently
    // ignores population_size on a --resume, so offering the field there
    // would be a control that quietly does nothing. Generations stays live
    // either way, just meaning "how many total" for a new campaign versus
    // "how many MORE" (--extend) for an existing one.
    var popField = $("run-population").parentElement;
    popField.hidden = !isNew;
    $("run-population").disabled = !isNew;
    $("run-generations-label").textContent = isNew ? "Generations" : "Extend by";
    $("run-generations").placeholder = isNew ? "default" : "e.g. 100";
  }

  function renderCampaignStats() {
    var host = $("run-campaign-stats");
    clear(host);
    var sel = $("run-campaign-select");
    var isNew = sel.value === NEW_CAMPAIGN;
    $("run-campaign-new-field").hidden = !isNew;
    runEvoFieldsSet(isNew);
    if (isNew) {
      host.appendChild(el("span", null,
        "A new campaign starts a fresh search from the strategy above."));
      $("run-evo-settings-note").textContent = "Blank uses config defaults.";
      return;
    }
    var c = CAMPAIGNS.filter(function (x) { return x.label === sel.value; })[0];
    if (!c) return;
    host.appendChild(el("span", "chip plain",
      c.generations_done + " of " + c.generations + " generations"));
    host.appendChild(el("span", "chip plain", c.trials + " trials charged"));
    host.appendChild(el("span", "chip plain", c.status));
    host.appendChild(el("p", "note mono", "campaign id " + c.campaign_id));
    host.appendChild(el("p", "note",
      "Start evolution CONTINUES this campaign — it breeds on from the "
      + c.generations_done + " generations already paid for rather than "
      + "restarting."));
    $("run-evo-settings-note").textContent =
      "Population is locked to what this campaign started with. "
      + "“Extend by” is generations to run BEYOND the "
      + c.generations_done + " already completed — blank continues to "
      + "its current ceiling of " + c.generations + ".";
  }

  function fillCampaignSelect(sel, list, opts) {
    var keep = sel.value;
    clear(sel);
    if (opts && opts.allowNew) {
      var o = el("option", null, "+ New campaign…");
      o.value = NEW_CAMPAIGN;
      sel.appendChild(o);
    }
    var values = list.map(function (c) {
      var oc = el("option", null,
        c.label + "  (" + c.generations_done + " gens, " + c.trials + " trials)");
      oc.value = opts && opts.byId ? c.campaign_id : c.label;
      sel.appendChild(oc);
      return oc.value;
    });
    // Keep the operator's choice only if it still exists for this strategy.
    // NEW_CAMPAIGN must NOT be sticky: switching the composer to a strategy that
    // already has campaigns should surface them, which is the whole point of
    // binding the two — otherwise the existing history stays invisible.
    if (keep && values.indexOf(keep) !== -1) sel.value = keep;
    else if (values.length) sel.value = values[0];
    else if (sel.options.length) sel.value = sel.options[0].value;
  }

  function refreshCampaigns() {
    var strategy = $("strategy-name").value.trim();
    return api("GET", "/api/campaigns").then(function (all) {
      var every = (all && all.campaigns) || [];
      // Evolution page: every campaign, addressed by id (what /api/generations
      // keys on). Run card: only this strategy's, addressed by label.
      fillCampaignSelect($("generations-campaign-select"), every, { byId: true });
      CAMPAIGNS = every.filter(function (c) { return c.strategy_name === strategy; });
      fillCampaignSelect($("run-campaign-select"), CAMPAIGNS, { allowNew: true });
      renderCampaignStats();
    }).catch(function (e) { showError("campaigns: " + e.message); });
  }

  function loadStrategy(name) {
    if (!name) return;
    clearError();
    api("GET", "/api/strategies/" + encodeURIComponent(name)).then(function (d) {
      $("strategy-name").value = d.name || name;
      // The campaign list is strategy-scoped, so it must follow the composer.
      refreshCampaigns();
      CURRENT_EDITABLE = !!d.editable;
      STAGES = (d.stages || []).map(function (s) {
        return { kind: s.kind, key: s.key, params: Object.assign({}, s.params || {}) };
      });
      // Deep-copy: META is mutated in place by the eligibility checkboxes, and
      // must never alias the response object.
      META = (d.meta && typeof d.meta === "object") ? JSON.parse(JSON.stringify(d.meta)) : {};
      renderPipe();
      if (d.error) showError("strategy " + name + ": " + d.error);
    }).catch(function (e) { showError("load failed: " + e.message); });
  }

  function saveStrategy() {
    return api("POST", "/api/strategies/" + encodeURIComponent($("strategy-name").value.trim()),
               { stages: STAGES, meta: META });
  }

  function validateStrategy() {
    clearError();
    var out = $("validate-result");
    clear(out);
    api("POST", "/api/validate", { name: $("strategy-name").value.trim(), stages: STAGES, meta: META })
      .then(function (d) {
        var p = el("p", "note");
        if (d.valid) {
          out.appendChild(el("span", "chip good", "valid"));
          p.textContent = "graph hash " + String(d.graph_hash || "").slice(0, 12);
        } else {
          out.appendChild(el("span", "chip crit", "invalid"));
          p.textContent = d.error || "unknown reason";
        }
        out.appendChild(p);
      })
      .catch(function (e) {
        out.appendChild(el("span", "chip crit", "invalid"));
        out.appendChild(el("p", "note", e.message));
      });
  }

  // ------------------------------------------------------------------ runs
  function renderTabs() {
    var host = $("run-tabs");
    clear(host);
    if (!RUNS.length) {
      $("run-empty").hidden = false;
      $("run-detail").hidden = true;
      $("results-body").hidden = true;
      $("log-panel").hidden = true;
      return;
    }
    $("run-empty").hidden = true;
    // Shape note: status is an OBJECT ({state, tier, exit_code, ...}) and the
    // kind/strategy/campaign live under `cmd`, not on the run root.
    RUNS.slice(0, 12).forEach(function (r) {
      var cmd = r.cmd || {};
      var state = (r.status || {}).state || "";
      var b = el("button", "tab");
      b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", r.run_id === ACTIVE_RUN ? "true" : "false");
      b.appendChild(el("span", "led " + state));
      b.appendChild(el("span", null, (cmd.kind || "run") + " · " + (cmd.strategy || "?")));
      b.title = r.run_id + " — " + state +
                (cmd.campaign ? " — campaign " + cmd.campaign : "");
      b.onclick = function () { selectRun(r.run_id); };
      host.appendChild(b);
    });
  }

  function selectRun(rid) {
    ACTIVE_RUN = rid;
    renderTabs();
    $("run-detail").hidden = false;
    $("log-panel").hidden = false;
    renderLog();   // show THIS run's buffer immediately, not the last run's
    api("GET", "/api/runs/" + encodeURIComponent(rid)).then(function (d) {
      renderRun(d);
      watchRun(rid);   // replays a finished run's log too; no-op once complete
    }).catch(function (e) { showError("run " + rid + ": " + e.message); });
  }

  function statusChip(status) {
    var cls = status === "done" ? "good" : (status === "running" ? "info"
            : (status === "error" || status === "failed" ? "crit" : "plain"));
    return el("span", "chip " + cls, status || "unknown");
  }

  function renderRun(run) {
    var payload = run.result || run.payload || null;
    var cmd = run.cmd || {};
    var state = (run.status || {}).state || "unknown";
    LAST_PAYLOAD = payload;

    var v = $("verdict");
    clear(v);
    var gate = payload && payload.gate;
    if (gate && gate.conditions) {
      var passed = !!gate.passed;
      var nOk = gate.conditions.filter(function (c) { return c.ok; }).length;
      v.appendChild(el("div", "badge " + (passed ? "pass" : "fail"),
        passed ? "GATE PASS" : "GATE FAIL"));
      var tally = el("div", "tally");
      tally.appendChild(el("b", null, nOk + " of " + gate.conditions.length));
      tally.appendChild(el("span", null, " conditions met"));
      v.appendChild(tally);
      if (payload.n_trials_used !== undefined && payload.n_trials_used !== null) {
        var tr = el("span", "chip plain", payload.n_trials_used + " trials charged");
        tr.title = "DSR is deflated by this many evaluations. It grows for the life of the campaign.";
        v.appendChild(tr);
      }
      if (payload.campaign) v.appendChild(el("span", "chip plain", "campaign " + payload.campaign));
      if (cmd.campaign_id) {
        v.appendChild(el("span", "chip plain", "id " + cmd.campaign_id));
      }
    } else {
      var lab = el("div", "tally");
      lab.appendChild(el("b", null, (cmd.kind || "run") + " · " + (cmd.strategy || "?")));
      if (cmd.start_ms) {
        lab.appendChild(el("span", null, "  " + day(cmd.start_ms) + " → " + day(cmd.end_ms)));
      }
      v.appendChild(lab);
      v.appendChild(statusChip(state));
      if (cmd.symbols && cmd.symbols.length) {
        v.appendChild(el("span", "chip plain", cmd.symbols.length + " symbols"));
      }
      // An evolve run has no gate payload, so without this its campaign was
      // nameless on this panel — the run tab was the only place it appeared.
      if (cmd.campaign) v.appendChild(el("span", "chip plain", "campaign " + cmd.campaign));
      var evolvedId = cmd.campaign_id || (run.status || {}).campaign_id;
      if (evolvedId) v.appendChild(el("span", "chip plain", "id " + evolvedId));
      if ((run.status || {}).continued) {
        v.appendChild(el("span", "chip good", "continued"));
      }
    }
    // WHICH graph produced these numbers. A campaign that has evolved is scored
    // as its champion, not as the seed file on disk, and a verdict that did not
    // say so was indistinguishable from a re-run of generation zero.
    var src = payload && payload.graph_source;
    if (src && src.kind === "champion") {
      var ch = el("span", "chip good",
        "champion · gen " + src.gen_index);
      ch.title = "Evaluated the best member of campaign " + (src.campaign || "?") +
                 " (member " + (src.member_id || "?") + ", graph " +
                 String(src.graph_hash || "").slice(0, 8) +
                 "), not the saved seed graph.";
      v.appendChild(ch);
    } else if (src && src.kind === "seed") {
      var sd = el("span", "chip plain", "seed graph");
      sd.title = "This campaign has no scored member yet, so the saved strategy " +
                 "file was evaluated as-is.";
      v.appendChild(sd);
    }
    if (payload && payload.honesty) {
      var h = el("p", "callout");
      h.style.margin = "0 14px 12px";
      h.style.width = "100%";
      h.appendChild(el("strong", null, "Diagnostic only. "));
      h.appendChild(el("span", null, payload.honesty));
      v.appendChild(h);
    }

    var prog = $("run-progress");
    clear(prog);
    if (state === "running") {
      var bar = el("div", "bar indet");
      bar.appendChild(el("span"));
      bar.style.margin = "0 14px 12px";
      prog.appendChild(bar);
      $("btn-stop").hidden = false;
    } else {
      $("btn-stop").hidden = true;
    }

    // The log panel STAYS VISIBLE. It reports "no output was captured" when a
    // finished run genuinely produced none — hiding it was indistinguishable
    // from never having fetched the log at all.
    $("log-panel").hidden = false;

    renderTiles(payload);
    if (!payload) {
      // CLEAR, don't just hide. Hiding leaves the previous run's rows in the DOM,
      // so any later reveal would show another run's numbers under this header.
      ["gate-table", "per-symbol", "bench-table", "folds-table", "trades-table",
       "chart-equity", "chart-underwater", "chart-pnl", "heat-monthly",
       "legend-equity"].forEach(function (id) { clear($(id)); });
      $("trades-count").textContent = "";
      $("results-body").hidden = true;
      return;
    }
    $("results-body").hidden = false;
    renderGate(payload);
    renderCharts(payload);
    renderTables(payload);
  }

  function renderTiles(payload) {
    var host = $("tiles");
    clear(host);
    if (!payload) return;
    var m = payload.oos_metrics || payload.pooled_metrics || {};
    var pm = payload.pooled_metrics || {};
    // oos_equity FIRST. Sharpe, ann_return_pct and dsr are equity-path figures
    // and live ONLY on the equity dict — compute_metrics never produces them.
    // Reading pooled_equity alone left every gate run showing "—" for return
    // and Sharpe while the conditions table right below printed the real
    // numbers, because a gate payload has oos_equity and no pooled_equity.
    var eq = payload.oos_equity || payload.pooled_equity || {};

    var ann = firstFinite(m.ann_return_pct, eq.ann_return_pct);
    var sharpe = firstFinite(m.sharpe, eq.sharpe);
    // EQUITY drawdown first, deliberately. metrics.max_drawdown_pct is a
    // trade-sequence figure and equity.max_drawdown_pct is the daily-equity path
    // — they differ materially (0.9221 vs 0.5521 on the same run) and it is the
    // EQUITY one the gate scores and the underwater chart draws. Showing the
    // trade-sequence number beside that chart made the tile contradict the plot.
    var ddEquity = firstFinite(eq.max_drawdown_pct);
    var ddTrades = firstFinite(m.max_drawdown_pct, m.max_dd_pct);
    var dd = ddEquity !== null ? ddEquity : ddTrades;
    var wr = firstFinite(m.win_rate, pm.win_rate);
    var nt = firstFinite(m.n_trades, pm.n_trades);
    var exp = firstFinite(m.expectancy_pct, pm.expectancy_pct);
    var dsr = firstFinite(m.dsr, eq.deflated_sharpe, eq.dsr);

    var tiles = [
      { k: "Ann. return", v: fp(ann, 2), cls: polarity(ann) },
      { k: "Sharpe", v: signed(sharpe, 3), cls: polarity(sharpe) },
      { k: "Max drawdown", v: fpa(dd, 2),
        s: ddEquity !== null
           ? (ddTrades !== null ? "equity path · " + fpa(ddTrades, 2) + " by trade sequence"
                                : "equity path")
           : "trade sequence" },
      { k: "Expectancy / trade", v: fp(exp, 4), cls: polarity(exp) },
      { k: "Win rate", v: fpa(wr, 2) },
      { k: "Trades", v: nt === null ? "—" : String(nt) }
    ];
    // DSR is NEVER shown without the trial count it was charged (contract §4).
    if (dsr !== null) {
      tiles.push({ k: "DSR", v: fmt(dsr, 4),
        s: (payload.n_trials_used !== undefined && payload.n_trials_used !== null)
           ? "at " + payload.n_trials_used + " trials"
           : "in-sample, no trial charged" });
    }
    tiles.forEach(function (t) {
      var d = el("div", "tile");
      d.appendChild(el("span", "k", t.k));
      d.appendChild(el("span", "v " + (t.cls || ""), t.v));
      if (t.s) d.appendChild(el("span", "s", t.s));
      host.appendChild(d);
    });
  }

  function renderGate(payload) {
    var panel = $("panel-gate"), host = $("gate-table");
    clear(host);
    if (!payload.gate || !payload.gate.conditions) { panel.hidden = true; return; }
    panel.hidden = false;
    var basket = (payload.benchmark && payload.benchmark.basket) || null;
    var rows = payload.gate.conditions.map(function (c) {
      var extra = "";
      // A benchmark pass must NEVER appear without the basket's own absolute number.
      if (String(c.name).indexOf("beats_benchmark") === 0 && basket) {
        var isRet = String(c.name).indexOf("return") !== -1;
        var bv = isRet ? basket.ann_return_pct : basket.sharpe;
        if (bv !== undefined && bv !== null && isFinite(Number(bv)) && Number(bv) < 0) {
          extra = "basket itself " + (isRet ? signedPct(bv, 2) : signed(bv, 3)) +
                  " — beating it means losing less than holding";
        }
      }
      return {
        name: c.name,
        measured: c.measured,
        threshold: c.threshold,
        verdict: el("span", "chip " + (c.ok ? "good" : "crit"), c.ok ? "PASS" : "FAIL"),
        extra: extra
      };
    });
    host.appendChild(table([
      { key: "name", label: "Condition" },
      { key: "measured", label: "Measured", num: true },
      { key: "threshold", label: "Threshold", num: true },
      { key: "verdict", label: "" },
      { key: "extra", label: "Note" }
    ], rows));
  }

  function renderCharts(payload) {
    var curves = payload.curves || {};
    var startMs = curves.oos_start_ms || payload.oos_start_ms || payload.start_ms || null;

    if (curves.strategy && curves.strategy.length > 1) {
      $("panel-equity").hidden = false;
      var series = [{ label: "Strategy", data: curves.strategy, cls: "s1" }];
      var hasBasket = !!(curves.basket && curves.basket.length > 1);
      if (hasBasket) {
        series.push({ label: "Buy & hold basket", data: curves.basket, cls: "s2" });
      }
      // Only CLAIM a benchmark comparison when a benchmark leg is actually
      // plotted. Gate runs carry the basket; a backtest does not, and the
      // static title asserted a comparison that was not on the chart.
      $("equity-title").textContent = hasBasket
        ? "Cumulative return vs buy-and-hold"
        : "Cumulative return (no benchmark leg on this run type)";
      $("equity-note").hidden = !hasBasket;
      lineChart($("chart-equity"), series, startMs, { includeOne: true, asPct: true });
      var lg = $("legend-equity");
      clear(lg);
      // Legend is always present for >= 2 series; identity never colour-alone.
      series.forEach(function (s) {
        var w = el("span", "lg");
        var sw = el("span", "sw");
        sw.style.background = "var(--" + s.cls + ")";
        w.appendChild(sw);
        w.appendChild(el("span", null, s.label));
        lg.appendChild(w);
      });
      $("panel-underwater").hidden = false;
      underwaterChart($("chart-underwater"), curves.strategy, startMs);
      $("panel-monthly").hidden = false;
      monthlyHeatmap($("heat-monthly"), curves.strategy, startMs);
    } else {
      $("panel-equity").hidden = true;
      $("panel-underwater").hidden = true;
      $("panel-monthly").hidden = true;
    }

    var trades = payload.trades || curves.trades || [];
    if (trades.length) {
      $("panel-pnl").hidden = false;
      // pnl_pct is a FRACTION; the histogram plots and labels percent.
      pnlHistogram($("chart-pnl"), trades.map(function (t) { return Number(t.pnl_pct) * 100; }));
    } else {
      $("panel-pnl").hidden = true;
    }
  }

  function renderTables(payload) {
    var ps = payload.per_symbol_expectancy || null;
    var psm = payload.per_symbol_metrics || null;
    var host = $("per-symbol");
    clear(host);
    if (ps && Object.keys(ps).length) {
      $("panel-persym").hidden = false;
      host.appendChild(table([
        { key: "sym", label: "Symbol" },
        { key: "exp", label: "OOS expectancy", num: true },
        { key: "verdict", label: "" }
      ], Object.keys(ps).map(function (s) {
        var v = Number(ps[s]);
        return { sym: s, exp: fp(v, 4),
                 verdict: el("span", "chip " + (v > 0 ? "good" : "crit"), v > 0 ? "OK" : "FAIL") };
      })));
    } else if (psm && Object.keys(psm).length) {
      $("panel-persym").hidden = false;
      host.appendChild(table([
        { key: "sym", label: "Symbol" },
        { key: "n", label: "Trades", num: true },
        { key: "wr", label: "Win rate", num: true },
        { key: "exp", label: "Expectancy", num: true },
        { key: "pf", label: "PF", num: true },
        { key: "dd", label: "Max DD", num: true }
      ], Object.keys(psm).map(function (s) {
        var m = psm[s] || {};
        return { sym: s, n: m.n_trades, wr: fpa(m.win_rate, 1),
                 exp: fp(m.expectancy_pct, 4), pf: fmt(m.profit_factor, 2),
                 dd: fpa(m.max_drawdown_pct, 2) };
      })));
    } else {
      $("panel-persym").hidden = true;
    }

    var bh = $("bench-table");
    clear(bh);
    var bm = payload.benchmark;
    if (bm && (bm.per_symbol || bm.basket)) {
      $("panel-bench").hidden = false;
      var rows = [];
      Object.keys(bm.per_symbol || {}).forEach(function (s) {
        var r = bm.per_symbol[s] || {};
        rows.push({ sym: s, total: fmt(r.total_return, 4) + "×", ann: fp(r.ann_return_pct, 2),
                    sharpe: signed(r.sharpe, 3), dd: fpa(r.max_drawdown_pct, 2) });
      });
      if (bm.basket) {
        var b = bm.basket;
        rows.push({ sym: el("strong", null, "BASKET"), total: fmt(b.total_return, 4) + "×",
                    ann: fp(b.ann_return_pct, 2), sharpe: signed(b.sharpe, 3),
                    dd: fpa(b.max_drawdown_pct, 2) });
      }
      bh.appendChild(table([
        { key: "sym", label: "Symbol" },
        { key: "total", label: "Total ×", num: true },
        { key: "ann", label: "Annualized", num: true },
        { key: "sharpe", label: "Sharpe", num: true },
        { key: "dd", label: "Max DD", num: true }
      ], rows));
    } else {
      $("panel-bench").hidden = true;
    }

    var fh = $("folds-table");
    clear(fh);
    var folds = payload.folds || [];
    if (folds.length) {
      $("panel-folds").hidden = false;
      fh.appendChild(table([
        { key: "i", label: "#", num: true },
        { key: "train", label: "Train" },
        { key: "test", label: "Test" },
        { key: "exp", label: "Test expectancy", num: true },
        { key: "n", label: "Trades", num: true }
      ], folds.map(function (f, i) {
        return {
          i: i + 1,
          train: day(f.train_start) + " → " + day(f.train_end),
          test: day(f.test_start) + " → " + day(f.test_end),
          exp: fp(firstFinite(f.test_expectancy_pct, f.test_expectancy, f.expectancy_pct), 4),
          n: firstFinite(f.n_trades, f.test_n_trades)
        };
      })));
    } else {
      $("panel-folds").hidden = true;
    }

    var th = $("trades-table");
    clear(th);
    var trades = payload.trades || [];
    if (trades.length) {
      $("panel-trades").hidden = false;
      $("trades-count").textContent = trades.length + " closed";
      th.appendChild(table([
        { key: "sym", label: "Symbol" },
        { key: "dir", label: "Dir" },
        { key: "entry", label: "Entry", cls: "mono" },
        { key: "exit", label: "Exit", cls: "mono" },
        { key: "pnl", label: "P&L after costs", num: true },
        { key: "outcome", label: "Outcome" },
        { key: "rr", label: "Planned R:R", num: true },
        { key: "pattern", label: "Pattern" }
      ], trades.map(function (t) {
        var v = Number(t.pnl_pct);
        var pnl = el("span", null, fp(v, 3));
        pnl.style.color = v > 0 ? "var(--good)" : (v < 0 ? "var(--crit)" : "");
        pnl.style.fontWeight = "600";
        return { sym: t.symbol, dir: t.direction, entry: day(t.entry_ts), exit: day(t.exit_ts),
                 pnl: pnl, outcome: t.outcome, rr: fmt(t.planned_rr, 2), pattern: t.pattern };
      })));
    } else {
      $("panel-trades").hidden = true;
      $("trades-count").textContent = "";
    }
  }

  /* Jesse-style: copy every metric as aligned two-column plain text. */
  function copyMetrics() {
    var p = LAST_PAYLOAD;
    if (!p) return;
    var lines = [];
    function push(k, v) {
      if (v !== undefined && v !== null && v !== "") lines.push([k, String(v)]);
    }
    if (p.gate) {
      push("GATE", p.gate.passed ? "PASS" : "FAIL");
      (p.gate.conditions || []).forEach(function (c) {
        push("  " + c.name, (c.ok ? "PASS" : "FAIL") +
             "  (" + c.measured + " vs " + c.threshold + ")");
      });
      push("n_trials charged", p.n_trials_used);
    }
    if (p.honesty) push("NOTE", p.honesty);
    var m = p.oos_metrics || p.pooled_metrics || {};
    Object.keys(m).forEach(function (k) {
      if (m[k] !== null && typeof m[k] === "object") return;
      push(k, m[k]);
    });
    var b = p.benchmark && p.benchmark.basket;
    if (b) {
      push("benchmark basket total_return", b.total_return);
      push("benchmark basket ann_return_pct", b.ann_return_pct);
      push("benchmark basket sharpe", b.sharpe);
      push("benchmark basket max_drawdown_pct", b.max_drawdown_pct);
    }
    if (p.oos_start_ms) push("span", day(p.oos_start_ms) + " -> " + day(p.oos_end_ms));

    var w = 0;
    lines.forEach(function (l) { if (l[0].length > w) w = l[0].length; });
    var txt = lines.map(function (l) {
      var pad = "";
      while (pad.length < Math.max(1, w - l[0].length + 2)) pad += " ";
      return l[0] + pad + l[1];
    }).join("\n");

    var btn = $("btn-copy-metrics");
    var done = function () {
      var old = btn.textContent;
      btn.textContent = "✓ copied";
      setTimeout(function () { btn.textContent = old; }, 1400);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(txt).then(done, function () { fallbackCopy(txt, done); });
    } else {
      fallbackCopy(txt, done);
    }
  }
  function fallbackCopy(txt, done) {
    var ta = el("textarea");
    ta.value = txt;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); done(); } catch (e) { showError("copy failed"); }
    document.body.removeChild(ta);
  }

  /* Tier B emits `progress` frames (generation / population counts). Nothing
     consumed them before, so an evolve run showed only an indeterminate bar.
     Becomes DETERMINATE as soon as a current/total pair is available. */
  function renderProgress(p) {
    var host = $("run-progress");
    if (!p) return;
    var cur = firstFinite(p.generation, p.current, p.done, p.completed);
    var tot = firstFinite(p.generations, p.total, p.n_generations);
    clear(host);
    var lab = el("p", "note");
    lab.style.margin = "0 14px 4px";
    var bits = [];
    if (cur !== null && tot !== null) bits.push("generation " + cur + " of " + tot);
    else if (cur !== null) bits.push("generation " + cur);
    var pop = firstFinite(p.population_size, p.population, p.members);
    if (pop !== null) bits.push("population " + pop);
    var evals = firstFinite(p.evaluations, p.n_evaluations, p.trials);
    if (evals !== null) bits.push(evals + " evaluations");
    lab.textContent = bits.length ? bits.join(" · ") : "running…";
    host.appendChild(lab);

    var bar = el("div", "bar" + (cur !== null && tot ? "" : " indet"));
    bar.style.margin = "0 14px 12px";
    var fill = el("span");
    if (cur !== null && tot) {
      var frac = Math.max(0, Math.min(1, cur / tot));
      fill.style.width = (frac * 100).toFixed(1) + "%";
    }
    bar.appendChild(fill);
    host.appendChild(bar);
  }

  /* Render whichever run is active, from ITS OWN buffer. Previously the log pane
     was appended to directly, so switching tabs showed the previous run's output
     under the new run's header. */
  function renderLog() {
    var pre = $("run-log");
    var txt = LOGS[ACTIVE_RUN] || "";
    if (txt) {
      pre.textContent = txt;
      pre.style.color = "";
      pre.scrollTop = pre.scrollHeight;
    } else if (LOG_DONE[ACTIVE_RUN]) {
      // Say so explicitly. Hiding the panel implied "nothing to see", which is
      // indistinguishable from "we never fetched it".
      pre.textContent = "no output was captured for this run";
      pre.style.color = "var(--muted)";
    } else {
      pre.textContent = "loading…";
      pre.style.color = "var(--muted)";
    }
  }

  /* The SSE route replays the log from byte 0 and closes once the run reaches a
     terminal state, so a FINISHED run's log is retrievable — connect regardless
     of state. Because every connection replays from 0, the buffer is reset on
     open, which makes reconnects idempotent instead of duplicating output. */
  function watchRun(rid) {
    if (WATCHED[rid] || LOG_DONE[rid]) return;
    var src = new EventSource("/api/runs/" + encodeURIComponent(rid) + "/events");
    WATCHED[rid] = src;
    LOGS[rid] = "";

    /* EVERY frame api.sse_frames emits is a NAMED event — `log`, `progress`,
       `done`. EventSource.onmessage fires ONLY for unnamed (default-type)
       events, so an onmessage-based handler is dead code and never runs. Use
       addEventListener per name. Also: a `log` frame's data is a RAW LINE of
       stdout, not JSON — only `progress` and `done` carry JSON. */
    src.addEventListener("log", function (ev) {
      LOGS[rid] = (LOGS[rid] || "") + ev.data + "\n";
      if (rid === ACTIVE_RUN) renderLog();
    });

    src.addEventListener("progress", function (ev) {
      var p = null;
      try { p = JSON.parse(ev.data); } catch (e) { return; }
      if (rid === ACTIVE_RUN) renderProgress(p);
    });

    src.addEventListener("done", function (ev) {
      var d = null;
      try { d = JSON.parse(ev.data); } catch (e) { d = {}; }
      src.close();
      delete WATCHED[rid];
      LOG_DONE[rid] = true;
      if (rid === ACTIVE_RUN) renderLog();
      // Re-fetch so the result payload (charts, gate verdict) appears.
      refreshRuns().then(function () {
        if (rid === ACTIVE_RUN && (d.has_result || d.state === "done")) selectRun(rid);
      });
    });

    src.onerror = function () {
      // Do NOT close on a transient error: EventSource reconnects on its own and
      // resumes from the `id:` byte offset via Last-Event-ID, which is how a long
      // evolve run survives a dropped socket. Only clean up once it has really
      // given up.
      if (src.readyState === EventSource.CLOSED) {
        delete WATCHED[rid];
        if (rid === ACTIVE_RUN) renderLog();
      }
    };
  }

  function refreshRuns() {
    return api("GET", "/api/runs").then(function (d) {
      RUNS = d.runs || [];
      if (!ACTIVE_RUN && RUNS.length) ACTIVE_RUN = RUNS[0].run_id;
      renderTabs();
    }).catch(function (e) { showError("runs: " + e.message); });
  }

  function startRun(kind) {
    clearError();
    var campaign = campaignValue();
    if (!campaign) {
      showError("name the new campaign before starting a run");
      return;
    }
    var body = {
      kind: kind,
      strategy: $("strategy-name").value.trim(),
      campaign: campaign
    };
    // Population/generations are evolve-only, and blank means "use the
    // server's default" -- send a value ONLY when the operator actually typed
    // one, never a stray 0 from an empty numeric input.
    if (kind === "evolve") {
      var popRaw = $("run-population").value.trim();
      var genRaw = $("run-generations").value.trim();
      if (!$("run-population").disabled && popRaw) body.population = Number(popRaw);
      if (genRaw) body.generations = Number(genRaw);
    }
    var chain = CURRENT_EDITABLE ? saveStrategy() : Promise.resolve(null);
    chain.then(function () { return api("POST", "/api/runs", body); })
      .then(function (d) {
        show("results");
        // A brand-new campaign only becomes selectable once it exists, and a
        // continued one's generation count just moved.
        refreshCampaigns();
        return refreshRuns().then(function () { selectRun(d.run_id); });
      })
      .catch(function (e) { showError("run failed to start: " + e.message); });
  }

  function stopRun() {
    if (!ACTIVE_RUN) return;
    // api() only sets Content-Type: application/json when a body is passed
    // (see its `if (body !== undefined)` guard above) — and the server
    // requires that header on every POST, /stop included, even though /stop
    // itself ignores the body. Without the {} here this 415s every time.
    api("POST", "/api/runs/" + encodeURIComponent(ACTIVE_RUN) + "/stop", {})
      .then(function () { refreshRuns(); })
      .catch(function (e) { showError("stop failed: " + e.message); });
  }

  // ------------------------------------------------------------ other views
  function renderCoverage(data) {
    var host = $("coverage-table");
    clear(host);
    var rows = (data.coverage || []).map(function (c) {
      var gaps = firstFinite(c.gaps);
      return { sym: c.symbol, tf: c.timeframe, bars: c.bars,
               first: day(c.first_ts), last: day(c.last_ts),
               gaps: gaps === null ? "—"
                     : el("span", "chip " + (gaps ? "crit" : "good"),
                          gaps ? gaps + " gaps" : "clean") };
    });
    if (!rows.length) { host.appendChild(el("p", "empty", "No stored candles.")); return; }
    host.appendChild(table([
      { key: "sym", label: "Symbol" },
      { key: "tf", label: "Timeframe" },
      { key: "bars", label: "Bars", num: true },
      { key: "first", label: "First" },
      { key: "last", label: "Last stored" },
      { key: "gaps", label: "Interior gaps" }
    ], rows));
  }

  function renderPluginList() {
    var host = $("plugins-list");
    clear(host);
    var want = $("plugin-kind").value;
    var rows = [];
    KINDS.concat(["reviewer", "mutator"]).forEach(function (k) {
      if (want && k !== want) return;
      (PLUGINS[k] || []).forEach(function (p) {
        rows.push({ kind: k, key: p.key,
                    tier: (p.tier === null || p.tier === undefined) ? "—" : p.tier,
                    dof: (p.dof === undefined || p.dof === null) ? "—" : p.dof,
                    rationale: p.rationale || "" });
      });
    });
    if (!rows.length) { host.appendChild(el("p", "empty", "No plug-ins for that kind.")); return; }
    var t = el("table");
    var thead = el("thead"), tr = el("tr");
    ["Kind", "Key", "Tier", "DoF"].forEach(function (h, i) {
      tr.appendChild(el("th", i >= 2 ? "num" : null, h));
    });
    thead.appendChild(tr);
    t.appendChild(thead);
    var tb = el("tbody");
    rows.forEach(function (r) {
      var a = el("tr");
      a.appendChild(el("td", null, r.kind));
      var kd = el("td");
      kd.appendChild(el("strong", null, r.key));
      a.appendChild(kd);
      a.appendChild(el("td", "num", r.tier));
      a.appendChild(el("td", "num", r.dof));
      tb.appendChild(a);
      if (r.rationale) {
        var b = el("tr");
        var c = el("td", null, r.rationale);
        c.colSpan = 4;
        c.style.color = "var(--muted)";
        c.style.fontSize = "11.5px";
        c.style.whiteSpace = "normal";
        c.style.paddingTop = "0";
        b.appendChild(c);
        tb.appendChild(b);
      }
    });
    t.appendChild(tb);
    host.appendChild(t);
  }

  function loadReviews() {
    api("GET", "/api/reviews").then(function (d) {
      var host = $("reviews-table");
      clear(host);
      var rows = (d.reviews || []).map(function (r) {
        return { sym: r.symbol, entry: day(r.entry_ts), pnl: signedPct(r.pnl_pct, 3),
                 outcome: r.outcome, tp: r.tp_verdict, sl: r.sl_verdict,
                 ver: String(r.strategy_version || "").slice(0, 12) };
      });
      if (!rows.length) {
        host.appendChild(el("p", "empty",
          "No review records yet. Run `cli review`, or a gate run."));
        return;
      }
      host.appendChild(table([
        { key: "sym", label: "Symbol" },
        { key: "entry", label: "Entry" },
        { key: "pnl", label: "P&L", num: true },
        { key: "outcome", label: "Outcome" },
        { key: "tp", label: "TP verdict" },
        { key: "sl", label: "SL verdict" },
        { key: "ver", label: "Version", cls: "mono" }
      ], rows));
    }).catch(function (e) {
      var host = $("reviews-table");
      clear(host);
      host.appendChild(el("p", "empty", "reviews unavailable: " + e.message));
    });
  }

  function loadGenerations() {
    // The dropdown carries campaign_ids, which is what the generations table is
    // keyed on; the text input stays as a fallback for a hand-typed id.
    var campaign = $("generations-campaign-select").value
                || $("generations-campaign").value.trim();
    var host = $("generations-table");
    clear(host);
    if (!campaign) { host.appendChild(el("p", "empty", "No campaigns yet.")); return; }
    api("GET", "/api/generations?campaign=" + encodeURIComponent(campaign)).then(function (d) {
      var gens = d.generations || [];
      if (gens.length) {
        host.appendChild(table([
          { key: "i", label: "Gen", num: true },
          { key: "n", label: "Members", num: true },
          { key: "best", label: "Best fitness", num: true },
          { key: "member", label: "Best member", cls: "mono" }
        ], gens.map(function (g) {
          // gen_index / best_member_id are the names /api/generations actually
          // emits (and what the generations table stores). Reading g.generation
          // and g.best_member silently rendered both columns as "—".
          return { i: g.gen_index, n: g.population_size,
                   best: signed(firstFinite(g.best_fitness), 4),
                   member: g.best_member_id };
        })));
      } else {
        host.appendChild(el("p", "empty",
          "No generations for campaign “" + campaign + "”."));
      }
      var top = d.top_members || d.leaderboard || [];
      if (top.length) {
        var cap = el("p", "note",
          "Leaderboard — fitness is excess_sharpe (benchmark-relative and trial-count free). " +
          "DSR moves with the trial count, so it is a verdict statistic and is deliberately " +
          "not used for ranking.");
        cap.style.padding = "10px 14px 0";
        host.appendChild(cap);
        host.appendChild(table([
          { key: "m", label: "Member", cls: "mono" },
          { key: "tier", label: "Tier" },
          { key: "fit", label: "Fitness", num: true },
          { key: "n", label: "Trades", num: true }
        ], top.map(function (t) {
          return { m: t.member_id || t.member, tier: t.tier,
                   fit: signed(firstFinite(t.fitness), 4), n: firstFinite(t.n_trades) };
        })));
      }
    }).catch(function (e) {
      clear(host);
      host.appendChild(el("p", "empty", "generations unavailable: " + e.message));
    });
  }

  // ------------------------------------------------- evolution replay player
  /* Sidebar (campaign -> generation -> member) plus an animated candlestick
     replay of a member's exact evaluation window. The replay endpoint re-runs
     a real backtest server-side (deterministic re-derivation, never the trial
     ledger) so EVERYTHING here is read-only against already-scored members.

     EVO holds all mutable player state. There is exactly ONE timer for the
     whole app: EVO.timer. It is cancelled (a) in show() when navigating away
     from the evolution view, (b) at the top of loadReplay() before loading a
     new member/symbol/timeframe, and (c) when switching to the Table toggle.
     Orphaned intervals were an explicit edge case in the plan, hence the
     single shared handle rather than one per chart. */
  var EVO = {
    campaign: null,   // campaign id the sidebar is currently showing
    member: null,     // member_id of the loaded replay, or null
    chart: null,       // built by evoBuildChart/renderReplay, or null
    cursor: 0,          // bars revealed so far
    speed: 1,           // bars advanced per timer tick
    playing: false,
    timer: null
  };

  function evoCancelTimer() {
    if (EVO.timer) { clearInterval(EVO.timer); EVO.timer = null; }
    EVO.playing = false;
    var b = $("evo-btn-play");
    if (b) b.textContent = "▶ play";
  }

  function evoBsearchFirstGE(arr, val) {
    // First index i such that arr[i] >= val (arr sorted ascending). Bar
    // timeframe may differ from the trade's native spacing, so entry_ts/
    // exit_ts are not guaranteed to land exactly on a bar's ts.
    var lo = 0, hi = arr.length;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (arr[mid] < val) lo = mid + 1; else hi = mid;
    }
    return lo >= arr.length ? arr.length - 1 : lo;
  }
  function evoClamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function evoOutcomeColor(t) {
    // "colored by outcome" per the plan, but "time"/"end" exits are outcome-
    // neutral labels (the engine's vocabulary is stop/trail/channel/target/
    // time/end) -- fall back to the realized pnl sign for those.
    var o = t.outcome;
    if (o === "target" || o === "trail" || o === "channel") return "--good";
    if (o === "stop") return "--crit";
    var v = Number(t.pnl_pct);
    if (!isFinite(v)) return "--muted";
    return v > 0 ? "--good" : (v < 0 ? "--crit" : "--muted");
  }

  /* Legend for the replay chart's trade overlays -- rendered once per chart
     build into the sibling #evo-legend element (the .legend/.lg/.sw idiom
     also used by the equity chart), never re-drawn per frame. */
  function evoBuildLegend() {
    var host = $("evo-legend");
    if (!host) return;
    clear(host);
    [["entry", "s1"], ["stop", "crit"], ["target", "good"]].forEach(function (item) {
      var w = el("span", "lg");
      var sw = el("span", "sw");
      sw.style.background = "var(--" + item[1] + ")";
      w.appendChild(sw);
      w.appendChild(el("span", null, item[0]));
      host.appendChild(w);
    });
    var zw = el("span", "lg");
    var zsw = el("span", "sw sw-zone");
    zw.appendChild(zsw);
    zw.appendChild(el("span", null, "pattern zone"));
    host.appendChild(zw);
  }

  /* ---------------------------------------------------------------- viewport
     The price chart draws a WINDOW of bars -- [view.from, view.from+view.count)
     -- not the whole replay. At full extent a 400-bar window gives each candle
     under two pixels and the pattern geometry this panel exists to show is
     unreadable, which is the defect this replaces. Zoom (wheel / buttons),
     pan (drag), and range-select (drag on the overview strip) all move that
     window and nothing else; every mutation goes through evoSetView so the
     clamping, the redraw and the overview stay in one place.

     Redraw is a full rebuild of the visible slice rather than the append-only
     scheme this had before: with a viewport the "already drawn" set changes on
     every pan, so incremental bookkeeping buys nothing and costs correctness.
     view.count is bounded by what fits on screen, so the node count per frame
     is bounded too. */
  var EVO_MIN_BARS = 15;

  function evoSetView(st, from, count) {
    if (!st) return;
    count = evoClamp(Math.round(count), Math.min(EVO_MIN_BARS, st.n), st.n);
    from = evoClamp(Math.round(from), 0, st.n - count);
    st.view.from = from;
    st.view.count = count;
    evoDraw(st);
    evoDrawOverview(st);
    // The bar readout and the performance panel's view marker both describe the
    // window, so they are part of a view change -- not just of a cursor change.
    evoRenderPosition();
    evoDrawPerf(st);
  }

  /* Zoom about a fixed point: the bar under `anchorFrac` (0..1 across the plot)
     stays put, the way a chart tool is expected to behave under the cursor. */
  function evoZoom(st, factor, anchorFrac) {
    if (!st) return;
    var v = st.view;
    var anchorBar = v.from + v.count * anchorFrac;
    var count = evoClamp(Math.round(v.count * factor), Math.min(EVO_MIN_BARS, st.n), st.n);
    evoSetView(st, Math.round(anchorBar - count * anchorFrac), count);
  }

  function evoBuildChart(bars) {
    var host = $("evo-chart");
    clear(host);
    var n = (bars.ts || []).length;
    if (!n) {
      // No stored candles for this symbol/timeframe in the window -- the
      // existing `.empty` idiom, not a broken chart.
      clear($("evo-legend"));
      clear($("evo-overview"));
      clear($("evo-perf"));
      host.appendChild(el("p", "empty",
        "No stored candles for this symbol/timeframe in the replay window."));
      return null;
    }
    // A fixed 400px-tall plot: the height no longer has to grow with the trade
    // count, because trade detail moved out of the chart (see evoDrawPerf) and
    // is no longer a stack of callout labels fighting for vertical room.
    var W = 900, m = { t: 12, r: 62, b: 26, l: 58 };
    var ih = 400, H = ih + m.t + m.b, iw = W - m.l - m.r;
    var st = {
      host: host, bars: bars, n: n, W: W, H: H, m: m, ih: ih, iw: iw,
      view: { from: 0, count: n },
      trades: [], cursor: 0, perfKey: null, panning: false,
      X: null, Y: null, bw: 1
    };
    evoBuildLegend();
    return st;
  }

  /* Price-domain for the CURRENT view: the visible candles, widened to keep the
     levels of any visible revealed trade on screen (a stop just off the low is
     the thing you are trying to read), but never widened past twice the candle
     range -- one runaway target must not flatten the candles into a line. */
  function evoDomain(st) {
    var v = st.view, hiIdx = Math.min(st.n - 1, v.from + v.count - 1);
    var lo = Infinity, hi = -Infinity;
    for (var i = v.from; i <= hiIdx; i++) {
      if (st.bars.l[i] < lo) lo = st.bars.l[i];
      if (st.bars.h[i] > hi) hi = st.bars.h[i];
    }
    if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
    if (lo === hi) { lo -= 1; hi += 1; }
    var cap = hi - lo;
    st.trades.forEach(function (tr) {
      if (tr.entryIdx > hiIdx || tr.exitIdx < v.from) return;
      if (st.cursor - 1 < tr.entryIdx) return;
      [tr.trade.entry, tr.trade.stop, tr.trade.target].forEach(function (raw) {
        var val = Number(raw);
        if (!isFinite(val)) return;
        if (val < lo && val > lo - cap) lo = val;
        if (val > hi && val < hi + cap) hi = val;
      });
    });
    var pad = (hi - lo) * 0.06;
    return { lo: lo - pad, hi: hi + pad };
  }

  function evoDraw(st) {
    if (!st) return;
    clear(st.host);
    var v = st.view, m = st.m, n = st.n;
    var last = Math.min(n - 1, v.from + v.count - 1);
    var d = evoDomain(st);
    var slot = st.iw / v.count;
    // Below ~2px a candle carries no readable shape, and a 51k-bar 15m window
    // drawn one node per bar cost ~300ms a frame -- unusable for panning or
    // playback. Aggregate into buckets (first open, last close, extreme high
    // and low) so the node count is capped by the plot's own width, exactly
    // what the geometry can actually show.
    var stride = Math.max(1, Math.ceil(v.count / Math.floor(st.iw / 2)));
    st.bw = Math.max(1, slot * stride * 0.66);
    st.X = function (i) { return m.l + (i - v.from + 0.5) * slot; };
    st.Y = function (val) { return m.t + st.ih - ((val - d.lo) / (d.hi - d.lo)) * st.ih; };

    var svg = svgEl("svg", { class: "plot evo-plot", viewBox: "0 0 " + st.W + " " + st.H,
                             preserveAspectRatio: "none", role: "img",
                             "aria-label": "Replay candles with trade overlays" });

    niceTicks(d.lo, d.hi, 6).forEach(function (val) {
      if (val < d.lo || val > d.hi) return;
      var y = st.Y(val);
      svg.appendChild(svgEl("line", { class: "grid-line", x1: m.l, x2: m.l + st.iw, y1: y, y2: y }));
      svg.appendChild(svgText(m.l - 8, y + 3.5, fmt(val, 2), "tick", "end"));
    });
    // Up to six time labels across the visible span, never the fixed three the
    // old chart drew -- when you zoom to twenty bars you need to know WHICH
    // twenty, and the endpoints alone do not tell you.
    var step = Math.max(1, Math.round(v.count / 5));
    for (var i = v.from; i <= last; i += step) {
      svg.appendChild(svgEl("line", { class: "grid-line grid-vert",
        x1: st.X(i), x2: st.X(i), y1: m.t, y2: m.t + st.ih }));
      svg.appendChild(svgText(st.X(i), st.H - 8, day(st.bars.ts[i]), "tick", "middle"));
    }
    svg.appendChild(svgEl("line", { class: "axis-line",
      x1: m.l, x2: m.l + st.iw, y1: m.t + st.ih, y2: m.t + st.ih }));

    var gCandles = svgEl("g", { class: "evo-candles" });
    var gOverlays = svgEl("g", { class: "evo-overlays" });
    svg.appendChild(gCandles);
    svg.appendChild(gOverlays);

    // Only bars that are BOTH inside the view and already revealed by the
    // player cursor: the replay's whole point is that the future is hidden.
    var revealedTo = Math.min(last, st.cursor - 1);
    for (var b = v.from; b <= revealedTo; b += stride) {
      gCandles.appendChild(evoCandleNode(st, b, Math.min(b + stride - 1, revealedTo)));
    }

    st.trades.forEach(function (tr) {
      if (st.cursor - 1 < tr.entryIdx) return;
      if (tr.entryIdx > last || tr.exitIdx < v.from) return;
      evoDrawTrade(st, gOverlays, tr, revealedTo);
    });

    // Crosshair + a single plot-wide hit rect. One listener replaces the old
    // per-candle handlers, and it is what carries wheel-zoom and drag-pan.
    var crossV = svgEl("line", { class: "crosshair", y1: m.t, y2: m.t + st.ih, x1: m.l, x2: m.l });
    var crossH = svgEl("line", { class: "crosshair", x1: m.l, x2: m.l + st.iw, y1: m.t, y2: m.t });
    crossV.style.opacity = "0"; crossH.style.opacity = "0";
    svg.appendChild(crossV); svg.appendChild(crossH);
    var hit = svgEl("rect", { class: "hit evo-hit",
      x: m.l, y: m.t, width: st.iw, height: st.ih });
    svg.appendChild(hit);
    st.svg = svg; st.hit = hit; st.crossV = crossV; st.crossH = crossH;
    evoWireInteractions(st);
    st.host.appendChild(svg);
  }

  /* Pointer x (client space) -> bar index, through the SVG's own viewBox so it
     stays correct at any rendered width. */
  function evoBarAt(st, clientX) {
    var box = st.svg.getBoundingClientRect();
    var vx = (clientX - box.left) / box.width * st.W;
    var i = st.view.from + Math.floor((vx - st.m.l) / (st.iw / st.view.count));
    return evoClamp(i, 0, st.n - 1);
  }

  function evoWireInteractions(st) {
    var hit = st.hit;

    hit.addEventListener("mousemove", function (ev) {
      if (st.panning) return;
      var i = evoBarAt(st, ev.clientX);
      if (i > st.cursor - 1) {
        st.crossV.style.opacity = "0"; st.crossH.style.opacity = "0"; tipHide(); return;
      }
      var box = st.svg.getBoundingClientRect();
      st.crossV.setAttribute("x1", st.X(i)); st.crossV.setAttribute("x2", st.X(i));
      var vy = (ev.clientY - box.top) / box.height * st.H;
      st.crossH.setAttribute("y1", vy); st.crossH.setAttribute("y2", vy);
      st.crossV.style.opacity = "1"; st.crossH.style.opacity = "1";
      tipShow("<div style=\"color:var(--muted);margin-bottom:3px\">" + day(st.bars.ts[i]) + "</div>" +
              "O " + fmt(st.bars.o[i], 2) + " H " + fmt(st.bars.h[i], 2) +
              " L " + fmt(st.bars.l[i], 2) + " C " + fmt(st.bars.c[i], 2) +
              "<br>vol " + fmt(st.bars.v[i], 0), ev.clientX, ev.clientY);
    });
    hit.addEventListener("mouseleave", function () {
      st.crossV.style.opacity = "0"; st.crossH.style.opacity = "0"; tipHide();
    });

    // Wheel zoom about the cursor. passive:false because the whole point is to
    // preventDefault -- otherwise the page scrolls out from under the chart.
    hit.addEventListener("wheel", function (ev) {
      ev.preventDefault();
      var box = st.svg.getBoundingClientRect();
      var vx = (ev.clientX - box.left) / box.width * st.W;
      var frac = evoClamp((vx - st.m.l) / st.iw, 0, 1);
      evoZoom(st, ev.deltaY > 0 ? 1.18 : 1 / 1.18, frac);
    }, { passive: false });

    // Drag to pan. The move/up listeners live on window so a drag that leaves
    // the SVG still tracks, and they are removed on mouseup -- a redraw
    // replaces the hit rect entirely, so nothing accumulates across frames.
    hit.addEventListener("mousedown", function (ev) {
      ev.preventDefault();
      tipHide();
      st.panning = true;
      st.host.classList.add("panning");
      var startX = ev.clientX, startFrom = st.view.from;
      var box = st.svg.getBoundingClientRect();
      // Client px -> bars, via the viewBox scale (the plot area is iw/W of the
      // rendered width).
      var barsPerPx = st.view.count / (box.width * (st.iw / st.W));
      function move(e) {
        evoSetView(st, startFrom - (e.clientX - startX) * barsPerPx, st.view.count);
      }
      function up() {
        st.panning = false;
        st.host.classList.remove("panning");
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", up);
      }
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
    });

    hit.addEventListener("dblclick", function () { evoSetView(st, 0, st.n); });
  }

  /* One candle for bars [i..j]. j === i is the ordinary case; a wider bucket is
     the zoomed-out aggregate, which is a real OHLC of the range (first open,
     last close, extreme high and low) rather than a sample of one bar. */
  function evoCandleNode(st, i, j) {
    if (j === undefined) j = i;
    var o = st.bars.o[i], c = st.bars.c[j];
    var h = st.bars.h[i], l = st.bars.l[i];
    for (var k = i + 1; k <= j; k++) {
      if (st.bars.h[k] > h) h = st.bars.h[k];
      if (st.bars.l[k] < l) l = st.bars.l[k];
    }
    var up = c >= o;
    var x = st.X((i + j) / 2);
    var g = svgEl("g", { class: "candle" });
    g.appendChild(svgEl("line", { class: up ? "wick-pos" : "wick-neg",
      x1: x.toFixed(1), x2: x.toFixed(1), y1: st.Y(h).toFixed(1), y2: st.Y(l).toFixed(1) }));
    var yTop = st.Y(Math.max(o, c)), yBot = st.Y(Math.min(o, c));
    g.appendChild(svgEl("rect", { class: up ? "bar-pos" : "bar-neg",
      x: (x - st.bw / 2).toFixed(1), y: yTop.toFixed(1),
      width: st.bw.toFixed(1), height: Math.max(1, yBot - yTop).toFixed(1) }));
    return g;
  }

  /* Chronological role of each named meta key a detector emits (reversal.py,
     continuation.py), used only to place a pivot marker along the pattern's
     real x-span -- never to invent a position outside it. Confirmed against
     the detector source, not guessed:
       doubles          (reversal.py:447-459): a.index < m.index < b.index,
                         so peak_a is the FIRST extreme, peak_b the LAST, and
                         trough (the level BOTH share) is the between-pivot.
       head-and-shoulders (reversal.py:281-297): window[0]=left, window[-1]=right,
                         head strictly between -- left_shoulder first, head
                         mid, right_shoulder last.
       flag/pole        (continuation.py:578-591): pole_high/pole_low are both
                         extremes of the SAME pole_start..pole_end slice, i.e.
                         both sit in the opening portion of the whole span
                         (pole, then consolidation) -- "start" for both is the
                         honest placement; there is no finer-grained ordering
                         to report from these two floats alone. */
  var PATTERN_META_ROLE = {
    peak_a: "start", trough: "mid", peak_b: "end",
    left_shoulder: "start", head: "mid", right_shoulder: "end",
    pole_low: "start", pole_high: "start"
  };

  /* Tier-3 upgrade: a detector that emits a `<key>_ts` alongside a `<key>`
     price (peak_a/peak_a_ts, head/head_ts, ...) gives us the pivot's REAL bar,
     not just its known chronological ROLE (PATTERN_META_ROLE's start/mid/end
     approximation). Paired generically -- no second hardcoded table naming
     "peak_a"/"head"/etc a second time -- so a future detector that starts
     emitting timestamped meta draws the connected shape with no JS change.
     Returns vertices sorted by bar index, each index clamped into
     [patternStartIdx, min(patternEndIdx, entryIdx)] so a pivot ts that lands
     outside the loaded bar range (or outside the pattern's own span, which
     should not happen but must never crash or draw off-canvas if it does)
     is pulled back onto the pattern's own segment instead of thrown or
     rendered off-plot. */
  function evoPatternVertices(meta, ts, patternStartIdx, patternEndIdx, entryIdx) {
    var lo = patternStartIdx, hi = Math.max(lo, Math.min(patternEndIdx, entryIdx));
    var verts = [];
    Object.keys(meta).forEach(function (key) {
      if (/_ts$/.test(key)) return; // consumed via its paired base key below
      var tsKey = key + "_ts";
      if (!(tsKey in meta)) return;
      var price = Number(meta[key]), rawTs = Number(meta[tsKey]);
      if (!isFinite(price) || !isFinite(rawTs)) return;
      var idx = evoClamp(evoBsearchFirstGE(ts, rawTs), lo, hi);
      verts.push({ key: key, idx: idx, price: price });
    });
    verts.sort(function (a, b) { return a.idx - b.idx; });
    return verts;
  }

  /* One trade's geometry, drawn only as far as the cursor has revealed.

     Trade NUMBERS no longer live on the chart at all. They used to stack in a
     right-margin callout column that, on a busy window, became forty lines of
     text with leader lines crossing the whole plot -- the "right side is too
     compact" defect. The chart now shows SHAPE (zone, levels, direction,
     outcome); the performance panel below shows FIGURES; hovering a trade
     gives its full detail. */
  function evoDrawTrade(st, host, tr, revealedTo) {
    var t = tr.trade;
    var isLong = String(t.direction).toLowerCase() === "long";
    var closed = st.cursor - 1 >= tr.exitIdx;
    var entryX = st.X(tr.entryIdx);
    var rightX = st.X(Math.max(tr.entryIdx, Math.min(tr.exitIdx, revealedTo)));
    var entryY = st.Y(Number(t.entry));
    var g = svgEl("g", { class: "evo-trade" + (closed ? " closed" : " open") });

    // The pattern's REAL geometry (v0.3.2 D4 fix): drawn only when the trade
    // actually carries it (graph-path trades -- see evoBuildTrades). A trade
    // with no geometry (legacy-engine trades, runs persisted before v0.3.2)
    // gets NO zone at all here -- degrading honestly instead of falling back
    // to the old hardcoded 20-bar/34px box, which had no relationship to the
    // detected pattern and is why this block used to lie.
    if (tr.hasPattern) {
      var psX = st.X(tr.patternStartIdx);
      var peX = Math.max(psX, st.X(Math.min(tr.patternEndIdx, tr.entryIdx)));
      var midX = (psX + peX) / 2;
      var roleX = { start: psX, mid: midX, end: peX };
      var meta = t.pattern_meta || {};

      var vertices = evoPatternVertices(
        meta, st.bars.ts, tr.patternStartIdx, tr.patternEndIdx, tr.entryIdx
      );

      if (vertices.length >= 2) {
        // TIER 3: real pivot timestamps are present (peak_a_ts/trough_ts/
        // peak_b_ts for doubles, left_shoulder_ts/head_ts/right_shoulder_ts
        // for H&S, ...) -- draw the CONNECTED shape through the pattern's own
        // bars, not a role-based approximation. This is what makes a
        // double-bottom actually render as a W anchored on real candles.
        var pts = vertices.map(function (v) {
          return st.X(v.idx).toFixed(1) + "," + st.Y(v.price).toFixed(1);
        }).join(" ");
        var poly = svgEl("polyline", { points: pts });
        poly.style.fill = "none";
        poly.style.stroke = isLong ? "var(--good)" : "var(--crit)";
        poly.style.strokeWidth = "1.6";
        poly.style.opacity = "0.85";
        g.appendChild(poly);
        vertices.forEach(function (v) {
          var vx = st.X(v.idx), vy = st.Y(v.price);
          var dot = svgEl("circle", { cx: vx.toFixed(1), cy: vy.toFixed(1), r: 2.8 });
          dot.style.fill = isLong ? "var(--good)" : "var(--crit)";
          g.appendChild(dot);
          if (peX - psX > 40) {
            g.appendChild(svgText(vx, vy - 6, v.key.replace(/_/g, " "), "evo-zone-label", "middle"));
          }
        });
      } else {
        // TIER 2 fallback: prices only, no pivot timestamps (a detector that
        // does not emit `<key>_ts`, or a run persisted before this change).
        // One horizontal line per named shape level, spanning the pattern's
        // own start->end (not a fixed 20 bars), at that level's ACTUAL price,
        // with the pivot marker at its KNOWN chronological ROLE rather than
        // its real bar -- see PATTERN_META_ROLE's docstring for why that is
        // an honest approximation and not an invented position.
        Object.keys(meta).forEach(function (key) {
          var role = PATTERN_META_ROLE[key];
          if (!role) return; // unlabeled diagnostic meta (separation_bars, etc.)
          var val = Number(meta[key]);
          if (!isFinite(val)) return;
          var y = st.Y(val);
          var line = svgEl("line", { x1: psX.toFixed(1), x2: peX.toFixed(1),
            y1: y.toFixed(1), y2: y.toFixed(1) });
          line.style.stroke = "var(--text-secondary)";
          line.style.strokeWidth = "1";
          line.style.strokeDasharray = "2 2";
          line.style.opacity = "0.65";
          g.appendChild(line);

          var dot = svgEl("circle", { cx: roleX[role].toFixed(1), cy: y.toFixed(1), r: 2.6 });
          dot.style.fill = isLong ? "var(--good)" : "var(--crit)";
          g.appendChild(dot);
          if (peX - psX > 40) {
            g.appendChild(svgText(roleX[role], y - 5, key.replace(/_/g, " "), "evo-zone-label",
              role === "end" ? "end" : role === "start" ? "start" : "middle"));
          }
        });
      }

      // The breakout level itself ("neckline" for reversals, the consolidation
      // edge for continuations) -- DetectedEvent.level, extended from the
      // pattern's end up to the entry so it reads as the line the trigger bar
      // actually broke through.
      if (isFinite(Number(t.pattern_level)) && Number(t.pattern_level) !== 0) {
        var lvlY = st.Y(Number(t.pattern_level));
        var lvl = svgEl("line", { x1: psX.toFixed(1), x2: entryX.toFixed(1),
          y1: lvlY.toFixed(1), y2: lvlY.toFixed(1) });
        lvl.style.stroke = isLong ? "var(--good)" : "var(--crit)";
        lvl.style.strokeWidth = "1.2";
        lvl.style.strokeDasharray = "5 3";
        lvl.style.opacity = "0.8";
        g.appendChild(lvl);
      }

      if (peX - psX > 54) {
        g.appendChild(svgText(midX, st.m.t + 10, t.pattern || "pattern",
          "evo-zone-label", "middle"));
      }
    }

    [[t.entry, "evo-line-entry"], [t.stop, "evo-line-stop"], [t.target, "evo-line-target"]]
      .forEach(function (spec) {
        var val = Number(spec[0]);
        if (!isFinite(val)) return;
        var y = st.Y(val);
        g.appendChild(svgEl("line", { class: spec[1],
          x1: entryX.toFixed(1), x2: rightX.toFixed(1), y1: y.toFixed(1), y2: y.toFixed(1) }));
      });

    g.appendChild(svgText(entryX, entryY + (isLong ? 15 : -9), isLong ? "▲" : "▼",
      "evo-marker " + (isLong ? "evo-long" : "evo-short"), "middle"));

    if (closed) {
      var mark = svgEl("circle", { class: "evo-exit-marker",
        cx: st.X(tr.exitIdx).toFixed(1), cy: st.Y(Number(t.exit_price)).toFixed(1), r: 4 });
      mark.style.fill = "var(" + evoOutcomeColor(t) + ")";
      g.appendChild(mark);
    }

    // Transparent hover band over the trade's own span and level range: this is
    // where the numbers now live. It sits above the candles by construction
    // (gOverlays is appended after gCandles), which is the right trade-off --
    // inside a trade's span, the trade is what you are reading.
    var ys = [entryY, st.Y(Number(t.stop)), st.Y(Number(t.target))].filter(isFinite);
    var yTop = Math.min.apply(null, ys), yBot = Math.max.apply(null, ys);
    var band = svgEl("rect", { class: "evo-trade-hit",
      x: (entryX - st.bw).toFixed(1), y: (yTop - 4).toFixed(1),
      width: Math.max(6, rightX - entryX + st.bw * 2).toFixed(1),
      height: Math.max(10, yBot - yTop + 8).toFixed(1) });
    band.addEventListener("mousemove", function (ev) {
      ev.stopPropagation();
      tipShow(evoTradeTip(t, closed), ev.clientX, ev.clientY);
    });
    band.addEventListener("mouseleave", tipHide);
    g.appendChild(band);

    host.appendChild(g);
  }

  function evoTradeTip(t, closed) {
    var rows = [
      "<div style=\"color:var(--muted);margin-bottom:3px\">" +
        String(t.direction || "?").toUpperCase() + " · " + (t.pattern || "—") + "</div>",
      "entry <b>" + fmt(t.entry, 2) + "</b> on " + day(t.entry_ts),
      "stop <b>" + fmt(t.stop, 2) + "</b> · target <b>" + fmt(t.target, 2) + "</b>",
      "planned R:R <b>" + fmt(t.planned_rr, 2) + "</b>"
    ];
    if (closed) {
      rows.push("exit <b>" + fmt(t.exit_price, 2) + "</b> on " + day(t.exit_ts) +
                " (" + (t.outcome || "—") + ")");
      rows.push("P&amp;L <b style=\"color:var(" + evoOutcomeColor(t) + ")\">" +
                fp(Number(t.pnl_pct), 2) + "</b>");
    } else {
      rows.push("<i>still open</i>");
    }
    return rows.join("<br>");
  }

  /* ---- overview strip ----------------------------------------------------
     The whole replay window at a glance: a close-price sparkline, a tick under
     every trade entry, and a box showing where the main chart is looking. Drag
     across it to SELECT a range (the "select a certain range" ask); click once
     to recentre at the current zoom; the ⤢ button resets to the full window. */
  function evoDrawOverview(st) {
    var host = $("evo-overview");
    if (!host) return;
    clear(host);
    if (!st || st.n < 2) return;
    var W = 900, H = 54, m = { t: 6, b: 6, l: 58, r: 62 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var lo = Math.min.apply(null, st.bars.l), hi = Math.max.apply(null, st.bars.h);
    if (!isFinite(lo) || !isFinite(hi) || lo === hi) { lo -= 1; hi += 1; }
    var X = function (i) { return m.l + (i / (st.n - 1)) * iw; };
    var Y = function (v) { return m.t + ih - ((v - lo) / (hi - lo)) * ih; };

    var svg = svgEl("svg", { class: "plot evo-overview-plot", viewBox: "0 0 " + W + " " + H,
                             preserveAspectRatio: "none", role: "img",
                             "aria-label": "Replay overview and range selector" });
    var d = "";
    for (var i = 0; i < st.n; i++) {
      d += (i ? " L" : "M") + X(i).toFixed(1) + "," + Y(st.bars.c[i]).toFixed(1);
    }
    svg.appendChild(svgEl("path", { class: "series s1 evo-ov-line", d: d }));

    st.trades.forEach(function (tr) {
      var isLong = String(tr.trade.direction).toLowerCase() === "long";
      svg.appendChild(svgEl("line", { class: "evo-ov-tick " + (isLong ? "long" : "short"),
        x1: X(tr.entryIdx).toFixed(1), x2: X(tr.entryIdx).toFixed(1), y1: m.t, y2: m.t + ih }));
    });
    // Where the player cursor is, so the strip doubles as a progress read.
    if (st.cursor > 0) {
      var cx = X(evoClamp(st.cursor - 1, 0, st.n - 1));
      svg.appendChild(svgEl("line", { class: "evo-ov-cursor",
        x1: cx.toFixed(1), x2: cx.toFixed(1), y1: m.t, y2: m.t + ih }));
    }

    var vx = X(st.view.from);
    var vw = Math.max(2, X(st.view.from + st.view.count - 1) - vx);
    svg.appendChild(svgEl("rect", { class: "evo-ov-window",
      x: vx.toFixed(1), y: m.t, width: vw.toFixed(1), height: ih }));

    var sel = svgEl("rect", { class: "evo-ov-select", x: 0, y: m.t, width: 0, height: ih });
    sel.style.opacity = "0";
    svg.appendChild(sel);
    var hit = svgEl("rect", { class: "hit evo-ov-hit", x: m.l, y: m.t, width: iw, height: ih });
    svg.appendChild(hit);

    function barAt(clientX) {
      var box = svg.getBoundingClientRect();
      var px = (clientX - box.left) / box.width * W;
      return evoClamp(Math.round((px - m.l) / iw * (st.n - 1)), 0, st.n - 1);
    }
    hit.addEventListener("mousedown", function (ev) {
      ev.preventDefault();
      var a = barAt(ev.clientX), moved = false;
      function move(e) {
        moved = true;
        var b = barAt(e.clientX);
        sel.style.opacity = "1";
        sel.setAttribute("x", X(Math.min(a, b)).toFixed(1));
        sel.setAttribute("width", Math.max(1, Math.abs(X(b) - X(a))).toFixed(1));
      }
      function up(e) {
        window.removeEventListener("mousemove", move);
        window.removeEventListener("mouseup", up);
        var b = barAt(e.clientX);
        // A plain click (or a drag too short to be a range) recentres at the
        // current zoom rather than selecting a one-bar view, which is useless.
        if (moved && Math.abs(b - a) >= EVO_MIN_BARS / 2) {
          evoSetView(st, Math.min(a, b), Math.abs(b - a) + 1);
        } else {
          evoSetView(st, a - st.view.count / 2, st.view.count);
        }
      }
      window.addEventListener("mousemove", move);
      window.addEventListener("mouseup", up);
    });
    host.appendChild(svg);
  }

  /* ---- performance panel -------------------------------------------------
     What used to be a right-margin column of price callouts is now a proper
     performance read on the trades closed SO FAR: stat tiles, a compounded
     equity curve with its underwater plot on the same bar timeline as the
     price chart above, and the trade log. Same shape as the standalone report
     scripts/build_performance_chart.py produces, on the same tokens. */
  function evoPerfStats(closed) {
    var eq = 1, peak = 1, maxDd = 0, wins = 0, gain = 0, loss = 0, sum = 0;
    var curve = [];
    closed.forEach(function (tr) {
      var r = Number(tr.trade.pnl_pct);
      if (!isFinite(r)) r = 0;
      sum += r;
      if (r > 0) { wins++; gain += r; } else { loss -= r; }
      eq *= 1 + r;
      peak = Math.max(peak, eq);
      maxDd = Math.max(maxDd, peak > 0 ? 1 - eq / peak : 0);
      curve.push({ idx: tr.exitIdx, eq: eq, dd: peak > 0 ? 1 - eq / peak : 0 });
    });
    return {
      curve: curve, n: closed.length, wins: wins,
      winRate: closed.length ? wins / closed.length : null,
      expectancy: closed.length ? sum / closed.length : null,
      profitFactor: loss > 0 ? gain / loss : null,
      total: eq - 1, maxDd: maxDd
    };
  }

  function evoPerfTiles(s) {
    var host = el("div", "tiles");
    [
      { k: "Closed trades", v: String(s.n) },
      { k: "Win rate", v: s.winRate === null ? "—" : fpa(s.winRate, 1),
        s: s.n ? s.wins + " of " + s.n : "" },
      { k: "Cumulative", v: fp(s.total, 2), cls: polarity(s.total) },
      { k: "Expectancy / trade", v: fp(s.expectancy, 2), cls: polarity(s.expectancy) },
      { k: "Profit factor", v: fmt(s.profitFactor, 2) },
      { k: "Max drawdown", v: fpa(s.maxDd, 2), s: "compounded, closed trades" }
    ].forEach(function (t) {
      var d = el("div", "tile");
      d.appendChild(el("span", "k", t.k));
      d.appendChild(el("span", "v " + (t.cls || ""), t.v));
      if (t.s) d.appendChild(el("span", "s", t.s));
      host.appendChild(d);
    });
    return host;
  }

  function evoPerfChart(st, s) {
    var W = 900, m = { t: 12, r: 62, b: 20, l: 58 };
    var eqH = 132, ddH = 56, gap = 16;
    var H = m.t + eqH + gap + ddH + m.b, iw = W - m.l - m.r;
    var svg = svgEl("svg", { class: "plot", viewBox: "0 0 " + W + " " + H,
                             preserveAspectRatio: "none", role: "img",
                             "aria-label": "Equity curve and drawdown for trades closed so far" });
    // The x axis is the SAME bar timeline as the price chart, so a dip here
    // lines up with the candles that caused it.
    var X = function (i) { return m.l + (st.n <= 1 ? iw / 2 : (i / (st.n - 1)) * iw); };
    var vals = s.curve.map(function (p) { return p.eq; }).concat([1]);
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    var pad = (hi - lo) * 0.12 || 0.02;
    lo -= pad; hi += pad;
    var Y = function (v) { return m.t + eqH - ((v - lo) / (hi - lo)) * eqH; };

    niceTicks(lo, hi, 4).forEach(function (v) {
      if (v < lo || v > hi) return;
      svg.appendChild(svgEl("line", { class: "grid-line", x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v) }));
      svg.appendChild(svgText(m.l - 8, Y(v) + 3.5, fmt(v, 2) + "×", "tick", "end"));
    });
    svg.appendChild(svgEl("line", { class: "baseline", x1: m.l, x2: m.l + iw, y1: Y(1), y2: Y(1) }));

    // A STEP path, not a smoothed line: equity only moves when a trade closes,
    // and sloping between exits would invent intermediate values.
    if (s.curve.length) {
      var d = "M" + X(0).toFixed(1) + "," + Y(1).toFixed(1);
      var prev = 1;
      s.curve.forEach(function (p) {
        d += " L" + X(p.idx).toFixed(1) + "," + Y(prev).toFixed(1) +
             " L" + X(p.idx).toFixed(1) + "," + Y(p.eq).toFixed(1);
        prev = p.eq;
      });
      d += " L" + X(st.n - 1).toFixed(1) + "," + Y(prev).toFixed(1);
      svg.appendChild(svgEl("path", { class: "series s1", d: d }));
      // Direct end label: identity and value never by colour alone.
      svg.appendChild(svgText(m.l + iw + 8, Y(prev) + 4, fmt(prev, 3) + "×", "dlabel s1t", "start"));
    } else {
      svg.appendChild(svgText(m.l + iw / 2, m.t + eqH / 2, "No closed trades yet.", "tick", "middle"));
    }

    // ---- underwater
    var dTop = m.t + eqH + gap;
    var dMax = Math.max(s.maxDd, 0.02);
    var DY = function (v) { return dTop + (v / dMax) * ddH; };
    svg.appendChild(svgEl("line", { class: "axis-line", x1: m.l, x2: m.l + iw, y1: dTop, y2: dTop }));
    svg.appendChild(svgText(m.l - 8, dTop + 4, "0%", "tick", "end"));
    svg.appendChild(svgText(m.l - 8, DY(dMax) + 4, "-" + fmt(dMax * 100, 1) + "%", "tick", "end"));
    if (s.curve.length) {
      var a = "M" + X(0).toFixed(1) + "," + dTop.toFixed(1);
      var pd = 0;
      s.curve.forEach(function (p) {
        a += " L" + X(p.idx).toFixed(1) + "," + DY(pd).toFixed(1) +
             " L" + X(p.idx).toFixed(1) + "," + DY(p.dd).toFixed(1);
        pd = p.dd;
      });
      a += " L" + X(st.n - 1).toFixed(1) + "," + DY(pd).toFixed(1) +
           " L" + X(st.n - 1).toFixed(1) + "," + dTop.toFixed(1) + " Z";
      svg.appendChild(svgEl("path", { class: "evo-uw", d: a }));
    }

    // The price chart's view window, marked across both panels, so it is
    // obvious which slice of the performance you are inspecting. Skipped at
    // full extent, where a box around the entire plot says nothing and only
    // tints it.
    if (st.view.count < st.n) {
      var vx = X(st.view.from), vw = Math.max(2, X(st.view.from + st.view.count - 1) - vx);
      svg.appendChild(svgEl("rect", { class: "evo-ov-window",
        x: vx.toFixed(1), y: m.t, width: vw.toFixed(1), height: eqH + gap + ddH }));
    }
    return svg;
  }

  function evoPerfTable(closed) {
    var rows = closed.map(function (tr, i) {
      var t = tr.trade;
      return {
        n: String(i + 1),
        dir: String(t.direction || "?").toUpperCase(),
        pattern: t.pattern || "—",
        entry: day(t.entry_ts), exit: day(t.exit_ts),
        px: fmt(t.entry, 2) + " → " + fmt(t.exit_price, 2),
        rr: fmt(t.planned_rr, 2),
        outcome: el("span", "chip " + (Number(t.pnl_pct) > 0 ? "good" : "crit"), t.outcome || "—"),
        pnl: fp(Number(t.pnl_pct), 2)
      };
    }).reverse();   // most recent first: that is what you are reviewing
    return table([
      { key: "n", label: "#", num: true }, { key: "dir", label: "Side" },
      { key: "pattern", label: "Pattern" }, { key: "entry", label: "Entry" },
      { key: "exit", label: "Exit" }, { key: "px", label: "Price", num: true },
      { key: "rr", label: "R:R", num: true }, { key: "outcome", label: "Outcome" },
      { key: "pnl", label: "P&L", num: true }
    ], rows);
  }

  function evoDrawPerf(st, force) {
    var host = $("evo-perf");
    if (!host) return;
    if (!st) { clear(host); return; }
    var closed = st.trades.filter(function (tr) { return st.cursor - 1 >= tr.exitIdx; });
    // Rebuilding the panel on every timer tick is wasted work: it only changes
    // when a trade closes or the view window moves.
    var key = closed.length + ":" + st.view.from + ":" + st.view.count;
    if (!force && key === st.perfKey) return;
    st.perfKey = key;
    clear(host);
    var s = evoPerfStats(closed);
    host.appendChild(evoPerfTiles(s));
    var chartBox = el("div", "chart");
    chartBox.appendChild(evoPerfChart(st, s));
    host.appendChild(chartBox);
    if (closed.length) {
      var wrap = el("div", "scroll-x");
      wrap.appendChild(evoPerfTable(closed));
      host.appendChild(wrap);
    }
  }

  function evoBuildTrades(st, trades) {
    var ts = st.bars.ts;
    return (trades || []).map(function (t) {
      var entryIdx = evoClamp(evoBsearchFirstGE(ts, t.entry_ts), 0, st.n - 1);
      var exitIdx = evoClamp(evoBsearchFirstGE(ts, t.exit_ts), 0, st.n - 1);
      if (exitIdx < entryIdx) exitIdx = entryIdx;
      var out = { trade: t, entryIdx: entryIdx, exitIdx: exitIdx, hasPattern: false };
      // v0.3.2 WS-B (D4): pattern_start_ts is the "no geometry" sentinel
      // (engine.Trade never lets a real epoch-ms timestamp be 0 — see its
      // docstring). Only trades built on the graph path carry real pattern
      // geometry; a legacy-engine trade or a pre-v0.3.2 persisted run must
      // draw with NO zone rather than the old hardcoded box, so this stays
      // false for them and evoDrawTrade skips the whole block.
      if (t.pattern_start_ts && t.pattern_end_ts) {
        out.hasPattern = true;
        out.patternStartIdx = evoClamp(evoBsearchFirstGE(ts, t.pattern_start_ts), 0, st.n - 1);
        out.patternEndIdx = evoClamp(evoBsearchFirstGE(ts, t.pattern_end_ts), 0, st.n - 1);
        if (out.patternEndIdx < out.patternStartIdx) out.patternEndIdx = out.patternStartIdx;
      }
      return out;
    }).sort(function (a, b) { return a.entryIdx - b.entryIdx; });
  }

  function evoRenderPosition() {
    var st = EVO.chart;
    var host = $("evo-position");
    if (!host) return;
    clear(host);
    if (!st) return;
    var cur = evoClamp(EVO.cursor - 1, 0, st.n - 1);
    host.appendChild(el("p", "note",
      "Bar " + evoClamp(EVO.cursor, 0, st.n) + " of " + st.n + " · " + day(st.bars.ts[cur]) +
      " · viewing bars " + (st.view.from + 1) + "–" +
      Math.min(st.n, st.view.from + st.view.count) +
      (st.trades.length ? "" : " · no trades in this window")));
    st.trades.forEach(function (tr) {
      if (EVO.cursor - 1 < tr.entryIdx || EVO.cursor - 1 >= tr.exitIdx) return;
      var t = tr.trade;
      var isLong = String(t.direction).toLowerCase() === "long";
      var p = el("p", "evo-pos-line");
      p.appendChild(el("span", "chip " + (isLong ? "good" : "warn"),
        String(t.direction || "?").toUpperCase()));
      p.appendChild(el("span", null,
        " open · " + (t.pattern || "—") + "  entry " + fmt(t.entry, 2) +
        "  stop " + fmt(t.stop, 2) + "  target " + fmt(t.target, 2) +
        "  R:R " + fmt(t.planned_rr, 2)));
      host.appendChild(p);
    });
  }

  function evoSetCursor(cursor) {
    var st = EVO.chart;
    if (!st) return;
    cursor = evoClamp(Math.round(cursor), 0, st.n);
    EVO.cursor = cursor;
    st.cursor = cursor;
    // Follow the playhead when it runs off the edge of a zoomed view --
    // otherwise pressing play while zoomed in looks like nothing is happening.
    var last = st.view.from + st.view.count - 1;
    if (cursor - 1 > last) {
      evoSetView(st, cursor - 1 - Math.round(st.view.count * 0.8), st.view.count);
    } else if (cursor - 1 < st.view.from && st.view.count < st.n) {
      evoSetView(st, cursor - 1, st.view.count);
    } else {
      evoDraw(st);
      evoDrawOverview(st);
    }
    var scrub = $("evo-scrub");
    if (scrub) scrub.value = String(cursor);
    evoRenderPosition();
    evoDrawPerf(st);
  }

  function evoTogglePlay() {
    if (!EVO.chart) return;
    if (EVO.playing) { evoCancelTimer(); return; }
    EVO.playing = true;
    $("evo-btn-play").textContent = "⏸ pause";
    EVO.timer = setInterval(function () {
      if (!EVO.chart) { evoCancelTimer(); return; }
      var next = EVO.cursor + EVO.speed;
      if (next >= EVO.chart.n) { evoSetCursor(EVO.chart.n); evoCancelTimer(); return; }
      evoSetCursor(next);
    }, 50);
  }

  function evoResetPlayer() {
    EVO.member = null;
    EVO.chart = null;
    EVO.cursor = 0;
    $("evo-replay-body").hidden = true;
    $("evo-replay-empty").hidden = false;
    clear($("evo-chart"));
    clear($("evo-overview"));
    clear($("evo-position"));
    clear($("evo-perf"));
    $("evo-scrub").value = "0";
    $("evo-scrub").max = "0";
    $("evo-scrub").disabled = true;
    $("evo-btn-play").disabled = true;
    evoSetZoomEnabled(false);
  }

  function evoSetZoomEnabled(on) {
    ["evo-zoom-out", "evo-zoom-in", "evo-zoom-fit", "evo-zoom-trades"].forEach(function (id) {
      var b = $(id);
      if (b) b.disabled = !on;
    });
  }

  /* Fit the view to the span the trades actually occupy (plus their pattern
     zones and a little air) -- the fastest route from "loaded a member" to
     "looking at the geometry I care about". */
  function evoZoomTrades() {
    var st = EVO.chart;
    if (!st || !st.trades.length) return;
    var from = st.n, to = 0;
    st.trades.forEach(function (tr) {
      from = Math.min(from, tr.entryIdx - 24);
      to = Math.max(to, tr.exitIdx + 6);
    });
    from = evoClamp(from, 0, st.n - 1);
    to = evoClamp(to, from, st.n - 1);
    evoSetView(st, from, to - from + 1);
  }

  function renderReplay(d) {
    var bars = d.bars || { ts: [], o: [], h: [], l: [], c: [], v: [] };
    $("evo-replay-empty").hidden = true;
    $("evo-replay-body").hidden = false;
    clear($("evo-position"));
    var st = evoBuildChart(bars);
    EVO.chart = st;
    if (!st) {
      $("evo-scrub").value = "0"; $("evo-scrub").max = "0"; $("evo-scrub").disabled = true;
      $("evo-btn-play").disabled = true;
      evoSetZoomEnabled(false);
      // Zero-trade / zero-bar members (tier D, or an unstored window) still
      // need SOME position-panel text rather than a blank area.
      if (d.trades && d.trades.length) {
        $("evo-position").appendChild(el("p", "empty",
          d.trades.length + " trade(s) recorded, but no candles are stored for " +
          "this symbol/timeframe in the window."));
      } else {
        $("evo-position").appendChild(el("p", "note", "No trades in this window."));
      }
      return;
    }
    st.trades = evoBuildTrades(st, d.trades || []);
    $("evo-scrub").max = String(Math.max(0, st.n - 1));
    $("evo-scrub").disabled = false;
    $("evo-btn-play").disabled = false;
    evoSetZoomEnabled(true);
    evoSetCursor(Math.min(st.n, 1));
    evoDrawPerf(st, true);
  }

  function loadReplay(memberId, symbol, timeframe) {
    if (!memberId) return;
    // Never more than one timer alive: a fresh load always cancels first.
    evoCancelTimer();
    EVO.member = memberId;
    $("evo-replay-empty").hidden = true;
    $("evo-replay-body").hidden = false;
    $("evo-loading").hidden = false;
    clear($("evo-chart"));
    clear($("evo-position"));
    $("evo-scrub").disabled = true;
    $("evo-btn-play").disabled = true;
    var qs = "member=" + encodeURIComponent(memberId);
    if (symbol) qs += "&symbol=" + encodeURIComponent(symbol);
    if (timeframe) qs += "&timeframe=" + encodeURIComponent(timeframe);
    api("GET", "/api/evolution/replay?" + qs).then(function (d) {
      $("evo-loading").hidden = true;
      if (d.symbol && $("evo-symbol").querySelector('option[value="' + d.symbol + '"]')) {
        $("evo-symbol").value = d.symbol;
      }
      if (d.timeframe && $("evo-timeframe").querySelector('option[value="' + d.timeframe + '"]')) {
        $("evo-timeframe").value = d.timeframe;
      }
      renderReplay(d);
    }).catch(function (e) {
      $("evo-loading").hidden = true;
      showError("replay failed: " + e.message);
    });
  }

  function evoTierChip(tier) {
    var cls = tier === "A" ? "good" : tier === "B" ? "info" : tier === "C" ? "warn"
            : tier === "D" ? "crit" : "plain";
    return el("span", "chip " + cls, tier || "—");
  }

  function evoMemberNode(m, idx) {
    var b = el("button", "evo-member");
    b.type = "button";
    var ix = (m.member_index === null || m.member_index === undefined) ? idx : m.member_index;
    b.appendChild(el("span", "idx", "#" + ix));
    b.appendChild(el("span", "role", m.role || "—"));
    b.appendChild(evoTierChip(m.tier));
    b.appendChild(el("span", "fit", signed(firstFinite(m.fitness), 2)));
    if (m.error) b.title = "errored: " + m.error;
    b.onclick = function () {
      Array.prototype.forEach.call(document.querySelectorAll(".evo-member.active"), function (x) {
        x.classList.remove("active");
      });
      b.classList.add("active");
      loadReplay(m.member_id, $("evo-symbol").value, $("evo-timeframe").value);
    };
    return b;
  }

  function evoGenNode(campaign, g) {
    var det = el("details", "evo-gen");
    var sum = el("summary");
    sum.appendChild(el("span", "evo-gen-idx", "Gen " + g.gen_index));
    sum.appendChild(el("span", "evo-gen-best", "best " + signed(firstFinite(g.best_fitness), 2)));
    det.appendChild(sum);
    var body = el("div", "evo-member-list");
    body.appendChild(el("p", "note", "expand to load members"));
    det.appendChild(body);
    var loaded = false;
    det.addEventListener("toggle", function () {
      if (!det.open || loaded) return;
      loaded = true;
      clear(body);
      body.appendChild(el("p", "note", "loading…"));
      api("GET", "/api/evolution/members?campaign=" + encodeURIComponent(campaign) +
                 "&gen=" + encodeURIComponent(g.gen_index)).then(function (d) {
        clear(body);
        var members = d.members || [];
        if (!members.length) { body.appendChild(el("p", "empty", "No members.")); return; }
        members.forEach(function (m, idx) { body.appendChild(evoMemberNode(m, idx)); });
      }).catch(function (e) {
        clear(body);
        body.appendChild(el("p", "empty", "members unavailable: " + e.message));
        loaded = false;
      });
    });
    return det;
  }

  function evoLoadGenList() {
    var campaign = $("generations-campaign-select").value
                || $("generations-campaign").value.trim();
    var host = $("evo-gen-list");
    clear(host);
    EVO.campaign = campaign;
    if (!campaign) { host.appendChild(el("p", "empty", "No campaigns yet.")); return; }
    api("GET", "/api/generations?campaign=" + encodeURIComponent(campaign)).then(function (d) {
      var gens = (d.generations || []).slice().sort(function (a, b) { return a.gen_index - b.gen_index; });
      clear(host);
      if (!gens.length) { host.appendChild(el("p", "empty", "No generations yet.")); return; }
      gens.forEach(function (g) { host.appendChild(evoGenNode(campaign, g)); });
    }).catch(function (e) {
      clear(host);
      host.appendChild(el("p", "empty", "generations unavailable: " + e.message));
    });
  }

  function evoShowPane(name) {
    $("evo-replay-pane").hidden = name !== "replay";
    $("evo-table-pane").hidden = name !== "table";
    $("evo-tab-replay").setAttribute("aria-selected", name === "replay" ? "true" : "false");
    $("evo-tab-table").setAttribute("aria-selected", name === "table" ? "true" : "false");
  }

  function fillEvoSelectors() {
    var symSel = $("evo-symbol"), tfSel = $("evo-timeframe");
    clear(symSel); clear(tfSel);
    // NO hardcoded symbol/timeframe here -- both selects come straight from
    // /api/config, same as everywhere else in this file.
    (CONFIG.symbols || []).forEach(function (s) {
      var o = el("option", null, s); o.value = s; symSel.appendChild(o);
    });
    (CONFIG.timeframes || []).forEach(function (t) {
      var o = el("option", null, t); o.value = t; tfSel.appendChild(o);
    });
  }

  // --------------------------------------------------------------- routing
  var VIEWS = ["compose", "results", "data", "evolution", "reviews", "plugins"];
  function show(name) {
    if (VIEWS.indexOf(name) === -1) name = "compose";
    // The replay player's single timer must never survive a view switch away
    // from Evolution -- an orphaned interval is an explicit edge case in the
    // plan. This call is a no-op when no timer is running.
    if (name !== "evolution") evoCancelTimer();
    VIEWS.forEach(function (v) {
      var node = $("view-" + v);
      if (node) node.hidden = v !== name;
    });
    Array.prototype.forEach.call(document.querySelectorAll(".navlink"), function (b) {
      if (b.getAttribute("data-view") === name) b.setAttribute("aria-current", "page");
      else b.removeAttribute("aria-current");
    });
    if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
    if (name === "reviews") loadReviews();
    if (name === "plugins") renderPluginList();
    if (name === "evolution") evoLoadGenList();
  }

  function applyTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    try { localStorage.setItem("tb-theme", t); } catch (e) { /* private mode */ }
  }

  function fillStrategies(names) {
    var sel = $("load-strategy");
    var cur = sel.value;
    clear(sel);
    var first = el("option", null, names.length ? "(choose)" : "(none saved)");
    first.value = "";
    sel.appendChild(first);
    names.forEach(function (n) {
      var o = el("option", null, n);
      o.value = n;
      sel.appendChild(o);
    });
    if (cur) sel.value = cur;
  }

  // ------------------------------------------------------------------ boot
  function boot() {
    try {
      var saved = localStorage.getItem("tb-theme");
      if (saved) document.documentElement.setAttribute("data-theme", saved);
    } catch (e) { /* ignore */ }

    $("btn-theme").onclick = function () {
      applyTheme(document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light");
    };
    Array.prototype.forEach.call(document.querySelectorAll(".navlink"), function (b) {
      b.onclick = function () { show(b.getAttribute("data-view")); };
    });
    $("btn-add-detector").onclick = function () { addStage("detector"); };
    $("btn-add-confirmation").onclick = function () { addStage("confirmation"); };
    $("btn-add-filter").onclick = function () { addStage("filter"); };
    $("btn-validate").onclick = validateStrategy;
    $("btn-save").onclick = function () {
      saveStrategy()
        .then(function () { return api("GET", "/api/strategies"); })
        .then(function (d) { fillStrategies(d.strategies || []); })
        .catch(function (e) { showError("save failed: " + e.message); });
    };
    $("load-strategy").onchange = function () { loadStrategy(this.value); };
    $("run-campaign-select").onchange = renderCampaignStats;
    // This select is shared by the Evolution sidebar (Replay pane) AND the
    // Table pane's unchanged loadGenerations() -- both must refresh together.
    $("generations-campaign-select").onchange = function () {
      evoCancelTimer();
      evoResetPlayer();
      loadGenerations();
      evoLoadGenList();
    };
    $("btn-backtest").onclick = function () { startRun("backtest"); };
    $("btn-gate").onclick = function () { startRun("gate"); };
    $("btn-evolve").onclick = function () { startRun("evolve"); };
    $("btn-stop").onclick = stopRun;
    $("btn-copy-metrics").onclick = copyMetrics;
    $("btn-load-generations").onclick = loadGenerations;
    $("plugin-kind").onchange = renderPluginList;

    // ---- evolution replay player controls ----
    $("evo-tab-replay").onclick = function () { evoShowPane("replay"); };
    $("evo-tab-table").onclick = function () {
      // Pause while the Table pane is showing -- nothing needs to tick behind
      // a hidden chart, and this keeps the "at most one timer" invariant easy
      // to reason about (the ONLY places a timer is ever running are inside
      // the visible Replay pane).
      evoCancelTimer();
      evoShowPane("table");
      loadGenerations();
    };
    $("evo-btn-reset").onclick = function () { evoCancelTimer(); evoSetCursor(0); };
    // Zoom buttons are the trackpad-free path to the same viewport the wheel
    // and the overview strip drive; they anchor on the view's centre.
    $("evo-zoom-in").onclick = function () { evoZoom(EVO.chart, 1 / 1.6, 0.5); };
    $("evo-zoom-out").onclick = function () { evoZoom(EVO.chart, 1.6, 0.5); };
    $("evo-zoom-fit").onclick = function () {
      if (EVO.chart) evoSetView(EVO.chart, 0, EVO.chart.n);
    };
    $("evo-zoom-trades").onclick = evoZoomTrades;
    $("evo-btn-play").onclick = evoTogglePlay;
    Array.prototype.forEach.call(document.querySelectorAll("#evo-speed-group .seg-btn"), function (b) {
      b.onclick = function () {
        EVO.speed = Number(b.getAttribute("data-speed")) || 1;
        Array.prototype.forEach.call(document.querySelectorAll("#evo-speed-group .seg-btn"), function (x) {
          x.classList.toggle("active", x === b);
        });
      };
    });
    $("evo-scrub").oninput = function () { evoCancelTimer(); evoSetCursor(Number(this.value)); };
    $("evo-symbol").onchange = function () {
      if (EVO.member) loadReplay(EVO.member, this.value, $("evo-timeframe").value);
    };
    $("evo-timeframe").onchange = function () {
      if (EVO.member) loadReplay(EVO.member, $("evo-symbol").value, this.value);
    };

    Promise.all([
      api("GET", "/api/config"),
      api("GET", "/api/plugins"),
      api("GET", "/api/data/coverage"),
      api("GET", "/api/strategies"),
      api("GET", "/api/runs")
    ]).then(function (r) {
      var cfg = r[0] || {}, plug = r[1] || {}, cov = r[2] || {},
          strat = r[3] || {}, runs = r[4] || {};
      CONFIG = cfg;

      $("meta-strategy-dir").textContent = cfg.strategy_dir || "data/strategies";
      $("meta-state-db").textContent = cfg.state_db || "data/state.db";
      fillEvoSelectors();

      // /api/plugins returns `plugins` ALREADY KEYED BY KIND (a dict of arrays),
      // not a flat list. Copy it rather than re-grouping.
      var byKind = plug.plugins || {};
      Object.keys(byKind).forEach(function (kind) {
        PLUGINS[kind] = (byKind[kind] || []).slice();
      });
      var kindSel = $("plugin-kind");
      (plug.kinds || Object.keys(PLUGINS)).forEach(function (k) {
        var o = el("option", null, k);
        o.value = k;
        kindSel.appendChild(o);
      });

      renderCoverage(cov);
      fillStrategies(strat.strategies || []);
      // After fillStrategies, so the campaign list is scoped to whichever
      // strategy the composer ends up showing.
      refreshCampaigns();

      RUNS = runs.runs || [];
      if (RUNS.length) ACTIVE_RUN = RUNS[0].run_id;
      renderTabs();

      // Seed the composer with a minimal valid-shaped pipeline.
      if (PLUGINS.detector && PLUGINS.detector.length) {
        STAGES = [
          { kind: "data",
            key: (PLUGINS.data && PLUGINS.data[0]) ? PLUGINS.data[0].key : "data.ohlcv",
            params: {} },
          { kind: "detector", key: PLUGINS.detector[0].key, params: {} }
        ];
        if (PLUGINS.policy && PLUGINS.policy.length) {
          STAGES.push({ kind: "policy", key: PLUGINS.policy[0].key, params: {} });
        }
      }
      renderPipe();

      show(String(location.hash || "#compose").slice(1));
      if (ACTIVE_RUN) selectRun(ACTIVE_RUN);
      RUNS.forEach(function (r2) { if (r2.status === "running") watchRun(r2.run_id); });
    }).catch(function (e) {
      showError("startup failed: " + e.message);
      show("compose");
    });
  }

  window.addEventListener("hashchange", function () {
    show(String(location.hash || "#compose").slice(1));
  });
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
