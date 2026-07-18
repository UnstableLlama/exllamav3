"""Self-contained local run logging + HTML report -- the default replacement for
wandb on the exl3-qlora trainers.

Why this exists: the wandb free tier under a team entity can't host public
projects, and sharing training runs publicly is a requirement. So instead of a
third-party platform, each run writes its own data to disk and renders a single
self-contained ``report.html`` (inline CSS + vanilla-JS SVG charts, **no CDN**,
no account) that opens locally or hosts anywhere.

Layout, all under the run's ``out`` dir so the report travels with the adapter::

    <out>/run_report/
        config.json     # run identity + hyperparameters (mirrors the CSV row)
        metrics.jsonl   # one JSON object per log() call: {"step": N, "<metric>": v, ...}
        summary.json    # final scalars (best val, throughput, peak VRAM, ...)
        report.html     # rendered from the three files above

Metric names use wandb-style ``group/name`` keys (``train/loss``,
``eval/held_out``, ``perf/tot_tok_s``); the report groups charts by the prefix.
The HTML is rebuildable from ``metrics.jsonl`` alone, so a crashed run still
renders whatever it logged (``render_report(out_dir)`` regenerates in place).

Comparison: ``compare_reports([outA, outB, ...])`` overlays several runs on one
page -- each metric chart draws one colored line per run with a shared legend --
for A/B runs (e.g. SFT vs EBFT). Same from the CLI::

    python training/run_report.py <out_dir>                 # (re)render one run
    python training/run_report.py <outA> <outB> -o cmp.html # overlay runs

Drop-in shape mirrors the wandb calls it replaces::

    rep = RunLogger(out_dir, run_name, config)     # ~ wandb.init(...)
    rep.log({"train/loss": x}, step=step)          # ~ run.log(..., step=step)
    rep.summary.update({"best_val": v})            # ~ run.summary.update(...)
    rep.finish(exit_code=0)                        # ~ run.finish(...) -> writes report.html
"""

import datetime
import json
import os


REPORT_SUBDIR = "run_report"


def _report_dir(out_dir):
    return os.path.join(out_dir, REPORT_SUBDIR)


def _json_safe(v):
    """Best-effort coerce a value into something json.dump can write."""
    if isinstance(v, bool) or v is None or isinstance(v, (int, float, str)):
        return v
    # torch scalars / numpy floats expose .item(); fall back to str.
    item = getattr(v, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    try:
        json.dumps(v)
        return v
    except TypeError:
        return str(v)


class RunLogger:
    """Local, self-contained run logger. One instance per training run.

    Streams every ``log()`` call to ``metrics.jsonl`` (flushed each write, so a
    hard kill still leaves the rows already logged) and renders ``report.html``
    on ``finish()``. ``finish()`` is idempotent -- the first caller (normal
    finish, Ctrl-C, or the failure logger) wins -- matching the wandb teardown
    it stands in for.
    """

    def __init__(self, out_dir, run_name, config=None, meta=None):
        self.out_dir = out_dir
        self.dir = _report_dir(out_dir)
        os.makedirs(self.dir, exist_ok=True)
        self.run_name = run_name
        self.summary = {}
        self._closed = False
        self._meta = {
            "run_name": run_name,
            "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "status": "running",
        }
        if meta:
            self._meta.update(meta)
        # Fresh metrics stream each run (an --out reuse starts a new report).
        self._metrics_path = os.path.join(self.dir, "metrics.jsonl")
        self._fh = open(self._metrics_path, "w", encoding="utf-8")
        self._write_json("config.json", {
            "meta": self._meta,
            "config": {k: _json_safe(v) for k, v in (config or {}).items()},
        })

    def _write_json(self, name, obj):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=False)

    def log(self, metrics, step=None):
        """Append one metrics row. Keys are wandb-style ``group/name`` strings."""
        if self._closed or self._fh is None:
            return
        row = {"step": step}
        for k, v in metrics.items():
            row[k] = _json_safe(v)
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def update_summary(self, d):
        self.summary.update({k: _json_safe(v) for k, v in d.items()})

    def finish(self, exit_code=0, status=None):
        if self._closed:
            return
        self._closed = True
        self._meta["status"] = status or ("completed" if exit_code == 0 else "failed")
        self._meta["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
        self._meta["exit_code"] = exit_code
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None
        # Re-persist config (meta now has final status) and the summary.
        cfg = _read_json(os.path.join(self.dir, "config.json")) or {}
        cfg["meta"] = self._meta
        self._write_json("config.json", cfg)
        self._write_json("summary.json", self.summary)
        try:
            render_report(self.out_dir)
            print(f"[report] {os.path.join(self.dir, 'report.html')}")
        except Exception as exc:  # a render bug must never mask the run's own exit
            print(f"[report] render failed: {exc}")


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _read_metrics(path):
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue  # skip a torn final line from a hard kill
    except OSError:
        return []
    return rows


def _load_run(out_dir, label=None):
    """Read one run's config/metrics/summary off disk into the payload shape the
    HTML template consumes: {meta, config, summary, series}. ``series`` maps each
    metric name to its [[step, value], ...] points (non-finite values dropped)."""
    rdir = _report_dir(out_dir)
    cfg = _read_json(os.path.join(rdir, "config.json")) or {}
    meta = dict(cfg.get("meta", {}))
    config = cfg.get("config", {})
    summary = _read_json(os.path.join(rdir, "summary.json")) or {}
    rows = _read_metrics(os.path.join(rdir, "metrics.jsonl"))

    series = {}
    for row in rows:
        step = row.get("step")
        for k, v in row.items():
            if k == "step":
                continue
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            if v != v or v in (float("inf"), float("-inf")):  # NaN / inf
                continue
            series.setdefault(k, []).append([step, v])

    meta["label"] = (label if label is not None
                     else meta.get("run_name")
                     or os.path.basename(os.path.normpath(out_dir)))
    return {"meta": meta, "config": config, "summary": summary, "series": series}


def _write_html(payload, out_path):
    html = _HTML_TEMPLATE.replace(
        "/*__DATA__*/", json.dumps(payload, separators=(",", ":")))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def render_report(out_dir):
    """(Re)render ``<out>/run_report/report.html`` from the on-disk config/
    metrics/summary. Usable standalone to regenerate a report (e.g. after a
    crash left only the jsonl)."""
    payload = {"runs": [_load_run(out_dir)]}
    return _write_html(payload, os.path.join(_report_dir(out_dir), "report.html"))


def compare_reports(out_dirs, output_path=None, labels=None):
    """Render one HTML page overlaying several runs for comparison. Each metric
    chart draws a colored line per run with a shared legend; summary and config
    become per-run columns (differing config rows highlighted). Runs with no
    metrics are skipped with a warning."""
    runs = []
    for i, d in enumerate(out_dirs):
        lbl = labels[i] if labels and i < len(labels) else None
        run = _load_run(d, label=lbl)
        if not run["series"]:
            print(f"[report] warning: no metrics under {d}, skipping")
            continue
        runs.append(run)
    if not runs:
        raise SystemExit("no runs with metrics to compare")
    # Disambiguate duplicate labels (e.g. two runs both basename 'exl3_ebft').
    seen = {}
    for r in runs:
        lbl = r["meta"]["label"]
        seen[lbl] = seen.get(lbl, 0) + 1
        if seen[lbl] > 1:
            r["meta"]["label"] = f"{lbl} #{seen[lbl]}"
    output_path = output_path or "compare_report.html"
    _write_html({"runs": runs}, output_path)
    print(f"[report] {output_path}  ({len(runs)} runs)")
    return output_path


# --------------------------------------------------------------------------- #
# Self-contained HTML template. All CSS/JS inline; the run data is injected as a
# single JSON blob at /*__DATA__*/ in the shape {runs: [{meta,config,summary,
# series}, ...]}. One run -> single-run report (summary cards); several runs ->
# overlay-comparison report (legend + one line per run per chart, per-run
# summary/config columns). Charts are drawn to SVG by the vanilla JS below -- no
# external requests, so the file works offline and hosts anywhere. Light/dark
# follow the viewer's OS preference.
# --------------------------------------------------------------------------- #
_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>training run report</title>
<style>
  :root {
    --bg: #ffffff; --panel: #f7f8fa; --border: #e3e6ea; --fg: #1a1d21;
    --muted: #6b7280; --grid: #eceef1; --accent: #2563eb;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f1216; --panel: #171b21; --border: #262b33; --fg: #e6e9ee;
      --muted: #8a93a0; --grid: #21262e; --accent: #60a5fa;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 28px 20px 80px; }
  header { border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 16px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  h2 { font-size: 15px; margin: 28px 0 12px; color: var(--muted);
       text-transform: uppercase; letter-spacing: .04em; }
  .sub { color: var(--muted); font-size: 13px; }
  .badge { display: inline-block; padding: 2px 9px; border-radius: 999px;
           font-size: 12px; font-weight: 600; vertical-align: middle; }
  .b-completed { background: #10b98122; color: #10b981; }
  .b-failed, .b-interrupted { background: #ef444422; color: #ef4444; }
  .b-running { background: #f59e0b22; color: #f59e0b; }
  .legend { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 4px; }
  .chip { display: inline-flex; align-items: center; gap: 7px; background: var(--panel);
          border: 1px solid var(--border); border-radius: 999px; padding: 4px 12px; font-size: 13px; }
  .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; flex: none; }
  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px,1fr));
           gap: 10px; margin: 8px 0 4px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
          padding: 11px 13px; }
  .card .k { color: var(--muted); font-size: 12px; }
  .card .v { font-size: 18px; font-weight: 600; margin-top: 2px;
             font-variant-numeric: tabular-nums; word-break: break-word; }
  .charts { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px,1fr)); gap: 14px; }
  .chart { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
           padding: 12px 12px 6px; }
  .chart .title { font-size: 13px; font-weight: 600; margin-bottom: 2px;
                  display: flex; justify-content: space-between; align-items: baseline; }
  .chart .cur { color: var(--muted); font-weight: 400; font-variant-numeric: tabular-nums; }
  .svg-holder { position: relative; }
  svg { width: 100%; display: block; }
  .grid-line { stroke: var(--grid); stroke-width: 1; }
  .axis-txt { fill: var(--muted); font-size: 10px; }
  .plot { fill: none; stroke-width: 1.6; }
  .dot { stroke: var(--bg); stroke-width: 1; }
  .crosshair { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; opacity: 0; }
  .tt { position: absolute; pointer-events: none; background: var(--bg);
        border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px;
        font-size: 11px; line-height: 1.6; opacity: 0; transition: opacity .05s;
        white-space: nowrap; box-shadow: 0 2px 10px rgba(0,0,0,.18); z-index: 2; top: 6px; }
  .tt .sw { margin-right: 5px; vertical-align: baseline; }
  .tt .row { font-variant-numeric: tabular-nums; }
  table.cfg, table.cmp { border-collapse: collapse; width: 100%; font-size: 13px; }
  table.cfg td { border-bottom: 1px solid var(--border); padding: 5px 10px 5px 0; vertical-align: top; }
  table.cfg td.k { color: var(--muted); white-space: nowrap; width: 1%; padding-right: 22px; }
  table.cfg td.v { font-variant-numeric: tabular-nums; word-break: break-word; }
  .cfg-wrap { columns: 2; column-gap: 34px; }
  @media (max-width: 640px) { .cfg-wrap { columns: 1; } }
  .cmp-wrap { overflow-x: auto; }
  table.cmp th, table.cmp td { border-bottom: 1px solid var(--border); padding: 6px 16px 6px 0;
                               text-align: left; font-variant-numeric: tabular-nums; vertical-align: top; }
  table.cmp th { color: var(--fg); font-weight: 600; }
  table.cmp td.k, table.cmp th.k { color: var(--muted); font-weight: 400; white-space: nowrap; }
  table.cmp th .sw { margin-right: 6px; }
  tr.diff td { background: color-mix(in srgb, var(--accent) 10%, transparent); }
  .toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-top: 12px; }
  .tbtn { font: inherit; font-size: 13px; cursor: pointer; background: var(--panel);
          color: var(--fg); border: 1px solid var(--border); border-radius: 8px; padding: 6px 12px; }
  .tbtn:hover { border-color: var(--accent); }
  #importMsg { margin-left: 2px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1 id="title"></h1>
    <div class="sub" id="subtitle"></div>
    <div id="legend"></div>
    <div class="toolbar">
      <button id="importBtn" class="tbtn" type="button">＋ Overlay another report…</button>
      <button id="resetBtn" class="tbtn" type="button" hidden>Reset</button>
      <input id="importInput" type="file" accept=".html,text/html" multiple hidden>
      <span class="sub" id="importMsg"></span>
    </div>
  </header>
  <div id="summary"></div>
  <h2>metrics</h2>
  <div class="charts" id="charts"></div>
  <h2>config</h2>
  <div id="config"></div>
</div>
<script>
const DATA = /*__DATA__*/;
const PALETTE = ["#3b82f6","#ef4444","#10b981","#f59e0b","#8b5cf6","#ec4899","#14b8a6","#f97316"];
const BASE_RUNS = (DATA.runs || []);
let RUNS = BASE_RUNS.slice();  // working set; overlay import appends, reset restores
function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

function el(tag, attrs, kids) {
  const svgTags = {svg:1, path:1, line:1, circle:1, text:1, g:1};
  const e = document.createElementNS(
    svgTags[tag] ? "http://www.w3.org/2000/svg" : "http://www.w3.org/1999/xhtml", tag);
  for (const k in (attrs||{})) {
    if (k === "html") e.innerHTML = attrs[k];
    else if (k === "text") e.textContent = attrs[k];
    else e.setAttribute(k, attrs[k]);
  }
  (kids||[]).forEach(c => e.appendChild(c));
  return e;
}
function fmt(v) {
  if (v === null || v === undefined) return "–";
  if (typeof v !== "number") return String(v);
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e6 || (a < 1e-3 && a > 0)) return v.toExponential(2);
  if (Number.isInteger(v)) return v.toLocaleString();
  if (a >= 100) return v.toFixed(1);
  return v.toFixed(4);
}
function swatch(color) { const s = el("span", {class: "sw"}); s.setAttribute("style", "background:" + color); return s; }

// Nice x-axis ticks: endpoints always shown, plus a few interior ticks at round
// step values; interior ticks too close to an endpoint are dropped to avoid
// label overlap.
function xticks(min, max, count) {
  min = Math.round(min); max = Math.round(max);
  if (min === max) return [min];
  const span = max - min, raw = span / (count - 1);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const t = new Set([min, max]);
  for (let v = Math.ceil(min / step) * step; v < max; v += step)
    if (v > min) t.add(Math.round(v));
  const arr = [...t].sort((a, b) => a - b);
  const tol = span * 0.08;
  return arr.filter((v, i) => i === 0 || i === arr.length - 1 || (v - min > tol && max - v > tol));
}

function unionKeys(objs) {
  const seen = [], set = new Set();
  objs.forEach(o => Object.keys(o || {}).forEach(k => { if (!set.has(k)) { set.add(k); seen.push(k); } }));
  return seen;
}

// ---- render the whole page from a runs array; called on load and on every
// overlay import, so all chart scales recompute across the current run set ----
function render(runs) {
  const MULTI = runs.length > 1;

  // header
  const title = document.getElementById("title");
  const subtitle = document.getElementById("subtitle");
  const legendRoot = document.getElementById("legend");
  clear(legendRoot); title.textContent = ""; subtitle.textContent = "";
  if (MULTI) {
    title.textContent = "Comparison — " + runs.length + " runs";
    const leg = el("div", {class: "legend"});
    runs.forEach(r => {
      const st = (r.meta && r.meta.status || "").toLowerCase();
      const chip = el("span", {class: "chip"}, [swatch(r.color), el("span", {text: r.meta.label})]);
      if (st) chip.appendChild(el("span", {class: "badge b-" + st, text: st}));
      leg.appendChild(chip);
    });
    legendRoot.appendChild(leg);
  } else {
    const m = (runs[0] && runs[0].meta) || {};
    const cfg0 = (runs[0] && runs[0].config) || {};
    title.textContent = m.run_name || "training run";
    const status = (m.status || "running").toLowerCase();
    const dur = (m.started && m.finished)
      ? Math.round((new Date(m.finished) - new Date(m.started)) / 1000) : null;
    subtitle.textContent = [
      cfg0.model || cfg0.model_name || "",
      m.started ? ("started " + m.started.replace("T", " ")) : "",
      dur != null ? (dur + "s wall") : "",
    ].filter(Boolean).join("  ·  ") + "  ";
    subtitle.appendChild(el("span", {class: "badge b-" + status, text: status}));
  }

  // summary
  const summaryRoot = document.getElementById("summary"); clear(summaryRoot);
  if (!MULTI) {
    const sum = (runs[0] && runs[0].summary) || {};
    const kk = Object.keys(sum);
    if (kk.length) {
      const cards = el("div", {class: "cards"});
      kk.forEach(k => cards.appendChild(el("div", {class: "card"}, [
        el("div", {class: "k", text: k}), el("div", {class: "v", text: fmt(sum[k])})])));
      summaryRoot.appendChild(cards);
    }
  } else {
    const skeys = unionKeys(runs.map(r => r.summary));
    if (skeys.length) {
      const tbl = el("table", {class: "cmp"});
      const head = el("tr", {}, [el("th", {class: "k", text: ""})]);
      runs.forEach(r => head.appendChild(el("th", {}, [swatch(r.color), el("span", {text: r.meta.label})])));
      tbl.appendChild(head);
      skeys.forEach(k => {
        const tr = el("tr", {}, [el("td", {class: "k", text: k})]);
        runs.forEach(r => tr.appendChild(el("td", {text: fmt((r.summary || {})[k])})));
        tbl.appendChild(tr);
      });
      summaryRoot.appendChild(el("div", {class: "cmp-wrap"}, [tbl]));
    }
  }

  // charts
function chart(name, seriesList, multi) {
  const W = 360, H = 158, PL = 44, PR = 12, PT = 10, PB = 26;
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  seriesList.forEach(s => s.points.forEach(p => {
    if (p[0] < xmin) xmin = p[0]; if (p[0] > xmax) xmax = p[0];
    if (p[1] < ymin) ymin = p[1]; if (p[1] > ymax) ymax = p[1];
  }));
  if (xmin === xmax) xmax = xmin + 1;
  if (ymin === ymax) { const p = Math.abs(ymin) || 1; ymin -= p * 0.05; ymax += p * 0.05; }
  const padY = (ymax - ymin) * 0.08; ymin -= padY; ymax += padY;
  const sx = v => PL + (v - xmin) / (xmax - xmin) * (W - PL - PR);
  const sy = v => PT + (1 - (v - ymin) / (ymax - ymin)) * (H - PT - PB);

  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`});
  for (let i = 0; i <= 3; i++) {
    const yv = ymin + (ymax - ymin) * i / 3, y = sy(yv);
    svg.appendChild(el("line", {class: "grid-line", x1: PL, y1: y, x2: W - PR, y2: y}));
    svg.appendChild(el("text", {class: "axis-txt", x: PL - 5, y: y + 3, "text-anchor": "end", text: fmt(yv)}));
  }
  xticks(xmin, xmax, 5).forEach(xv => {
    const x = sx(xv);
    if (xv > xmin && xv < xmax)
      svg.appendChild(el("line", {class: "grid-line", x1: x, y1: PT, x2: x, y2: H - PB}));
    svg.appendChild(el("text", {class: "axis-txt", x: x, y: H - 8, "text-anchor": "middle", text: String(xv)}));
  });

  const dots = [];
  seriesList.forEach(s => {
    const d = s.points.map((p, i) => (i ? "L" : "M") + sx(p[0]).toFixed(1) + " " + sy(p[1]).toFixed(1)).join(" ");
    const path = el("path", {class: "plot", d}); path.setAttribute("stroke", s.color); svg.appendChild(path);
    const dot = el("circle", {class: "dot", r: 3, cx: -10, cy: -10, opacity: 0}); dot.setAttribute("fill", s.color);
    svg.appendChild(dot); dots.push({s, dot});
  });
  const cross = el("line", {class: "crosshair", x1: 0, y1: PT, x2: 0, y2: H - PB}); svg.appendChild(cross);

  const box = el("div", {class: "chart"});
  const titleRow = el("div", {class: "title"}, [el("span", {text: name})]);
  if (!multi) {
    const pts = seriesList[0].points;
    titleRow.appendChild(el("span", {class: "cur", text: "last " + fmt(pts[pts.length - 1][1])}));
  }
  box.appendChild(titleRow);
  const holder = el("div", {class: "svg-holder"});
  const tip = el("div", {class: "tt"});
  holder.appendChild(svg); holder.appendChild(tip);
  box.appendChild(holder);

  svg.addEventListener("pointermove", ev => {
    const r = svg.getBoundingClientRect();
    const px = (ev.clientX - r.left) / r.width * W;
    let bx = null, bd = Infinity;
    seriesList.forEach(s => s.points.forEach(p => { const dd = Math.abs(sx(p[0]) - px); if (dd < bd) { bd = dd; bx = p[0]; } }));
    if (bx === null) return;
    cross.setAttribute("x1", sx(bx)); cross.setAttribute("x2", sx(bx)); cross.setAttribute("opacity", 1);
    let rows = '<div class="row" style="color:var(--muted)">step ' + bx + '</div>';
    dots.forEach(({s, dot}) => {
      let pp = null, pd = Infinity;
      s.points.forEach(p => { const dd = Math.abs(p[0] - bx); if (dd < pd) { pd = dd; pp = p; } });
      if (pp) {
        dot.setAttribute("cx", sx(pp[0])); dot.setAttribute("cy", sy(pp[1])); dot.setAttribute("opacity", 1);
        rows += '<div class="row"><span class="sw" style="background:' + s.color + '"></span>'
             + (multi ? s.label + ": " : "") + fmt(pp[1]) + '</div>';
      }
    });
    tip.innerHTML = rows; tip.style.opacity = 1;
    const w = r.width;
    const left = Math.min(sx(bx) / W * w + 12, w - tip.offsetWidth - 6);
    tip.style.left = Math.max(4, left) + "px";
  });
  svg.addEventListener("pointerleave", () => {
    cross.setAttribute("opacity", 0); dots.forEach(({dot}) => dot.setAttribute("opacity", 0)); tip.style.opacity = 0;
  });
  return box;
}

  // charts: metric union across runs, grouped by prefix; chart() recomputes
  // its own x/y domains from the union of series handed to it.
  const groupOrder = ["train", "eval", "perf"];
  const metricNames = unionKeys(runs.map(r => r.series));
  const groups = {};
  metricNames.forEach(k => {
    const g = k.includes("/") ? k.split("/")[0] : "other";
    (groups[g] = groups[g] || []).push(k);
  });
  const gnames = Object.keys(groups).sort((a, b) => {
    const ia = groupOrder.indexOf(a), ib = groupOrder.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
  });
  const chartsRoot = document.getElementById("charts"); clear(chartsRoot);
  gnames.forEach(g => {
    groups[g].sort().forEach(metric => {
      const seriesList = [];
      runs.forEach(r => {
        const pts = (r.series[metric] || []).filter(p => p[0] !== null && p[0] !== undefined);
        if (pts.length) seriesList.push({label: r.meta.label, color: r.color, points: pts});
      });
      if (seriesList.length) chartsRoot.appendChild(chart(metric, seriesList, MULTI));
    });
  });
  if (!chartsRoot.children.length)
    chartsRoot.appendChild(el("div", {class: "sub", text: "no metrics logged"}));

  // config
  const configRoot = document.getElementById("config"); clear(configRoot);
  if (!MULTI) {
    const cfg = (runs[0] && runs[0].config) || {};
    const wrap = el("div", {class: "cfg-wrap"});
    const t = el("table", {class: "cfg"});
    Object.keys(cfg).forEach(k => t.appendChild(el("tr", {}, [
      el("td", {class: "k", text: k}), el("td", {class: "v", text: fmt(cfg[k])})])));
    wrap.appendChild(t); configRoot.appendChild(wrap);
  } else {
    const ckeys = unionKeys(runs.map(r => r.config));
    const tbl = el("table", {class: "cmp"});
    const head = el("tr", {}, [el("th", {class: "k", text: ""})]);
    runs.forEach(r => head.appendChild(el("th", {}, [swatch(r.color), el("span", {text: r.meta.label})])));
    tbl.appendChild(head);
    ckeys.forEach(k => {
      const vals = runs.map(r => (r.config || {})[k]);
      const differ = vals.some(v => JSON.stringify(v) !== JSON.stringify(vals[0]));
      const tr = el("tr", differ ? {class: "diff"} : {}, [el("td", {class: "k", text: k})]);
      vals.forEach(v => tr.appendChild(el("td", {text: fmt(v)})));
      tbl.appendChild(tr);
    });
    configRoot.appendChild(el("div", {class: "cmp-wrap"}, [tbl]));
  }
}

// ---- assign per-run colors + disambiguate duplicate labels, then render ----
function prep(runs) {
  const out = runs.map((r, i) => ({
    meta: Object.assign({}, r.meta || {}),
    config: r.config || {}, summary: r.summary || {}, series: r.series || {},
    color: PALETTE[i % PALETTE.length],
  }));
  const seen = {};
  out.forEach(r => {
    const base = r.meta.label || r.meta.run_name || "run";
    seen[base] = (seen[base] || 0) + 1;
    r.meta.label = seen[base] > 1 ? base + " #" + seen[base] : base;
  });
  return out;
}
function rerender() {
  render(prep(RUNS));
  document.getElementById("resetBtn").hidden = RUNS.length <= BASE_RUNS.length;
}

// ---- overlay import: read another report.html, pull its DATA.runs, append ----
function extractRuns(text) {
  const marker = "const DATA = ";
  const i = text.indexOf(marker);
  if (i < 0) throw new Error("not a run report");
  const rest = text.slice(i + marker.length);
  const end = rest.indexOf(";\n");
  let blob = (end < 0 ? rest : rest.slice(0, end)).trim();
  if (blob.endsWith(";")) blob = blob.slice(0, -1);
  const runs = JSON.parse(blob).runs || [];
  if (!runs.length) throw new Error("no runs in file");
  return runs;
}
const importInput = document.getElementById("importInput");
document.getElementById("importBtn").addEventListener("click", () => importInput.click());
document.getElementById("resetBtn").addEventListener("click", () => {
  RUNS = BASE_RUNS.slice(); rerender();
  document.getElementById("importMsg").textContent = "";
});
importInput.addEventListener("change", async ev => {
  const files = Array.from(ev.target.files || []);
  let added = 0; const errs = [];
  for (const f of files) {
    try { extractRuns(await f.text()).forEach(r => RUNS.push(r)); added += 1; }
    catch (e) { errs.push(f.name + " — " + e.message); }
  }
  ev.target.value = "";  // let the same file be re-picked later
  rerender();
  const msg = [];
  if (added) msg.push("overlaid " + added + " report" + (added > 1 ? "s" : ""));
  if (errs.length) msg.push("skipped: " + errs.join("; "));
  document.getElementById("importMsg").textContent = msg.join("  ·  ");
});

// initial paint
rerender();
</script>
</body>
</html>
"""


def _main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Render or compare exl3-qlora run reports. One dir -> render "
                    "that run's report.html in place; several dirs -> overlay them "
                    "on one comparison page.")
    ap.add_argument("out_dirs", nargs="+",
                    help="run --out dir(s) (each containing a run_report/ subdir).")
    ap.add_argument("-o", "--output", default=None,
                    help="output HTML path for compare mode (default: compare_report.html).")
    ap.add_argument("--labels", default=None,
                    help="comma-separated legend labels for the runs (compare mode; "
                         "default: each run's name).")
    args = ap.parse_args()
    if len(args.out_dirs) == 1 and not args.labels:
        print(render_report(args.out_dirs[0]))
    else:
        labels = args.labels.split(",") if args.labels else None
        compare_reports(args.out_dirs, args.output, labels)


if __name__ == "__main__":
    _main()
