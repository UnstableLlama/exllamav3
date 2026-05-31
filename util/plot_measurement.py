import sys, os, json, argparse, html

# Standalone dark-mode SVG bar chart for a measurement.json produced by util/measure.py.
#
# X axis: tensor group buckets in numerical (idx) order.
# Y axis: delta KL-div of the candidate (alt) bpw vs the base bpw in that file.
#         Negative = KL improved when raising bitrate (group benefits from more bits).
#         Positive = KL got worse when raising bitrate (group is "better" at the lower bpw).
#
# No third-party dependencies; emits an SVG directly.

# Dark theme palette (github-dark-ish)
COL_BG      = "#0d1117"
COL_PANEL   = "#161b22"
COL_TEXT    = "#c9d1d9"
COL_MUTED   = "#8b949e"
COL_GRID    = "#30363d"
COL_ZERO    = "#6e7681"
COL_NEG     = "#3fb950"  # improves with more bits (what we're usually hunting for)
COL_POS     = "#f85149"  # regresses with more bits (better at lower bpw)


def lcp(seqs):
    if not seqs: return []
    m = min(len(s) for s in seqs)
    i = 0
    while i < m and all(s[i] == seqs[0][i] for s in seqs):
        i += 1
    return seqs[0][:i]


def short_name(layers, global_prefix_len):
    # Lightweight, torch-free group label: strip the common model prefix and collapse.
    if not layers:
        return "(empty)"
    stripped = [l[global_prefix_len:] for l in layers]
    base = stripped[0].replace("_proj", "")
    if len(stripped) > 1:
        return f"{base} (+{len(stripped) - 1})"
    return base


def nice_ticks(lo, hi, target = 6):
    # Produce ~target nicely rounded tick values spanning [lo, hi].
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    raw = span / target
    mag = 10 ** math_floor_log10(raw)
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if span / step <= target * 1.5:
            break
    start = math_ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 1e-6:
        ticks.append(round(v, 12))
        v += step
    return ticks


def math_floor_log10(x):
    import math
    if x <= 0: return 0
    return math.floor(math.log10(x))


def math_ceil(x):
    import math
    return math.ceil(x)


def build_svg(meas, cand_idx, title):
    groups = meas["groups"]
    n = len(groups)
    base_bpw = meas.get("base", {}).get("bpw")
    alts = meas.get("alts", [])
    alt_bpw = alts[cand_idx]["bpw"] if cand_idx < len(alts) else None
    base_kld = meas.get("base_kld")

    # Pull the chosen candidate's dkld per group
    vals = []
    dbits = []
    for g in groups:
        cands = g["candidates"]
        c = cands[cand_idx] if cand_idx < len(cands) else cands[-1]
        vals.append(c["dkld"])
        dbits.append(c["dbits"])

    all_layers = []
    for g in groups:
        all_layers += g["layers"]
    gpl = len(".".join(lcp([l.split(".") for l in all_layers]))) if all_layers else 0
    if gpl and gpl < len(all_layers[0]):
        gpl += 1  # drop trailing dot

    # Geometry
    W, H = 1500, 640
    ml, mr, mt, mb = 80, 30, 88, 76
    pw, ph = W - ml - mr, H - mt - mb

    vmin = min(vals + [0.0])
    vmax = max(vals + [0.0])
    pad = (vmax - vmin) * 0.06 or 1.0
    ymin, ymax = vmin - pad, vmax + pad

    def yof(v):
        return mt + (ymax - v) / (ymax - ymin) * ph

    def xof(i):
        return ml + (i + 0.5) / n * pw

    y0 = yof(0.0)
    slot = pw / n
    bw = max(1.0, slot * 0.78)

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="DejaVu Sans, Segoe UI, sans-serif">'
    )
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{COL_BG}"/>')
    parts.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="{COL_PANEL}"/>')

    # Y gridlines + labels
    for t in nice_ticks(ymin, ymax):
        y = yof(t)
        if not (mt - 1 <= y <= mt + ph + 1):
            continue
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" '
                     f'stroke="{COL_GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" fill="{COL_MUTED}" '
                     f'font-size="12" text-anchor="end">{t:+.4f}</text>')

    # Zero line (emphasised)
    parts.append(f'<line x1="{ml}" y1="{y0:.1f}" x2="{ml + pw}" y2="{y0:.1f}" '
                 f'stroke="{COL_ZERO}" stroke-width="1.6"/>')

    # Bars
    for i, v in enumerate(vals):
        cx = xof(i)
        x = cx - bw / 2
        yv = yof(v)
        top = min(y0, yv)
        h = abs(yv - y0)
        col = COL_NEG if v < 0 else COL_POS
        label = short_name(groups[i]["layers"], gpl)
        tip = (f"group {groups[i].get('idx', i)}: {label}\n"
               f"dKL={v:+.6f}  dbits={dbits[i]:,}")
        parts.append(
            f'<rect x="{x:.2f}" y="{top:.2f}" width="{bw:.2f}" height="{max(h,0.5):.2f}" '
            f'fill="{col}"><title>{html.escape(tip)}</title></rect>'
        )

    # X ticks (sparse)
    step = max(1, n // 28)
    for i in range(0, n, step):
        x = xof(i)
        parts.append(f'<line x1="{x:.1f}" y1="{mt + ph}" x2="{x:.1f}" y2="{mt + ph + 5}" '
                     f'stroke="{COL_MUTED}" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{mt + ph + 20}" fill="{COL_MUTED}" '
                     f'font-size="11" text-anchor="middle">{groups[i].get("idx", i)}</text>')

    # Axis labels
    parts.append(f'<text x="{ml + pw / 2:.0f}" y="{H - 16}" fill="{COL_TEXT}" '
                 f'font-size="14" text-anchor="middle">tensor group index</text>')
    parts.append(f'<text x="20" y="{mt + ph / 2:.0f}" fill="{COL_TEXT}" font-size="14" '
                 f'text-anchor="middle" transform="rotate(-90 20 {mt + ph / 2:.0f})">'
                 f'&#916; KL-div (alt vs base)</text>')

    # Title + subtitle
    parts.append(f'<text x="{ml}" y="30" fill="{COL_TEXT}" font-size="19" '
                 f'font-weight="bold">{html.escape(title)}</text>')
    sub = []
    if base_bpw is not None and alt_bpw is not None:
        sub.append(f"base {base_bpw:.3f} bpw → alt {alt_bpw:.3f} bpw")
    if base_kld is not None:
        sub.append(f"base KL-div {base_kld:.5f}")
    sub.append(f"{n} groups")
    parts.append(f'<text x="{ml}" y="52" fill="{COL_MUTED}" font-size="13">'
                 f'{html.escape("   •   ".join(sub))}</text>')

    # Legend
    lx, ly = ml + pw - 360, 36
    parts.append(f'<rect x="{lx}" y="{ly}" width="14" height="14" fill="{COL_NEG}"/>')
    parts.append(f'<text x="{lx + 20}" y="{ly + 12}" fill="{COL_TEXT}" font-size="12">'
                 f'&#916;KL &lt; 0  (improves with more bits)</text>')
    parts.append(f'<rect x="{lx}" y="{ly + 20}" width="14" height="14" fill="{COL_POS}"/>')
    parts.append(f'<text x="{lx + 20}" y="{ly + 32}" fill="{COL_TEXT}" font-size="12">'
                 f'&#916;KL &gt; 0  (better at lower bpw)</text>')

    parts.append('</svg>')
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description = "Plot a measurement.json as a dark-mode SVG bar chart.")
    parser.add_argument("-m", "--measurement", type = str, required = True, help = "Input measurement.json")
    parser.add_argument("-o", "--out", type = str, default = None, help = "Output SVG (default: <input>.svg)")
    parser.add_argument("-c", "--cand", type = int, default = 0, help = "Candidate (alt) index to plot, default: 0")
    parser.add_argument("-t", "--title", type = str, default = None, help = "Chart title")
    args = parser.parse_args()

    with open(args.measurement, "r", encoding = "utf8") as f:
        meas = json.load(f)

    out = args.out or (os.path.splitext(args.measurement)[0] + ".svg")
    title = args.title or os.path.basename(args.measurement)
    svg = build_svg(meas, args.cand, title)
    with open(out, "w", encoding = "utf8") as f:
        f.write(svg)
    print(f" -- Wrote {out}")


if __name__ == "__main__":
    main()
