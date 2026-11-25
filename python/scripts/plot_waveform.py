#!/usr/bin/env python3
"""Waveform and space-time schedule of the 2-D DCT processor from a VCD dump.

    vvp build/tb_dct2d_top_tsc.vvp +vcd          # writes build/tb_dct2d_top.vcd
    python3 python/scripts/plot_waveform.py build/tb_dct2d_top.vcd docs/figures

The signal names follow rtl/tb/tb_dct2d_top.v (scope tb_dct2d_top, DUT 'dut').
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Polygon, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
from figstyle import INK, INK2, S, save  # noqa: E402

plt = fs.plt
PERIOD = 10.0
FIRST_EDGE = 5.0


def parse_vcd(path: Path, wanted: set[str]):
    """Return {hierarchical name: (width, [(time, value or None)])} for the wanted signals."""
    ids, out, scope = {}, {}, []
    t = 0
    in_defs = True
    with open(path) as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if in_defs:
                if tok[0] == "$scope":
                    scope.append(tok[2])
                elif tok[0] == "$upscope":
                    scope.pop()
                elif tok[0] == "$var":
                    width, code, name = int(tok[2]), tok[3], tok[4]
                    full = ".".join(scope + [name])
                    if full in wanted:
                        ids.setdefault(code, []).append(full)
                        out[full] = (width, [])
                elif tok[0] == "$enddefinitions":
                    in_defs = False
                continue
            c = tok[0][0]
            if c == "#":
                t = int(tok[0][1:])
            elif c in "01xzXZ" and len(tok) == 1:
                code = tok[0][1:]
                if code in ids:
                    v = int(c) if c in "01" else None
                    for n in ids[code]:
                        out[n][1].append((t, v))
            elif c in "bB":
                code = tok[1]
                if code in ids:
                    bits = tok[0][1:]
                    v = int(bits, 2) if set(bits) <= {"0", "1"} else None
                    for n in ids[code]:
                        out[n][1].append((t, v))
    return out


def sample(changes, times):
    """Value of a signal just before each time instant (what a rising edge samples)."""
    vals, j, cur = [], 0, None
    for tt in times:
        while j < len(changes) and changes[j][0] < tt:
            cur = changes[j][1]
            j += 1
        vals.append(cur)
    return vals


def signed(v, bits):
    if v is None:
        return None
    return v - (1 << bits) if v >> (bits - 1) else v


TB = "tb_dct2d_top"
DUT = TB + ".dut"
SIGS = {
    "clk": TB + ".clk", "in_valid": TB + ".in_valid", "in_tag": DUT + ".ctl_in_row",
    "row_v": DUT + ".row_v", "row_tag": DUT + ".row_tag", "tsc_v": DUT + ".tsc_v", "tsc_row": DUT + ".tsc_row",
    "tsc_rep": DUT + ".tsc_rep", "tsc_tidx": DUT + ".tsc_tidx", "wr_orient": DUT + ".tb_wr_orient",
    "rd_v": DUT + ".ctl_rd_valid", "rd_col": DUT + ".ctl_rd_col", "col_v": DUT + ".col_v",
    "out_valid": TB + ".out_valid", "out_col": TB + ".out_col", "out_last": TB + ".out_last",
    "out_coef": TB + ".out_coef", "out_replaced": TB + ".out_replaced", "out_approx": TB + ".out_approx",
}


def load(vcd: Path, n_cycles: int):
    raw = parse_vcd(vcd, set(SIGS.values()))
    edges = [FIRST_EDGE + PERIOD * k for k in range(n_cycles + 1)]
    data = {}
    for key, name in SIGS.items():
        if name not in raw:
            raise KeyError(f"signal {name} not found in {vcd}")
        data[key] = sample(raw[name][1], edges)
    return data


def block_ids(valid):
    """Block index of every valid cycle (8 valid cycles per block)."""
    ids, n = [], 0
    for v in valid:
        if v == 1:
            ids.append(n // 8)
            n += 1
        else:
            ids.append(None)
    return ids


def block_color(b):
    if b is None:
        return None
    return S[b] if b < 4 else "#d9d9d9"


def schedule(data, out: Path, n_cycles: int):
    streams = [("pixel rows in", "in_valid", "in_tag", "row"), ("row engine out", "row_v", "row_tag", "row"),
               ("TSC out = buffer write", "tsc_v", "tsc_row", "row"),
               ("buffer read = column issue", "rd_v", "rd_col", "col"),
               ("column engine out", "col_v", None, "col"), ("coefficient columns out", "out_valid", "out_col", "col")]
    fig, ax = plt.subplots(figsize=(16, 4.6))
    for i, (lab, vk, ik, kind) in enumerate(streams):
        y = len(streams) - 1 - i
        ids = block_ids(data[vk][:n_cycles])
        cnt = 0
        for c in range(n_cycles):
            if ids[c] is None:
                continue
            idx = data[ik][c] if ik else cnt % 8
            cnt += 1
            ax.add_patch(Rectangle((c + 0.06, y - 0.36), 0.88, 0.72, fc=block_color(ids[c]), ec="none"))
            txt = f"{'r' if kind == 'row' else 'c'}{idx}" if idx is not None else ""
            ax.text(c + 0.5, y, txt, ha="center", va="center", fontsize=6.8,
                    color="#ffffff" if ids[c] is not None and ids[c] in (0, 5, 6) else INK)
        ax.text(-0.6, y, lab, ha="right", va="center", fontsize=9.5, color=INK)
    ax.set_xlim(-0.2, n_cycles)
    ax.set_ylim(-0.7, len(streams) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("clock cycle")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    first_out = next(c for c in range(n_cycles) if data["out_valid"][c] == 1)
    last_in7 = [c for c in range(n_cycles) if data["in_valid"][c] == 1][7]
    yb = -0.05
    ax.plot([last_in7, last_in7, first_out, first_out], [0.35, yb, yb, 0.35], color=INK2, lw=1.0)
    ax.plot([last_in7, last_in7], [0.35, len(streams) - 1 - 0.38], color=INK2, lw=0.9, ls=(0, (2, 2)))
    ax.text((last_in7 + first_out) / 2, yb + 0.12, f"{first_out - last_in7} clk: row 7 of block 0 in → "
            "column 0 of block 0 out", ha="center", va="bottom", fontsize=8.8, color=INK2)
    for b in range(4):
        ax.add_patch(Rectangle((0.3 + b * 2.4, 1.55), 0.8, 0.5, fc=S[b], ec="none"))
        ax.text(1.2 + b * 2.4, 1.8, f"{b}", fontsize=8.5, color=INK2, va="center")
    ax.text(0.3, 2.4, "block", fontsize=8.5, color=INK2, va="center")
    ax.set_title("Space-time schedule from the RTL simulation: one row in and one column out per clock, block k+1 "
                 "written while block k is read", fontsize=10.5, pad=18)
    save(fig, out)


def waveform(data, out: Path, c0: int, c1: int):
    rows = [("clk", "clk", "bit", None), ("in_valid", "in_valid", "bit", None),
            ("row tag", "in_tag", "bus", "in_valid"), ("row engine valid", "row_v", "bit", None),
            ("TSC row", "tsc_row", "bus", "tsc_v"), ("TSC replaced", "tsc_rep", "bit", "tsc_v"),
            ("threshold index m", "tsc_tidx", "bus", "tsc_v"), ("write orientation", "wr_orient", "bit", None),
            ("column issue", "rd_col", "bus", "rd_v"), ("out_valid", "out_valid", "bit", None),
            ("out_col  u", "out_col", "bus", "out_valid"), ("DC  F[0][0]", "out_coef", "coef", "out_valid"),
            ("out_last", "out_last", "bit", None), ("{approx, replaced}", "out_replaced", "status", "out_last")]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(16, 0.42 * n + 1.2))
    h = 0.62
    for i, (lab, key, kind, qual) in enumerate(rows):
        y = n - 1 - i
        ax.text(c0 - 0.4, y + h / 2, lab, ha="right", va="center", fontsize=9, color=INK)
        if key == "clk":
            xs, ys = [], []
            for c in range(c0, c1):
                xs += [c, c, c + 0.5, c + 0.5]
                ys += [y, y + h, y + h, y]
            ax.plot(xs, ys, color=INK2, lw=0.9)
            continue
        vals = data[key]
        qv = data[qual] if qual else None
        if kind == "bit":
            xs, ys = [], []
            for c in range(c0, c1):
                v = vals[c] or 0
                if qv is not None and qv[c] != 1:
                    v = 0
                xs += [c, c + 1]
                ys += [y + h * v, y + h * v]
                if v:
                    ax.add_patch(Rectangle((c, y), 1, h, fc=S[0], alpha=0.12, ec="none"))
            ax.plot(xs, ys, color=S[0] if key != "tsc_rep" else S[1], lw=1.4, drawstyle="steps-post")
            continue
        c = c0
        while c < c1:
            v = vals[c]
            active = qv is None or qv[c] == 1
            if kind == "coef":
                active = active and data["out_col"][c] == 0
            e = c + 1
            while e < c1 and vals[e] == v and (qv is None or (qv[e] == 1) == active) and kind != "coef":
                e += 1
            if not active or v is None:
                ax.plot([c, e], [y + h / 2, y + h / 2], color=fs.AXIS, lw=1.0)
            else:
                s = 0.12
                ax.add_patch(Polygon([(c, y + h / 2), (c + s, y + h), (e - s, y + h), (e, y + h / 2),
                                      (e - s, y), (c + s, y)], closed=True, fc="#eef4fd", ec=S[0], lw=0.9))
                if kind == "coef":
                    txt = str(signed(v & 0xFFF, 12))
                elif kind == "status":
                    txt = f"{(data['out_approx'][c] or 0)}·{v:02x}"
                else:
                    txt = str(v)
                ax.text((c + e) / 2, y + h / 2, txt, ha="center", va="center",
                        fontsize=6.6 if kind in ("coef", "status") else 7.6, color=INK)
            c = e
    ax.set_xlim(c0 - 0.2, c1)
    ax.set_ylim(-0.3, n)
    ax.set_yticks([])
    ax.set_xticks(range(c0, c1 + 1, 2))
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("clock cycle (rising edges)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=fs.GRID, lw=0.5)
    ax.spines["left"].set_visible(False)
    ax.set_title("RTL waveform of dct2d_top, tsc vectors: blocks 0-5 have identical rows (rows 1-7 replaced, zone "
                 "applied); blocks 6-7 change once at row 4, so m steps from 1 to 2", fontsize=10.5)
    save(fig, out)


def main(vcd: str, outdir: str, n_cycles: int = 64):
    fs.setup()
    data = load(Path(vcd), max(n_cycles, 80) + 2)
    o = Path(outdir)
    o.mkdir(parents=True, exist_ok=True)
    schedule(data, o / "pipeline_schedule.png", n_cycles)
    waveform(data, o / "rtl_waveform.png", 0, 80)


if __name__ == "__main__":
    main(*sys.argv[1:3])
