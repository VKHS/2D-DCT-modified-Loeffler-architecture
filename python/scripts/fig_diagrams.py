"""Block diagrams of the processor (architecture, engine, TSC, transpose, controller)."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

import figstyle as fs
from figstyle import INK, INK2, ROLE, SURF, arrow, block, bus_label, canvas, poly, reg_glyph, save


# ---------------------------------------------------------------------------
# pictogram of an 8x8 block: rows, spectra or columns highlighted
# ---------------------------------------------------------------------------
def block_icon(ax, x, y, size, mode, color, hl=None):
    c = size / 8
    for r in range(8):
        for k in range(8):
            if mode == "rows":
                on = (r % 2 == 0)
            elif mode == "cols":
                on = (k % 2 == 0)
            elif mode == "spec":
                on = (r + k) <= 5
            else:
                on = False
            fc = color if on else "#ffffff"
            ax.add_patch(Rectangle((x + k * c, y + (7 - r) * c), c, c, fc=fc, ec="#d0d0d0", lw=0.4, zorder=3,
                                   alpha=0.85 if on else 1))
    ax.add_patch(Rectangle((x, y), size, size, fc="none", ec=INK2, lw=0.8, zorder=4))


def architecture_overview(out: Path):
    fig, ax = canvas(17, 7.6, (0, 170), (0, 76))

    ax.text(2, 73, "8×8 2-D DCT processor based on the modified Loeffler architecture",
            fontsize=15, fontweight="bold", color=INK, va="center")
    ax.text(2, 69.3, "row–column decomposition  ·  one 8-pixel row in and one 8-coefficient column out per clock  ·  "
                     "one block every 8 clocks", fontsize=10.5, color=INK2, va="center")

    y0, h = 34, 20
    boxes = [
        (10, 15, "Level shift", ["(p − 128) << RF", "MSB inversion", "RF = 2 guard bits"], "io"),
        (33, 20, "Row engine", ["modified Loeffler", "8-point DCT", "11 mult · 29 add", "7-stage pipeline"], "engine"),
        (61, 21, "TSC unit", ["threshold set and", "comparison of rows", "dual-reference SAD", "adaptive threshold"], "tsc"),
        (90, 19, "Transpose", ["in-place 8×8 buffer", "64 × 13 bit", "orientation flips", "every block"], "mem"),
        (117, 20, "Column engine", ["modified Loeffler", "8-point DCT", "11 mult · 29 add", "7-stage pipeline"], "engine"),
        (145, 17, "Normalise", ["(Z + 16) >>> 5", "triangular zone", "u + v ≤ Z", "(stage 2)"], "out"),
    ]
    for x, w, t, lines, role in boxes:
        block(ax, x, y0, w, h, t, lines, role=role, fs=9.5, title_fs=11.5)

    yc = y0 + h / 2
    arrow(ax, (1.5, yc), (10, yc), lw=2.0, color=INK)
    ax.text(1.5, yc + 3.4, "pixels", fontsize=9.5, color=INK, ha="left")
    bus_label(ax, 5.5, yc, "8 × 8 b")
    links = [(25, 33, "8 × 10 b"), (53, 61, "8 × 13 b"), (82, 90, "8 × 13 b"), (109, 117, "8 × 13 b"),
             (137, 145, "8 × 16 b")]
    for xa, xb, lab in links:
        arrow(ax, (xa, yc), (xb, yc), lw=2.0, color=INK)
        bus_label(ax, (xa + xb) / 2, yc, lab)
    arrow(ax, (162, yc), (169.5, yc), lw=2.0, color=INK)
    bus_label(ax, 165.5, yc, "8 × 12 b")
    ax.text(169.5, yc - 3.4, "F[v][u]", fontsize=9.5, color=INK, ha="right")

    lat = [(33, 53, "P1–P7: 7 clk"), (61, 82, "stages A, B: 2 clk"), (90, 109, "write / read"),
           (117, 137, "P1–P7: 7 clk"), (145, 162, "1 clk")]
    for xa, xb, lab in lat:
        ax.plot([xa, xa, xb, xb], [57.2, 58.2, 58.2, 57.2], color=fs.AXIS, lw=1.0)
        ax.text((xa + xb) / 2, 59.4, lab, ha="center", fontsize=9, color=INK2)

    block_icon(ax, 2.5, 22, 6, "rows", fs.S[0])
    ax.text(5.5, 19.6, "rows in", ha="center", fontsize=8.5, color=INK2)
    block_icon(ax, 161.5, 22, 6, "cols", fs.S[5])
    ax.text(164.5, 19.6, "columns out", ha="center", fontsize=8.5, color=INK2)

    cx0, cy0, cw, ch = 46, 4, 76, 13
    block(ax, cx0, cy0, cw, ch, "Controller",
          ["row tags 0..7  ·  replacement mask  ·  read FSM: IDLE → READ (column 0..7)",
           "a block that completes at column 7 restarts at once: 1 block / 8 clk"],
          role="ctrl", fs=9.2, title_fs=11)
    ctl = ROLE["ctrl"][1]
    poly(ax, [(52, cy0 + ch), (52, 25), (43, 25), (43, y0)], color=ctl, lw=1.2)
    ax.text(44, 26.4, "row index tag", fontsize=8.5, color=ctl)
    poly(ax, [(64, y0), (64, cy0 + ch)], color=ctl, lw=1.2)
    ax.text(65.2, 21.0, "replaced, approx", fontsize=8.5, color=ctl)
    poly(ax, [(99.5, cy0 + ch), (99.5, y0)], color=ctl, lw=1.2)
    ax.text(100.7, 21.0, "rd_col, orientation", fontsize=8.5, color=ctl)
    poly(ax, [(116, cy0 + ch), (116, 25), (127, 25), (127, y0)], color=ctl, lw=1.2)
    ax.text(128.2, 26.4, "valid, column, flag", fontsize=8.5, color=ctl)
    poly(ax, [(cx0 + cw, 10.5), (153.5, 10.5), (153.5, y0)], color=ctl, lw=1.2)
    ax.text(154.7, 21.0, "zone mask", fontsize=8.5, color=ctl)

    tc = ROLE["tsc"][1]
    ax.text(65.2, 31.2, "stage 1: SAD(row r, row r−1) < TINT[m] → replace row r", fontsize=8.6, color=tc)
    ax.text(65.2, 28.6, "stage 2: rows 0..3 similar → column-wise approximation", fontsize=8.6, color=tc)

    ax.add_patch(FancyBboxPatch((2, 2.5), 40, 14.5, boxstyle="round,pad=0,rounding_size=1.0", fc=fs.PLANE,
                                ec=fs.AXIS, lw=0.8))
    txt = ["throughput  1 block / 8 clk", "latency     18 clk last row → column 0",
           "arithmetic  2 × (11 mult + 29 add)", "memory      64 words × 13 bit"]
    for i, t in enumerate(txt):
        ax.text(3.6, 14.2 - i * 3.3, t, fontsize=9.0, color=INK, family="DejaVu Sans Mono", va="center")
    save(fig, out)


# ---------------------------------------------------------------------------
# Loeffler signal-flow graph with pipeline cut-sets
# ---------------------------------------------------------------------------
LANE_Y = [80 - 8.4 * k for k in range(8)]
OUT_NAMES = ["Y0", "Y4", "Y2", "Y6", "Y7", "Y3", "Y5", "Y1"]


def _wire(ax, x0, x1, k, color=INK, z=2):
    ax.plot([x0, x1], [LANE_Y[k], LANE_Y[k]], color=color, lw=1.3, zorder=z)


def _butterfly(ax, x0, x1, i, j, neg=None, color=INK):
    """Sum and difference of lanes i, j; '-1' marks the lane whose own input is subtracted."""
    yi, yj = LANE_Y[i], LANE_Y[j]
    for (xa, ya, xb, yb) in [(x0, yi, x1, yi), (x0, yj, x1, yj), (x0, yi, x1, yj), (x0, yj, x1, yi)]:
        ax.plot([xa, xb], [ya, yb], color=color, lw=1.3, zorder=2)
    yn = LANE_Y[j if neg is None else neg]
    ax.text(x1 - 2.2, yn + 0.9, "−1", fontsize=7.8, color="#c4531c", ha="center", va="bottom", zorder=5)
    for y in (yi, yj):
        ax.add_patch(Circle((x1, y), 0.5, fc=INK, ec=INK, zorder=4))


def _gate(ax, x, w, p, q, top_label, bottom_label, role="engine"):
    """Plane rotation on lanes p < q drawn as a two-terminal gate."""
    fill, edge = ROLE[role]
    yp, yq = LANE_Y[p], LANE_Y[q]
    ax.add_patch(Rectangle((x + w / 2 - 0.6, yq), 1.2, yp - yq, fc=fill, ec=edge, lw=1.0, zorder=1))
    for y, lab in ((yp, top_label), (yq, bottom_label)):
        ax.add_patch(FancyBboxPatch((x, y - 2.4), w, 4.8, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc=fill, ec=edge, lw=1.3, zorder=3))
        ax.text(x + w / 2, y, lab, ha="center", va="center", fontsize=8.8, color=INK, zorder=4)


def _scaler(ax, x, k, label="×√2"):
    ax.add_patch(Circle((x, LANE_Y[k]), 2.1, fc=ROLE["engine"][0], ec=ROLE["engine"][1], lw=1.3, zorder=3))
    ax.text(x, LANE_Y[k], label, ha="center", va="center", fontsize=8.4, color=INK, zorder=4)


def loeffler_sfg(out: Path):
    fig, ax = canvas(17, 12.4, (0, 170), (-42, 96))
    ax.text(2, 94, "Modified Loeffler 8-point DCT engine", fontsize=15, fontweight="bold", color=INK, va="center")
    ax.text(2, 90.4, "Y = √8 · C8 · x   ·   11 multiplications, 29 additions   ·   every rotation in 3-multiplier "
                     "form, products kept at full precision and rounded once", fontsize=10.5, color=INK2, va="center")

    stages = [(11, 33, "stage 1"), (33, 80, "stage 2"), (80, 118, "stage 3"), (118, 152, "stage 4")]
    for i, (a, b, lab) in enumerate(stages):
        ax.add_patch(Rectangle((a, 18.5), b - a, 66, fc="#f5f8fc" if i % 2 == 0 else SURF, ec="none", zorder=0))
        ax.text((a + b) / 2, 85.2, lab, ha="center", fontsize=10, color=INK2, fontweight="bold")

    for k in range(8):
        ax.text(8.5, LANE_Y[k], f"x{k}", ha="right", va="center", fontsize=10.5, color=INK)
    # ---- stage 1
    for k in range(8):
        _wire(ax, 9.5, 14, k)
    for i in range(4):
        _butterfly(ax, 14, 30, i, 7 - i)
    for k in range(8):
        _wire(ax, 30, 37, k)
    # ---- stage 2
    _butterfly(ax, 37, 53, 0, 3)
    _butterfly(ax, 37, 53, 1, 2)
    for k in range(4):
        _wire(ax, 53, 84, k)
    for k in range(4, 8):
        _wire(ax, 37, 80, k)
    _gate(ax, 40, 14, 4, 7, "rotate 3π/16", "(c3, c5)")
    _gate(ax, 60, 14, 5, 6, "rotate π/16", "(c1, c7)")
    for k in range(4, 8):
        _wire(ax, 80, 84, k)
    # ---- stage 3
    _butterfly(ax, 84, 100, 0, 1)
    fill, edge = ROLE["engine"]
    ax.add_patch(FancyBboxPatch((88, LANE_Y[3] - 2.6), 20, LANE_Y[2] - LANE_Y[3] + 5.2,
                                boxstyle="round,pad=0,rounding_size=0.8", fc=fill, ec=edge, lw=1.3, zorder=3))
    ax.text(98, (LANE_Y[2] + LANE_Y[3]) / 2 + 1.4, "√2 · rotate 6π/16", ha="center", va="center", fontsize=9,
            fontweight="bold", color=INK, zorder=4)
    ax.text(98, (LANE_Y[2] + LANE_Y[3]) / 2 - 1.8, "(√2·c6, √2·c2)", ha="center", va="center", fontsize=8.2,
            color=INK2, zorder=4)
    _wire(ax, 84, 88, 2)
    _wire(ax, 84, 88, 3)
    _butterfly(ax, 84, 100, 4, 6, neg=6)
    _butterfly(ax, 84, 100, 5, 7, neg=5)
    for k in (0, 1):
        _wire(ax, 100, 152, k)
    for k in (2, 3):
        _wire(ax, 108, 152, k)
    for k in range(4, 8):
        _wire(ax, 100, 124, k)
    # ---- stage 4
    _butterfly(ax, 124, 140, 4, 7, neg=4)
    for k in (5, 6):
        _wire(ax, 124, 140, k)
    for k in range(4, 8):
        _wire(ax, 140, 152, k)
    _scaler(ax, 146, 5)
    _scaler(ax, 146, 6)
    for k in range(8):
        ax.text(153.5, LANE_Y[k], OUT_NAMES[k], ha="left", va="center", fontsize=10.5, color=INK, fontweight="bold")

    # node names after each stage
    for x, names in [(33, "a0 a1 a2 a3 a4 a5 a6 a7"), (80, "b0 b1 b2 b3 b4 b5 b6 b7"), (118, "· · · · d4 d5 d6 d7")]:
        for k, n in enumerate(names.split()):
            if n != "·":
                ax.text(x, LANE_Y[k] + 1.3, n, fontsize=7.6, color=INK2, ha="center", zorder=5)

    for (a, b, lab) in [(0, 3, "even part\n4-point DCT"), (4, 7, "odd part")]:
        ya, yb = LANE_Y[a] + 2, LANE_Y[b] - 2
        ax.plot([4.8, 3.8, 3.8, 4.8], [ya, ya, yb, yb], color=fs.AXIS, lw=1.0)
        ax.text(3.0, (ya + yb) / 2, lab, ha="right", va="center", fontsize=8.5, color=INK2, rotation=90)
    for (a, b, lab) in [(0, 3, "ready after stage 3"), (4, 7, "ready after stage 4")]:
        ya, yb = LANE_Y[a] + 2, LANE_Y[b] - 2
        ax.plot([159.5, 160.5, 160.5, 159.5], [ya, ya, yb, yb], color=fs.AXIS, lw=1.0)
        ax.text(162, (ya + yb) / 2, lab, ha="left", va="center", fontsize=8.5, color=INK2, rotation=90)

    # ---- pipeline ruler
    ax.text(2, 13.6, "pipeline registers (latency 7 clocks, one vector per clock)", fontsize=10, fontweight="bold",
            color=INK)
    steps = [("P1", "stage-1 butterflies"), ("P2", "even butterflies,\nrotation pre-adds"),
             ("P3", "6 odd products,\nY0 / Y4, pre-add"), ("P4", "odd post-add + round,\n3 even products"),
             ("P5", "odd butterflies,\nY2 / Y6  (even done)"), ("P6", "Y1 / Y7,\n2 √2 products"),
             ("P7", "Y3 / Y5 round,\noutput  (odd done)")]
    w = 23.0
    for i, (tag, txt) in enumerate(steps):
        x = 2 + i * (w + 0.85)
        ax.add_patch(FancyBboxPatch((x, 1.2), w, 9.6, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc=ROLE["engine"][0], ec=ROLE["engine"][1], lw=1.0))
        reg_glyph(ax, x + 2.2, 6.0, color=ROLE["engine"][1])
        ax.text(x + 4.0, 8.4, tag, fontsize=9.6, fontweight="bold", color=ROLE["engine"][1], va="center")
        ax.text(x + 4.0, 4.6, txt, fontsize=8.0, color=INK, va="center", linespacing=1.15)

    # ---- inset: rotator
    ox, oy = 2, -41
    ax.add_patch(FancyBboxPatch((ox, oy), 96, 37, boxstyle="round,pad=0,rounding_size=1.2", fc=fs.PLANE,
                                ec=fs.AXIS, lw=0.8, zorder=0))
    ax.text(ox + 2, oy + 34, "3-multiplier rotation with fused rounding", fontsize=10, fontweight="bold", color=INK)
    ax.text(ox + 2, oy + 30.8, "maps onto a DSP slice: pre-adder → multiplier (M register) → post-adder (P register)",
            fontsize=8.8, color=INK2)
    yp, yq = oy + 22, oy + 8

    def op(x, y, t, role="engine", r=2.3):
        f, e = ROLE[role]
        ax.add_patch(Circle((x, y), r, fc=f, ec=e, lw=1.2, zorder=3))
        ax.text(x, y, t, ha="center", va="center", fontsize=9, color=INK, zorder=4)

    ax.text(ox + 4, yp, "xp", fontsize=10, color=INK, va="center")
    ax.text(ox + 4, yq, "xq", fontsize=10, color=INK, va="center")
    op(ox + 18, oy + 15, "+")
    op(ox + 33, oy + 15, "×K")
    op(ox + 33, yp + 4.5, "×B")
    op(ox + 33, yq - 3.5, "×A")
    op(ox + 55, yp, "−")
    op(ox + 55, yq, "+")
    for (x, y) in [(ox + 68, yp), (ox + 68, yq)]:
        ax.add_patch(FancyBboxPatch((x - 4.5, y - 2.2), 9, 4.4, boxstyle="round,pad=0,rounding_size=0.6",
                                    fc=ROLE["out"][0], ec=ROLE["out"][1], lw=1.1, zorder=3))
        ax.text(x, y, "+2^12 >>> 13", ha="center", va="center", fontsize=7.2, color=INK, zorder=4)
    poly(ax, [(ox + 7, yp), (ox + 12, yp), (ox + 12, oy + 16.4), (ox + 15.7, oy + 16.4)], lw=1.1)
    poly(ax, [(ox + 7, yq), (ox + 12, yq), (ox + 12, oy + 13.6), (ox + 15.7, oy + 13.6)], lw=1.1)
    arrow(ax, (ox + 20.3, oy + 15), (ox + 30.7, oy + 15), lw=1.1)
    poly(ax, [(ox + 12, yp), (ox + 12, yp + 4.5), (ox + 30.7, yp + 4.5)], lw=1.1)
    poly(ax, [(ox + 12, yq), (ox + 12, yq - 3.5), (ox + 30.7, yq - 3.5)], lw=1.1)
    poly(ax, [(ox + 35.3, oy + 15), (ox + 44, oy + 15), (ox + 44, yp), (ox + 52.7, yp)], lw=1.1)
    poly(ax, [(ox + 44, oy + 15), (ox + 44, yq), (ox + 52.7, yq)], lw=1.1)
    poly(ax, [(ox + 35.3, yp + 4.5), (ox + 55, yp + 4.5), (ox + 55, yp + 2.3)], lw=1.1)
    poly(ax, [(ox + 35.3, yq - 3.5), (ox + 55, yq - 3.5), (ox + 55, yq - 2.3)], lw=1.1)
    for y in (yp, yq):
        arrow(ax, (ox + 57.3, y), (ox + 63.5, y), lw=1.1)
        arrow(ax, (ox + 72.5, y), (ox + 78, y), lw=1.1)
    ax.text(ox + 79, yp, "yq = z − B·xp", fontsize=9.5, color=INK, va="center")
    ax.text(ox + 79, yq, "yp = z + A·xq", fontsize=9.5, color=INK, va="center")
    ax.text(ox + 26, oy + 17.4, "z = K·(xp + xq)", fontsize=8.5, color=INK2, ha="center")

    # ---- inset: constants and operations
    tx, ty = 101, -41
    ax.add_patch(FancyBboxPatch((tx, ty), 67, 37, boxstyle="round,pad=0,rounding_size=1.2", fc=fs.PLANE,
                                ec=fs.AXIS, lw=0.8, zorder=0))
    ax.text(tx + 2, ty + 34, "Constants round(c · 2^13) and operations", fontsize=10, fontweight="bold", color=INK)
    rows = [("rotation", "K", "A", "B"),
            ("3π/16  (a4, a7)", "6811", "−2260", "11363"),
            ("π/16   (a5, a6)", "8035", "−6436", "9633"),
            ("√2·6π/16 (b2, b3)", "4433", "6270", "15137"),
            ("√2 scaling", "11585", "", "")]
    for i, r in enumerate(rows):
        y = ty + 30.4 - i * 3.6
        col = INK if i else INK2
        wt = "bold" if i == 0 else "normal"
        ax.text(tx + 3, y, r[0], fontsize=8.8, color=col, fontweight=wt, va="center")
        for j, v in enumerate(r[1:]):
            ax.text(tx + 39 + j * 9.5, y, v, fontsize=8.8, color=col, fontweight=wt, va="center", ha="right",
                    family="DejaVu Sans Mono" if i else "DejaVu Sans")
    ax.plot([tx + 2, tx + 65], [ty + 28.5, ty + 28.5], color=fs.AXIS, lw=0.7)
    ops = [("stage", "add", "mult"), ("1", "8", "0"), ("2", "10", "6"), ("3", "9", "3"), ("4", "2", "2"),
           ("total", "29", "11")]
    for i, r in enumerate(ops):
        x = tx + 3 + i * 10.5
        ax.text(x, ty + 9.0, r[0], fontsize=8.8, color=INK2 if i == 0 else INK, fontweight="bold", va="center")
        ax.text(x, ty + 5.2, r[1], fontsize=8.8, color=INK, va="center", fontweight="bold" if i == 5 else "normal")
        ax.text(x, ty + 1.8, r[2], fontsize=8.8, color=INK, va="center", fontweight="bold" if i == 5 else "normal")
    ax.text(tx + 3, ty + 12.2, "operations per stage", fontsize=8.6, color=INK2, va="center")
    save(fig, out)


# ---------------------------------------------------------------------------
# TSC unit micro-architecture
# ---------------------------------------------------------------------------
def tsc_unit(out: Path):
    fig, ax = canvas(17, 8.9, (0, 170), (-1.5, 90))
    ax.text(2, 87, "Threshold Set and Comparison (TSC) unit", fontsize=15, fontweight="bold", color=INK, va="center")
    ax.text(2, 83.3, "one row spectrum per clock  ·  a speculative dual-reference SAD reduces the decision loop to a "
                     "2:1 mux and one comparator", fontsize=10.5, color=INK2, va="center")
    tsc_e = ROLE["tsc"][1]
    hl = fs.S[1]

    ax.add_patch(Rectangle((14, 7.5), 62, 69.5, fc="#fbf6f3", ec="none", zorder=0))
    ax.text(45, 74.5, "stage A  (1 clk)", ha="center", fontsize=10, color=INK2, fontweight="bold")
    ax.text(122, 79.5, "stage B  (1 clk)", ha="center", fontsize=10, color=INK2, fontweight="bold")

    # input and stage-A registers
    ax.text(1.5, 52, "Y_r\n8 × 13 b", fontsize=9.5, color=INK, va="center")
    arrow(ax, (9, 52), (14, 52), lw=2.0, color=INK)
    block(ax, 17, 61, 16, 9, "prev_raw", ["Y(r−1)"], role="mem", fs=8.8, title_fs=9.8)
    block(ax, 17, 31, 16, 9, "ref", ["kept spectrum"], role="mem", fs=8.8, title_fs=9.8)
    block(ax, 40, 58, 27, 15, "SAD a", ["8 × sub, |·|, adder tree", "against row r−1", "valid if r−1 kept"],
          role="tsc", fs=8.6)
    block(ax, 40, 27, 27, 15, "SAD b", ["8 × sub, |·|, adder tree", "against kept spectrum",
                                         "valid if r−1 replaced"], role="tsc", fs=8.6)
    poly(ax, [(14, 52), (36, 52), (36, 67), (40, 67)], lw=1.4, color=INK)
    poly(ax, [(36, 52), (36, 36), (40, 36)], lw=1.4, color=INK)
    arrow(ax, (33, 65.5), (40, 65.5), lw=1.2)
    arrow(ax, (33, 35.5), (40, 35.5), lw=1.2)
    poly(ax, [(36, 52), (36, 48), (25, 48), (25, 61)], lw=1.1, color=INK2)
    ax.text(26.2, 49.4, "Y_r → prev_raw", fontsize=8, color=INK2)
    for y in (65.5, 34.5, 52):
        reg_glyph(ax, 72, y, color=tsc_e)
    arrow(ax, (67, 65.5), (71.5, 65.5), lw=1.1)
    arrow(ax, (67, 34.5), (71.5, 34.5), lw=1.1)
    poly(ax, [(36, 52), (71.5, 52)], lw=1.1, color=INK2, arrow_end=False)
    ax.text(48, 53.3, "Y_r, row index", fontsize=8.2, color=INK2)

    # mux
    mx = 82
    ax.add_patch(Polygon([(mx, 40), (mx, 62), (mx + 6, 58), (mx + 6, 44)], closed=True, fc=ROLE["tsc"][0],
                         ec=tsc_e, lw=1.3, zorder=3))
    ax.text(mx + 3, 51, "mux", rotation=90, ha="center", va="center", fontsize=8.5, color=INK)
    poly(ax, [(72.5, 65.5), (78, 65.5), (78, 57), (82, 57)], lw=1.2)
    poly(ax, [(72.5, 34.5), (78, 34.5), (78, 45), (82, 45)], lw=1.2)
    ax.text(79, 58.3, "a", fontsize=8.5, color=INK2)
    ax.text(79, 42.6, "b", fontsize=8.5, color=INK2)

    # comparator, ROM, threshold index
    block(ax, 94, 45, 17, 12, "SAD < TINT[m]", ["similar"], role="tsc", fs=8.6, title_fs=9.4)
    arrow(ax, (88, 51), (94, 51), lw=1.2)
    block(ax, 94, 63, 17, 10, "TINT ROM", ["16 × 10 b"], role="io", fs=8.6, title_fs=9.4)
    arrow(ax, (102.5, 63), (102.5, 57), lw=1.2)
    block(ax, 115, 63, 15, 10, "m", ["threshold index"], role="ctrl", fs=8.6, title_fs=9.6)
    arrow(ax, (115, 68), (111, 68), lw=1.2)
    block(ax, 134, 62, 34, 12, "threshold update", ["row 0: m ← t0",
                                                     "row kept: m ← min(m + step, tmax)"],
          role="ctrl", fs=8.4, title_fs=9.6)
    arrow(ax, (134, 68), (130, 68), lw=1.2)

    # similar distribution
    arrow(ax, (111, 51), (146, 51), lw=1.4, color=INK)
    poly(ax, [(120, 51), (120, 41)], lw=1.4, color=hl)
    poly(ax, [(140, 51), (140, 62)], lw=1.1, color=INK2)
    ax.text(112.5, 52.6, "similar", fontsize=8.4, color=INK)

    # last_rep loop
    block(ax, 113, 30, 15, 11, "last_rep", ["decision r−1"], role="warn", fs=8.4, title_fs=9.4)
    poly(ax, [(113, 35.5), (85, 35.5), (85, 41.8)], lw=1.6, color=hl)
    ax.text(86, 28.4, "decision loop: mux select → compare → register", fontsize=8.4, color=hl)

    # replace and outputs
    block(ax, 146, 42, 17, 17, "replace", ["similar ? ref : Y_r", "ref ← Y_r if kept"], role="tsc",
          fs=8.3, title_fs=9.6)
    poly(ax, [(72.5, 52), (75, 52), (75, 22), (152, 22), (152, 42)], lw=1.1, color=INK2)
    ax.text(110, 23.2, "Y_r (registered)", fontsize=8.2, color=INK2)
    poly(ax, [(25, 31), (25, 17), (154.5, 17), (154.5, 42)], lw=1.1, color=INK2)
    ax.text(110, 18.2, "ref", fontsize=8.2, color=INK2)
    poly(ax, [(160, 42), (160, 11), (29, 11), (29, 31)], lw=1.1, color=tsc_e)
    ax.text(110, 12.2, "ref update when the row is kept", fontsize=8.2, color=tsc_e)
    arrow(ax, (163, 50.5), (169.8, 50.5), lw=2.0, color=INK)
    ax.text(166.4, 52.3, "Y'_r", fontsize=9, color=INK, ha="center")

    block(ax, 134, 26, 9, 12, "stage 2", ["rows", "1..3"], role="out", fs=7.6, title_fs=8.4)
    poly(ax, [(138.5, 51), (138.5, 38)], lw=1.1, color=INK2)

    lines = ["TINT[m] = round(m · 0.001 · 255 · 8 · √8 · 2^RF)",
             "m: 4-bit index, 0.001 per step; t0 = 1, step = 1, tmax = 10"]
    for i, t in enumerate(lines):
        ax.text(2, 4.2 - i * 3.1, t, fontsize=8.8, color=INK, family="DejaVu Sans Mono", va="center")
    save(fig, out)


# ---------------------------------------------------------------------------
# in-place transpose buffer
# ---------------------------------------------------------------------------
def transpose_buffer(out: Path):
    fig, ax = canvas(17, 7.0, (0, 170), (0, 70))
    ax.text(2, 67.5, "In-place transpose buffer with alternating orientation", fontsize=15, fontweight="bold",
            color=INK, va="center")
    ax.text(2, 63.8, "block k+1 is written into exactly the cells that block k releases, so one 64-word register "
                     "file sustains one row in and one column out per clock", fontsize=10.5, color=INK2, va="center")
    cA, cB, cC = "#9dc8f7", "#f9b996", "#8fdcc8"
    cell = 4.0

    def grid(x0, y0, owner, read_col=None, read_row=None, title="", sub=""):
        for i in range(8):
            for j in range(8):
                o = owner[i][j]
                fc = {"A": cA, "B": cB, "C": cC, "": "#ffffff"}[o[0] if o else ""]
                ax.add_patch(Rectangle((x0 + j * cell, y0 + (7 - i) * cell), cell, cell, fc=fc, ec="#ffffff",
                                       lw=1.2, zorder=2))
                if o:
                    ax.text(x0 + j * cell + cell / 2, y0 + (7 - i) * cell + cell / 2, o, ha="center",
                            va="center", fontsize=5.9, color=INK, zorder=3)
        ax.add_patch(Rectangle((x0, y0), 8 * cell, 8 * cell, fc="none", ec=INK2, lw=0.8, zorder=3))
        if read_col is not None:
            ax.add_patch(Rectangle((x0 + read_col * cell, y0), cell, 8 * cell, fc="none", ec=INK, lw=2.0, zorder=4))
        if read_row is not None:
            ax.add_patch(Rectangle((x0, y0 + (7 - read_row) * cell), 8 * cell, cell, fc="none", ec=INK, lw=2.0,
                                   zorder=4))
        ax.text(x0, y0 + 8 * cell + 5.0, title, fontsize=10.5, fontweight="bold", color=INK)
        ax.text(x0, y0 + 8 * cell + 1.8, sub, fontsize=9, color=INK2)
        for j in range(8):
            ax.text(x0 + j * cell + cell / 2, y0 - 1.8, str(j), ha="center", fontsize=7.5, color=INK2)
        for i in range(8):
            ax.text(x0 - 1.4, y0 + (7 - i) * cell + cell / 2, str(i), ha="center", va="center", fontsize=7.5,
                    color=INK2)

    def own(fn):
        return [[fn(i, j) for j in range(8)] for i in range(8)]

    y0 = 22
    # (1) block A complete, written row-wise: mem[8r+u] = A(r,u)
    g1 = own(lambda i, j: f"A{i}{j}")
    grid(4, y0, g1, read_col=0, title="1  block A written row-wise", sub="clock t: read column 0 of A")
    # (2) three columns consumed and refilled by rows of B (B row r -> mem column r)
    g2 = own(lambda i, j: f"B{j}{i}" if j < 3 else f"A{i}{j}")
    grid(46, y0, g2, read_col=3, title="2  B fills the released columns", sub="clock t+3: read A col 3, write B row 3")
    # (3) block B complete, stored transposed; now read by mem rows while C is written row-wise
    g3 = own(lambda i, j: f"B{j}{i}")
    grid(88, y0, g3, read_row=0, title="3  block B stored transposed", sub="clock t+8: B column 0 = mem row 0")
    # (4) C written row-wise over B's released rows
    g4 = own(lambda i, j: f"C{i}{j}" if i < 3 else f"B{j}{i}")
    grid(130, y0, g4, read_row=3, title="4  C fills the released rows", sub="clock t+11: orientation flipped back")
    for x in (40, 82, 124):
        arrow(ax, (x, y0 + 16), (x + 4, y0 + 16), lw=1.6, color=INK2)

    # legend + rule
    for k, (c, t) in enumerate([(cA, "block A"), (cB, "block B"), (cC, "block C")]):
        ax.add_patch(Rectangle((4 + k * 22, 8.5), 3.2, 3.2, fc=c, ec="none"))
        ax.text(8.5 + k * 22, 10.1, t, fontsize=9, color=INK2, va="center")
    ax.add_patch(Rectangle((70, 8.5), 3.2, 3.2, fc="none", ec=INK, lw=2.0))
    ax.text(74.5, 10.1, "read this clock (combinational, read-before-write)", fontsize=9, color=INK2, va="center")
    ax.text(4, 3.5, "orientation 0: element (r, u) at mem[8r + u]     orientation 1: element (r, u) at mem[8u + r]"
                    "     cell label = block, row r, column u", fontsize=9, color=INK, family="DejaVu Sans Mono")
    save(fig, out)


# ---------------------------------------------------------------------------
# controller FSM (graphviz)
# ---------------------------------------------------------------------------
def controller_fsm(out: Path):
    e_ctrl = ROLE["ctrl"][1]
    f_ctrl = ROLE["ctrl"][0]
    dot = f"""
digraph fsm {{
  graph [rankdir=LR, bgcolor="white", pad="0.3", nodesep="0.6", ranksep="1.3", fontname="DejaVu Sans",
         label=<<b>Controller read FSM</b><br/><font point-size="11" color="#666666">row tags count 0..7 on every input row; the replacement mask and stage-2 flag of a block are latched when its row 7 enters the transpose buffer</font>>,
         labelloc="t", labeljust="l", fontsize="16"];
  node  [shape=circle, style="filled", fontname="DejaVu Sans", fontsize="12", fixedsize=true, width="1.45",
         color="{e_ctrl}", fillcolor="{f_ctrl}", penwidth="1.6", fontcolor="#191919"];
  edge  [fontname="DejaVu Sans", fontsize="10.5", color="#666666", fontcolor="#191919", penwidth="1.3",
         arrowsize="0.8"];
  start [shape=point, width="0.12", color="#191919"];
  IDLE  [label=<<b>S_IDLE</b><br/><font point-size="10">rd_valid = 0</font>>];
  READ  [label=<<b>S_READ</b><br/><font point-size="10">issue column<br/>rd_col</font>>];
  start -> IDLE [label=" rst"];
  IDLE -> READ  [label=<row 7 written<br/><font point-size="9.5" color="#666666">latch mask, approx; rd_col = 0</font>>];
  READ -> READ  [label=<rd_col &lt; 7<br/><font point-size="9.5" color="#666666">rd_col + 1</font>>];
  READ -> IDLE  [label=<rd_col = 7 and no new block>];
  READ -> READ  [label=<rd_col = 7 and row 7 written<br/><font point-size="9.5" color="#666666">back-to-back: rd_col = 0</font>>, color="{fs.S[1]}", penwidth="1.8"];
  IDLE -> IDLE  [label=<no block completes>];
}}"""
    dot_path = Path(out).with_suffix(".dot")
    dot_path.write_text(dot)
    subprocess.run(["dot", "-Tpng", "-Gdpi=170", str(dot_path), "-o", str(out)], check=True)
    subprocess.run(["dot", "-Tsvg", str(dot_path), "-o", str(Path(out).with_suffix(".svg"))], check=True)
    dot_path.unlink()


def make_all(outdir: Path):
    fs.setup()
    outdir.mkdir(parents=True, exist_ok=True)
    architecture_overview(outdir / "architecture_overview.png")
    loeffler_sfg(outdir / "loeffler_flowgraph.png")
    tsc_unit(outdir / "tsc_unit.png")
    transpose_buffer(outdir / "transpose_buffer.png")
    controller_fsm(outdir / "controller_fsm.png")
