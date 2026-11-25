"""Shared look of every figure: palette, ink, block-diagram primitives."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

# categorical slots, fixed order (validated: adjacent CVD dE >= 11.8 on white)
S = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
BLUE = {100: "#cce3fd", 150: "#b6d5f8", 200: "#9dc8f7", 250: "#84b9f3", 300: "#6aabf1", 350: "#519ced",
        400: "#338ced", 450: "#257de0", 500: "#2370cd", 550: "#1b64be", 600: "#1757ac", 650: "#0f4a96",
        700: "#0e3d7e"}
INK = "#191919"
INK2 = "#666666"
GRID = "#e7e7e7"
AXIS = "#c2c2c2"
SURF = "#ffffff"
PLANE = "#f9f9f9"
NEUTRAL = "#efefef"

# block-diagram roles: (fill, edge)
ROLE = {
    "engine": ("#e6f0fc", "#1b64be"),
    "tsc": ("#fde9df", "#c4531c"),
    "mem": ("#dff5ef", "#00866b"),
    "ctrl": ("#ece8fb", "#4f35b8"),
    "io": ("#f2f2f2", "#666666"),
    "out": ("#e2f1e4", "#00701a"),
    "warn": ("#fdf1d6", "#a87300"),
}


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 5,
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
        "savefig.facecolor": SURF,
        "savefig.dpi": 170,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def canvas(w, h, xlim, ylim):
    """Figure with one axis in data units and no decoration (for diagrams)."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def block(ax, x, y, w, h, title, lines=(), role="engine", fs=10, title_fs=11, r=1.2, align="center", lw=1.4):
    fill, edge = ROLE[role]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fill, ec=edge, lw=lw, zorder=2))
    cx = x + w / 2 if align == "center" else x + 1.0
    ha = "center" if align == "center" else "left"
    n = len(lines)
    top = y + h / 2 + (n * 0.5 * fs * 0.11) + 0.9
    ax.text(cx, top, title, ha=ha, va="center", fontsize=title_fs, fontweight="bold", color=INK, zorder=3)
    for i, line in enumerate(lines):
        ax.text(cx, top - 2.0 - i * fs * 0.205, line, ha=ha, va="center", fontsize=fs, color=INK2, zorder=3)


def arrow(ax, p0, p1, color=INK2, lw=1.5, style="-|>", ms=12, zorder=1, rad=0.0):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, lw=lw, color=color,
                        shrinkA=0, shrinkB=0, zorder=zorder,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)


def poly(ax, pts, color=INK2, lw=1.5, arrow_end=True, zorder=1, ms=12):
    """Orthogonal connector through a list of points."""
    for p0, p1 in zip(pts[:-2], pts[1:-1]):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, zorder=zorder, solid_capstyle="butt")
    if arrow_end:
        arrow(ax, pts[-2], pts[-1], color=color, lw=lw, zorder=zorder, ms=ms)
    else:
        ax.plot([pts[-2][0], pts[-1][0]], [pts[-2][1], pts[-1][1]], color=color, lw=lw, zorder=zorder)


def bus_label(ax, x, y, text, color=INK2, fs=8.5, slash=True, dy=1.1):
    if slash:
        ax.plot([x - 0.45, x + 0.45], [y - 0.7, y + 0.7], color=color, lw=1.1, zorder=3)
    ax.text(x, y + dy, text, ha="center", va="bottom", fontsize=fs, color=color, zorder=3)


def reg_glyph(ax, x, y, w=0.9, h=1.6, color="#1b64be"):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=SURF, ec=color, lw=1.0, zorder=4))
    ax.plot([x - w / 2, x, x + w / 2], [y - h / 2, y - h / 2 + 0.5, y - h / 2], color=color, lw=0.8, zorder=5)


def axes_row(fig, widths, gaps, height, left=0.6, bottom=0.7):
    """Axes placed left to right with sizes in inches, all of the same height.

    Panels with a fixed aspect (images, square maps) then share the top edge
    with the plots beside them, so the panel titles sit on one line.
    """
    fw, fh = fig.get_size_inches()
    axs, x = [], left
    for k, w in enumerate(widths):
        axs.append(fig.add_axes((x / fw, bottom / fh, w / fw, height / fh)))
        x += w + (gaps[k] if k < len(gaps) else 0.0)
    return axs


def save(fig, path, svg=False):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.12)
    if svg:
        fig.savefig(str(path).replace(".png", ".svg"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
