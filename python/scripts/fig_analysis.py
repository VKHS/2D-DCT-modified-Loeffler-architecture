"""Measured figures: energy compaction, word length, TSC behaviour, trade-offs, images."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, Rectangle

import figstyle as fs
from figstyle import BLUE, INK, INK2, S, save

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loeffler2d import EXACT, FixedPointConfig, ModifiedLoeffler2D, TSCConfig, dct_matrix, psnr  # noqa: E402
from loeffler2d.codec import codec_psnr, codec_roundtrip  # noqa: E402
from loeffler2d.fixed_point import dct2d_fixed_exact, engine_ranges  # noqa: E402
from loeffler2d.images import available_samples  # noqa: E402
from loeffler2d.opcount import image_counts  # noqa: E402
from loeffler2d.processor import float_dct_image  # noqa: E402
from loeffler2d.reference import blockify, dct2d  # noqa: E402
from loeffler2d.tsc import threshold_table, triangle_mask  # noqa: E402

plt = fs.plt
SEQ = LinearSegmentedColormap.from_list("seq_blue", [BLUE[k] for k in (100, 200, 300, 400, 500, 600, 700)])
FOCUS = ("camera", "moon", "coffee", "brick")


def _images():
    imgs = available_samples()
    return {k: imgs[k] for k in imgs}


def _triangle_outline(ax, zone, color=INK, lw=1.6, offset=-0.5):
    """Staircase outline of the cells u + v <= zone on an imshow grid."""
    pts = [(offset, offset), (zone + 1 + offset, offset)]
    for k in range(zone + 1):
        pts.append((zone + 1 - k + offset, k + 1 + offset))
        pts.append((zone - k + offset, k + 1 + offset))
    pts.append((offset, zone + 1 + offset))
    ax.add_patch(Polygon(pts, closed=True, fill=False, ec=color, lw=lw, zorder=5))


# ---------------------------------------------------------------------------
def fig_energy(out: Path, imgs):
    fig = plt.figure(figsize=(16.6, 5.4))
    ax_a, ax_b, cax_b, ax_c = fs.axes_row(fig, widths=[4.2, 4.2, 0.16, 5.0], gaps=[1.0, 0.14, 1.55],
                                          height=4.2, left=0.75, bottom=0.75)

    # (a) basis images
    ax = ax_a
    c = dct_matrix(8)
    canvas_img = np.ones((8 * 9 + 1, 8 * 9 + 1))
    for v in range(8):
        for u in range(8):
            b = np.outer(c[v], c[u])
            b = (b - b.min()) / (b.max() - b.min() + 1e-12)
            canvas_img[1 + v * 9:1 + v * 9 + 8, 1 + u * 9:1 + u * 9 + 8] = b
    ax.imshow(canvas_img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for v in range(8):
        for u in range(8):
            if u + v <= 5:
                ax.add_patch(Rectangle((u * 9 + 0.5, v * 9 + 0.5), 8, 8, fill=False, ec=S[1], lw=1.6))
    ax.set_xticks([4.5 + 9 * k for k in range(8)], [str(k) for k in range(8)])
    ax.set_yticks([4.5 + 9 * k for k in range(8)], [str(k) for k in range(8)])
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("horizontal frequency u")
    ax.set_ylabel("vertical frequency v")
    ax.set_title("a  8×8 DCT basis; outlined: zone u + v ≤ 5")

    # (b) mean energy map
    ax = ax_b
    acc = np.zeros((8, 8))
    for name in FOCUS:
        b = blockify(imgs[name]).astype(float) - 128.0
        f = dct2d(b)
        e = (f ** 2).reshape(-1, 8, 8).mean(0)
        acc += e / e.sum()
    acc /= len(FOCUS)
    im = ax.imshow(np.log10(acc), cmap=SEQ, interpolation="nearest")
    _triangle_outline(ax, 5, color=S[1])
    for (v, u) in [(0, 0), (0, 1), (1, 0), (0, 7), (7, 0), (7, 7)]:
        ax.text(u, v, f"{100 * acc[v, u]:.1f}" if acc[v, u] >= 0.001 else f"{100 * acc[v, u]:.3f}",
                ha="center", va="center", fontsize=7.8,
                color="#ffffff" if np.log10(acc[v, u]) > -1.5 else INK)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.grid(False)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_title("b  block energy share (%), log colour")
    cb = fig.colorbar(im, cax=cax_b)
    cb.set_label("log10(energy share)", color=INK2)
    cb.outline.set_visible(False)

    # (c) cumulative energy vs zone
    ax = ax_c
    zones = np.arange(0, 15)
    for i, name in enumerate(FOCUS):
        b = blockify(imgs[name]).astype(float) - 128.0
        f = dct2d(b).reshape(-1, 8, 8)
        ac = f.copy()
        ac[:, 0, 0] = 0
        tot = (ac ** 2).sum()
        frac = [((ac ** 2)[:, triangle_mask(z)].sum()) / tot for z in zones]
        ax.plot(zones, 100 * np.array(frac), "-o", color=S[i], label=name, ms=4.5,
                markeredgecolor="#ffffff", markeredgewidth=1.2)
    ax.axvline(5, color=fs.AXIS, lw=1.0)
    ax.text(5.2, 40, "default zone Z = 5\n(21 of 64 coefficients)", fontsize=8.8, color=INK2)
    ax.set_xlabel("zone Z (keep u + v ≤ Z)")
    ax.set_ylabel("AC energy kept (%)")
    ax.set_ylim(0, 102)
    ax.set_xticks(range(0, 15, 2))
    ax.legend(loc="lower right", title="image", title_fontsize=9)
    ax.set_title("c  AC energy inside the triangular zone")
    save(fig, out)


# ---------------------------------------------------------------------------
def fig_fixed_point(out: Path, imgs):
    rng = np.random.default_rng(5)
    blocks = np.concatenate([rng.integers(0, 256, (6000, 8, 8)),
                             blockify(imgs["camera"]).reshape(-1, 8, 8)[:4096]])
    ref = dct2d(blocks - 128.0)
    cfs = list(range(8, 16))
    rfs = list(range(0, 5))
    rms = np.zeros((len(rfs), len(cfs)))
    mx = np.zeros_like(rms)
    for i, rf in enumerate(rfs):
        for j, cf in enumerate(cfs):
            f = dct2d_fixed_exact(blocks, FixedPointConfig(coef_frac=cf, row_frac=rf))
            e = f - ref
            rms[i, j] = np.sqrt(np.mean(e ** 2))
            mx[i, j] = np.abs(e).max()

    fig = plt.figure(figsize=(16, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.9, 1.15], wspace=0.42)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(mx, cmap=SEQ, aspect="auto", origin="lower", vmin=0.5, vmax=1.8)
    for i in range(len(rfs)):
        for j in range(len(cfs)):
            ax.text(j, i, f"{mx[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if mx[i, j] > 1.15 else INK)
    j0, i0 = cfs.index(13), rfs.index(2)
    ax.add_patch(Rectangle((j0 - 0.5, i0 - 0.5), 1, 1, fill=False, ec=S[1], lw=2.4))
    ax.set_xticks(range(len(cfs)), cfs)
    ax.set_yticks(range(len(rfs)), rfs)
    ax.grid(False)
    ax.set_xlabel("constant fractional bits CF")
    ax.set_ylabel("guard bits RF (row → column)")
    ax.set_title("a  error vs word length (cell = max |error|)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("max |F(fixed) − F(float)|  (LSB)", color=INK2)
    cb.outline.set_visible(False)
    ax.text(j0 + 0.55, i0 + 0.55, "chosen", color=S[1], fontsize=8.5, fontweight="bold")

    ax = fig.add_subplot(gs[1])
    f = dct2d_fixed_exact(blocks, FixedPointConfig())
    e = (f - ref).ravel()
    ax.hist(e, bins=np.linspace(-1, 1, 41), color=S[0], edgecolor="#ffffff", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylim(1, 3e6)
    ax.set_xlabel("F(fixed) − F(float)")
    ax.set_ylabel("coefficients")
    ax.set_title("b  error distribution, CF = 13, RF = 2")
    ax.text(0.02, 0.97, f"max |e| = {np.abs(e).max():.3f}\nRMS = {np.sqrt(np.mean(e ** 2)):.3f}\n"
                        f"≠ round(float): {100 * np.mean(f != np.round(ref)):.1f} %",
            transform=ax.transAxes, va="top", fontsize=9, color=INK)

    ax = fig.add_subplot(gs[2])
    cfg = FixedPointConfig()
    lo, hi = -128 << cfg.row_frac, 127 << cfg.row_frac
    rr = engine_ranges(lo, hi)
    ylo = min(v[0] for k, v in rr.items() if k.startswith("Y"))
    yhi = max(v[1] for k, v in rr.items() if k.startswith("Y"))
    cr = engine_ranges(ylo, yhi)
    groups = [("stage 1", ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"]),
              ("stage 2", ["b0", "b1", "b2", "b3", "s47", "s56", "b4", "b5", "b6", "b7"]),
              ("stage 3", ["Y0", "Y4", "s23", "Y2", "Y6", "d4", "d5", "d6", "d7"]),
              ("stage 4", ["Y1", "Y7", "Y3", "Y5"])]
    names = [n for _, g in groups for n in g]
    x = np.arange(len(names))
    rb = [rr[n][2] for n in names]
    cbits = [cr[n][2] for n in names]
    ax.bar(x - 0.21, rb, width=0.4, color=S[0], label="row engine (input 10 b)")
    ax.bar(x + 0.21, cbits, width=0.4, color=S[1], label="column engine (input 13 b)")
    ax.axhline(13, color=S[0], lw=1.0)
    ax.axhline(16, color=S[1], lw=1.0)
    ax.set_xticks(x, names, rotation=90, fontsize=7.6)
    ax.set_ylim(8, 17.5)
    ax.set_ylabel("required two's-complement bits")
    pos = 0
    for lab, g in groups:
        ax.text(pos + len(g) / 2 - 0.5, 17.1, lab, ha="center", fontsize=8.5, color=INK2)
        pos += len(g)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8.5)
    ax.set_title("c  worst-case word length per node", pad=24)
    save(fig, out)


# ---------------------------------------------------------------------------
def fig_tsc(out: Path, imgs):
    cfg = FixedPointConfig()
    proc = ModifiedLoeffler2D(cfg, TSCConfig())
    fig = plt.figure(figsize=(16.4, 5.6))
    ax_a, ax_b, cax_b, ax_c = fs.axes_row(fig, widths=[4.2, 4.2, 0.16, 5.2], gaps=[0.45, 0.14, 1.5],
                                          height=4.2, left=0.3, bottom=0.75)

    for k, name in enumerate(("camera", "moon")):
        img = imgs[name]
        _, res = proc.forward(img)
        bh, bw = img.shape[0] // 8, img.shape[1] // 8
        nrep = res.replaced.sum(-1).reshape(bh, bw)
        apx = res.approx.reshape(bh, bw)
        ax = (ax_a, ax_b)[k]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ov = np.ma.masked_where(nrep == 0, nrep)
        im = ax.imshow(np.kron(ov, np.ones((8, 8))), cmap=SEQ, vmin=0, vmax=7, alpha=0.55,
                       interpolation="nearest")
        ys, xs = np.nonzero(apx)
        for y, x in zip(ys, xs):
            ax.add_patch(Rectangle((x * 8 - 0.5, y * 8 - 0.5), 8, 8, fill=False, ec=S[1], lw=1.0))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        rate = 100 * res.replaced[..., 1:].mean()
        ax.set_title(f"{'ab'[k]}  {name}\n{rate:.0f} % rows replaced · {100 * res.approx.mean():.1f} % blocks "
                     f"approximated", fontsize=9.8)
        if k == 1:
            cb = fig.colorbar(im, cax=cax_b)
            cb.set_label("rows replaced in the block", color=INK2)
            cb.outline.set_visible(False)

    ax = ax_c
    scale = threshold_table(cfg.row_frac)[1] / 0.001
    xs = np.logspace(-4, 0, 400)
    for i, name in enumerate(FOCUS):
        y = ModifiedLoeffler2D(cfg, EXACT).transform_blocks(blockify(imgs[name]).astype(np.int64)).row_coefs
        d = (np.abs(np.diff(y, axis=-2)).sum(-1) / scale).ravel()
        cdf = [100 * np.mean(d < x) for x in xs]
        ax.plot(xs, cdf, color=S[i], label=name)
        if np.mean(d == 0) > 0.05:
            ax.text(1.2e-4, 100 * np.mean(d == 0) - 1.8, f"{name}: {100 * np.mean(d == 0):.0f} %\nidentical (d = 0)",
                    fontsize=8.5, color=INK2, va="top", linespacing=1.25)
    for t, lab, yy in ((0.001, "t0 = 0.001", 92), (0.01, "tmax = 0.010", 8)):
        ax.axvline(t, color=fs.AXIS, lw=1.2)
        ax.text(t * 1.1, yy, lab, fontsize=8.8, color=INK2)
    ax.axvspan(0.001, 0.01, color=S[1], alpha=0.06, lw=0)
    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.set_xlabel("difference of consecutive row spectra  d = mean|ΔX| / 255")
    ax.set_ylabel("row pairs with difference < d  (%)")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.3), title="image", title_fontsize=9, fontsize=8.5)
    ax.set_title("c  share of row pairs below a threshold", fontsize=10.5)
    save(fig, out)


# ---------------------------------------------------------------------------
def _sweep(imgs):
    cfg = FixedPointConfig()
    t0s = list(range(1, 11))
    data = {}
    for name in FOCUS:
        img = imgs[name]
        ex = ModifiedLoeffler2D(cfg, EXACT)
        ce, _ = ex.forward(img)
        base = {q: codec_psnr(img, ce, q)[0] for q in (50, 75, 90)}
        rows = []
        for t0 in t0s:
            tsc = TSCConfig(t0=t0)
            cp, res = ModifiedLoeffler2D(cfg, tsc).forward(img)
            oc = image_counts(res.replaced, res.approx, tsc.zone)
            rows.append(dict(t0=t0, rep=100 * res.replaced[..., 1:].mean(), apx=100 * res.approx.mean(),
                             d50=codec_psnr(img, cp, 50)[0] - base[50], d75=codec_psnr(img, cp, 75)[0] - base[75],
                             d90=codec_psnr(img, cp, 90)[0] - base[90], cm=oc["col_mult"], ca=oc["col_add"]))
        data[name] = rows
    return t0s, data


def fig_tradeoff(out: Path, imgs, sweep=None):
    t0s, data = sweep or _sweep(imgs)
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.4))
    fig.subplots_adjust(wspace=0.32)
    specs = [("rep", "rows replaced (%)", "a  stage 1: row replacement"),
             ("apx", "blocks approximated (%)", "b  stage 2: column approximation"),
             ("d75", "ΔPSNR vs exact, JPEG q75 (dB)", "c  quality cost in a codec"),
             ("cm", "column-engine multiplications / block", "d  column-pass multiplications")]
    for ax, (key, ylab, title) in zip(axs, specs):
        for i, name in enumerate(FOCUS):
            ax.plot(t0s, [r[key] for r in data[name]], "-o", color=S[i], label=name, ms=4.5,
                    markeredgecolor="#ffffff", markeredgewidth=1.1)
        ax.set_xlabel("default threshold t0 (× 0.001)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10.5)
        ax.set_xticks([1, 2, 4, 6, 8, 10])
    axs[3].axhline(88, color=fs.AXIS, lw=1.0)
    axs[3].text(10, 87.0, "exact transform: 88", ha="right", va="top", fontsize=8.5, color=INK2)
    axs[1].legend(loc="upper left", title="image", title_fontsize=9)
    fig.text(0.125, -0.04, "step = 0.001, tmax = 0.010, stage 2 needs 3 similar pairs in rows 0..3, zone Z = 5; "
                           "row engine always runs (88 mult + 232 add per block)", fontsize=9, color=INK2)
    save(fig, out)
    return t0s, data


# ---------------------------------------------------------------------------
def fig_operations(out: Path, imgs):
    fig = plt.figure(figsize=(16, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.4], wspace=0.25)
    ax = fig.add_subplot(gs[0])
    algs = [("direct matrix", 64, 56), ("Chen", 16, 26), ("Lee", 12, 29), ("Loeffler", 11, 29)]
    y = np.arange(len(algs))[::-1]
    ax.barh(y + 0.2, [a[1] for a in algs], height=0.38, color=S[0], label="multiplications")
    ax.barh(y - 0.2, [a[2] for a in algs], height=0.38, color=S[1], label="additions")
    for yy, a in zip(y, algs):
        ax.text(a[1] + 1, yy + 0.2, str(a[1]), va="center", fontsize=9, color=INK)
        ax.text(a[2] + 1, yy - 0.2, str(a[2]), va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [a[0] for a in algs])
    ax.set_xlabel("operations per 8-point DCT")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("a  8-point DCT algorithms")

    ax = fig.add_subplot(gs[1])
    names = list(FOCUS)
    x = np.arange(len(names))
    cfg = FixedPointConfig()
    tsc = TSCConfig()
    cm, ca, ov = [], [], []
    for name in names:
        _, res = ModifiedLoeffler2D(cfg, tsc).forward(imgs[name])
        oc = image_counts(res.replaced, res.approx, tsc.zone)
        cm.append(oc["col_mult"])
        ca.append(oc["col_add"])
        ov.append(oc["tsc_add"] + oc["tsc_abs"] + oc["tsc_cmp"])
    w = 0.2
    ax.bar(x - 1.5 * w, [88] * len(names), width=w * 0.92, color=fs.NEUTRAL, edgecolor=fs.AXIS, lw=0.6,
           label="exact: column multiplications")
    ax.bar(x - 0.5 * w, cm, width=w * 0.92, color=S[0], label="TSC: column multiplications")
    ax.bar(x + 0.5 * w, [232] * len(names), width=w * 0.92, color="#f7f7f7", edgecolor=fs.AXIS, lw=0.6,
           hatch="////", label="exact: column additions")
    ax.bar(x + 1.5 * w, ca, width=w * 0.92, color=S[1], label="TSC: column additions")
    for xi, m_, a_ in zip(x, cm, ca):
        ax.text(xi - 0.5 * w, m_ + 4, f"{m_:.0f}", ha="center", fontsize=8.5, color=INK)
        ax.text(xi + 1.5 * w, a_ + 4, f"{a_:.0f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, names)
    ax.set_ylabel("operations per 8×8 block")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 260)
    ax.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=8.3, handlelength=1.4,
              columnspacing=1.0)
    ax.set_title("b  column-pass arithmetic, default TSC setting (t0 = 0.001)", pad=28)
    ax.text(0.0, -0.13, f"each block also needs 7 row comparisons ({ov[0]:.0f} add / abs / compare operations, no "
                        "multiplications); the row pass is unchanged (88 mult, 232 add)",
            transform=ax.transAxes, fontsize=8.6, color=INK2)
    save(fig, out)


# ---------------------------------------------------------------------------
def fig_images(out: Path, imgs):
    cfg = FixedPointConfig()
    q = 50
    fig, axs = plt.subplots(2, 5, figsize=(16, 7.2))
    fig.subplots_adjust(wspace=0.04, hspace=0.16)
    for r, name in enumerate(("camera", "moon")):
        img = imgs[name]
        crop = (slice(0, 256), slice(128, 384)) if name == "camera" else (slice(128, 384), slice(128, 384))
        ce, _ = ModifiedLoeffler2D(cfg, EXACT).forward(img)
        cp, res = ModifiedLoeffler2D(cfg, TSCConfig(t0=3)).forward(img)
        je, _ = codec_roundtrip(ce, q, img.shape)
        jp, _ = codec_roundtrip(cp, q, img.shape)
        diff = np.abs(je.astype(int) - jp.astype(int))
        panels = [(img, "input", "gray", 0, 255),
                  (je, f"exact, JPEG q{q}: {psnr(img, je):.2f} dB", "gray", 0, 255),
                  (jp, f"TSC t0 = 0.003, q{q}: {psnr(img, jp):.2f} dB", "gray", 0, 255),
                  (diff, f"|exact − TSC|, scale 0..8 (max {diff.max()})", SEQ, 0, 8)]
        for c, (im, title, cmap, lo, hi) in enumerate(panels):
            ax = axs[r, c]
            ax.imshow(im[crop], cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_title(title, fontsize=9.5, fontweight="normal", color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
        ax = axs[r, 4]
        bh, bw = img.shape[0] // 8, img.shape[1] // 8
        nrep = res.replaced.sum(-1).reshape(bh, bw)
        cm = np.kron(nrep, np.ones((8, 8)))[crop]
        ax.imshow(img[crop], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.imshow(np.ma.masked_where(cm == 0, cm), cmap=SEQ, vmin=0, vmax=7, alpha=0.75, interpolation="nearest")
        apx = np.kron(res.approx.reshape(bh, bw), np.ones((8, 8)))[crop]
        ax.contour(apx, levels=[0.5], colors=[S[1]], linewidths=1.0)
        ax.set_title("rows replaced (blue), approximated (orange)", fontsize=9.5, fontweight="normal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        axs[r, 0].set_ylabel(name, fontsize=11, color=INK)
    save(fig, out)


# ---------------------------------------------------------------------------
def fig_rate_distortion(out: Path, imgs):
    cfg = FixedPointConfig()
    qs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.3), sharey=False)
    fig.subplots_adjust(wspace=0.3)
    setups = [("float DCT", None), ("exact fixed point", EXACT), ("TSC t0 = 0.001", TSCConfig(t0=1)),
              ("TSC t0 = 0.005", TSCConfig(t0=5))]
    cols = [fs.AXIS, S[0], S[1], S[2]]
    for ax, name in zip(axs, FOCUS):
        img = imgs[name]
        for (lab, tsc), col in zip(setups, cols):
            if tsc is None:
                coef = float_dct_image(img)
            else:
                coef, _ = ModifiedLoeffler2D(cfg, tsc).forward(img)
            pts = [codec_psnr(img, coef, q) for q in qs]
            rate = [p[1] / img.size for p in pts]
            ax.plot(rate, [p[0] for p in pts], "-o" if tsc is not None else "-", color=col, label=lab,
                    ms=3.8, lw=1.8 if tsc is not None else 3.2, markeredgecolor="#ffffff", markeredgewidth=0.9,
                    zorder=3 if tsc is not None else 2)
        ax.set_title(name, fontsize=10.5)
        ax.set_xlabel("nonzero quantised coefficients / pixel")
        ax.set_ylabel("PSNR (dB)")
    axs[0].legend(loc="lower right", fontsize=8.5)
    fig.text(0.125, -0.04, "JPEG luminance table scaled for quality 10 ... 95; the float curve is the reference "
                           "floating-point orthonormal DCT", fontsize=9, color=INK2)
    save(fig, out)


def make_all(outdir: Path):
    fs.setup()
    outdir.mkdir(parents=True, exist_ok=True)
    imgs = _images()
    fig_energy(outdir / "energy_compaction.png", imgs)
    fig_fixed_point(outdir / "fixed_point_accuracy.png", imgs)
    fig_tsc(outdir / "tsc_behaviour.png", imgs)
    fig_tradeoff(outdir / "tsc_tradeoff.png", imgs)
    fig_operations(outdir / "operation_counts.png", imgs)
    fig_images(outdir / "image_results.png", imgs)
    fig_rate_distortion(outdir / "rate_distortion.png", imgs)
