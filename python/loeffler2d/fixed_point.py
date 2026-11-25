"""Bit-accurate fixed-point model of the pipelined Loeffler engine.

This is the golden model of the RTL (``rtl/loeffler_dct8.v``) and the HLS
code (``hls/dct2d_loeffler.cpp``); all three produce identical integers.

Arithmetic rules
----------------
* Data words are two's-complement integers; the 1-D engine keeps the LSB
  weight of its input (no internal truncation of additions).
* Constants are ``round(c * 2**CF)``.
* Each rotation is fused: the three products are formed at full precision and
  every rotation output is rounded once,
  ``yp = (K*(xp+xq) + A*xq + 2**(CF-1)) >>> CF``.
* ``Y3`` and ``Y5`` are rounded the same way after the sqrt(2) product.

2-D datapath (row-column, Loeffler scale sqrt(8) per pass)
---------------------------------------------------------
    x   = (pixel - 128) << RF                    level shift + RF guard bits
    Y   = engine(x)            per row           LSB = 2**-RF, scale sqrt(8)
    Z   = engine(Y^T)          per column        LSB = 2**-RF, scale 8
    F   = (Z + 2**(RF+2)) >>> (RF + 3)           orthonormal 2-D DCT, integer

The two sqrt(8) scale factors multiply to exactly 8, so the final
normalisation is a plain rounding shift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import quantized_constants


@dataclass(frozen=True)
class FixedPointConfig:
    """Word-length parameters shared by Python, MATLAB, RTL and HLS."""

    coef_frac: int = 13  # CF : fractional bits of the rotation constants
    row_frac: int = 2  # RF : guard bits carried from the row to the column pass
    pixel_bits: int = 8  # unsigned input pixels
    out_bits: int = 12  # signed 2-D DCT coefficients

    @property
    def final_shift(self) -> int:
        return self.row_frac + 3

    @property
    def row_in_bits(self) -> int:
        """Width of the level-shifted, pre-scaled row-engine input."""
        return self.pixel_bits + self.row_frac


def _rshift_round(v, s: int):
    return (v + (1 << (s - 1))) >> s


def dct8_fixed(x, coef_frac: int = 13, return_nodes: bool = False):
    """Integer 8-point Loeffler DCT along the last axis (scale sqrt(8))."""
    x = np.asarray(x, dtype=np.int64)
    k = quantized_constants(coef_frac)
    rnd = np.int64(1 << (coef_frac - 1))
    x0, x1, x2, x3, x4, x5, x6, x7 = (x[..., i] for i in range(8))

    # stage 1
    a0, a7 = x0 + x7, x0 - x7
    a1, a6 = x1 + x6, x1 - x6
    a2, a5 = x2 + x5, x2 - x5
    a3, a4 = x3 + x4, x3 - x4

    # stage 2
    b0, b3 = a0 + a3, a0 - a3
    b1, b2 = a1 + a2, a1 - a2
    s47 = a4 + a7
    s56 = a5 + a6
    m_o3k, m_o3a, m_o3b = k["O3_K"] * s47, k["O3_A"] * a7, k["O3_B"] * a4
    m_o1k, m_o1a, m_o1b = k["O1_K"] * s56, k["O1_A"] * a6, k["O1_B"] * a5
    b4 = (m_o3k + m_o3a + rnd) >> coef_frac
    b7 = (m_o3k - m_o3b + rnd) >> coef_frac
    b5 = (m_o1k + m_o1a + rnd) >> coef_frac
    b6 = (m_o1k - m_o1b + rnd) >> coef_frac

    # stage 3
    y0, y4 = b0 + b1, b0 - b1
    s23 = b2 + b3
    m_ek, m_ea, m_eb = k["E_K"] * s23, k["E_A"] * b3, k["E_B"] * b2
    y2 = (m_ek + m_ea + rnd) >> coef_frac
    y6 = (m_ek - m_eb + rnd) >> coef_frac
    d4, d6 = b4 + b6, b4 - b6
    d7, d5 = b7 + b5, b7 - b5

    # stage 4
    y1, y7 = d7 + d4, d7 - d4
    m_r3, m_r5 = k["R2"] * d5, k["R2"] * d6
    y3 = (m_r3 + rnd) >> coef_frac
    y5 = (m_r5 + rnd) >> coef_frac

    y = np.stack([y0, y1, y2, y3, y4, y5, y6, y7], axis=-1)
    if not return_nodes:
        return y
    nodes = dict(
        x0=x0, x1=x1, x2=x2, x3=x3, x4=x4, x5=x5, x6=x6, x7=x7,
        a0=a0, a1=a1, a2=a2, a3=a3, a4=a4, a5=a5, a6=a6, a7=a7,
        b0=b0, b1=b1, b2=b2, b3=b3, s47=s47, s56=s56,
        p_o3k=m_o3k, p_o3a=m_o3a, p_o3b=m_o3b, p_o1k=m_o1k, p_o1a=m_o1a, p_o1b=m_o1b,
        b4=b4, b5=b5, b6=b6, b7=b7, s23=s23,
        p_ek=m_ek, p_ea=m_ea, p_eb=m_eb,
        Y0=y0, Y4=y4, Y2=y2, Y6=y6, d4=d4, d5=d5, d6=d6, d7=d7,
        p_r3=m_r3, p_r5=m_r5, Y1=y1, Y7=y7, Y3=y3, Y5=y5,
    )
    return y, nodes


def level_shift_and_scale(pixels, cfg: FixedPointConfig):
    """(pixel - 2**(P-1)) << RF  — the row-engine input word."""
    p = np.asarray(pixels, dtype=np.int64)
    return (p - (1 << (cfg.pixel_bits - 1))) << cfg.row_frac


def row_pass(blocks, cfg: FixedPointConfig):
    """Row transform of (..., 8, 8) unsigned pixel blocks."""
    return dct8_fixed(level_shift_and_scale(blocks, cfg), cfg.coef_frac)


def column_pass(row_coefs, cfg: FixedPointConfig):
    """Column transform of row spectra (..., r, u) -> (..., v, u), Loeffler scale 8."""
    z = dct8_fixed(np.swapaxes(np.asarray(row_coefs, dtype=np.int64), -1, -2), cfg.coef_frac)
    return np.swapaxes(z, -1, -2)


def final_scale(z, cfg: FixedPointConfig):
    return _rshift_round(np.asarray(z, dtype=np.int64), cfg.final_shift)


def dct2d_fixed_exact(blocks, cfg: FixedPointConfig = FixedPointConfig()):
    """Fixed-point 2-D DCT without TSC replacement or column approximation."""
    return final_scale(column_pass(row_pass(blocks, cfg), cfg), cfg)


# --------------------------------------------------------------------------
# dynamic-range analysis
# --------------------------------------------------------------------------

def _node_functionals():
    """Linear functional (8 weights) of every node of the float data flow."""
    from .constants import LOEFFLER_CONSTANTS

    kf = {n: v for n, (v, _, _) in LOEFFLER_CONSTANTS.items()}
    eye = np.eye(8)
    x = [eye[i] for i in range(8)]
    f = {f"x{i}": x[i] for i in range(8)}
    f["a0"], f["a7"] = x[0] + x[7], x[0] - x[7]
    f["a1"], f["a6"] = x[1] + x[6], x[1] - x[6]
    f["a2"], f["a5"] = x[2] + x[5], x[2] - x[5]
    f["a3"], f["a4"] = x[3] + x[4], x[3] - x[4]
    f["b0"], f["b3"] = f["a0"] + f["a3"], f["a0"] - f["a3"]
    f["b1"], f["b2"] = f["a1"] + f["a2"], f["a1"] - f["a2"]
    f["s47"], f["s56"] = f["a4"] + f["a7"], f["a5"] + f["a6"]
    f["b4"] = kf["O3_K"] * f["s47"] + kf["O3_A"] * f["a7"]
    f["b7"] = kf["O3_K"] * f["s47"] - kf["O3_B"] * f["a4"]
    f["b5"] = kf["O1_K"] * f["s56"] + kf["O1_A"] * f["a6"]
    f["b6"] = kf["O1_K"] * f["s56"] - kf["O1_B"] * f["a5"]
    f["Y0"], f["Y4"] = f["b0"] + f["b1"], f["b0"] - f["b1"]
    f["s23"] = f["b2"] + f["b3"]
    f["Y2"] = kf["E_K"] * f["s23"] + kf["E_A"] * f["b3"]
    f["Y6"] = kf["E_K"] * f["s23"] - kf["E_B"] * f["b2"]
    f["d4"], f["d6"] = f["b4"] + f["b6"], f["b4"] - f["b6"]
    f["d7"], f["d5"] = f["b7"] + f["b5"], f["b7"] - f["b5"]
    f["Y1"], f["Y7"] = f["d7"] + f["d4"], f["d7"] - f["d4"]
    f["Y3"], f["Y5"] = kf["R2"] * f["d5"], kf["R2"] * f["d6"]
    return f


def signed_bits(lo: int, hi: int) -> int:
    """Smallest two's-complement width holding [lo, hi]."""
    b = 1
    while not (-(1 << (b - 1)) <= lo and hi <= (1 << (b - 1)) - 1):
        b += 1
    return b


def engine_ranges(lo: int, hi: int, coef_frac: int = 13, margin: int = 2):
    """Worst-case value range of every data node for inputs in [lo, hi].

    For each node the extreme input vector (lo/hi chosen by the sign of the
    node's linear weights) is pushed through the bit-accurate engine; a
    small margin covers the bounded rounding error of other input vectors.
    Product nodes are bounded analytically.
    """
    f = _node_functionals()
    exact = {f"x{i}" for i in range(8)} | {f"a{i}" for i in range(8)} | {
        "b0", "b1", "b2", "b3", "s47", "s56", "s23", "Y0", "Y4"}
    out = {}
    for name, w in f.items():
        xmax = np.where(w >= 0, hi, lo)
        xmin = np.where(w >= 0, lo, hi)
        _, nmax = dct8_fixed(xmax, coef_frac, return_nodes=True)
        _, nmin = dct8_fixed(xmin, coef_frac, return_nodes=True)
        m = 0 if name in exact else margin
        vmax = int(nmax[name]) + m
        vmin = int(nmin[name]) - m
        out[name] = (vmin, vmax, signed_bits(vmin, vmax))
    return out


def required_engine_width(lo: int, hi: int, coef_frac: int = 13) -> int:
    """Uniform data width of an engine whose inputs lie in [lo, hi]."""
    return max(b for (_, _, b) in engine_ranges(lo, hi, coef_frac).values())
