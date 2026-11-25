"""Floating-point 8-point Loeffler DCT engine, stage by stage.

Data flow (Loeffler, Ligtenberg & Moschytz, ICASSP 1989), output scaled by sqrt(8):

    stage 1  a_i = x_i + x_{7-i},  a_{7-i} = x_i - x_{7-i}            8 add
    stage 2  even: b0=a0+a3, b3=a0-a3, b1=a1+a2, b2=a1-a2              4 add
             odd : (b4,b7) = rot_{3pi/16}(a4,a7)                       3 mul 3 add
                   (b5,b6) = rot_{pi/16}(a5,a6)                        3 mul 3 add
    stage 3  even: Y0=b0+b1, Y4=b0-b1                                  2 add
                   (Y2,Y6) = sqrt2*rot_{6pi/16}(b2,b3)                 3 mul 3 add
             odd : d4=b4+b6, d6=b4-b6, d7=b7+b5, d5=b7-b5              4 add
    stage 4  odd : Y1=d7+d4, Y7=d7-d4, Y3=sqrt2*d5, Y5=sqrt2*d6        2 mul 2 add

The even outputs (Y0, Y4, Y2, Y6) are complete after three stages and the
odd outputs (Y1, Y7, Y3, Y5) after four.
"""

from __future__ import annotations

import numpy as np

from .constants import LOEFFLER_CONSTANTS

_K = {name: value for name, (value, _, _) in LOEFFLER_CONSTANTS.items()}

OUTPUT_ORDER = ("Y0", "Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7")


def _rot(xp, xq, k, a, b):
    z = k * (xp + xq)
    return z + a * xq, z - b * xp


def loeffler_dct8(x, return_stages: bool = False):
    """Scaled DCT Y = sqrt(8) * C8 x of the last axis of ``x``."""
    x = np.asarray(x, dtype=float)
    x0, x1, x2, x3, x4, x5, x6, x7 = (x[..., i] for i in range(8))

    # stage 1: input butterflies
    a0, a7 = x0 + x7, x0 - x7
    a1, a6 = x1 + x6, x1 - x6
    a2, a5 = x2 + x5, x2 - x5
    a3, a4 = x3 + x4, x3 - x4

    # stage 2
    b0, b3 = a0 + a3, a0 - a3
    b1, b2 = a1 + a2, a1 - a2
    b4, b7 = _rot(a4, a7, _K["O3_K"], _K["O3_A"], _K["O3_B"])
    b5, b6 = _rot(a5, a6, _K["O1_K"], _K["O1_A"], _K["O1_B"])

    # stage 3
    y0, y4 = b0 + b1, b0 - b1
    y2, y6 = _rot(b2, b3, _K["E_K"], _K["E_A"], _K["E_B"])
    d4, d6 = b4 + b6, b4 - b6
    d7, d5 = b7 + b5, b7 - b5

    # stage 4
    y1, y7 = d7 + d4, d7 - d4
    y3, y5 = _K["R2"] * d5, _K["R2"] * d6

    y = np.stack([y0, y1, y2, y3, y4, y5, y6, y7], axis=-1)
    if not return_stages:
        return y
    stages = {
        "stage1": np.stack([a0, a1, a2, a3, a4, a5, a6, a7], axis=-1),
        "stage2": np.stack([b0, b1, b2, b3, b4, b5, b6, b7], axis=-1),
        "stage3": np.stack([y0, y4, y2, y6, d4, d5, d6, d7], axis=-1),
        "stage4": y,
    }
    return y, stages


def loeffler_dct8_orthonormal(x):
    """Orthonormal 8-point DCT computed with the Loeffler data flow."""
    return loeffler_dct8(x) / np.sqrt(8.0)


def loeffler_dct2d(block):
    """Row-column 2-D DCT; two sqrt(8) passes give an exact factor 8."""
    rows = loeffler_dct8(np.asarray(block, dtype=float))
    cols = loeffler_dct8(np.swapaxes(rows, -1, -2))
    return np.swapaxes(cols, -1, -2) / 8.0
