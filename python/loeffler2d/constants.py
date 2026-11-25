"""Rotation constants of the 8-point Loeffler DCT engine.

The Loeffler-Ligtenberg-Moschytz factorisation computes the scaled DCT
Y = sqrt(8) * C8 * x with 11 multiplications and 29 additions.  The
multiplications live in three plane rotations and two sqrt(2) scalings:

    odd part,  stage 2 : rotation by 3*pi/16 on (a4, a7)        3 mult / 3 add
    odd part,  stage 2 : rotation by   pi/16 on (a5, a6)        3 mult / 3 add
    even part, stage 3 : sqrt(2)-scaled rotation by 6*pi/16     3 mult / 3 add
    odd part,  stage 4 : two sqrt(2) scalings                    2 mult

Every rotation uses the 3-multiplier form

    z  = K * (xp + xq)
    yp = z + A * xq
    yq = z - B * xp

with (K, A, B) listed below.  The same names are used in the MATLAB,
Verilog and HLS sources.
"""

from __future__ import annotations

import math

C = [math.cos(k * math.pi / 16.0) for k in range(8)]
SQRT2 = math.sqrt(2.0)

#: name -> (value, symbolic expression, role)
LOEFFLER_CONSTANTS = {
    "E_K": (SQRT2 * C[6], "sqrt2*cos(6pi/16)", "even rotation, shared product"),
    "E_A": (SQRT2 * (C[2] - C[6]), "sqrt2*(cos(2pi/16)-cos(6pi/16))", "even rotation -> Y2"),
    "E_B": (SQRT2 * (C[2] + C[6]), "sqrt2*(cos(2pi/16)+cos(6pi/16))", "even rotation -> Y6"),
    "O3_K": (C[3], "cos(3pi/16)", "odd rotation 3pi/16, shared product"),
    "O3_A": (C[5] - C[3], "cos(5pi/16)-cos(3pi/16)", "odd rotation 3pi/16 -> b4"),
    "O3_B": (C[3] + C[5], "cos(3pi/16)+cos(5pi/16)", "odd rotation 3pi/16 -> b7"),
    "O1_K": (C[1], "cos(pi/16)", "odd rotation pi/16, shared product"),
    "O1_A": (C[7] - C[1], "cos(7pi/16)-cos(pi/16)", "odd rotation pi/16 -> b5"),
    "O1_B": (C[1] + C[7], "cos(pi/16)+cos(7pi/16)", "odd rotation pi/16 -> b6"),
    "R2": (SQRT2, "sqrt2", "stage-4 scaling -> Y3, Y5"),
}

CONSTANT_ORDER = tuple(LOEFFLER_CONSTANTS)

#: number of non-trivial operations of the 8-point engine
ENGINE_MULTIPLICATIONS = 11
ENGINE_ADDITIONS = 29


def quantized_constants(coef_frac: int) -> dict[str, int]:
    """Round every constant to the nearest integer multiple of 2**-coef_frac."""
    scale = 1 << coef_frac
    return {name: int(round(value * scale)) for name, (value, _, _) in LOEFFLER_CONSTANTS.items()}


def constant_bits(coef_frac: int) -> int:
    """Two's-complement width that holds every quantised constant."""
    q = quantized_constants(coef_frac)
    largest = max(max(v, -v - 1) for v in q.values())
    return largest.bit_length() + 1
