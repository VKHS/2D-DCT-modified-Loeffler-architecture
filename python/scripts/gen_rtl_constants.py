#!/usr/bin/env python3
"""Print the Verilog/HLS constants and threshold table for a given CF and RF.

    python3 python/scripts/gen_rtl_constants.py --cf 13 --rf 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loeffler2d.constants import LOEFFLER_CONSTANTS, constant_bits, quantized_constants  # noqa: E402
from loeffler2d.fixed_point import engine_ranges, required_engine_width  # noqa: E402
from loeffler2d.tsc import threshold_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf", type=int, default=13)
    ap.add_argument("--rf", type=int, default=2)
    ap.add_argument("--pixel-bits", type=int, default=8)
    a = ap.parse_args()

    q = quantized_constants(a.cf)
    wc = constant_bits(a.cf)
    lo, hi = -(1 << (a.pixel_bits - 1)) << a.rf, ((1 << (a.pixel_bits - 1)) - 1) << a.rf
    wr = required_engine_width(lo, hi, a.cf)
    yr = engine_ranges(lo, hi, a.cf)
    ylo = min(v[0] for k, v in yr.items() if k.startswith("Y"))
    yhi = max(v[1] for k, v in yr.items() if k.startswith("Y"))
    wk = required_engine_width(ylo, yhi, a.cf)

    print(f"// CF = {a.cf}, RF = {a.rf}: WC = {wc}, row engine WI = {a.pixel_bits + a.rf} WD = {wr}, column engine WD = {wk}")
    print("// Verilog parameters (rtl/loeffler_dct8.v)")
    for name, (value, expr, _) in LOEFFLER_CONSTANTS.items():
        print(f"    parameter signed [WC-1:0] {name:<4s} = {q[name]:6d},      // {expr} = {value:+.10f}")
    print("// HLS constants (hls/dct2d_loeffler.cpp)")
    for name, (_, expr, _) in LOEFFLER_CONSTANTS.items():
        print(f"static const int {name} = {q[name]};  // {expr}")
    t = threshold_table(a.rf)
    print("// TSC threshold table TINT[m], m = 0..15")
    print("parameter [159:0] TINT = {" + ", ".join(f"10'd{v}" for v in reversed(t)) + "};")
    print("static const uint16_t TINT[16] = {" + ", ".join(map(str, t)) + "};")


if __name__ == "__main__":
    main()
