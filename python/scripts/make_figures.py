#!/usr/bin/env python3
"""Regenerate every figure of docs/figures.

    python3 python/scripts/make_figures.py                 # diagrams + measured plots
    python3 python/scripts/make_figures.py --vcd build/tb_dct2d_top.vcd   # + waveform, schedule

The waveform and schedule figures need a VCD of rtl/tb/tb_dct2d_top.v run with
the tsc vectors and +vcd (see scripts/run_rtl_sim.sh).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fig_analysis  # noqa: E402
import fig_diagrams  # noqa: E402
import plot_waveform  # noqa: E402

ROOT = HERE.parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures"))
    ap.add_argument("--vcd", default=None)
    ap.add_argument("--only", choices=("diagrams", "analysis", "waveform"), default=None)
    a = ap.parse_args()
    out = Path(a.out)
    if a.only in (None, "diagrams"):
        fig_diagrams.make_all(out)
        print("diagrams  ->", out)
    if a.only in (None, "analysis"):
        fig_analysis.make_all(out)
        print("analysis  ->", out)
    if a.vcd:
        plot_waveform.main(a.vcd, str(out))
        print("waveform  ->", out)


if __name__ == "__main__":
    main()
