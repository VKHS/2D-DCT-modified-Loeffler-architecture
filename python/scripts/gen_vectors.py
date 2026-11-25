#!/usr/bin/env python3
"""Generate the golden vectors in ../../vectors used by every test bench."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from loeffler2d.images import vector_blocks  # noqa: E402
from loeffler2d.vectors import MODES, generate_engine, generate_mode  # noqa: E402


def main() -> None:
    out = ROOT / "vectors"
    blocks = vector_blocks()
    for mode in MODES:
        info = generate_mode(out, blocks, mode)
        print(f"{mode:6s}: {info}")
    generate_engine(out)
    print(f"engine: 512 vectors -> {out / 'engine'}")


if __name__ == "__main__":
    main()
