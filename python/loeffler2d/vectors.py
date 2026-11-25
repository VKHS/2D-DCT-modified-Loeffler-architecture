"""Golden test-vector files shared by the Verilog, HLS and MATLAB test benches.

Layout of ``vectors/<mode>/``
    config.txt           key = value parameters of the run
    input_rows.hex       one 8x8 block = 8 lines; a line is one row of 8 pixels,
                         16 hex digits, pixel 0 in bits [7:0]
    expected_cols.hex    one block = 8 lines; line u holds column u of the 2-D
                         DCT, coefficient F[v][u] in bits [12v+11:12v]
                         (12-bit two's complement), 24 hex digits
    expected_status.hex  one line per block: {approx, replaced[7:0]} (3 hex digits)
    expected_coefs.csv   readable copy: block, v, u, F
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fixed_point import FixedPointConfig, dct8_fixed
from .processor import ModifiedLoeffler2D
from .tsc import TSCConfig, threshold_table

MODES = {
    "exact": TSCConfig(enable=False, approx_mode="off"),
    "tsc": TSCConfig(enable=True, t0=1, step=1, tmax=10, min_similar=3, approx_mode="tsc", zone=5),
}


def pack(values, bits: int) -> int:
    word = 0
    for i, v in enumerate(values):
        word |= (int(v) & ((1 << bits) - 1)) << (bits * i)
    return word


def unpack(word: int, bits: int, n: int = 8, signed: bool = True) -> list[int]:
    out = []
    for i in range(n):
        v = (word >> (bits * i)) & ((1 << bits) - 1)
        if signed and v >> (bits - 1):
            v -= 1 << bits
        out.append(v)
    return out


def write_hex(path: Path, words, bits: int) -> None:
    digits = (bits + 3) // 4
    path.write_text("".join(f"{w:0{digits}x}\n" for w in words))


def read_hex(path: Path) -> list[int]:
    return [int(line, 16) for line in Path(path).read_text().split() if line.strip()]


def config_lines(fixed: FixedPointConfig, tsc: TSCConfig, n_blocks: int) -> str:
    approx_code = {"off": 0, "tsc": 1, "always": 2}[tsc.approx_mode]
    kv = {
        "n_blocks": n_blocks,
        "coef_frac": fixed.coef_frac,
        "row_frac": fixed.row_frac,
        "pixel_bits": fixed.pixel_bits,
        "out_bits": fixed.out_bits,
        "tsc_enable": int(tsc.enable),
        "t0": tsc.t0,
        "step": tsc.step,
        "tmax": tsc.tmax,
        "min_similar": tsc.min_similar,
        "approx_mode": approx_code,
        "zone": tsc.zone,
        "threshold_table": " ".join(map(str, threshold_table(fixed.row_frac, tsc.unit))),
    }
    return "".join(f"{k} = {v}\n" for k, v in kv.items())


def generate_mode(out_dir: Path, blocks: np.ndarray, mode: str,
                  fixed: FixedPointConfig = FixedPointConfig()) -> dict:
    tsc = MODES[mode]
    proc = ModifiedLoeffler2D(fixed, tsc)
    res = proc.transform_blocks(blocks.astype(np.int64))
    d = Path(out_dir) / mode
    d.mkdir(parents=True, exist_ok=True)
    rows = [pack(blocks[b, r], 8) for b in range(len(blocks)) for r in range(8)]
    cols = [pack(res.coefs[b, :, u], fixed.out_bits) for b in range(len(blocks)) for u in range(8)]
    status = [pack(res.replaced[b], 1) | (int(res.approx[b]) << 8) for b in range(len(blocks))]
    write_hex(d / "input_rows.hex", rows, 64)
    write_hex(d / "expected_cols.hex", cols, 8 * fixed.out_bits)
    write_hex(d / "expected_status.hex", status, 9)
    (d / "config.txt").write_text(config_lines(fixed, tsc, len(blocks)))
    lines = ["block,v,u,F"]
    for b in range(len(blocks)):
        for v in range(8):
            for u in range(8):
                lines.append(f"{b},{v},{u},{int(res.coefs[b, v, u])}")
    (d / "expected_coefs.csv").write_text("\n".join(lines) + "\n")
    return {"blocks": len(blocks), "replaced_rows": int(res.replaced.sum()), "approx_blocks": int(res.approx.sum())}


def generate_engine(out_dir: Path, n: int = 512, seed: int = 11, fixed: FixedPointConfig = FixedPointConfig()) -> None:
    """Vectors for the stand-alone 1-D engine (row-engine input range)."""
    rng = np.random.default_rng(seed)
    lo = -(1 << (fixed.pixel_bits - 1)) << fixed.row_frac
    hi = ((1 << (fixed.pixel_bits - 1)) - 1) << fixed.row_frac
    x = rng.integers(lo, hi + 1, (n, 8))
    x[:16] = np.where(rng.integers(0, 2, (16, 8)) > 0, hi, lo)
    y = dct8_fixed(x, fixed.coef_frac)
    w_in = fixed.row_in_bits
    w_out = w_in + 3
    d = Path(out_dir) / "engine"
    d.mkdir(parents=True, exist_ok=True)
    write_hex(d / "input.hex", [pack(r, w_in) for r in x], 8 * w_in)
    write_hex(d / "expected.hex", [pack(r, w_out) for r in y], 8 * w_out)
    (d / "config.txt").write_text(f"n_vectors = {n}\nw_in = {w_in}\nw_out = {w_out}\ncoef_frac = {fixed.coef_frac}\n")
