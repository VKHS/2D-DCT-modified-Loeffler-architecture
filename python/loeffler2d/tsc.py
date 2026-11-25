"""Threshold Set and Comparison (TSC) unit and column-wise approximation.

Operation reduction of the paper works in two stages per 8x8 block:

Stage 1 - row replacement
    After the row DCT of row r, its spectrum is compared with the spectrum
    kept for row r-1.  When the DCT-domain difference is below the current
    threshold the row is *replaced* by its predecessor.  Whenever two
    consecutive rows are not similar, the threshold is raised by one step
    (0.001) for the next comparison, up to a maximum of 0.01.

Stage 2 - column-wise approximation
    When the first four rows of the block show enough similarity, the energy
    of the block is concentrated in the low-frequency triangle of the
    spectrum.  The column pass then only produces the coefficients of the
    triangular zone u + v <= Z; the rest are forced to zero.

Difference measure
    d_r = mean_k |X_r[k] - X_ref[k]| / 255, with X the orthonormal row DCT of
    8-bit pixels (i.e. the mean absolute DCT difference on a [0, 1] intensity
    scale).  In hardware this is a sum of absolute differences of the
    Loeffler-scaled row words compared with an integer threshold table:

        SAD < TINT[m],   TINT[m] = round(m * 0.001 * 255 * 8 * sqrt(8) * 2**RF)

    with threshold index m = 1..10 for thresholds 0.001..0.010.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

APPROX_MODES = ("off", "tsc", "always")


@dataclass(frozen=True)
class TSCConfig:
    """Parameters of the TSC unit (threshold indices in units of ``unit``)."""

    enable: bool = True
    t0: int = 1  # default threshold index  (0.001)
    step: int = 1  # adaptive increment       (0.001)
    tmax: int = 10  # maximum threshold index  (0.010)
    min_similar: int = 3  # similar pairs among rows 0..3 that trigger stage 2
    approx_mode: str = "tsc"  # 'off' | 'tsc' | 'always'
    zone: int = 5  # keep coefficients with u + v <= zone
    unit: float = 0.001

    def __post_init__(self):
        if self.approx_mode not in APPROX_MODES:
            raise ValueError(f"approx_mode must be one of {APPROX_MODES}")
        if not (0 <= self.t0 <= 15 and 0 <= self.tmax <= 15 and 0 <= self.step <= 15):
            raise ValueError("threshold indices are 4-bit values")
        if not (0 <= self.min_similar <= 3):
            raise ValueError("min_similar must be in 0..3")
        if not (0 <= self.zone <= 14):
            raise ValueError("zone must be in 0..14")


EXACT = TSCConfig(enable=False, approx_mode="off")


def threshold_table(row_frac: int, unit: float = 0.001, size: int = 16) -> list[int]:
    """Integer SAD thresholds TINT[m] for the Loeffler-scaled row words."""
    scale = unit * 255.0 * 8.0 * np.sqrt(8.0) * (1 << row_frac)
    return [int(np.floor(m * scale + 0.5)) for m in range(size)]


def triangle_mask(zone: int) -> np.ndarray:
    """Boolean 8x8 mask [v, u] of the kept low-frequency triangle u + v <= zone."""
    v, u = np.mgrid[0:8, 0:8]
    return (u + v) <= zone


@dataclass
class TSCResult:
    rows: np.ndarray  # (..., 8, 8) row spectra after replacement
    replaced: np.ndarray  # (..., 8) bool, row r replaced by row r-1
    approx: np.ndarray  # (...,) bool, stage-2 approximation active
    sad: np.ndarray  # (..., 8) SAD of row r vs its reference (row 0: 0)
    tindex: np.ndarray  # (..., 8) threshold index used for row r
    extra: dict = field(default_factory=dict)


def apply_tsc(row_coefs: np.ndarray, cfg: TSCConfig, row_frac: int) -> TSCResult:
    """Run the TSC unit on integer row spectra of shape (..., 8, 8) = (row r, u)."""
    y = np.asarray(row_coefs, dtype=np.int64)
    lead = y.shape[:-2]
    out = y.copy()
    replaced = np.zeros(lead + (8,), dtype=bool)
    sad_all = np.zeros(lead + (8,), dtype=np.int64)
    tidx = np.zeros(lead + (8,), dtype=np.int64)
    tint = np.asarray(threshold_table(row_frac, cfg.unit), dtype=np.int64)

    m = np.full(lead, cfg.t0, dtype=np.int64)
    ref = y[..., 0, :].copy()
    n_sim_top = np.zeros(lead, dtype=np.int64)
    tidx[..., 0] = m
    for r in range(1, 8):
        tidx[..., r] = m
        sad = np.abs(y[..., r, :] - ref).sum(axis=-1)
        sad_all[..., r] = sad
        similar = (sad < tint[m]) & bool(cfg.enable)
        replaced[..., r] = similar
        out[..., r, :] = np.where(similar[..., None], ref, y[..., r, :])
        ref = np.where(similar[..., None], ref, y[..., r, :])
        m = np.where(similar, m, np.minimum(m + cfg.step, cfg.tmax))
        if r <= 3:
            n_sim_top += similar

    if cfg.approx_mode == "always":
        approx = np.ones(lead, dtype=bool)
    elif cfg.approx_mode == "tsc" and cfg.enable:
        approx = n_sim_top >= cfg.min_similar
    else:
        approx = np.zeros(lead, dtype=bool)
    return TSCResult(out, replaced, approx, sad_all, tidx)
