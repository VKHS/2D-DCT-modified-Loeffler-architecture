"""Complete 2-D DCT processor model (row engine -> TSC -> transpose -> column engine)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fixed_point import FixedPointConfig, column_pass, final_scale, row_pass
from .reference import blockify, dct2d, idct2d, unblockify
from .tsc import EXACT, TSCConfig, apply_tsc, triangle_mask


@dataclass
class BlockResult:
    coefs: np.ndarray  # (..., 8, 8) integer orthonormal 2-D DCT, [v, u]
    replaced: np.ndarray  # (..., 8) bool
    approx: np.ndarray  # (...,) bool
    sad: np.ndarray  # (..., 8)
    tindex: np.ndarray  # (..., 8)
    row_coefs: np.ndarray = field(repr=False, default=None)  # row spectra after TSC


class ModifiedLoeffler2D:
    """Bit-accurate model of the 2-D DCT processor.

    Parameters
    ----------
    fixed : FixedPointConfig
        word lengths (CF, RF, pixel and output widths)
    tsc : TSCConfig
        TSC / column-approximation settings; ``loeffler2d.tsc.EXACT`` turns
        both reduction stages off.
    """

    def __init__(self, fixed: FixedPointConfig = FixedPointConfig(), tsc: TSCConfig = TSCConfig()):
        self.fixed = fixed
        self.tsc = tsc
        self._mask = triangle_mask(tsc.zone)

    # ------------------------------------------------------------------ blocks
    def transform_blocks(self, blocks: np.ndarray) -> BlockResult:
        """Transform (..., 8, 8) uint8 blocks."""
        y = row_pass(blocks, self.fixed)
        t = apply_tsc(y, self.tsc, self.fixed.row_frac)
        z = column_pass(t.rows, self.fixed)
        f = final_scale(z, self.fixed)
        f = np.where(t.approx[..., None, None] & ~self._mask, 0, f)
        return BlockResult(f, t.replaced, t.approx, t.sad, t.tindex, t.rows)

    # ------------------------------------------------------------------ images
    def forward(self, img: np.ndarray) -> tuple[np.ndarray, BlockResult]:
        """Blockwise forward transform of a grey image -> coefficient image."""
        img = np.asarray(img)
        res = self.transform_blocks(blockify(img).astype(np.int64))
        return unblockify(res.coefs, img.shape), res

    @staticmethod
    def inverse(coef_img: np.ndarray, shape=None) -> np.ndarray:
        """Floating-point inverse DCT, +128, clipped to 8 bits."""
        blocks = blockify(coef_img).astype(float)
        rec = idct2d(blocks) + 128.0
        return np.clip(np.round(unblockify(rec, shape)), 0, 255).astype(np.uint8)


def float_dct_image(img: np.ndarray) -> np.ndarray:
    """Floating-point orthonormal blockwise DCT of (img - 128)."""
    b = blockify(np.asarray(img, float)) - 128.0
    return unblockify(dct2d(b), np.asarray(img).shape)


def exact_processor(fixed: FixedPointConfig = FixedPointConfig()) -> ModifiedLoeffler2D:
    return ModifiedLoeffler2D(fixed, EXACT)
