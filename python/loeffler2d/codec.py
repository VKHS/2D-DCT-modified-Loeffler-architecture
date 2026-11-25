"""JPEG-style quantisation used to evaluate the transform inside a codec loop."""

from __future__ import annotations

import numpy as np

from .reference import blockify, idct2d, psnr, unblockify

#: ITU-T T.81 Annex K, Table K.1 (luminance)
JPEG_LUMA_Q50 = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.int64,
)


def zigzag_order(n: int = 8) -> np.ndarray:
    """Flat indices (v*n+u) of the JPEG zig-zag scan."""
    order = sorted(((v, u) for v in range(n) for u in range(n)),
                   key=lambda p: (p[0] + p[1], p[0] if (p[0] + p[1]) % 2 else p[1]))
    return np.array([v * n + u for v, u in order])


def quant_table(quality: int) -> np.ndarray:
    """IJG quality scaling of the luminance table."""
    q = int(np.clip(quality, 1, 100))
    s = 5000 // q if q < 50 else 200 - 2 * q
    t = (JPEG_LUMA_Q50 * s + 50) // 100
    return np.clip(t, 1, 255)


def quantize(coef_img: np.ndarray, quality: int) -> np.ndarray:
    qt = quant_table(quality).astype(float)
    b = blockify(coef_img).astype(float)
    return np.round(b / qt).astype(np.int64)


def codec_roundtrip(coef_img: np.ndarray, quality: int, shape) -> tuple[np.ndarray, int]:
    """Quantise, dequantise and inverse-transform; returns (image, nonzero count)."""
    qb = quantize(coef_img, quality)
    rec = idct2d(qb * quant_table(quality)) + 128.0
    img = np.clip(np.round(unblockify(rec, shape)), 0, 255).astype(np.uint8)
    return img, int(np.count_nonzero(qb))


def codec_psnr(img: np.ndarray, coef_img: np.ndarray, quality: int) -> tuple[float, int]:
    rec, nz = codec_roundtrip(coef_img, quality, img.shape)
    return psnr(img, rec), nz
