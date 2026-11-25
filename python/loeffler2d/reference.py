"""Floating-point references: orthonormal DCT-II, blocking helpers, metrics."""

from __future__ import annotations

import numpy as np

N = 8


def dct_matrix(n: int = N) -> np.ndarray:
    """Orthonormal DCT-II matrix C with X = C @ x."""
    k = np.arange(n)[:, None]
    m = np.arange(n)[None, :]
    c = np.cos((2 * m + 1) * k * np.pi / (2 * n))
    c[0, :] *= np.sqrt(1.0 / n)
    c[1:, :] *= np.sqrt(2.0 / n)
    return c


_C8 = dct_matrix(N)


def dct1d(x: np.ndarray) -> np.ndarray:
    """Orthonormal 8-point DCT along the last axis."""
    return np.asarray(x, dtype=float) @ _C8.T


def idct1d(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=float) @ _C8


def dct2d(block: np.ndarray) -> np.ndarray:
    """Orthonormal 2-D DCT F = C X C^T of (..., 8, 8) blocks."""
    b = np.asarray(block, dtype=float)
    return _C8 @ b @ _C8.T


def idct2d(coefs: np.ndarray) -> np.ndarray:
    c = np.asarray(coefs, dtype=float)
    return _C8.T @ c @ _C8


def blockify(img: np.ndarray, n: int = N) -> np.ndarray:
    """Split an (H, W) image into (H/n, W/n, n, n) blocks (edge-padded)."""
    img = np.asarray(img)
    h, w = img.shape
    ph, pw = (-h) % n, (-w) % n
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw)), mode="edge")
    hh, ww = img.shape
    return img.reshape(hh // n, n, ww // n, n).swapaxes(1, 2)


def unblockify(blocks: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    bh, bw, n, _ = blocks.shape
    img = blocks.swapaxes(1, 2).reshape(bh * n, bw * n)
    if shape is not None:
        img = img[: shape[0], : shape[1]]
    return img


def psnr(ref: np.ndarray, test: np.ndarray, peak: float = 255.0) -> float:
    err = np.mean((np.asarray(ref, float) - np.asarray(test, float)) ** 2)
    return float("inf") if err == 0 else float(10 * np.log10(peak ** 2 / err))


def ssim(ref: np.ndarray, test: np.ndarray, peak: float = 255.0) -> float:
    """Gaussian-window SSIM (Wang et al. 2004), single scale."""
    from scipy.ndimage import gaussian_filter

    x = np.asarray(ref, float)
    y = np.asarray(test, float)
    c1, c2 = (0.01 * peak) ** 2, (0.03 * peak) ** 2
    mx, my = gaussian_filter(x, 1.5), gaussian_filter(y, 1.5)
    sxx = gaussian_filter(x * x, 1.5) - mx * mx
    syy = gaussian_filter(y * y, 1.5) - my * my
    sxy = gaussian_filter(x * y, 1.5) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return float(s.mean())
