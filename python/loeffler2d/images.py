"""Test images: scikit-image samples (if installed) and deterministic synthetic scenes."""

from __future__ import annotations

import numpy as np

#: scikit-image sample images used in the evaluation (all CC0 / public domain)
SAMPLE_IMAGES = ("camera", "moon", "coffee", "chelsea", "brick", "gravel", "clock", "rocket")


def _to_gray_u8(im: np.ndarray) -> np.ndarray:
    im = np.asarray(im)
    if im.dtype == bool:
        im = im.astype(np.uint8) * 255
    if im.ndim == 3:
        im = im[..., :3].astype(float) @ np.array([0.299, 0.587, 0.114])
    if im.dtype != np.uint8:
        im = np.asarray(im, float)
        if im.max() <= 1.0:
            im = im * 255.0
        im = np.clip(np.round(im), 0, 255).astype(np.uint8)
    h, w = im.shape
    return np.ascontiguousarray(im[: h - h % 8, : w - w % 8])


def load_sample(name: str) -> np.ndarray:
    """Grey 8-bit sample image cropped to a multiple of 8 (synthetic fallback)."""
    try:
        import skimage.data as skd

        return _to_gray_u8(getattr(skd, name)())
    except Exception:  # pragma: no cover - fallback without scikit-image
        return synthetic_scene(256, seed=sum(map(ord, name)))


def available_samples() -> dict[str, np.ndarray]:
    return {n: load_sample(n) for n in SAMPLE_IMAGES}


def _fractal_noise(n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    f = np.fft.fftfreq(n)
    r = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    r[0, 0] = 1.0
    spec = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / r ** (beta / 2)
    spec[0, 0] = 0
    img = np.real(np.fft.ifft2(spec))
    return (img - img.mean()) / (img.std() + 1e-12)


def synthetic_scene(n: int = 256, seed: int = 7) -> np.ndarray:
    """Natural-looking test scene: 1/f^2 texture, sky gradient, disc, bars, text-like edges."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n
    img = 128 + 38 * _fractal_noise(n, 3.0, rng)
    sky = y < 0.32
    img[sky] = (210 - 60 * y[sky] / 0.32 + 1.5 * _fractal_noise(n, 4.0, rng)[sky])
    disc = (x - 0.72) ** 2 + (y - 0.62) ** 2 < 0.018
    img[disc] = 235
    bars = (x > 0.08) & (x < 0.40) & (y > 0.70) & (y < 0.92)
    img[bars] = np.where((np.floor(x[bars] * 48) % 2) == 0, 30, 220)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# blocks for the RTL / HLS / MATLAB golden vectors
# --------------------------------------------------------------------------

def vector_blocks(seed: int = 2019, n_random: int = 24) -> np.ndarray:
    """Deterministic set of 8x8 uint8 blocks covering corner cases.

    Includes constant and extreme blocks, worst-case sign patterns of the DCT
    basis (maximum coefficient magnitudes), gradients, edges, checkerboards,
    vertically uniform blocks (TSC replaces every row), near-uniform blocks
    with sub-threshold noise, the 'camera'-like texture and uniform noise.
    """
    rng = np.random.default_rng(seed)
    from .reference import dct_matrix

    c = dct_matrix(8)
    blocks = [
        np.zeros((8, 8)), np.full((8, 8), 255), np.full((8, 8), 128), np.full((8, 8), 127),
    ]
    for v, u in [(0, 1), (1, 0), (1, 1), (3, 5), (7, 7), (2, 6), (5, 2), (4, 4)]:
        basis = np.outer(c[v], c[u])
        blocks.append(np.where(basis >= 0, 255, 0))
        blocks.append(np.where(basis >= 0, 0, 255))
    r, q = np.mgrid[0:8, 0:8]
    blocks += [
        16 + 28 * q, 16 + 28 * r, 8 * (q + r) + 10,
        np.where(q < 4, 20, 230), np.where(r < 3, 200, 60),
        np.where((q + r) % 2 == 0, 0, 255), np.where((q // 2 + r // 2) % 2 == 0, 40, 210),
    ]
    row = rng.integers(0, 256, 8)
    blocks.append(np.tile(row, (8, 1)))  # all rows identical -> full replacement
    row = rng.integers(60, 200, 8)
    blocks.append(np.clip(np.tile(row, (8, 1)) + rng.integers(-1, 2, (8, 8)), 0, 255))
    top = np.tile(rng.integers(90, 160, 8), (4, 1))
    bottom = rng.integers(0, 256, (4, 8))
    blocks.append(np.vstack([top, bottom]))  # similar top half -> stage 2
    blocks.append(np.clip(150 + np.round(4 * np.sin(q / 2.0)) + 0 * r, 0, 255))
    for _ in range(n_random):
        blocks.append(rng.integers(0, 256, (8, 8)))
    smooth = synthetic_scene(64, seed=seed)
    for i in range(0, 64, 8):
        blocks.append(smooth[i:i + 8, (i * 3) % 56:(i * 3) % 56 + 8])
    return np.asarray(blocks, dtype=np.int64).astype(np.uint8)
