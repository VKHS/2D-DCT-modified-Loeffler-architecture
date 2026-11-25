#!/usr/bin/env python3
"""Quality and arithmetic of the processor model on the sample images (Markdown tables).

    python3 python/scripts/evaluate.py            # default TSC setting (t0 = 0.001)
    python3 python/scripts/evaluate.py --t0 3     # other default threshold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loeffler2d import EXACT, FixedPointConfig, ModifiedLoeffler2D, TSCConfig, psnr  # noqa: E402
from loeffler2d.codec import codec_psnr  # noqa: E402
from loeffler2d.images import available_samples  # noqa: E402
from loeffler2d.opcount import image_counts  # noqa: E402
from loeffler2d.processor import float_dct_image  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", type=int, default=1)
    ap.add_argument("--zone", type=int, default=5)
    a = ap.parse_args()
    cfg = FixedPointConfig()
    tsc = TSCConfig(t0=a.t0, zone=a.zone)
    print(f"TSC: t0 = {a.t0 / 1000:.3f}, step = 0.001, tmax = 0.010, min_similar = 3, zone = {a.zone}\n")
    print("| image | size | max \\|F − F_float\\| exact | rows replaced | blocks approx. | PSNR exact / TSC (dB) "
          "| JPEG q50 exact / TSC (dB) | JPEG q75 exact / TSC (dB) | column mult / add per block |")
    print("|---|---|---|---|---|---|---|---|---|")
    tot = []
    for name, img in available_samples().items():
        ex = ModifiedLoeffler2D(cfg, EXACT)
        pr = ModifiedLoeffler2D(cfg, tsc)
        ce, _ = ex.forward(img)
        cp, res = pr.forward(img)
        err = np.abs(ce - float_dct_image(img)).max()
        pe, pp = psnr(img, ex.inverse(ce, img.shape)), psnr(img, pr.inverse(cp, img.shape))
        q50 = codec_psnr(img, ce, 50)[0], codec_psnr(img, cp, 50)[0]
        q75 = codec_psnr(img, ce, 75)[0], codec_psnr(img, cp, 75)[0]
        oc = image_counts(res.replaced, res.approx, tsc.zone)
        rep = 100 * res.replaced[..., 1:].mean()
        apx = 100 * res.approx.mean()
        tot.append((rep, apx, q75[1] - q75[0], oc["col_mult"], oc["col_add"]))
        print(f"| {name} | {img.shape[1]}×{img.shape[0]} | {err:.2f} | {rep:.1f} % | {apx:.1f} % | "
              f"{pe:.2f} / {pp:.2f} | {q50[0]:.2f} / {q50[1]:.2f} | {q75[0]:.2f} / {q75[1]:.2f} | "
              f"{oc['col_mult']:.1f} / {oc['col_add']:.1f} |")
    t = np.array(tot)
    print(f"\nmean: rows replaced {t[:, 0].mean():.1f} %, blocks approximated {t[:, 1].mean():.1f} %, "
          f"JPEG q75 ΔPSNR {t[:, 2].mean():+.3f} dB, column engine {t[:, 3].mean():.1f} mult / "
          f"{t[:, 4].mean():.1f} add per block (exact: 88 / 232)")


if __name__ == "__main__":
    main()
