"""loeffler2d - models of the modified-Loeffler 2-D DCT processor.

    from loeffler2d import ModifiedLoeffler2D, FixedPointConfig, TSCConfig
    proc = ModifiedLoeffler2D(FixedPointConfig(), TSCConfig())
    coefs, info = proc.forward(image_u8)
"""

from .constants import LOEFFLER_CONSTANTS, quantized_constants
from .fixed_point import FixedPointConfig, dct2d_fixed_exact, dct8_fixed
from .loeffler import loeffler_dct2d, loeffler_dct8, loeffler_dct8_orthonormal
from .processor import ModifiedLoeffler2D, exact_processor, float_dct_image
from .reference import dct2d, dct_matrix, idct2d, psnr, ssim
from .tsc import EXACT, TSCConfig, apply_tsc, threshold_table, triangle_mask

__all__ = [
    "LOEFFLER_CONSTANTS", "quantized_constants", "FixedPointConfig", "dct2d_fixed_exact", "dct8_fixed",
    "loeffler_dct2d", "loeffler_dct8", "loeffler_dct8_orthonormal", "ModifiedLoeffler2D",
    "exact_processor", "float_dct_image", "dct2d", "dct_matrix", "idct2d", "psnr", "ssim",
    "EXACT", "TSCConfig", "apply_tsc", "threshold_table", "triangle_mask",
]

__version__ = "1.0.0"
