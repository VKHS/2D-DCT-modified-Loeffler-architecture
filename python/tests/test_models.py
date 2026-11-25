import itertools
from pathlib import Path

import numpy as np
import pytest

from loeffler2d import (EXACT, FixedPointConfig, ModifiedLoeffler2D, TSCConfig, apply_tsc, dct2d,
                        dct2d_fixed_exact, dct8_fixed, dct_matrix, loeffler_dct2d, loeffler_dct8,
                        loeffler_dct8_orthonormal, quantized_constants, threshold_table, triangle_mask)
from loeffler2d.constants import ENGINE_ADDITIONS, ENGINE_MULTIPLICATIONS, constant_bits
from loeffler2d.fixed_point import engine_ranges, required_engine_width, row_pass
from loeffler2d.images import vector_blocks
from loeffler2d.opcount import count_ops, full_counts, labels_from_replaced
from loeffler2d.vectors import MODES, read_hex, unpack

ROOT = Path(__file__).resolve().parents[2]
RNG = np.random.default_rng(1234)


def test_dct_matrix_orthonormal():
    c = dct_matrix(8)
    assert np.allclose(c @ c.T, np.eye(8))


def test_loeffler_float_equals_orthonormal_dct():
    x = RNG.normal(0, 100, (2000, 8))
    assert np.allclose(loeffler_dct8_orthonormal(x), x @ dct_matrix().T, atol=1e-10)


def test_loeffler_scale_is_sqrt8_and_2d_scale_is_8():
    x = RNG.normal(0, 50, (100, 8, 8))
    assert np.allclose(loeffler_dct8(x[..., 0, :]) / np.sqrt(8), x[..., 0, :] @ dct_matrix().T)
    assert np.allclose(loeffler_dct2d(x), dct2d(x), atol=1e-9)


def test_operation_count_is_11_mult_29_add():
    assert full_counts() == {"mult": ENGINE_MULTIPLICATIONS, "add": ENGINE_ADDITIONS} == {"mult": 11, "add": 29}


def test_constants_fit_15_bits_for_cf13():
    q = quantized_constants(13)
    assert q["E_B"] == 15137 and q["R2"] == 11585
    assert constant_bits(13) == 15


def test_fixed_point_matches_float_within_one_lsb():
    blocks = RNG.integers(0, 256, (5000, 8, 8))
    f = dct2d_fixed_exact(blocks)
    r = dct2d(blocks - 128.0)
    assert np.abs(f - r).max() < 1.0
    assert np.mean(f != np.round(r)) < 0.06


def test_extreme_sign_patterns_stay_in_width():
    cfg = FixedPointConfig()
    lo, hi = -128 << cfg.row_frac, 127 << cfg.row_frac
    assert required_engine_width(lo, hi) == 13
    yr = engine_ranges(lo, hi)
    ylo = min(v[0] for k, v in yr.items() if k.startswith("Y"))
    yhi = max(v[1] for k, v in yr.items() if k.startswith("Y"))
    assert required_engine_width(ylo, yhi) == 16
    # exhaustive sign patterns of a full block through the 2-D datapath
    pats = np.array(list(itertools.product([0, 255], repeat=8)))
    y = row_pass(pats[:, None, :].repeat(8, axis=1), cfg)
    assert y.min() >= -(1 << 12) and y.max() < (1 << 12)


def test_output_range_12_bits():
    blocks = vector_blocks()
    for mode in MODES.values():
        f = ModifiedLoeffler2D(FixedPointConfig(), mode).transform_blocks(blocks).coefs
        assert f.min() >= -2048 and f.max() <= 2047


def test_threshold_table():
    t = threshold_table(2)
    assert t[:11] == [0, 23, 46, 69, 92, 115, 138, 162, 185, 208, 231]


def test_triangle_mask_sizes():
    assert triangle_mask(7).sum() == 36
    assert triangle_mask(5).sum() == 21
    assert triangle_mask(14).all()


def test_tsc_identical_rows_are_replaced_and_trigger_stage2():
    cfg = FixedPointConfig()
    block = np.tile(RNG.integers(0, 256, 8), (8, 1))
    res = apply_tsc(row_pass(block, cfg), TSCConfig(), cfg.row_frac)
    assert res.replaced.tolist() == [False] + [True] * 7
    assert bool(res.approx)


def test_tsc_threshold_adapts_on_dissimilar_rows():
    cfg = FixedPointConfig()
    block = RNG.integers(0, 256, (8, 8))
    res = apply_tsc(row_pass(block, cfg), TSCConfig(t0=1, step=1, tmax=4), cfg.row_frac)
    assert not res.replaced.any()
    assert res.tindex.tolist() == [1, 1, 2, 3, 4, 4, 4, 4]


def test_tsc_disabled_equals_exact_fixed_point():
    blocks = RNG.integers(0, 256, (300, 8, 8))
    a = ModifiedLoeffler2D(FixedPointConfig(), EXACT).transform_blocks(blocks).coefs
    assert np.array_equal(a, dct2d_fixed_exact(blocks))


def test_approximation_zeroes_outside_triangle():
    block = np.full((8, 8), 77)
    block[:, 3] = 200
    res = ModifiedLoeffler2D(FixedPointConfig(), TSCConfig(approx_mode="always", zone=2)).transform_blocks(block)
    assert np.all(res.coefs[~triangle_mask(2)] == 0)


def test_equal_inputs_reduce_operations():
    assert count_ops(tuple(range(8))) == (11, 29)
    assert count_ops((0,) * 8) == (0, 0)
    assert count_ops((0, 0, 1, 1, 2, 2, 3, 3)) == (11, 17)
    assert labels_from_replaced([False, True, False, True, True, False, False, True]) == (0, 0, 1, 1, 1, 2, 3, 3)
    m, a = count_ops(tuple(range(8)), ("Y0",))
    assert (m, a) == (0, 7)


@pytest.mark.parametrize("mode", list(MODES))
def test_committed_vectors_match_model(mode):
    d = ROOT / "vectors" / mode
    if not d.exists():
        pytest.skip("vectors not generated")
    rows = read_hex(d / "input_rows.hex")
    cols = read_hex(d / "expected_cols.hex")
    status = read_hex(d / "expected_status.hex")
    nb = len(rows) // 8
    blocks = np.array([unpack(w, 8, signed=False) for w in rows]).reshape(nb, 8, 8)
    res = ModifiedLoeffler2D(FixedPointConfig(), MODES[mode]).transform_blocks(blocks)
    exp = np.array([unpack(w, 12) for w in cols]).reshape(nb, 8, 8)  # [block, u, v]
    assert np.array_equal(np.swapaxes(exp, 1, 2), res.coefs)
    for b in range(nb):
        assert status[b] & 0xFF == sum(int(r) << i for i, r in enumerate(res.replaced[b]))
        assert (status[b] >> 8) & 1 == int(res.approx[b])
