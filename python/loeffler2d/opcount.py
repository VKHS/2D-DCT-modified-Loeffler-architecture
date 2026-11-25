"""Operation accounting on the Loeffler flow graph.

The engine is described as a list of two-operand operations.  Two kinds of
structural savings are evaluated:

* output pruning - only the operations that feed the requested outputs are
  executed (column-wise approximation keeps the triangle u + v <= Z);
* repeated inputs - rows replaced by the TSC unit make column inputs equal.
  Operations whose result is zero, that pass an operand through, that reduce
  to a shift (x + x) or that recompute an existing value are free.

The counts are what a data-aware engine with operand isolation would really
execute; they do not change the bit-exact results of the processor.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .constants import LOEFFLER_CONSTANTS

# (destination, kind, operand-1, operand-2 / constant name)
GRAPH = (
    ("a0", "add", "x0", "x7"), ("a7", "sub", "x0", "x7"),
    ("a1", "add", "x1", "x6"), ("a6", "sub", "x1", "x6"),
    ("a2", "add", "x2", "x5"), ("a5", "sub", "x2", "x5"),
    ("a3", "add", "x3", "x4"), ("a4", "sub", "x3", "x4"),
    ("b0", "add", "a0", "a3"), ("b3", "sub", "a0", "a3"),
    ("b1", "add", "a1", "a2"), ("b2", "sub", "a1", "a2"),
    ("s47", "add", "a4", "a7"),
    ("p_o3k", "mul", "s47", "O3_K"), ("p_o3a", "mul", "a7", "O3_A"), ("p_o3b", "mul", "a4", "O3_B"),
    ("b4", "add", "p_o3k", "p_o3a"), ("b7", "sub", "p_o3k", "p_o3b"),
    ("s56", "add", "a5", "a6"),
    ("p_o1k", "mul", "s56", "O1_K"), ("p_o1a", "mul", "a6", "O1_A"), ("p_o1b", "mul", "a5", "O1_B"),
    ("b5", "add", "p_o1k", "p_o1a"), ("b6", "sub", "p_o1k", "p_o1b"),
    ("Y0", "add", "b0", "b1"), ("Y4", "sub", "b0", "b1"),
    ("s23", "add", "b2", "b3"),
    ("p_ek", "mul", "s23", "E_K"), ("p_ea", "mul", "b3", "E_A"), ("p_eb", "mul", "b2", "E_B"),
    ("Y2", "add", "p_ek", "p_ea"), ("Y6", "sub", "p_ek", "p_eb"),
    ("d4", "add", "b4", "b6"), ("d6", "sub", "b4", "b6"),
    ("d7", "add", "b7", "b5"), ("d5", "sub", "b7", "b5"),
    ("Y1", "add", "d7", "d4"), ("Y7", "sub", "d7", "d4"),
    ("Y3", "mul", "d5", "R2"), ("Y5", "mul", "d6", "R2"),
)

OUTPUTS = tuple(f"Y{k}" for k in range(8))
STAGE_OF = {
    **{d: 1 for d in ("a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7")},
    **{d: 2 for d in ("b0", "b1", "b2", "b3", "s47", "s56", "p_o3k", "p_o3a", "p_o3b",
                      "p_o1k", "p_o1a", "p_o1b", "b4", "b5", "b6", "b7")},
    **{d: 3 for d in ("Y0", "Y4", "s23", "p_ek", "p_ea", "p_eb", "Y2", "Y6", "d4", "d5", "d6", "d7")},
    **{d: 4 for d in ("Y1", "Y7", "Y3", "Y5")},
}

#: SAD comparison of two 8-word spectra: 8 sub + 8 abs + 7 add + 1 compare
TSC_OPS_PER_COMPARISON = {"add": 15, "abs": 8, "cmp": 1}


def full_counts() -> dict[str, int]:
    mult = sum(1 for _, k, _, _ in GRAPH if k == "mul")
    add = sum(1 for _, k, _, _ in GRAPH if k in ("add", "sub"))
    return {"mult": mult, "add": add}


def _needed(outputs) -> set[str]:
    need = set(outputs)
    for dst, kind, a, b in reversed(GRAPH):
        if dst in need:
            need.add(a)
            if kind != "mul":
                need.add(b)
    return need


@lru_cache(maxsize=None)
def count_ops(input_labels: tuple[int, ...] = tuple(range(8)), outputs: tuple[str, ...] = OUTPUTS):
    """Non-trivial (mult, add) operations for an equality pattern and output subset.

    ``input_labels[i] == input_labels[j]`` declares x_i == x_j.
    """
    kconst = {n: v for n, (v, _, _) in LOEFFLER_CONSTANTS.items()}
    labels = sorted(set(input_labels))
    dim = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}

    def key(v):
        return tuple(np.round(v, 9))

    val: dict[str, np.ndarray] = {}
    for i, lab in enumerate(input_labels):
        e = np.zeros(dim)
        e[idx[lab]] = 1.0
        val[f"x{i}"] = e
    seen = {key(v) for v in val.values()}
    need = _needed(outputs)
    mult = add = 0
    for dst, kind, a, b in GRAPH:
        va = val[a]
        if kind == "mul":
            v = kconst[b] * va
            trivial = not np.any(np.abs(va) > 1e-12)
        else:
            vb = val[b]
            v = va + vb if kind == "add" else va - vb
            za = not np.any(np.abs(va) > 1e-12)
            zb = not np.any(np.abs(vb) > 1e-12)
            same = np.allclose(va, vb, atol=1e-12) or np.allclose(va, -vb, atol=1e-12)
            trivial = za or zb or same
        if not np.any(np.abs(v) > 1e-12):
            trivial = True
        if key(v) in seen or key(-v) in seen:
            trivial = True
        val[dst] = v
        seen.add(key(v))
        if dst in need and not trivial:
            if kind == "mul":
                mult += 1
            else:
                add += 1
    return mult, add


def labels_from_replaced(replaced) -> tuple[int, ...]:
    """Equality labels of a column vector given the TSC replacement mask."""
    lab, cur = [], 0
    for r, rep in enumerate(replaced):
        if r > 0 and not rep:
            cur += 1
        lab.append(cur)
    return tuple(lab)


def block_counts(replaced, approx: bool, zone: int) -> dict[str, int]:
    """Operations of one 8x8 block: row engine, column engine and TSC overhead."""
    fc = full_counts()
    row = {"mult": 8 * fc["mult"], "add": 8 * fc["add"]}
    labels = labels_from_replaced(tuple(bool(x) for x in replaced))
    cm = ca = 0
    for u in range(8):
        if approx:
            outs = tuple(f"Y{v}" for v in range(8) if u + v <= zone)
        else:
            outs = OUTPUTS
        if not outs:
            continue
        m, a = count_ops(labels, outs)
        cm += m
        ca += a
    return {
        "row_mult": row["mult"], "row_add": row["add"],
        "col_mult": cm, "col_add": ca,
        "tsc_add": 7 * TSC_OPS_PER_COMPARISON["add"],
        "tsc_abs": 7 * TSC_OPS_PER_COMPARISON["abs"],
        "tsc_cmp": 7 * TSC_OPS_PER_COMPARISON["cmp"],
    }


def image_counts(replaced: np.ndarray, approx: np.ndarray, zone: int, tsc_enabled: bool = True) -> dict[str, float]:
    """Average per-block operation counts over many blocks."""
    rep = np.asarray(replaced).reshape(-1, 8)
    apx = np.asarray(approx).reshape(-1)
    tot: dict[str, float] = {}
    cache = {}
    for r, a in zip(map(tuple, rep), apx):
        k = (r, bool(a))
        if k not in cache:
            cache[k] = block_counts(r, bool(a), zone)
        for name, v in cache[k].items():
            tot[name] = tot.get(name, 0.0) + v
    n = max(len(apx), 1)
    out = {k: v / n for k, v in tot.items()}
    if not tsc_enabled:
        for k in ("tsc_add", "tsc_abs", "tsc_cmp"):
            out[k] = 0.0
    return out
