"""
OceanMaster v13.2 — Syrjala Spatial Distribution Test
=====================================================
CATCH 使用的更嚴格空間驗證指標。

Syrjala (1996) test: 比較兩個空間分布是否統計上不可區分。
SSIM 對空間稀疏資料（大面積空白海域）偏寬鬆。
Syrjala 基於累積分布差異 + 排列檢驗，更能捕捉熱點位置偏差。

Reference:
  Syrjala, S.E. (1996). A statistical test for a difference between
  the spatial distributions of two populations. Ecology, 77(1), 75-80.

Usage:
    from engine.ml.syrjala_test import syrjala_test
    result = syrjala_test(predicted_map, observed_map)
    print(f"p-value = {result['p_value']:.3f}")
    # p > 0.05 → 預測與觀測分布無顯著差異 (CATCH 使用此準則)
"""

import numpy as np
import logging
from typing import Dict

log = logging.getLogger("OceanMaster.Syrjala")


def syrjala_test(
    predicted: np.ndarray,
    observed: np.ndarray,
    n_permutations: int = 999,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Syrjala two-sample spatial distribution test.

    Tests H0: the two spatial distributions are identical.
    Uses the Cramér-von Mises type statistic on cumulative
    row/column distributions.

    Args:
        predicted: 2D array (H, W), predicted density/probability
        observed: 2D array (H, W), observed density/probability
        n_permutations: number of permutation resamples
        seed: random seed for reproducibility

    Returns:
        {
            "statistic": float — Syrjala test statistic (Ψ)
            "p_value": float — p-value from permutation test
            "n_permutations": int
            "significant": bool — True if p < 0.05 (distributions differ)
        }
    """
    assert predicted.shape == observed.shape, \
        f"Shape mismatch: {predicted.shape} vs {observed.shape}"

    # Flatten NaN
    p = np.nan_to_num(predicted, nan=0.0).astype(np.float64)
    o = np.nan_to_num(observed, nan=0.0).astype(np.float64)

    # Normalize to probability distributions (sum = 1)
    p_sum = p.sum()
    o_sum = o.sum()
    if p_sum < 1e-10 or o_sum < 1e-10:
        log.warning("  Syrjala: one distribution is all zeros")
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n_permutations": n_permutations,
            "significant": False,
        }

    p_norm = p / p_sum
    o_norm = o / o_sum

    # Compute test statistic: Ψ = Σ (CDF_p - CDF_o)²
    observed_stat = _compute_psi(p_norm, o_norm)

    # Permutation test:
    # Pool the two distributions, randomly reassign to two groups,
    # recompute Ψ each time.
    rng = np.random.RandomState(seed)
    ny, nx = p.shape
    pooled = np.stack([p, o], axis=0)  # (2, H, W)

    n_greater = 0
    for _ in range(n_permutations):
        # Randomly swap values at each cell
        swap_mask = rng.randint(0, 2, size=(ny, nx))
        perm_p = np.where(swap_mask == 0, pooled[0], pooled[1])
        perm_o = np.where(swap_mask == 0, pooled[1], pooled[0])

        # Normalize
        ps = perm_p.sum()
        os = perm_o.sum()
        if ps > 1e-10 and os > 1e-10:
            perm_stat = _compute_psi(perm_p / ps, perm_o / os)
            if perm_stat >= observed_stat:
                n_greater += 1

    p_value = (n_greater + 1) / (n_permutations + 1)

    log.info(
        f"  Syrjala test: Ψ={observed_stat:.6f}, "
        f"p={p_value:.3f} ({n_permutations} permutations)"
    )

    return {
        "statistic": float(observed_stat),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
        "significant": p_value < 0.05,
    }


def _compute_psi(p: np.ndarray, o: np.ndarray) -> float:
    """
    Compute the Syrjala Ψ statistic.

    Ψ = average of row-cumsum and column-cumsum CvM statistics.

    Ψ = 0.5 * [Σ(CDF_row_p - CDF_row_o)² + Σ(CDF_col_p - CDF_col_o)²]
    """
    # Row-wise cumulative sums (east-west scan)
    cdf_row_p = np.cumsum(p, axis=1)
    cdf_row_o = np.cumsum(o, axis=1)
    psi_row = np.sum((cdf_row_p - cdf_row_o) ** 2)

    # Column-wise cumulative sums (north-south scan)
    cdf_col_p = np.cumsum(p, axis=0)
    cdf_col_o = np.cumsum(o, axis=0)
    psi_col = np.sum((cdf_col_p - cdf_col_o) ** 2)

    return 0.5 * (psi_row + psi_col)
