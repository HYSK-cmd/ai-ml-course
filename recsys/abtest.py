"""A/B testing statistics for the launch decision (stdlib statistics.NormalDist + numpy)."""
from __future__ import annotations


def sample_size_per_arm(p_baseline: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Two-sided two-proportion z-test. With p1 = p_baseline, p2 = p1 + mde_abs, p_bar = (p1 + p2) / 2:
        n = (z_{1-alpha/2} * sqrt(2 p_bar (1 - p_bar)) + z_{power} * sqrt(p1 (1-p1) + p2 (1-p2)))^2 / mde_abs^2
    rounded UP."""
    raise NotImplementedError


def two_proportion_ztest(conv_a: int, n_a: int, conv_b: int, n_b: int) -> tuple[float, float]:
    """Pooled two-sided z-test for B vs A. Returns (z, p_value); z > 0 means B converts better."""
    raise NotImplementedError


def simulate_power(p_a: float, p_b: float, n: int, n_sims: int = 2000, alpha: float = 0.05, seed: int = 0) -> float:
    """Monte-Carlo power: draw binomial conversions for both arms n_sims times, return the fraction of
    simulations where two_proportion_ztest gives p < alpha. Vectorize over simulations."""
    raise NotImplementedError
