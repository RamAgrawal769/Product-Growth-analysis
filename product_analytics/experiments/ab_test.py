"""A/B test analysis engine: two-proportion z-test + sample-size/MDE calculator.

Applied to the onboarding-nudge experiment ("prompt new users to invite a
teammate during setup") simulated in the generator with a true +4.5pp
activation lift, days 420-480 of the simulation window.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def analyze_proportion_test(users: pd.DataFrame, group_col: str = "ab_group",
                             outcome_col: str = "activated") -> dict:
    df = users[users[group_col].isin(["control", "treatment"])]
    control = df[df[group_col] == "control"][outcome_col]
    treatment = df[df[group_col] == "treatment"][outcome_col]

    n_c, n_t = len(control), len(treatment)
    p_c, p_t = control.mean(), treatment.mean()
    lift = p_t - p_c

    p_pool = (control.sum() + treatment.sum()) / (n_c + n_t)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    z = lift / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    se_unpooled = math.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    ci_low = lift - 1.96 * se_unpooled
    ci_high = lift + 1.96 * se_unpooled

    return {
        "n_control": int(n_c), "n_treatment": int(n_t),
        "control_rate": float(p_c), "treatment_rate": float(p_t),
        "absolute_lift": float(lift),
        "relative_lift_pct": float(lift / p_c * 100) if p_c > 0 else float("nan"),
        "z_statistic": float(z), "p_value": float(p_value),
        "ci_95_low": float(ci_low), "ci_95_high": float(ci_high),
        "significant_at_0.05": bool(p_value < 0.05),
    }


def required_sample_size(baseline_rate: float, min_detectable_effect: float,
                          alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-arm sample size for a two-proportion z-test (standard formula)."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p1 = baseline_rate
    p2 = baseline_rate + min_detectable_effect
    p_bar = (p1 + p2) / 2
    numerator = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) +
                 z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = (p2 - p1) ** 2
    return math.ceil(numerator / denominator)


def sequential_boundary_check(daily_control: np.ndarray, daily_treatment: np.ndarray,
                               alpha: float = 0.05, n_looks: int = 5) -> pd.DataFrame:
    """Naive repeated-looks simulation showing why peeking without correction
    inflates false-positive rate -- compares uncorrected vs. Bonferroni-
    corrected significance at each interim look on cumulative data."""
    rows = []
    look_points = np.linspace(len(daily_control) / n_looks, len(daily_control), n_looks).astype(int)
    corrected_alpha = alpha / n_looks
    for i, look in enumerate(look_points, start=1):
        c = daily_control[:look]
        t = daily_treatment[:look]
        n_c, n_t = len(c), len(t)
        p_c, p_t = c.mean(), t.mean()
        p_pool = (c.sum() + t.sum()) / (n_c + n_t)
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t)) if n_c and n_t else float("nan")
        z = (p_t - p_c) / se if se else 0.0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        rows.append({
            "look": i, "n_per_arm": look, "p_value": p_value,
            "significant_uncorrected": p_value < alpha,
            "significant_bonferroni": p_value < corrected_alpha,
        })
    return pd.DataFrame(rows)
