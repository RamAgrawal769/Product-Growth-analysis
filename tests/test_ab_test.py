import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from product_analytics.experiments.ab_test import analyze_proportion_test, required_sample_size, sequential_boundary_check


def test_analyze_proportion_test_detects_large_effect():
    rng = np.random.default_rng(0)
    n = 5000
    control = pd.DataFrame({"ab_group": ["control"] * n, "activated": rng.random(n) < 0.30})
    treatment = pd.DataFrame({"ab_group": ["treatment"] * n, "activated": rng.random(n) < 0.45})
    df = pd.concat([control, treatment], ignore_index=True)
    result = analyze_proportion_test(df)
    assert result["absolute_lift"] > 0.10
    assert result["significant_at_0.05"] is True
    assert result["p_value"] < 0.001


def test_analyze_proportion_test_null_effect_not_significant_most_of_the_time():
    rng = np.random.default_rng(1)
    n = 400
    control = pd.DataFrame({"ab_group": ["control"] * n, "activated": rng.random(n) < 0.30})
    treatment = pd.DataFrame({"ab_group": ["treatment"] * n, "activated": rng.random(n) < 0.30})
    df = pd.concat([control, treatment], ignore_index=True)
    result = analyze_proportion_test(df)
    assert abs(result["absolute_lift"]) < 0.15


def test_required_sample_size_decreases_with_larger_mde():
    small_mde = required_sample_size(baseline_rate=0.3, min_detectable_effect=0.02)
    large_mde = required_sample_size(baseline_rate=0.3, min_detectable_effect=0.10)
    assert large_mde < small_mde


def test_sequential_boundary_check_shape():
    rng = np.random.default_rng(2)
    c = rng.random(500) < 0.3
    t = rng.random(500) < 0.3
    seq = sequential_boundary_check(c, t, n_looks=5)
    assert len(seq) == 5
    assert (seq["p_value"].between(0, 1)).all()
    # Bonferroni-corrected significance is never more permissive than uncorrected.
    assert (seq["significant_bonferroni"].astype(int) <= seq["significant_uncorrected"].astype(int)).all()
