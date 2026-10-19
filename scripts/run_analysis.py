"""Regenerates every number and plot referenced in the README, from scratch."""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from product_analytics.data.generator import generate_users, simulate_activation_and_events, AB_TEST_START_DAY, AB_TEST_END_DAY
from product_analytics.metrics.funnel import funnel_by, overall_funnel
from product_analytics.metrics.retention import weekly_cohort_retention, retention_by, matured_cohort_summary
from product_analytics.metrics.feature_adoption import build_feature_matrix, fit_retention_model
from product_analytics.metrics.north_star import weekly_north_star
from product_analytics.experiments.ab_test import analyze_proportion_test, required_sample_size, sequential_boundary_check

ASSETS = "assets"
GENERATED = "data/generated"
os.makedirs(ASSETS, exist_ok=True)
os.makedirs(GENERATED, exist_ok=True)


def main() -> None:
    users = generate_users(12000, seed=42)
    users, events = simulate_activation_and_events(users, seed=42)
    users.to_csv(f"{GENERATED}/users.csv", index=False)
    events.to_csv(f"{GENERATED}/events.csv", index=False)

    results: dict = {}

    # --- Funnel ---
    results["overall_funnel"] = overall_funnel(users)
    funnel_channel = funnel_by(users, "channel")
    funnel_segment = funnel_by(users, "segment")
    funnel_channel.to_csv(f"{GENERATED}/funnel_by_channel.csv", index=False)
    funnel_segment.to_csv(f"{GENERATED}/funnel_by_segment.csv", index=False)
    results["funnel_by_channel"] = funnel_channel.round(4).to_dict(orient="records")

    fig, ax = plt.subplots(figsize=(8, 5))
    fc = funnel_channel.sort_values("activation_rate")
    ax.barh(fc["channel"], fc["activation_rate"], color="#2563eb")
    ax.set_xlabel("Activation rate"); ax.set_title("Activation rate by acquisition channel")
    for i, v in enumerate(fc["activation_rate"]):
        ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{ASSETS}/activation_by_channel.png", dpi=140); plt.close()

    # --- Retention ---
    cohort_ret = weekly_cohort_retention(users)
    cohort_ret.to_csv(f"{GENERATED}/weekly_cohort_retention.csv", index=False)
    results["matured_cohort_summary"] = matured_cohort_summary(users)
    ret_by_feat_seg = retention_by(users, "segment")
    results["retention_by_segment"] = ret_by_feat_seg.round(4).to_dict(orient="records")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(cohort_ret["cohort_week"], cohort_ret["retention_90d"], marker="o", ms=3, color="#16a34a")
    ax.set_xlabel("Signup cohort (week #)"); ax.set_ylabel("90-day retention")
    ax.set_title("90-day retention by weekly signup cohort")
    plt.tight_layout(); plt.savefig(f"{ASSETS}/retention_by_cohort.png", dpi=140); plt.close()

    # --- North Star ---
    ns = weekly_north_star(events)
    ns.to_csv(f"{GENERATED}/north_star_weekly.csv", index=False)
    results["north_star_latest_4wk_avg"] = float(ns.tail(4)["north_star_users"].mean())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ns["week"], ns["north_star_users"], color="#7c3aed")
    ax.set_xlabel("Week"); ax.set_ylabel("Weekly Teams Completing Collaborative Work")
    ax.set_title("North Star Metric trend")
    plt.tight_layout(); plt.savefig(f"{ASSETS}/north_star_trend.png", dpi=140); plt.close()

    # --- Feature adoption model ---
    feat_df = build_feature_matrix(users, events)
    model_results = fit_retention_model(feat_df)
    results["feature_adoption_model"] = {
        "ranked_features": model_results["ranked_features"],
        "train_accuracy": model_results["train_accuracy"],
        "n_observations": model_results["n_observations"],
        "base_retention_rate": model_results["base_retention_rate"],
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    ranked = model_results["ranked_features"]
    names = [r[0] for r in ranked]; vals = [r[1] for r in ranked]
    colors = ["#16a34a" if v > 0 else "#dc2626" for v in vals]
    ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Logistic regression coefficient (standardized)")
    ax.set_title("Which features predict 90-day retention (segment/channel-controlled)")
    plt.tight_layout(); plt.savefig(f"{ASSETS}/feature_retention_coefficients.png", dpi=140); plt.close()

    # --- A/B test ---
    ab_result = analyze_proportion_test(users)
    results["ab_test"] = ab_result

    mde_needed = required_sample_size(baseline_rate=ab_result["control_rate"], min_detectable_effect=0.02)
    results["sample_size_for_2pp_mde"] = mde_needed

    ab_users = users[(users["ab_group"].isin(["control", "treatment"]))].copy()
    ab_users["day_in_test"] = ab_users["signup_day"] - AB_TEST_START_DAY
    daily_c = ab_users[ab_users["ab_group"] == "control"].sort_values("day_in_test")["activated"].values
    daily_t = ab_users[ab_users["ab_group"] == "treatment"].sort_values("day_in_test")["activated"].values
    n = min(len(daily_c), len(daily_t))
    seq = sequential_boundary_check(daily_c[:n], daily_t[:n])
    seq.to_csv(f"{GENERATED}/ab_test_sequential_looks.csv", index=False)
    results["ab_test_sequential_looks"] = seq.round(4).to_dict(orient="records")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Control", "Treatment"], [ab_result["control_rate"], ab_result["treatment_rate"]],
           color=["#94a3b8", "#2563eb"])
    for i, v in enumerate([ab_result["control_rate"], ab_result["treatment_rate"]]):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=11)
    ax.set_ylabel("Activation rate"); ax.set_title("Onboarding-nudge A/B test: activation rate")
    plt.tight_layout(); plt.savefig(f"{ASSETS}/ab_test_result.png", dpi=140); plt.close()

    with open(f"{ASSETS}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("Analysis complete.")
    print(f"Overall activation rate: {results['overall_funnel']['activation_rate']:.1%}")
    print(f"Overall conversion rate: {results['overall_funnel']['conversion_rate']:.1%}")
    print(f"A/B test lift: {ab_result['absolute_lift']:.2%} (p={ab_result['p_value']:.4f})")
    print(f"Top retention-predicting feature: {model_results['ranked_features'][0]}")


if __name__ == "__main__":
    main()
