"""Synthetic B2B SaaS product-usage data generator.

Simulates signups, an activation funnel, feature-usage events, and
subscription conversion for a hypothetical project-management SaaS product
("Flowboard"). Every distribution and effect size below is authored by hand
to be *plausible*, not copied from any real company's numbers -- see
data/README.md for the full rationale.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CHANNELS = ["organic_search", "referral", "content", "paid_search", "paid_social", "outbound"]
# Channel quality varies -- this is the ground truth the analysis should recover.
CHANNEL_ACTIVATION_LIFT = {
    "organic_search": 0.00,
    "referral": 0.12,
    "content": 0.05,
    "paid_search": -0.08,
    "paid_social": -0.14,
    "outbound": -0.05,
}
CHANNEL_WEIGHTS = {
    "organic_search": 0.22,
    "referral": 0.10,
    "content": 0.13,
    "paid_search": 0.28,
    "paid_social": 0.19,
    "outbound": 0.08,
}
SEGMENTS = ["solo", "small_team", "mid_market", "enterprise"]
SEGMENT_WEIGHTS = [0.35, 0.35, 0.22, 0.08]
SEGMENT_BASE_ACTIVATION = {"solo": 0.38, "small_team": 0.52, "mid_market": 0.47, "enterprise": 0.41}
SEGMENT_BASE_CONVERSION = {"solo": 0.06, "small_team": 0.16, "mid_market": 0.24, "enterprise": 0.31}

FEATURES = [
    "create_project", "invite_teammate", "create_task", "assign_task",
    "add_comment", "connect_integration", "create_automation", "view_report",
]
# Each feature's causal lift on 90-day retention (ground truth for the
# feature-adoption analysis to recover).
FEATURE_RETENTION_LIFT = {
    "create_project": 0.02,
    "invite_teammate": 0.16,
    "create_task": 0.03,
    "assign_task": 0.05,
    "add_comment": 0.02,
    "connect_integration": 0.11,
    "create_automation": 0.09,
    "view_report": 0.01,
}

N_DAYS = 540  # ~18 months
START_DATE = datetime(2025, 3, 1)

# The onboarding-flow A/B test: treatment nudges new users to invite a
# teammate during setup. Ground-truth effect: +4.5pp activation rate.
AB_TEST_TRUE_LIFT = 0.045
AB_TEST_START_DAY = 420
AB_TEST_END_DAY = 480


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_users(n_users: int, seed: int) -> pd.DataFrame:
    rng = _rng(seed)
    signup_day = rng.integers(0, N_DAYS, size=n_users)
    channel = rng.choice(list(CHANNEL_WEIGHTS.keys()), size=n_users, p=list(CHANNEL_WEIGHTS.values()))
    segment = rng.choice(SEGMENTS, size=n_users, p=SEGMENT_WEIGHTS)

    in_ab_window = (signup_day >= AB_TEST_START_DAY) & (signup_day < AB_TEST_END_DAY)
    ab_group = np.where(in_ab_window, rng.choice(["control", "treatment"], size=n_users), "not_in_test")

    df = pd.DataFrame({
        "user_id": [f"U-{i:06d}" for i in range(n_users)],
        "signup_day": signup_day,
        "signup_date": [START_DATE + timedelta(days=int(d)) for d in signup_day],
        "channel": channel,
        "segment": segment,
        "ab_group": ab_group,
    })
    return df


def _activation_prob(row: pd.Series) -> float:
    p = SEGMENT_BASE_ACTIVATION[row["segment"]] + CHANNEL_ACTIVATION_LIFT[row["channel"]]
    if row["ab_group"] == "treatment":
        p += AB_TEST_TRUE_LIFT
    return float(np.clip(p, 0.02, 0.95))


def simulate_activation_and_events(users: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (users with activation/retention outcomes, feature-event log)."""
    rng = _rng(seed + 1)
    users = users.copy()
    activation_p = users.apply(_activation_prob, axis=1)
    users["activated"] = rng.random(len(users)) < activation_p

    events = []
    for idx, row in users.iterrows():
        if not row["activated"]:
            # Unactivated users still get 0-1 shallow events sometimes.
            if rng.random() < 0.3:
                feat = rng.choice(FEATURES, p=_feature_probs(shallow=True))
                events.append((row["user_id"], row["signup_day"] + int(rng.integers(0, 3)), feat))
            continue
        n_events = rng.poisson(6) + 1
        feat_probs = _feature_probs(shallow=False)
        feats = rng.choice(FEATURES, size=n_events, p=feat_probs)
        days_offset = np.sort(rng.integers(0, 14, size=n_events))
        for f, d in zip(feats, days_offset):
            events.append((row["user_id"], row["signup_day"] + int(d), f))

    event_log = pd.DataFrame(events, columns=["user_id", "event_day", "feature"])

    # 90-day retention: activated users retain based on base rate + feature lifts,
    # gated so a user can only be evaluated if signup_day + 90 <= N_DAYS (observed).
    users["observable_90d"] = users["signup_day"] + 90 <= N_DAYS
    used_features = event_log.groupby("user_id")["feature"].apply(set)
    users["_features"] = users["user_id"].map(used_features).apply(lambda s: s if isinstance(s, set) else set())

    def retention_prob(row: pd.Series) -> float:
        if not row["activated"]:
            return 0.04
        base = 0.30
        lift = sum(FEATURE_RETENTION_LIFT[f] for f in row["_features"])
        return float(np.clip(base + lift, 0.03, 0.97))

    ret_p = users.apply(retention_prob, axis=1)
    users["retained_90d"] = np.where(users["observable_90d"], rng.random(len(users)) < ret_p, np.nan)

    # Trial -> paid conversion (only among activated users who are observable at day 30).
    users["observable_30d"] = users["signup_day"] + 30 <= N_DAYS
    conv_p = users["segment"].map(SEGMENT_BASE_CONVERSION) * users["activated"].astype(float)
    converted = rng.random(len(users)) < conv_p
    users["converted_paid"] = np.where(users["observable_30d"], converted, np.nan)

    users = users.drop(columns=["_features"])
    return users, event_log


def _feature_probs(shallow: bool) -> np.ndarray:
    if shallow:
        weights = np.array([3, 1, 2, 1, 1, 1, 1, 1], dtype=float)
    else:
        weights = np.array([5, 3, 6, 4, 4, 2, 2, 3], dtype=float)
    return weights / weights.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/generated")
    args = ap.parse_args()

    users = generate_users(args.n_users, args.seed)
    users, events = simulate_activation_and_events(users, args.seed)

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    users.to_csv(f"{args.out_dir}/users.csv", index=False)
    events.to_csv(f"{args.out_dir}/events.csv", index=False)
    print(f"Wrote {len(users)} users and {len(events)} events to {args.out_dir}/")


if __name__ == "__main__":
    main()
