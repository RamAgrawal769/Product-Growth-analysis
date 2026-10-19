"""Weekly-cohort 90-day retention."""
from __future__ import annotations

import numpy as np
import pandas as pd


def weekly_cohort_retention(users: pd.DataFrame) -> pd.DataFrame:
    obs = users[users["observable_90d"]].copy()
    obs["cohort_week"] = obs["signup_day"] // 7
    grouped = obs.groupby("cohort_week").agg(
        cohort_size=("user_id", "count"),
        retained=("retained_90d", "sum"),
    ).reset_index()
    grouped["retention_90d"] = grouped["retained"] / grouped["cohort_size"]
    return grouped


def retention_by(users: pd.DataFrame, by: str) -> pd.DataFrame:
    obs = users[users["observable_90d"]].copy()
    grouped = obs.groupby(by).agg(
        n=("user_id", "count"),
        retained=("retained_90d", "sum"),
    ).reset_index()
    grouped["retention_90d"] = grouped["retained"] / grouped["n"]
    return grouped.sort_values("retention_90d", ascending=False)


def matured_cohort_summary(users: pd.DataFrame) -> dict:
    obs = users[users["observable_90d"]]
    n = len(obs)
    retained = int(obs["retained_90d"].sum())
    return {"matured_cohort_users": n, "retained_90d": retained, "retention_90d": retained / n if n else 0.0}
