"""Signup -> activation -> paid-conversion funnel, sliced by channel and segment."""
from __future__ import annotations

import pandas as pd


def funnel_by(users: pd.DataFrame, by: str) -> pd.DataFrame:
    obs = users[users["observable_30d"]].copy()
    grouped = obs.groupby(by).agg(
        signups=("user_id", "count"),
        activated=("activated", "sum"),
        converted=("converted_paid", "sum"),
    ).reset_index()
    grouped["activation_rate"] = grouped["activated"] / grouped["signups"]
    grouped["conversion_rate"] = grouped["converted"] / grouped["signups"]
    grouped["activated_to_paid_rate"] = grouped["converted"] / grouped["activated"].replace(0, pd.NA)
    return grouped.sort_values("signups", ascending=False)


def overall_funnel(users: pd.DataFrame) -> dict:
    obs = users[users["observable_30d"]]
    n = len(obs)
    activated = int(obs["activated"].sum())
    converted = int(obs["converted_paid"].sum())
    return {
        "signups": n,
        "activated": activated,
        "converted_paid": converted,
        "activation_rate": activated / n,
        "conversion_rate": converted / n,
        "activated_to_paid_rate": converted / activated if activated else 0.0,
    }
