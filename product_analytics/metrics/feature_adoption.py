"""Which feature-usage signals actually predict 90-day retention?

Uses L2-regularized logistic regression with segment/channel dummy controls
so a feature's coefficient reflects its association with retention *net of*
who's using it (bigger teams both invite teammates more and retain better --
controlling for segment is what keeps this from just re-discovering that).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from product_analytics.data.generator import FEATURES


def build_feature_matrix(users: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    obs = users[users["observable_90d"] & users["activated"]].copy()
    used = events[events["user_id"].isin(obs["user_id"])]
    pivot = pd.crosstab(used["user_id"], used["feature"]).clip(upper=1)
    pivot = pivot.reindex(columns=FEATURES, fill_value=0)
    df = obs.set_index("user_id").join(pivot, how="left").fillna(0)
    return df.reset_index()


def fit_retention_model(feature_df: pd.DataFrame) -> dict:
    df = feature_df.dropna(subset=["retained_90d"]).copy()
    y = df["retained_90d"].astype(int).values

    seg_dummies = pd.get_dummies(df["segment"], prefix="segment", drop_first=True)
    chan_dummies = pd.get_dummies(df["channel"], prefix="channel", drop_first=True)
    X = pd.concat([df[FEATURES], seg_dummies, chan_dummies], axis=1)
    feature_names = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values.astype(float))

    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(X_scaled, y)

    coefs = dict(zip(feature_names, model.coef_[0]))
    feature_coefs = {f: coefs[f] for f in FEATURES}
    ranked = sorted(feature_coefs.items(), key=lambda kv: kv[1], reverse=True)

    train_acc = model.score(X_scaled, y)
    return {
        "feature_coefficients": feature_coefs,
        "ranked_features": ranked,
        "all_coefficients": coefs,
        "train_accuracy": train_acc,
        "n_observations": len(df),
        "base_retention_rate": float(y.mean()),
    }
