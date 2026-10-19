import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from product_analytics.metrics.funnel import funnel_by, overall_funnel
from product_analytics.metrics.retention import weekly_cohort_retention, retention_by, matured_cohort_summary
from product_analytics.metrics.north_star import weekly_north_star
from product_analytics.metrics.feature_adoption import build_feature_matrix, fit_retention_model
from product_analytics.experiments.ab_test import analyze_proportion_test, required_sample_size

st.set_page_config(page_title="Product Growth Analytics", layout="wide")
st.title("Flowboard Product Growth Analytics")

DATA_DIR = "data/generated"


@st.cache_data
def load_data():
    users = pd.read_csv(f"{DATA_DIR}/users.csv", parse_dates=["signup_date"])
    events = pd.read_csv(f"{DATA_DIR}/events.csv")
    return users, events


if not os.path.exists(f"{DATA_DIR}/users.csv"):
    st.error("Run `python scripts/run_analysis.py` first to generate data.")
    st.stop()

users, events = load_data()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Funnel", "Retention", "North Star", "Feature Adoption", "A/B Test"]
)

with tab1:
    st.subheader("Signup -> Activation -> Paid Conversion")
    overall = overall_funnel(users)
    c1, c2, c3 = st.columns(3)
    c1.metric("Activation rate", f"{overall['activation_rate']:.1%}")
    c2.metric("Conversion rate", f"{overall['conversion_rate']:.1%}")
    c3.metric("Activated -> Paid", f"{overall['activated_to_paid_rate']:.1%}")
    st.bar_chart(funnel_by(users, "channel").set_index("channel")[["activation_rate", "conversion_rate"]])
    st.dataframe(funnel_by(users, "segment"))

with tab2:
    st.subheader("90-day retention by weekly signup cohort")
    ret = weekly_cohort_retention(users)
    st.line_chart(ret.set_index("cohort_week")["retention_90d"])
    st.dataframe(retention_by(users, "segment"))
    st.write(matured_cohort_summary(users))

with tab3:
    st.subheader("North Star: Weekly Teams Completing Collaborative Work")
    ns = weekly_north_star(events)
    st.line_chart(ns.set_index("week")["north_star_users"])

with tab4:
    st.subheader("Which features predict 90-day retention?")
    feat_df = build_feature_matrix(users, events)
    result = fit_retention_model(feat_df)
    coef_df = pd.DataFrame(result["ranked_features"], columns=["feature", "coefficient"]).set_index("feature")
    st.bar_chart(coef_df)
    st.caption(f"Train accuracy: {result['train_accuracy']:.1%} | n={result['n_observations']}")

with tab5:
    st.subheader("Onboarding-nudge experiment")
    ab = analyze_proportion_test(users)
    c1, c2, c3 = st.columns(3)
    c1.metric("Control activation", f"{ab['control_rate']:.1%}")
    c2.metric("Treatment activation", f"{ab['treatment_rate']:.1%}", f"{ab['absolute_lift']:.2%}")
    c3.metric("p-value", f"{ab['p_value']:.3f}")
    st.write(f"95% CI on lift: [{ab['ci_95_low']:.2%}, {ab['ci_95_high']:.2%}]")
    baseline = st.slider("Baseline rate", 0.05, 0.6, float(ab["control_rate"]))
    mde = st.slider("Minimum detectable effect (pp)", 0.5, 10.0, 2.0) / 100
    st.write(f"Required sample size per arm: **{required_sample_size(baseline, mde):,}**")
