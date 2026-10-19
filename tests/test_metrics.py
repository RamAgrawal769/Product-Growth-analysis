import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from product_analytics.data.generator import generate_users, simulate_activation_and_events
from product_analytics.metrics.funnel import funnel_by, overall_funnel
from product_analytics.metrics.retention import weekly_cohort_retention, matured_cohort_summary
from product_analytics.metrics.north_star import weekly_north_star
from product_analytics.metrics.feature_adoption import build_feature_matrix, fit_retention_model


def _sample():
    users = generate_users(3000, seed=11)
    return simulate_activation_and_events(users, seed=11)


def test_overall_funnel_rates_in_range():
    users, _ = _sample()
    f = overall_funnel(users)
    assert 0 <= f["activation_rate"] <= 1
    assert 0 <= f["conversion_rate"] <= f["activation_rate"] + 1e-9


def test_funnel_by_channel_sums_to_overall():
    users, _ = _sample()
    obs = users[users["observable_30d"]]
    by_channel = funnel_by(users, "channel")
    assert by_channel["signups"].sum() == len(obs)


def test_weekly_cohort_retention_monotonic_index():
    users, _ = _sample()
    ret = weekly_cohort_retention(users)
    assert (ret["retention_90d"].between(0, 1)).all()
    assert list(ret["cohort_week"]) == sorted(ret["cohort_week"])


def test_north_star_nonnegative():
    users, events = _sample()
    ns = weekly_north_star(events)
    assert (ns["north_star_users"] >= 0).all()


def test_feature_adoption_model_recovers_invite_as_top_or_near_top():
    users, events = generate_users(8000, seed=99), None
    users, events = simulate_activation_and_events(users, seed=99)
    feat_df = build_feature_matrix(users, events)
    result = fit_retention_model(feat_df)
    top_two = [f for f, _ in result["ranked_features"][:2]]
    assert "invite_teammate" in top_two


def test_matured_cohort_summary_consistent():
    users, _ = _sample()
    s = matured_cohort_summary(users)
    assert s["retained_90d"] <= s["matured_cohort_users"]
