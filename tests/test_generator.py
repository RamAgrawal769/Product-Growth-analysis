import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from product_analytics.data.generator import generate_users, simulate_activation_and_events, CHANNELS, SEGMENTS


def test_generate_users_shape():
    users = generate_users(500, seed=1)
    assert len(users) == 500
    assert set(users["channel"].unique()) <= set(CHANNELS)
    assert set(users["segment"].unique()) <= set(SEGMENTS)


def test_simulate_activation_bounds():
    users = generate_users(500, seed=1)
    users, events = simulate_activation_and_events(users, seed=1)
    assert users["activated"].dtype == bool
    assert set(events["feature"]) <= set([
        "create_project", "invite_teammate", "create_task", "assign_task",
        "add_comment", "connect_integration", "create_automation", "view_report",
    ])
    assert events["user_id"].isin(users["user_id"]).all()


def test_retention_only_defined_when_observable():
    users = generate_users(2000, seed=3)
    users, _ = simulate_activation_and_events(users, seed=3)
    unobservable = users[~users["observable_90d"]]
    assert unobservable["retained_90d"].isna().all()


def test_ab_group_assignment_only_in_window():
    users = generate_users(5000, seed=7)
    from product_analytics.data.generator import AB_TEST_START_DAY, AB_TEST_END_DAY
    in_window = (users["signup_day"] >= AB_TEST_START_DAY) & (users["signup_day"] < AB_TEST_END_DAY)
    assert (users.loc[~in_window, "ab_group"] == "not_in_test").all()
    assert users.loc[in_window, "ab_group"].isin(["control", "treatment"]).all()
