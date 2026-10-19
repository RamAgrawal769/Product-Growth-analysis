"""North Star Metric: Weekly Teams Completing Collaborative Work (WTCCW).

Chosen over a simpler metric like WAU because it requires an event that
indicates *multi-user, in-product collaboration* (invite_teammate or
add_comment/assign_task -- i.e. the product's actual value proposition),
not just any login. See README's "Why this North Star" section for the
full reasoning against the alternatives considered.
"""
from __future__ import annotations

import pandas as pd

COLLABORATION_EVENTS = {"invite_teammate", "add_comment", "assign_task"}


def weekly_north_star(events: pd.DataFrame) -> pd.DataFrame:
    collab = events[events["feature"].isin(COLLABORATION_EVENTS)].copy()
    collab["week"] = collab["event_day"] // 7
    weekly = collab.groupby("week")["user_id"].nunique().reset_index(name="north_star_users")
    return weekly.sort_values("week")
