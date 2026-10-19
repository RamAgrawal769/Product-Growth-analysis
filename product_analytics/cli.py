"""python -m product_analytics.cli <command>"""
from __future__ import annotations

import argparse
import json
import sys

from product_analytics.data.generator import generate_users, simulate_activation_and_events
from product_analytics.metrics.funnel import overall_funnel, funnel_by
from product_analytics.experiments.ab_test import analyze_proportion_test, required_sample_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["funnel", "ab-test", "sample-size"])
    ap.add_argument("--n-users", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baseline-rate", type=float, default=0.3)
    ap.add_argument("--mde", type=float, default=0.02)
    args = ap.parse_args()

    if args.command == "sample-size":
        n = required_sample_size(args.baseline_rate, args.mde)
        print(f"Required sample size per arm: {n}")
        return

    users = generate_users(args.n_users, args.seed)
    users, events = simulate_activation_and_events(users, args.seed)

    if args.command == "funnel":
        print(json.dumps(overall_funnel(users), indent=2))
        print(json.dumps(funnel_by(users, "channel").round(4).to_dict(orient="records"), indent=2))
    elif args.command == "ab-test":
        print(json.dumps(analyze_proportion_test(users), indent=2))


if __name__ == "__main__":
    main()
