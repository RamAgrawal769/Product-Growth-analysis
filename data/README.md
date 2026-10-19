# Data

Everything in `data/generated/` is produced by `product_analytics/data/generator.py`
and is **fully synthetic** -- there is no real product, no real company, and
no real user data anywhere in this repository.

## What's simulated

A hypothetical B2B project-management SaaS ("Flowboard"), 12,000 signups over
an 18-month window:

- **Acquisition channel** (6 channels) and **customer segment** (solo / small
  team / mid-market / enterprise), each with a hand-authored base activation
  and conversion rate.
- **Activation** (did the user complete a meaningful first session) and
  **90-day retention**, generated from those base rates plus a set of
  per-feature retention "lifts" that are known by construction (e.g.
  `invite_teammate` is deliberately the strongest driver) -- this ground
  truth is what the feature-adoption model in the README is checked against.
- **A simulated onboarding-nudge A/B test** (days 420-480 of the window) with
  a true injected effect of **+4.5 percentage points** on activation, used to
  evaluate the A/B-testing engine honestly, including what happens when the
  test is underpowered.

## Why synthetic, and why these specific effect sizes

The point of this repository is to demonstrate product-analytics
*methodology* -- funnel/cohort construction, causal-aware feature analysis,
proper A/B test statistics -- against data where the ground truth is known,
so the analysis can be checked for correctness rather than taken on faith.
No real INHAUS, EMUSKI, or any other company's user data is used or implied
anywhere in this project.
