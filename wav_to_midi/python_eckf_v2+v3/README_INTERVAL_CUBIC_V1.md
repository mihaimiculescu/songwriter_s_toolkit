# ECKF V2 — Interval Juror Cubic V1

Based on `python_eckf_v2_interval_quadratic_v1`.

This build changes only the interval juror experiment:

- interval demand curve: quadratic -> cubic
- `W_INTERVAL`: `0.15` -> `0.30`

For the interval juror only, when a candidate violates the historical required-time prior:

    base_penalty = historical_vocal_transition_penalty(...)
    demand_excess = max(0, required_time / available_time - 1)
    juror_penalty = base_penalty * (1 + demand_excess**3)

The existing conversion to juror evidence remains capped at -1.0.

Unchanged:

- historical transition prior used elsewhere in the ECKF pipeline
- <=4-semitone historical free zone
- no two-sided excursion multiplier
- spectral, temporal, range jurors and their weights
- tournament and abstention logic
