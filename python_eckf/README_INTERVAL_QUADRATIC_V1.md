# ECKF V2 — Interval Juror Quadratic V1

Based on `python_eckf_v2_interval_interruption_barrier_v1`.

This build changes **only the interval juror's interpretation of the existing vocal transition prior**. It does **not** alter the historical transition penalty used by ECKF initialization, validity/rescue, octave reacquisition, range-reference evaluation, or the pitch-candidate field.

No two-sided excursion multiplier is added. `W_INTERVAL` remains `0.15` in this production test build.

For the interval juror only, when a candidate violates the historical required-time prior:

    base_penalty = historical_vocal_transition_penalty(...)
    demand_excess = max(0, required_time / available_time - 1)
    juror_penalty = base_penalty * (1 + demand_excess**2)

The existing conversion to juror evidence remains capped at -1.0, so grossly impossible leap/time combinations become maximum negative interval evidence rather than numerically unbounded.

The <=4-semitone historical free zone is unchanged in this build. No two-sided multiplier is used.
