# ECKF V2 variable-aperture active v1

This build starts from `python_eckf_v2_gain_invariance_fix_v1` and activates variable-aperture target formation in the offline V2 trajectory interpreter.

## What changes

The historical ladder is active:

- 24 ms
- 40 ms
- 64 ms
- 96 ms

The old 120 ms rule remains exported as audit evidence only; it no longer decides the production stable-target mask.

## Production rule

The implementation is deliberately conservative.

1. At every trusted trajectory point, evaluate all four apertures using the existing robust pitch-spread test (`q90-q10 <= 0.65 st`).
2. Compute the **largest contiguous passing aperture**, starting at 24 ms. A wide-window pass cannot confirm a target if a narrower aperture already failed.
3. A new stable-target **core** requires continuous coherence through at least **64 ms**.
4. The existing minimum stable duration remains **80 ms**.
5. After a core is confirmed, the 24 ms aperture may refine its left/right boundaries while pitch remains within the existing same-target tolerance (**0.45 st**) of the frozen core median.
6. Boundary growth cannot create a new target and cannot "walk" along a glissando by moving the target center.
7. The existing short-gap same-target merge remains unchanged.

This keeps 24 ms as the minimum/localization aperture without allowing every 24 ms local coincidence to become an independent stable note.

## New audit fields

The target-formation CSV adds:

- `va_largest_contiguous_passing_aperture_ms`
- `va_core_candidate`
- `va_core_after_duration`
- `va_edge_grown`

New reasons include:

- `VARIABLE_MIN_APERTURE_SPREAD_TOO_WIDE`
- `VARIABLE_NO_CONFIRMING_APERTURE`
- `VARIABLE_CORE_TOO_SHORT`
- `VARIABLE_STABLE_TARGET`
- `VARIABLE_EDGE_REFINEMENT`
- `MERGED_SAME_TARGET_GAP`

## Intentionally unchanged

- gain-invariance fix remains active
- `--matlab` remains frozen
- silence/noise threshold remains the existing fixed rule; adaptive silence is postponed until after V2 feature completion
- validity/rescue logic is unchanged
- smoothing remains 50 ms
- stable-range threshold remains 0.65 semitone
- minimum stable duration remains 80 ms
- same-target tolerance remains 0.45 semitone
- merge-gap rule remains unchanged

