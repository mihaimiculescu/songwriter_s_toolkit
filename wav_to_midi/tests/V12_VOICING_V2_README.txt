V12 NOTE-GROUP TOURNAMENT + VOICING V2 (shadow diagnostics, not production)

Place audit_shadow_adjudicator_v22_fixed_v12.py and audit_voicing_juror_v2.py in tests/.
Run at repository root, with original V13/V19/V21 inputs extracted and the four mono WAVs in tests/.

V12 replay of V11_04 settings (V12's CLI defaults intentionally retain V11 original values; specify the tuned weights):
python tests/audit_shadow_adjudicator_v22_fixed_v12.py --case all --tests-root tests --wav-root tests --w-range 0.50 --w-spectral 0.90 --w-temporal 0.40 --w-interval 0.30 --out tests/shadow_adjudicator_v22_fixed_v12
7z a -r tests/shadow_adjudicator_v22_fixed_v12.zip tests/shadow_adjudicator_v22_fixed_v12

V2 voicing audit against V12, optional V11_04 baseline for transitions:
python tests/audit_voicing_juror_v2.py --v12-zip tests/shadow_adjudicator_v22_fixed_v12.zip --v11-zip tests/shadow_adjudicator_v22_fixed_v11_04.zip --wav-root tests --out tests/voicing_juror_v2
7z a -r tests/voicing_juror_v2.zip tests/voicing_juror_v2

V12 note grouping: upstream range gate; cluster hypotheses by V11 observation_cluster_cents; independently assign nearest equal-tempered note; group only if strictly <49 cents; out-of-tolerance survivors are kept separate rather than deleted; score only acoustically admitted candidates; highest candidate median ACF per note advances; an exact highest-ACF tie abstains; solo must be acoustically admitted. Compare representatives using ONLY their own direct recorded V21 pairs and the unmodified V11 graph logic. Never transfer eliminated contestants' victories, never infer unmeasured matches. Full V11 evidence measurement, weights, thresholds, and original transition curve unchanged. Voicing V2 reuses V1 waveform heuristic unmodified: results are observational periodicity labels, NOT trained human-voice classification or MIDI rests.

Outputs: V12 per-case pair_decisions, new_direct_comparisons, observation_summary with note_groups_json and grouping diagnostics, manifest. Voicing V2 individual periodicity CSV, cross_tab_by_song, representative examples, manifest including V11->V12 transitions when baseline supplied.
