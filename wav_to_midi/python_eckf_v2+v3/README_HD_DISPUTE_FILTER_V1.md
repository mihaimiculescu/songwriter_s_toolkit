# Harmonic Detective Jury-Dispute Filter V1

This version retains the interval-cubic-v1 jury, including W_INTERVAL=0.30.

The only architectural change is in the conditional Harmonic Detective:

- it still scores every acoustically admissible candidate exactly as before;
- the full candidate audit remains exported;
- before ranking top/runner for a verdict, candidates are retained only if they
  are within 2.0 cents of a representative in the unresolved jury dispute;
- for multiple-undefeated-candidate abstentions, the dispute set is the
  undefeated tournament nodes;
- for no-decisive-pair abstentions, the dispute set is the admissible unresolved
  pair endpoints;
- existing harmonic score/support/margin thresholds are unchanged.

This makes the Harmonic Detective a tie-breaker for the dispute actually handed
to it, while preserving its independent spectral scoring and candidate audit.
