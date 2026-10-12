# MIDI Renderer V14 — Stable-Target Owner Arbitration

## Status

V3R9 is frozen.  This revision changes the MIDI renderer only.

V14 is deliberately a narrow successor to V13.  It retains V13's complete
object-corridor ownership, interruption barriers, short-island protection,
reset support, primary-periodicity rescue, rearticulation logic, CC11 dynamics,
pitch-bend behavior, and monophonic cleanup.  Those mechanisms remain useful
as fail-safes even when V3R9 normally supplies clean input.

## Problem

V10 introduced the correct principle that a trusted pitch island may own a
physical attack that precedes the first trustworthy F0 frame.  V13 protects
that backdating with acoustic-object corridors and trusted-island topology.

One edge case remained: two different pitch owners could compete for the same
physical attack.  A materially later trusted island could borrow the attack
backward even though a CLI-frozen stable target existed essentially on the
attack itself.  Generic monophonic overlap cleanup then favored the later event
and could erase or severely shorten the nearer owner.

Historical examples on the pre-V3R9 corpus include:

- `MaInnecArseneEN` around 25.818 s: the nearby ~D4 stable owner lost its attack
  to a much later low trusted island around 26.070 s.
- `Comomelo` around 50.113 s: a ~10 ms low trusted island around 50.240 s was
  inflated backward by roughly 127 ms and displaced the nearby ~D4 owner.

These were renderer ownership failures.  V3R9 happens to make the two examples
much less visible by supplying saner pitch evidence, but the renderer should
still defend itself against the same class on future recordings.

## V14 rule

When V13 is about to backdate a trusted island to an earlier WAV attack, V14
asks whether that physical attack already has a materially nearer CLI-frozen
stable-target owner.

Backdating is denied only when all of the following are true:

1. a stable-target boundary lies within `max(30 ms, 3 * pitch_dt)` of the WAV
   attack;
2. that stable target precedes the future trusted island by at least
   `max(80 ms, 6 * pitch_dt)`;
3. stable-target pitch and future-island median pitch differ by at least
   4 semitones; and
4. the stable target is physically closer to the attack than the future island.

If the certificate fires, V14 does **not** invent a new pitch or delete the
future trusted island.  It merely refuses the future island's renderer-created
backdating.  The future island starts from its own trustworthy anchor and all
ordinary V13 mechanisms continue to operate.

The attack audit records the rejected borrowing as:

`nearer_stable_target_owns_physical_attack_v14`

## Regression behavior

On the historical V3R5 twelve-patient artifacts the certificate fires only four
times: twice in `MaInnecArseneEN`, once in `Comomelo`, and once in
`PREDESTINATI`.  The two known V9→V10 regression-family cases are repaired, and
the other two hits exhibit the same physical geometry: a real nearby stable
pitch owner followed by a later different-pitch trusted owner.

On the frozen V3R9 twelve-patient corpus the certificate is almost entirely
dormant: it fires once, in `MaInnecArseneEN` around 18.933 s, where the waveform
supports the nearby high stable owner before the later lower trusted island.
The other eleven songs keep V13 event assembly unchanged.

## Scope intentionally deferred

V14 does not attempt to solve the other renderer-side mechanisms identified
during the V3R6→V3R9 tracker work.  In particular, later revisions can address
object-onset peak selection, over-early primary-bridge backdating, local versus
global rearticulation gating, and the remaining corrected-valid spans not
consumed by the renderer.  Keeping these surgeries separate makes regressions
traceable.
