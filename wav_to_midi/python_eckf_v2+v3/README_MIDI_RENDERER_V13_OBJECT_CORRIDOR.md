# MIDI Renderer V13 — Object-Corridor Attack Ownership

V13 is a clean-sheet-renderer continuation on top of the frozen V3R5 pitch
tracker.  It does **not** change tracker output or pitch authority.

## Why V13 exists

V10-V12 correctly separated WAV attack timing from V3R5 pitch authority, but
V12 could still assign future pitch to a weak precursor.  Its interruption test
looked only at short windows beside a valley.  If the precursor decayed slowly,
the short left window was already quiet and the true collapse was underestimated.

ElTelefono examples around 15.709, 153.443, 159.590 and 169.781 exposed the
pattern:

    weak/noisy precursor -> decay -> deep valley -> strong new restart -> trusted F0

V13 adjudicates the entire pre-pitch corridor before ownership is finalized.

## V13 rules

### 1. Full-corridor acoustic-object test

For every candidate attack that wants to own a later trusted V3R5 pitch island,
V13 evaluates the complete candidate->pitch corridor.  For every possible
collapse it measures:

- the earlier object's peak level from the candidate through its decaying lobe;
- the valley level;
- the later object's level around an independent spectral restart;
- recording-adaptive active-floor context.

A barrier is accepted when the morphology proves two acoustic objects.  It can
be established either by a recording-near-inactive valley with bilateral depth,
or by an exceptionally deep relative valley in a noisier recording.  A weak
detached precursor may also be rejected when it dies by >=8 dB and the later
object restarts >=15 dB above the valley and >=8 dB above the precursor object.
These are corpus-independent evidence gates; there are no song/time exceptions.

### 2. No short-closure pardon

There is no `short_closure_tolerated` rule.  Closure duration is recorded for
audit only.  If a short collapse truly separates two objects, ownership ends.
Correct attacks are preferred over preserving a few milliseconds of a preceding
/p/, /t/ or /k/ tail.

### 3. Final acoustic object owns future pitch

Candidates before the last proven object barrier are rejected.  Candidate
selection therefore happens *after* object separation, rather than allowing the
first plausible precursor to reserve the future pitch before stronger later
evidence is considered.

### 4. Primary rescue cannot bypass the object barrier

The V12 primary-periodicity bridge is retained, but it must pass the same V13
full-corridor continuity test before it may backdate a trusted island.  Primary
periodicity still cannot override trusted V3R5 pitch.

### 5. V12 invariants retained

- V3R5 corrected-valid pitch remains normal pitch authority.
- WAV attack evidence supplies timing, not pitch.
- Independent attacks may split a trusted owner; Pitch Bend cannot swallow a
  proven rearticulation.
- CC11 sustained level is measured from the voiced-expression anchor, not from
  an early consonant/breath Note-On.
- Every Note Off is followed at the same tick by Pitch Bend reset to zero.
- MIDI remains Type 1, PPQ 960, GM #65 Soprano Sax, channel 1, velocity 100,
  CC11 compression 0.3.

## Local preflight

Using the same frozen V3R5 sidecars and WAVs, V13 moves the four newly reported
ElTelefono regressions back to the final acoustic object:

- ~15.709 -> ~15.746
- ~153.443 -> ~153.453
- ~159.590 -> ~159.609
- ~169.781 -> ~169.786

The prior ElTelefono ~64 s attack topology / CC11 repair remains intact in the
preflight, and RATATA ~25.276 improves from the V12 ~25.401 start to ~25.291.
These are diagnostics only, not song-specific rules.
