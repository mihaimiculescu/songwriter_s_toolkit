# V3R6 Hotfix 2 — Persistent Drift + Isolated False-Valid Repair

This release keeps the V3R6 primary-evidence recovery path unchanged and keeps
MIDI Renderer V13 byte-for-byte unchanged.

It narrows only replacement of already-valid tracker pitch.

## 1. Generic tracker-drift replacement now requires disagreement persistence

Previously, a strong primary frame could replace an already-valid 100-Hz ECKF
sample when the pitch disagreement was large and non-integer, provided primary
periodicity itself was sustained nearby.  That could flatten legitimate fast
mordent/gliss/melismatic motion inside one ~46-ms primary frame.

Hotfix 2 requires the *large non-integer disagreement itself* to be present in
an adjacent primary frame before generic `PRIMARY_TRACKER_DRIFT_REPLACED` is
authorized.

This preserves sustained repairs such as Damelo ~78.7–79.0 while suppressing
single-frame replacements such as the previously observed Comomelo ~47.60,
Afurisita ~24.10, MaInnec ~23.73, and Bodies ~19.69 cases.

## 2. Separate narrow certificate for isolated false-valid islands

A different failure class should not be forced through the persistence rule.
When the accepted ECKF trajectory contains only 1–2 valid 100-Hz samples,
rejected on both sides, those samples can still be plainly wrong even though
the error fits inside one primary frame.

Hotfix 2 may replace such an island only when all of the following hold:

- island length <= 2 samples;
- rejected samples immediately bound it on both sides;
- same-frame primary evidence is very strong (`ACF >= 0.95`, `CMNDF <= 0.05`);
- candidate differs from the accepted island by >= 3.5 semitones;
- disagreement is not an integer-family/register dispute;
- trusted primary context supports the candidate;
- nearest trusted valid pitch on BOTH sides, within 200 ms, lies within 2
  semitones of the primary candidate.

Across the frozen 12-song V3R5 corpus, replay of this certificate selected only:

- Comomelo ~50.24: false ~187.24 Hz -> primary ~285.44 Hz;
- Bodies ~116.83–116.84: false ~186 Hz -> primary ~393.75 Hz.

No song/time exceptions or MIDI information are used.

## 3. Snapshot discipline

Replacement certificates are evaluated against a snapshot of the V3R5 state
entering V3R6.  An earlier V3R6 promotion/replacement cannot manufacture
support for a later V3R6 replacement.

## Intended architectural boundary

- rejected/lost primary pitch recovery: unchanged V3R6 behavior;
- sustained tracker-state drift: V3R6 responsibility;
- isolated demonstrably false accepted island: V3R6 responsibility;
- continuous expressive motion inside an already-valid trajectory: preserve
  unless disagreement persists across primary frames;
- musical Note-On / competing-owner arbitration: remains renderer responsibility.

Renderer V13 is frozen in this package.
