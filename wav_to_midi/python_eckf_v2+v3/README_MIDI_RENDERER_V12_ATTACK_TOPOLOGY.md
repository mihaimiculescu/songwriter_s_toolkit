# MIDI Renderer V12 — attack topology and expression ownership

V12 is built on the frozen V3R5 pitch tracker.  The tracker is unchanged,
including `BORROWING_MIN_HIGH_ACF = 0.77`.

The renderer remains clean-sheet: WAV evidence answers *when* an articulation
begins; V3R5 corrected-valid/structural evidence remains the normal pitch
authority; V3R5 primary frame periodicity is only a bounded rescue/corroboration
lane.

## Changes from V11

### 1. Short-closure tolerance removed

V11 provisionally allowed very short near-inactive closures to remain inside one
attack-to-pitch ownership object.  V12 removes that exception.

An acoustic collapse is a hard ownership barrier whenever all of the following
are true:

- the envelope reaches the recording-adaptive near-inactive region;
- the trough is materially below the acoustic material on both sides, or it
  terminates a detached weak precursor;
- an independent spectral restart follows.

Closure duration no longer pardons the interruption.  This deliberately prefers
correct Note-On separation over preserving the last few milliseconds of a
stop-consonant tail.

### 2. Trusted pitch-owner topology is a hard guard

A future corrected-valid island may associate its Note-On with an earlier
physical attack, including an attack in the decaying tail of the immediately
preceding owner.  It may not reach backward across the *start* of that preceding
trusted island.

This prevents the V11 failure where a future owner was backdated across an
intervening trusted owner and later monophonic overlap cleanup destroyed the
future owner's real corrected-valid span.

### 3. Proven attacks can split a trusted pitch island

V10/V11 wrongly coupled two decisions:

- primary periodicity may not replace trusted V3R5 pitch; and
- an attack discovered/corroborated by that lane may not split the trusted
  owner.

V12 separates them.

Inside a trusted corrected-valid island, a sufficiently strong independent WAV
attack with persistent primary-periodicity support may create a Note-On
boundary.  The pitch on both sides is still taken from V3R5 corrected-valid
pitch.  Primary periodicity corroborates the articulation; it does not seize
pitch authority.

This enforces the renderer principle:

> Pitch Bend may describe continuous pitch motion inside one articulation
> object.  It may not absorb an independently proven rearticulation.

The internal corroborated-attack gate remains recording-adaptive (the 80th
percentile of attack strength, with the existing minimum floor) to avoid turning
every spectral ripple into a note.

### 4. Expression timing is separated from attack timing

A Note-On may precede trustworthy voiced F0 because of a consonant, aspiration,
or half-spoken attack.  V10/V11 sometimes measured the whole note's CC11
baseline from that quiet early attack.

Each event now carries an `expression_anchor_s`:

- trusted events: first trustworthy V3R5 pitch point owned by that segment;
- primary-rescue events: first accepted primary pitch point.

`attack_dbfs` remains in the audit as the physical attack level.
`expression_dbfs` is the level used for the sustained CC11 baseline.

### 5. Pitch Bend reset is unconditional at every Note Off

Every rendered Note Off is followed at the same MIDI tick by Pitch Bend = 0.
At same-tick legato boundaries the event ordering is:

1. old Note Off;
2. PB reset to zero;
3. next note's RPN/PB setup;
4. next Note On.

The renderer remains monophonic and does not overlap notes merely to preserve
legato.

## New/updated audit fields

`*.mid.events.csv` now includes:

- `attack_dbfs`
- `expression_anchor_s`
- `expression_dbfs`
- `cc11_baseline`

`*.mid.attacks.csv` retains interruption depth/duration evidence but removes the
V11 `short_closure_tolerated` field because the exception no longer exists.

## Acceptance priorities

V12 is intentionally optimized for:

1. correct musical attacks;
2. preservation of V3R5 pitch authority;
3. no corrected-valid ownership loss caused by renderer overlap;
4. pitch bend only within one articulation object;
5. CC11 sustained loudness derived from the voiced owner rather than a preceding
   consonant/breath.

No song name, timestamp, or manually annotated wishlist time is encoded in the
implementation.
