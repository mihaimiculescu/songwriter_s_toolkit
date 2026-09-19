"""Run: python -m unittest discover -s tests -p test_selector_priority_regressions.py

Synthetic, selector-only regression tests; no MIDI and no generated pitches.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import python_eckf.initialization_candidates as selector


class SelectorPriorityRegression(unittest.TestCase):
    def choose(self, periods, harmonic_map, proposal, observed, prior=None):
        periodicity = SimpleNamespace(
            voiced=True, acf_frequency_hz=observed,
            cmndf_frequency_hz=observed, acf_peak=.975, cmndf_minimum=.026,
        )
        frame = [0.0] * 2048
        with (patch.object(selector, '_measured_period_candidates', return_value=[
                (hz, lag, .975, .026) for hz, lag in periods]),
              patch.object(selector, '_harmonic_support', side_effect=lambda d, f, fs, hz: harmonic_map.get(round(hz), ())),
              patch.object(selector, '_measured_amplitude_phase', return_value=(.1, 0.))):
            return selector.choose_initialization(
                None, frame, 44100, 2349056,
                SimpleNamespace(f0_hz=proposal, amplitude=.1, phase=0.),
                periodicity, previous_hz=prior,
                elapsed_ms=2048 / 44100 * 1000 if prior else None,
            )

    def test_third_submultiple_does_not_steal_479_period(self):
        choice = self.choose(
            [(479.347826, 92), (239.673913, 184), (159.782609, 276), (120.163, 367)],
            {479: tuple(range(2, 13)), 240: tuple(range(4, 13)),
             160: (1, 6, 9, 12), 120: (4, 8, 10, 11, 12)},
            32., 479.347826, 473.468988,
        )
        self.assertAlmostEqual(choice.frequency_hz, 479.347826, places=3)

    def test_previous_octave_case_remains_at_measured_fundamental(self):
        choice = self.choose(
            [(400.90909, 110), (200.454545, 220)],
            {401: (2, 3, 4, 5, 6), 200: (4, 6, 8)},
            797., 400.90909, 395.,
        )
        self.assertAlmostEqual(choice.frequency_hz, 400.90909, places=3)

    def test_missing_previous_pitch_does_not_make_submultiple_win(self):
        choice = self.choose(
            [(479.347826, 92), (159.782609, 276)],
            {479: tuple(range(2, 13)), 160: (1, 6, 9, 12)},
            32., 479.347826,
        )
        self.assertAlmostEqual(choice.frequency_hz, 479.347826, places=3)


if __name__ == '__main__':
    unittest.main()
