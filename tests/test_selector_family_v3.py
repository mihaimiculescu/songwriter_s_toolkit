"""Candidate-family regression fixtures; mocked acoustic observations, never MIDI."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import python_eckf.initialization_candidates as selector

class FamilyRegressions(unittest.TestCase):
    def choose(self, periods, harmonics, amplitudes, observed, proposal=363., prior=None):
        p = SimpleNamespace(voiced=True, acf_frequency_hz=observed,
                            cmndf_frequency_hz=observed, acf_peak=.813,
                            cmndf_minimum=.183)
        with (patch.object(selector, '_measured_period_candidates', return_value=periods),
              patch.object(selector, '_harmonic_support', side_effect=lambda d, f, fs, hz: harmonics.get(round(hz), ())),
              patch.object(selector, '_measured_amplitude_phase', side_effect=lambda f, fs, hz, start: (amplitudes.get(round(hz), .001), 0.))):
            return selector.choose_initialization(None, [0.]*2048, 44100, 2363392,
                SimpleNamespace(f0_hz=proposal, amplitude=.1, phase=0.), p,
                previous_hz=prior, elapsed_ms=46.44 if prior else None)

    def test_53_5917_shared_harmonics_do_not_steal_fundamental(self):
        choice=self.choose([(400.90909,110,.813171,.182524), (201.36986,219,.751191,.234847)],
            {401:tuple(range(5,13)), 201:(1,5,6,7,10,11,12)},
            {401:.08,201:.003},400.90909)
        self.assertAlmostEqual(choice.frequency_hz,400.90909,places=3)

    def test_genuinely_supported_lower_fundamental_not_suppressed(self):
        choice=self.choose([(400.90909,110,.813171,.182524), (201.36986,219,.85,.15)],
            {401:tuple(range(5,13)), 201:(1,5,6,7,10,11,12)},
            {401:.08,201:.035},201.36986)
        self.assertAlmostEqual(choice.frequency_hz,201.36986,places=3)

if __name__=='__main__': unittest.main()
