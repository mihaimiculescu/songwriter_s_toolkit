"""Regression tests for waveform-authoritative ECKF initialization.
Run from repository root: python -m unittest discover -s tests -p test_initialization_candidates.py
"""
import unittest
from types import SimpleNamespace
import numpy as np

from python_eckf.config import ECKFConfig
from python_eckf.harmonic_change import HarmonicChangeDetector
from python_eckf.periodicity import assess_periodicity
from python_eckf.initialization_candidates import choose_initialization
from python_eckf.pitch_status import PitchStatus, note_boundary_permission


class TestInitializationCandidates(unittest.TestCase):
    def setUp(self):
        self.sr = 44100
        self.start = 2048 * 100
        self.detector = HarmonicChangeDetector(ECKFConfig(mode='offline'))
        t = (np.arange(2048) + self.start) / self.sr
        # Real periodic waveform with harmonic energy; not based on MIDI.
        self.frame = (0.18 * np.cos(2*np.pi*400*t) +
                      0.60 * np.cos(2*np.pi*800*t) +
                      0.25 * np.cos(2*np.pi*1200*t))
        self.periodicity = assess_periodicity(self.frame, self.sr)

    def test_octave_proposal_is_not_forced_on_waveform(self):
        self.assertTrue(self.periodicity.voiced)
        choice = choose_initialization(
            self.detector, self.frame, self.sr, self.start,
            SimpleNamespace(f0_hz=800., amplitude=.1, phase=0.),
            self.periodicity, previous_hz=400., elapsed_ms=2048/self.sr*1000)
        self.assertEqual(choice.source, 'waveform_periodicity')
        self.assertAlmostEqual(choice.frequency_hz, 400., delta=12.)
        self.assertGreaterEqual(choice.spectral_harmonics, 2)
        self.assertIsNotNone(choice.transition_penalty)

    def test_true_unresolved_does_not_grant_midi_boundary(self):
        status = np.array([PitchStatus.VOICED_VALID,
                           PitchStatus.VOICED_UNRESOLVED,
                           PitchStatus.VOICED_VALID], dtype=np.uint8)
        self.assertEqual(note_boundary_permission(status, 0, 3), 'DEFER_UNRESOLVED')


if __name__ == '__main__':
    unittest.main()
