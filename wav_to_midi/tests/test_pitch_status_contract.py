import unittest
import numpy as np
from python_eckf.pitch_status import (PitchStatus as S, validate_pitch_status,
    validity_masks, note_boundary_permission)


class TestPitchStatus(unittest.TestCase):
    def test_unresolved_is_not_unvoiced(self):
        f = np.array([390., 0., 0., 392.])
        s = np.array([S.VOICED_VALID, S.VOICED_UNRESOLVED,
                      S.VOICED_UNRESOLVED, S.VOICED_VALID], dtype=np.uint8)
        valid, unresolved, unvoiced = validity_masks(f, s)
        self.assertEqual(valid.tolist(), [True, False, False, True])
        self.assertEqual(unresolved.tolist(), [False, True, True, False])
        self.assertFalse(unvoiced.any())
        self.assertEqual(note_boundary_permission(s, 0, 4), 'DEFER_UNRESOLVED')

    def test_breath_is_unvoiced(self):
        f = np.zeros(3)
        s = np.zeros(3, dtype=np.uint8)
        validate_pitch_status(f, s)
        self.assertEqual(note_boundary_permission(s, 0, 3), 'UNVOICED_INTERVAL')

    def test_positive_pitch_cannot_be_unresolved(self):
        with self.assertRaises(ValueError):
            validate_pitch_status(np.array([797.]), np.array([S.VOICED_UNRESOLVED]))


if __name__ == '__main__':
    unittest.main()
