"""Protocol selection for legacy and identified magnetic-encoder servos."""
import unittest
from unittest.mock import patch

from tools.arm_small_test import resolve_position_protocol


class ProfileTests(unittest.TestCase):
    def test_legacy_position_mode_is_explicit(self):
        result = resolve_position_protocol(b'#000PMOD!#000PMOD1!', b'')
        self.assertTrue(result['mode_confirmed'])
        self.assertEqual(result['mode'], 1)

    def test_motor_mode_is_never_accepted_by_a_version_profile(self):
        with self.assertRaisesRegex(RuntimeError, 'Motor'):
            resolve_position_protocol(b'#000PMOD7!', b'#000@ ZL-ZServo_AD_CBM V2.1.16STG!')

    def test_exact_encoder_version_uses_verified_manufacturer_position_examples(self):
        result = resolve_position_protocol(b'#000PMOD!', b'#000PVER!#000@ ZL-ZServo_AD_CBM V2.1.16STG!')
        self.assertFalse(result['mode_confirmed'])
        self.assertTrue(result['single_axis_commissioning_only'])
        self.assertEqual(result['base_test_envelope_p'], [1010, 1990])

    def test_echo_unknown_version_or_changed_vendor_evidence_is_rejected(self):
        with self.assertRaises(RuntimeError):
            resolve_position_protocol(b'#000PMOD!', b'#000PVER!')
        with patch('pathlib.Path.read_bytes', return_value=b'changed'):
            with self.assertRaises(RuntimeError):
                resolve_position_protocol(b'#000PMOD!', b'#000@ ZL-ZServo_AD_CBM V2.1.16STG!')


if __name__ == '__main__':
    unittest.main()
