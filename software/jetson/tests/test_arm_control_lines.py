"""No-device regression tests for the STM32 automatic-program signal pair."""
import struct
import termios
import unittest
from unittest.mock import patch

from vision.arm_serial import LinkedControlSerial, ArmSerial


class ControlLineTests(unittest.TestCase):
    def test_open_hooks_update_both_lines_in_one_ioctl(self):
        connection = LinkedControlSerial(port=None)
        connection.fd = 123
        connection.dtr = False
        connection.rts = False
        with patch('vision.arm_serial.fcntl.ioctl') as ioctl:
            connection._update_dtr_state()
            connection._update_rts_state()
        self.assertEqual(ioctl.call_count, 2)
        expected = struct.pack('I', termios.TIOCM_DTR | termios.TIOCM_RTS)
        for call in ioctl.call_args_list:
            self.assertEqual(call.args, (123, termios.TIOCMBIC, expected))

    def test_mixed_levels_fail_before_changing_hardware(self):
        connection = LinkedControlSerial(port=None)
        connection.fd = 123
        connection.dtr = False
        connection.rts = True
        with patch('vision.arm_serial.fcntl.ioctl') as ioctl:
            with self.assertRaises(ValueError): connection._update_control_pair()
            ioctl.assert_not_called()

    def test_external_motion_or_startup_is_not_mistaken_for_query_feedback(self):
        connection = ArmSerial.__new__(ArmSerial)
        connection.expected_motion_echoes = {}
        with self.assertRaisesRegex(RuntimeError, 'External motion'):
            connection._check_received(b'#000P2400T2000!#000P1772!')
        with self.assertRaisesRegex(RuntimeError, 'reset/startup'):
            connection._check_received(b'uart1 check ok!#255P1500T2000!')
        connection._check_received(b'#000PRAD!#000P1772!')

    def test_current_command_echo_is_allowed_but_expired_echo_is_rejected(self):
        import time
        connection = ArmSerial.__new__(ArmSerial)
        command = b'#000P1782T2000!'
        connection.expected_motion_echoes = {command: time.monotonic() + 1}
        connection._check_received(command)
        connection.expected_motion_echoes[command] = 0
        with self.assertRaisesRegex(RuntimeError, 'External motion'):
            connection._check_received(command)


if __name__ == '__main__':
    unittest.main()
