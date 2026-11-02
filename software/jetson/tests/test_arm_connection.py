"""Connection diagnosis must distinguish controller echoes from servo feedback."""
import unittest
from unittest.mock import Mock,patch

from tools.check_arm_connection import check_connection
from vision.arm_serial import ArmSerial, MissingServoFeedback


class ConnectionTests(unittest.TestCase):
    def transport(self):
        arm = ArmSerial.__new__(ArmSerial)
        arm.connection_report = {'handshake':'AAA', 'startup':''}
        return arm

    def test_absent_controller_never_opens_a_port_or_reuses_old_success(self):
        factory=Mock()
        with patch('tools.check_arm_connection.list_ports.comports',return_value=[]):
            report=check_connection(transport_factory=factory)
        factory.assert_not_called()
        self.assertFalse(report['controller_connected'])
        self.assertFalse(report['ready'])
        self.assertEqual(report['detected_ports'],[])

    def test_reports_all_missing_servos_and_preserves_partial_feedback(self):
        arm = self.transport()
        def position(axis):
            if axis == 0:
                return 1504
            raise MissingServoFeedback({}, {axis:f'#{axis:03d}PRAD!'.encode()}, arm.connection_report)
        arm.position = Mock(side_effect=position)
        with self.assertRaises(MissingServoFeedback) as caught:
            arm.positions()
        self.assertEqual(arm.position.call_count, 6)
        self.assertEqual(caught.exception.positions, {0:1504})
        self.assertEqual(set(caught.exception.replies), set(range(1,6)))
        self.assertEqual(caught.exception.connection_report['handshake'], 'AAA')
        self.assertIn('001、002、003、004、005', str(caught.exception))

    def test_reset_or_external_command_aborts_remaining_queries(self):
        arm = self.transport()
        arm.position = Mock(side_effect=RuntimeError('Controller reset/startup detected'))
        with self.assertRaisesRegex(RuntimeError, 'reset/startup'):
            arm.positions()
        arm.position.assert_called_once_with(0)

    def test_incomplete_bus_is_not_a_ready_connection_and_never_moves(self):
        arm = Mock()
        arm.connection_report = {'handshake':'AAA'}
        arm.positions.side_effect = MissingServoFeedback({0:1504}, {1:b'#001PRAD!'}, arm.connection_report)
        report = check_connection('/dev/ttyUSB0', lambda _:arm)
        self.assertTrue(report['controller_connected'])
        self.assertFalse(report['ready'])
        self.assertFalse(report['motion_commands_sent'])
        self.assertEqual(report['positions_p'], {'0':1504})
        self.assertEqual(report['missing_axes'], [1])
        arm.move.assert_not_called()
        arm.close.assert_called_once_with()

    def test_complete_bus_and_failed_cleanup_are_reported_separately(self):
        for failure in (None, OSError('USB disconnected')):
            with self.subTest(failure=failure):
                arm = Mock()
                arm.connection_report = {'handshake':'AAA'}
                arm.positions.return_value = {axis:1500 for axis in range(6)}
                arm.close.side_effect = failure
                report = check_connection('/dev/ttyUSB0', lambda _:arm)
                self.assertEqual(report['ready'], failure is None)
                self.assertEqual(report['missing_axes'], [])
                arm.move.assert_not_called()


if __name__=='__main__':
    unittest.main()
