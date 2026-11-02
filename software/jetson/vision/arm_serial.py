"""Explicit CH340 connection. Importing this module never opens a port."""
import re
import fcntl
from pathlib import Path
import struct
import termios
import time

import serial
from serial.tools import list_ports


class MissingServoFeedback(RuntimeError):
    """No measured response arrived from one or more addressed servos."""
    def __init__(self, positions, replies, connection_report):
        self.positions = dict(positions)
        self.replies = dict(replies)
        self.connection_report = dict(connection_report)
        missing = '、'.join(f'{axis:03d}' for axis in sorted(replies))
        super().__init__(f'关节 {missing} 无位置反馈；请检查舵机独立供电、总线接线及舵机 ID')


class LinkedControlSerial(serial.Serial):
    """Apply CH340 DTR/RTS together, avoiding the board's reset combination.

    This board routes DTR/RTS to the STM32 auto-program circuit. pySerial's
    separate updates briefly mix the two levels even when both are False.
    A single Linux modem-control ioctl updates both bits in one driver call.
    """
    def _update_control_pair(self):
        if self._dtr_state != self._rts_state:
            raise ValueError('Arm DTR/RTS must stay at matching levels')
        operation = termios.TIOCMBIS if self._dtr_state else termios.TIOCMBIC
        mask = struct.pack('I', termios.TIOCM_DTR | termios.TIOCM_RTS)
        fcntl.ioctl(self.fd, operation, mask)

    def _update_dtr_state(self):
        self._update_control_pair()

    def _update_rts_state(self):
        self._update_control_pair()


class ArmSerial:
    def __init__(self, port):
        matches = [p for p in list_ports.comports() if p.device == port]
        if len(matches) != 1 or (matches[0].vid, matches[0].pid) != (0x1A86, 0x7523):
            raise ValueError('Explicit port is not the documented CH340 arm controller')
        guard = Path('/sys/module/ch341/parameters/jibot_no_auto_reset')
        if not guard.exists() or guard.read_text().strip().lower() not in ('y', '1'):
            raise RuntimeError('Jibot driver reset-line guard is inactive; serial port left closed')
        self.serial = LinkedControlSerial(port=None, baudrate=115200, timeout=0.02,
                                          write_timeout=0.2, exclusive=True)
        self.connection_report = {'line_update': 'paired_dtr_rts_ioctl', 'driver_reset_guard': True}
        self.expected_motion_echoes = {}
        # Inactive physical levels are required outside firmware downloading.
        self.serial.dtr = False
        self.serial.rts = False
        self.serial.port = port
        try:
            self.serial.open()
            startup = self._read_for(0.6)
            self.connection_report['startup'] = startup.decode('ascii', errors='replace')
            self._check_received(startup)
            self.serial.reset_input_buffer()
            self._send('$GETA!')
            handshake = self._read_for(0.3)
            self.connection_report['handshake'] = handshake.decode('ascii', errors='replace')
            self._check_received(handshake)
            if b'AAA' not in handshake:
                raise RuntimeError('Controller handshake failed')
        except Exception as exc:
            exc.connection_report = self.connection_report
            if self.serial.is_open:
                try:
                    self.stop()
                finally:
                    self.serial.close()
            raise

    def _send(self, text):
        data = text.encode('ascii')
        echoes = getattr(self, 'expected_motion_echoes', {})
        now = time.monotonic()
        previous = getattr(self, 'last_send_at', None)
        # The STM32 ASCII parser consumes one receive buffer at a time. Sending
        # PRAD immediately behind a move can lose the second transaction.
        if previous is not None and now - previous < 0.05:
            time.sleep(0.05 - (now - previous))
            now = time.monotonic()
        self.expected_motion_echoes = {key: end for key, end in echoes.items() if end > now}
        for motion in re.findall(rb'#\d{3}P\d{4}T\d{4}!', data):
            self.expected_motion_echoes[motion] = now + 4
        if self.serial.write(data) != len(data):
            raise IOError('Incomplete serial write')
        self.serial.flush()
        self.last_send_at = now

    def _check_received(self, data):
        if b'uart1 check' in data or b'#255P1500' in data:
            raise RuntimeError('Controller reset/startup detected; motion inhibited')
        echoes = getattr(self, 'expected_motion_echoes', {})
        for motion in re.findall(rb'#\d{3}P\d{4}T\d{4}!', data):
            if echoes.get(motion, 0) <= time.monotonic():
                raise RuntimeError(f'External motion command detected: {motion.decode()}')

    def _read_for(self, seconds):
        result = bytearray()
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            result.extend(self.serial.read(max(1, self.serial.in_waiting)))
            if len(result) > 16384:
                raise RuntimeError('Unexpected serial output size')
        return bytes(result)

    def _position_once(self, axis):
        if self.serial.in_waiting:
            self._check_received(self.serial.read(self.serial.in_waiting))
        self._send(f'#{axis:03d}PRAD!')
        reply = bytearray()
        pattern = rb'#' + f'{axis:03d}'.encode() + rb'P(\d{4})!'
        end = time.monotonic() + 0.18
        while time.monotonic() < end:
            reply.extend(self.serial.read(max(1, self.serial.in_waiting)))
            if re.search(pattern, reply):
                break
            if len(reply) > 16384:
                raise RuntimeError('Unexpected serial output size')
        self._check_received(reply)
        # Query echo has PRAD; only a complete measured Pxxxx reply is accepted.
        matches = re.findall(pattern, reply)
        if not matches:
            self.last_position_reply = bytes(reply)
            return None
        value = int(matches[-1])
        if not 500 <= value <= 2500:
            raise RuntimeError('Invalid measured servo position')
        return value

    def position(self, axis):
        if type(axis) is not int or axis not in range(6):
            raise ValueError('Invalid servo axis')
        for _ in range(2):
            value = self._position_once(axis)
            if value is not None:
                return value
        reply = getattr(self, 'last_position_reply', b'')
        raise MissingServoFeedback({}, {axis: reply}, getattr(self, 'connection_report', {}))

    def positions(self):
        """Check all six joints; echoes alone never count as servo feedback."""
        positions, replies = {}, {}
        for axis in range(6):
            try:
                positions[axis] = self.position(axis)
            except MissingServoFeedback as exc:
                replies.update(exc.replies)
        if replies:
            raise MissingServoFeedback(positions, replies, self.connection_report)
        return positions

    def move(self, targets, duration_ms, *, allowed_axes=(0, 3)):
        if not 400 <= duration_ms <= 1000 or not targets or not set(targets) <= set(allowed_axes) or not set(allowed_axes) <= set(range(6)):
            raise ValueError('Targets must belong to explicitly enabled joints')
        if any(type(p) is not int or not 500 <= p <= 2500 for p in targets.values()):
            raise ValueError('Invalid servo target')
        self._send('{' + ''.join(f'#{axis:03d}P{p:04d}T{duration_ms:04d}!'
                               for axis, p in sorted(targets.items())) + '}')

    def stop(self):
        self._send('$DST!')

    def stop_axes(self, axes):
        selected=set(axes)
        if not selected or not selected<=set(range(6)):
            raise ValueError('Invalid stop axes')
        for axis in sorted(selected):
            self._send(f'$DST:{axis}!')

    def close(self):
        try:
            if self.serial.is_open:
                self.stop()
        finally:
            self.serial.close()
