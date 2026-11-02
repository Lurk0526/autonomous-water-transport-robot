"""Hardware-free checks for binary camera packets and port isolation."""
import io
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from vision.openmv_client import OpenMVClient, camera_ports, decode_frame


class PartialSerial:
    def __init__(self, response):
        self.response = bytearray(response)
        self.writes = []

    def read(self, size):
        result = bytes(self.response[:min(3, size)])
        del self.response[:len(result)]
        return result

    def write(self, packet):
        self.writes.append(packet)
        return len(packet)

    def flush(self):
        pass


class ProtocolTests(unittest.TestCase):
    def test_rgb565_primary_colors(self):
        payload = struct.pack('<4H', 0xF800, 0x07E0, 0x001F, 0xFFFF)
        image = decode_frame(2, 2, 2, payload)
        np.testing.assert_array_equal(image.reshape(-1, 3),
                                      [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]])

    def test_gray(self):
        image = decode_frame(2, 1, 1, bytes([20, 220]))
        np.testing.assert_array_equal(image, [[[20, 20, 20], [220, 220, 220]]])

    def test_jpeg(self):
        stream = io.BytesIO()
        Image.new('RGB', (8, 6), (220, 30, 10)).save(stream, format='JPEG')
        self.assertEqual(decode_frame(8, 6, len(stream.getvalue()), stream.getvalue()).shape, (6, 8, 3))
        with self.assertRaises(ValueError):
            decode_frame(6, 8, len(stream.getvalue()), stream.getvalue())

    def test_packet_layout_and_partial_reads(self):
        payload = bytes([0, 127, 255, 20])
        connection = OpenMVClient.__new__(OpenMVClient)
        connection.serial = PartialSerial(struct.pack('<III', 2, 2, 1) + payload)
        image = connection.frame()
        self.assertEqual(image.shape, (2, 2, 3))
        self.assertEqual(connection.serial.writes,
                         [struct.pack('<BBI', 0x30, 0x81, 12), struct.pack('<BBI', 0x30, 0x82, 4)])

    def test_only_openmv_not_arm_or_system_ports(self):
        ports = [SimpleNamespace(device='/dev/ttyUSB0', vid=0x1A86, pid=0x7523, description='USB Serial'),
                 SimpleNamespace(device='/dev/ttyS0', vid=None, pid=None, description='n/a'),
                 SimpleNamespace(device='/dev/ttyACM0', vid=0x1209, pid=0xABD1, description='OpenMV')]
        with patch('vision.openmv_client.list_ports.comports', return_value=ports):
            self.assertEqual([p.device for p in camera_ports()], ['/dev/ttyACM0'])
            with patch('vision.openmv_client.serial.Serial') as serial_open:
                with self.assertRaises(ValueError):
                    OpenMVClient('/dev/ttyUSB0')
                serial_open.assert_not_called()

    def test_modern_frame_and_console_state(self):
        payload = bytes([20, 220])
        connection = OpenMVClient.__new__(OpenMVClient)
        connection.use_state = True
        connection.text = ''
        state = struct.pack('<IIII48s', 7, 2, 1, 1, b'camera ready\n')
        connection.serial = PartialSerial(state + payload)
        self.assertEqual(connection.frame().shape, (1, 2, 3))
        self.assertEqual(connection.console(), 'camera ready\n')
        self.assertEqual(connection.serial.writes[0], struct.pack('<BBI', 0x30, 0x93, 64))

    def test_payload_uses_separate_usb_phase(self):
        connection = OpenMVClient.__new__(OpenMVClient)
        connection.serial = PartialSerial(b'')
        connection._send(0x0D, 4, struct.pack('<I', 1))
        self.assertEqual(connection.serial.writes,
                         [struct.pack('<BBI', 0x30, 0x0D, 4), struct.pack('<I', 1)])


if __name__ == '__main__':
    unittest.main()
