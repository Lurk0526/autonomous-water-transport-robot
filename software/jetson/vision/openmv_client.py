"""OpenMV USB debugger client; does not write flash or send servo commands.

Protocol reference: openmv/openmv v4.5.9 tools/pyopenmv.py (MIT).
"""
import io
import struct
import time

import numpy as np
from PIL import Image
import serial
from serial.tools import list_ports


CAPTURE_SCRIPT = """import sensor, time
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=1200)
clock = time.clock()
while True:
    clock.tick()
    sensor.snapshot()
"""


def camera_ports():
    """Never probe CH340 arm controllers or arbitrary system serial ports."""
    return [p for p in list_ports.comports()
            if (p.vid, p.pid) in {(0x1209, 0xABD1), (0xF055, 0x9800)}
            or (p.vid is not None and 'openmv' in (p.description or '').lower())]


def decode_frame(width, height, kind, data):
    if not (0 < width <= 4096 and 0 < height <= 4096):
        raise ValueError('OpenMV 帧尺寸无效')
    if kind == 1:
        gray = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
        return np.repeat(gray[:, :, None], 3, axis=2)
    if kind == 2:
        pixels = np.frombuffer(data, dtype='<u2').reshape(height, width)
        return np.stack((((pixels >> 11) * 255 // 31),
                         (((pixels >> 5) & 63) * 255 // 63),
                         ((pixels & 31) * 255 // 31)), axis=-1).astype(np.uint8)
    with Image.open(io.BytesIO(data)) as image:
        rgb = np.asarray(image.convert('RGB')).copy()
    if rgb.shape != (height, width, 3):
        raise ValueError('JPEG 尺寸与 OpenMV 帧头不符')
    return rgb


class OpenMVClient:
    def __init__(self, port):
        candidates = {p.device for p in camera_ports()}
        if port not in candidates:
            raise ValueError('该端口未识别为 OpenMV；不向机械臂控制板发送调试协议')
        self.serial = serial.Serial(port, baudrate=115200, timeout=0.2,
                                    write_timeout=1, exclusive=True)
        # Switching out of and back into IDE baud mode resets stale USB state.
        time.sleep(0.05)
        self.serial.baudrate = 921600
        time.sleep(0.05)
        self.serial.reset_input_buffer()
        self.started = False
        self.use_state = False
        self.text = ''

    def _send(self, command, length=0, payload=b''):
        # The debugger has distinct command and data phases. Keep writes apart.
        self.serial.write(struct.pack('<BBI', 0x30, command, length))
        self.serial.flush()
        if payload:
            self.serial.write(payload)
            self.serial.flush()

    def _read(self, size):
        if not 0 <= size <= 16 * 1024 * 1024:
            raise ValueError('OpenMV 数据长度超出限制')
        result = bytearray()
        deadline = time.monotonic() + 2
        while len(result) < size and time.monotonic() < deadline:
            result.extend(self.serial.read(size - len(result)))
        if len(result) != size:
            raise TimeoutError(f'OpenMV USB 超时：{len(result)}/{size} 字节')
        return bytes(result)

    def identify(self):
        self._send(0x80, 12)
        version = struct.unpack('<III', self._read(12))
        if not 1 <= version[0] <= 20:
            raise ValueError('OpenMV 固件版本回复异常')
        self.use_state = version >= (4, 5, 0)
        self._send(0x83, 64)
        architecture = self._read(64).split(b'\0', 1)[0].decode(errors='replace')
        return {'firmware': '.'.join(map(str, version)), 'architecture': architecture}

    def start(self):
        # Temporary script in RAM. SCRIPT_SAVE / SYS_RESET / UART are not used.
        self._send(0x06)
        time.sleep(0.15)
        self._send(0x0D, 4, struct.pack('<I', 1))
        script = CAPTURE_SCRIPT.encode('ascii')
        self._send(0x05, len(script), script)
        self.started = True
        time.sleep(0.3)

    def frame(self):
        if getattr(self, 'use_state', False):
            self._send(0x93, 64)
            flags, width, height, kind, text = struct.unpack('<IIII48s', self._read(64))
            if flags & 2:
                self.text += text.split(b'\0', 1)[0].decode(errors='replace')
            if not flags & 4:
                return None
        else:
            self._send(0x81, 12)
            width, height, kind = struct.unpack('<III', self._read(12))
        if width == 0:
            return None
        if not (0 < width <= 4096 and 0 < height <= 4096) or kind == 0:
            raise ValueError('OpenMV 帧头无效')
        size = width * height * kind if kind in (1, 2) else kind
        if size > 16 * 1024 * 1024:
            raise ValueError('OpenMV 帧过大')
        self._send(0x82, size)
        return decode_frame(width, height, kind, self._read(size))

    def console(self):
        if self.use_state:
            text, self.text = self.text, ''
            return text
        self._send(0x8E, 4)
        count, = struct.unpack('<I', self._read(4))
        if count > 1024 * 1024:
            raise ValueError('OpenMV 控制台数据长度异常')
        if not count:
            return ''
        self._send(0x8F, count)
        return self._read(count).decode(errors='replace')

    def close(self):
        try:
            if self.started:
                self._send(0x06)
                self._send(0x0D, 4, struct.pack('<I', 0))
        finally:
            self.serial.close()
