#!/usr/bin/env python3
"""Live OpenMV preview and explicit start/stop controls for bowl tracking."""
import argparse
import codecs
from datetime import datetime
import json
import os
import signal
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PyQt5 import QtCore, QtGui, QtWidgets, QtNetwork
import numpy as np
from PIL import Image
from vision.openmv_client import OpenMVClient, camera_ports


class CameraWorker(QtCore.QThread):
    state = QtCore.pyqtSignal(str)
    frame_ready = QtCore.pyqtSignal(object)
    message = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()
        self.active = True
        self.latest = None
        self.lock = threading.Lock()

    def run(self):
        while not self.stop_event.is_set():
            client = None
            try:
                if not self.active:
                    self.stop_event.wait(0.2)
                    continue
                ports = camera_ports()
                if not ports:
                    self.state.emit('等待 OpenMV USB 连接')
                    self.stop_event.wait(2)
                    continue
                if len(ports) > 1:
                    self.state.emit('检测到多块 OpenMV，请只连接一块再取流')
                    self.stop_event.wait(2)
                    continue
                client = OpenMVClient(ports[0].device)
                identity = client.identify()
                self.message.emit(f"OpenMV {ports[0].device} · {identity}")
                client.start()
                self.state.emit('正在启动相机')
                last_frame = time.monotonic()
                last_console = 0
                while self.active and not self.stop_event.is_set():
                    rgb = client.frame()
                    if rgb is not None:
                        last_frame = time.monotonic()
                        with self.lock:
                            self.latest = (rgb, last_frame)
                    if time.monotonic() - last_console > 1:
                        console = client.console()
                        if console.strip():
                            self.message.emit(console.strip())
                        last_console = time.monotonic()
                    if time.monotonic() - last_frame > 10:
                        raise TimeoutError('OpenMV 10 秒没有提供图像，请检查传感器与固件')
                    self.stop_event.wait(0.02)
            except Exception as exc:
                self.state.emit('相机连接异常')
                self.message.emit(str(exc))
                self.stop_event.wait(2)
            finally:
                with self.lock:
                    self.latest = None
                if client:
                    try:
                        client.close()
                    except Exception as exc:
                        self.message.emit(f'关闭相机：{exc}')


class Window(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle('Jibot1 · OpenMV 视觉工作台')
        self.setWindowIcon(QtGui.QIcon(str(ROOT/'assets/jibot-bowl.svg')))
        self.resize(1080, 760)
        self.camera_state = '等待 OpenMV USB 连接'
        self.rgb = None
        self.frame_time = None
        self.frame_count = 0
        self.start_time = time.monotonic()
        self.bridge = None
        self.follow_operation = None
        self.follow_pending_stop = False
        self.follow_pending_close = False
        self.follow_message = ''
        self.follow_error = ''
        self.follow_output = ''
        self.follow_progress = ''
        self.follow_decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.follow_process = QtCore.QProcess(self)
        self.follow_process.setWorkingDirectory(str(ROOT))
        self.follow_process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.follow_process.readyReadStandardOutput.connect(self.read_follow_output)
        self.follow_process.finished.connect(self.follow_control_finished)
        self.follow_process.errorOccurred.connect(self.follow_control_error)
        self.worker = CameraWorker()
        self.worker.state.connect(self.set_state)
        self.worker.message.connect(self.log)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        heading = QtWidgets.QLabel('OpenMV 实时视觉')
        font = heading.font()
        font.setPointSize(20)
        heading.setFont(font)
        layout.addWidget(heading)
        self.state_label = QtWidgets.QLabel(self.camera_state)
        layout.addWidget(self.state_label)
        self.preview = QtWidgets.QLabel('等待摄像头画面\n请将 OpenMV 板的 USB 接口连接至 Jetson')
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumSize(320, 240)
        self.preview.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.preview.setStyleSheet('QLabel { background: #151b22; color: #dbe4ed; border-radius: 8px; }')
        layout.addWidget(self.preview, 1)
        controls = QtWidgets.QHBoxLayout()
        self.pause_button = QtWidgets.QPushButton('暂停取流')
        self.pause_button.clicked.connect(self.toggle_camera)
        controls.addWidget(self.pause_button)
        self.capture_button = QtWidgets.QPushButton('保存当前图像')
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(self.save_frame)
        controls.addWidget(self.capture_button)
        controls.addStretch()
        self.stats = QtWidgets.QLabel('尚无图像')
        controls.addWidget(self.stats)
        layout.addLayout(controls)
        arm_controls = QtWidgets.QHBoxLayout()
        arm_heading = QtWidgets.QLabel('机械臂跟随')
        arm_heading.setStyleSheet('font-size: 17px; font-weight: bold;')
        arm_controls.addWidget(arm_heading)
        arm_controls.addStretch()
        self.arm_start_button = QtWidgets.QPushButton('开始跟踪')
        self.arm_start_button.clicked.connect(self.start_follow)
        self.arm_start_button.setEnabled(False)
        self.arm_start_button.setMinimumSize(160,44)
        self.arm_start_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.arm_start_button.setStyleSheet('QPushButton {font-size: 16px; font-weight: bold; color: white; background: #19764b; border: none; border-radius: 7px; padding: 8px 20px;} QPushButton:disabled {background: #dce5df; color: #7e8d83;}')
        arm_controls.addWidget(self.arm_start_button)
        self.arm_stop_button = QtWidgets.QPushButton('停止跟踪')
        self.arm_stop_button.clicked.connect(self.stop_follow)
        self.arm_stop_button.setMinimumSize(150,44)
        self.arm_stop_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaStop))
        self.arm_stop_button.setStyleSheet('QPushButton {font-size: 16px; font-weight: bold; color: #b42318; background: #fdebe9; border: none; border-radius: 7px; padding: 8px 20px;} QPushButton:disabled {background: #eeeeee; color: #999999;}')
        arm_controls.addWidget(self.arm_stop_button)
        layout.addLayout(arm_controls)
        self.ros_label = QtWidgets.QLabel('ROS 2：初始化中')
        layout.addWidget(self.ros_label)
        self.detection_label = QtWidgets.QLabel('碗识别：等待推理节点')
        self.detection_label.setWordWrap(True)
        layout.addWidget(self.detection_label)
        self.follow_label = QtWidgets.QLabel('机械臂跟踪：未启动')
        self.follow_label.setWordWrap(True)
        layout.addWidget(self.follow_label)
        self.follow_help = QtWidgets.QLabel('将碗放入画面 → 点击开始；六关节 P500～P2500 内从当前位置启动，新姿态自动适配。')
        self.follow_help.setWordWrap(True)
        layout.addWidget(self.follow_help)
        self.isaac_label = QtWidgets.QLabel('Isaac：正在检查组件')
        self.isaac_label.setWordWrap(True)
        layout.addWidget(self.isaac_label)
        self.calibration_label = QtWidgets.QLabel('相机尚未标定 · 单目 RGB · 当前仅采图与识别')
        self.calibration_label.setWordWrap(True)
        layout.addWidget(self.calibration_label)
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(115)
        self.console.document().setMaximumBlockCount(100)
        layout.addWidget(self.console)
        self.log('将碗放入画面，点击绿色按钮开始跟踪；停止后可再次点击绿色按钮继续。')
        try:
            from vision.ros_bridge import ImageBridge
            self.bridge = ImageBridge()
            self.ros_label.setText('ROS 2 已启动 · /openmv/image_raw · 等待真实图像')
        except Exception as exc:
            self.ros_label.setText('ROS 2 未启动')
            self.log(f'ROS 2：{exc}')
        self.worker.start()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(50)
        if args.quit_after:
            QtCore.QTimer.singleShot(int(args.quit_after * 1000), self.close)

    def log(self, text):
        self.console.appendPlainText(f'{datetime.now():%H:%M:%S}  {text}')
        print(text, flush=True)

    def set_state(self, state):
        if state != self.camera_state:
            self.log(state)
        self.camera_state = state
        self.state_label.setText(state)

    def toggle_camera(self):
        self.worker.active = not self.worker.active
        self.pause_button.setText('暂停取流' if self.worker.active else '继续取流')
        if not self.worker.active:
            self.set_state('取流已暂停')
            self.rgb = None
            self.capture_button.setEnabled(False)
            self.preview.clear()
            self.preview.setText('取流已暂停')

    def stop_follow(self):
        if self.bridge:
            self.bridge.request_follow_stop()
        self.follow_message = '正在停止，等待控制节点确认……'
        if self.follow_process.state() != QtCore.QProcess.NotRunning:
            if self.follow_operation == 'start':
                self.follow_pending_stop = True
                self.follow_process.terminate()
        else:
            self.run_follow_control('stop')
        self.update_follow_controls()

    def start_follow(self):
        if self.follow_process.state() != QtCore.QProcess.NotRunning:
            return
        if not self.bridge or self.rgb is None or not self.worker.active:
            self.follow_message = '请先恢复取流，将目标碗放入画面。'
            self.update_follow_controls()
            return
        self.follow_message = '正在检查目标碗与机械臂，准备开启……'
        self.run_follow_control('start')

    def run_follow_control(self, action):
        self.follow_operation = action
        self.follow_error = ''
        self.follow_output = ''
        self.follow_progress = ''
        self.follow_decoder.reset()
        self.follow_process.start(sys.executable,[str(ROOT/'tools/follow_service.py'),action])
        self.update_follow_controls()

    def read_follow_output(self):
        if not self.follow_process.isOpen():
            return
        output = self.follow_decoder.decode(bytes(self.follow_process.readAllStandardOutput()))
        self.follow_output = (self.follow_output + output)[-16384:]
        for line in self.follow_output.splitlines():
            if line.startswith('正在自动适配当前位置：'):
                self.follow_progress = line
            if line.startswith('无法准备跟踪：'):
                self.follow_error = line
            elif line.startswith('开启失败：') and not self.follow_error:
                self.follow_error = line
        if output.strip():
            self.log(output.strip())

    def follow_control_error(self, error):
        if error == QtCore.QProcess.FailedToStart:
            self.follow_message = '无法启动跟踪控制程序，请查看下方日志。'
            self.log(self.follow_process.errorString())
            self.follow_operation = None
            self.update_follow_controls()

    def follow_control_finished(self, exit_code, _exit_status):
        self.read_follow_output()
        action = self.follow_operation
        self.follow_operation = None
        if self.follow_pending_stop:
            self.follow_pending_stop = False
            self.run_follow_control('stop')
            return
        if action == 'stop' and exit_code == 0:
            self.follow_message = '已停止；点击“继续跟踪”即可重新开启。'
            if self.bridge:
                self.bridge.follow_status = ({'state':'stopped','motion_enabled':False},time.monotonic())
        elif action == 'start' and exit_code == 0:
            self.follow_message = ''
        else:
            self.follow_message = (self.follow_error or '未能开启跟踪，请查看下方原因；处理后可再次点击开启。') if action == 'start' else '停止尚未确认，请查看下方日志。'
        self.update_follow_controls()
        if self.follow_pending_close:
            self.follow_pending_close = False
            self.close()

    def update_follow_controls(self):
        follow = self.bridge.follow_status if self.bridge else None
        fresh = bool(follow and time.monotonic()-follow[1]<=1.5)
        value = follow[0] if fresh else {}
        running = value.get('motion_enabled') is True
        busy = self.follow_process.state() != QtCore.QProcess.NotRunning
        camera_ready = bool(self.bridge and self.rgb is not None and self.worker.active
                            and self.frame_time is not None and time.monotonic()-self.frame_time<=1.5)
        self.arm_start_button.setEnabled(camera_ready and not running and not busy)
        paused=value.get('state') in ('stopped','operator_stopped') or '已停止' in self.follow_message
        self.arm_start_button.setText('正在开启……' if busy and self.follow_operation=='start' else '继续跟踪' if paused else '开始跟踪')
        self.arm_start_button.setToolTip('重新锁定当前碗并检查机械臂位置后开启跟随。' if camera_ready else '请先恢复取流，等待新画面。')
        can_stop=fresh and value.get('state') not in ('stopped','operator_stopped')
        self.arm_stop_button.setEnabled(self.follow_operation!='stop' and (running or busy or can_stop))
        if busy:
            text = '正在检查目标碗与机械臂，准备开启……' if self.follow_operation=='start' and not self.follow_pending_stop else '正在停止，等待确认……'
            if self.follow_operation=='start' and not self.follow_pending_stop and self.follow_progress:
                text = self.follow_progress
            self.follow_label.setText('机械臂跟踪：'+text)
        elif self.follow_message and not running:
            self.follow_label.setText('机械臂跟踪：'+self.follow_message)

    def tick(self):
        with self.worker.lock:
            latest = self.worker.latest
            self.worker.latest = None
        if latest is not None:
            self.rgb, self.frame_time = latest
            self.frame_count += 1
            self.set_state('实时画面已连接')
            height, width = self.rgb.shape[:2]
            self.capture_button.setEnabled(True)
            if self.bridge:
                self.bridge.publish(self.rgb)
            age = time.monotonic() - self.start_time
            self.stats.setText(f'{width} × {height} · {self.frame_count / max(age, 0.1):.1f} FPS')
            if self.args.first_frame and not Path(self.args.first_frame).exists():
                self.save_to(Path(self.args.first_frame))
        if self.frame_time is not None and time.monotonic() - self.frame_time > 2 and self.rgb is not None:
            self.rgb = None
            self.capture_button.setEnabled(False)
            self.preview.clear()
            self.preview.setText('画面已中断 · 等待重新连接')
            self.set_state('画面已中断')
        if self.bridge:
            try:
                self.bridge.tick(self.camera_state)
            except Exception as exc:
                self.log(f'ROS 2 发布已停止：{exc}')
                self.bridge.close()
                self.bridge = None
                self.ros_label.setText('ROS 2 发布异常，请查看下方日志')
                self.update_follow_controls()
                return
            self.ros_label.setText(f'ROS 2 已启动 · /openmv/image_raw · 已发布 {self.bridge.count} 帧')
            follow = self.bridge.follow_status
            if follow and time.monotonic()-follow[1] <= 1.5:
                value = follow[0]
                labels = {'acquiring': '确认目标中', 'centered': '已对准', 'move': '正在调整',
                          'rate_limited': '等待动作完成', 'stale': '画面过期，保持停止',
                          'no_target': '等待目标碗', 'low_confidence': '目标置信度不足',
                          'operator_stopped': '用户已停止', 'locked_target_lost': '已丢失锁定目标，停止',
                          'waiting_for_target': '等待重新识别目标碗',
                          'outside_locked_target': '候选与锁定目标不符，停止'}
                labels.update(stopped='已停止，可点击继续跟踪',
                              ambiguous='多个目标，等待锁定目标碗',limited='已到当前跟随范围边界')
                state = value.get('state', '等待')
                actuator = value.get('actuator_state', '')
                if actuator.startswith('fault:'):
                    summary = '已停止 · ' + actuator[len('fault:'):]
                else:
                    summary = labels.get(state, state)
                    if value.get('limit_handoff'):
                        summary += ' · 肩肘接力'
                mode = '小范围联调' if value.get('supervised_trial') else '预览'
                self.follow_label.setText(f'机械臂跟踪：{mode} · {summary}')
                self.calibration_label.setText('仅图像中心跟随 · 无三维抓取坐标 · 可点击上方按钮停止')
            else:
                self.follow_label.setText('机械臂跟踪：未开启 · 点击绿色按钮启动')
        display_rgb = self.rgb
        if self.bridge:
            overlay = self.bridge.fresh_overlay()
            status = self.bridge.detection_status
            if status and time.monotonic() - status[1] <= 1.5:
                result = status[0]
                if result.get('valid') and overlay is not None and self.rgb is not None:
                    display_rgb = overlay
                    targets = result.get('detections', [])
                    details = '未检测到目标碗'
                    if targets:
                        best = max(targets, key=lambda item: item['confidence'])
                        cx, cy = best['mask_centroid_px']
                        details = f"目标 {len(targets)} 个 · 置信度 {best['confidence']:.3f} · 像素中心 ({cx:.1f}, {cy:.1f})"
                    self.detection_label.setText(
                        f"碗识别：{details} · {'CUDA GPU · ' if result.get('inference_backend')=='pytorch_cuda' else ''}{result['fps']:.1f} FPS · {result['inference_ms']:.1f} ms")
                else:
                    self.detection_label.setText('碗识别：等待新画面（旧结果已失效）')
            else:
                self.detection_label.setText('碗识别：推理节点未连接或结果已过期')
        if display_rgb is not None:
            height, width = display_rgb.shape[:2]
            image = QtGui.QImage(display_rgb.data, width, height, width * 3, QtGui.QImage.Format_RGB888).copy()
            self.preview.setPixmap(QtGui.QPixmap.fromImage(image).scaled(
                self.preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        if self.bridge and self.bridge.isaac_last and time.monotonic() - self.bridge.isaac_last <= 2:
            fps = (self.bridge.isaac_frames - 1) / max(self.bridge.isaac_last - self.bridge.isaac_first, 0.1)
            self.isaac_label.setText(f'Isaac ROS：GPU BGR 实时输出 · {fps:.1f} FPS · {self.bridge.isaac_frames} 帧')
        else:
            self.isaac_label.setText('Isaac ROS：等待 GPU 图像输出')
        self.update_follow_controls()

    def save_to(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.rgb).save(path)
        self.log(f'图像已保存：{path}')

    def save_frame(self):
        if self.rgb is not None:
            self.save_to(ROOT / 'work/openmv/captures' / f'{datetime.now():%Y%m%d-%H%M%S-%f}.png')

    def closeEvent(self, event):
        if self.follow_process.state()!=QtCore.QProcess.NotRunning:
            self.follow_pending_close = True
            self.stop_follow()
            event.ignore()
            return
        self.timer.stop()
        self.worker.stop_event.set()
        if not self.worker.wait(6000):
            self.log('正在等待相机通信结束')
            self.timer.start(50)
            event.ignore()
            return
        if self.bridge:
            self.bridge.close()
        event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quit-after', type=float, default=0)
    parser.add_argument('--first-frame')
    parser.add_argument('--screenshot')
    parser.add_argument('--activate-only',action='store_true')
    args = parser.parse_args()
    application = QtWidgets.QApplication(['jibot-bowl-workbench'])
    application.setApplicationName('jibot-bowl-workbench')
    application.setDesktopFileName('jibot-bowl-follow')
    socket_name=f'jibot-bowl-workbench-{os.getuid()}'
    attempts=20 if args.activate_only else 1
    for _ in range(attempts):
        client=QtNetwork.QLocalSocket()
        client.connectToServer(socket_name)
        if client.waitForConnected(200):
            client.write(b'activate\n');client.waitForBytesWritten(200)
            client.disconnectFromServer()
            return 0
        if args.activate_only:time.sleep(.1)
    if args.activate_only:
        print('未能连接视觉工作台窗口',file=sys.stderr)
        return 1
    server=QtNetwork.QLocalServer()
    server.setSocketOptions(QtNetwork.QLocalServer.UserAccessOption)
    if not server.listen(socket_name):
        QtNetwork.QLocalServer.removeServer(socket_name)
        if not server.listen(socket_name):raise RuntimeError('无法建立工作台窗口连接')
    application.setStyle('Fusion')
    window = Window(args)
    def activate_window():
        while server.hasPendingConnections():
            connection=server.nextPendingConnection()
            window.showNormal();window.raise_();window.activateWindow()
            connection.disconnectFromServer();connection.deleteLater()
    server.newConnection.connect(activate_window)
    signal.signal(signal.SIGTERM, lambda *_: window.close())
    signal.signal(signal.SIGINT, lambda *_: window.close())
    window.show()
    if args.screenshot:
        def screenshot():
            Path(args.screenshot).parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(args.screenshot)
        QtCore.QTimer.singleShot(2000, screenshot)
    return application.exec_()


if __name__ == '__main__':
    raise SystemExit(main())
