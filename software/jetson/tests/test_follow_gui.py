"""GUI start/stop lifecycle with mocked camera, ROS, and service processes."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock,patch
import numpy as np
from PyQt5 import QtCore,QtWidgets
from tools.openmv_gui import Window


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.bridge=MagicMock()
        self.bridge.follow_status=({'state':'operator_stopped','motion_enabled':False},time.monotonic())
        self.patches=[patch('tools.openmv_gui.CameraWorker.start'),
                      patch('vision.ros_bridge.ImageBridge',return_value=self.bridge)]
        for item in self.patches:item.start()
        self.window=Window(SimpleNamespace(quit_after=0,first_frame=None))
        self.window.timer.stop()
        self.window.rgb=np.zeros((240,320,3),np.uint8)
        self.window.frame_time=time.monotonic()

    def tearDown(self):
        self.window.follow_process=QtCore.QProcess(self.window)
        self.window.close()
        for item in reversed(self.patches):item.stop()

    def test_stopped_state_has_working_start_button(self):
        self.window.update_follow_controls()
        self.assertTrue(self.window.arm_start_button.isEnabled())
        self.assertFalse(self.window.arm_stop_button.isEnabled())
        with patch.object(self.window,'run_follow_control') as start:
            self.window.arm_start_button.click();start.assert_called_once_with('start')

    def test_camera_absent_does_not_launch_hardware_control(self):
        self.window.rgb=None
        with patch.object(self.window,'run_follow_control') as start:
            self.window.start_follow();start.assert_not_called()
        self.assertFalse(self.window.arm_start_button.isEnabled())

    def test_running_state_disables_duplicate_start(self):
        self.bridge.follow_status=({'state':'move','motion_enabled':True},time.monotonic())
        self.window.update_follow_controls()
        self.assertFalse(self.window.arm_start_button.isEnabled())
        self.assertTrue(self.window.arm_stop_button.isEnabled())

    def test_stop_cancels_start_then_confirms_service_stop(self):
        process=MagicMock();process.state.return_value=QtCore.QProcess.Running
        process.readAllStandardOutput.return_value=QtCore.QByteArray()
        self.window.follow_process=process;self.window.follow_operation='start'
        with patch.object(self.window,'run_follow_control') as run:
            self.window.stop_follow()
            self.bridge.request_follow_stop.assert_called_once()
            process.terminate.assert_called_once();run.assert_not_called()
            self.window.follow_control_finished(2,QtCore.QProcess.NormalExit)
            run.assert_called_once_with('stop')

    def test_confirmed_stop_reenables_start(self):
        self.window.follow_operation='stop'
        self.window.follow_control_finished(0,QtCore.QProcess.NormalExit)
        self.assertTrue(self.window.arm_start_button.isEnabled())
        self.assertIn('重新开启',self.window.follow_label.text())

    def test_failed_start_keeps_specific_bus_error_in_visible_label(self):
        process=MagicMock();process.state.return_value=QtCore.QProcess.NotRunning
        process.isOpen.return_value=True
        reason='无法准备跟踪：关节 001、002 无位置反馈；请检查舵机独立供电、总线接线及舵机 ID'
        data=(reason+'\n').encode()
        chunks=[data[:16],data[16:], '开启失败：未能恢复跟踪；请查看上方具体原因。\n'.encode()]
        process.readAllStandardOutput.side_effect=[QtCore.QByteArray(c) for c in chunks]
        self.window.follow_process=process;self.window.follow_operation='start'
        self.window.read_follow_output()
        self.window.read_follow_output()
        self.window.follow_control_finished(1,QtCore.QProcess.NormalExit)
        self.assertIn(reason,self.window.follow_label.text())
        self.assertTrue(self.window.arm_start_button.isEnabled())
