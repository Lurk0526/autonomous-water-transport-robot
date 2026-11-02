"""ROS activation must also work when launched from the strict GUI wrapper."""
from pathlib import Path
import os
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(Path('/opt/ros/jazzy/setup.bash').exists() and (ROOT/'.venv/bin/python').exists(),
                     'Jetson ROS environment integration check')
class ActivationTests(unittest.TestCase):
    def test_default_dds_uses_local_interface_and_preserves_explicit_config(self):
        for custom in ('', '<CycloneDDS><Domain Id="99"/></CycloneDDS>'):
            with self.subTest(custom=bool(custom)):
                env=dict(os.environ,RMW_IMPLEMENTATION='rmw_cyclonedds_cpp',CYCLONEDDS_URI=custom)
                result=subprocess.run(['bash','-euc','source "$1"; printf "%s" "$CYCLONEDDS_URI"',
                    'activation-test',str(ROOT/'activate_project.sh')],env=env,
                    capture_output=True,text=True,timeout=20)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(result.stdout,custom or f'file://{ROOT}/config/cyclonedds-local.xml')

    def test_strict_caller_keeps_nounset_and_can_import_ros(self):
        result=subprocess.run(['bash','-euc','source "$1"; [[ $- == *u* ]]; python -c "import rclpy"',
                               'activation-test',str(ROOT/'activate_project.sh')],
                              capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_normal_caller_does_not_gain_nounset(self):
        result=subprocess.run(['bash','-ec','source "$1"; [[ $- != *u* ]]',
                               'activation-test',str(ROOT/'activate_project.sh')],
                              capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
