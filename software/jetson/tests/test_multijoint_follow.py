"""Multi-joint image control with redundant and non-pointing joints."""
import time
import unittest
import numpy as np
from unittest.mock import MagicMock
from vision.bowl_follow import FollowPolicy, FollowSettings
from vision.arm_serial import ArmSerial
from vision.follow_calibration import estimate_jacobian
from tests.test_bowl_follow import detection


class MultiJointTests(unittest.TestCase):
    def test_redundant_axes_reduce_image_error_and_leave_gripper_held(self):
        matrix=np.array([[1.2,.03,-.02,0,0],[-.08,.9,-.85,.87,0]])
        policy=FollowPolicy(FollowSettings(jacobian=matrix,joint_axes=(0,1,2,3,4),
            damped_solver=True,max_step=65,deadband_compensation=(0,)*5,alpha=.65))
        center=np.array([230.,170.]);start=time.time();used=set()
        for i in range(250):
            stamp=start+i*.05
            plan=policy.update(detection(stamp,center),stamp+.01,i*.05)
            delta=np.array([plan['delta_p'].get(j,0) for j in range(5)])
            center+=matrix@delta
            used.update(plan['delta_p'])
            self.assertNotIn(5,plan['delta_p'])
            self.assertNotIn(4,plan['delta_p'])
        self.assertLess(abs(center[0]-160),12)
        self.assertLess(abs(center[1]-120),10)
        self.assertTrue({0,1,2,3} <= used)

    def test_five_axis_calibration_requires_both_directions(self):
        matrix=np.array([[1.2,.03,-.02,0,0],[-.08,.9,-.85,.87,0]])
        rows=[]
        for j in range(5):
            for dp in (30,-30):
                rows.append(dict(axis=j,actual_delta_p=dp,before={'center':[160,120]},
                    after={'center':(np.array([160,120])+matrix[:,j]*dp).tolist()}))
        measured,_=estimate_jacobian(rows,(0,1,2,3,4))
        np.testing.assert_allclose(measured,matrix,atol=1e-10)
        with self.assertRaises(ValueError):estimate_jacobian(rows[:-1],(0,1,2,3,4))

    def test_six_channel_transport_requires_explicit_axis_selection(self):
        arm=ArmSerial.__new__(ArmSerial);arm._send=MagicMock()
        with self.assertRaises(ValueError):arm.move({5:1510},400)
        arm.move({i:1500 for i in range(6)},400,allowed_axes=range(6))
        self.assertIn('#005P1500T0400!',arm._send.call_args.args[0])
