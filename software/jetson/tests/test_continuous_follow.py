"""A moving trajectory can only be extended after verified measured progress."""
import time
import unittest
from vision.bowl_follow import BoundedArm
from tests.test_bowl_follow import FakeArm,calibrated_fixture


class ContinuousTests(unittest.TestCase):
    def test_coordinated_mode_can_decelerate_before_the_old_endpoint(self):
        arm,fake,now=self.setup_arm();arm.config['coordinated_retarget']=True
        fake.positions[0]=1520
        self.assertEqual(arm.tick(self.plan(30),now+.45),'command_retargeted')
        self.assertEqual(fake.commands[-1][0],{0:1550})
        self.assertLess(1550,1560)

    def setup_arm(self):
        config=calibrated_fixture()
        config.update(continuous_follow=True,max_step_p=100,retarget_period_s=.4,
                      duration_ms=500,settle_seconds=.2,session_travel_p=12000,
                      position_windows_p={'0':200,'3':200})
        for axis in ('0','3'):config['axes'][axis].update(min_p=1300,max_p=1700)
        fake=FakeArm();now=time.monotonic();arm=BoundedArm(fake,config,now)
        self.assertEqual(arm.tick(self.plan(60),now),'command_sent')
        return arm,fake,now

    def plan(self,delta,center=(240,120),stamp=None):
        return dict(state='move',delta_p={0:delta},source_stamp_unix=time.time() if stamp is None else stamp,
                    center_px=center,target_px=(160,120))

    def test_progress_extends_target_before_previous_motion_finishes(self):
        arm,fake,now=self.setup_arm();fake.positions[0]=1520
        self.assertEqual(arm.tick(self.plan(60),now+.45),'command_retargeted')
        self.assertEqual(fake.commands[-1][0],{0:1580})
        self.assertEqual(arm.pending_origin,{0:1520});self.assertEqual(arm.retarget_count,1)
        self.assertEqual(arm.travel[0],80)
        self.assertIsNotNone(arm.pending)

    def test_stalled_joint_or_near_center_is_not_retargeted(self):
        for case in ('stalled','near','stale'):
            arm,fake,now=self.setup_arm()
            if case!='stalled':fake.positions[0]=1520
            plan=self.plan(60,center=(170,120) if case=='near' else (240,120),
                           stamp=time.time()-1 if case=='stale' else None)
            self.assertEqual(arm.tick(plan,now+.45),'moving')
            self.assertEqual(len(fake.commands),1)

    def test_reversal_interrupts_old_trajectory(self):
        arm,fake,now=self.setup_arm();fake.positions[0]=1520
        self.assertEqual(arm.tick(self.plan(-40,center=(80,120)),now+.45),'direction_changed')
        self.assertEqual(fake.stops,1);self.assertIsNone(arm.pending)

    def test_center_or_lost_target_interrupts_extended_trajectory(self):
        for state in ('centered','no_target'):
            arm,fake,now=self.setup_arm();fake.positions[0]=1520
            arm.tick(self.plan(60),now+.45)
            self.assertEqual(arm.tick(dict(state=state,delta_p={}),now+.5),state)
            self.assertEqual(fake.stops,1);self.assertIsNone(arm.pending)
            self.assertEqual(arm.travel[0],20)

    def test_retarget_is_clamped_to_absolute_limit(self):
        arm,fake,now=self.setup_arm();fake.positions[0]=1545
        arm.config['axes']['0']['max_p']=1580
        result=arm.tick(self.plan(100),now+.45)
        self.assertEqual(result,'command_retargeted')
        self.assertEqual(fake.commands[-1][0],{0:1580})

    def test_unexplained_large_progress_never_retargets(self):
        arm,fake,now=self.setup_arm();fake.positions[0]=1620
        result=arm.tick(self.plan(60),now+.45)
        self.assertIn('unexpected_retarget_motion',result)
        self.assertEqual(len(fake.commands),1)
        self.assertEqual(fake.stops,1)
