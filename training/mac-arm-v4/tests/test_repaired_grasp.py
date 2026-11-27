"""Contracts behind the reproduced failures; physical success is evaluated separately."""
from collections import deque
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from arm_training.compliant_control import SynchronousGraspController
from arm_training.repaired_env import RepairedBowlEnv
from arm_training.model import config,make_model,JOINTS
from arm_training.linked_gripper import loop_residuals
from arm_training.precision_ik import PrecisionIK
from arm_training.fingerprints import model_fingerprint


class ContactControllerTests(unittest.TestCase):
    def sample(self,c,foam,**kwargs):
        return c.update('close',.01,.005,.008,foam,[0.]*4,[(0,3),(1,2)],**kwargs)

    def test_single_contact_can_establish_opposition_instead_of_freezing(self):
        c=SynchronousGraspController();out=self.sample(c,[1,0,0,0])
        self.assertGreater(out['command_m'],.005);self.assertFalse(out['confirmed'])
        for _ in range(20):out=self.sample(c,[1,0,0,0])
        self.assertLessEqual(out['command_m'],.0062);self.assertIsNone(out['failure'])

    def test_weak_bilateral_touch_is_not_a_loaded_grasp(self):
        c=SynchronousGraspController()
        for _ in range(40):out=self.sample(c,[.12,.12,.12,.12])
        self.assertFalse(out['confirmed']);self.assertGreater(out['command_m'],.005)

    def test_real_load_has_to_persist_and_does_not_stop_at_20n_rack(self):
        c=SynchronousGraspController()
        for _ in range(9):out=self.sample(c,[4.1,0,0,4.1],actuator_force_n=28.)
        self.assertFalse(out['confirmed'])
        out=self.sample(c,[4.1,0,0,4.1],actuator_force_n=28.)
        self.assertTrue(out['confirmed']);before=out['command_m']
        out=self.sample(c,[4.1,0,0,4.1],actuator_force_n=28.)
        self.assertGreater(out['command_m'],before)

    def test_raw_overload_is_not_hidden_by_filtering(self):
        c=SynchronousGraspController()
        for _ in range(20):self.sample(c,[4.1,0,0,4.1])
        out=self.sample(c,[8.1,0,0,1.])
        self.assertEqual(out['failure'],'foam_contact_force_limit')
        self.assertFalse(out['independent_finger_commands'])

    def test_raw_lost_contact_is_cleared_from_regulation_estimate(self):
        c=SynchronousGraspController()
        for _ in range(20):self.sample(c,[4.1,0,0,4.1])
        self.sample(c,[0,0,0,0])
        np.testing.assert_array_equal(c.filtered_foam,np.zeros(4))

    def test_table_touch_cancels_both_queues_and_gripper_preload(self):
        e=RepairedBowlEnv.__new__(RepairedBowlEnv);e.cfg={'task':{'tilt_limit_rad':.2},'placement':{}}
        e.phase_index=5;e.support_lock_q=None;e.qids=np.arange(5);e.closure_qids=np.array([5]);e.site=0
        e.data=SimpleNamespace(qpos=np.r_[np.arange(5)*.1,.006],ctrl=np.r_[np.ones(5),.009],
            site_xpos=np.array([[.2,0,.40]]),time=10.)
        e.targets=[np.zeros(3) for _ in range(10)];e.commands=deque([(9.,np.ones(5))]);e.residual=np.ones(3)*.003
        e.servo_integral=np.ones(5)*.04;e.contact_control=SimpleNamespace(command=.009);e.guard_events=[]
        e._stop_on_support({'support_n':.6,'clearance_m':0.,'tilt_rad':.001})
        self.assertIsNotNone(e.support_lock_q);self.assertEqual(len(e.commands),1)
        self.assertAlmostEqual(e.data.ctrl[5],.006);self.assertAlmostEqual(e.contact_control.command,.006)
        np.testing.assert_array_equal(e.servo_integral,np.zeros(5));np.testing.assert_array_equal(e.residual,np.zeros(3))
        self.assertIn('first_real_support_touch',e.guard_events[0]['stop_kind'])

    def test_freezing_external_vision_requires_a_real_measurement(self):
        # Constructor-free check is not useful here; run actual reset, never physics.
        e=RepairedBowlEnv()
        with self.assertRaises(ValueError):e.reset(seed=1,options={'observation_source':'frozen_wrist_estimate'})

    def test_initial_scenario_does_not_change_when_target_is_refined(self):
        e=RepairedBowlEnv();_,initial=e.reset(seed=6)
        e.estimate+=np.array([.001,-.001,.0005])
        self.assertEqual(initial['scenario'],e._info(False)['scenario'])

    def test_search_to_home_uses_valid_branch_without_teleporting_commands(self):
        c=config();c['fixture']['bowl_world_xy_m']=[.24323056,.00533648]
        e=RepairedBowlEnv(c);e.reset(seed=123002,options={'parking':[0,0,0],
            'target_estimate_base_m':[.24323056,.00533648,.35053483],
            'observation_source':'frozen_wrist_estimate'})
        search=np.array([.17858831,.06444608,.83359561,-.91389327,-1.45963026])
        e.data.qpos[e.qids]=search;e.command=search.copy();e.desired_q=search.copy();e.data.ctrl[:5]=search
        e.commands=deque([(0.,search.copy())]);mujoco.mj_forward(e.model,e.data)
        _,_,done,_,info=e.step(np.zeros(3))
        self.assertFalse(done,info['failure_reason'])
        self.assertLessEqual(np.max(np.abs(e.command-search)),c['arm']['max_joint_velocity_rad_s']*e.dt+1e-10)


class LinkedGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=config();cls.model,_,cls.cad=make_model(cls.cfg,visual=False)

    def test_one_rack_and_physical_loops_do_not_attach_bowl(self):
        m=self.model;self.assertEqual(m.nu,6);self.assertEqual(m.neq,12)
        names=[m.joint(i).name for i in range(m.njnt)]
        self.assertEqual([n for n in names if n.startswith('closure_')],['closure_0'])
        self.assertEqual(len([n for n in names if n.startswith('hinge_')]),32)
        bowl=m.body('bowl').id
        self.assertNotIn(bowl,m.eq_obj1id.tolist());self.assertNotIn(bowl,m.eq_obj2id.tolist())
        self.assertEqual(m.body_mass[bowl],.65)

    def test_reference_pin_loops_are_closed(self):
        d=mujoco.MjData(self.model);mujoco.mj_forward(self.model,d)
        self.assertLess(max(r['separation_m'] for r in loop_residuals(self.model,d)),1e-8)

    def test_almost_touching_foot_does_not_report_deep_table_penetration(self):
        m=self.model;d=mujoco.MjData(m)
        q=m.joint('bowl_free').qposadr[0];foot=m.geom('bowl_foot').id;table=m.geom('platform').id
        height=self.cfg['fixture']['top_height_m']
        for tilt in (0.,1e-6,1e-5,2e-5):
            for gap in (-1e-7,0.,2.5e-8,5e-8,1e-7):
                d.qpos[q:q+3]=[.2,0,height+gap]
                d.qpos[q+3:q+7]=[np.cos(tilt/2),np.sin(tilt/2),0,0]
                mujoco.mj_forward(m,d)
                contacts=[c for c in d.contact if {c.geom1,c.geom2}=={foot,table}]
                self.assertTrue(all(c.dist>-.00001 for c in contacts),(tilt,gap))

    def test_hard_collisions_remain_and_bowl_radius_is_unchanged(self):
        m=self.model;hard=[i for i in range(m.ngeom) if (m.geom(i).name or '').endswith('_hard')]
        self.assertGreater(len(hard),100)
        self.assertTrue(all(m.geom_contype[i] and m.geom_conaffinity[i] for i in hard))
        self.assertEqual(self.cfg['bowl']['rim_radius_m'],.078)
        error=.078*(1-np.cos(np.pi/self.cfg['bowl']['sectors']))
        self.assertLess(error,.0001)

    def test_fixed_orientation_vertical_path_does_not_spin_unobservable_axis(self):
        ik=PrecisionIK(self.model);q,ok=ik.solve([.2,0,.46]);self.assertTrue(ok)
        ik.data.qpos[ik.qids]=q;mujoco.mj_forward(self.model,ik.data)
        ik.orientation_target=ik.data.site_xmat[ik.site].reshape(3,3).copy();start=q.copy()
        for z in np.linspace(.46,.3985,30):
            next_q,ok=ik.solve([.2,0,z],seed=q);self.assertTrue(ok)
            self.assertLess(np.max(np.abs(next_q[[0,4]]-q[[0,4]])),.01)
            q=next_q
        self.assertLess(np.max(np.abs(q[[0,4]]-start[[0,4]])),.03)

    def test_ik_minimal_pipeline_matches_full_pose_and_jacobian(self):
        ik=PrecisionIK(self.model);full=mujoco.MjData(self.model);rng=np.random.default_rng(707)
        for _ in range(32):
            full.qpos[ik.qids]=rng.uniform(-1,1,5);ik.data.qpos[:]=full.qpos
            mujoco.mj_forward(self.model,full);ik._forward()
            np.testing.assert_array_equal(ik.data.site_xpos[ik.site],full.site_xpos[ik.site])
            np.testing.assert_array_equal(ik.data.site_xmat[ik.site],full.site_xmat[ik.site])
            a,b,c,d=[np.zeros((3,self.model.nv)) for _ in range(4)]
            mujoco.mj_jacSite(self.model,full,a,b,ik.site);mujoco.mj_jacSite(self.model,ik.data,c,d,ik.site)
            np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)

    def test_collision_decomposition_changes_model_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'assets/cad').mkdir(parents=True)
            (root/'assets/candidate-model.json').write_text(json.dumps({'parts':[]}))
            (root/'assets/cad/cad-manifest.json').write_text('{}')
            path=root/'assets/cad/collision-parts.json';path.write_text('{"v":1}');first=model_fingerprint(root)
            path.write_text('{"v":2}');self.assertNotEqual(first,model_fingerprint(root))


if __name__=='__main__':unittest.main()
