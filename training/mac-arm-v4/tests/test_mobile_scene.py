"""Camera frames and sampled geometry contracts, on Linux or the receiving Mac."""
import copy
import math
import os
import platform
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


MAC=platform.system()=='Darwin' and platform.machine() in ('arm64','aarch64')


class MobileSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        from arm_training.model import ROOT,config
        from arm_training.mobile_scene import MobileScene,ArmScreen
        cls.arm_cfg=config();cls.mobile_cfg=yaml.safe_load((ROOT/'config/mobile.yaml').read_text())
        cls.scene=MobileScene(cls.arm_cfg,cls.mobile_cfg)
        cls.screen=ArmScreen(cls.arm_cfg,cls.mobile_cfg)

    @classmethod
    def tearDownClass(cls):cls.scene.close()

    def setUp(self):
        self.table={'center_xy_m':[1.,.02],'half_size_m':[.18,.18],
                    'top_height_m':.35,'yaw_rad':.2}
        self.scene.set_scenario(self.table,[1.,.02,.35],[.1,-.1,.3])

    def test_mocap_props_are_world_children_without_joints(self):
        for name in ('mount','mobile_chassis','platform_body'):
            body=self.scene.model.body(name).id
            self.assertEqual(int(self.scene.model.body_parentid[body]),0)
            self.assertEqual(int(self.scene.model.body_jntnum[body]),0)
            self.assertGreaterEqual(int(self.scene.model.body_mocapid[body]),0)
        self.assertEqual(int(self.scene.model.cam_bodyid[self.scene.model.camera('wrist').id]),self.scene.model.body('wrist_pitch').id)

    def test_optical_look_at_is_rigid_and_points_forward(self):
        from arm_training.mobile_scene import optical_look_at
        eye=np.array([.1,-.2,.8]);target=np.array([.7,.2,.4])
        transform=optical_look_at(eye,target)
        np.testing.assert_allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(transform[:3,:3]),1.)
        optical=np.linalg.inv(transform)@np.r_[target,1.]
        np.testing.assert_allclose(optical[:2],[0,0],atol=1e-12);self.assertGreater(optical[2],0)

    def test_camera_intrinsics_match_render_size_and_fovy(self):
        for name in ('main_rgbd','wrist'):
            K,T=self.scene.camera(name);cid=self.scene.model.camera(name).id
            expected=self.scene.height/(2*math.tan(math.radians(float(self.scene.model.cam_fovy[cid]))/2))
            self.assertAlmostEqual(K[0,0],expected);self.assertAlmostEqual(K[1,1],expected)
            np.testing.assert_allclose(K[:2,2],[(self.scene.width-1)/2,(self.scene.height-1)/2])
            np.testing.assert_allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-10)
            gl_rotation=self.scene.data.cam_xmat[cid].reshape(3,3)
            np.testing.assert_allclose(T[:3,:3],gl_rotation@np.diag([1,-1,-1]),atol=1e-12)

    def test_wrist_camera_does_not_rotate_with_servo_004(self):
        q=self.scene.observation_q.copy();before=self.scene.wrist_camera_pose_in_model(q)
        q[-1]+=.35;after=self.scene.wrist_camera_pose_in_model(q)
        np.testing.assert_allclose(after,before,atol=1e-10)

    def test_wrist_query_restores_scene_and_uses_model_ground_origin(self):
        self.scene.data.qpos[self.scene.ids[5:]]=.012
        saved=self.scene.data.qpos.copy();_,world_before=self.scene.camera('wrist')
        local=self.scene.wrist_camera_pose_in_model(self.scene.observation_q)
        np.testing.assert_array_equal(self.scene.data.qpos,saved)
        np.testing.assert_allclose(self.scene.arm_pose4@local,world_before,atol=1e-9)
        self.assertEqual(self.scene.arm_pose4[2,3],0.)
        _,world_after=self.scene.camera('wrist');np.testing.assert_allclose(world_after,world_before,atol=1e-9)

    def test_wrist_query_restores_even_when_camera_extraction_fails(self):
        saved=self.scene.data.qpos.copy();q=self.scene.observation_q.copy();q[0]+=.1
        with patch.object(self.scene,'camera',side_effect=ValueError('forced extraction failure')):
            with self.assertRaises(ValueError):self.scene.wrist_camera_pose_in_model(q)
        np.testing.assert_array_equal(self.scene.data.qpos,saved)

    def test_model_relative_camera_pose_does_not_depend_on_parking(self):
        q=self.scene.observation_q.copy();first=self.scene.wrist_camera_pose_in_model(q)
        self.scene.set_vehicle([-.3,.25,-.8]);second=self.scene.wrist_camera_pose_in_model(q)
        np.testing.assert_allclose(second,first,atol=1e-9)

    def test_grasp_state_world_transform_preserves_height(self):
        from scipy.spatial.transform import Rotation
        local=np.array([.19,.015,.395]);source=self.screen.data
        qpos=source.qpos.copy();qpos[self.screen.free:self.screen.free+3]=local
        local_rotation=Rotation.from_euler('x',.1)
        x,y,z,w=local_rotation.as_quat();qpos[self.screen.free+3:self.screen.free+7]=[w,x,y,z]
        qpos[self.screen.qids]=self.scene.observation_q
        for jid in range(self.screen.model.njnt):
            if self.screen.model.jnt_type[jid] in (2,3) and jid!=self.screen.model.joint('bowl_free').id:
                if (self.screen.model.joint(jid).name or '').startswith(('closure_','hinge_')):qpos[int(self.screen.model.jnt_qposadr[jid])]=.002
        env=SimpleNamespace(data=SimpleNamespace(qpos=qpos),model=self.screen.model,free_q=self.screen.free)
        self.scene.apply_grasp_state(env)
        actual=self.scene.data.qpos[self.scene.free:self.scene.free+7]
        np.testing.assert_allclose(actual[:3],(self.scene.arm_pose4@np.r_[local,1.])[:3],atol=1e-10)
        self.assertAlmostEqual(actual[2],local[2])
        actual_rotation=Rotation.from_quat([actual[4],actual[5],actual[6],actual[3]])
        expected=Rotation.from_euler('z',self.scene.arm_pose[2])*local_rotation
        np.testing.assert_allclose(actual_rotation.as_matrix(),expected.as_matrix(),atol=1e-10)
        np.testing.assert_allclose(self.scene.data.qpos[self.scene.ids[5:]],.002)

    def test_platform_sizes_bounds_and_pose_refresh_for_each_table(self):
        from arm_training.mobile_scene import _set_platform
        for half,height in (([.25,.12],.4),([.08,.15],.31)):
            table={**self.table,'half_size_m':half,'top_height_m':height}
            for owner in (self.scene,self.screen):
                _set_platform(owner.model,owner.data,owner.platform,table)
                gid=owner.model.geom('platform').id;size=np.array([*half,height/2])
                np.testing.assert_allclose(owner.model.geom_size[gid],size)
                np.testing.assert_allclose(owner.model.geom_aabb[gid],[0,0,0,*size])
                self.assertAlmostEqual(float(owner.model.geom_rbound[gid]),np.linalg.norm(size))
                np.testing.assert_allclose(owner.data.mocap_pos[owner.platform],[*table['center_xy_m'],height/2])

    def test_arm_screen_refreshes_dimensions_without_stale_midphase(self):
        import mujoco
        table={**self.table,'center_xy_m':[3.,2.],'half_size_m':[.1,.22],'top_height_m':.38}
        self.screen.geometry_check(self.scene.observation_q.copy(),np.array([.2,0,.4]),'home',table)
        gid=self.screen.model.geom('platform').id
        np.testing.assert_allclose(self.screen.model.geom_size[gid],[.1,.22,.19])
        self.assertTrue(int(self.screen.model.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE))
        self.assertFalse(int(self.screen.model.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_CONTACT))

    def test_arm_screen_lower_clearance_uses_physical_hard_finger_meshes(self):
        import mujoco
        self.assertTrue(self.screen.hard_finger_geoms)
        for gid in self.screen.hard_finger_geoms:
            name=self.screen.model.geom(gid).name
            self.assertTrue(name.endswith('_hard'));self.assertFalse(name.endswith('_visual'))
            self.assertEqual(int(self.screen.model.geom_type[gid]),int(mujoco.mjtGeom.mjGEOM_MESH))
            self.assertIn(int(self.screen.model.geom_bodyid[gid]),self.screen.finger_bodies)

    def test_snapshot_round_trip_restores_visual_state(self):
        row=self.scene.snapshot(2.,'test');saved=self.scene.data.qpos.copy()
        self.scene.set_vehicle([.5,.5,-.5]);self.scene.data.qpos[self.scene.ids[:5]]=0.
        self.scene.apply_snapshot(row)
        np.testing.assert_allclose(self.scene.data.qpos,saved)
        np.testing.assert_allclose(self.scene.vehicle_pose,row['vehicle_xy_yaw'])

    def test_invalid_camera_and_planar_pose_input_is_rejected(self):
        from arm_training.mobile_scene import optical_look_at,pose4
        for eye,target in (([0,0,0],[0,0,0]),([0,0,0],[0,0,1]),([math.nan,0,0],[1,0,0])):
            with self.assertRaises(ValueError):optical_look_at(eye,target)
        with self.assertRaises(ValueError):pose4([0,0,float('inf')])


class DeviceContractTests(unittest.TestCase):
    def test_cpu_device_is_explicit_and_recorded(self):
        from arm_training.device import resolve_device,device_record
        selected=resolve_device('cpu');record=device_record('cpu',selected)
        self.assertEqual(selected.type,'cpu');self.assertEqual(record['actual_policy_device'],'cpu')
        self.assertEqual(record['physics_device'],'cpu')

    def test_auto_and_silent_mps_fallback_are_rejected(self):
        from arm_training.device import resolve_device
        with self.assertRaises(ValueError):resolve_device('auto')
        with patch.dict(os.environ,{'PYTORCH_ENABLE_MPS_FALLBACK':'1'}):
            with self.assertRaises(RuntimeError):resolve_device('mps')

    def test_unavailable_explicit_mps_raises(self):
        from arm_training.device import resolve_device
        with patch.dict(os.environ,{'PYTORCH_ENABLE_MPS_FALLBACK':'0'}):
            with patch('torch.backends.mps.is_available',return_value=False):
                with self.assertRaises(RuntimeError):resolve_device('mps')


if __name__=='__main__':unittest.main()
