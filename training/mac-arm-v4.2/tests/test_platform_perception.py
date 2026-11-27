"""Geometry-only platform detection and conservative hidden-edge bounds."""
import copy
import unittest
import numpy as np

from arm_training.platform_perception import detect_platform,platform_search_prior,fuse_static_platform
from arm_training.mobile_planner import MobileCoordinator,default_mobile_config


def scene_depth(occluded=False):
    h=w=240;K=np.array([[240.,0.,119.5],[0.,240.,119.5],[0.,0.,1.]])
    T=np.eye(4);T[:3,:3]=np.diag([1.,-1.,-1.]);T[:3,3]=[0.,0.,1.]
    v,u=np.mgrid[:h,:w];x=(u-K[0,2])*.65/K[0,0];y=-(v-K[1,2])*.65/K[1,1]
    support=(np.abs(x)<=.18)&(np.abs(y)<=.18)
    depth=np.ones((h,w));depth[support]=.65
    if occluded:depth[support&(x<-.08)]=.3
    return depth,K,T


class PlatformPerceptionTests(unittest.TestCase):
    def test_depth_detects_plane_without_scene_metadata(self):
        depth,K,T=scene_depth();result=detect_platform(depth,K,T,1.)
        self.assertTrue(result['valid'],result)
        self.assertEqual(result['kind'],'platform')
        self.assertFalse(result['uses_simulator_truth'])
        self.assertFalse(result['learned_platform_detector'])
        self.assertAlmostEqual(result['top_height_m'],.35,places=6)
        self.assertLess(np.linalg.norm(result['center_xy_m']),.005)

    def test_hidden_edges_are_bounded_without_claiming_full_footprint_observed(self):
        depth,K,T=scene_depth(occluded=True);result=detect_platform(depth,K,T,1.)
        self.assertTrue(result['valid'],result)
        table=result['conservative_table'];center=np.array(table['center_xy_m']);half=np.array(table['half_size_m'])
        self.assertTrue(np.all(center-half<=[-.18,-.18]))
        self.assertTrue(np.all(center+half>=[.18,.18]))
        self.assertFalse(table['full_footprint_measured'])
        self.assertFalse(result['quality']['complete_platform_edges_verified'])
        self.assertTrue(result['quality']['hidden_edges_bounded_by_prior'])

    def test_platform_search_prior_is_never_a_detected_bowl(self):
        result=detect_platform(*scene_depth(),1.)
        prior=platform_search_prior(result)
        self.assertFalse(prior['is_bowl_detection'])
        self.assertEqual(prior['kind'],'platform_center_search_prior')
        self.assertEqual(prior['source'],'platform_search_prior')
        self.assertEqual(np.array(prior['covariance_m2']).shape,(3,3))

    def test_unknown_orientation_uses_larger_diagonal_bound(self):
        depth,K,T=scene_depth()
        aligned=detect_platform(depth,K,T,1.)
        unknown=detect_platform(depth,K,T,1.,config={'orientation_prior_yaw_rad':None})
        self.assertTrue(unknown['valid'],unknown)
        self.assertTrue(np.all(np.array(unknown['conservative_table']['half_size_m'])>aligned['conservative_table']['half_size_m']))

    def test_invalid_or_empty_depth_does_not_return_a_known_table(self):
        depth,K,T=scene_depth()
        for invalid in [np.full_like(depth,np.nan),np.zeros_like(depth)]:
            result=detect_platform(invalid,K,T,1.)
            self.assertFalse(result['valid'])
            self.assertIsNone(result['conservative_table'])

    def test_old_frames_wrong_units_and_bad_transform_fail_closed(self):
        depth,K,T=scene_depth()
        self.assertFalse(detect_platform(depth,K,T,1.,now_s=2.)['valid'])
        self.assertFalse(detect_platform(depth*1000,K,T,1.)['valid'])
        bad=T.copy();bad[0,0]=2.
        self.assertFalse(detect_platform(depth,K,bad,1.)['valid'])

    def test_platform_prior_is_typed_and_does_not_borrow_bowl_identity(self):
        platform=detect_platform(*scene_depth(),1.);prior=platform_search_prior(platform,target_id='station-1')
        cfg=default_mobile_config();cfg['coordinator'].update(main_target_source='platform_search_prior',main_max_sigma_m=.08)
        flow=MobileCoordinator(cfg)
        def odom(stamp):return {'valid':True,'frame':'odom','stamp_s':stamp,'xy_yaw':[0.,0.,0.],'speed_m_s':0.,'yaw_rate_rad_s':0.}
        self.assertTrue(flow.update(1.,odom(1.),main_target=prior)['plan_requested'])
        flow.reset();fake=copy.deepcopy(prior);fake['is_bowl_detection']=True
        self.assertFalse(flow.update(1.,odom(1.),main_target=fake)['plan_requested'])
        flow.reset();flow.select_plan({'valid':True,'target_id':'station-1','target_kind':'platform_center_search_prior',
            'selected':{'parking_xy_yaw':[0.,0.,0.]}})
        for tick in range(7):
            stamp=1.1+tick*.1;fresh=copy.deepcopy(prior);fresh.update(stamp_s=stamp,valid_until_s=stamp+.5)
            result=flow.update(stamp,odom(stamp),main_target=fresh)
        self.assertEqual(result['state'],'acquire_wrist')
        wrist={'valid':True,'source':'wrist','frame':'odom','stamp_s':1.8,'valid_until_s':2.3,
            'target_id':'actual-bowl-1','position_m':[0.,0.,.35],'covariance_m2':(np.eye(3)*.003**2).tolist(),
            'station_id':'station-1','station_association':'estimated_platform_bounds'}
        result=flow.update(1.8,odom(1.8),wrist_target=wrist,precise_grasp_feasible=True)
        self.assertTrue(result['grasp_ready'])
        self.assertEqual(result['wrist_target']['target_id'],'actual-bowl-1')

    def test_static_map_intersects_real_observation_bounds_instead_of_following_hidden_edges(self):
        depth,K,T=scene_depth(occluded=True)
        one=detect_platform(depth,K,T,1.)
        two=detect_platform(np.fliplr(depth),K,T,1.1)
        fused=fuse_static_platform([one,two])
        self.assertTrue(fused['valid'],fused)
        self.assertLess(np.linalg.norm(fused['center_xy_m']),.005)
        width=lambda row:np.array(row['center_bounds_xy_m']['maximum'])-row['center_bounds_xy_m']['minimum']
        self.assertTrue(np.all(width(fused)<=width(one)+1e-12))
        self.assertEqual(fused['stamp_s'],1.1)


if __name__=='__main__':unittest.main()
