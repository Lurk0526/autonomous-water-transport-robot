"""Known-geometry PnP and conservative occupancy checks; synthetic fixtures only."""
import importlib.util
import unittest

AVAILABLE=bool(importlib.util.find_spec('cv2') and importlib.util.find_spec('numpy'))


@unittest.skipUnless(AVAILABLE,'Optional camera checks require numpy/OpenCV')
class CameraGeometryTests(unittest.TestCase):
    def test_pnp_recovers_known_camera_transform_despite_outliers(self):
        import cv2,numpy as np
        from cart.vision_checks import pnp_motion
        rng=np.random.default_rng(31)
        xyz=np.column_stack([rng.uniform(-.6,.6,120),rng.uniform(-.4,.4,120),rng.uniform(1,3,120)])
        k=np.array([[500.,0,320],[0,500,240],[0,0,1]])
        rotation=np.array([[np.cos(.02),0,np.sin(.02)],[0,1,0],[-np.sin(.02),0,np.cos(.02)]])
        target=np.array([.03,-.01,.005])
        projected=xyz@rotation.T+target
        uv=(projected[:,:2]/projected[:,2,None])*500+np.array([320,240])
        uv+=rng.normal(0,.1,uv.shape);uv[:20]=rng.uniform([0,0],[640,480],(20,2))
        result=pnp_motion(xyz,uv,k,np.zeros(5))
        self.assertTrue(result['valid']);self.assertGreater(result['inliers'],90)
        self.assertLess(np.linalg.norm(np.array(result['translation_vector_m'])-target),.003)

    def test_insufficient_or_nan_correspondences_do_not_produce_a_pose(self):
        import numpy as np
        from cart.vision_checks import pnp_motion
        self.assertFalse(pnp_motion([[0,0,1]]*5,[[0,0]]*5,np.eye(3),np.zeros(5))['valid'])
        self.assertFalse(pnp_motion([[0,0,float('nan')]]*40,[[0,0]]*40,np.eye(3),np.zeros(5))['valid'])

    def test_unknown_mount_and_attitude_do_not_make_a_free_grid(self):
        import numpy as np
        from cart.obstacles import build_grid
        for args in [dict(),dict(mount_verified=True,roll_rad=.2,pitch_rad=0),dict(mount_verified=True)]:
            result=build_grid([[1,0,0]],[0],[.9],np.eye(4),0,**args)
            self.assertFalse(result['valid']);self.assertTrue((result['occupancy']==-1).all())

    def test_obstacle_and_drop_override_ground_and_unknown_stays_unknown(self):
        import numpy as np
        from cart.obstacles import build_grid
        result=build_grid([[1,0,0],[1.02,0,.3],[2,0,0],[3,0,-.12],[4,0,float('nan')]],
                          [0,0,2,0,0],[.9]*5,np.eye(4),0,mount_verified=True,roll_rad=0,pitch_rad=0)
        grid=result['occupancy']
        self.assertEqual(grid[20,10],100)
        self.assertEqual(grid[20,20],-1)
        self.assertEqual(grid[20,30],100)
        self.assertEqual(grid[20,40],-1)


if __name__=='__main__':unittest.main()
