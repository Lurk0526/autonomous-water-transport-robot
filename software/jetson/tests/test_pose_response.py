"""Restarts select real saved geometry without assuming a startup posture."""
import copy
import unittest
from vision.pose_response import select_pose_response,response_offsets
from vision.follow_calibration import estimate_jacobian


def measured(pose):
    rows=[]
    for axis,column in ((0,(2,0)),(2,(0,-1)),(3,(0,1))):
        for dp in (20,-20):
            rows.append(dict(axis=axis,actual_delta_p=dp,actual_before_p=pose[str(axis)],
                actual_after_p=pose[str(axis)]+dp,target_p=pose[str(axis)]+dp,
                before={'center':[160,120]},after={'center':[160+column[0]*dp,120+column[1]*dp]}))
    matrix,condition=estimate_jacobian(rows,(0,2,3))
    return dict(result='LOCAL_JACOBIAN_PASS',axes=[0,2,3],trials=rows,jacobian=matrix,
                condition=condition,final_pose=pose)


class PoseResponseTests(unittest.TestCase):
    def test_reboot_chooses_the_closest_measured_geometry_and_ignores_yaw(self):
        first={str(a):1500 for a in range(6)}
        second=dict(first);second.update({'1':1300,'2':2400,'3':1775})
        current={int(a):p for a,p in second.items()};current[0]=1800
        chosen,meta=select_pose_response([('old',measured(first)),('new',measured(second))],current)
        self.assertEqual(meta['source'],'new');self.assertEqual(meta['distance'],0)
        self.assertEqual(response_offsets(chosen)['2'],dict(positive=0,negative=0))

    def test_inconsistent_or_failed_models_never_enable_direct_start(self):
        pose={str(a):1500 for a in range(6)};bad=measured(pose)
        bad['jacobian'][0][0]=99
        with self.assertRaises(ValueError):
            select_pose_response([('corrupt',bad),('failed',{'result':'ABORTED'})],{a:1500 for a in range(6)})
