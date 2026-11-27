"""Run on the receiving Mac. Synthetic geometry and mocked YOLO, no hardware.

These checks do not establish the recognition rate of the real trained model.
Actual RGB/registered-depth inference is a separate Mac simulation acceptance.
"""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

import numpy as np

from arm_training.perception import Segmenter,_fit_center,default_perception_config,estimate_target,infer_tabletop,infer_wrist


PROFILE={'radial_profile_m':[[.03,0],[.036,.008],[.055,.028],[.069,.047],[.078,.06]],
         'height_m':.06,'rim_radius_m':.078}


def fixture():
    K=np.array([[600.,0,320.],[0,600.,240.],[0,0,1.]])
    T=np.diag([1.,-1.,-1.,1.]);T[:3,3]=[.20,-.03,.95]
    table=.35;center=np.array([.21,.015]);distance=T[2,3]-table-PROFILE['height_m']
    yy,xx=np.mgrid[:480,:640]
    world_x=T[0,3]+(xx-K[0,2])*distance/K[0,0]
    world_y=T[1,3]-(yy-K[1,2])*distance/K[1,1]
    mask=(world_x-center[0])**2+(world_y-center[1])**2<=PROFILE['rim_radius_m']**2
    detection={'mask':mask,'class_id':0,'class_name':'target_bowl','confidence':.95}
    return detection,K,T,table,center


class GeometryTests(unittest.TestCase):
    def estimate(self,**overrides):
        detection,K,T,table,_=fixture()
        args={'detection':detection,'K':K,'T_camera_optical_to_odom':T,
              'table_top_m':table,'bowl_profile':PROFILE,'stamp_s':1.,'source':'wrist',
              'config':{'target_id':'bowl-1','now_s':1.05,'calibration_status':'simulation_only'}}
        args.update(overrides)
        return estimate_target(**args)

    def test_known_plane_recovers_bottom_center(self):
        result=self.estimate();self.assertTrue(result['valid'],result)
        _,_,_,table,center=fixture()
        np.testing.assert_allclose(result['position_m'],[*center,table],atol=.002)
        self.assertEqual(result['source'],'wrist');self.assertEqual(result['frame'],'odom')
        self.assertEqual(result['target_id'],'bowl-1');self.assertFalse(result['hardware_ready'])
        self.assertFalse(result['quality']['uses_simulator_truth'])
        self.assertGreater(result['quality']['visible_arc_deg'],300)
        covariance=np.array(result['covariance_m2'])
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance)>0))
        json.dumps(result,allow_nan=False)

    def test_missing_detection_wrong_class_low_confidence_and_identity_fail_closed(self):
        self.assertEqual(self.estimate(detection=None)['reason'],'no_detection')
        detection,*_=fixture();detection['class_name']='other'
        self.assertEqual(self.estimate(detection=detection)['reason'],'wrong_class')
        detection,*_=fixture();detection['confidence']=.2
        self.assertEqual(self.estimate(detection=detection)['reason'],'low_confidence')
        result=self.estimate(config={'now_s':1.05})
        self.assertEqual(result['reason'],'target_id_missing')

    def test_stale_and_future_observations_are_rejected(self):
        result=self.estimate(config={'target_id':'bowl-1','now_s':2.})
        self.assertEqual(result['reason'],'target_expired');self.assertIsNone(result['position_m'])
        result=self.estimate(config={'target_id':'bowl-1','now_s':.5})
        self.assertEqual(result['reason'],'target_from_future')

    def test_invalid_values_stay_json_safe(self):
        result=self.estimate(stamp_s=float('nan'))
        self.assertFalse(result['valid']);json.dumps(result,allow_nan=False)

    def test_main_camera_requires_depth_and_rejects_table_water(self):
        self.assertEqual(self.estimate(source='main_rgbd')['reason'],'registered_depth_required')
        _,_,T,table,_=fixture()
        for plane in (table,table+PROFILE['height_m']):
            depth=np.full((480,640),T[2,3]-plane)
            result=self.estimate(source='main_rgbd',depth_m=depth)
            self.assertFalse(result['valid']);self.assertEqual(result['reason'],'depth_sidewall_insufficient')

    def test_depth_alignment_and_monocular_contract(self):
        result=self.estimate(source='main_rgbd',depth_m=np.ones((20,20)))
        self.assertEqual(result['reason'],'rgb_depth_shape_mismatch')
        self.assertEqual(self.estimate(depth_m=np.ones((480,640)))['reason'],'wrist_mode_is_monocular_do_not_supply_depth')

    def test_unregistered_range_depth_is_rejected(self):
        result=self.estimate(source='main_rgbd',depth_m=np.ones((480,640)),
            config={'target_id':'bowl-1','depth_kind':'euclidean_range'})
        self.assertEqual(result['reason'],'depth_must_be_registered_optical_z_metres')

    def test_clipped_mask_and_nonrigid_calibration_rejected(self):
        detection,*_=fixture();detection['mask'][:,0:20]=True
        self.assertFalse(self.estimate(detection=detection)['valid'])
        _,_,T,_,_=fixture();T[0,0]=2
        self.assertEqual(self.estimate(T_camera_optical_to_odom=T)['reason'],'camera_transform_not_rigid')

    def test_profile_center_tolerates_outliers(self):
        rng=np.random.default_rng(7);angles=np.linspace(0,2*math.pi,700,endpoint=False)
        radii=np.linspace(.04,.075,len(angles));center=np.array([.3,-.2])
        xy=center+np.column_stack((np.cos(angles),np.sin(angles)))*radii[:,None]
        xy+=rng.normal(0,.0005,xy.shape);xy[:50]+=rng.normal(0,.03,(50,2))
        estimate,cov,quality=_fit_center(xy,radii,.004,default_perception_config(),80)
        np.testing.assert_allclose(estimate,center,atol=.001)
        self.assertGreater(quality['inlier_fraction'],.85)
        self.assertTrue(np.isfinite(cov).all())

    def test_short_visible_arc_does_not_create_precise_target(self):
        angles=np.linspace(0,np.deg2rad(25),200);radius=.078
        xy=np.column_stack((np.cos(angles),np.sin(angles)))*radius
        with self.assertRaises(ValueError):
            _fit_center(xy,np.full(len(xy),radius),.001,default_perception_config(),150)

    def test_main_depth_geometry_uses_sidewall_profile(self):
        # Isolate centre fitting from depth sampling, which is checked separately.
        angles=np.linspace(0,2*math.pi,360,endpoint=False)
        heights=np.linspace(.012,.052,360)
        radii=np.interp(heights,np.array(PROFILE['radial_profile_m'])[:,1],np.array(PROFILE['radial_profile_m'])[:,0])
        center=np.array([.21,.015])
        points=np.column_stack((center[0]+radii*np.cos(angles),center[1]+radii*np.sin(angles),.35+heights))
        with patch('arm_training.perception._sidewall_points',return_value=(points,radii)):
            result=self.estimate(source='main_rgbd',depth_m=np.ones((480,640)))
        self.assertTrue(result['valid'],result)
        np.testing.assert_allclose(result['position_m'][:2],center,atol=.001)
        self.assertEqual(result['quality']['method'],'edge_sidewall_profile_rgbd')

    def test_registered_depth_unprojection_and_edge_sampling_together(self):
        # Analytic upright sloped-wall depth, independent of any MuJoCo state.
        # A camera directly above the centre sees this same radial profile.
        _,K,T,table,_=fixture();yy,xx=np.mgrid[:480,:640]
        angular_radius=np.hypot((xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1])
        height=PROFILE['height_m'];distance=T[2,3]-table
        profile=np.array(PROFILE['radial_profile_m'])
        mask=angular_radius<=PROFILE['rim_radius_m']/(distance-height)
        lower=np.zeros(mask.shape);upper=np.full(mask.shape,height)
        for _ in range(32):
            middle=(lower+upper)/2
            residual=angular_radius*(distance-middle)-np.interp(middle,profile[:,1],profile[:,0])
            lower=np.where(residual>0,middle,lower);upper=np.where(residual<=0,middle,upper)
        depth=distance-(lower+upper)/2;depth[~mask]=0
        detection={'mask':mask,'class_name':'target_bowl','class_id':0,'confidence':.95}
        result=self.estimate(detection=detection,source='main_rgbd',depth_m=depth)
        self.assertTrue(result['valid'],result)
        np.testing.assert_allclose(result['position_m'],[T[0,3],T[1,3],table],atol=.001)
        self.assertGreater(result['quality']['sidewall_height_span_m'],.01)


class TabletopCropTests(unittest.TestCase):
    def test_wrist_hint_cannot_create_a_detection(self):
        rgb=np.zeros((480,640,3),np.uint8);detector=Mock();detector.infer.return_value=[];detector.last_inference_s=.01
        camera=np.array([[400.,0.,320.],[0.,400.,240.],[0.,0.,1.]])
        for z,expected in [(.8,2),(-.2,1)]:
            predictions,record=infer_wrist(detector,rgb,camera,np.eye(4),[0,0,z],PROFILE,1.45)
            self.assertEqual(predictions,[]);self.assertEqual(len(record['crops']),expected)
        mask=np.ones((10,10),bool)
        def detected(frame):
            if frame.shape==rgb.shape:return []
            return [{'mask':np.resize(mask,frame.shape[:2]),'bbox':[0,0,5,5],'center':[2,2],'confidence':.9}]
        detector.infer.side_effect=detected
        predictions,record=infer_wrist(detector,rgb,camera,np.eye(4),[0,0,.8],PROFILE,1.45)
        json.dumps({'record':record,'prediction':{k:v for k,v in predictions[0].items() if k!='mask'}},allow_nan=False)

    def test_overlapping_learned_masks_are_mapped_back_once(self):
        image=np.zeros((96,128,3),np.uint8);image[32:52,64:88]=255
        depth=np.ones((96,128));depth[32:52,64:88]=.97
        transform=np.diag([1.,-1.,-1.,1.]);transform[2,3]=1.
        camera=np.array([[100.,0.,64.],[0.,100.,48.],[0.,0.,1.]])
        def detect(frame):
            mask=frame[:,:,0]>200;y,x=np.where(mask)
            return [{'mask':mask,'bbox':[x.min(),y.min(),x.max()+1,y.max()+1],
                     'center':[x.mean(),y.mean()],'confidence':.8 if frame.shape==image.shape else .9}]
        detector=Mock();detector.infer.side_effect=detect;detector.last_inference_s=.01
        options={'crop_scale':2.2,'gamma':.6,'maximum_crops':8}
        predictions,record=infer_tabletop(detector,image,depth,camera,transform,0.,.06,options)
        self.assertEqual(len(record['crops']),2);self.assertEqual(len(predictions),1)
        np.testing.assert_array_equal(predictions[0]['mask'],image[:,:,0]>200)
        self.assertEqual(predictions[0]['bbox'],[64,32,88,52]);self.assertAlmostEqual(record['inference_s'],.02)
        detector.infer.side_effect=lambda frame:[]
        self.assertEqual(infer_tabletop(detector,image,depth,camera,transform,0.,.06,options)[0],[])


class SegmenterContractTests(unittest.TestCase):
    def test_constructor_does_not_predict_or_warm_up(self):
        fake=Mock();fake.names={0:'target_bowl'};fake.task='segment'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'target-bowl.pt';path.write_bytes(b'mocked weights')
            with patch('ultralytics.YOLO',return_value=fake):
                detector=Segmenter(path,device='cpu')
            fake.predict.assert_not_called();self.assertIsNone(detector.last_inference_s)

    def test_wrong_weight_classes_rejected_without_inference(self):
        fake=Mock();fake.names={0:'person'};fake.task='segment'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.pt';path.write_bytes(b'mocked weights')
            with patch('ultralytics.YOLO',return_value=fake):
                with self.assertRaises(ValueError):Segmenter(path)
            fake.predict.assert_not_called()

    def test_rgb_to_bgr_and_aligned_boolean_masks(self):
        import torch
        fake=Mock();fake.names={0:'target_bowl'};fake.task='segment'
        fake.predictor.model.model.parameters.return_value=iter([torch.nn.Parameter(torch.zeros(1))])
        # MagicMock supplies the special __len__ used by the Ultralytics result API.
        from unittest.mock import MagicMock
        boxes=MagicMock();boxes.__len__.return_value=1
        boxes.xyxy=torch.tensor([[1.,1.,6.,6.]])
        boxes.conf=torch.tensor([.9]);boxes.cls=torch.tensor([0.])
        masks=Mock();masks.data=torch.ones((1,8,8))
        result=Mock();result.boxes=boxes;result.masks=masks
        fake.predict.return_value=[result]
        image=np.zeros((8,8,3),dtype=np.uint8);image[0,0]=[12,34,56]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'target-bowl.pt';path.write_bytes(b'mocked weights')
            with patch('ultralytics.YOLO',return_value=fake):detector=Segmenter(path)
            detections=detector.infer(image)
        np.testing.assert_array_equal(fake.predict.call_args.kwargs['source'][0,0],[56,34,12])
        self.assertEqual(len(detections),1);self.assertEqual(detections[0]['mask'].dtype,np.bool_)
        self.assertEqual(detections[0]['mask'].shape,(8,8))
        self.assertEqual(detections[0]['class_name'],'target_bowl')


if __name__=='__main__':unittest.main()
