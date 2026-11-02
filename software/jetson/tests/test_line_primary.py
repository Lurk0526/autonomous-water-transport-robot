"""Primary-line observation and bounded switching; synthetic images only."""
import json
from pathlib import Path
import unittest
import importlib.util

from cart.line_tracking import NavigationMode

ROOT=Path(__file__).resolve().parents[1]


class ModeSwitchTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/'config/cart-line-tracking.json').read_text())
        self.good=dict(valid=True,confidence=.95,reason='line_observed')
        self.lost=dict(valid=False,confidence=0,reason='no_continuous_line')

    def tick(self,policy,now,line,stamp,**changes):
        readiness=dict(base_ready=True,obstacles_ready=True);readiness.update(changes)
        return policy.update(now,line,stamp,**readiness)

    def test_unspecified_readiness_does_not_default_to_permission_to_move(self):
        policy=NavigationMode(self.cfg)
        self.assertEqual(policy.update(0,self.good,0),"STOP")

    def active(self):
        p=NavigationMode(self.cfg)
        for t in [0,.1,.2]:self.tick(p,t,self.good,t)
        self.assertEqual(p.mode,'LINE');return p

    def test_multiple_distinct_frames_are_required_before_line_mode(self):
        p=NavigationMode(self.cfg)
        for t in [0,.05,.1,.15]:self.tick(p,t,self.good,0)
        self.assertEqual(p.mode,'WAIT_LINE')
        self.tick(p,.2,self.good,.2);self.tick(p,.3,self.good,.3)
        self.assertEqual(p.mode,'LINE')

    def test_brief_loss_slows_then_valid_backup_can_take_over(self):
        p=self.active()
        self.assertEqual(self.tick(p,.3,self.lost,.3,backup_ready=True),'LINE_LOST_SLOW')
        self.assertEqual(self.tick(p,.6,self.lost,.6,backup_ready=True),'BACKUP')
        for t in [.7,.8,.9]:self.tick(p,t,self.good,t,backup_ready=True)
        self.assertEqual(p.mode,'LINE')

    def test_no_backup_means_stop_and_later_line_cannot_unlatch_it(self):
        p=self.active();self.tick(p,.6,self.lost,.6,backup_ready=False)
        self.assertEqual(p.mode,'STOP');self.assertEqual(p.reason,'line_loss_stop')
        self.tick(p,.7,self.good,.7,backup_ready=True);self.assertEqual(p.mode,'STOP')

    def test_base_feedback_and_obstacle_validity_override_line_confidence(self):
        for changes in [dict(base_ready=False),dict(obstacles_ready=False),dict(base_ready='true')]:
            p=self.active();self.assertEqual(self.tick(p,.3,self.good,.3,**changes),'STOP')

    def test_ambiguous_junction_does_not_get_solved_by_blind_line_following(self):
        p=self.active();line=dict(self.good,reason='junction_or_wide_marking')
        self.assertEqual(self.tick(p,.3,line,.3,backup_ready=True),'STOP')

    def test_stale_or_future_image_cannot_establish_primary_mode(self):
        for stamp in [-1,1,float('nan')]:
            p=NavigationMode(self.cfg);self.tick(p,0,self.good,stamp)
            self.assertEqual(p.mode,'WAIT_LINE')

    def test_clock_reversal_is_a_latched_stop(self):
        p=self.active();self.assertEqual(self.tick(p,.1,self.good,.1),'STOP')


@unittest.skipUnless(importlib.util.find_spec('cv2') and importlib.util.find_spec('numpy'),'line image checks require OpenCV/NumPy')
class PixelLineTests(unittest.TestCase):
    def test_auto_detects_each_supported_color_without_color_retraining(self):
        import cv2,numpy as np
        from cart.line_tracking import detect_line
        for color,paint in [('black',(20,20,20)),('white',(240,240,240)),('yellow',(235,205,25))]:
            with self.subTest(color=color):
                rgb=np.full((240,320,3),140,np.uint8);cv2.line(rgb,(180,239),(170,100),paint,8)
                result=detect_line(rgb)
                self.assertTrue(result['valid'],result['reason']);self.assertEqual(result['color'],color)
                self.assertGreater(result['lateral_error_normalized'],0)

    def test_blank_floor_is_not_a_line(self):
        import numpy as np
        from cart.line_tracking import detect_line
        for level in [0,140,255]:
            self.assertFalse(detect_line(np.full((240,320,3),level,np.uint8))['valid'])

    def test_parallel_candidates_and_cross_bar_are_not_arbitrarily_selected(self):
        import cv2,numpy as np
        from cart.line_tracking import detect_line
        rgb=np.full((240,320,3),140,np.uint8)
        cv2.line(rgb,(130,239),(130,100),(20,20,20),8);cv2.line(rgb,(190,239),(190,100),(20,20,20),8)
        self.assertFalse(detect_line(rgb)['valid'])
        rgb[:]=140;cv2.line(rgb,(160,239),(160,100),(20,20,20),8);cv2.line(rgb,(0,170),(319,170),(20,20,20),14)
        self.assertEqual(detect_line(rgb)['reason'],'junction_or_wide_marking')

    def test_pixels_cannot_become_metric_path_without_measured_mount(self):
        import numpy as np
        from cart.line_tracking import line_to_ground
        self.assertFalse(line_to_ground([[160,200]]*8,np.eye(3),np.zeros(5),np.eye(4),0)['valid'])

    def test_calibrated_ground_ray_recovers_known_lateral_offset(self):
        import numpy as np,math
        from cart.line_tracking import line_to_ground
        beta=.45
        t=np.array([[0,-math.sin(beta),math.cos(beta),0],[-1,0,0,0],[0,-math.cos(beta),-math.sin(beta),.35],[0,0,0,1]])
        pixels=[]
        for x in np.linspace(.3,1,10):
            z=math.cos(beta)*x+math.sin(beta)*.35
            pixels.append([160-220*.08/z,120+220*(-math.sin(beta)*x+math.cos(beta)*.35)/z])
        result=line_to_ground(pixels,[[220,0,160],[0,220,120],[0,0,1]],[0]*5,t,0,True)
        self.assertTrue(result['valid'])
        self.assertTrue(all(abs(p[1]-.08)<1e-8 for p in result['points_xy_m']))


if __name__=='__main__':unittest.main()
