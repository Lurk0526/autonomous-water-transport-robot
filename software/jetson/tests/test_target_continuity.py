"""Only the same recently trusted bowl may briefly use reduced confidence."""
import time
import unittest
from vision.bowl_follow import FollowPolicy,FollowSettings
from tests.test_bowl_follow import detection


class ContinuityTests(unittest.TestCase):
    def setUp(self):
        self.start=time.time()
        self.policy=FollowPolicy(FollowSettings(confidence=.75,locked_confidence=.6,
            stable_frames=2,period=.05))

    def update(self,offset,center=(200,90),confidence=.9):
        return self.policy.update(detection(self.start+offset,center,confidence),
                                  self.start+offset+.02,offset)

    def acquire(self):
        self.update(0);self.update(.05)

    def test_single_confidence_dip_keeps_recent_locked_bowl(self):
        self.acquire();out=self.update(.1,(200,85),.65)
        self.assertEqual(out['state'],'move');self.assertTrue(out['locked_continuity'])

    def test_low_confidence_cannot_start_a_new_track(self):
        out=self.update(0,confidence=.65)
        self.assertEqual(out['state'],'low_confidence');self.assertFalse(out['delta_p'])

    def test_continuity_expires_without_new_trusted_measurement(self):
        self.acquire()
        for step in range(1,9):out=self.update(.05+step*.05,confidence=.65)
        self.assertEqual(out['state'],'low_confidence')

    def test_spatially_different_low_score_candidate_is_not_followed(self):
        self.acquire();out=self.update(.1,(100,90),.65)
        self.assertEqual(out['state'],'low_confidence')

    def test_lost_target_removes_confidence_grace(self):
        self.acquire()
        self.policy.update(dict(valid=False,state='live'),self.start+.08,.08)
        out=self.update(.1,confidence=.65)
        self.assertEqual(out['state'],'low_confidence')

    def test_multiple_candidates_still_stop_motion(self):
        self.acquire();record=detection(self.start+.1,(200,90),.65)
        record['detections']*=2
        out=self.policy.update(record,self.start+.12,.1)
        self.assertEqual(out['state'],'ambiguous');self.assertFalse(out['delta_p'])
