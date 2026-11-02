"""Recover a clear bowl across the frame while retaining ambiguity checks."""
import time
import unittest

from vision.bowl_follow import FollowPolicy,FollowSettings
from vision.follow_worker import FollowWorker
from vision.target_association import VisibleTargetReacquisition
from tests.test_bowl_follow import detection


def record(stamp,center=(240,120),confidence=.95):
    result=detection(stamp,center,confidence)
    result['weights_sha256']='verified-model'
    return result


class ReacquisitionTests(unittest.TestCase):
    def test_three_distinct_clear_frames_can_reacquire_anywhere_in_view(self):
        acquirer=VisibleTargetReacquisition();now=time.time()
        self.assertIsNone(acquirer.consider(record(now),'verified-model',now))
        self.assertIsNone(acquirer.consider(record(now+.05),'verified-model',now+.05))
        result=acquirer.consider(record(now+.1),'verified-model',now+.1)
        self.assertEqual(result['detections'][0]['mask_centroid_px'],[240,120])

    def test_duplicates_weak_candidates_ambiguity_and_wrong_models_never_relock(self):
        for kind in ('duplicate','weak','multiple','wrong_model','stale','missing'):
            with self.subTest(kind=kind):
                acquirer=VisibleTargetReacquisition();now=time.time()
                for i in range(6):
                    stamp=now if kind=='duplicate' else now+i*.05
                    result=record(stamp,confidence=.7 if kind=='weak' else .95)
                    if kind=='multiple':result['detections']+=record(stamp,(180,120))['detections']
                    if kind=='wrong_model':result['weights_sha256']='other-model'
                    if kind=='stale':result['stamp']['sec']-=1
                    if kind=='missing':result['detections']=[]
                    self.assertIsNone(acquirer.consider(result,'verified-model',now+i*.05))

    def test_weak_background_proposal_does_not_block_unique_clear_bowl(self):
        acquirer=VisibleTargetReacquisition();now=time.time()
        for i in range(3):
            value=record(now+i*.05)
            value['detections']+=record(now+i*.05,(100,120),.55)['detections']
            recovered=acquirer.consider(value,'verified-model',now+i*.05)
        self.assertEqual(len(recovered['detections']),1)
        self.assertEqual(recovered['raw_detection_count'],2)

    def test_fast_continuous_bowl_does_not_reset_acquisition_on_vertical_motion(self):
        now=time.time()
        for responsive in (False,True):
            with self.subTest(responsive=responsive):
                policy=FollowPolicy(FollowSettings(confidence=.75,locked_confidence=.6,
                    stable_frames=2,period=.05,responsive_tracking=responsive))
                for i,cy in enumerate((70,70,105)):
                    value=record(now+i*.06,(160,cy))
                    value['detections'][0]['bbox_xyxy_px']=[100,cy-50,220,cy+50]
                    plan=policy.update(value,now+i*.06,i*.06)
                self.assertEqual(plan['state'],'move' if responsive else 'acquiring')

    def test_skipped_but_fresh_frames_do_not_force_reacquisition(self):
        now=time.time()
        for responsive in (False,True):
            with self.subTest(responsive=responsive):
                policy=FollowPolicy(FollowSettings(confidence=.75,locked_confidence=.6,
                    stable_frames=2,period=.05,responsive_tracking=responsive))
                for offset in (0,.06,.3):
                    plan=policy.update(record(now+offset,(200,120)),now+offset,offset)
                self.assertEqual(plan['state'],'move' if responsive else 'acquiring')

    def test_worker_recovers_a_distant_bowl_after_loss_without_operator_restart(self):
        self.run_worker_case(ambiguous=False)

    def test_worker_keeps_multiple_distant_bowls_stopped(self):
        self.run_worker_case(ambiguous=True)

    def run_worker_case(self,ambiguous):
        config=dict(reacquire_visible_target=True,operator_target_lock=dict(
            center=[40,120],bbox=[5,95,75,145],weights_sha256='verified-model'))
        worker=FollowWorker(FollowSettings(confidence=.75,stable_frames=2,period=.05,
                            locked_confidence=.6,responsive_tracking=True),config)
        worker.start()
        try:
            for i in range(9):
                value=record(time.time(),(230,120))
                if ambiguous:value['detections']+=record(time.time(),(170,120))['detections']
                worker.submit(value);time.sleep(.06)
            result=worker.snapshot()
            if ambiguous:
                self.assertEqual(result['target_reacquisitions'],0)
                self.assertEqual(result['state'],'ambiguous');self.assertFalse(result['delta_p'])
            else:
                self.assertGreaterEqual(result['target_reacquisitions'],1)
                self.assertEqual(result['state'],'move');self.assertTrue(result['delta_p'])
        finally:worker.close()


if __name__=='__main__':unittest.main()
