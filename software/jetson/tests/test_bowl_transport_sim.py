"""Meaningful gate checks; these tests never access robot hardware."""
import json
from pathlib import Path
import sys
import unittest

SIM=Path(__file__).resolve().parents[1]/'sim'/'bowl_transport'
sys.path.insert(0,str(SIM))
from flow import Supervisor, Observation, CONTACT_PHASES, edge_rise_proxy
from test_bowl_transport_faults import nominal_observation


class TransportGates(unittest.TestCase):
    def setUp(self):
        self.g=json.loads((SIM/'scenario.json').read_text())['initial_gate_thresholds_require_calibration']
        self.t=0.

    def ready(self,dt=.01,**changes):
        self.t+=dt
        return nominal_observation(self.t,**changes)

    def test_cannot_start_grasp_or_place_while_base_moves(self):
        for phase in ['PICK_WAIT','PLACE_WAIT']:
            f=Supervisor(self.g);f.phase=phase
            for _ in range(100):f.update(self.ready(speed=.03),.01)
            self.assertEqual(f.phase,phase)

    def test_settle_requires_continuous_fresh_observations(self):
        f=Supervisor(self.g);f.phase='PICK_WAIT'
        for _ in range(40):f.update(self.ready(),.01)
        f.update(self.ready(target_age=.5),.01)
        for _ in range(40):f.update(self.ready(),.01)
        self.assertEqual(f.phase,'PICK_WAIT')
        for _ in range(11):f.update(self.ready(),.01)
        self.assertEqual(f.phase,'DESCEND')

    def test_motion_during_contact_latches_hold(self):
        for phase in CONTACT_PHASES:
            f=Supervisor(self.g);f.phase=phase
            self.assertEqual(f.update(self.ready(dt=.02,speed=.03),.02),'HOLD')
            self.assertEqual(f.update(self.ready(dt=.02),.02),'HOLD')

    def test_grasp_and_support_are_not_inferred_from_time(self):
        for phase,change in [('CLOSE',{'grasp_verified':False}),('LOWER',{'support_verified':False})]:
            f=Supervisor(self.g);f.phase=phase
            for _ in range(50):f.update(self.ready(dt=.02,**change),.02)
            self.assertEqual(f.phase,phase)

    def test_release_requires_support(self):
        f=Supervisor(self.g);f.phase='OPEN'
        self.assertEqual(f.update(self.ready(dt=.02,support_verified=False),.02),'HOLD')

    def test_nan_observation_holds(self):
        f=Supervisor(self.g)
        self.assertEqual(f.update(self.ready(dt=.02,speed=float('nan')),.02),'HOLD')

    def test_to_rim_has_no_positive_proxy_margin(self):
        self.assertEqual(edge_rise_proxy(.078,0,0),0)
        self.assertGreater(edge_rise_proxy(.078,.05,0),0)
        self.assertGreater(edge_rise_proxy(.078,0,.01),0)


if __name__=='__main__':unittest.main()
