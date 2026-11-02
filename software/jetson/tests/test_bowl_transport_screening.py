import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim/bowl_transport'))
from screening import response,quintic
from actuator_screening import audit

class NumericalScreening(unittest.TestCase):
    def test_constant_forcing_has_single_static_response(self):
        r=response(lambda t:(.1,0,0),10,.01,2.4,.3)
        self.assertAlmostEqual(r['final_edge_response_m'],.078*.1/9.81,places=7)
    def test_mirror_forcing_mirrors_response(self):
        a=response(lambda t:(.1*math.sin(t),0,0),3,.01,2.4,.12)
        b=response(lambda t:(-.1*math.sin(t),0,0),3,.01,2.4,.12)
        self.assertAlmostEqual(a['final_edge_response_m'],-b['final_edge_response_m'])
    def test_out_of_domain_does_not_return_plausible_number(self):
        for fn in (lambda t:(0,-8,0),lambda t:(0,0,.5),lambda t:(math.nan,0,0)):
            with self.assertRaises(ValueError):response(fn,1,.01,2.4,.12)
    def test_quintic_endpoints_and_velocity_acceleration_continuity(self):
        for t in (-1,0,2,3):
            p,v,a,j=quintic(t,2,.12)
            self.assertAlmostEqual(p,0 if t<=0 else .12)
            self.assertAlmostEqual(v,0);self.assertAlmostEqual(a,0)
    def test_shortening_same_motion_increases_peak_acceleration(self):
        def peak(T):return max(abs(quintic(i*T/100,T,.12)[2]) for i in range(101))
        self.assertAlmostEqual(peak(1)/peak(2),4)
    def test_actuator_delay_audit(self):self.assertTrue(audit()['all_checks_passed'])

if __name__=='__main__':unittest.main()
