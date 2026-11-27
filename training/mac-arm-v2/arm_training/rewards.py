"""One-shot task events and time-consistent costs; simulator forces are not policy inputs."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RewardWeights:
    trial_lift:float=10.
    hold_verified:float=20.
    complete:float=100.
    failure:float=-200.
    potential:float=5.
    time_per_second:float=.2
    tilt_per_second:float=.5
    residual_per_second:float=.02
    action_rate_per_second:float=.0005


class Reward:
    """Each event is emitted once. A failure cannot receive any success event."""
    def __init__(self,weights=None,*,gamma=.995,control_dt=.05):
        if not 0<gamma<=1 or control_dt<=0:raise ValueError('Invalid reward discount')
        self.discount_rate=-math.log(gamma)/control_dt
        self.w=weights or RewardWeights();self.seen=set();self.closed=False

    def step(self,*,dt,distance_before,distance_after,tilt,residual,previous_residual,
             trial_lift=False,hold_verified=False,complete=False,failure=False):
        values=[dt,distance_before,distance_after,tilt,*residual,*previous_residual]
        if not all(math.isfinite(float(v)) for v in values):raise ValueError('Nonfinite reward evidence')
        if dt<=0:raise ValueError('Nonpositive reward timestep')
        if self.closed:raise RuntimeError('Episode already terminated')
        if failure and complete:raise ValueError('Contradictory terminal evidence')
        if complete and not (hold_verified or 'hold_verified' in self.seen):raise ValueError('Completion without hold evidence')
        phi_before=-min(max(distance_before,0.)/.05,1.)
        phi_after=0. if complete or failure else -min(max(distance_after,0.)/.05,1.)
        parts={'shaping':self.w.potential*(math.exp(-self.discount_rate*dt)*phi_after-phi_before)}
        parts['time']=-self.w.time_per_second*dt
        parts['tilt']=-self.w.tilt_per_second*min((tilt/.15)**2,4.)*dt
        parts['residual']=-self.w.residual_per_second*sum(min(float(a)**2,1.) for a in residual)*dt
        parts['action_rate']=-self.w.action_rate_per_second*sum(min(((a-b)/dt)**2,400.) for a,b in zip(residual,previous_residual))*dt
        if failure:
            parts['failure']=self.w.failure;self.closed=True
        else:
            for key,flag in [('trial_lift',trial_lift),('hold_verified',hold_verified),('complete',complete)]:
                if flag and key not in self.seen:
                    parts[key]=getattr(self.w,key);self.seen.add(key)
            if complete:self.closed=True
        return sum(parts.values()),parts
