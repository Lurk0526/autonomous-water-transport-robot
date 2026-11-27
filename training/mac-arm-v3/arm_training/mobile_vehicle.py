"""Forward-only planar vehicle surrogate; unmeasured, no hardware transport."""
from dataclasses import dataclass,asdict
import math
import numpy as np
from .mobile_planner import wrap


def clip(x,a,b):return min(b,max(a,float(x)))


def _finite(value,name):
    value=float(value)
    if not math.isfinite(value):raise ValueError('nonfinite_'+name)
    return value


def _validate_state(state):
    for name,value in state.record().items():_finite(value,name)
    if state.speed_m_s<0:raise ValueError('forward_vehicle_has_negative_speed')


@dataclass
class VehicleState:
    x_m:float=0.
    y_m:float=0.
    yaw_rad:float=0.
    speed_m_s:float=0.
    steering_rad:float=0.
    acceleration_m_s2:float=0.
    yaw_rate_rad_s:float=0.

    @property
    def pose(self):return [self.x_m,self.y_m,self.yaw_rad]
    def odometry(self,t):
        _validate_state(self);t=_finite(t,'odometry_time')
        return {'valid':True,'frame':'odom','stamp_s':t,'xy_yaw':self.pose,
            'speed_m_s':self.speed_m_s,'yaw_rate_rad_s':self.yaw_rate_rad_s,
            'source':'simulated_bicycle_state_not_real_odometry'}
    def record(self):return asdict(self)


def advance(state,target_speed,target_steer,dt,cfg):
    """Integrate bounded speed/steering at <=10ms; never teleport or spin."""
    _validate_state(state);dt=_finite(dt,'vehicle_step')
    if dt<0:raise ValueError('negative_vehicle_step')
    v=cfg['vehicle'];control=cfg['drive']
    for key in ('wheelbase_m','max_steer_rad'):
        if _finite(v[key],key)<=0:raise ValueError('nonpositive_'+key)
    if v['max_steer_rad']>=math.pi/2:raise ValueError('invalid_bicycle_steering_limit')
    for key in ('steering_time_constant_s','steering_rate_rad_s','max_speed_m_s',
                'speed_time_constant_s','max_deceleration_m_s2','max_acceleration_m_s2'):
        if _finite(control[key],key)<=0:raise ValueError('nonpositive_'+key)
    if abs(state.steering_rad)>v['max_steer_rad']+1e-12:
        raise ValueError('initial_steering_out_of_bounds')
    target_speed=clip(_finite(target_speed,'target_speed'),0.,control['max_speed_m_s'])
    target_steer=clip(_finite(target_steer,'target_steering'),-v['max_steer_rad'],v['max_steer_rad'])
    if dt==0:return []
    n=max(1,math.ceil(dt/.01));h=dt/n;samples=[]
    for _ in range(n):
        steer_rate=clip((target_steer-state.steering_rad)/control['steering_time_constant_s'],
            -control['steering_rate_rad_s'],control['steering_rate_rad_s'])
        next_steer=clip(state.steering_rad+steer_rate*h,
            min(state.steering_rad,target_steer),max(state.steering_rad,target_steer))
        acceleration=clip((target_speed-state.speed_m_s)/control['speed_time_constant_s'],
            -control['max_deceleration_m_s2'],control['max_acceleration_m_s2'])
        # Small configured time constants must not make an Euler step overshoot
        # its bounded target, or make the reported acceleration disagree with it.
        next_v=clip(state.speed_m_s+acceleration*h,
            min(state.speed_m_s,target_speed),max(state.speed_m_s,target_speed))
        stop_epsilon=float(control.get('stationary_speed_epsilon_m_s',.0001))
        if not 0<=stop_epsilon<=.001:raise ValueError('invalid_stationary_speed_epsilon')
        if target_speed==0 and next_v<=stop_epsilon and state.speed_m_s<=control['max_deceleration_m_s2']*h:
            next_v=0.  # Finite parking-brake capture, within the acceleration bound.
        acceleration=(next_v-state.speed_m_s)/h
        vmid=(next_v+state.speed_m_s)/2;smid=(next_steer+state.steering_rad)/2
        yaw_rate=vmid*math.tan(smid)/v['wheelbase_m']
        state.x_m+=vmid*math.cos(state.yaw_rad+yaw_rate*h/2)*h
        state.y_m+=vmid*math.sin(state.yaw_rad+yaw_rate*h/2)*h
        state.yaw_rad=wrap(state.yaw_rad+yaw_rate*h)
        state.speed_m_s=next_v;state.steering_rad=next_steer;state.acceleration_m_s2=acceleration
        state.yaw_rate_rad_s=next_v*math.tan(next_steer)/v['wheelbase_m'];samples.append(state.pose.copy())
    return samples


class PathFollower:
    def __init__(self,path,cfg):
        self.samples=path['samples'];self.cfg=cfg;self.index=0;self.failure_reason=None
        if not self.samples:raise ValueError('Empty forward path')
        previous=-math.inf
        for sample in self.samples:
            for key in ('x_m','y_m','yaw_rad','s_m'):_finite(sample[key],'path_'+key)
            if sample['s_m']<0 or sample['s_m']<previous:raise ValueError('nonmonotonic_forward_path')
            previous=sample['s_m']
    def command(self,state):
        _validate_state(state)
        if self.failure_reason:return 0.,0.
        points=self.samples;max_advance=int(self.cfg['drive'].get('projection_search_samples',30))
        if max_advance<1:raise ValueError('Invalid projection search window')
        end=min(len(points),self.index+max_advance)
        distances=[math.hypot(p['x_m']-state.x_m,p['y_m']-state.y_m) for p in points[self.index:end]]
        self.index+=int(np.argmin(distances))
        progress=points[self.index]['s_m'];look=self.cfg['drive']['lookahead_m']
        ref=self.index
        while ref+1<len(points) and points[ref]['s_m']<progress+look:ref+=1
        p=points[ref];dx,dy=p['x_m']-state.x_m,p['y_m']-state.y_m
        local_y=-math.sin(state.yaw_rad)*dx+math.cos(state.yaw_rad)*dy
        ld2=max(dx*dx+dy*dy,.0016)
        desired_steer=math.atan(self.cfg['vehicle']['wheelbase_m']*2*local_y/ld2)
        remaining=max(0.,points[-1]['s_m']-progress)
        distance_goal=math.hypot(points[-1]['x_m']-state.x_m,points[-1]['y_m']-state.y_m)
        stop_distance=self.cfg['drive']['goal_stop_distance_m']
        control=self.cfg['drive'];gates=self.cfg['coordinator']
        # The analytic path has instantaneous curvature changes; the vehicle
        # surrogate has steering/speed lag. Include lag in the braking margin.
        lag_margin=state.speed_m_s*(control['speed_time_constant_s']+
            self.cfg.get('runtime',{}).get('control_dt_s',.05))
        speed=min(control['max_speed_m_s'],math.sqrt(2*control['max_deceleration_m_s2']*
            max(0.,remaining-stop_distance-lag_margin)))
        yaw_error=abs(wrap(points[-1]['yaw_rad']-state.yaw_rad))
        valid_pose=distance_goal<=gates['parking_position_tolerance_m'] and yaw_error<=gates['parking_yaw_tolerance_rad']
        final_along=(state.x_m-points[-1]['x_m'])*math.cos(points[-1]['yaw_rad'])+(state.y_m-points[-1]['y_m'])*math.sin(points[-1]['yaw_rad'])
        if valid_pose:speed=0.
        elif remaining<=stop_distance or (remaining<=look and final_along>gates['parking_position_tolerance_m']):
            # Forward-only steering cannot fix a missed terminal heading by
            # creeping forever or rotating at zero speed. Stop and report it.
            self.failure_reason='forward_path_terminal_pose_mismatch';return 0.,0.
        return speed,clip(desired_steer,-self.cfg['vehicle']['max_steer_rad'],self.cfg['vehicle']['max_steer_rad'])
