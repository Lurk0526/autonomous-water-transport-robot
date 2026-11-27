"""SE(2) parking changes represented as target and fixture poses in arm-base coordinates."""
import math
import numpy as np


def world_to_base(point_world,parking_xy_yaw):
    point=np.asarray(point_world,dtype=float);parking=np.asarray(parking_xy_yaw,dtype=float)
    if point.shape!=(3,) or parking.shape!=(3,) or not np.isfinite([*point,*parking]).all():
        raise ValueError('Expected finite point and parking pose')
    c,s=math.cos(parking[2]),math.sin(parking[2])
    r=np.array([[c,s,0],[-s,c,0],[0,0,1]])
    return r@(point-np.array([parking[0],parking[1],0.]))


def sample_parking(rng,config):
    """Ranges are experiment settings, not estimates of the user's parking distribution."""
    return np.array([rng.uniform(*config['x_m']),rng.uniform(*config['y_m']),
                     math.radians(rng.uniform(*config['yaw_deg']))])
