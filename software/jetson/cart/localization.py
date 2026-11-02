"""Planar wheel-speed/gyro odometry with covariance and optional absolute anchors.

This is a testable baseline for later real inputs, not a running localization
system. Without anchors it has an arbitrary origin and accumulates drift.
"""
import math
from .feedback import finite


def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


def transpose(a):
    return [list(row) for row in zip(*a)]


def matmul(a, b):
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]


def add(a,b):
    return [[x+y for x,y in zip(ar,br)] for ar,br in zip(a,b)]


def inverse(a):
    n=len(a)
    aug=[list(row)+[float(i==j) for j in range(n)] for i,row in enumerate(a)]
    for i in range(n):
        pivot=max(range(i,n),key=lambda j:abs(aug[j][i]))
        if abs(aug[pivot][i])<1e-15:
            raise ValueError("singular covariance")
        aug[i],aug[pivot]=aug[pivot],aug[i]
        scale=aug[i][i]
        aug[i]=[v/scale for v in aug[i]]
        for j in range(n):
            if j!=i:
                ratio=aug[j][i]
                aug[j]=[v-ratio*w for v,w in zip(aug[j],aug[i])]
    return [r[n:] for r in aug]


class PlanarLocalization:
    def __init__(self, speed_sigma=.03, gyro_sigma=.02):
        if not all(finite(v) and v>0 for v in (speed_sigma,gyro_sigma)):
            raise ValueError("noise sigmas must be positive")
        self.pose=[0.0,0.0,0.0]
        self.cov=[[.0001,0,0],[0,.0001,0],[0,0,.0004]]
        self.last_time=None
        self.speed_sigma,self.gyro_sigma=speed_sigma,gyro_sigma
        self.has_absolute_anchor=False
        self.last_anchor_time=None
        self.faults=[]

    def predict(self, stamp_s, speed_m_s, gyro_z_rad_s):
        if not all(finite(v) for v in (stamp_s,speed_m_s,gyro_z_rad_s)) or stamp_s<0:
            self.faults.append("nonfinite_input");return False
        if abs(speed_m_s)>2 or abs(gyro_z_rad_s)>3:
            self.faults.append("input_out_of_range");return False
        if self.last_time is None:
            self.last_time=stamp_s;return True
        dt=stamp_s-self.last_time
        if not 0<dt<=.25:
            self.faults.append("time_discontinuity");return False
        self.last_time=stamp_s
        theta=self.pose[2]+gyro_z_rad_s*dt/2
        c,s=math.cos(theta),math.sin(theta)
        self.pose[0]+=speed_m_s*c*dt
        self.pose[1]+=speed_m_s*s*dt
        self.pose[2]=wrap(self.pose[2]+gyro_z_rad_s*dt)
        f=[[1,0,-speed_m_s*s*dt],[0,1,speed_m_s*c*dt],[0,0,1]]
        g=[[c*dt,0],[s*dt,0],[0,dt]]
        q=[[self.speed_sigma**2,0],[0,self.gyro_sigma**2]]
        process=matmul(matmul(g,q),transpose(g))
        # Unmodelled slip/bias grows uncertainty even with many tiny samples.
        for i,var in enumerate((.015**2,.015**2,.005**2)):
            process[i][i]+=var*dt
        self.cov=add(matmul(matmul(f,self.cov),transpose(f)),process)
        return True

    def anchor(self, stamp_s, pose, position_sigma_m, yaw_sigma_rad):
        values=[stamp_s,*pose,position_sigma_m,yaw_sigma_rad]
        if len(pose)!=3 or not all(finite(v) for v in values) or min(position_sigma_m,yaw_sigma_rad)<=0:
            self.faults.append("anchor_invalid");return False
        if self.last_time is None or abs(stamp_s-self.last_time)>.05:
            self.faults.append("anchor_not_time_aligned");return False
        if self.last_anchor_time is not None and stamp_s<=self.last_anchor_time:
            self.faults.append("anchor_replayed");return False
        r=[[position_sigma_m**2,0,0],[0,position_sigma_m**2,0],[0,0,yaw_sigma_rad**2]]
        innovation=[pose[i]-self.pose[i] for i in range(3)]
        innovation[2]=wrap(innovation[2])
        inv_s=inverse(add(self.cov,r))
        mahal=sum(innovation[i]*sum(inv_s[i][j]*innovation[j] for j in range(3)) for i in range(3))
        if mahal>16:
            self.faults.append("anchor_outlier");return False
        k=matmul(self.cov,inv_s)
        correction=[sum(k[i][j]*innovation[j] for j in range(3)) for i in range(3)]
        self.pose=[self.pose[i]+correction[i] for i in range(3)]
        self.pose[2]=wrap(self.pose[2])
        ik=[[float(i==j)-k[i][j] for j in range(3)] for i in range(3)]
        self.cov=add(matmul(matmul(ik,self.cov),transpose(ik)),matmul(matmul(k,r),transpose(k)))
        self.has_absolute_anchor=True;self.last_anchor_time=stamp_s
        return True

    def snapshot(self):
        return {"pose":list(self.pose),"covariance":[list(r) for r in self.cov],
                "pose_sigma_m":math.sqrt(max(self.cov[0][0],self.cov[1][1])),
                "yaw_sigma_rad":math.sqrt(max(0,self.cov[2][2])),
                "origin":"anchored" if self.has_absolute_anchor else "arbitrary_odometry_origin",
                "faults":self.faults[-30:]}
