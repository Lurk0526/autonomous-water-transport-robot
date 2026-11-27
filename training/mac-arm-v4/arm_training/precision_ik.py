"""Pose-continuous IK for vertical bowl handling, with a retained tool orientation."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .model import IK


class PrecisionIK(IK):
    def __init__(self,model):super().__init__(model);self.orientation_target=None
    def solve(self,target,seed=None,iterations=120):
        upright_only=self.orientation_target is None
        target=np.asarray(target,dtype=float);q=np.zeros(5) if seed is None else np.asarray(seed).copy()
        jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp);weight=.12
        for _ in range(iterations):
            self.data.qpos[self.qids]=q;self._forward()
            position=target-self.data.site_xpos[self.site]
            current=self.data.site_xmat[self.site].reshape(3,3)
            axis=current[:,1]
            angle=np.array([0.,0.,1.])-axis if upright_only else Rotation.from_matrix(self.orientation_target@current.T).as_rotvec()
            if np.linalg.norm(position)<.00003 and np.linalg.norm(angle)<.0008:return q,True
            mujoco.mj_jacSite(self.model,self.data,jp,jr,self.site)
            if upright_only:
                skew=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
                J=np.vstack([jp[:,self.dofs],-weight*skew@jr[:,self.dofs]])
            else:J=np.vstack([jp[:,self.dofs],weight*jr[:,self.dofs]])
            error=np.r_[position,weight*angle]
            delta=J.T@np.linalg.solve(J@J.T+np.eye(6)*1e-7,error)
            # Scale a coordinated Newton direction, never clip each joint differently.
            delta*=min(1.,.08/max(float(np.max(np.abs(delta))),1e-12))
            old_cost=float(error@error);accepted=False
            for fraction in (1.,.5,.25,.125,.0625):
                candidate=np.clip(q+fraction*delta,self.ranges[:,0]+.01,self.ranges[:,1]-.01)
                residual=self._residual(candidate,target,upright_only,weight)
                if float(residual@residual)<old_cost:
                    q=candidate;accepted=True;break
            if not accepted:break
        # Near a kinematic singularity, bounded least squares is a numerical
        # fallback. The physical reachability tolerances stay unchanged.
        from scipy.optimize import least_squares
        fit=least_squares(lambda values:self._residual(values,target,upright_only,weight),q,
            bounds=(self.ranges[:,0]+.01,self.ranges[:,1]-.01),jac='3-point',
            x_scale='jac',ftol=1e-11,xtol=1e-11,gtol=1e-11,max_nfev=100)
        q=fit.x;residual=self._residual(q,target,upright_only,weight)
        return q,bool(np.linalg.norm(residual[:3])<.00015 and np.linalg.norm(residual[3:])/weight<.003)

    def _residual(self,q,target,upright_only,weight):
        self.data.qpos[self.qids]=q;self._forward()
        current=self.data.site_xmat[self.site].reshape(3,3)
        rotation=np.array([0.,0.,1.])-current[:,1] if upright_only else Rotation.from_matrix(self.orientation_target@current.T).as_rotvec()
        return np.r_[np.asarray(target)-self.data.site_xpos[self.site],weight*rotation]
