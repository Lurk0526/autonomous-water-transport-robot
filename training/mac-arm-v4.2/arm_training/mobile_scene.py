"""Mac-rendered dual-camera scene and sampled kinematic screening.

Vehicle motion is planar bicycle kinematics; arm grasp physics runs separately
in BowlEnv after a stationary handoff. This is not calibrated full-car dynamics.
Camera rotations are synthetic calibration assumptions, never real extrinsics.
No physics, inference or local effect validation is performed by importing it.
"""
import copy
import math
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .model import make_model,JOINTS,IK,text,quaternion
from .precision_ik import PrecisionIK
from .mobile_planner import compose_pose,point_in_model


def scene_model(cfg,visual=False):
    return make_model(cfg,visual=visual)


def optical_look_at(eye,target,up=(0.,0.,1.)):
    eye=np.asarray(eye,float);target=np.asarray(target,float);up=np.asarray(up,float)
    if eye.shape!=(3,) or target.shape!=(3,) or up.shape!=(3,) or not np.isfinite(np.r_[eye,target,up]).all():
        raise ValueError('Camera eye, target and up must be finite three-vectors')
    forward=target-eye
    if np.linalg.norm(forward)<1e-8:raise ValueError('Camera eye and target coincide')
    forward/=np.linalg.norm(forward);right=np.cross(forward,up)
    if np.linalg.norm(right)<1e-8:raise ValueError('Degenerate camera up')
    right/=np.linalg.norm(right);down=np.cross(forward,right)
    t=np.eye(4);t[:3,:3]=np.column_stack([right,down,forward]);t[:3,3]=eye
    return t


def pose4(xy_yaw,z=0.):
    xy_yaw=np.asarray(xy_yaw,float)
    if xy_yaw.shape!=(3,) or not np.isfinite(xy_yaw).all() or not np.isfinite(z):raise ValueError('Invalid planar pose')
    c,s=np.cos(xy_yaw[2]),np.sin(xy_yaw[2]);t=np.eye(4)
    t[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];t[:3,3]=[*xy_yaw[:2],z];return t


def _set_platform(model,data,mocap_id,table):
    """Update one box prop's actual dimensions and direct broadphase bounds.

    ArmScreen disables midphase BVH traversal because compiled BVH bounds are
    otherwise stale after changing geom_size. The render-only scene never steps
    this prop dynamically. This helper must not be used to resize dynamic bodies.
    """
    center=np.asarray(table['center_xy_m'],float);half=np.asarray(table['half_size_m'],float)
    height=float(table['top_height_m']);yaw=float(table['yaw_rad'])
    if center.shape!=(2,) or half.shape!=(2,) or not np.isfinite(np.r_[center,half,height,yaw]).all() or min(*half,height)<=0:
        raise ValueError('Invalid table dimensions/pose')
    gid=model.geom('platform').id
    if int(model.geom_type[gid])!=int(mujoco.mjtGeom.mjGEOM_BOX):raise ValueError('Platform must be a box')
    size=np.r_[half,height/2]
    model.geom_size[gid]=size;model.geom_rbound[gid]=np.linalg.norm(size)
    model.geom_aabb[gid,:3]=0.;model.geom_aabb[gid,3:]=size
    data.mocap_pos[mocap_id]=[*center,height/2]
    data.mocap_quat[mocap_id]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]


class MobileScene:
    def __init__(self,arm_cfg,mobile_cfg):
        self.cfg=copy.deepcopy(arm_cfg);self.mobile=copy.deepcopy(mobile_cfg)
        _,xml,self.cad=scene_model(self.cfg,visual=True)
        root=ET.fromstring(xml);world=root.find('worldbody')
        mount=world.find("body[@name='mount']");mount.set('mocap','true')
        v=mobile_cfg['vehicle'];front=v['wheelbase_m']+v['front_overhang_m'];rear=v['rear_overhang_m']
        vehicle=ET.SubElement(world,'body',{'name':'mobile_chassis','mocap':'true'})
        ET.SubElement(vehicle,'geom',{'name':'mobile_chassis_display','type':'box','size':text([(front+rear)/2,v['width_m']/2,v['height_m']/2]),
            'pos':text([(front-rear)/2,0,v['height_m']/2]),'rgba':'.12 .3 .5 .8','contype':'0','conaffinity':'0'})
        for x in [0.,v['wheelbase_m']]:
            for y in [-v['width_m']/2,v['width_m']/2]:
                ET.SubElement(vehicle,'geom',{'type':'cylinder','size':'.045 .012','pos':text([x,y,.045]),'euler':'1.57079632679 0 0',
                    'rgba':'.06 .07 .08 1','contype':'0','conaffinity':'0'})
        cameras=mobile_cfg['render_cameras'];self.width=int(cameras['width']);self.height=int(cameras['height'])
        main=cameras['main'];tmain=optical_look_at(main['position_in_vehicle_m'],main['look_at_in_vehicle_m'])
        tmain[:3,:3]=tmain[:3,:3]@Rotation.from_euler('xyz',main.get('mount_rpy_deg',[0.,0.,0.]),degrees=True).as_matrix()
        ET.SubElement(vehicle,'camera',{'name':'main_rgbd','pos':text(tmain[:3,3]),
            'quat':text(quaternion(tmain[:3,:3]@np.diag([1,-1,-1]))),'fovy':str(main['fovy_deg'])})
        # Compute one fixed mount rotation in a declared nominal observation pose,
        # rather than using the actual bowl location to turn the wrist camera.
        temp_model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));temp=mujoco.MjData(temp_model)
        nominal=np.asarray(cameras['wrist']['nominal_workspace_target_m']);ik=IK(temp_model)
        # The optical installation is frozen in a declared calibration posture.
        # Changing a search posture must not secretly retarget camera look_at.
        calibration_height=cameras['wrist'].get('mount_reference_tcp_above_plane_m',.095)
        reference_q,ok=ik.solve(nominal+[0,0,calibration_height])
        if not ok:raise ValueError('Nominal observation arm pose is unreachable')
        qids=[int(temp_model.joint(j).qposadr[0]) for j in JOINTS];temp.qpos[qids]=reference_q
        mujoco.mj_forward(temp_model,temp);ref=temp_model.site('camera_mount_reference').id;parent=temp_model.body('wrist_pitch').id
        eye=temp.site_xpos[ref];topt=optical_look_at(eye,nominal+[0,0,self.cfg['bowl']['height_m']/2])
        parent_r=temp.xmat[parent].reshape(3,3);local_r=parent_r.T@topt[:3,:3]
        local_p=parent_r.T@(eye-temp.xpos[parent]);wrist=root.find(".//body[@name='wrist_pitch']")
        local_p+=np.asarray(cameras['wrist'].get('mount_position_offset_in_support_m',[0.,0.,0.]),float)
        local_p+=local_r@np.asarray(cameras['wrist'].get('mount_position_offset_optical_m',[0.,0.,0.]),float)
        local_r=local_r@Rotation.from_euler('xyz',cameras['wrist'].get('mount_rpy_deg',[0.,0.,0.]),degrees=True).as_matrix()
        ET.SubElement(wrist,'camera',{'name':'wrist','pos':text(local_p),'quat':text(quaternion(local_r@np.diag([1,-1,-1]))),
            'fovy':str(cameras['wrist']['fovy_deg'])})
        visual=root.find('visual')
        if visual is None:visual=ET.SubElement(root,'visual')
        global_visual=visual.find('global')
        if global_visual is None:global_visual=ET.SubElement(visual,'global')
        global_visual.set('offwidth',str(self.width));global_visual.set('offheight',str(self.height))
        self.model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));self.data=mujoco.MjData(self.model)
        self.mount=int(self.model.body('mount').mocapid[0]);self.chassis=int(self.model.body('mobile_chassis').mocapid[0])
        self.platform=int(self.model.body('platform_body').mocapid[0]);self.free=int(self.model.joint('bowl_free').qposadr[0])
        self.joint_names=JOINTS+[self.model.joint(i).name for i in range(self.model.njnt)
            if self.model.joint(i).name not in JOINTS and int(self.model.jnt_type[i]) in
            (int(mujoco.mjtJoint.mjJNT_HINGE),int(mujoco.mjtJoint.mjJNT_SLIDE))]
        self.ids=np.array([int(self.model.joint(j).qposadr[0]) for j in self.joint_names])
        observation_height=cameras['wrist'].get('observation_tcp_above_plane_m',.14)
        observation_offset=np.asarray(cameras['wrist'].get('observation_xy_offset_m',[0.,0.]),float)
        self.observation_q,ok=ik.solve(nominal+np.r_[observation_offset,observation_height],seed=reference_q)
        if not ok:raise ValueError('Configured wrist observation pose is unreachable')
        self.navigation_q=self.observation_q.copy()
        if 'navigation_tcp_above_plane_m' in mobile_cfg['arm']:
            self.navigation_q,ok=ik.solve(nominal+[0.,0.,mobile_cfg['arm']['navigation_tcp_above_plane_m']],seed=reference_q)
            if not ok:raise ValueError('Configured navigation arm pose is unreachable')
        self.navigation_q[0]+=math.radians(mobile_cfg['arm'].get('navigation_base_yaw_offset_deg',0.))
        if np.any(self.navigation_q<ik.ranges[:,0]) or np.any(self.navigation_q>ik.ranges[:,1]):raise ValueError('Navigation arm joints outside limits')
        self.data.qpos[self.ids[:5]]=self.observation_q;self.renderer=None;self.vehicle_pose=np.zeros(3)
        # arm_model shares world Z=0 with BowlEnv: its mount already includes base_height_m.
        self.arm_pose=np.zeros(3);self.arm_pose4=np.eye(4)
        self.table=None;self.bowl_position=None

    def set_scenario(self,table,bowl_position,vehicle_pose):
        self.table=copy.deepcopy(table);self.bowl_position=np.asarray(bowl_position,float).copy()
        _set_platform(self.model,self.data,self.platform,table)
        self.data.qpos[self.free:self.free+7]=[*self.bowl_position,1,0,0,0]
        self.data.qpos[self.ids[:5]]=self.observation_q;self.data.qpos[self.ids[5:]]=0.
        self.set_vehicle(vehicle_pose)

    def set_vehicle(self,pose):
        self.vehicle_pose=np.asarray(pose,float).copy()
        if self.vehicle_pose.shape!=(3,) or not np.isfinite(self.vehicle_pose).all():raise ValueError('Invalid vehicle pose')
        arm=compose_pose(self.vehicle_pose,self.mobile['arm']['mount_xy_yaw_m_rad'])
        self.arm_pose=arm;self.arm_pose4=pose4(arm)
        self.data.mocap_pos[self.chassis]=[*pose[:2],0.]
        self.data.mocap_quat[self.chassis]=[np.cos(pose[2]/2),0,0,np.sin(pose[2]/2)]
        self.data.mocap_pos[self.mount]=[*arm[:2],self.cfg['arm']['base_height_m']]
        self.data.mocap_quat[self.mount]=[np.cos(arm[2]/2),0,0,np.sin(arm[2]/2)]
        mujoco.mj_forward(self.model,self.data)

    def camera(self,name):
        # MuJoCo camera: x right, y up, -z forward. Optical: x right, y down, +z forward.
        cid=self.model.camera(name).id;t=np.eye(4)
        t[:3,:3]=self.data.cam_xmat[cid].reshape(3,3)@np.diag([1,-1,-1]);t[:3,3]=self.data.cam_xpos[cid]
        fy=self.height/(2*np.tan(np.deg2rad(float(self.model.cam_fovy[cid]))/2))
        k=np.array([[fy,0,(self.width-1)/2],[0,fy,(self.height-1)/2],[0,0,1.]])
        return k,t

    def set_arm_joints(self,q):
        q=np.asarray(q,float)
        if q.shape!=(len(JOINTS),) or not np.isfinite(q).all():raise ValueError('Invalid arm joint vector')
        ranges=np.array([self.model.joint(name).range for name in JOINTS])
        if np.any(q<ranges[:,0]) or np.any(q>ranges[:,1]):raise ValueError('Arm joint range exceeded')
        self.data.qpos[self.ids[:5]]=q;mujoco.mj_forward(self.model,self.data)

    set_navigation_joints=set_arm_joints

    def wrist_search_candidates(self,platform,offsets=None):
        """Real arm postures around the estimated platform; no bowl truth input."""
        if platform.get('valid') is not True or platform.get('kind')!='platform':raise ValueError('Detected platform required')
        relative=point_in_model([*platform['center_xy_m'],platform['top_height_m']],self.arm_pose)
        settings=self.mobile['render_cameras']['wrist']
        choices=offsets if offsets is not None else settings.get('search_offsets_m',
            [[0.,0.,.14],[0.,.04,.14],[0.,-.04,.14],[.025,0.,.16],[-.025,0.,.16]])
        ik=IK(self.model);result=[]
        for index,offset in enumerate(choices):
            offset=np.asarray(offset,float)
            if offset.shape!=(3,) or not np.isfinite(offset).all():raise ValueError('Invalid wrist search offset')
            q,ok=ik.solve(relative+offset,seed=self.observation_q)
            result.append({'view_id':index,'q_rad':q.tolist(),'valid':bool(ok),
                'target_tcp_model_m':(relative+offset).tolist(),'search_offset_m':offset.tolist(),
                'search_type':'whole_arm_joint_posture_fixed_optical_mount',
                'camera_rotated_by_servo004':False})
        return result

    def wrist_camera_pose_in_model(self,q):
        """Joint-dependent optical pose for planner FOV screening, without a target.

        Restore scene generalized positions and derived transforms even if camera
        extraction fails. Do not subtract base_height_m: it is part of BowlEnv's
        kinematic model, while arm_pose4 represents only the planar odom offset.
        """
        q=np.asarray(q,float)
        if q.shape!=(len(JOINTS),) or not np.isfinite(q).all():raise ValueError('Invalid arm joints')
        saved=self.data.qpos.copy()
        try:
            self.data.qpos[self.ids[:len(JOINTS)]]=q;mujoco.mj_forward(self.model,self.data)
            _,optical_to_odom=self.camera('wrist')
            return np.linalg.inv(self.arm_pose4)@optical_to_odom
        finally:
            self.data.qpos[:]=saved;mujoco.mj_forward(self.model,self.data)

    def render(self,name,with_depth=False):
        if self.renderer is None:self.renderer=mujoco.Renderer(self.model,height=self.height,width=self.width)
        self.renderer.disable_depth_rendering();self.renderer.update_scene(self.data,camera=name);rgb=self.renderer.render().copy()
        depth=None
        if with_depth:
            # MuJoCo 3.3.7 Renderer undoes the OpenGL projection into metric optical-Z,
            # NOT Euclidean ray range. RGB and depth share the same projection/pixels.
            self.renderer.enable_depth_rendering()
            try:self.renderer.update_scene(self.data,camera=name);depth=self.renderer.render().copy()
            finally:self.renderer.disable_depth_rendering()
        return rgb,depth,*self.camera(name)

    def joint_values(self):return {name:float(self.data.qpos[idx]) for name,idx in zip(self.joint_names,self.ids)}

    def apply_grasp_state(self,env):
        for name,idx in zip(self.joint_names,self.ids):
            self.data.qpos[idx]=env.data.qpos[int(env.model.joint(name).qposadr[0])]
        local=env.data.qpos[env.free_q:env.free_q+3];world=self.arm_pose4@np.r_[local,1.]
        q=env.data.qpos[env.free_q+3:env.free_q+7]
        local_r=Rotation.from_quat([q[1],q[2],q[3],q[0]])
        global_r=Rotation.from_euler('z',self.arm_pose[2])*local_r;x,y,z,w=global_r.as_quat()
        self.data.qpos[self.free:self.free+7]=[*world[:3],w,x,y,z];mujoco.mj_forward(self.model,self.data)

    def snapshot(self,time_s,stage):
        return {'time_s':time_s,'stage':stage,'vehicle_xy_yaw':self.vehicle_pose.tolist(),'joint_values':self.joint_values(),
            'bowl_world_qpos':self.data.qpos[self.free:self.free+7].tolist()}

    def apply_snapshot(self,row):
        self.set_vehicle(row['vehicle_xy_yaw'])
        for name,idx in zip(self.joint_names,self.ids):self.data.qpos[idx]=row['joint_values'][name]
        self.data.qpos[self.free:self.free+7]=row['bowl_world_qpos'];mujoco.mj_forward(self.model,self.data)

    def close(self):
        if self.renderer is not None:self.renderer.close();self.renderer=None


class ArmScreen:
    """Separate kinematic data for candidate screening; never runs mj_step."""
    def __init__(self,arm_cfg,mobile_cfg):
        self.cfg=arm_cfg;self.mobile=mobile_cfg;self.model,_,cad=scene_model(arm_cfg,visual=False);self.data=mujoco.MjData(self.model)
        self.ik=PrecisionIK(self.model);self.qids=[int(self.model.joint(j).qposadr[0]) for j in JOINTS]
        self.platform=int(self.model.body('platform_body').mocapid[0]);self.free=int(self.model.joint('bowl_free').qposadr[0])
        finger_roots={self.model.body(f'finger_{i}').id for i in range(4)
            if mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,f'finger_{i}')>=0}
        self.finger_bodies=set(finger_roots)
        self.finger_bodies.update(self.model.body(name).id for name in cad.get('linked_gripper',{}).get('body_to_finger',{}))
        for bid in range(self.model.nbody):
            ancestor=bid
            while ancestor>0:
                if ancestor in finger_roots or (self.model.body(ancestor).name or '').startswith('finger_'):
                    self.finger_bodies.add(bid);break
                ancestor=int(self.model.body_parentid[ancestor])
        self.robot_bodies=self.finger_bodies|{self.model.body(name).id for name in JOINTS}
        # Keep collision detection, but bypass compiled BVH bounds when the table
        # dimensions vary. geom_size/rbound/aabb are refreshed by _set_platform.
        self.model.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        self.hard_finger_geoms=[gid for gid in range(self.model.ngeom)
            if int(self.model.geom_bodyid[gid]) in self.finger_bodies
            and (self.model.geom(gid).name or '').endswith('_hard')
            and int(self.model.geom_type[gid])==int(mujoco.mjtGeom.mjGEOM_MESH)
            and (self.model.geom_contype[gid] or self.model.geom_conaffinity[gid])]
        if not self.hard_finger_geoms:raise ValueError('No physical hard-finger meshes for platform screening')

    def solve_grasp(self,point,seed=None):
        # Match RepairedBowlEnv: select home from a declared zero seed, then
        # retain that complete tool orientation throughout the vertical path.
        if seed is None:self.ik.orientation_target=None
        q,ok=self.ik.solve(point,seed=seed)
        if seed is None and ok:
            self.ik.data.qpos[self.ik.qids]=q;mujoco.mj_forward(self.model,self.ik.data)
            self.ik.orientation_target=self.ik.data.site_xmat[self.ik.site].reshape(3,3).copy()
        return q,ok

    def geometry_check(self,q,target,phase,table):
        _set_platform(self.model,self.data,self.platform,table);yaw=table['yaw_rad']
        # The non-interacting bowl is removed from this arm-vs-platform screen.
        self.data.qpos[:]=self.model.qpos0
        self.data.qpos[self.qids]=q
        self.data.qpos[self.free:self.free+3]=[20,20,20]
        mujoco.mj_forward(self.model,self.data)
        for c in self.data.contact[:self.data.ncon]:
            g1,g2=int(c.geom1),int(c.geom2);n1=self.model.geom(g1).name or '';n2=self.model.geom(g2).name or ''
            b1,b2=int(self.model.geom_bodyid[g1]),int(self.model.geom_bodyid[g2])
            if b1 in self.finger_bodies and b2 in self.finger_bodies:continue
            robot=bool({b1,b2}&self.robot_bodies)
            if float(c.dist)<-.0001 and robot:
                return {'valid':False,'reason':'sampled_arm_platform_or_self_collision','clearance_m':float(c.dist),
                    'contact_geoms':[n1,n2]}
        # Conservative hard-finger AABB clearance against the platform top.
        cy,sy=np.cos(-yaw),np.sin(-yaw);r=np.array([[cy,-sy],[sy,cy]])
        clearance=1.
        for gid in self.hard_finger_geoms:
            mid=int(self.model.geom_dataid[gid]);start=int(self.model.mesh_vertadr[mid]);n=int(self.model.mesh_vertnum[mid])
            vertices=self.model.mesh_vert[start:start+n]@self.data.geom_xmat[gid].reshape(3,3).T+self.data.geom_xpos[gid]
            xy=(vertices[:,:2]-np.asarray(table['center_xy_m']))@r.T;half=np.asarray(table['half_size_m'])
            if np.all(xy.min(axis=0)<=half) and np.all(xy.max(axis=0)>=-half):
                clearance=min(clearance,float(vertices[:,2].min()-table['top_height_m']))
        minimum=self.mobile['planning'].get('minimum_hard_finger_clearance_m',.003)
        return {'valid':clearance>=minimum,'reason':None if clearance>=minimum else 'hard_finger_platform_clearance','clearance_m':clearance,
            'hard_finger_geoms_checked':len(self.hard_finger_geoms),
            'clearance_method':'physical_hard_finger_mesh_lower_AABB_envelope_conservative'}
