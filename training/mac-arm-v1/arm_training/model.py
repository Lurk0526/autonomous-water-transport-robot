"""CAD-derived candidate geometry and explicit contact approximations for CPU simulation."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT=Path(__file__).resolve().parents[1]
JOINTS=['base_yaw','shoulder','elbow','wrist_pitch','tool_roll']


def config(path=None):
    result=yaml.safe_load(Path(path or ROOT/'config/train.yaml').read_text())
    if result['simulation']['dt_s']<=0 or result['simulation']['control_dt_s']<=0:raise ValueError('Invalid timestep')
    ratio=result['simulation']['control_dt_s']/result['simulation']['dt_s']
    if abs(ratio-round(ratio))>1e-8:raise ValueError('Control timestep must contain an integer number of physics steps')
    return result


def text(values):return ' '.join(f'{float(v):.12g}' for v in np.asarray(values).flatten())
def quaternion(matrix):
    x,y,z,w=Rotation.from_matrix(matrix).as_quat();return [w,x,y,z]


def make_model(cfg,*,visual=True):
    cad=json.loads((ROOT/'assets/candidate-model.json').read_text())
    root=ET.Element('mujoco',model=cad['identity'])
    def add(parent,tag,**args):return ET.SubElement(parent,tag,{k:str(v) for k,v in args.items()})
    add(root,'compiler',angle='radian',autolimits='true',inertiafromgeom='false')
    add(root,'option',timestep=cfg['simulation']['dt_s'],integrator='implicitfast',cone='pyramidal',iterations=80)
    add(root,'size',njmax=4000,nconmax=1000)
    defaults=add(root,'default');add(defaults,'joint',damping='.4',armature='.004')
    add(defaults,'geom',condim=3,friction='.5 .001 .0001',solref='.006 1',solimp='.95 .99 .001')
    assets=add(root,'asset')
    selected_parts=cad['parts'] if visual else [p for p in cad['parts'] if p['name'].startswith('指')]
    for p in selected_parts:add(assets,'mesh',name=p['id'],file=str(ROOT/'assets/cad/meshes'/p['mesh']))
    b=cfg['bowl'];profile=b['radial_profile_m'];sectors=int(b['sectors'])
    for band,((r0,z0),(r1,z1)) in enumerate(zip(profile[:-1],profile[1:])):
        for sector in range(sectors):
            a0=2*np.pi*sector/sectors;a1=2*np.pi*(sector+1)/sectors;vertices=[]
            for radius,z in [(r0,z0),(r1,z1)]:
                for rr in [max(.001,radius-b['wall_thickness_m']),radius]:
                    vertices.extend([[rr*np.cos(a0),rr*np.sin(a0),z],[rr*np.cos(a1),rr*np.sin(a1),z]])
            add(assets,'mesh',name=f'bowl_band_{band}_{sector}',vertex=text(vertices))
    world=add(root,'worldbody');add(world,'light',pos='0 -1 2',dir='0 0 -1')
    add(world,'geom',name='floor',type='plane',size='2 2 .1',rgba='.12 .17 .21 1')
    platform=add(world,'body',name='platform_body',mocap='true',pos=text([.2,0,cfg['fixture']['top_height_m']/2]))
    add(platform,'geom',name='platform',type='box',size=text([*cfg['fixture']['half_size_m'],cfg['fixture']['top_height_m']/2]),rgba='.55 .39 .23 1')
    bodies={'mount':add(world,'body',name='mount',pos=text([0,0,cfg['arm']['base_height_m']]))}
    previous=np.zeros(3);parent=bodies['mount'];pivots={}
    for i,j in enumerate(cad['joints']):
        point=np.array(j['point_m']);body=add(parent,'body',name=j['name'],pos=text(point-previous))
        add(body,'inertial',pos='0 0 0',mass=cfg['arm']['masses_kg'][i],diaginertia='.001 .001 .001')
        add(body,'joint',name=j['name'],axis=text(j['axis']),range=text(cfg['arm']['joint_range_rad']))
        bodies[j['name']]=body;pivots[j['name']]=point;parent=body;previous=point
    tool_point=np.array(cad['joints'][-1]['point_m'])
    add(bodies['tool_roll'],'site',name='tcp',pos=text(np.array(cad['tcp_m'])-tool_point),quat=text(quaternion(cad['tool_rotation'])),size='.004',rgba='0 .9 .7 1')
    for i,axis in enumerate(cad['finger_closure_axes']):
        body=add(bodies['tool_roll'],'body',name=f'finger_{i}')
        add(body,'inertial',pos=text(np.array(cad['tcp_m'])-tool_point),mass='.04',diaginertia='.0001 .0001 .0001')
        add(body,'joint',name=f'closure_{i}',type='slide',axis=text(axis),range='0 .055',damping=1.,armature='.001')
        bodies[f'finger_{i}']=body;pivots[f'finger_{i}']=tool_point
    parts={p['id']:p for p in cad['parts']}
    for o in cad['occurrences']:
        part=parts[o['part_id']];body_name=o['body'];T=np.array(o['canonical_pose_m'])
        pivot=pivots.get(body_name,np.zeros(3));position=T[:3,3]-pivot;rotation=T[:3,:3]
        if visual:
            color='.85 .53 .23 1'
            if body_name.startswith('finger'):color='.3 .65 .55 1'
            if 'Cam' in o['path'] or '摄像' in o['path']:color='.3 .5 .8 1'
            add(bodies[body_name],'geom',name=o['id']+'_visual',type='mesh',mesh=o['part_id'],pos=text(position),quat=text(quaternion(rotation)),contype=0,conaffinity=0,group=2,rgba=color)
        if o['name']=='防撞缓冲垫':
            center=position+rotation@np.array([0,0,.0015])
            add(bodies[body_name],'geom',name=o['id']+'_foam',type='box',size='.010 .010 .0015',pos=text(center),quat=text(quaternion(rotation)),
                friction=text([cfg['gripper']['foam_friction'][1],.001,.0001]),solref=text([cfg['gripper']['foam_time_constant_s'],cfg['gripper']['foam_damping_ratio']]),solimp='.9 .95 .001',rgba='.2 .24 .2 .5',group=3)
        elif o['name'].startswith('指'):
            add(bodies[body_name],'geom',name=o['id']+'_hard',type='mesh',mesh=o['part_id'],pos=text(position),quat=text(quaternion(rotation)),friction='.1 .001 .0001',rgba='.2 .2 .2 .1',group=3)
    camera=next(o for o in cad['occurrences'] if o['name']=='Lens')
    camera_pose=np.array(camera['canonical_pose_m']);camera_part=parts[camera['part_id']]
    camera_center=camera_pose[:3,:3]@np.array(camera_part['center_of_volume_m'])+camera_pose[:3,3]
    add(bodies['wrist_pitch'],'site',name='camera_mount_reference',pos=text(camera_center-pivots['wrist_pitch']),size='.003',rgba='.2 .5 1 1')
    # Conservative arm collision capsules, separate from detailed display meshes.
    jp=[np.array(j['point_m']) for j in cad['joints']]
    for index in range(1,4):
        add(bodies[JOINTS[index]],'geom',name=JOINTS[index]+'_collision',type='capsule',fromto=text([[0,0,0],jp[index+1]-jp[index]]),size='.014',rgba='.8 .4 .2 .15',group=3)
    bowl=add(world,'body',name='bowl',pos=text([.2,0,cfg['fixture']['top_height_m']]))
    add(bowl,'freejoint',name='bowl_free')
    mass=b['total_mass_kg'];radius=b['rim_radius_m'];height=b['height_m']
    add(bowl,'inertial',pos=text([0,0,height/2]),mass=mass,diaginertia=text([mass*(3*radius*radius+height*height)/12]*2+[mass*radius*radius/2]))
    add(bowl,'geom',name='bowl_foot',type='cylinder',size='.03 .003',pos='0 0 .003',rgba='.06 .07 .08 1')
    for band in range(len(profile)-1):
        for sector in range(sectors):add(bowl,'geom',name=f'bowl_{band}_{sector}',type='mesh',mesh=f'bowl_band_{band}_{sector}',rgba='.06 .07 .08 1')
    equal=add(root,'equality')
    for i in range(1,4):add(equal,'joint',joint1=f'closure_{i}',joint2='closure_0',polycoef='0 1 0 0 0',solref='.004 1')
    actuators=add(root,'actuator')
    for name,limit in zip(JOINTS,cfg['arm']['torque_limits_nm']):
        add(actuators,'position',name=name+'_actuator',joint=name,kp=cfg['arm']['kp'],kv=cfg['arm']['kv'],ctrlrange=text(cfg['arm']['joint_range_rad']),forcerange=text([-limit,limit]))
    add(actuators,'position',name='gripper_actuator',joint='closure_0',kp=1500,kv=15,ctrlrange='0 .055',forcerange=text([-cfg['gripper']['force_limit_n'],cfg['gripper']['force_limit_n']]))
    xml=ET.tostring(root,encoding='unicode')
    model=mujoco.MjModel.from_xml_string(xml)
    return model,xml,cad


class IK:
    def __init__(self,model):
        self.model=model;self.data=mujoco.MjData(model);self.site=model.site('tcp').id
        self.qids=np.array([model.joint(j).qposadr[0] for j in JOINTS]);self.dofs=np.array([model.joint(j).dofadr[0] for j in JOINTS])
        self.ranges=np.array([model.joint(j).range for j in JOINTS])
    def solve(self,target,seed=None,iterations=100):
        q=np.zeros(5) if seed is None else np.asarray(seed).copy();target=np.asarray(target,dtype=float)
        jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp)
        for _ in range(iterations):
            self.data.qpos[self.qids]=q;mujoco.mj_forward(self.model,self.data)
            pos=self.data.site_xpos[self.site];axis=self.data.site_xmat[self.site].reshape(3,3)[:,1]
            error=np.r_[target-pos,.12*(np.array([0,0,1])-axis)]
            if np.linalg.norm(error[:3])<.0005 and np.linalg.norm(error[3:])<.003:return q,True
            mujoco.mj_jacSite(self.model,self.data,jp,jr,self.site)
            skew=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
            J=np.vstack([jp[:,self.dofs],-.12*skew@jr[:,self.dofs]])
            step=J.T@np.linalg.solve(J@J.T+np.eye(6)*1e-5,error)
            q=np.clip(q+np.clip(step,-.15,.15),self.ranges[:,0]+.01,self.ranges[:,1]-.01)
        self.data.qpos[self.qids]=q;mujoco.mj_forward(self.model,self.data)
        return q,bool(np.linalg.norm(self.data.site_xpos[self.site]-target)<.005 and np.linalg.norm(error[3:])<.01)
