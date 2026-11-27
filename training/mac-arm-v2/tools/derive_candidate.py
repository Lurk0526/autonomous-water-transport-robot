"""Derive a geometry-based candidate chain; preserve explicit uncalibrated status."""
from pathlib import Path
import json
import re
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def derive():
    cad=json.loads((ROOT/'assets/cad/cad-manifest.json').read_text());parts={p['id']:p for p in cad['parts']}
    objects=cad['occurrences'];rotation=np.array([[0,0,-1],[0,1,0],[1,0,0.]])
    base=next(o for o in objects if o['name']=='PRODUCT_10168');bt=np.array(base['pose_m'])
    origin=bt[:3,:3]@np.array([.0255,.0104,.01016])+bt[:3,3]
    C=np.eye(4);C[:3,:3]=rotation;C[:3,3]=-rotation@origin
    def point(o,local):
        t=C@np.array(o['pose_m']);return t[:3,:3]@local+t[:3,3]
    cases=[o for o in objects if o['name']=='PRODUCT_21630']
    case_info=[]
    for o in cases:
        t=C@np.array(o['pose_m']);axis=t[:3,0]
        p=point(o,np.array([.0255,.0105,.01]))
        case_info.append({'object':o,'point':p,'axis':axis,'prefix':re.sub(r'\[\d+\]$','',o['path'].rsplit('/',1)[0])})
    roll=next(x for x in case_info if abs(x['axis'][2])>.9)
    pitches=sorted([x for x in case_info if x is not roll],key=lambda x:x['point'][0])
    if len(pitches)!=3:raise ValueError('Expected three pitch servo cases')
    joints=[{'name':'base_yaw','point_m':[0,0,0],'axis':[0,0,1],'source':base['id']}]
    for nm,x in zip(['shoulder','elbow','wrist_pitch','tool_roll'],[*pitches,roll]):
        p=x['point'].copy()
        if nm!='tool_roll':p[1]=0. # Shift along the same pitch axis; this does not move its line.
        joints.append({'name':nm,'point_m':p.tolist(),'axis':x['axis'].tolist(),'source':x['object']['id']})
    grip_assembly=next(a for a in cad['assemblies'] if a['name']=='【MG23】装配体')
    grip_pose=C@np.array(grip_assembly['pose_m']);invgrip=np.linalg.inv(grip_pose)
    pads=[o for o in objects if o['name']=='防撞缓冲垫']
    pad_local=[]
    for o in pads:
        p=point(o,np.array([0,0,.0015]));pad_local.append((o,(invgrip@np.r_[p,1])[:3]))
    distal=sorted(pad_local,key=lambda x:x[1][1])[:4]
    axis_center=np.mean([p for _,p in distal],axis=0)
    tcp=(grip_pose@np.r_[axis_center,1])[:3]
    radial_directions=[]
    for _,p in distal:
        v=p-axis_center;v[1]=0;v/=np.linalg.norm(v)
        radial_directions.append(v)
    # Four radially translated rigid finger envelopes, coupled to ONE closure actuator.
    # This is an explicit approximation, not a decoded MG23 linkage law.
    previous=['mount','base_yaw','shoulder','elbow','wrist_pitch']
    prefixes={x['prefix']:(j,previous[j]) for j,x in enumerate([*pitches,roll],start=1)}
    assignment=[]
    points=np.array([j['point_m'] for j in joints])
    def segment_distance(p,a,b):
        v=b-a;u=np.clip(np.dot(p-a,v)/max(np.dot(v,v),1e-12),0,1)
        return np.linalg.norm(p-a-u*v)
    for o in objects:
        pose=C@np.array(o['pose_m']);p=pose[:3,:3]@np.array(parts[o['part_id']]['center_of_volume_m'])+pose[:3,3]
        group=None;claw=None
        if '【MG23】装配体' in o['path']:
            if o['name'].startswith(('指','拉杆')) or o['name']=='防撞缓冲垫':
                pgrip=(invgrip@np.r_[p,1])[:3]-axis_center;pgrip[1]=0
                claw=int(np.argmax([np.dot(pgrip,v) for v in radial_directions]));group=f'finger_{claw}'
            else:group='tool_roll'
        elif o['name']=='夹爪与机械臂连接件':group='tool_roll'
        elif any(s in o['path'] for s in ['摄像头stm32板','CamAssembly']):group='wrist_pitch'
        elif 'KSZ001_ASM' in o['path'] or o['name'] in ['J1DB001','PRODUCT_10168']:group='mount'
        else:
            prefix=re.sub(r'\[\d+\]$','',o['path'].rsplit('/',1)[0])
            if prefix in prefixes and o['name'] in ['PRODUCT_21630','PRODUCT_68796']:
                j,upstream=prefixes[prefix];group=upstream if o['name']=='PRODUCT_21630' else joints[j]['name']
            else:
                distances=[segment_distance(p,points[i],points[i+1]) for i in range(4)]
                group=['base_yaw','shoulder','elbow','wrist_pitch'][int(np.argmin(distances))]
        assignment.append({**o,'body':group,'claw':claw,'canonical_pose_m':pose.tolist()})
    result={'identity':'user_step_geometry_with_candidate_servo_chain_and_mg23_envelope',
        'source_sha256':cad['source_sha256'],'canonical_from_cad':C.tolist(),'joints':joints,
        'tcp_m':tcp.tolist(),'tool_rotation':grip_pose[:3,:3].tolist(),
        'finger_closure_axes':[(-grip_pose[:3,:3]@v).tolist() for v in radial_directions],
        'occurrences':assignment,'parts':cad['parts'],
        'status':{'cad_geometry_valid':all(p['brep_valid'] for p in cad['parts']),
                  'joint_axes':'candidate axes derived from CAD servo-shaft cylinders; sign/zero/range require hardware calibration',
                  'rigid_group_assignment':'motor cases/horns and camera assigned explicitly; remaining arm pieces assigned by nearest bone and require review',
                  'gripper':'four rigid finger envelopes with a single coupled radial closure; actual linkage and compliant finger motion NOT validated',
                  'mass_and_actuators':'provisional simulation parameters, not measurements',
                  'hardware_deployment_ready':False}}
    dest=ROOT/'assets/candidate-model.json';dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('JOINTS',[(j['name'],np.round(j['point_m'],4).tolist()) for j in joints])
    print('TCP',tcp.round(4).tolist(),'DISTAL_PADS',[o['id'] for o,_ in distal])
    return result

if __name__=='__main__':derive()
