"""CAD-pin closed-chain four-finger gripper with ONE axial rack actuator.

The nine pin locations are recovered from coincident 1.5 mm cylindrical axes.
Eight rigid links per branch preserve the STEP assembly at q=0. Three connect
constraints close each branch; proximal and middle phalanges remain passive.
Their springs/friction, density and rack servo are declared assumptions pending
hardware measurement. No prescribed per-finger shape law or bowl attachment.
"""
from __future__ import annotations

import copy
import json
from itertools import combinations
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .model import ROOT,make_rigid_reference_model as base_make_model,text,quaternion


def defaults():
    return {'rack_range_m':[0.,.009],'rack_kp_n_m':4000.,'rack_kv_n_s_m':8.,
        'rack_force_limit_n':40.,'rack_armature_kg':.003,'rack_damping_n_s_m':.15,
        'rack_frictionloss_n':.05,'rack_timeconst_s':.01,
        'plastic_density_kg_m3':1200.,'foam_density_kg_m3':80.,
        'minimum_link_mass_kg':.00015,'hinge_armature_kg_m2':1e-6,
        'hinge_damping_n_m_s':.002,'hinge_frictionloss_n_m':.0005,
        'passive_stiffness_n_m_rad':.015,'passive_springref_rad':0.,
        'hinge_range_rad':[-1.4,1.4],'loop_solref':[.004,1.],
        'loop_solimp':[.95,.99,.001,.5,2.],
        'hard_friction':[.1,.001,.0001]}


def _branch(cad,index):
    """Recover unambiguous planar pin topology, in canonical metres."""
    parts={p['id']:p for p in cad['parts']};origin=np.asarray(cad['tcp_m'])
    outward=-np.asarray(cad['finger_closure_axes'][index],float)
    up=np.asarray(cad['tool_rotation'],float)[:,1];normal=np.cross(outward,up)
    frame=np.stack([outward,up]);axes={};pads=[]
    for obj in cad['occurrences']:
        if obj['body']!=f'finger_{index}' and obj['name'] not in ('MG23基座','齿条30×2'):continue
        tf=np.asarray(obj['canonical_pose_m']);part=parts[obj['part_id']];name=obj['name']
        if name=='防撞缓冲垫':
            pads.append(frame@(tf[:3,:3]@part['center_of_volume_m']+tf[:3,3]-origin))
        for face in part['cylinders']:
            if not .0014<=face['radius_m']<=.0017:continue
            if abs(normal@(tf[:3,:3]@face['axis_direction']))<.999:continue
            projected=frame@(tf[:3,:3]@face['axis_point_m']+tf[:3,3]-origin)
            if projected[0]>0:axes.setdefault(name,[]).append(projected)
    def common(*names):
        points=[]
        for point in axes[names[0]]:
            if all(min(np.linalg.norm(other-point) for other in axes[name])<.00008 for name in names[1:]):
                if all(np.linalg.norm(point-old)>.001 for old in points):points.append(point)
        if len(points)!=1:raise ValueError(f'Ambiguous branch {index} CAD pins {names}: {points}')
        return points[0]
    nodes={'base':common('MG23基座','拉杆2','指2'),'rack':common('齿条30×2','拉杆3'),
        'crank_in':common('拉杆3','拉杆2'),'crank_out':common('拉杆2','拉杆1'),
        'middle':common('指2','指3'),'tip':common('指3','指1'),
        'coupler':common('指3','拉杆4'),'hub':common('拉杆4','拉杆1','拉杆5'),
        'end':common('指1','拉杆5')}
    if len(pads)!=3:raise ValueError('Exactly three CAD foam pads per branch are required')
    return nodes,sorted(pads,key=lambda p:-p[1]),frame,normal


def _inertial(body,objects,parts,tool_point,options):
    """CAD-volume masses and rotated bounding-box inertia, explicitly approximate."""
    contributions=[]
    for obj in objects:
        part=parts[obj['part_id']];tf=np.asarray(obj['canonical_pose_m'])
        density=options['foam_density_kg_m3'] if obj['name']=='防撞缓冲垫' else options['plastic_density_kg_m3']
        mass=max(float(part['volume_m3'])*density,1e-8)
        center=tf[:3,:3]@np.asarray(part['center_of_volume_m'])+tf[:3,3]-tool_point
        dimensions=np.diff(np.asarray(part['bbox_m']),axis=0)[0]
        diagonal=mass/12*np.array([dimensions[1]**2+dimensions[2]**2,
            dimensions[0]**2+dimensions[2]**2,dimensions[0]**2+dimensions[1]**2])
        inertia=tf[:3,:3]@np.diag(diagonal)@tf[:3,:3].T
        contributions.append((mass,center,inertia))
    if not contributions:raise ValueError(f'No CAD objects assigned to {body.get("name")}')
    mass=sum(row[0] for row in contributions)
    center=sum(m*p for m,p,_ in contributions)/mass
    tensor=np.zeros((3,3))
    for m,p,I in contributions:
        d=p-center;tensor+=I+m*(np.eye(3)*(d@d)-np.outer(d,d))
    if mass<options['minimum_link_mass_kg']:
        tensor*=options['minimum_link_mass_kg']/mass;mass=options['minimum_link_mass_kg']
    eigen,rotation=np.linalg.eigh(tensor);eigen=np.maximum(eigen,1e-10)
    if np.linalg.det(rotation)<0:rotation[:,0]*=-1
    ET.SubElement(body,'inertial',pos=text(center),mass=str(mass),
        quat=text(quaternion(rotation)),diaginertia=text(eigen))


def make_model(cfg,*,visual=False):
    """Replacement for model.make_model; 5 arm + one public closure_0 actuator."""
    options=defaults();options.update(cfg.get('linked_gripper',{}))
    if np.asarray(options['rack_range_m']).shape!=(2,) or not options['rack_range_m'][0]<=0<options['rack_range_m'][1]:
        raise ValueError('Rack range must contain the supplied CAD reference configuration')
    for key in ('rack_kp_n_m','rack_force_limit_n','plastic_density_kg_m3','minimum_link_mass_kg'):
        if not np.isfinite(options[key]) or options[key]<=0:raise ValueError('Invalid linked gripper '+key)
    # Start from complete named CAD visuals to preserve exact rod placements.
    _,xml,cad=base_make_model(cfg,visual=True);cad=copy.deepcopy(cad)
    root=ET.fromstring(xml);tool=root.find(".//body[@name='tool_roll']")
    parts={p['id']:p for p in cad['parts']};objects={o['id']:o for o in cad['occurrences']}
    tool_point=np.asarray(cad['joints'][-1]['point_m']);origin=np.asarray(cad['tcp_m'])-tool_point
    up=np.asarray(cad['tool_rotation'])[:,1]
    for i in range(4):tool.remove(tool.find(f"body[@name='finger_{i}']"))
    equality=root.find('equality')
    for item in list(equality):equality.remove(item)
    contact=root.find('contact')
    if contact is None:contact=ET.SubElement(root,'contact')
    rack=ET.SubElement(tool,'body',name='gripper_rack')
    ET.SubElement(rack,'joint',name='closure_0',type='slide',axis=text(up),range=text(options['rack_range_m']),
        damping=str(options['rack_damping_n_s_m']),armature=str(options['rack_armature_kg']),frictionloss=str(options['rack_frictionloss_n']))
    rack_objects=[o for o in cad['occurrences'] if o['name']=='齿条30×2']
    if len(rack_objects)!=1:raise ValueError('Expected one physical common CAD rack')
    rack_ids={o['id'] for o in rack_objects}
    # Its old display was attached rigidly to the tool; move that display onto rack.
    for geom in list(tool.findall('geom')):
        if geom.get('name','').rsplit('_',1)[0] in rack_ids:tool.remove(geom);rack.append(geom)
    _inertial(rack,rack_objects,parts,tool_point,options)
    body_to_finger={};branch_records=[];loop_pairs=[];occurrence_bodies={}
    for index in range(4):
        nodes,pads,frame,normal=_branch(cad,index)
        # All body frames coincide at q=0. Hinge anchors use those common local
        # CAD coordinates; nested rotations then provide exact rigid transforms.
        definitions=[('crank',None,'base','拉杆2'),('rod3','crank','crank_in','拉杆3'),
            ('prox',None,'base','指2'),('mid','prox','middle','指3'),('dist','mid','tip','指1'),
            ('rod1','crank','crank_out','拉杆1'),('rod5','rod1','hub','拉杆5'),('rod4','mid','coupler','拉杆4')]
        names={'prox':f'link_{index}_0','mid':f'link_{index}_1','dist':f'finger_{index}'}
        names.update({k:f'link_{index}_{k}' for k,_,_,_ in definitions if k not in names})
        bodies={};assigned={key:[] for key,_,_,_ in definitions}
        branch_objects=[o for o in cad['occurrences'] if o['body']==f'finger_{index}']
        part_key={name:key for key,_,_,name in definitions}
        for obj in branch_objects:
            name=obj['name']
            if name in part_key:assigned[part_key[name]].append(obj)
            elif name=='防撞缓冲垫':
                tf=np.asarray(obj['canonical_pose_m']);part=parts[obj['part_id']]
                point=frame@(tf[:3,:3]@part['center_of_volume_m']+tf[:3,3]-np.asarray(cad['tcp_m']))
                key=('prox','mid','dist')[int(np.argmin(np.linalg.norm(np.asarray(pads)-point,axis=1)))]
                assigned[key].append(obj)
        for key,parent,pivot,_ in definitions:
            body=ET.SubElement(tool if parent is None else bodies[parent],'body',name=names[key]);bodies[key]=body
            passive=key in ('prox','mid')
            ET.SubElement(body,'joint',name=f'hinge_{index}_{key}',type='hinge',
                pos=text(origin+frame.T@nodes[pivot]),axis=text(normal),range=text(options['hinge_range_rad']),
                damping=str(options['hinge_damping_n_m_s']),armature=str(options['hinge_armature_kg_m2']),
                frictionloss=str(options['hinge_frictionloss_n_m']),
                stiffness=str(options['passive_stiffness_n_m_rad'] if passive else 0.),
                springref=str(options['passive_springref_rad'] if passive else 0.))
            _inertial(body,assigned[key],parts,tool_point,options);body_to_finger[names[key]]=index
            for obj in assigned[key]:
                occurrence_bodies[obj['id']]=names[key]
                tf=np.asarray(obj['canonical_pose_m']);position=tf[:3,3]-tool_point;rotation=tf[:3,:3]
                if visual:
                    ET.SubElement(body,'geom',name=obj['id']+'_visual',type='mesh',mesh=obj['part_id'],
                        pos=text(position),quat=text(quaternion(rotation)),contype='0',conaffinity='0',group='2',rgba='.3 .65 .55 1')
                if obj['name']=='防撞缓冲垫':
                    center=position+rotation@np.array([0,0,.0015])
                    ET.SubElement(body,'geom',name=obj['id']+'_foam',type='box',size='.010 .010 .0015',
                        pos=text(center),quat=text(quaternion(rotation)),friction=text([cfg['gripper']['foam_friction'][1],.001,.0001]),
                        solref=text([cfg['gripper']['foam_time_constant_s'],cfg['gripper']['foam_damping_ratio']]),
                        solimp='.9 .95 .001',rgba='.2 .24 .2 .5',group='3')
                else:
                    ET.SubElement(body,'geom',name=obj['id']+'_hard',type='mesh',mesh=obj['part_id'],
                        pos=text(position),quat=text(quaternion(rotation)),friction=text(options['hard_friction']),
                        rgba='.2 .2 .2 .3',group='3')
        for first,second,pivot in [('rod3','rack','rack'),('rod4','rod1','hub'),('rod5','dist','end')]:
            body2='gripper_rack' if second=='rack' else names[second]
            name=f'loop_{index}_{pivot}'
            ET.SubElement(equality,'connect',name=name,body1=names[first],body2=body2,
                anchor=text(origin+frame.T@nodes[pivot]),solref=text(options['loop_solref']),solimp=text(options['loop_solimp']))
            loop_pairs.append([names[first],body2])
        # Only bodies sharing a real pin are excluded. Nonadjacent hard collisions
        # and every hard-body/bowl collision remain active.
        shared_pins={'base':['crank','prox'],'crank_in':['crank','rod3'],
            'crank_out':['crank','rod1'],'middle':['prox','mid'],'tip':['mid','dist'],
            'coupler':['mid','rod4'],'hub':['rod1','rod4','rod5'],'end':['rod5','dist']}
        for keys in shared_pins.values():
            for first,second in combinations(keys,2):
                ET.SubElement(contact,'exclude',body1=names[first],body2=names[second])
        ET.SubElement(contact,'exclude',body1=names['rod3'],body2='gripper_rack')
        branch_records.append({'index':index,'pins_radial_axial_m':{k:v.tolist() for k,v in nodes.items()},
            'body_names':names,'hinge_axis_canonical':normal.tolist(),
            'passive_joints':[f'hinge_{index}_prox',f'hinge_{index}_mid']})
    act=root.find(".//position[@name='gripper_actuator']")
    act.attrib.update(joint='closure_0',kp=str(options['rack_kp_n_m']),kv=str(options['rack_kv_n_s_m']),
        ctrlrange=text(options['rack_range_m']),forcerange=text([-options['rack_force_limit_n'],options['rack_force_limit_n']]),
        timeconst=str(options['rack_timeconst_s']))
    if not visual:
        for body in root.iter('body'):
            for geom in list(body.findall('geom')):
                if geom.get('name','').endswith('_visual'):body.remove(geom)
    decomposition_path=ROOT/'assets/cad/collision-parts.json'
    decomposed={}
    if options.get('use_convex_decomposition',True) and decomposition_path.is_file():
        decomposition=json.loads(decomposition_path.read_text())['parts']
        assets=root.find('asset')
        for body in root.iter('body'):
            for geom in list(body.findall('geom')):
                if not geom.get('name','').endswith('_hard'):continue
                part_id=geom.get('mesh')
                if part_id not in decomposition:continue
                source=decomposition[part_id]
                body.remove(geom)
                for number,hull in enumerate(source['hulls']):
                    meshname=part_id+f'_convex_{number}'
                    if assets.find(f"mesh[@name='{meshname}']") is None:
                        ET.SubElement(assets,'mesh',name=meshname,vertex=text(hull['vertices']),
                            face=' '.join(str(int(v)) for row in hull['faces'] for v in row))
                    attributes=dict(geom.attrib);attributes['mesh']=meshname
                    attributes['name']=geom.get('name')[:-5]+f'_c{number:02d}_hard'
                    ET.SubElement(body,'geom',attributes)
                decomposed[part_id]=len(source['hulls'])
    cad['linked_gripper']={'status':'CAD_pin_topology_physical_loops_unmeasured_springs_servo_masses',
        'decomposed_parts':decomposed,'control_dimensions':1,'control_joint':'closure_0','rack_body':'gripper_rack',
        'rack_axis_canonical':up.tolist(),'body_to_finger':body_to_finger,
        'occurrence_bodies':occurrence_bodies,'branches':branch_records,
        'loop_count':12,'passive_dofs_at_fixed_rack':8,'options':options,
        'inertia_status':'CAD_volume_density_and_bbox_approximation_not_measured',
        'hard_collision_status':'Original CAD convex meshes; decomposition may be required to preserve concavities.'}
    xml=ET.tostring(root,encoding='unicode');model=mujoco.MjModel.from_xml_string(xml)
    if model.nu!=6:raise ValueError('Exactly five arm and one gripper actuator required')
    return model,xml,cad


def loop_residuals(model,data):
    """Metre separation of each connect anchor after mj_forward/mj_step."""
    rows=[]
    for eid in range(model.neq):
        name=model.equality(eid).name or ''
        if not name.startswith('loop_'):continue
        first,second=int(model.eq_obj1id[eid]),int(model.eq_obj2id[eid])
        one=data.xpos[first]+data.xmat[first].reshape(3,3)@model.eq_data[eid,:3]
        two=data.xpos[second]+data.xmat[second].reshape(3,3)@model.eq_data[eid,3:6]
        rows.append({'name':name,'separation_m':float(np.linalg.norm(one-two))})
    return rows
