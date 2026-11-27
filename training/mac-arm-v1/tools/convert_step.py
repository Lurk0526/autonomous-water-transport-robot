"""OCCT 8 STEP-to-local-mesh conversion preserving every named assembly occurrence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.collections import Sequence_TDF_Label
from OCP.TDF import TDF_Label
from OCP.TDataStd import TDataStd_Name
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE,TopAbs_REVERSED
from OCP.TopoDS import TopoDS
from OCP.TopLoc import TopLoc_Location
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder
from OCP.IFSelect import IFSelect_RetDone


def xyz(p):return np.array([p.X(),p.Y(),p.Z()])
def matrix(loc):
    t=loc.Transformation();m=np.eye(4)
    m[:3,:4]=[[t.Value(i,j) for j in range(1,5)] for i in range(1,4)]
    m[:3,3]*=.001
    return m
def name(label):
    attr=TDataStd_Name()
    return attr.Get().ToExtString() if label.FindAttribute(TDataStd_Name.GetID_s(),attr) else str(label.Tag())


def mesh_shape(shape,path,tolerance_mm):
    mesher=BRepMesh_IncrementalMesh(shape,tolerance_mm,False,.35,True);mesher.Perform()
    triangles=[];cylinders=[];it=TopExp_Explorer(shape,TopAbs_FACE)
    while it.More():
        face=(getattr(TopoDS,'Face_s',None) or TopoDS.Face)(it.Current());location=TopLoc_Location()
        mesh=BRep_Tool.Triangulation_s(face,location)
        if mesh is not None:
            transform=location.Transformation()
            for i in range(1,mesh.NbTriangles()+1):
                triangle=mesh.Triangle(i)
                indices=[triangle.Value(j) for j in (1,2,3)]
                vertices=np.array([xyz(mesh.Node(j).Transformed(transform))*.001 for j in indices])
                if face.Orientation()==TopAbs_REVERSED:vertices=vertices[[0,2,1]]
                if np.linalg.norm(np.cross(vertices[1]-vertices[0],vertices[2]-vertices[0]))>1e-15:triangles.append(vertices)
        surf=BRepAdaptor_Surface(face,True)
        if surf.GetType()==GeomAbs_Cylinder:
            cyl=surf.Cylinder();axis=cyl.Axis();direction=xyz(axis.Direction());point=xyz(axis.Location())
            v1,v2=surf.FirstVParameter(),surf.LastVParameter()
            if all(math.isfinite(v) and abs(v)<1e6 for v in [v1,v2]):
                cylinders.append({'radius_m':cyl.Radius()*.001,'axis_direction':direction.tolist(),
                    'axis_point_m':(point*.001).tolist(),'face_midpoint_m':((point+direction*(v1+v2)/2)*.001).tolist()})
        it.Next()
    data=np.asarray(triangles,dtype=np.float64)
    if not len(data):raise ValueError('No triangles for '+str(path))
    with path.open('wb') as f:
        f.write(b'Converted from user STEP, metres; not a kinematic calibration'.ljust(80,b' '))
        f.write(struct.pack('<I',len(data)))
        for tri in data:
            normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal/=np.linalg.norm(normal)
            f.write(struct.pack('<12fH',*normal,*tri.flatten(),0))
    properties=GProp_GProps();BRepGProp.VolumeProperties_s(shape,properties)
    return {'mesh':path.name,'triangles':len(data),'bbox_m':[data.min(axis=(0,1)).tolist(),data.max(axis=(0,1)).tolist()],
        'volume_m3':properties.Mass()*1e-9,'center_of_volume_m':(xyz(properties.CentreOfMass())*.001).tolist(),
        'brep_valid':bool(BRepCheck_Analyzer(shape).IsValid()),'cylinders':cylinders,
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def convert(source,output,tolerance_mm=.35):
    output.mkdir(parents=True,exist_ok=True);meshdir=output/'meshes';meshdir.mkdir(exist_ok=True)
    doc=TDocStd_Document(TCollection_ExtendedString('MDTV-XCAF'))
    reader=STEPCAFControl_Reader();reader.SetNameMode(True);reader.SetColorMode(True)
    if reader.ReadFile(str(source))!=IFSelect_RetDone or not reader.Transfer(doc):raise ValueError('STEP transfer failed')
    tool=XCAFDoc_DocumentTool.ShapeTool_s(doc.Main());roots=Sequence_TDF_Label();tool.GetFreeShapes(roots)
    definitions={};occurrences=[];assemblies=[]
    def walk(label,parent,path):
        definition=TDF_Label()
        if not tool.GetReferredShape_s(label,definition):definition=label
        nm=name(definition);local=matrix(tool.GetLocation_s(label));global_pose=parent@local
        path=path+'/'+nm
        if tool.IsAssembly_s(definition):
            assemblies.append({'path':path,'name':nm,'pose_m':global_pose.tolist()})
            children=Sequence_TDF_Label();tool.GetComponents_s(definition,children)
            for i in range(1,children.Length()+1):walk(children.Value(i),global_pose,path+f'[{i:03d}]')
        else:
            key=nm
            if key not in definitions:
                ident=f'part_{len(definitions):03d}'
                print('MESH',ident,nm,flush=True)
                item=mesh_shape(tool.GetShape_s(definition),meshdir/(ident+'.stl'),tolerance_mm)
                definitions[key]={'id':ident,'name':nm,**item}
            occurrences.append({'id':f'instance_{len(occurrences):03d}','path':path,'name':nm,
                'part_id':definitions[key]['id'],'pose_m':global_pose.tolist()})
    for i in range(1,roots.Length()+1):walk(roots.Value(i),np.eye(4),'')
    result={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'length_scale_to_m':.001,
        'tolerance_mm':tolerance_mm,'parts':list(definitions.values()),'occurrences':occurrences,'assemblies':assemblies,
        'geometry_kernel':'cadquery-ocp-novtk 8.0.1.1.0','joint_model_validated':False,'hardware_calibration_validated':False}
    (output/'cad-manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('CONVERTED',len(definitions),'unique parts;',len(occurrences),'leaf occurrences',flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();convert(args.source,args.output)
