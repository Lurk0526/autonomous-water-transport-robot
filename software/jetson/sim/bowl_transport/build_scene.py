"""Build the parameterized bowl/platform USD fixture; no robot or fluid dynamics."""
import argparse
import json
import math
from pathlib import Path


def build(config, output):
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics
    stage=Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage,1.0)
    world=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(world.GetPrim())
    world.GetPrim().SetCustomDataByKey('status','PARAMETRIC_FIXTURE_ONLY: no robot articulation, no fluid simulation; provisional bowl profile')
    phys=UsdPhysics.Scene.Define(stage,'/World/PhysicsScene')
    phys.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));phys.CreateGravityMagnitudeAttr(9.81)
    light=UsdLux.DomeLight.Define(stage,'/World/Light');light.CreateIntensityAttr(900.)
    assumed=config['preview_assumptions_not_measurements'];known=config['confirmed']
    def box(path,xyz,size,color):
        shape=UsdGeom.Cube.Define(stage,path);shape.CreateSizeAttr(1.)
        shape.AddTranslateOp().Set(Gf.Vec3d(*xyz));shape.AddScaleOp().Set(Gf.Vec3f(*size))
        shape.CreateDisplayColorAttr([Gf.Vec3f(*color)]);UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        return shape
    box('/World/Ground',(1,0,-.025),(5,3,.05),(.24,.27,.30))
    h=assumed['platform_height_m'];side=assumed['platform_side_m']
    for name,x in [('Pick',0),('Place',assumed['platform_separation_m'])]:
        box('/World/'+name,(x,0,h/2),(side,side,h),(.9,.38,.08))
    mount=UsdGeom.Xform.Define(stage,'/World/RobotMount')
    mount.GetPrim().SetCustomDataByKey('status','EMPTY: insert validated articulated robot here; placement must be calibrated')
    bowl=UsdGeom.Xform.Define(stage,'/World/Bowl')
    bowl.AddTranslateOp().Set(Gf.Vec3d(0,0,h+.001))
    UsdPhysics.RigidBodyAPI.Apply(bowl.GetPrim())
    UsdPhysics.MassAPI.Apply(bowl.GetPrim()).CreateMassAttr(known['filled_total_mass_kg'])
    bowl.GetPrim().SetCustomDataByKey('mass_note','Lumped bowl+water mass 0.65 kg for rigid fixture only. Replace with EMPTY bowl mass before adding physical fluid.')
    profile=assumed['outer_profile_r_z_m'];th=assumed['wall_thickness_m'];bt=assumed['bottom_thickness_m']
    # Each wall panel is a separate convex collider on ONE rigid body.
    # A single convex hull of the complete bowl would seal the opening.
    count=32
    face_counts=[4]*6
    face_indices=[0,3,2,1,4,5,6,7,0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7]
    for j,((r0,z0),(r1,z1)) in enumerate(zip(profile,profile[1:])):
        for i in range(count):
            a=2*math.pi*i/count;b=2*math.pi*(i+1)/count
            polar=[(r0,a,z0),(r0,b,z0),(r1,b,z1),(r1,a,z1),
                   (r0-th,a,z0),(r0-th,b,z0),(r1-th,b,z1),(r1-th,a,z1)]
            pts=[Gf.Vec3f(r*math.cos(t),r*math.sin(t),z) for r,t,z in polar]
            mesh=UsdGeom.Mesh.Define(stage,f'/World/Bowl/Wall_{j}_{i}')
            mesh.CreatePointsAttr(pts);mesh.CreateFaceVertexCountsAttr(face_counts)
            mesh.CreateFaceVertexIndicesAttr(face_indices);mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateDoubleSidedAttr(True);mesh.CreateDisplayColorAttr([Gf.Vec3f(.035,.045,.05)])
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
    bottom=UsdGeom.Cylinder.Define(stage,'/World/Bowl/Bottom')
    bottom.CreateAxisAttr('Z');bottom.CreateRadiusAttr(profile[0][0]);bottom.CreateHeightAttr(bt)
    bottom.AddTranslateOp().Set(Gf.Vec3d(0,0,bt/2));bottom.CreateDisplayColorAttr([Gf.Vec3f(.035,.045,.05)])
    UsdPhysics.CollisionAPI.Apply(bottom.GetPrim())
    water=UsdGeom.Cylinder.Define(stage,'/World/Bowl/WaterSurface_VISUAL_ONLY')
    water.CreateAxisAttr('Z');water.CreateRadiusAttr(profile[-1][0]-th)
    water.CreateHeightAttr(.0004);water.AddTranslateOp().Set(Gf.Vec3d(0,0,known['height_m']-.0002))
    water.CreateDisplayColorAttr([Gf.Vec3f(.25,.65,.78)]);water.CreateDisplayOpacityAttr([.5])
    water.GetPrim().SetCustomDataByKey('status','visual disk only, not a fluid and has no collision')
    marker=UsdGeom.Cylinder.Define(stage,'/World/Bowl/RedMarker')
    marker.CreateAxisAttr('Z');marker.CreateRadiusAttr(.006);marker.CreateHeightAttr(.0001)
    marker.AddTranslateOp().Set(Gf.Vec3d(0,0,bt+.00005));marker.CreateDisplayColorAttr([Gf.Vec3f(.9,.02,.02)])
    camera=UsdGeom.Camera.Define(stage,'/World/OverviewCamera')
    look=Gf.Matrix4d().SetLookAt(Gf.Vec3d(2.8,-3,2),Gf.Vec3d(.8,0,.25),Gf.Vec3d(0,0,1))
    camera.AddTransformOp().Set(look.GetInverse());camera.CreateClippingRangeAttr(Gf.Vec2f(.01,100))
    stage.GetRootLayer().Save();print('Saved fixture:',output)
    return stage


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('scenario.json'))
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'output'/'bowl_platform_fixture.usda')
    args=p.parse_args();args.out=args.out.resolve();args.out.parent.mkdir(parents=True,exist_ok=True)
    try:build(json.loads(args.config.read_text()),args.out)
    except ModuleNotFoundError as e:
        if e.name=='pxr':raise SystemExit('Requires USD Python (included in Isaac Sim). Run this with the cloud Isaac Lab environment; use the bundled .usda for viewing.')
        raise
