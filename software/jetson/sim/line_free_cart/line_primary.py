"""Closed-loop steering from rendered line pixels; line-free navigation is backup."""
import base64
from dataclasses import replace
import math
from pathlib import Path
import json

from cart.line_tracking import detect_line,NavigationMode,line_to_ground
from .model import Command,Obstacle,clamp
from .navigation import Experiment,NavigationSimulation
from .scenarios import straight,s_curve,bend

ROOT=Path(__file__).resolve().parents[2]


def render_line(route,state,color='black',visible=True,junction=False):
    import cv2
    import numpy as np
    image=np.full((240,320,3),140,np.uint8)
    paint={'black':(20,20,20),'white':(240,240,240),'yellow':(235,205,25)}[color]
    if not visible:return image
    progress=route.project(state.x,state.y)[0]
    sides=[[],[]]
    beta=.45;fx=fy=220.;height=.35
    for i in range(70):
        distance=progress+.10+i*.055
        x,y,yaw=route.at(distance)
        if distance>route.length:
            extra=distance-route.length;x+=extra*math.cos(yaw);y+=extra*math.sin(yaw)
        for side,sign in enumerate((-1,1)):
            px=x-sign*.018*math.sin(yaw);py=y+sign*.018*math.cos(yaw)
            dx,dy=px-state.x,py-state.y
            forward=math.cos(state.yaw)*dx+math.sin(state.yaw)*dy
            left=-math.sin(state.yaw)*dx+math.cos(state.yaw)*dy
            z=math.cos(beta)*forward+math.sin(beta)*height
            if z>.05:
                u=160-fx*left/z;v=120+fy*(-math.sin(beta)*forward+math.cos(beta)*height)/z
                sides[side].append([int(clamp(u,-1000,1320)),int(clamp(v,-1000,1240))])
    if sides[0] and sides[1]:cv2.fillPoly(image,[np.array(sides[0]+sides[1][::-1],np.int32)],paint)
    if junction:cv2.line(image,(0,170),(319,170),paint,14)
    return image


class LinePrimarySimulation(NavigationSimulation):
    def __init__(self,config,case,color='black',loss=None,backup_ready=True,junction_at=None):
        super().__init__(config,case)
        self.color,self.loss,self.backup_ready,self.junction_at=color,loss,backup_ready,junction_at
        self.policy=NavigationMode(json.loads((ROOT/'config/cart-line-tracking.json').read_text()))
        self.line=None;self.last_line_steer=0.;self.example_image=None

    def _decide(self,motion_enabled,external_hold):
        backup,distance=super()._decide(motion_enabled,external_hold)
        if backup.emergency or backup.reason=='waiting_inputs':return backup,distance
        visible=not self.loss or not self.loss[0]<=self.t<self.loss[1]
        image=render_line(self.case.route,self.vehicle.state,self.color,visible,
                          self.junction_at is not None and self.t>=self.junction_at)
        line=detect_line(image)
        if self.example_image is None and self.t>=2:
            import cv2
            self.example_image=base64.b64encode(cv2.imencode('.jpg',cv2.cvtColor(image,cv2.COLOR_RGB2BGR))[1]).decode()
        self.line={k:v for k,v in line.items() if k not in {'mask','candidates'}}
        mode=self.policy.update(self.t,line,self.t,base_ready=True,obstacles_ready=True,backup_ready=self.backup_ready)
        if mode=='STOP':
            reason='junction_stop' if self.policy.reason=='junction_or_line_ambiguity' else self.policy.reason
            self.latched_fault=reason
            return Command(0,self.vehicle.state.steer,True,reason),distance
        if mode=='WAIT_LINE':return Command(0,0,False,'WAIT_LINE'),distance
        if mode=='BACKUP':return replace(backup,reason='BACKUP'),distance
        if mode=='LINE_LOST_SLOW':return Command(min(backup.speed,.12),self.last_line_steer,False,mode),distance
        # Camera geometry is known ONLY in this synthetic renderer. No guessed
        # transform is installed into the actual camera configuration.
        import numpy as np
        beta=.45
        camera_to_base=np.array([[0,-math.sin(beta),math.cos(beta),0],[-1,0,0,0],
                                 [0,-math.cos(beta),-math.sin(beta),.35],[0,0,0,1]])
        ground=line_to_ground(line['path_pixels'],[[220,0,160],[0,220,120],[0,0,1]],[0]*5,camera_to_base,0,mount_verified=True)
        if not ground['valid']:
            self.latched_fault='line_ground_projection_invalid'
            return Command(0,self.vehicle.state.steer,True,self.latched_fault),distance
        target=min(ground['points_xy_m'],key=lambda p:abs(p[0]-(.50+.45*self.vehicle.state.v)))
        curvature=2*target[1]/max(.02,target[0]**2+target[1]**2)
        steer=clamp(math.atan(self.vcfg['wheelbase_m']*curvature),-self.vcfg['max_steer_rad'],self.vcfg['max_steer_rad'])
        self.last_line_steer=steer
        return Command(min(backup.speed,.35),steer,False,'LINE'),distance

    def result(self,outcome):
        result=super().result(outcome)
        result.update(navigation_role='primary_track_line',line_color=self.color,
                      image_driven_steering=True,backup_available=self.backup_ready,
                      example_image_jpeg_base64=self.example_image,
                      notes='循迹转向来自合成RGB图像；定位、终点距离和障碍为合成输入。真实相机安装/转向映射及现场线形尚未验收。')
        return result


def run_line_cases(config):
    import cv2
    cv2.setNumThreads(1)
    definitions=[
        ('line_black_straight','黑线直线主导航',straight(6),'black',None,True,None,'arrived'),
        ('line_white_straight','白线直线主导航',straight(6),'white',None,True,None,'arrived'),
        ('line_yellow_straight','黄线直线主导航',straight(6),'yellow',None,True,None,'arrived'),
        ('line_black_s','黑线S弯主导航',s_curve(),'black',None,True,None,'arrived'),
        ('line_white_left','白线左弯主导航',bend(1),'white',None,True,None,'arrived'),
        ('line_yellow_right','黄线右弯主导航',bend(-1),'yellow',None,True,None,'arrived'),
        ('line_short_loss','短时遮挡后恢复循迹',s_curve(),'black',(5,5.4),True,None,'arrived'),
        ('line_backup','长时丢线切换无标线备用',s_curve(),'yellow',(5,12),True,None,'arrived'),
        ('line_loss_stop','丢线且备用不可用时停车',straight(6),'white',(5,30),False,None,'line_loss_stop'),
        ('line_junction','路口标记不明确时停车',straight(6),'black',None,True,4,'junction_stop'),
    ]
    results=[]
    for name,label,route,color,loss,backup,junction,expected in definitions:
        case=Experiment(name,label,route,expected=expected,initial_offset_m=.04,
                        initial_yaw_error_rad=.01,max_time_s=130)
        sim=LinePrimarySimulation(config,case,color,loss,backup,junction)
        results.append(sim.run())
    return results
