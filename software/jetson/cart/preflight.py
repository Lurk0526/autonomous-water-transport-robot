"""Produce an honest hardware/input readiness snapshot and offline software checks."""
import argparse
import base64
from datetime import datetime
import html
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from zoneinfo import ZoneInfo

from .parameters import apply_measurements

ROOT=Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def usb_inventory():
    devices=[]
    for path in Path('/sys/bus/usb/devices').glob('*'):
        if (path/'idVendor').exists():
            values={name:(path/name).read_text().strip() for name in ['idVendor','idProduct','product'] if (path/name).exists()}
            if values.get('idVendor')=='2bc5':devices.append(values)
    return devices


def run(out,audit_path,observe_seconds=8,recover_camera=False):
    import yaml
    out.mkdir(parents=True,exist_ok=True)
    cfg=read_json(ROOT/'config/cart-integration.json')
    base=read_json(ROOT/'sim/line_free_cart/config.json')
    registry_path=ROOT/cfg['vehicle_measurements']
    line_profile=read_json(ROOT/'config/cart-line-tracking.json')
    field_validation=read_json(ROOT/'config/cart-field-validation.json')
    profile,measurements=apply_measurements(base,read_json(registry_path),ROOT)
    (out/'vehicle-profile.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2,allow_nan=False))
    devices=usb_inventory()
    recovery=[]
    monitor_path=out/'monitor'
    if observe_seconds:
        def observe():
            result=subprocess.run([sys.executable,'-m','cart.ros_monitor','--seconds',str(observe_seconds),'--output',str(monitor_path)],cwd=ROOT,capture_output=True,text=True,timeout=observe_seconds+15)
            (out/'monitor-command.log').write_text(result.stdout+result.stderr)
            if result.returncode:
                raise RuntimeError('Input monitor failed; see monitor-command.log')
            return read_json(monitor_path/'health.json')
        monitor=observe()
        streams=monitor.get('camera_streams',{})
        if recover_camera and devices and not all(streams.get(k,{}).get('fresh') for k in ('rgb','depth')):
            # One bounded recovery, only when explicitly requested; never repeatedly restart.
            result=subprocess.run(['systemctl','--user','restart','gemini-max.service'],capture_output=True,text=True,timeout=25)
            recovery.append({'action':'restart_camera_service_once','returncode':result.returncode})
            if result.returncode==0:monitor=observe()
    else:
        monitor=read_json(ROOT/'work/cart_integration/live-monitor/health.json')
    audit=read_json(audit_path)
    mount_path=ROOT/cfg['mount_measurements']
    mount=yaml.safe_load(mount_path.read_text()) if mount_path.exists() else {}
    tf=mount.get('camera_link_in_base_link') or {}
    mount_complete=mount.get('stage') in {'measured','calibrated'} and all(type(tf.get(k)) in (int,float) and math.isfinite(tf[k]) for k in ['x','y','z','roll','pitch','yaw']) and type(mount.get('ground_z_in_base_link_m')) in (int,float) and math.isfinite(mount['ground_z_in_base_link_m'])
    blockers=[]
    def require(ok,key,detail):
        if not ok:blockers.append({'id':key,'detail':detail})
    require(measurements['all_required_measurements_present'],'vehicle_measurements','真实轴距、车宽、悬长、转向及制动参数尚无可追溯实测记录。')
    require(bool(cfg['serial'].get('protocol')) or monitor.get('real_chassis_feedback_seen') is True,'chassis_protocol','尚未提供底盘STM32协议/固件，也没有现成ROS底盘反馈；没有猜测串口协议。')
    for kind,label in [('odometry','里程计'),('imu','IMU'),('steering','实测转角')]:
        entry=monitor.get('inputs',{}).get(kind,{})
        require(entry.get('fresh_and_valid') is True and entry.get('source')=='ros_live',kind,'没有接收到可用的真实'+label+'反馈。')
    require(monitor.get('rear_axle_frame_semantics_confirmed') is True,'rear_axle_frame','里程计车体原点与后轴中心的关系尚未确认。')
    require(mount_complete,'camera_mount','相机未完成车体外参和地面高度实测；没有生成车体可行驶区域。')
    require(audit.get('geometric_input_usable') is True,'rgbd_geometry','当前记录未通过RGB-D同步、内参和有效深度检查。')
    require(audit.get('location')=='field','field_capture','这次是台架/当前位置采集，实际砖路现场数据仍需现场人员录制。')
    require(field_validation.get('status')=='validated' and bool(field_validation.get('evidence')),'field_accuracy','现场标注/已知距离/定位参考真值尚未取得，未验收识别、测距或定位精度。')
    require(line_profile.get('scene_profile_validated') is True and type(line_profile.get('measured_steering_sign')) in (int,float) and line_profile.get('measured_steering_sign') in [-1,1],
            'primary_line_profile','循迹为主方案，黑/白/黄兼容已实现；现场线宽、线形、转向映射与路口/站点标记尚未验收。')
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_cart_runtime.py')
    tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (out/'tests.txt').write_text(stream.getvalue())
    report={'schema_version':1,'created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
            'software_checks_passed':tests.wasSuccessful(),'test_methods':tests.testsRun,
            'real_vehicle_ready':not blockers,'hardware_commands_sent':False,'motion_output_enabled':False,
            'camera_devices':devices,'camera_recovery':recovery,'vehicle_measurements':measurements,
            'live_input_snapshot':monitor,'rgbd_audit':audit,'blockers':blockers,
            'mount_measurements_complete':mount_complete,
            'navigation_strategy':{'primary':'track_line','backup':'line_free'},'line_profile':line_profile,
            'cad_status':'原生CAD与引用文件齐全，但车体坐标、尺寸和运动学尚未导出核验。'}
    (out/'readiness.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    rows=''.join('<tr><td>'+html.escape(b['id'])+'</td><td>'+html.escape(b['detail'])+'</td></tr>' for b in blockers)
    metrics=audit.get('pairing',{})
    pictures=''
    for name in ('rgb.jpg','depth.png'):
        path=audit_path.parent/name
        if path.is_file():
            mime='image/jpeg' if name.endswith('jpg') else 'image/png'
            pictures+=f'<figure><img alt="{name}" src="data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"><figcaption>{"原始RGB（未改变图像坐标）" if name.startswith("rgb") else "原始深度伪彩色；黑色为无效"}</figcaption></figure>'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>小车真实输入接入验收</title><style>body{font:16px/1.75 system-ui,sans-serif;color:#183640;background:#f1f5f5;margin:0}main{max-width:1050px;margin:32px auto;padding:0 22px}section{background:white;border:1px solid #dbe4e7;border-radius:12px;padding:22px;margin:18px 0}.good{color:#087f74}.pending{color:#a56622}h1{font-size:29px}h2{font-size:20px}table{width:100%;border-collapse:collapse;font-size:14px}td{border-bottom:1px solid #e3ebed;padding:10px;vertical-align:top}.pictures{display:flex;gap:14px;flex-wrap:wrap}figure{margin:0;flex:1;min-width:250px}img{width:100%;border-radius:8px}figcaption{font-size:12px;color:#627b85}a{color:#216c9b}code{background:#edf2f3;padding:3px 6px}.note{border-left:4px solid #c18a3d;padding:10px 16px;background:#fff7e8}</style><main><h1>小车真实输入接入验收</h1>'''
    page+=f'<p>记录时间 {html.escape(report["created_at"])}。这是本次采集/检查快照，不是持续实时监控。</p><section><h2 class="good">已完成的软件与真实相机检查</h2><p>{tests.testsRun}项输入、参数来源、定位积分和制动日志拟合测试：{"通过" if tests.wasSuccessful() else "失败"}。</p><p>RGB {audit.get("timing",{}).get("rgb",{}).get("frames",0)}帧；深度 {audit.get("timing",{}).get("depth",{}).get("frames",0)}帧。60ms内配对比例 {metrics.get("within_60ms_fraction",0)*100:.1f}%；时间差中位数 {metrics.get("median_delta_ms",0):.2f}ms。</p><p>有效深度帧间均值 {(audit.get("depth_valid_fraction_mean") or 0)*100:.1f}%；深度投影有效比例 {(audit.get("projected_valid_fraction_mean") or 0)*100:.1f}%。这些是当前场景的数据有效率，不是测距或道路识别精度。</p><div class="pictures">'+pictures+'</div></section>'
    page+='<section><h2 class="pending">实车闭环尚待以下输入</h2><p>底盘未连接；软件适配器已经实现并用录制/合成数据检查，不能把模拟反馈写成实机连接成功。</p><table>'+rows+'</table></section>'
    vo=audit.get('visual_motion_screening',{})
    page+=f'<section><h2>定位输入的额外检查</h2><p>真实RGB-D抽样帧间PnP：{vo.get("accepted_pairs",0)} / {vo.get("attempted_pairs",0)}组通过几何筛查。没有独立位姿真值、回环或车体外参，这不能算完成整车定位。</p><p>轮速/IMU积分、协方差增长、绝对位置校正和异常输入逻辑已经实现；目前没有真实轮速或IMU输入。</p></section>'
    page+='<section><h2>现成操作入口</h2><p><code>./cart_preflight.sh</code>：只读检查实时相机、定位、障碍与反馈。</p><p><code>./cart_preflight.sh --recover-camera</code>：缺少新鲜RGB/深度且设备存在时，最多尝试一次相机服务恢复。</p><p><code>./cart_capture.sh --seconds 15 --location field --camera-placement handheld --route-id brick-01 --condition shade</code>：现场人员就位后录制真实RGB-D及实际存在的定位/IMU/感知话题，自动保存元数据与审计。</p><p class="note">用户负责算法，无需承担机构改造；轴距、车宽、安装外参和底盘协议由负责同学提供后填入测量表，软件会校验单位和来源并替换对应参数。未填项保持未知。</p><p><a href="../report.html">返回扩展仿真报告</a> · <a href="readiness.json">真实输入验收明细</a> · <a href="tests.txt">输入软件测试日志</a> · <a href="vehicle-profile.json">参数替换结果</a></p></section></main></html>'
    (out/'report.html').write_text(page)
    print(json.dumps({'software_checks_passed':report['software_checks_passed'],'real_vehicle_ready':report['real_vehicle_ready'],
                      'rgbd_geometric_input_usable':audit.get('geometric_input_usable'),'blockers':blockers},ensure_ascii=False,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'sim/line_free_cart/output/integration')
    p.add_argument('--audit',type=Path,default=ROOT/'work/cart_integration/rgbd-audit-adjusted/audit.json')
    p.add_argument('--observe-seconds',type=float,default=8)
    p.add_argument('--recover-camera',action='store_true')
    a=p.parse_args()
    if not 0<=a.observe_seconds<=600:raise ValueError('observe-seconds must be 0..600')
    r=run(a.out.resolve(),a.audit.resolve(),a.observe_seconds,a.recover_camera)
    return 0 if r['software_checks_passed'] else 1


if __name__=='__main__':raise SystemExit(main())
