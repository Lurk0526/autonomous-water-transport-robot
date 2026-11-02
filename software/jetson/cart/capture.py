"""Bounded RGB-D + available odometry/IMU/status capture with scene metadata."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]


def main():
    import rclpy
    from sensor_msgs.msg import Image
    from rclpy.qos import qos_profile_sensor_data
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds",type=float,default=15)
    parser.add_argument("--location",choices=["bench","field","unconfirmed"],default="unconfirmed")
    parser.add_argument("--camera-placement",choices=["handheld","temporary_support","mounted","unconfirmed"],default="unconfirmed")
    parser.add_argument("--route-id",default="unconfirmed")
    parser.add_argument("--condition",default="unconfirmed")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if not 1<=args.seconds<=600:
        raise ValueError("capture duration must be 1..600 s")
    folder=(args.output or ROOT/"work/cart_integration/recordings"/datetime.now(ZoneInfo('Asia/Shanghai')).strftime("%Y%m%d-%H%M%S")).resolve()
    if folder.exists():
        raise FileExistsError("Recording folder already exists")
    # A topic appearing in the graph is not sufficient: check actual publishers.
    rclpy.init();node=rclpy.create_node("cart_capture_discovery")
    raw_seen={}
    def raw(topic,msg):
        raw_seen[topic]=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
    raw_subs=[node.create_subscription(Image,topic,lambda msg,t=topic:raw(t,msg),qos_profile_sensor_data)
              for topic in ('/camera/color/image_raw','/camera/depth/image_rect_raw')]
    try:
        import time
        started=time.monotonic()
        while time.monotonic()-started<2:
            rclpy.spin_once(node,timeout_sec=.1)
        cfg=json.loads((ROOT/'config/cart-integration.json').read_text())
        candidates=[v for k,v in cfg['ros'].items() if k.endswith('_topic') and isinstance(v,str)]+['/tf','/road_perception/mask','/road_perception/confidence']
        available=[topic for topic in candidates if node.count_publishers(topic)>0]
        missing=[topic for topic in candidates if topic not in available]
        required=['/camera/color/image_raw','/camera/depth/image_rect_raw']
        now=node.get_clock().now().nanoseconds*1e-9
        if any(node.count_publishers(t)==0 or t not in raw_seen or not -.05<=now-raw_seen[t]<=.75 for t in required):
            raise RuntimeError("Fresh RGB/depth frames are missing; run cart_preflight.sh --recover-camera first")
    finally:
        node.destroy_node();rclpy.shutdown()
    command=[sys.executable,str(ROOT/'tools/record_road_session.py'),'--seconds',str(args.seconds),'--output',str(folder),
             '--purpose','Real RGB-D input capture; field recognition, localization and driving remain unvalidated']
    for topic in available:
        command += ['--extra-topic',topic]
    result=subprocess.run(command,cwd=ROOT)
    if not folder.exists():
        return result.returncode or 1
    context={'schema_version':1,'location':args.location,'camera_placement':args.camera_placement,
             'route_id':args.route_id,'condition':args.condition,'available_extra_topics':available,
             'unavailable_extra_topics':missing,'field_validated':False,'vehicle_mount_validated':False,
             'ground_truth_labels_present':False,'hardware_commands_sent':False,
             'note':'手持现场数据可检查外观与测距，但不能替代最终装车视角/外参/制动验收。'}
    (folder/'integration-context.json').write_text(json.dumps(context,ensure_ascii=False,indent=2))
    if result.returncode:
        return result.returncode
    from .rgbd_audit import audit
    report=audit(folder,folder/'audit',args.location)
    print(json.dumps({'bag':str(folder),'raw_transport_complete':report['raw_transport_complete'],
                      'geometric_input_usable':report['geometric_input_usable'],'missing_feedback_topics':missing},ensure_ascii=False,indent=2))
    return 0 if report['geometric_input_usable'] else 3


if __name__=='__main__':
    raise SystemExit(main())
