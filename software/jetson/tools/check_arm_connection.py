#!/usr/bin/env python3
"""Check the controller and six servo replies without sending motion targets."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serial.tools import list_ports
from vision.arm_serial import ArmSerial, MissingServoFeedback


def check_connection(port=None, transport_factory=ArmSerial):
    report = dict(checked_unix=time.time(), port=port, motion_commands_sent=False,
                  controller_connected=False, ready=False)
    if not port:
        ports = [p.device for p in list_ports.comports() if (p.vid,p.pid)==(0x1A86,0x7523)]
        report['detected_ports'] = ports
        if len(ports)!=1:
            report['error'] = '请连接一块机械臂 CH340 控制板，或用 --port 指定控制板端口'
            return report
        port = ports[0]
        report['port'] = port
    arm = None
    try:
        arm = transport_factory(port)
        report.update(controller_connected=True, connection=dict(arm.connection_report))
        positions = arm.positions()
        report.update(ready=True, positions_p={str(a):p for a,p in positions.items()},
                      missing_axes=[])
    except MissingServoFeedback as exc:
        report.update(error=str(exc), positions_p={str(a):p for a,p in exc.positions.items()},
                      missing_axes=sorted(exc.replies),
                      replies={str(a):r.decode('ascii',errors='replace') for a,r in exc.replies.items()})
    except (RuntimeError, ValueError, OSError) as exc:
        report.update(error=str(exc))
        if hasattr(exc, 'connection_report'):
            report['connection'] = exc.connection_report
    finally:
        if arm:
            try:
                arm.close()
            except (RuntimeError, ValueError, OSError) as exc:
                report.update(ready=False, close_error=str(exc))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', help='CH340 controller; auto-detect only if exactly one exists')
    parser.add_argument('--output', type=Path, default=ROOT/'work/arm-small-test/connection-status.json')
    args = parser.parse_args()
    report = check_connection(args.port)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp = args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    temp.replace(args.output)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return 0 if report['ready'] else 1


if __name__=='__main__':
    raise SystemExit(main())
