"""Audit actual rosbag2 RGB-D transport/calibration; never infer field accuracy."""
import argparse
from bisect import bisect_left
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3


def timing(stamps):
    delta = [b-a for a, b in zip(stamps, stamps[1:])]
    return {"frames": len(stamps), "rate_hz": (len(stamps)-1)/(stamps[-1]-stamps[0]) if len(stamps)>1 and stamps[-1]>stamps[0] else None,
            "non_increasing": sum(d <= 0 for d in delta),
            "max_gap_s": max(delta, default=None),
            "duration_s": stamps[-1]-stamps[0] if len(stamps)>1 else 0}


def nearest_index(values, target):
    if not values:
        return None
    i = bisect_left(values, target)
    candidates = [j for j in (i-1, i) if 0 <= j < len(values)]
    return min(candidates, key=lambda j: abs(values[j]-target))


def audit(bag, out, location="unconfirmed"):
    import cv2
    import numpy as np
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from vision.road_geometry import (Calibration, associate_depth, depth_metres,
                                      image_array, rotation_matrix)
    bag, out = Path(bag).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    databases = sorted(bag.glob("*.db3"), key=lambda p: int(p.stem.rsplit('_', 1)[-1]))
    if not databases:
        raise ValueError("No sqlite3 rosbag files")
    counts, frames, infos, transforms = Counter(), {"rgb": [], "depth": []}, {}, []
    depth_fractions = []
    depth_all_zero = 0
    encodings = {"rgb": set(), "depth": set()}
    handles = []
    classes = {}

    def decode(typename, data):
        if typename not in classes:
            classes[typename] = get_message(typename)
        return deserialize_message(data, classes[typename])

    def message(handle, row_id):
        row = handles[handle].execute("SELECT topics.type, messages.data FROM messages JOIN topics ON messages.topic_id=topics.id WHERE messages.id=?", (row_id,)).fetchone()
        return decode(*row)

    try:
        for file_index, database in enumerate(databases):
            connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
            handles.append(connection)
            query = "SELECT messages.id,topics.name,topics.type,messages.data FROM messages JOIN topics ON topics.id=messages.topic_id ORDER BY messages.timestamp,messages.id"
            for row_id, name, typename, data in connection.execute(query):
                counts[name] += 1
                if name.endswith("camera_info"):
                    infos[name] = decode(typename, data)
                elif name == "/tf_static":
                    transforms.extend(decode(typename, data).transforms)
                elif name in {"/camera/color/image_raw", "/camera/depth/image_rect_raw"}:
                    msg = decode(typename, data)
                    key = "rgb" if "/color/" in name else "depth"
                    stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
                    frames[key].append((stamp, file_index, row_id))
                    encodings[key].add(msg.encoding)
                    if key == "depth":
                        value = depth_metres(msg)
                        valid = np.isfinite(value)&(value>=.3)&(value<=5)
                        depth_fractions.append(float(valid.mean()))
                        depth_all_zero += int(not np.any(np.isfinite(value)&(value>0)))
        summary = {"kind": "real_rgbd_recording_audit", "bag_path": str(bag),
                   "location": location, "field_validated": False,
                   "counts": dict(counts), "timing": {k: timing([r[0] for r in v]) for k,v in frames.items()},
                   "encodings": {k: sorted(v) for k,v in encodings.items()},
                   "depth_all_zero_frames": depth_all_zero,
                   "depth_all_zero_fraction": depth_all_zero/max(1, len(depth_fractions)),
                   "depth_window_m": [.3, 5.0],
                   "depth_valid_fraction_mean": float(np.mean(depth_fractions)) if depth_fractions else None,
                   "depth_valid_fraction_median": float(np.median(depth_fractions)) if depth_fractions else None,
                   "raw_transport_complete": all(counts[k]>0 for k in ["/camera/color/image_raw", "/camera/depth/image_rect_raw", "/camera/color/camera_info", "/camera/depth/camera_info", "/tf_static"]),
                   "vehicle_mount_validated": False, "obstacle_distance_accuracy_validated": False,
                   "localization_accuracy_validated": False}
        rgb_stamps = [r[0] for r in frames["rgb"]]
        depth_stamps = sorted(r[0] for r in frames["depth"])
        pairs = [(r, nearest_index(depth_stamps, r[0])) for r in frames["rgb"]]
        differences = [abs(r[0]-depth_stamps[i]) for r,i in pairs if i is not None]
        summary["pairing"] = {"within_60ms_fraction": sum(d <= .06 for d in differences)/max(1,len(rgb_stamps)),
                              "median_delta_ms": float(np.median(differences)*1000) if differences else None,
                              "max_delta_ms": max(differences)*1000 if differences else None}
        summary["timing_checks_passed"] = bool(summary["raw_transport_complete"] and
            summary["pairing"]["within_60ms_fraction"] >= .95 and
            all(v["non_increasing"] == 0 and v["max_gap_s"] is not None and v["max_gap_s"] < .2 for v in summary["timing"].values()))
        calibration_error = ""
        try:
            color_info = infos["/camera/color/camera_info"]
            depth_info = infos["/camera/depth/camera_info"]
            cc = Calibration(color_info.width, color_info.height, color_info.k, color_info.d,
                             color_info.header.frame_id, color_info.distortion_model).validate()
            dc = Calibration(depth_info.width, depth_info.height, np.array(depth_info.p).reshape(3,4)[:,:3], [],
                             depth_info.header.frame_id).validate()
            edges = {}
            for transform in transforms:
                q, v = transform.transform.rotation, transform.transform.translation
                t = np.eye(4)
                t[:3,:3] = rotation_matrix([q.x,q.y,q.z,q.w])
                t[:3,3] = [v.x,v.y,v.z]
                # parent_from_child, and its inverse
                edges.setdefault(transform.child_frame_id, []).append((transform.header.frame_id,t))
                edges.setdefault(transform.header.frame_id, []).append((transform.child_frame_id,np.linalg.inv(t)))
            frontier, seen = [(dc.frame,np.eye(4))], set()
            depth_to_color = None
            while frontier:
                frame, t = frontier.pop(0)
                if frame == cc.frame:
                    depth_to_color = t
                    break
                if frame in seen:
                    continue
                seen.add(frame)
                frontier.extend((target, step@t) for target, step in edges.get(frame,[]) if target not in seen)
            if depth_to_color is None:
                raise ValueError("No camera-internal TF between depth and color optical frames")
            summary["calibration"] = {"color_frame": cc.frame, "depth_frame": dc.frame,
                                      "color_size": [cc.width,cc.height], "depth_size": [dc.width,dc.height],
                                      "depth_to_color": depth_to_color.tolist(), "valid": True}
            projected = []
            motion_pairs=[]
            depth_sorted = sorted(frames["depth"])
            for index in range(0,len(frames["rgb"]),max(1,len(frames["rgb"])//12)):
                rgb_item = frames["rgb"][index]
                j = nearest_index([x[0] for x in depth_sorted],rgb_item[0])
                if j is None or abs(depth_sorted[j][0]-rgb_item[0]) > .06:
                    continue
                rgb = image_array(message(rgb_item[1],rgb_item[2]))
                depth = depth_metres(message(depth_sorted[j][1],depth_sorted[j][2]))
                labels=np.full((cc.height,cc.width),255,np.uint8)
                confidence=np.zeros(labels.shape,np.float32)
                associated=associate_depth(depth,dc,cc,depth_to_color,labels,confidence)
                valid=np.isfinite(associated["aligned_depth"])
                projected.append(float(valid.mean()))
                motion_pairs.append({'stamp_s':rgb_item[0],'rgb':rgb,'aligned_depth':associated['aligned_depth']})
                # Preserve the first sampled pair, even when unusable. Invalid is not fabricated into free space.
                if len(projected)==1:
                    cv2.imwrite(str(out/"rgb.jpg"),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                    preview=cv2.applyColorMap(np.uint8(np.clip(np.nan_to_num(depth)/5*255,0,255)),cv2.COLORMAP_TURBO)
                    preview[~np.isfinite(depth)|(depth<=0)]=0
                    cv2.imwrite(str(out/"depth.png"),preview)
                    np.savez_compressed(out/"sample-pair.npz",rgb=rgb,depth_m=depth,color_k=cc.k,color_d=cc.d,depth_k=dc.k,depth_to_color=depth_to_color)
            summary["projected_valid_fraction_mean"] = float(np.mean(projected)) if projected else None
            summary["projection_samples"] = len(projected)
            from .vision_checks import screen_pairs
            summary['visual_motion_screening']=screen_pairs(motion_pairs,cc)
        except (ValueError, KeyError) as error:
            calibration_error = str(error)
            summary["calibration"] = {"valid": False, "reason": calibration_error}
        summary["geometric_input_usable"] = bool(summary["timing_checks_passed"] and
            not calibration_error and (summary.get("projected_valid_fraction_mean") or 0) >= .01 and
            (summary["depth_valid_fraction_median"] or 0) >= .01)
        summary["files"] = {p.name: {"bytes": p.stat().st_size,"sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in databases}
        (out/"audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return summary
    finally:
        for handle in handles:
            handle.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--location",choices=["bench","unconfirmed","field"],default="unconfirmed")
    args=parser.parse_args()
    r=audit(args.bag,args.out,args.location)
    print(json.dumps({k:v for k,v in r.items() if k not in {"files","calibration"}},ensure_ascii=False,indent=2))
    return 0 if r["geometric_input_usable"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
