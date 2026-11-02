"""Inspect candidate black/white/yellow lines in a captured real image, no drive."""
import argparse
import json
from pathlib import Path
from .line_tracking import detect_line


def inspect(rgb,rotation=0,color='auto_black_white_yellow'):
    import cv2,numpy as np
    raw_h,raw_w=rgb.shape[:2]
    if rotation==90:processed=cv2.rotate(rgb,cv2.ROTATE_90_CLOCKWISE)
    elif rotation==180:processed=cv2.rotate(rgb,cv2.ROTATE_180)
    elif rotation==270:processed=cv2.rotate(rgb,cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation==0:processed=rgb.copy()
    else:raise ValueError('Rotation must be 0/90/180/270')
    line=detect_line(processed,color)
    raw=[]
    for x,y in line['path_pixels']:
        if rotation==90:x,y=y,raw_h-1-x
        elif rotation==180:x,y=raw_w-1-x,raw_h-1-y
        elif rotation==270:x,y=raw_w-1-y,x
        raw.append([x,y])
    overlay=processed.copy()
    if line['path_pixels']:cv2.polylines(overlay,[np.asarray(line['path_pixels'],np.int32)],False,(0,220,90),2)
    cv2.putText(overlay,f"{line['color']} {line['confidence']:.2f} {line['reason']}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(250,120,40),1)
    report={k:v for k,v in line.items() if k!='mask'}
    report.update(input_rotation_degrees=rotation,path_pixels_raw=raw,scene_profile_validated=False,
                  vehicle_steering_output_enabled=False,
                  note='仅输出候选线；使用真实内参投影时须用path_pixels_raw，且必须有实测车体外参。')
    return report,overlay,line['mask']


def main():
    import cv2,numpy as np
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--image',type=Path);source.add_argument('--pair',type=Path)
    p.add_argument('--rotation',type=int,default=180)
    p.add_argument('--color',choices=['auto_black_white_yellow','black','white','yellow'],default='auto_black_white_yellow')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.pair:
        with np.load(a.pair) as data:rgb=data['rgb']
    else:
        bgr=cv2.imread(str(a.image))
        if bgr is None:raise ValueError('Image could not be read')
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    report,overlay,mask=inspect(rgb,a.rotation,a.color)
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'line.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    cv2.imwrite(str(a.out/'line-overlay.jpg'),cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(a.out/'line-mask.png'),mask)
    print(json.dumps({k:v for k,v in report.items() if k not in {'candidates','path_pixels','path_pixels_raw'}},ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
