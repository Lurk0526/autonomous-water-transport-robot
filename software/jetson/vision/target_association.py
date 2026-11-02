"""Associate detections with the already locked bowl, never an arbitrary object."""
import math


def overlap(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0


def select_locked_target(record, center, bbox, confidence):
    detections=record.get('detections',[])
    if not isinstance(detections,list) or len(detections)<=1 or center is None:
        return record
    candidates=[]
    for item in detections:
        try:
            xy=item['mask_centroid_px'];box=item['bbox_xyxy_px'];score=float(item['confidence'])
            if (item['class']!='target_bowl' or len(xy)!=2 or len(box)!=4
                    or not all(math.isfinite(float(v)) for v in [*xy,*box,score])
                    or not 0<=box[0]<box[2]<=record['width']
                    or not 0<=box[1]<box[3]<=record['height']):continue
            if score>=confidence and math.dist(xy,center)<=80:
                candidates.append(item)
        except (KeyError,TypeError,ValueError):continue
    selected=None
    if len(candidates)==1:
        selected=candidates[0]
    elif len(candidates)>1:
        candidates.sort(key=lambda x:overlap(x['bbox_xyxy_px'],bbox),reverse=True)
        first,second=candidates[:2]
        duplicate_radius=max(15,.15*math.hypot(bbox[2]-bbox[0],bbox[3]-bbox[1]))
        same=(overlap(first['bbox_xyxy_px'],second['bbox_xyxy_px'])>.65
              and math.dist(first['mask_centroid_px'],second['mask_centroid_px'])<duplicate_radius)
        distinct=(overlap(first['bbox_xyxy_px'],bbox)>.25
                  and overlap(second['bbox_xyxy_px'],bbox)<.1)
        if same or distinct:selected=first
    if selected is None:return record
    return dict(record,detections=[selected],raw_detection_count=len(detections))


class VisibleTargetReacquisition:
    """Relock a unique clear bowl over distinct frames anywhere in the image."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous=None
        self.streak=0
        self.first_stamp=None

    def consider(self,record,weights_sha256,now):
        try:
            stamp=record['stamp']['sec']+record['stamp']['nanosec']/1e9
            width,height=record['width'],record['height']
            if (record.get('valid') is not True or record.get('state')!='live'
                    or not isinstance(weights_sha256,str) or not weights_sha256
                    or not 0<=now-stamp<=.35 or record.get('weights_sha256')!=weights_sha256):
                self.reset();return None
            candidates=[]
            for item in record.get('detections',[]):
                center=item['mask_centroid_px'];bbox=item['bbox_xyxy_px'];score=item['confidence']
                area=item['mask_area_px']
                if (item.get('class')=='target_bowl' and len(center)==2 and len(bbox)==4
                        and all(math.isfinite(v) for v in (*center,*bbox,score,area))
                        and .85<=score<=1 and 0<=bbox[0]<bbox[2]<=width
                        and 0<=bbox[1]<bbox[3]<=height and bbox[0]<=center[0]<=bbox[2]
                        and bbox[1]<=center[1]<=bbox[3] and .02<=area/(width*height)<=.65):
                    candidates.append(item)
            if len(candidates)!=1:
                self.reset();return None
            item=candidates[0];center=item['mask_centroid_px'];bbox=item['bbox_xyxy_px']
            if self.previous is not None:
                if stamp<=self.previous['stamp']:return None
                same=(stamp-self.previous['stamp']<=.2
                      and math.dist(center,self.previous['center'])<=60
                      and overlap(bbox,self.previous['bbox'])>=.3)
            else:same=False
            if not same:
                self.streak=0;self.first_stamp=stamp
            self.streak+=1
            self.previous=dict(stamp=stamp,center=list(center),bbox=list(bbox))
            if self.streak>=3 and stamp-self.first_stamp>=.08:
                return dict(record,detections=[item],raw_detection_count=len(record['detections']))
        except (KeyError,TypeError,ValueError,OverflowError,ZeroDivisionError):
            self.reset()
        return None
