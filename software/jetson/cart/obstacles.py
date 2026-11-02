"""Conservative observed-ground grid, gated on measured mounting and attitude."""


def build_grid(xyz_camera,semantic_ids,confidence,camera_to_base,ground_z_m,
               mount_verified=False,roll_rad=None,pitch_rad=None,resolution_m=.1):
    import numpy as np
    from vision.road_geometry import valid_transform
    grid=np.full((40,50),-1,dtype=np.int8)
    invalid=lambda reason:{'valid':False,'reason':reason,'occupancy':grid,'frame':'base_link','resolution_m':resolution_m,'origin_xy_m':[0,-2]}
    if mount_verified is not True:
        return invalid('measured_mount_required')
    if any(v is None or not np.isfinite(v) for v in [ground_z_m,roll_rad,pitch_rad]):
        return invalid('ground_and_current_attitude_required')
    if abs(roll_rad)>.1 or abs(pitch_rad)>.1:
        return invalid('terrain_or_attitude_outside_validated_flat_ground_model')
    if resolution_m!=.1:
        raise ValueError('Current grid bounds use 0.1 m cells')
    t=valid_transform(camera_to_base)
    xyz=np.asarray(xyz_camera,dtype=float).reshape(-1,3)
    labels=np.asarray(semantic_ids)
    scores=np.asarray(confidence,dtype=float)
    if len(xyz)!=len(labels) or len(xyz)!=len(scores):
        raise ValueError('Point/semantic arrays differ')
    points=xyz@t[:3,:3].T+t[:3,3]
    finite=np.isfinite(points).all(axis=1)
    for point,label,score in zip(points[finite],labels[finite],scores[finite]):
        x,y,z=point
        col,row=int(np.floor(x/resolution_m)),int(np.floor((y+2)/resolution_m))
        if not 0<=col<50 or not 0<=row<40:
            continue
        height=z-ground_z_m
        if height>=.1 or height<=-.08:
            grid[row,col]=100
        elif abs(height)<=.06 and label==0 and np.isfinite(score) and .65<=score<=1 and grid[row,col]!=100:
            grid[row,col]=0
    return {'valid':True,'reason':'observed_ground_only','occupancy':grid,'frame':'base_link',
            'resolution_m':resolution_m,'origin_xy_m':[0,-2],
            'note':'阈值为待实测的设计值；未观测、无效深度、非铺装和平地模型之外均不自动标为空闲。'}
