"""RGB-D relative camera-motion screening, not global/base-frame localization."""
import math


def pnp_motion(points_xyz, image_uv, camera_k, distortion):
    import cv2
    import numpy as np
    xyz=np.asarray(points_xyz,dtype=np.float64).reshape(-1,3)
    uv=np.asarray(image_uv,dtype=np.float64).reshape(-1,2)
    if len(xyz)!=len(uv) or len(xyz)<30 or not np.isfinite(xyz).all() or not np.isfinite(uv).all() or (xyz[:,2]<=0).any():
        return {'valid':False,'reason':'insufficient_or_invalid_correspondences'}
    cv2.setRNGSeed(7)
    ok,rvec,tvec,inliers=cv2.solvePnPRansac(xyz,uv,np.asarray(camera_k),np.asarray(distortion),iterationsCount=150,reprojectionError=2.5,confidence=.99,flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers)<30 or len(inliers)/len(xyz)<.5:
        return {'valid':False,'reason':'insufficient_ransac_inliers','correspondences':len(xyz),'inliers':len(inliers) if inliers is not None else 0}
    indexes=inliers.reshape(-1)
    cv2.solvePnP(xyz[indexes],uv[indexes],np.asarray(camera_k),np.asarray(distortion),rvec,tvec,True,flags=cv2.SOLVEPNP_ITERATIVE)
    projected,_=cv2.projectPoints(xyz[indexes],rvec,tvec,np.asarray(camera_k),np.asarray(distortion))
    rms=float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-uv[indexes])**2,axis=1))))
    return {'valid':rms<=2.0,'reason':'accepted_relative_camera_motion' if rms<=2 else 'reprojection_error',
            'correspondences':len(xyz),'inliers':len(indexes),'reprojection_rms_px':rms,
            'rotation_vector_rad':rvec.reshape(-1).tolist(),'translation_vector_m':tvec.reshape(-1).tolist(),
            'translation_norm_m':float(np.linalg.norm(tvec))}


def screen_pairs(pairs,camera):
    import cv2
    import numpy as np
    detector=cv2.ORB_create(nfeatures=1800)
    matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    results=[]
    previous=None
    for sample in pairs:
        gray=cv2.cvtColor(sample['rgb'],cv2.COLOR_RGB2GRAY)
        keypoints,descriptors=detector.detectAndCompute(gray,None)
        current=(sample,keypoints,descriptors)
        if previous is None:
            previous=current;continue
        old,old_kp,old_des=previous
        previous=current
        summary={'from_stamp_s':old['stamp_s'],'to_stamp_s':sample['stamp_s']}
        if old_des is None or descriptors is None:
            results.append(dict(summary,valid=False,reason='insufficient_rgb_features'));continue
        matches=[items[0] for items in matcher.knnMatch(old_des,descriptors,k=2) if len(items)==2 and items[0].distance<.75*items[1].distance]
        xyz,uv,cells=[],[],set()
        for match in matches:
            u,v=old_kp[match.queryIdx].pt
            x,y=round(u),round(v)
            if not 1<=x<camera.width-1 or not 1<=y<camera.height-1:
                continue
            neighborhood=old['aligned_depth'][y-1:y+2,x-1:x+2]
            values=neighborhood[np.isfinite(neighborhood)&(neighborhood>=.3)&(neighborhood<=5)]
            if len(values)<3 or float(values.max()-values.min())>.06:
                continue
            z=float(np.median(values))
            normal=cv2.undistortPoints(np.array([[[u,v]]],dtype=np.float64),camera.k,camera.d).reshape(2)
            xyz.append([normal[0]*z,normal[1]*z,z])
            uv.append(keypoints[match.trainIdx].pt)
            cells.add((min(2,int(u/camera.width*3)),min(2,int(v/camera.height*3))))
        if len(cells)<4:
            results.append(dict(summary,valid=False,reason='features_not_spatially_distributed',correspondences=len(xyz)));continue
        result=pnp_motion(xyz,uv,camera.k,camera.d)
        results.append(dict(summary,**result))
    return {'kind':'real_rgbd_relative_camera_motion_screening','pairs':results,
            'accepted_pairs':sum(r['valid'] for r in results),'attempted_pairs':len(results),
            'continuous_chain':bool(results) and all(r['valid'] for r in results),
            'base_pose_usable':False,'global_localization_validated':False,
            'note':'仅评估当前图像/深度是否支持帧间PnP；没有独立位姿真值、回环、车体外参或漂移校正，不能当整车定位验收。'}
