import copy
import numpy as np
import open3d as o3d
from probreg import cpd
from optimize_rot_scipy import affine_alignment
from umeyama_ransac import getRANSACInliers,WeightedRansacAffine3D
from utils import transform_pointcloud

def ecpd(source_point,target_point,correspondence,save=True):
    if save:
      data = {'source_point':source_point,
        'target_point':target_point,
        'correspondence':correspondence}
      with open('ecpd_debug.pkl','wb') as f:
          pickle.dump(data,f)
    
    # 有负作用
    # affine_estimater = WeightedRansacAffine3D(with_scale='affine') # 用整个物体算初值，服务于非刚性点云配准
    # scales,rotation,t,sim_transform  = affine_estimater.weighted_affine_alignment(source_point[correspondence[:,0]].transpose(),target_point[correspondence[:,1]].transpose())
    # # source_point = transform_pointcloud(source_point,sim_transform)
    
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_point)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_point)

    tf_param, _, _ = cpd.registration_cpd(source_pcd, target_pcd,tf_type_name='nonrigid_constrained',
                                          idx_source=correspondence[:,0],
                                        idx_target=correspondence[:,1],
                                          alpha=1e-3,
                                        beta=0.1,   # （径向基函数）核的参数，控制点云变形的影响范围。值越大，影响范围越大，变形越平滑。
                                        lmd=2.0,    # 正则化参数，控制模型的平滑性。值越大，越倾向于产生平滑的变形。
                                        w=0,
                                        tol=1e-7,
                                        
                                          maxiter=50)

    transfered_source_point = np.asarray(tf_param.transform(source_pcd.points))   # n,3

    tmp_pcd = copy.deepcopy(source_pcd)
    tmp_pcd.points = tf_param.transform(source_pcd.points)
    source_pcd.paint_uniform_color([1, 0, 0])
    tmp_pcd.paint_uniform_color([1, 1, 0])
    target_pcd.paint_uniform_color([0, 1, 0])
    # o3d.visualization.draw_geometries([source_pcd,target_pcd,tmp_pcd ])

    distance = np.linalg.norm(transfered_source_point[:,None] - target_point[None],axis=-1) # n,m,3 -> n,m
    
    new_corr_target_id = np.argmin(distance,axis=-1)   # n
    new_corr_source_id = np.arange(distance.shape[0])
    new_corr_set_from_source = np.stack([new_corr_source_id,new_corr_target_id]).transpose(1,0)  # n,2
    distance_mask = distance[new_corr_set_from_source[:,0],new_corr_set_from_source[:,1]] < 0.01
    new_corr_set_from_source = new_corr_set_from_source[distance_mask]    # n',
    
    new_corr_source_id = np.argmin(distance,axis=0)
    new_corr_target_id = np.arange(distance.shape[1])
    new_corr_set_from_target = np.stack([new_corr_source_id,new_corr_target_id]).transpose(1,0)  # n,2
    distance_mask = distance[new_corr_set_from_target[:,0],new_corr_set_from_target[:,1]] < 0.01
    new_corr_set_from_target = new_corr_set_from_target[distance_mask]    # n',

    # scales, rotation, translation, transform = affine_alignment(source_point.transpose(),transfered_source_point.transpose())

    if len(new_corr_set_from_source)>len(correspondence)*0.25 and len(new_corr_set_from_target)>len(correspondence)*0.25:
        new_corr_set = np.concatenate([new_corr_set_from_source,new_corr_set_from_target],axis=0)
        print('ecpd success')
    else:
        new_corr_set = correspondence    
        print('ecpd failed')
    return new_corr_set # , rotation

import pickle
if __name__ == '__main__':
    with open('ecpd_debug.pkl','rb') as f:
        data = pickle.load(f)
    source_point = data['source_point']
    target_point = data['target_point']
    correspondence = data['correspondence']
    # correspondence = np.arange(len(source_point))[:,None]
    new_corr_set = ecpd(source_point,target_point,correspondence,save=False)