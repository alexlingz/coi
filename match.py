# category-level point match
# step1 feature extraction
# step2 feature matching and group-based local filtering
from scipy.stats import zscore
from torchmetrics.functional import pairwise_cosine_similarity
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from utils import calculate_2d_projections, select_keypoint, rotation_to_quaternion_np, quaternion_to_matrix_np,transform_pointcloud_torch
import cv2
import random
import open3d as o3d
# import sys
# sys.setrecursionlimit(1000)
from diffusion_pruning import Prune
from umeyama_ransac import getRANSACInliers,WeightedRansacAffine3D
import matplotlib.pyplot as plt
from functional_map import RegularizedFM
from utils import select_keypoint, transform_pointcloud, remove_outliers_with_open3d
from optimize_rot_scipy import affine_alignment
from optimize_rot_torch import run_optimize
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from ecpd import ecpd
import sys
sys.path.append('/home/yan20/tianshuwu/coi/vfc_pybind/build')
import pybind_vfc
import time
from scipy.stats import zscore
from pytorch3d.ops import sample_farthest_points

class ZeroRecursor:
    def __init__(self,selected_flag,walk_through_flag,source_neighbor_indices,target_neighbor_indices,final_match_indices):
        self.selected_flag = selected_flag
        self.walk_through_flag = walk_through_flag
        self.source_neighbor_indices = source_neighbor_indices
        self.target_neighbor_indices = target_neighbor_indices
        self.final_match_indices = final_match_indices

class Matcher:
    def __init__(self,save_dir):
        self.feature_extracter = FeatureExtracter()
        self.save_dir = save_dir

    def match_weighted_affine(self,frame_source,frame_target):

        match = self.match_samplek(frame_source,frame_target).cpu().numpy()

        estimator = WeightedRansacAffine3D(a=0.0,with_scale='sim')
        source_keypoint = select_keypoint(frame_source)
        target_keypoint = select_keypoint(frame_target)
        inliers,affine_transform,rigid_transform = estimator.estimate(match[:,:3], source_keypoint, match[:,3:], target_keypoint)

        match_visualization = self.draw_match(frame_source,frame_target,torch.tensor(match[inliers]),draw_num=32)
        plt.figure(figsize=(16, 8)) 
        plt.imshow(match_visualization)
        plt.axis('off')  # 隐藏坐标轴
        plt.show()

        self.show_open3d(self.match_nn(frame_source,frame_target),rigid_transform,np.array([[1,0,0,source_keypoint[0,0]],[0,1,0,source_keypoint[0,1]],[0,0,1,source_keypoint[0,2]],[0,0,0,1]]))

        return match[inliers], affine_transform,source_keypoint,target_keypoint,rigid_transform
    
    def match_nn_id(self,frame_source,frame_target,which_obj):
        point_source,feature_source, color_source = self.feature_extracter.extract(frame_source,which_obj)  
        point_target,feature_target, color_target = self.feature_extracter.extract(frame_target,which_obj)
        similarity_map = pairwise_cosine_similarity(feature_source,feature_target)  # ns,nt

        # source2target
        source2target_id = torch.argmax(similarity_map,dim=1).cpu().numpy()  # ns
        correspondence_set_s2t = torch.tensor(np.stack([np.arange(source2target_id.shape[0]),source2target_id]).transpose(),device='cuda')     # ns,2

        # target2source
        target2source_id = torch.argmax(similarity_map,dim=0).cpu().numpy()  # nt
        correspondence_set_s2t_2 = torch.tensor(np.stack([target2source_id,np.arange(target2source_id.shape[0])]).transpose(),device='cuda')     # nt,2

        correspondence_set = torch.cat([correspondence_set_s2t,correspondence_set_s2t_2])    # ns+nt,2
        # 只取相似度较高的match
        similarity_score = similarity_map[correspondence_set[:,0],correspondence_set[:,1]]  # ns+nt
        _, similarity_selected_by_score = torch.topk(similarity_score,int(correspondence_set.shape[0]*1.0))  
        correspondence_set = correspondence_set[similarity_selected_by_score].cpu().numpy()

        return correspondence_set, point_source.cpu().numpy(), point_target.cpu().numpy(),feature_source,feature_target, color_source.cpu().numpy(), color_target.cpu().numpy()

    
    def match_vfc_image(self,frame_source,frame_target,which_obj,vis=False):
    

        time_start = time.time()

        # match = self.match_nn(frame_source,frame_target,which_obj).cpu().numpy()
        correspondence_set, point_source, point_target,feature_source,feature_target,color_source,color_target = self.match_nn_id(frame_source,frame_target,which_obj)

        np.save(f'{self.save_dir}/{which_obj}_point_source.npy',point_source)
        np.save(f'{self.save_dir}/{which_obj}_point_target.npy',point_target)
        np.save(f'{self.save_dir}/{which_obj}_nn_match_result_id.npy',correspondence_set)
        np.save(f'{self.save_dir}/{which_obj}_feature_source.npy',feature_source.cpu().numpy())
        np.save(f'{self.save_dir}/{which_obj}_feature_target.npy',feature_target.cpu().numpy())
        np.save(f'{self.save_dir}/{which_obj}_color_source.npy',color_source)
        np.save(f'{self.save_dir}/{which_obj}_color_target.npy',color_target)
        
        vfc = pybind_vfc.VFC()
        # 50 0.9 3.0 0.1 0.75 10.0
        vfc.setMaxIter(100) 
        # print(vfc.getMaxIter())
        vfc.setGamma(0.7)      # 估计潜在的正确匹配数量
        # print(vfc.getGamma())  
        vfc.setLambda(3.0)      #   0.1-10  平滑程度，较大的值强调平滑型，忽略局部细节，较小的值允许更多自由度
        # print(vfc.getLambda())  
        vfc.setBeta(0.1)      #  0.01-1
        # print(vfc.getBeta())  
        vfc.setTheta(0.75)      #  0.6 - 0.9    匹配点的置信度阈值。用于决定哪些点被认为是正确匹配。
        # print(vfc.getTheta())     
        vfc.setA(10.0)   
        vfc.setNumCtrlPts(16)   # 控制点数量，仅在稀疏版本（SPARSE_VFC）中使用。通过减少控制点数量来提高效率。
        match_2d_source = calculate_2d_projections(point_source[correspondence_set[:,0]].transpose(),np.array([[frame_source['intrinsics']['fx'],0,frame_source['intrinsics']['cx']],
                                    [0,frame_source['intrinsics']['fy'],frame_source['intrinsics']['cy']],
                                    [0,0,1]]))
        match_2d_target = calculate_2d_projections(point_target[correspondence_set[:,1]].transpose(),np.array([[frame_source['intrinsics']['fx'],0,frame_source['intrinsics']['cx']],
                                    [0,frame_source['intrinsics']['fy'],frame_source['intrinsics']['cy']],
                                    [0,0,1]]))
        vfc_iter = 0
        while True:
            vfc_iter = vfc_iter+1
            inlier = pybind_vfc.process_and_return_correct_matches(frame_source['rgb'].cpu().numpy(),frame_target['rgb'].cpu().numpy(),np.concatenate([match_2d_source,match_2d_target],axis=1).astype('float').tolist(),"./vfc_pybind/result/tmp_result.txt",vfc)
            print('vfc results:', len(np.concatenate([match_2d_source,match_2d_target],axis=1)),'->',len(inlier))   # inlier是id的list，即[0,1,3,4,...]
            if len(inlier) > len(match_2d_source)*(0.6-vfc_iter*0.05):
                break
            if vfc_iter > 5:
                inlier = np.arange(len(np.concatenate([match_2d_source,match_2d_target],axis=1)))
                break
        time_vfc = time.time()
        print('time_vfc cost:',time_vfc-time_start)
        landmark_match_id = correspondence_set[inlier]
        
        # 直接不做vfc?
        # landmark_match_id = correspondence_set
        
        np.save(f'{self.save_dir}/{which_obj}_vfc_match_result_id.npy',landmark_match_id)

        
        affine_estimater = WeightedRansacAffine3D(with_scale='sim') # 用整个物体算初值，服务于非刚性点云配准
        # scales,rotation,t,sim_transform  = affine_estimater.weighted_affine_alignment(point_source[landmark_match_id[:,0]].transpose(),point_target[landmark_match_id[:,1]].transpose())
        # landmark_match_id = ecpd(transform_pointcloud(point_source,sim_transform),point_target,landmark_match_id)  # 感觉耗时有点长，而且效果不可靠，不如仅vfc
        
        landmark_match_id = ecpd(point_source,point_target,landmark_match_id)  # 感觉耗时有点长，而且效果不可靠，不如仅vfc
        

        np.save(f'{self.save_dir}/{which_obj}_ecpd_match_result_id.npy',landmark_match_id)

        inlier_source_point = point_source[landmark_match_id[:,0]]
        inlier_target_point = point_target[landmark_match_id[:,1]]
        
        if which_obj == 'positive':
            inlier_rigid_mask = np.linalg.norm(inlier_source_point[:,None] - frame_source['rigid_point'][None],axis=-1).min(axis=1)<0.01 # n,m,3 -> n,m -> n
            rigid_mask= np.linalg.norm(point_source[correspondence_set[:,0]][:,None] - frame_source['rigid_point'][None],axis=-1).min(axis=1)<0.01
        else:
            inlier_rigid_mask = np.ones(inlier_source_point.shape[0],dtype=bool)
            rigid_mask = np.ones(point_source[correspondence_set[:,0]].shape[0],dtype=bool)
        
        # scales,rotation,_,_ = affine_alignment(inlier_source_point[rigid_mask].transpose(),inlier_target_point[rigid_mask].transpose(),with_scale='sim')
        
        # 只用rigid部分来算rigid部分之间的相对变换，用非刚性点云配准的结果来算
        scales,rotation,t,_  = affine_estimater.weighted_affine_alignment(inlier_source_point[inlier_rigid_mask].transpose(),inlier_target_point[inlier_rigid_mask].transpose())

        source_keypoint = frame_source[f'{which_obj}_keypoint']    # 1，3
        # 从inlier的match中找最接近的对应点  不可靠，还是用dino+vfc
        # distance = np.linalg.norm((source_keypoint-point_source[new_corr_set[:,0]]),axis=1)
        # target_keypoint = point_target[new_corr_set[:,1]][np.argsort(distance)[:10]].mean(axis=0)
        # # 从dino+vfc的结果中拿匹配点
        distance = np.linalg.norm((source_keypoint-point_source[landmark_match_id[:,0]]),axis=1)
        target_keypoint_candidate = point_target[landmark_match_id[:,1]][np.argsort(distance)[:10]]   # .mean(axis=0,keepdims=True)
        source_keypoint_neighbor = point_source[landmark_match_id[:,0]][np.argsort(distance)[:10]]

        z_scores = np.abs(zscore(target_keypoint_candidate, axis=0))
        z_scores = np.nan_to_num(z_scores, nan=0.0)
        target_keypoint_inlier_mask = (z_scores<2).all(axis=1)
        target_keypoint_center = target_keypoint_candidate[target_keypoint_inlier_mask].mean(axis=0,keepdims=True)
        source_keypoint_center = source_keypoint_neighbor[target_keypoint_inlier_mask].mean(axis=0,keepdims=True)
        target_keypoint = source_keypoint + target_keypoint_center - source_keypoint_center

        # 优化
        # optimized_scales,optimized_rotation,optimized_t,deformation = run_optimize(point_source[correspondence_set[:,0]],feature_source[correspondence_set[:,0]],
        #              point_target[correspondence_set[:,1]],feature_target[correspondence_set[:,1]],
        #              rigid_mask,inlier,
        #              scales,rotation,t
        #              )
        # keypoint_id = np.argmin(np.linalg.norm(point_source[correspondence_set[:,0]]-source_keypoint,axis=1))
        # cs2ct = np.eye(4)
        # cs2ct[:3,:3] = optimized_rotation @ np.diag(optimized_scales)
        # cs2ct[:3,3] = optimized_t
        # target_keypoint = transform_pointcloud(source_keypoint,cs2ct) + deformation[keypoint_id]

        # 不优化 
        optimized_rotation = rotation
        
        # 优化后，

        final_transform = np.eye(4)
        final_transform[:3,:3] = optimized_rotation
        final_transform[:3,3] = target_keypoint[0] - optimized_rotation @ source_keypoint[0] 

        time_end = time.time()
        print('vfc+ecpd cost time:',time_end-time_start)


        if vis:
            # self.feature_extracter.show_patch_feature(frame_source,frame_target,which_obj,show_background=False)
            # self.feature_extracter.show_point_feature(frame_source,frame_target,which_obj)
            # self.show_open3d(self.match_nn(frame_source,frame_target,which_obj),final_transform,np.array([[1,0,0,source_keypoint[0,0]],[0,1,0,source_keypoint[0,1]],[0,0,1,source_keypoint[0,2]],[0,0,0,1]]))
            # self.show_match_color(np.concatenate([point_source[correspondence_set[:,0]],point_target[correspondence_set[:,1]]],axis=1))   # dino的结果
            # self.show_match_color(np.concatenate([point_source[correspondence_set[inlier][:,0]],point_target[correspondence_set[inlier][:,1]]],axis=1))   # vfc的结果
            # self.show_match_color(np.concatenate([point_source[landmark_match_id[:,0]],point_target[landmark_match_id[:,1]]],axis=1))   # ecpd的结果
            pass
        # return np.concatenate([point_source[new_corr_set[:,0]],point_target[new_corr_set[:,1]]]), final_transform, source_keypoint,target_keypoint
        return np.concatenate([inlier_source_point,inlier_target_point],axis=1), final_transform, source_keypoint,target_keypoint, scales,inlier_target_point[inlier_rigid_mask]


    def match_rigid(self,frame_source,frame_target):

        # 在变形不大的情况下非常work

        point_source,feature_source = self.feature_extracter.extract(frame_source)  # 固定数量，例如512？
        point_target,feature_target = self.feature_extracter.extract(frame_target)

        pcd_source = o3d.geometry.PointCloud()
        pcd_source.points = o3d.utility.Vector3dVector(point_source.cpu().numpy())
        pcd_feature_source = o3d.pipelines.registration.Feature()
        pcd_feature_source.data = feature_source.detach().cpu().numpy().T

        pcd_target = o3d.geometry.PointCloud()
        pcd_target.points = o3d.utility.Vector3dVector(point_target.cpu().numpy())
        pcd_feature_target = o3d.pipelines.registration.Feature()
        pcd_feature_target.data = feature_target.detach().cpu().numpy().T

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd_source,      # source
            pcd_target,       # target
            pcd_feature_source,
            pcd_feature_target,
            False,
            0.01,
            # checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
            #                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.01)],
            # checker=[],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000,1.0)
        )

        correspondence_set = np.asarray(result.correspondence_set)
        match = torch.cat([point_source[correspondence_set[:,0]],point_target[correspondence_set[:,1]]],dim=1)
        print('open3d match inliers:', match.shape[0])

        return match
    
    # def match_scale(self,frame_source,frame_target):

    #     # 错误的，似乎在inlier很少时，check on distance必须在两个scale上都添加约束。

    #     match = self.match_samplek(frame_source,frame_target)    # 512,6

    #     pcd_source = o3d.geometry.PointCloud()
    #     pcd_source.points = o3d.utility.Vector3dVector(match[:,:3].cpu().numpy())

    #     pcd_target = o3d.geometry.PointCloud()
    #     pcd_target.points = o3d.utility.Vector3dVector(match[:,3:].cpu().numpy())

    #     correspondence = np.arange(match.shape[0])[:,None].repeat(2,axis=1) # n,2
    #     correspondence = o3d.utility.Vector2iVector(correspondence)
        
    #     result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
    #         pcd_source,
    #         pcd_target,
    #         correspondence,
    #         0.005,
    #         o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
    #         ransac_n=5,
    #         checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
    #         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.005)],
    #     )

    #     return match[result.correspondence_set], result.transformation

    def match_umeyama(self,frame_source,frame_target):

        # match = self.match_nn(frame_source,frame_target)
        match = self.match_samplek(frame_source,frame_target)    # 512,6
        # 考虑局部一致性的nn
        # match = self.match_local_group_filter(frame_source,frame_target)    # 512,6
        # 刚性ransac的结果的match，较准确
        # match = self.match_ransac(frame_source,frame_target)

        # s2t
        point_source_um, point_target_um,sim_transform = getRANSACInliers(
            # point_source[correspondence_set[:,0]].cpu().numpy(),
            # point_target[correspondence_set[:,1]].cpu().numpy(),
            match[:,:3].cpu().numpy(),
            match[:,3:].cpu().numpy()
        )

        match = np.concatenate([point_source_um,point_target_um],axis=1)
        print('umeyama result:', match.shape[0])

        match_visualization = self.draw_match(frame_source,frame_target,torch.tensor(match),draw_num=32)
        plt.figure(figsize=(16, 8)) 
        plt.imshow(match_visualization)
        plt.axis('off')  # 隐藏坐标轴
        plt.show()

        return match, sim_transform

    def match_nn(self,frame_source,frame_target,which_obj):

        # nn取match
        point_source,feature_source,_ = self.feature_extracter.extract(frame_source,which_obj)  # 固定数量，例如512？
        point_target,feature_target,_ = self.feature_extracter.extract(frame_target,which_obj)
        similarity_map = pairwise_cosine_similarity(feature_source,feature_target)  # 512,512

        # source2target
        source2target_id = torch.argmax(similarity_map,dim=1).cpu().numpy()  # 512
        correspondence_set_s2t = torch.tensor(np.stack([np.arange(source2target_id.shape[0]),source2target_id]).transpose(),device='cuda')     # 512,2


        match_s2t = np.concatenate([point_source[correspondence_set_s2t[:,0]].cpu().numpy(),
             point_target[correspondence_set_s2t[:,1]].cpu().numpy(),],axis=1)

        # target2source
        target2source_id = torch.argmax(similarity_map,dim=0).cpu().numpy()  # 512
        correspondence_set_t2s = torch.tensor(np.stack([target2source_id,np.arange(target2source_id.shape[0])]).transpose(),device='cuda')     # 512,2
        match_t2s = np.concatenate([point_source[correspondence_set_t2s[:,0]].cpu().numpy(),
             point_target[correspondence_set_t2s[:,1]].cpu().numpy(),],axis=1)

        match = np.concatenate([match_s2t,match_t2s],axis=0)
        print('nn result:', match.shape[0])

        return torch.tensor(match)
    

    def match_samplek(self,frame_source,frame_target,which_obj):
        # nn取match
        point_source,feature_source = self.feature_extracter.extract(frame_source,which_obj)  # 固定数量，例如512？
        point_target,feature_target = self.feature_extracter.extract(frame_target,which_obj)
        similarity_map = pairwise_cosine_similarity(feature_source,feature_target)  # 512,512
        similarity_map = ((similarity_map+1)/2)**12    # 先放缩到0～1之间，再用幂次变换扩大相似度值之间的差距    


        sample_number = 5
        # source2target
        source2target_id = torch.multinomial(similarity_map,sample_number,).unsqueeze(-1)    # 512,5,1
        source_id = torch.arange(source2target_id.shape[0]).unsqueeze(-1).unsqueeze(-1).repeat(1,sample_number,1).to('cuda')  # 512,5,1
        correspondence_set_s2t = torch.cat([source_id,source2target_id],dim=-1).reshape(-1,2)     # 512,5,2 -> 512*5,2
        match_s2t = np.concatenate([point_source[correspondence_set_s2t[:,0]].cpu().numpy(), point_target[correspondence_set_s2t[:,1]].cpu().numpy(),],axis=1)

        # source2target
        target2source_id = torch.multinomial(similarity_map.permute(1,0),sample_number,).unsqueeze(-1)    # 512,5,1
        target_id = torch.arange(target2source_id.shape[0]).unsqueeze(-1).unsqueeze(-1).repeat(1,sample_number,1).to('cuda')  # 512,5,1
        correspondence_set_t2s = torch.cat([target2source_id,target_id],dim=-1).reshape(-1,2)     # 512,5,2 -> 512*5,2
        match_t2s = np.concatenate([point_source[correspondence_set_t2s[:,0]].cpu().numpy(), point_target[correspondence_set_t2s[:,1]].cpu().numpy(),],axis=1)

        match = np.concatenate([match_s2t,match_t2s],axis=0)
        print('sample result:', match.shape[0])

        return torch.tensor(match)

    # def match_estimateAffine3D(self,frame_source,frame_target):
    #     # 考虑局部一致性的nn
    #     # match = self.match_local_group_filter(frame_source,frame_target)    # 512,6
    #     # 刚性ransac的结果的match，较准确
    #     # match = self.match_ransac(frame_source,frame_target).cpu().numpy().astype('float32')
    #     # match = self.match_umeyama(frame_source,frame_target)  # 512,6
    #     match = self.match_samplek(frame_source,frame_target).cpu().numpy()

    #     retval, Rt, inliers = cv2.estimateAffine3D(match[:,:3].astype('float32'), match[:,3:].astype('float32'), ransacThreshold=0.02,confidence=0.999)

    #     match = np.concatenate([match[inliers[:,0].astype(bool)][:,:3],match[inliers[:,0].astype(bool)][:,3:]],axis=1)
    #     print('opencv affine3d result:', match.shape[0])

    #     return match

    # def match_pruning(self,frame_source,frame_target):
    #     point_source,feature_source = self.feature_extracter.extract(frame_source)  # 固定数量，例如512？
    #     point_target,feature_target = self.feature_extracter.extract(frame_target)
    #     similarity_map = pairwise_cosine_similarity(feature_source,feature_target)  # 512,512

    #     target_id = torch.argmax(similarity_map,dim=1).cpu().numpy()  # 512
    #     correspondence_set = np.stack([np.arange(target_id.shape[0]),target_id]).transpose()     # 512,2

    #     point_source, point_target = point_source.cpu().numpy(), point_target.cpu().numpy()
    #     pruner = Prune()
    #     pruner.initialize(point_source,point_target,correspondence_set)
    #     pruned_correspondence_set = pruner.run_pruning(correspondence_set)

    #     match = np.concatenate([point_source[pruned_correspondence_set[:,0]],point_target[pruned_correspondence_set[:,1]]],axis=1)

    #     return match

    # def match_functional_map(self,frame_source,frame_target):
    #     point_source,feature_source = self.feature_extracter.extract(frame_source)  # 固定数量，例如512？
    #     point_target,feature_target = self.feature_extracter.extract(frame_target)
    #     fmmatcher = RegularizedFM()
    #     source2target_id = fmmatcher.run(feat_x=feature_source,feat_y=feature_target,point_x=point_source,point_y=point_target)

    #     correspondence_set_s2t = torch.stack([torch.arange(source2target_id.shape[0],device='cuda'),source2target_id]).permute(1,0)
    #     match_s2t = np.concatenate([point_source[correspondence_set_s2t[:,0]].cpu().numpy(), point_target[correspondence_set_s2t[:,1]].cpu().numpy(),],axis=1)

    #     return match_s2t


    # def match_local_group_filter(self,frame_source,frame_target):
    #     point_source,feature_source = self.feature_extracter.extract(frame_source)  # 固定数量，例如512？
    #     point_target,feature_target = self.feature_extracter.extract(frame_target)
    #     similarity_map = pairwise_cosine_similarity(feature_source,feature_target)  # 512,512

    #     # 以corl workshop的思路，手动过滤，过滤的原则是邻居匹配一致性，同时基于特征相似度的权重来取匹配点，避免局部最优？   TODO 但是如果邻居的match全是局部最优，则该思路也会失效，需要通过某种方法实现聚合
    #     # step1 为每个点以相似度计算权重，来采样5个可能的match
    #     similarity_map = ((similarity_map+1)/2)**10    # 先放缩到0～1之间，再用幂次变换扩大相似度值之间的差距    
    #     match_candidate = torch.multinomial(similarity_map,10,)      # 512,5
    #     # step2 做group-based local filtering: match和邻居的match得是邻居

    #     # 获得邻居的索引
    #     source_distances = torch.cdist(point_source,point_source)
    #     _, source_neighbor_indices = torch.topk(source_distances, 15+1, largest=False)  # indices: 512,9

    #     target_distances = torch.cdist(point_target,point_target)
    #     _, target_neighbor_indices = torch.topk(target_distances, 15+1, largest=False)  # indices: 512,9

    #     # 获得邻居的match candidate
    #     match_candidate_group = match_candidate[source_neighbor_indices]  # 512,9,5  match(target)序号
    #     match_candidate_group_point = point_target[match_candidate_group]   # 512,9,5,3  match(target)坐标

    #     # 算出group对应的target group的距离矩阵，只求与group中心点的距离
    #     # 求5,3 与 8,5,3 的距离矩阵，即5,8,5
    #     match_candidate_group_point_center = match_candidate_group_point[:,0,:,:].unsqueeze(2).unsqueeze(3)  # 512,5,3 -> 512,5,1,1,3
    #     match_candidate_group_point_neighbor = match_candidate_group_point[:,1:,:,:].unsqueeze(1)  # 512,8,5,3 -> 512,1,8,5,3
    #     match_candidate_group_distance = torch.norm(match_candidate_group_point_center - match_candidate_group_point_neighbor, dim=-1)  # 512,5,8,5
    #     # 认为一致性最高的是正确的match
    #     local_match_indices = torch.min(torch.sum(torch.min(match_candidate_group_distance,dim=-1)[0],dim=-1),dim=-1)[1]   # 512,5,8,5 min-> 512,5,8 sum-> 512,5 min-> 512个indice(在5中的)
    #     final_match_indices = match_candidate_group[torch.arange(local_match_indices.shape[0]),0,local_match_indices]   # 512,9,5 -> 512,5 -> 512

    #     # 在全局用阈值移除outlier   # 不work
    #     # inliers = self.zero_recursion(source_neighbor_indices,target_neighbor_indices,final_match_indices,iter_num=10)

    #     final_match = torch.cat([point_source,point_target[final_match_indices]],dim=-1)    # 512,6

    #     # TODO(优先级低)
    #     # 1.不仅仅以center为match的位置，group中选取的其他match也会影响最终的match位置
    #     # 2.对final match做一个过滤，平均距离过大的丢弃
    #     # TODO(优先级高)
    #     # 如何聚合出全局信息？
    #     # 从几个点出发，计算几个候选match的代价
    #     # 递归地计算邻居的候选match的代价，同时要考虑到距离的代价

    #     return final_match

    # def zero_recursion(self,source_neighbor_indices,target_neighbor_indices,final_match_indices,iter_num=50):

    #     result = []

    #     for i in range(iter_num):
    #         starting_indice = torch.randint(0,len(final_match_indices),(1,)).item()
    #         selected_flag=torch.zeros(len(final_match_indices),dtype=torch.bool)
    #         selected_flag[starting_indice] = True
    #         walk_through_flag=torch.zeros(len(final_match_indices),dtype=torch.bool)
    #         walk_through_flag[starting_indice] = True
    #         zerorecursor = ZeroRecursor(
    #             selected_flag=selected_flag,
    #             walk_through_flag=walk_through_flag,
    #             source_neighbor_indices=source_neighbor_indices,
    #             target_neighbor_indices=target_neighbor_indices,
    #             final_match_indices=final_match_indices
    #             )
    #         self.zero_remove_outlier(zerorecursor,starting_indice)

    #         print(zerorecursor.selected_flag.sum())
    #         result.append(zerorecursor.selected_flag)

    #     result = torch.stack(result)    # 20,512
    #     return result[torch.max(result.sum(dim=-1),dim=0)[1]]            # 20,512 -> 20 -> 1 -> 512 bool


    # def zero_remove_outlier(self,zerorecursor,current_indice):
    #     current_match = zerorecursor.final_match_indices[current_indice]
    #     neighbors = zerorecursor.source_neighbor_indices[current_indice]
    #     for neighbor in neighbors[1:]:
    #         if zerorecursor.walk_through_flag[neighbor] is True:
    #             continue
    #         else:
    #             neighbor_match = zerorecursor.final_match_indices[neighbor]
    #             if neighbor_match in zerorecursor.target_neighbor_indices[current_match] and current_match in zerorecursor.target_neighbor_indices[neighbor_match]:
    #                 zerorecursor.selected_flag[neighbor] = True

    #     for neighbor in neighbors[1:]:
    #         if zerorecursor.selected_flag[neighbor] == True and zerorecursor.walk_through_flag[neighbor] == False:
    #             zerorecursor.walk_through_flag[neighbor] = True
    #             self.zero_remove_outlier(zerorecursor,neighbor)
    #         else:
    #             zerorecursor.walk_through_flag[neighbor] = True


    def draw_match(self,source_frame,target_frame,match,draw_num=10):

        def to_keypoints(coords):
            return [cv2.KeyPoint(x=float(coord[0]), y=float(coord[1]), size=1) for coord in coords]

        sampled_indices = random.sample(range(len(match)), draw_num)
        match = match[sampled_indices]

        img_source = source_frame['rgb'].cpu().numpy()
        intrinsics_source = np.array([[source_frame['intrinsics']['fx'],0,source_frame['intrinsics']['cx']],
                                    [0,source_frame['intrinsics']['fy'],source_frame['intrinsics']['cy']],
                                    [0,0,1]])
        match3d_source = match[:,:3].cpu().numpy()
        match2d_source = calculate_2d_projections(match3d_source.transpose(),intrinsics_source)

        img_target = target_frame['rgb'].cpu().numpy()
        intrinsics_target = np.array([[target_frame['intrinsics']['fx'],0,target_frame['intrinsics']['cx']],
                                    [0,target_frame['intrinsics']['fy'],target_frame['intrinsics']['cy']],
                                    [0,0,1]])
        match3d_target = match[:,3:].cpu().numpy()
        match2d_target = calculate_2d_projections(match3d_target.transpose(),intrinsics_target)


        match_result = cv2.drawMatches(img_source,to_keypoints(match2d_source),img_target,to_keypoints(match2d_target),[cv2.DMatch(i, i, 0) for i in range(len(match3d_source))],None)

        return match_result

    def draw_coordinate_transfer(self,source_frame,target_frame,sim_transform,source_init_obj2cam):

        image_source = source_frame['rgb'].cpu().numpy()
        intrinsics_source = np.array([[source_frame['intrinsics']['fx'],0,source_frame['intrinsics']['cx']],
                                    [0,source_frame['intrinsics']['fy'],source_frame['intrinsics']['cy']],
                                    [0,0,1]])

        image_target = target_frame['rgb'].cpu().numpy()
        intrinsics_target = np.array([[target_frame['intrinsics']['fx'],0,target_frame['intrinsics']['cx']],
                                    [0,target_frame['intrinsics']['fy'],target_frame['intrinsics']['cy']],
                                    [0,0,1]])

        source_rvec,_ = cv2.Rodrigues(source_init_obj2cam[:3,:3])       # 可能会有歧义，某两个轴调换了

        cv2.drawFrameAxes(image_source,
        intrinsics_source,
        np.array([0.,0.,0.,0.,0.]),
        source_rvec, source_init_obj2cam[:3,3], 0.05) 

        # target_o2c = sim_transform @ source_init_obj2cam
        # target_rvec,_ = cv2.Rodrigues((sim_transform[:3,:3]/np.linalg.norm(sim_transform[:3,:3],axis=1,keepdims=True)) @ source_init_obj2cam[:3,:3])
        target_o2c = np.eye(4)
        target_o2c[:3,3] = (sim_transform @ source_init_obj2cam)[:3,3]
        target_o2c[:3,:3] = (sim_transform[:3,:3]/np.linalg.norm(sim_transform[:3,:3],axis=1,keepdims=True)) @ source_init_obj2cam[:3,:3]

        target_rvec,_ = cv2.Rodrigues(target_o2c[:3,:3])

        cv2.drawFrameAxes(image_target,
        intrinsics_target,
        np.array([0.,0.,0.,0.,0.]),
        target_rvec, target_o2c[:3,3], 0.05) 

        return np.concatenate([image_source,image_target],axis=1), target_o2c

    def show_open3d(self,match,sim_transform,source_init_obj2cam):
        pcd_match_source = o3d.geometry.PointCloud()
        pcd_match_source.points = o3d.utility.Vector3dVector(match[:,:3])
        pcd_match_source.colors = o3d.utility.Vector3dVector(np.full((match.shape[0], 3), [1, 0, 0])) 
        
        source_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1,origin=[0,0,0])
        source_frame.transform(source_init_obj2cam)

        pcd_match_target = o3d.geometry.PointCloud()
        pcd_match_target.points = o3d.utility.Vector3dVector(match[:,3:])
        pcd_match_target.colors = o3d.utility.Vector3dVector(np.full((match.shape[0], 3), [0, 0, 1]))  # 蓝色
        
        target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1,origin=[0,0,0])
        target_o2c = np.eye(4)
        target_o2c[:3,3] = (sim_transform @ source_init_obj2cam)[:3,3]
        target_o2c[:3,:3] = (sim_transform[:3,:3]/np.linalg.norm(sim_transform[:3,:3],axis=1,keepdims=True)) @ source_init_obj2cam[:3,:3]
        target_frame.transform(target_o2c)
        
        transformed_source = transform_pointcloud(match[:,:3],sim_transform)
        pcd_match_transformed_source = o3d.geometry.PointCloud()
        pcd_match_transformed_source.points = o3d.utility.Vector3dVector(transformed_source)
        pcd_match_transformed_source.colors = o3d.utility.Vector3dVector(np.full((transformed_source.shape[0], 3), [1, 0.5, 0]))  # 橙色

        o3d.visualization.draw_geometries(
            [pcd_match_source,pcd_match_target,pcd_match_transformed_source,
             source_frame,target_frame,
             ]
        )

        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry([pcd_match_source,pcd_match_target,source_frame,target_frame])
        # vis.run()
        # vis.destroy_window()

    def show_match_color(self, match):
        # Extract coordinates for the first point cloud (x1, y1, z1)
        x1 = match[:, 0]
        y1 = match[:, 1]
        z1 = match[:, 2]

        cube_length_1 = 1.5*np.std(match[:, :3]) # 用3sigma方差来可视化吧
        point_center_1 = match[:, :3].mean(axis=(0))

        # Extract coordinates for the second point cloud (x2, y2, z2)
        x2 = match[:, 3]
        y2 = match[:, 4]
        z2 = match[:, 5]

        cube_length_2 = 1.5*np.std(match[:, 3:]) # 用3sigma方差来可视化吧
        point_center_2 = match[:, 3:].mean(axis=(0))

        # Normalize x1, y1, z1 to [0, 1] for RGB mapping
        norm_x1 = (x1 - np.min(x1)) / (np.max(x1) - np.min(x1))
        norm_y1 = (y1 - np.min(y1)) / (np.max(y1) - np.min(y1))
        norm_z1 = (z1 - np.min(z1)) / (np.max(z1) - np.min(z1))

        # Combine normalized x1, y1, z1 into RGB colors
        colors = np.stack([norm_x1, norm_y1, norm_z1], axis=1)

        # Convert RGB values to hexadecimal for Plotly
        rgb_colors = ['rgb({}, {}, {})'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]

        # Create subplots for visualizing both point clouds
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('source point', 'target point'),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            horizontal_spacing=0.05
        )

        # Point cloud 1
        trace1 = go.Scatter3d(
            x=x1,
            y=y1,
            z=z1,
            mode='markers',
            marker=dict(
                size=3,
                color=rgb_colors,  # Apply RGB colors
                opacity=0.5
            ),
            name='source point'
        )

        # Point cloud 2 (use the same colors as Point Cloud 1)
        trace2 = go.Scatter3d(
            x=x2,
            y=y2,
            z=z2,
            mode='markers',
            marker=dict(
                size=3,
                color=rgb_colors,  # Apply the same RGB colors
                opacity=0.5
            ),
            name='target point'
        )

        # Add traces to the figure
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=1, col=2)

        # Update layout
        fig.update_layout(
            title="3D Point Clouds with RGB Colors",
        scene=dict(
            xaxis=dict(title='X Axis', range=[point_center_1[0]-cube_length_1/2,point_center_1[0]+cube_length_1/2]),
            yaxis=dict(title='Y Axis', range=[point_center_1[1]-cube_length_1/2,point_center_1[1]+cube_length_1/2]),
            zaxis=dict(title='Z Axis', range=[point_center_1[2]-cube_length_1/2,point_center_1[2]+cube_length_1/2]),
            aspectmode='cube'
        ),
        scene2=dict(
            xaxis=dict(title='X Axis', range=[point_center_2[0]-cube_length_2/2,point_center_2[0]+cube_length_2/2]),
            yaxis=dict(title='Y Axis', range=[point_center_2[1]-cube_length_2/2,point_center_2[1]+cube_length_2/2]),
            zaxis=dict(title='Z Axis', range=[point_center_2[2]-cube_length_2/2,point_center_2[2]+cube_length_2/2]),
            aspectmode='cube'
        )
        )

        # Show the figure
        fig.show()
    



class FeatureExtracter:
    def __init__(self):
        self.device = 'cuda'
        self.dino = torch.hub.load(repo_or_dir="/home/yan20/Desktop/feepe/107/dinov2",source="local", model="dinov2_vitl14_reg", pretrained=False)
        self.dino.load_state_dict(torch.load('/home/yan20/Desktop/feepe/107/gripper_commit/dinov2_weights/dinov2_vitl14_reg4_pretrain.pth'))
        self.dino.to(self.device)
        self.dino.eval()
        self.dino_size = 448

        # self.sample_point_size = 512


    def extract(self,frame,which_obj,sample_point_size=2048):     # which_obj = positive or passive
        
        if 'cam2' in frame.keys(): 
            # frame = frame['cam1']
            
            return self.extract_multiview(frame,which_obj,sample_point_size=sample_point_size)
        else:
        
            rgb,depth,mask,intrinsics = frame['rgb'],frame['depth'],frame[f'{which_obj}_mask'],frame['intrinsics']
            cropped_rgb,cropped_mask,bbox_square = self.prepare_image(rgb=rgb,mask=mask)    # expanded

            with torch.no_grad():
                dino_feature_dinowise = self.dino.forward_features(cropped_rgb.unsqueeze(0))['x_norm_patchtokens']   # 1, (448/14)**2(num of tokens,1024), c(1536)
                dino_feature_dinowise = dino_feature_dinowise.permute(0,2,1).reshape(1,-1,self.dino_size//14,self.dino_size//14)    # 1,c,32,32 ， bbox,patch-level

                dino_feature_imagewise = self.get_imagewise_feature(dino_feature_dinowise,bbox_square,[rgb.shape[0],rgb.shape[1]])  # image,pixel-level
                pointcloud, dino_feature_pointwise,rgb_pointwise = self.get_pointwise_feature(depth,mask,intrinsics,dino_feature_imagewise,rgb)   # n,3; n,c
                
                # 过滤离群点
                # inlier_mask = torch.tensor((np.abs(zscore(pointcloud.cpu().numpy(), axis=0))<3.0).all(axis=1),device=self.device)
                # pointcloud, dino_feature_pointwise = pointcloud[inlier_mask], dino_feature_pointwise[inlier_mask]
                inlier = remove_outliers_with_open3d(pointcloud.cpu().numpy())
                pointcloud, dino_feature_pointwise,rgb_pointwise = pointcloud[inlier], dino_feature_pointwise[inlier], rgb_pointwise[inlier]
                
                if pointcloud.shape[0] > sample_point_size:
                    # fps_ids = farthest_point_sampling(pointcloud.cpu(),sample_point_size).to('cuda')
                    _,batch_ids = sample_farthest_points(pointcloud.unsqueeze(0),K=sample_point_size,random_start_point=True)
                    fps_ids = batch_ids[0]
                    
                    pointcloud, dino_feature_pointwise, rgb_pointwise = pointcloud[fps_ids], dino_feature_pointwise[fps_ids], rgb_pointwise[fps_ids]

                frame[f'{which_obj}_point_with_rgb'] = np.concatenate([pointcloud.cpu().numpy(),rgb_pointwise.cpu().numpy()],axis=1) # n,6
                
                return pointcloud, dino_feature_pointwise, rgb_pointwise

    def extract_multiview(self,multiview_frame,which_obj,sample_point_size=2048):     # which_obj = positive or passive
        
        pointcloud_list = []
        dino_feature_pointwise_list = []
        for cam in multiview_frame.keys():
            frame = multiview_frame[cam]
            
            rgb,depth,mask,intrinsics = frame['rgb'],frame['depth'],frame[f'{which_obj}_mask'],frame['intrinsics']
            cropped_rgb,cropped_mask,bbox_square = self.prepare_image(rgb=rgb,mask=mask)    # expanded

            with torch.no_grad():
                dino_feature_dinowise = self.dino.forward_features(cropped_rgb.unsqueeze(0))['x_norm_patchtokens']   # 1, (448/14)**2(num of tokens,1024), c(1536)
                dino_feature_dinowise = dino_feature_dinowise.permute(0,2,1).reshape(1,-1,self.dino_size//14,self.dino_size//14)    # 1,c,32,32 ， bbox,patch-level

                dino_feature_imagewise = self.get_imagewise_feature(dino_feature_dinowise,bbox_square,[rgb.shape[0],rgb.shape[1]])  # image,pixel-level
                pointcloud, dino_feature_pointwise,rgb_pointwise = self.get_pointwise_feature(depth,mask,intrinsics,dino_feature_imagewise,rgb)   # n,3; n,c
            
            pointcloud = transform_pointcloud_torch(pointcloud,torch.tensor(np.linalg.inv(multiview_frame['cam1']['cam2base'])@frame['cam2base'],device='cuda',dtype=torch.float32))
            pointcloud_list.append(pointcloud)     
            dino_feature_pointwise_list.append(dino_feature_pointwise)
        
        pointcloud = torch.cat(pointcloud_list,dim=0)
        dino_feature_pointwise = torch.cat(dino_feature_pointwise_list,dim=0)
        
        if pointcloud.shape[0] > sample_point_size:
            # fps_ids = farthest_point_sampling(pointcloud.cpu(),sample_point_size).to('cuda')
            _,batch_ids = sample_farthest_points(pointcloud.unsqueeze(0),K=sample_point_size,random_start_point=True)
            fps_ids = batch_ids[0]
            pointcloud, dino_feature_pointwise  = pointcloud[fps_ids], dino_feature_pointwise[fps_ids]
        inlier = remove_outliers_with_open3d(pointcloud.cpu().numpy())
        pointcloud, dino_feature_pointwise = pointcloud[inlier], dino_feature_pointwise[inlier]
            
        return pointcloud, dino_feature_pointwise,None
        
        
        
    def get_pointwise_feature(self,depth,mask,intrinsics,feature_map,rgb):
        # depth: h,w
        # feature_map: h,w,c
        # rgb: h,w,3
        depth_map = depth.clone()
        # 1, C, H, W = depth_map.shape
        _, C, H, W = feature_map.shape

        if mask is not None:
            depth_map[mask == 0] = -1

        # Create grid of pixel coordinates
        u, v = torch.meshgrid(torch.arange(W,device=depth_map.device), torch.arange(H,device=depth_map.device),indexing='xy')
        # Convert pixel coordinates to camera coordinates   h,w
        x = (u - intrinsics['cx']) * depth_map / intrinsics['fx']
        y = (v - intrinsics['cy']) * depth_map / intrinsics['fy']
        z = depth_map   # 

        # Reshape to (H*W)
        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)
        
        # Stack into point cloud
        pointcloud = torch.stack((x, y, z), dim=-1)
        pointcloud = pointcloud[z > 0]

        feature_pointwise = feature_map[0].reshape(C,-1).permute(1,0)
        feature_pointwise = feature_pointwise[z.cpu() > 0]

        rgb_pointwise = rgb.reshape((-1,3))
        rgb_pointwise = rgb_pointwise[z>0]

        return pointcloud, feature_pointwise.cuda(), rgb_pointwise


    def get_imagewise_feature(self,dino_feature_dinowise,bbox_square,image_size):
        h,w = image_size
        # 把448_feature恢复到图片尺寸，方便后续操作
        dino_feature_cropwise = self.get_cropwise_feature(bbox_square,dino_feature_dinowise,mode='bilinear')
        dino_feature_imagewise = torch.zeros((1,dino_feature_cropwise.shape[1],h,w),device='cpu')# self.device)
        y_min, x_min, y_max, x_max = bbox_square
        dino_feature_imagewise[:,:,y_min:y_max,x_min:x_max] = dino_feature_cropwise.to('cpu')

        return dino_feature_imagewise

    def prepare_image(self,rgb,mask,mask_ground=False):
        # rgb: h,w,3, np.uint8
        # mask: h,w, np.bool
        rgb = rgb.permute(2, 0, 1) / 255.0  # 3,h,w
        bbox_square = self.get_2dbboxes(mask)   # 4 

        y_min, x_min, y_max, x_max = bbox_square
        cropped_rgb = rgb[:,y_min:y_max, x_min:x_max]
        cropped_rgb = F.interpolate(cropped_rgb.unsqueeze(0),size=(self.dino_size,self.dino_size),mode='bilinear')[0]
        cropped_mask = mask[y_min:y_max, x_min:x_max]
        cropped_mask = F.interpolate(cropped_mask.unsqueeze(0).unsqueeze(0).to(torch.uint8),size=(self.dino_size,self.dino_size),mode='nearest')[0][0]
        cropped_rgb = transforms.Compose([
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # imagenet defaults
    ])(cropped_rgb)
        
        if mask_ground:
            cropped_rgb[cropped_mask.repeat(3,1,1) == 0] = 0
        
        return cropped_rgb,cropped_mask,bbox_square

    def get_2dbboxes(self,mask,expand=0.2):

        assert len(mask.shape) == 2     # make bbox square
        H, W = mask.shape

        # Find coordinates of non-zero elements in the mask
        non_zero_coords = torch.nonzero(mask.float())
        #print(non_zero_coords.shape)
        #print(non_zero_coords)

        # Extract bounding box coordinates
        ymin = non_zero_coords[:, 0].min()
        ymax = non_zero_coords[:, 0].max() + 1
        xmin = non_zero_coords[:, 1].min()
        xmax = non_zero_coords[:, 1].max() + 1

        if expand is not None:
            width_now = xmax-xmin
            xmax = min(xmax+width_now*expand, W)
            xmin = max(xmin-width_now*expand, 0)
            height_now = ymax-ymin
            ymax = min(ymax+height_now*expand, H)
            ymin = max(ymin-height_now*expand, 0)

        w = xmax-xmin
        h = ymax-ymin
        if h > w:
            dif = h-w
            xmin = xmin-dif//2
            xmax = xmax+dif//2
            if xmin < 0:
                xmax = xmax - xmin
                xmin = 0
            if xmax > W:
                xmin = xmin-xmax+W
                xmax = W
        elif w>h :
            dif = w-h
            ymin = ymin-dif/2
            ymax = ymax+dif/2
            if ymin < 0:
                ymax = ymax - ymin
                ymin = 0
            if ymax > H:
                ymin = ymin-ymax+H
                ymax = H
        ymin = max(ymin,0)
        xmin = max(xmin,0)
        ymax = min(ymax, H)
        xmax = min(xmax, W)

        # Store bounding box coordinates
        bbox = torch.tensor([ymin, xmin, ymax, xmax])
        bbox = torch.clamp(bbox,0)
        
        return bbox.int()

    def get_cropwise_feature(self,bbox,feature_map,mode='nearest'):
        assert len(feature_map.shape) == 4
        ymin, xmin, ymax, xmax = bbox
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        interpolate_feature = F.interpolate(feature_map,(bbox_height,bbox_width),mode=mode)
        return interpolate_feature
    
    def show_point_feature(self,frame1,frame2,which_obj):

        point1,feature1 = self.extract(frame1,which_obj=which_obj)
        point2,feature2 = self.extract(frame2,which_obj=which_obj)
        n1,n2 = point1.shape[0], point2.shape[0]
        point = torch.cat([point1,point2],dim=0).cpu().numpy()
        feature = torch.cat([feature1,feature2],dim=0).cpu().numpy()
        pca = PCA(n_components=3)
        pca.fit(feature)
        pca_feature = pca.transform(feature)
        color = minmax_scale(pca_feature)

        cube_length_1 = 1*np.std(point[:n1]) # 用3sigma方差来可视化吧
        point_center_1 = point[:n1].mean(axis=(0))

        cube_length_2 = 1*np.std(point[n1:]) # 用3sigma方差来可视化吧
        point_center_2 = point[n1:].mean(axis=(0))

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('source feature', 'target feature'),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            horizontal_spacing=0.05
        )

        # Point cloud 1
        trace1 = go.Scatter3d(
            x=point[:n1,0],
            y=point[:n1,1],
            z=point[:n1,2],
            mode='markers',
            marker=dict(
                size=3,
                color=color[:n1],  # Apply RGB colors
                opacity=0.5
            ),
            name='source feature'
        )

        # Point cloud 2 (use the same colors as Point Cloud 1)
        trace2 = go.Scatter3d(
            x=point[n1:,0],
            y=point[n1:,1],
            z=point[n1:,2],
            mode='markers',
            marker=dict(
                size=3,
                color=color[n1:],  # Apply the same RGB colors
                opacity=0.5
            ),
            name='target feature'
        )

        # Add traces to the figure
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=1, col=2)

        # Update layout
        fig.update_layout(
            title="feature",
        scene=dict(
            xaxis=dict(title='X Axis', range=[point_center_1[0]-cube_length_1/2,point_center_1[0]+cube_length_1/2]),
            yaxis=dict(title='Y Axis', range=[point_center_1[1]-cube_length_1/2,point_center_1[1]+cube_length_1/2]),
            zaxis=dict(title='Z Axis', range=[point_center_1[2]-cube_length_1/2,point_center_1[2]+cube_length_1/2]),
            aspectmode='cube'
        ),
        scene2=dict(
            xaxis=dict(title='X Axis', range=[point_center_2[0]-cube_length_2/2,point_center_2[0]+cube_length_2/2]),
            yaxis=dict(title='Y Axis', range=[point_center_2[1]-cube_length_2/2,point_center_2[1]+cube_length_2/2]),
            zaxis=dict(title='Z Axis', range=[point_center_2[2]-cube_length_2/2,point_center_2[2]+cube_length_2/2]),
            aspectmode='cube'
        )
        )

        # Show the figure
        fig.show()

    def show_patch_feature(self,frame1,frame2,which_obj,show_background=True):
        def extract_patch_feature(frame,which_obj):     # which_obj = positive or passive
            rgb,depth,mask,intrinsics = frame['rgb'],frame['depth'],frame[f'{which_obj}_mask'],frame['intrinsics']
            cropped_rgb,cropped_mask,bbox_square = self.prepare_image(rgb=rgb,mask=mask)    # expanded

            
            with torch.no_grad():
                dino_feature_dinowise = self.dino.forward_features(cropped_rgb.unsqueeze(0))['x_norm_patchtokens']   # 1, (448/14)**2(num of tokens,1024), c(1536)
                dino_feature_dinowise = dino_feature_dinowise.permute(0,2,1).reshape(1,-1,self.dino_size//14,self.dino_size//14)    # 1,c,32,32 ， bbox,patch-level

            return F.interpolate(dino_feature_dinowise,(448,448),mode='nearest')[0],F.interpolate(cropped_mask.unsqueeze(0).unsqueeze(0),(448,448),mode='nearest')[0][0]
        if show_background:
            feature1,mask1 = extract_patch_feature(frame1,which_obj)  # c,448,448
            feature2,mask2 = extract_patch_feature(frame2,which_obj)  # c,448,448
            c,h,w = feature1.shape
            feature1 = feature1.reshape(c,h*w).transpose(1,0)     # c,448,448 -> h*w,c
            feature2 = feature2.reshape(c,h*w).transpose(1,0) 
            feature = torch.cat([feature1,feature2],dim=0).cpu().numpy()  # 2*h*w,c

            pca = PCA(n_components=3)
            pca.fit(feature)
            pca_feature = pca.transform(feature)
            color = minmax_scale(pca_feature)
            color1,color2 = color[:h*w].reshape((h,w,3)), color[h*w:].reshape((h,w,3))

            # 创建一个新的画布，设置 1 行 2 列的子图
            plt.figure(figsize=(18, 6))

            # 第一个子图
            plt.subplot(1, 3, 1)
            plt.axis('off')  # 关闭坐标轴
            plt.imshow(color1)  # 显示第一张图片

            # 第二个子图
            plt.subplot(1, 3, 2)
            plt.axis('off')  # 关闭坐标轴
            plt.imshow(color2)  # 显示第二张图片
        else:
            feature1,mask1 = extract_patch_feature(frame1,which_obj)    # c,448,448; 448,448
            feature2,mask2 = extract_patch_feature(frame2,which_obj)
            c,h,w = feature1.shape
            feature1_masked = feature1[:,mask1] # c,n1
            feature2_masked = feature2[:,mask2] # c,n2
            
            feature = torch.cat([feature1_masked,feature2_masked],dim=1).cpu().numpy().transpose()  # c,n1+n2 -> n1+n2,c
            
            pca = PCA(n_components=3)
            pca.fit(feature)
            pca_feature = pca.transform(feature)
            color = minmax_scale(pca_feature)


            # 将 PCA 结果重新映射回 mask 的空间
            color1 = np.zeros((h, w, 3))
            color2 = np.zeros((h, w, 3))
            
            mask1,mask2 = mask1.cpu().numpy().astype(bool),mask2.cpu().numpy().astype(bool)

            # 仅填充 mask 区域的颜色
            color1[mask1] = color[:mask1.sum(), :]
            color2[mask2] = color[mask1.sum():, :]

            # 创建一个新的画布，设置 1 行 2 列的子图
            plt.figure(figsize=(18, 6))

            # 第一个子图
            plt.subplot(1, 3, 1)
            plt.axis('off')  # 关闭坐标轴
            plt.imshow(color1)  # 显示第一张图片

            # 第二个子图
            plt.subplot(1, 3, 2)
            plt.axis('off')  # 关闭坐标轴
            plt.imshow(color2)  # 显示第二张图片