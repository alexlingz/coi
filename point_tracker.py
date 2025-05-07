import torch
from co_tracker.cotracker.predictor import CoTrackerPredictor
import imageio.v3 as iio
from cotracker.utils.visualizer import Visualizer
from utils import image_coords_to_camera_space,depth_map_to_pointcloud_torch,transform_pointcloud,calculate_2d_projections
import open3d as o3d
import cv2
import os
import numpy as np
import imageio
# from SpaTracker.models.spatracker.predictor import SpaTrackerPredictor
from base64 import b64encode
from IPython.display import HTML
from umeyama_ransac import WeightedRansacAffine3D
from utils import rotation_to_quaternion_np, quaternion_apply,quaternion_to_matrix_np, smooth_trajectory_fitting_with_smoothness,quaternion_raw_multiply ,quaternion_invert
from torchmetrics.functional import pairwise_cosine_similarity
from umeyama_ransac import umeyama_alignment, evaluateModel
from tqdm import tqdm
def show_video(video_path):
    video_file = open(video_path, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video width="640" height="360" autoplay loop controls><source src="{video_url}"></video>""")

class PointTracker:
    def __init__(self,tracker_model='cotracker',feature_extracter=None,save_dir=None):
        self.feature_extracter = feature_extracter
        self.tracker_model = tracker_model
        if tracker_model == 'cotracker':
            self.cotracker = CoTrackerPredictor(checkpoint=os.path.join('/home/yan20/tianshuwu/coi/co_tracker/checkpoints/scaled_offline.pth')).to('cuda')
            self.cotracker.eval()
        elif tracker_model == 'spatracker':
            # seq_length [8, 12, 16] choose one you want
            self.S_lenth = 12
            self.spatracker = SpaTrackerPredictor(checkpoint=os.path.join('./SpaTracker/checkpoints/spaT_final.pth'),interp_shape = (384, 512),seq_length = self.S_lenth).to('cuda')
            self.spatracker.eval()
        else:
            print('no correct tracker!!!')
        self.vis= Visualizer(save_dir=save_dir, pad_value=100, linewidth=0.6)
        
        self.save_dir = save_dir

    def track_pose(self,frames,obj2caminit,which_obj,grid_size=640,savename='video',rigid_point=None,articulate=False):
        torch.cuda.empty_cache()
        mask_area = frames[0][f'{which_obj}_mask'].sum()
        grid_size = int(torch.sqrt((400)*frames[0]['rgb'].shape[0]*frames[0]['rgb'].shape[1]/(mask_area)).int().item())
        
        video = torch.stack([frame['rgb'] for frame in frames]).unsqueeze(0).to(torch.float32).permute(0,1,4,2,3)
        first_mask = frames[0][f'{which_obj}_mask'].to(torch.uint8).unsqueeze(0).unsqueeze(0)
        intrinsics = np.array([[frames[0]['intrinsics']['fx'],0,frames[0]['intrinsics']['cx']],
                                    [0,frames[0]['intrinsics']['fy'],frames[0]['intrinsics']['cy']],
                                    [0,0,1]])
        # video: B T C H W, torch.float32, 0~255
        # first_mask: B,1,h,w,  0/1, torch.uint8

        if self.tracker_model == 'cotracker':
            pred_tracks, pred_visibility = self.cotracker(video, grid_size=grid_size, segm_mask=first_mask,backward_tracking=True)   # B T N 2,  B T N
        elif self.tracker_model == 'spatracker':
            MonoDEst_M = None   # TODO 无深度时的pipeline尚未考虑
            depths = torch.stack([frame['depth'] for frame in frames])[:,None]
            pred_tracks, pred_visibility, T_Firsts = self.spatracker(video, video_depth=depths, grid_size=grid_size,segm_mask=first_mask,depth_predictor=MonoDEst_M, grid_query_frame=0, wind_length=self.S_lenth)    # B T N 3,  B T N,  B N

            pred_tracks = pred_tracks[:,:,int(grid_size**2):,:2]
            pred_visibility = pred_visibility[:,:,int(grid_size**2):]
            # pred_tracks = pred_tracks[0].reshape(-1,3).permute(1,0)    # b,t,n,3 -> t*n,3 -> 3,t*n
            # pred_tracks = torch.tensor(calculate_2d_projections(pred_tracks.cpu().numpy(),intrinsics).reshape(len(frames),-1,2)[None],device='cuda')  # t*n,2 -> t,n,2
        
        self.vis.visualize(torch.tensor(video,device='cuda'),torch.tensor(pred_tracks,device='cuda'),torch.tensor(pred_visibility,device='cuda'),filename='tmp_track')

        # 计算与上一帧的差值，累积误差勉勉强强；
        # 尝试计算与第一帧的差值，误差更大
        point_last = None
        visibility_last = None
        obj2camlast = None

        pose_estimater = WeightedRansacAffine3D(with_scale='transform')
        init_keypoint = obj2caminit[:3,3]


        pose_obj2cam_list = []
        pred_tracks, pred_visibility = pred_tracks.cpu().numpy(), pred_visibility.cpu().numpy()
        
        if rigid_point is None:
            inlier, rigid_point = self.find_rigid_area(frames,init_keypoint,pred_tracks[0], pred_visibility[0],views=1)
            if not articulate:
                inlier = np.ones(pred_tracks[0][0].shape[0]).astype(bool)
                rigid_point = image_coords_to_camera_space(frames[0]['depth'].cpu().numpy(),np.round(pred_tracks[0,0][:,[1,0]]).astype(np.int32),frames[0]['intrinsics'])
            # 权重不work，想想也是，连阈值都不是很能分的清楚，基于权重的模糊配准怎么可能更work？
            # 并且不用open3d的库效果也会更差
            # rigid_weight = self.compute_rigid_weight(frames,init_keypoint,pred_tracks[0], pred_visibility[0])

        else:
            track_points0 = np.array(image_coords_to_camera_space(frames[0]['depth'].cpu().numpy(),np.round(pred_tracks[0][0][:,[1,0]]).astype(np.int32),frames[0]['intrinsics']))
            inlier = np.linalg.norm(track_points0[:,None] - rigid_point[None], axis=-1).min(axis=1)<0.02  # n
            pass

        pred_tracks, pred_visibility = pred_tracks[:,:,inlier], pred_visibility[:,:,inlier]
        
        # if compute_last_only:
        #     pred_tracks = pred_tracks[:,[0,-1]]
        #     pred_visibility = pred_visibility[:   ,[0,-1]]
        for i in range(pred_tracks.shape[1]):
            track_2d, visibility = pred_tracks[0,i], pred_visibility[0,i]  # n,2  n
            track_2d_visiable = track_2d[visibility]   # n',2(x,y) 
            track_3d = image_coords_to_camera_space(frames[i]['depth'].cpu().numpy(), np.round(track_2d_visiable[:,[1,0]]).astype(np.int32), frames[0]['intrinsics'])  # n',3
            point_all = np.zeros((visibility.shape[0],3))   # n,3
            point_all[visibility] = track_3d

            if point_last is None:
                point_last = point_all
                visibility_last = visibility
                obj2camlast = obj2caminit
                pose_obj2cam_list.append(obj2camlast)

                # point_distance_to_keypoint = np.linalg.norm(init_keypoint - point_all,axis=1)  # n

                # keypoint_last = point_all[visibility][np.argmin(point_distance_to_keypoint[visibility])]
                

            else:
                correspondence = [[idx, idx] for idx in range(visibility.shape[0]) if visibility_last[idx] and visibility[idx]] # n,2
                correspondence = o3d.utility.Vector2iVector(correspondence)

                pcd_last = o3d.geometry.PointCloud()
                pcd_last.points = o3d.utility.Vector3dVector(point_last)

                pcd_current = o3d.geometry.PointCloud()
                pcd_current.points = o3d.utility.Vector3dVector(point_all)

                # # 全局写法
                result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                    pcd_last,
                    pcd_current,
                    correspondence,
                    0.01,
                    checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.01)],
                )
                pose_camlast2camcurr = result.transformation

                # weight 算法
                # _, transform,_ = pose_estimater.getRANSACInliers(point_last[visibility],point_all[visibility],rigid_weight[visibility])
                # pose_camlast2camcurr = transform


                pose_obj2camcurr =  pose_camlast2camcurr@obj2camlast
                pose_obj2cam_list.append(pose_obj2camcurr)

                # 注释掉，算的就一直是相对第一帧的pose
                # point_last = point_all
                # visibility_last = visibility
                # obj2camlast = pose_obj2camcurr
                # keypoint_last = keypoint_now



        video_np = self.draw_pose_tracked(video,pose_obj2cam_list,intrinsics)
        self.vis.visualize(torch.tensor(video_np,device='cuda'),torch.tensor(pred_tracks,device='cuda'),torch.tensor(pred_visibility,device='cuda'),filename=savename)

        # pose_obj2cam_list = smooth_trajectory_fitting_with_smoothness(pose_obj2cam_list,len(pose_obj2cam_list),t_smooth_factor=0.01,r_smooth_factor=0.1)
        pose_obj2cam_list = self.optimize_trajectory(frames,pose_obj2cam_list,which_obj,pred_tracks[0], pred_visibility[0],rigid_point)
        video_np = self.draw_pose_tracked(video,pose_obj2cam_list,intrinsics)
        self.vis.visualize(torch.tensor(video_np,device='cuda'),torch.tensor(pred_tracks,device='cuda'),torch.tensor(pred_visibility,device='cuda'),filename=f'optimized_{savename}')


        return pose_obj2cam_list,rigid_point
    
    def optimize_trajectory(self,frames,pose_obj2cam_list,which_obj,pred_tracks, pred_visibility,rigid_point):
        rigid_point = torch.tensor(rigid_point,device='cuda',dtype=torch.float32)
        # 没有cad模型，我们预测的只是相对第一帧的pose，所以第一帧的pose应该固定
        # pred_tracks, pred_visibility： t,n,2; t,n
        
        

        def compute_traj_smoothness_loss(tq_obj2cam, position_lambda, rotation_lambda):
            # cami2cam0： T,7 xyzwxyz
            T = len(tq_obj2cam)
            loss = torch.zeros((T-1), device='cuda', dtype=torch.float32)

            position_diff = (tq_obj2cam[1:,:3] - tq_obj2cam[:-1,:3])**2

            quat_diff = quaternion_raw_multiply(quaternion_invert(tq_obj2cam[1:,3:]),tq_obj2cam[:-1,3:])
            rotation_diff = (2*torch.acos(torch.clamp(quat_diff[:,0],min=-1+1e-6,max=1-1e-6)))**2

            loss = position_lambda * (torch.norm(position_diff,dim=-1)**3) + rotation_lambda * (torch.norm(rotation_diff)**3)

            return loss.mean()
        
        def compute_feature_loss(tq_obj2cam,point,feature,rigid_point):
            # 计算相邻帧之间，因为如果帧数相差较大的话视角可能不一致，会出问题
            # qt_cam02cami_list:T,7， 其中第一个元素是常量0,0,0,1,0,0,0
            # point: T,n(数量不一定一致),3
            # feature: T,n,c    
            # rigid_point: m,3,camspace下，第0帧
            rigid_point_objspace = quaternion_apply(quaternion_invert(tq_obj2cam[0][3:]),rigid_point) - quaternion_apply(quaternion_invert(tq_obj2cam[0][3:]),tq_obj2cam[0:1,:3])
            new_point = []
            new_feature = []
            for i in range(len(point)):
                point_objspace = quaternion_apply(quaternion_invert(tq_obj2cam[i][3:]),point[i]) - quaternion_apply(quaternion_invert(tq_obj2cam[i][3:]),tq_obj2cam[i:i+1,:3])  # ni,3
                distance_mask = torch.norm(point_objspace[:,None]-rigid_point_objspace[None],dim=-1).min(dim=-1)[0]<0.01  # n,m -> n
                new_point.append(point[i][distance_mask])
                new_feature.append(feature[i][distance_mask])

            T = len(tq_obj2cam)
            loss = torch.zeros(T-1,dtype=torch.float32,device='cuda')
            for i in range(T-1):
                # 投影到objspace下计算
                pointi_objspace = quaternion_apply(quaternion_invert(tq_obj2cam[i][3:]),new_point[i]) - quaternion_apply(quaternion_invert(tq_obj2cam[i][3:]),tq_obj2cam[i:i+1,:3])  # ni,3
                pointi1_objspace = quaternion_apply(quaternion_invert(tq_obj2cam[i+1][3:]),new_point[i+1]) - quaternion_apply(quaternion_invert(tq_obj2cam[i+1][3:]),tq_obj2cam[i+1:i+2,:3])  # ni1,3
                
                similarity_matrix = pairwise_cosine_similarity(new_feature[i],new_feature[i+1])# ni,ni1
                similarity_matrix = ((similarity_matrix+1)/2)**6
                
                distance_matrix = torch.norm(pointi_objspace.unsqueeze(1) - pointi1_objspace.unsqueeze(0), dim=-1)    # ni,ni1
                dis_weight = (1/(distance_matrix+1e-3) )**3     # ni,ni1
                similarity_matrix = dis_weight*similarity_matrix
                
                loss[i] += -similarity_matrix.mean()
            
            return loss.mean()
                
        def compute_tracking_loss(tq_obj2cam,track_points, pred_visibility):
            # qt_cami2cam0:T,7， 其中第一个元素是常量0,0,0,1,0,0,0
            # track_points      T,n,3
            # pred_visibility   T,n
            T = len(tq_obj2cam)
            loss = torch.zeros(T-1,dtype=torch.float32,device='cuda')
            track_points_objspace_t0 = quaternion_apply(quaternion_invert(tq_obj2cam[0][3:]),track_points[0]) - quaternion_apply(quaternion_invert(tq_obj2cam[0][3:]),tq_obj2cam[0:1,:3])
            for i in range(1,T):
                vis_mask = (pred_visibility[0]*pred_visibility[i]).bool()
                dis = torch.norm(track_points_objspace_t0[vis_mask] -(quaternion_apply(quaternion_invert(tq_obj2cam[i][3:]),track_points[i]) - quaternion_apply(quaternion_invert(tq_obj2cam[i][3:]),tq_obj2cam[i:i+1,:3]))[vis_mask] , dim=-1)   # n'
                loss[i-1] += dis.mean()
                
            return loss.mean()
        
        def compute_quat_loss(qt_cami2cam0):
            T = len(qt_cami2cam0)
            loss = (1 - torch.norm(qt_cami2cam0[1:,3:],dim=-1))**2

            return loss.mean()
        
        points_list,feature_list = [],[]
        for i in tqdm(range(len(frames))):
            point,feature,_ = self.feature_extracter.extract(frames[i],which_obj,sample_point_size=512)
            points_list.append(point)
            feature_list.append(feature)
            
        # track_points = [image_coords_to_camera_space(frames[i]['depth'].cpu().numpy(),
        #                                              np.concatenate([ np.clip(np.round(pred_tracks[i][:,1]),0,frames[i]['depth'].shape[0] )[:,None],np.clip(np.round(pred_tracks[i][:,0]),0,frames[i]['depth'].shape[1])[:,None]],axis=-1).astype(np.int32),
        #                                              frames[i]['intrinsics']) for i in range(len(frames))]
        # track_points, pred_visibility = torch.tensor(track_points,device='cuda',dtype=torch.float32),torch.tensor(pred_visibility,device='cuda',dtype=torch.float32)
    
        
        tq_obj2cam_list = []
        for i in range(len(pose_obj2cam_list)):
            tq = np.zeros(7)
            tq[:3] = pose_obj2cam_list[i][:3,3]
            tq[3:] = rotation_to_quaternion_np(pose_obj2cam_list[i][:3,:3])
            if i == 0:
                tq_obj2cam_list.append(torch.tensor(tq,device='cuda',requires_grad=False,dtype=torch.float32))
            else:
                tq_obj2cam_list.append(torch.tensor(tq,device='cuda',requires_grad=True,dtype=torch.float32))

        optimizer = torch.optim.Adam(tq_obj2cam_list[1:],lr=2e-3)
        
        iteration = 0
        a = 10
        b = 1e-4
        c = 1e-1
        d = 1e6
        while iteration<100:
        
            tq_obj2cam = torch.stack(tq_obj2cam_list,dim=0) # T,7
            
            
            iteration = iteration+1
            optimizer.zero_grad()
            
            traj_smoothness_loss = compute_traj_smoothness_loss(tq_obj2cam,1.0,1.0)
            feature_loss = compute_feature_loss(tq_obj2cam,points_list,feature_list,rigid_point)
            # tracking_loss = compute_tracking_loss(tq_obj2cam,track_points,pred_visibility)
            quat_loss = compute_quat_loss(tq_obj2cam)
            

                
            loss = a*traj_smoothness_loss + b*feature_loss + d*quat_loss
            loss.backward()
            optimizer.step()
            
            print('iter:',iteration,'traj_smoothness_loss:',a*traj_smoothness_loss.item(),'feature_loss:',b*feature_loss.item(),'quat_loss:',d*quat_loss.item())

        optimized_o2c_list = []
        for i in range(len(tq_obj2cam_list)):
            optimized_o2c = np.eye(4)
            optimized_o2c[:3,:3] = quaternion_to_matrix_np(tq_obj2cam_list[i][3:].detach().cpu().numpy())
            optimized_o2c[:3,3] = tq_obj2cam_list[i][:3].detach().cpu().numpy()
            optimized_o2c_list.append(optimized_o2c)

        return optimized_o2c_list

    def track_multiview_pose(self,record_frames,obj2caminit1,which_obj,grid_size=640,savename='video',rigid_point=None,articulate=False):
        torch.cuda.empty_cache()
        mask_area = record_frames[0]['cam1'][f'{which_obj}_mask'].sum()
        grid_size = int(torch.sqrt((400)*record_frames[0]['cam1']['rgb'].shape[0]*record_frames[0]['cam1']['rgb'].shape[1]/mask_area).int().item())

        multiview_frames = {}
        for camname in record_frames[0].keys():
            multiview_frames[camname] = [record_frames[i][camname] for i in range(len(record_frames))]  
        
        pred_tracks_list = []
        pred_visibility_list = []
        
        for camname in multiview_frames.keys():
            frames = multiview_frames[camname]
            
            video = torch.stack([frame['rgb'] for frame in frames]).unsqueeze(0).to(torch.float32).permute(0,1,4,2,3)
            first_mask = frames[0][f'{which_obj}_mask'].to(torch.uint8).unsqueeze(0).unsqueeze(0)
            intrinsics = np.array([[frames[0]['intrinsics']['fx'],0,frames[0]['intrinsics']['cx']],
                                        [0,frames[0]['intrinsics']['fy'],frames[0]['intrinsics']['cy']],
                                        [0,0,1]])
            # video: B T C H W, torch.float32, 0~255
            # first_mask: B,1,h,w,  0/1, torch.uint8

            if self.tracker_model == 'cotracker':
                pred_tracks, pred_visibility = self.cotracker(video, grid_size=grid_size, segm_mask=first_mask,backward_tracking=True)   # B T N 2,  B T N
            elif self.tracker_model == 'spatracker':
                MonoDEst_M = None   # TODO 无深度时的pipeline尚未考虑
                depths = torch.stack([frame['depth'] for frame in frames])[:,None]
                pred_tracks, pred_visibility, T_Firsts = self.spatracker(video, video_depth=depths, grid_size=grid_size,segm_mask=first_mask,depth_predictor=MonoDEst_M, grid_query_frame=0, wind_length=self.S_lenth)    # B T N 3,  B T N,  B N

                pred_tracks = pred_tracks[:,:,int(grid_size**2):,:2]
                pred_visibility = pred_visibility[:,:,int(grid_size**2):]
                # pred_tracks = pred_tracks[0].reshape(-1,3).permute(1,0)    # b,t,n,3 -> t*n,3 -> 3,t*n
                # pred_tracks = torch.tensor(calculate_2d_projections(pred_tracks.cpu().numpy(),intrinsics).reshape(len(frames),-1,2)[None],device='cuda')  # t*n,2 -> t,n,2
                
            pred_tracks_list.append(pred_tracks)            # nc,B,T,N,2
            pred_visibility_list.append(pred_visibility)    # nc,B,T,N
            
        pred_tracks, pred_visibility = pred_tracks.cpu().numpy(), pred_visibility.cpu().numpy()

        if rigid_point is None:
            inlier, rigid_point = self.find_rigid_area(frames,init_keypoint,pred_tracks[0], pred_visibility[0],views=2)
            if not articulate:
                inlier = np.ones(pred_tracks[0][0].shape[0]).astype(bool)
                rigid_point = image_coords_to_camera_space(frames[0]['depth'].cpu().numpy(),np.round(pred_tracks[0,0][:,[1,0]]).astype(np.int32),frames[0]['intrinsics'])
            # 权重不work，想想也是，连阈值都不是很能分的清楚，基于权重的模糊配准怎么可能更work？
            # 并且不用open3d的库效果也会更差
            # rigid_weight = self.compute_rigid_weight(frames,init_keypoint,pred_tracks[0], pred_visibility[0])
        else:
            track_points0 = np.array(image_coords_to_camera_space(frames[0]['depth'].cpu().numpy(),np.round(pred_tracks[0][0][:,[1,0]]).astype(np.int32),frames[0]['intrinsics']))
            inlier = np.linalg.norm(track_points0[:,None] - rigid_point[None], axis=-1).min(axis=1)<0.02  # n
            pass

        pred_tracks, pred_visibility = pred_tracks[:,:,inlier], pred_visibility[:,:,inlier]


        # 计算与上一帧的差值，累积误差勉勉强强；
        # 尝试计算与第一帧的差值，误差更大
        point_last = None
        visibility_last = None
        obj2camlast = None

        obj2caminit = obj2caminit1
        init_keypoint = obj2caminit[:3,3]
        

        pose_obj2cam_list = []
        # pred_tracks, pred_visibility = pred_tracks.cpu().numpy(), pred_visibility.cpu().numpy()

        for i in range(pred_tracks.shape[1]):
            
            point_all_list = []
            visibility_list = []
                
            for j,camname in enumerate(multiview_frames.keys()):
                frames = multiview_frames[camname]
                pred_tracks = pred_tracks_list[j].cpu().numpy()
                pred_visibility = pred_visibility_list[j].cpu().numpy()
            
                track_2d, visibility = pred_tracks[0,i], pred_visibility[0,i]  # n,2  n
                track_2d_visiable = track_2d[visibility]   # n',2(x,y) 
                track_3d = image_coords_to_camera_space(frames[i]['depth'].cpu().numpy(), np.round(track_2d_visiable[:,[1,0]]).astype(np.int32), frames[0]['intrinsics'])  # n',3
                
                # 转换到第一个相机的坐标系下
                track_3d = transform_pointcloud(track_3d,np.linalg.inv(multiview_frames['cam1'][0]['cam2base'])@multiview_frames[camname][0]['cam2base'])
                
                point_all = np.zeros((visibility.shape[0],3))   # n,3
                point_all[visibility] = track_3d
                
                point_all_list.append(point_all)
                visibility_list.append(visibility)
                
            point_all = np.concatenate(point_all_list,axis=0)
            visibility = np.concatenate(visibility_list,axis=0)

            if point_last is None:
                point_last = point_all
                visibility_last = visibility
                obj2camlast = obj2caminit
                pose_obj2cam_list.append(obj2camlast)

                point_distance_to_keypoint = np.linalg.norm(init_keypoint - point_all,axis=1)  # n

                keypoint_last = point_all[visibility][np.argmin(point_distance_to_keypoint[visibility])]
                

            else:
                correspondence = [[idx, idx] for idx in range(visibility.shape[0]) if visibility_last[idx] and visibility[idx]] # n,2
                correspondence = o3d.utility.Vector2iVector(correspondence)

                pcd_last = o3d.geometry.PointCloud()
                pcd_last.points = o3d.utility.Vector3dVector(point_last)

                pcd_current = o3d.geometry.PointCloud()
                pcd_current.points = o3d.utility.Vector3dVector(point_all)

                # 全局写法
                result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                    pcd_last,
                    pcd_current,
                    correspondence,
                    0.01,
                    checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.01)],
                )
                pose_camlast2camcurr = result.transformation

                # keypoint_now = point_all[visibility][np.argmin(point_distance_to_keypoint[visibility])]

                # weight 算法
                # inliers,affine_transform,_ = pose_estimater.estimate(point_last,keypoint_last,point_all,keypoint_now)
                # pose_camlast2camcurr = affine_transform


                pose_obj2camcurr =  pose_camlast2camcurr@obj2camlast

                pose_obj2cam_list.append(pose_obj2camcurr)

                # 注释掉，算的就一直是相对第一帧的pose
                # point_last = point_all
                # visibility_last = visibility
                # obj2camlast = pose_obj2camcurr
                # keypoint_last = keypoint_now


        video = torch.stack([frame['cam1']['rgb'] for frame in record_frames]).unsqueeze(0).to(torch.float32).permute(0,1,4,2,3)

        video_np = self.draw_pose_tracked(video,pose_obj2cam_list,intrinsics)
        # self.vis.visualize(torch.tensor(video_np,device='cuda'),torch.tensor(pred_tracks,device='cuda'),torch.tensor(pred_visibility,device='cuda'),filename=savename)


        # pose_obj2cam_list = self.optimize_trajectory(record_frames,pose_obj2cam_list,which_obj,pred_tracks[0], pred_visibility[0],rigid_point)
        video_np = self.draw_pose_tracked(video,pose_obj2cam_list,intrinsics)
        self.vis.visualize(torch.tensor(video_np,device='cuda'),torch.tensor(pred_tracks,device='cuda'),torch.tensor(pred_visibility,device='cuda'),filename=f'optimized_{savename}')


        return pose_obj2cam_list,rigid_point

    def draw_pose_tracked(self,video,pose_obj2cam_list,intrinsics):

        video_np = []
        for i,image in enumerate(video[0]):
            obj2cam = pose_obj2cam_list[i]
            rvec, _ = cv2.Rodrigues(obj2cam[:3,:3])

            video_np.append(cv2.drawFrameAxes(
                image.permute(1,2,0).cpu().numpy(),
                intrinsics,
                np.array([0.,0.,0.,0.,0.]),
                rvec, obj2cam[:3,3], 0.3
            ))
            

        return np.stack(video_np)[None].transpose(0,1,4,2,3)


    def show_objspace(self,frames,pose_obj2cam_list):
        objspace_point_list = []
        for i in range(len(frames)):
            camspace_point = depth_map_to_pointcloud_torch(frames[i]['depth'],frames[i]['mask'],frames[i]['intrinsics']).cpu().numpy()
            objspace_point = transform_pointcloud(camspace_point,np.linalg.inv(pose_obj2cam_list[i]))
            objspace_point_list.append(objspace_point)


        vis = o3d.visualization.Visualizer()
        vis.create_window()
        geometry = o3d.geometry.PointCloud()

        # 合成视频
        image_list = []

        for i, points in enumerate(objspace_point_list):
            geometry.points = o3d.utility.Vector3dVector(points)
            vis.add_geometry(geometry)
            vis.poll_events()
            vis.update_renderer()

            # 保存当前帧为图像
            vis.capture_screen_image(f"tmp_image/frame_{i:03d}.png")
            vis.remove_geometry(geometry)
            image_list.append(cv2.imread(f"tmp_image/frame_{i:03d}.png"))
            os.remove(f"tmp_image/frame_{i:03d}.png")

        vis.destroy_window()

        height, width, _ = image_list[0].shape


        video_writer = imageio.get_writer("tmp_image/objspace_point.mp4", fps=16)

        for frame in image_list:
            resized_frame = cv2.resize(frame,(width//2,height//2))
            video_writer.append_data(resized_frame)
        video_writer.close()

    def find_rigid_area(self,frames,init_keypoint,pred_tracks, pred_visibility,views):
        track_points = np.array([image_coords_to_camera_space(frames[i]['depth'].cpu().numpy(),np.round(pred_tracks[i][:,[1,0]]).astype(np.int32),frames[i]['intrinsics']) for i in range(len(frames))])
        
        mean_neighbor_distance = np.sort(np.linalg.norm(track_points[0][:,None] - track_points[0][None],axis=-1),axis=1)[:,1:4].mean()
        
        T,n,_ = track_points.shape
        # t,n,3; t,n
        
        distance = np.linalg.norm(track_points[0] - init_keypoint, axis=-1)  # n
        neighbor_sort_id = np.argsort(distance)  # n
        
        inlier_vote = np.zeros(n)
        MaxIterations = 100
        
        compute_time = 0
        for t in range(1,T):
            source = track_points[0]  # n,3
            target = track_points[t]
            current_visibility = (pred_visibility[0]*pred_visibility[t])
            
            point_translation = np.linalg.norm(target[current_visibility] - source[current_visibility],axis=-1)  # n
            
            if (point_translation>mean_neighbor_distance*3).sum()<0.02*current_visibility.sum():
                continue
            else:
                compute_time = compute_time+1
            
            SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
            TargetHom = np.transpose(np.hstack([target, np.ones([target.shape[0], 1])]))
            
            

            for i in range(MaxIterations):
                RandIdx = np.random.choice(neighbor_sort_id[current_visibility[neighbor_sort_id]][:15], 6, replace=False)
                _, rotation, translation, SimTransform = umeyama_alignment(SourceHom[:3, RandIdx], TargetHom[:3, RandIdx],with_scale=False)
                
                Diff_s2t = TargetHom - np.matmul(SimTransform, SourceHom)
                ResidualVec_s2t = np.linalg.norm(Diff_s2t[:3, :], axis=0)
                InlierIdx = np.where(ResidualVec_s2t < mean_neighbor_distance*3)      # todo 这两个scale应该不一样

                inlier_vote[InlierIdx] = inlier_vote[InlierIdx] + 1
        
        inlier_vote = inlier_vote / (MaxIterations*compute_time)
        
        np.save(f'{self.save_dir}/all_tracking_point_for_computing_rigid_source.npy',source)
        np.save(f'{self.save_dir}/rigid_source_score.npy',inlier_vote)

        threshold_ratio = (views-0.3)/views

        threshold = np.sort(inlier_vote)[int(len(inlier_vote)*threshold_ratio)]
        inlier = inlier_vote > threshold

        return inlier, source[inlier]

    def compute_rigid_weight(self,frames,init_keypoint,pred_tracks, pred_visibility):
        track_points = np.array([image_coords_to_camera_space(frames[i]['depth'].cpu().numpy(),np.round(pred_tracks[i][:,[1,0]]).astype(np.int32),frames[i]['intrinsics']) for i in range(len(frames))])
        
        mean_neighbor_distance = np.sort(np.linalg.norm(track_points[0][:,None] - track_points[0][None],axis=-1),axis=1)[:,1:4].mean()
        
        T,n,_ = track_points.shape
        # t,n,3; t,n
        
        distance = np.linalg.norm(track_points[0] - init_keypoint, axis=-1)  # n
        neighbor_sort_id = np.argsort(distance)  # n
        
        rigid_weight = np.zeros(n)
        MaxIterations = 100
        
        compute_time = 0
        for t in range(1,T):
            source = track_points[0]  # n,3
            target = track_points[t]
            current_visibility = (pred_visibility[0]*pred_visibility[t])
            
            point_translation = np.linalg.norm(target[current_visibility] - source[current_visibility],axis=-1)  # n
            
            if (point_translation>mean_neighbor_distance*3).sum()<0.02*current_visibility.sum():
                continue
            else:
                compute_time = compute_time+1
            
            SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
            TargetHom = np.transpose(np.hstack([target, np.ones([target.shape[0], 1])]))
            
            for i in range(MaxIterations):
                RandIdx = np.random.choice(neighbor_sort_id[current_visibility[neighbor_sort_id]][:15], 6, replace=False)
                _, rotation, translation, SimTransform = umeyama_alignment(SourceHom[:3, RandIdx], TargetHom[:3, RandIdx],with_scale=False)
                
                Diff_s2t = TargetHom - np.matmul(SimTransform, SourceHom)
                ResidualVec_s2t = np.linalg.norm(Diff_s2t[:3, :], axis=0)   # n
                
                currnet_weight =  ( (-ResidualVec_s2t)-(-ResidualVec_s2t).min() ) / ( (-ResidualVec_s2t).max()-(-ResidualVec_s2t).min() )  

                rigid_weight = rigid_weight + currnet_weight
        
        rigid_weight = ( (rigid_weight - rigid_weight.min()) / (rigid_weight.max() - rigid_weight.min()) ) ** 6
        

        return rigid_weight