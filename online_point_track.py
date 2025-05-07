import torch
import numpy as np
from utils import image_coords_to_camera_space
import open3d as o3d
import cv2
from cotracker.utils.visualizer import Visualizer
from co_tracker.cotracker.predictor import CoTrackerOnlinePredictor

class OnlineCoTracker:
    def __init__(self,obj2caminit):
        self.model = CoTrackerOnlinePredictor(checkpoint='/home/yan20/tianshuwu/coi/co_tracker/checkpoints/scaled_online.pth')
        self.model = self.model.to('cuda')
        self.window_frames = []
        self.is_first_step = True
        self.i = 0
        self.init_frame = None
        
        # self.point_last = None
        # self.visibility_last = None
        # self.obj2camlast = None
        # self.keypoint_last = None
        
        self.obj2caminit = obj2caminit
        
        self.current_pose = obj2caminit.copy()
        self.draw_pose_list = []
        self.draw_image_list = []
        # self.vis= Visualizer(save_dir="./online_tmp_track", pad_value=100, linewidth=0.6)
        
    def track_pose(self,frame):
        
        if self.i == 0:
            self.init_frame = frame
        def _process_step(window_frames, is_first_step, grid_size, grid_query_frame,mask=None):
            video_chunk = (
                torch.tensor(
                    np.stack(window_frames[-self.model.step * 2 :]), device='cuda'
                )
                .float()
                .permute(0, 3, 1, 2)[None]
            )  # (1, T, 3, H, W)
            return self.model(
                video_chunk,
                is_first_step=is_first_step,
                grid_size=grid_size,
                grid_query_frame=grid_query_frame,
                segm_mask=mask
            )
        
        if self.i % self.model.step == 0 and self.i != 0:
            pred_tracks, pred_visibility = _process_step(
                self.window_frames,
                self.is_first_step,
                grid_size=512,
                grid_query_frame=0,
                mask=frame['positive_mask'].to(torch.uint8).unsqueeze(0).unsqueeze(0)
            )
            
            if self.is_first_step:
                self.is_first_step = False
            else:
                pred_tracks, pred_visibility = pred_tracks.cpu().numpy(), pred_visibility.cpu().numpy()
                track_2d, visibility = pred_tracks[0,-1], pred_visibility[0,-1]  # n,2  n
                track_2d_visiable = track_2d[visibility]   # n',2(x,y) 
                track_3d = image_coords_to_camera_space(frame['depth'].cpu().numpy(), track_2d_visiable[:,[1,0]].astype(np.int32), frame['intrinsics'])  # n',3
                point_all = np.zeros((visibility.shape[0],3))   # n,3
                point_all[visibility] = track_3d
                
                track_2d_init,visibility_init = pred_tracks[0,0], pred_visibility[0,0]
                track_2d_init_visable = track_2d_init[visibility_init]
                track_3d_init = image_coords_to_camera_space(self.init_frame['depth'].cpu().numpy(), track_2d_init_visable[:,[1,0]].astype(np.int32), self.init_frame['intrinsics'])
                point_all_init = np.zeros((visibility_init.shape[0],3))
                point_all_init[visibility_init] = track_3d_init
                
                point_last = track_3d_init
                visibility_last = visibility_init
                obj2camlast = self.obj2caminit
                

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
                pose_obj2camcurr =  pose_camlast2camcurr@obj2camlast
                
                self.current_pose = pose_obj2camcurr
                
                self.draw_pose_list.append(pose_obj2camcurr)
                self.draw_image_list.append(frame['rgb'].cpu().numpy())
                print('update current tracking pose!!!!!!!!!!!!!!!!!!!!')
                
        self.window_frames.append(frame['rgb'].cpu().numpy())
        self.i = self.i + 1


    def draw_pose_tracked(self):

        video_np = []
        for i,image in enumerate(self.draw_image_list):
            obj2cam = self.draw_pose_list[i]
            rvec, _ = cv2.Rodrigues(obj2cam[:3,:3])

            video_np.append(cv2.drawFrameAxes(
                image,
                np.array([[self.init_frame['intrinsics']['fx'],0,self.init_frame['intrinsics']['cx']],
                                    [0,self.init_frame['intrinsics']['fy'],self.init_frame['intrinsics']['cy']],
                                    [0,0,1]]),
                np.array([0.,0.,0.,0.,0.]),
                rvec, obj2cam[:3,3], 0.3
            ))
            
        for drawed_image in video_np:
            cv2.imshow("drawed_image",drawed_image)
            cv2.waitKey(1000)
        
        
        