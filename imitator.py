import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import json
import numpy as np
import torch
import open3d as o3d
from dataset import D415Data
from match import Matcher
from point_tracker import PointTracker
from utils import transform_pointcloud

class Imitator:
    def __init__(self):
        self.dataloader = D415Data(estimate_depth=False)
        self.matcher = Matcher()
        self.tracker = PointTracker('cotracker',self.matcher.feature_extracter)

    def imitate_visualize(self,seq_source,frame_target):
        
        # step0 计算不同视角初始时passive和positive的相对变换
        # match_positive, init_positive_camsource2camtarget,source_positive_keypoint,target_positive_keypoint,init_rigid_transform_positive_camsource2camtarget = self.matcher.match_weighted_affine(seq_source_positive[0], frame_target_positive)
        # match_passive, init_passive_camsource2camtarget, source_passive_keypoint, target_passive_keypoint,init_rigid_transform_passive_camsource2camtarget = self.matcher.match_weighted_affine(seq_source_passive[0], frame_target_passive)
        match_positive, init_rigid_transform_positive_camsource2camtarget,source_positive_keypoint,target_positive_keypoint,_ = self.matcher.match_vfc_image(seq_source[0], frame_target,'positive')
        match_passive, init_rigid_transform_passive_camsource2camtarget, source_passive_keypoint,target_passive_keypoint,_ = self.matcher.match_vfc_image(seq_source[0], frame_target,'passive')


        # step1 确定source中的两个坐标系原点,再计算出target中坐标系原点
        # 暂时直接以点云中心为原点
        # init_positive2cam_source, init_passive2cam_source,  = self.get_init_obj2cam_each_frame([seq_source_positive[0],seq_source_passive[0]])
        # 现在是手动指定原点/关键点
        init_positive2cam_source = np.eye(4)
        init_positive2cam_source[:3,3] = source_positive_keypoint
        init_passive2cam_source = np.eye(4)
        init_passive2cam_source[:3,3] = source_passive_keypoint

        init_positive2cam_target = init_rigid_transform_positive_camsource2camtarget@init_positive2cam_source
        init_passive2cam_target = init_rigid_transform_passive_camsource2camtarget@init_passive2cam_source

        # step2 计算source sequence中,两个物体的obj2cam_list,可接着计算出positive2passive
        positive2cam_source_list = self.tracker.track_pose(seq_source,init_positive2cam_source,which_obj='positive',grid_size=160,savename='source_positive')
        passive2cam_source_list = self.tracker.track_pose(seq_source,init_passive2cam_source,which_obj='passive',grid_size=160,savename='source_passive')
        positive2passive_source_list = [np.linalg.inv(passive2cam_source_list[i])@positive2cam_source_list[i] for i in range(len(positive2cam_source_list))]


        # 计算target canonical下的positive与passive point,然后用计算出的source sequence中两个物体的obj2cam_list去对齐,即可展示轨迹复现的效果
        # positive_target_point_camspace, passive_target_point_camspace = frame_target_positive['point_with_rgb'][:,:3], frame_target_passive['point_with_rgb'][:,:3]
        # positive_target_point_canonicalspace = transform_pointcloud(positive_target_point_camspace,np.linalg.inv(init_positive2cam_target))
        # passive_target_point_canonicalspace = transform_pointcloud(passive_target_point_camspace,np.linalg.inv(init_passive2cam_target))
        # final_positive_target_point_camsourcespace = [transform_pointcloud(positive_target_point_canonicalspace,positive2cam_source) for positive2cam_source in positive2cam_source_list]
        # final_passive_target_point_camsourcespace = [transform_pointcloud(passive_target_point_canonicalspace,passive2cam_source) for passive2cam_source in passive2cam_source_list]

        # return final_positive_target_point_camsourcespace,final_passive_target_point_camsourcespace

        positive_camtarget2camsource_list = [positive2cam_source@np.linalg.inv(init_positive2cam_target) for positive2cam_source in positive2cam_source_list]
        passive_camtarget2camsource_list = [passive2cam_source@np.linalg.inv(init_passive2cam_target) for passive2cam_source in passive2cam_source_list]
        return positive_camtarget2camsource_list,passive_camtarget2camsource_list
    


    def get_init_obj2cam_each_frame(self,frame_list):
        init_obj2cam_list = []
        for frame in frame_list:
            init_obj2cam = np.eye(4)
            point,_ = self.matcher.feature_extracter.extract(frame)
            init_obj2cam[:3,3] = point.mean(dim=0).cpu().numpy()
            init_obj2cam_list.append(init_obj2cam)
        return init_obj2cam_list