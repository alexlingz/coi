
from imitator import Imitator
import rospy
from sensor_msgs.msg import Image,CameraInfo
import numpy as np
from mask_predictor import MaskPredictor
import cv2
import torch
from scipy.ndimage import binary_erosion,distance_transform_edt
from dataset import D415Data
from match import Matcher
from point_tracker import PointTracker
from utils import transform_pointcloud, compute_rot_distance
from detect_contact_point import find_contact_point
from generate_demo_keypoint import generate_keypoint
from frankapy import FrankaArm, SensorDataMessageType
import time
from utils import quaternion_to_matrix_np,rotation_to_quaternion_np, calculate_2d_projections, smooth_trajectory_fitting_with_smoothness,select_keypoint,depth_map_to_pointcloud_torch
from autolab_core import RigidTransform
from grasping.api import run_grasping
from online_point_track import OnlineCoTracker
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from franka_interface_msgs.msg import SensorDataGroup
import os
import requests
import open3d as o3d
import pickle

# class Observer:
#     def __init__(self,dynamic_keypoint):
        
        
#         self.dynamic_keypoint = dynamic_keypoint
#         self.rgb = None # np, h,w,3
#         self.depth = None
#         self.intrinsic = None
#         self.cam2base = np.load('./c2b.npy')
        
#         self.rgb2 = None # np, h,w,3
#         self.depth2 = None
#         self.intrinsic2 = None
#         self.cam2base2 = np.load('./c2b2.npy')

#         self._rgb_sub = rospy.Subscriber('/camera/color/image_raw', Image, self._rgb_callback)
#         self._depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self._depth_callback)
#         self.camera_info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self._camera_info_callback)
        
#         self._rgb2_sub = rospy.Subscriber('/camera2/color/image_raw', Image, self._rgb2_callback)
#         self._depth2_sub = rospy.Subscriber("/camera2/aligned_depth_to_color/image_raw", Image, self._depth2_callback)
#         self.camera2_info_sub = rospy.Subscriber('/camera2/color/camera_info', CameraInfo, self._camera2_info_callback)

        
#         self.max_w=640


#         self.mask_predictor = MaskPredictor()
#         self.mask_predictor2 = MaskPredictor()
#         self.device='cuda'
#         time.sleep(1)
#         self.robot = FrankaArm()
        
#     def _rgb_callback(self, msg):
#         try:
#             # 从消息中提取图像数据
#             # 注意：msg.data 是图像数据的字节数组
#             img_data = np.frombuffer(msg.data, dtype=np.uint8)
            
#             # 获取图像的宽度、高度和通道数
#             height = msg.height
#             width = msg.width
#             channels = 3  # 假设是 RGB 图像，3 个通道
            
#             # 重塑数据为图像格式
#             self.rgb = img_data.reshape((height, width, channels))
#         except Exception as e:
#             rospy.logerr("Error parsing raw image data: %s", str(e))

#     def _depth_callback(self, msg):
#         try:
#             # 从消息中提取图像数据
#             # 注意：msg.data 是图像数据的字节数组
#             depth_data = np.frombuffer(msg.data, dtype=np.uint16)
            
#             # 获取图像的宽度、高度和通道数
#             height = msg.height
#             width = msg.width
            
#             # 重塑数据为图像格式
#             depth = depth_data.reshape((height, width)).astype(np.float32)/1000.0

#             depth[depth>10] = 0
#             self.depth = depth

#         except Exception as e:
#             rospy.logerr("Error parsing depth data: %s", str(e))

#     def _camera_info_callback(self,msg):
#         if self.depth is None:
#             return
#         if self.intrinsic is None:
#             self.intrinsic = {'fx':msg.K[0],
#                         'fy':msg.K[4],
#                         'cx':msg.K[2],
#                         'cy':msg.K[5],
#                         'height':self.depth.shape[0],
#                         'weight':self.depth.shape[1]}
#             self.distortion_coefficients = np.array(msg.D)
            
            
#     def _rgb2_callback(self, msg):
#         try:
#             # 从消息中提取图像数据
#             # 注意：msg.data 是图像数据的字节数组
#             img_data = np.frombuffer(msg.data, dtype=np.uint8)
            
#             # 获取图像的宽度、高度和通道数
#             height = msg.height
#             width = msg.width
#             channels = 3  # 假设是 RGB 图像，3 个通道
            
#             # 重塑数据为图像格式
#             self.rgb2 = img_data.reshape((height, width, channels))
#         except Exception as e:
#             rospy.logerr("Error parsing raw image data: %s", str(e))

#     def _depth2_callback(self, msg):
#         try:
#             # 从消息中提取图像数据
#             # 注意：msg.data 是图像数据的字节数组
#             depth_data = np.frombuffer(msg.data, dtype=np.uint16)
            
#             # 获取图像的宽度、高度和通道数
#             height = msg.height
#             width = msg.width
            
#             # 重塑数据为图像格式
#             depth = depth_data.reshape((height, width)).astype(np.float32)/1000.0

#             depth[depth>10] = 0
#             self.depth2 = depth

#         except Exception as e:
#             rospy.logerr("Error parsing depth data: %s", str(e))

#     def _camera2_info_callback(self,msg):
#         if self.depth2 is None:
#             return
#         if self.intrinsic2 is None:
#             self.intrinsic2 = {'fx':msg.K[0],
#                         'fy':msg.K[4],
#                         'cx':msg.K[2],
#                         'cy':msg.K[5],
#                         'height':self.depth.shape[0],
#                         'weight':self.depth.shape[1]}
#             self.distortion_coefficients2 = np.array(msg.D)
            
            
#     def get_observation(self,reset=False,matcher=None,source_frame=None,positive_target_keypoint=None):
#         rgb = self.rgb.copy()
#         w = min(self.max_w, rgb.shape[1])
#         h = int(w*rgb.shape[0]/rgb.shape[1])
#         rgb = cv2.resize(rgb,(w, h), interpolation=cv2.INTER_NEAREST)
#         depth = self.depth.copy()
#         depth = cv2.resize(depth,(w, h), interpolation=cv2.INTER_NEAREST)
        
#         rgb2 = self.rgb2.copy()
#         w2 = min(self.max_w, rgb2.shape[1])
#         h2 = int(w2*rgb2.shape[0]/rgb2.shape[1])
#         rgb2 = cv2.resize(rgb2,(w2, h2), interpolation=cv2.INTER_NEAREST)
#         depth2 = self.depth2.copy()
#         depth2 = cv2.resize(depth2,(w2, h2), interpolation=cv2.INTER_NEAREST)

#         if reset:
#             self.mask_predictor.sequence_predictor_initialize(rgb)
#             self.mask_predictor2.sequence_predictor_initialize(rgb2) 
            
#         positive_mask,passive_mask = self.mask_predictor.sequence_predictor_track(rgb)
#         positive_mask = binary_erosion(positive_mask.cpu().numpy(),structure=np.ones((5,5),dtype=bool))
#         passive_mask = binary_erosion(passive_mask.cpu().numpy(),structure=np.ones((5,5),dtype=bool))
#         positive_mask = torch.tensor(positive_mask,device=self.device)
#         passive_mask = torch.tensor(passive_mask,device=self.device)
        
#         positive_mask2,passive_mask2 = self.mask_predictor2.sequence_predictor_track(rgb2)
#         positive_mask2 = binary_erosion(positive_mask2.cpu().numpy(),structure=np.ones((5,5),dtype=bool))
#         passive_mask2 = binary_erosion(passive_mask2.cpu().numpy(),structure=np.ones((5,5),dtype=bool))
#         positive_mask2 = torch.tensor(positive_mask2,device=self.device)
#         passive_mask2 = torch.tensor(passive_mask2,device=self.device)

#         if self.intrinsic['height'] == depth.shape[0]:
#             intrinsics = self.intrinsic
#         else:
#             intrinsics = {}
#             scale = depth.shape[0] / self.intrinsic['height']
#             intrinsics['fx'] = scale * self.intrinsic['fx']
#             intrinsics['fy'] = scale * self.intrinsic['fy']
#             intrinsics['cx'] = scale * self.intrinsic['cx']
#             intrinsics['cy'] = scale * self.intrinsic['cy']
            
            
#         if self.intrinsic2['height'] == depth2.shape[0]:
#             intrinsics2 = self.intrinsic2
#         else:
#             intrinsics2 = {}
#             scale2 = depth2.shape[0] / self.intrinsic2['height']
#             intrinsics2['fx'] = scale2 * self.intrinsic2['fx']
#             intrinsics2['fy'] = scale2 * self.intrinsic2['fy']
#             intrinsics2['cx'] = scale2 * self.intrinsic2['cx']
#             intrinsics2['cy'] = scale2 * self.intrinsic2['cy']

#         T_ee_world = self.robot.get_pose()
#         gripper2base = np.eye(4)
#         gripper2base[:3,3] = T_ee_world.translation
#         gripper2base[:3,:3] = quaternion_to_matrix_np(T_ee_world.quaternion)
        
#         frame = {'cam1':{
#             'rgb':torch.tensor(rgb,device=self.device),   # h,w,3, torch.uint8, 0~255
#             'depth':torch.tensor(depth,device=self.device), # h,w, torch.float32
#             'positive_mask':positive_mask,   # h,w, torch.bool
#             'passive_mask':passive_mask,
#             'intrinsics':intrinsics,
#             'cam2base':self.cam2base,
#             'gripper2base':gripper2base,},

#             'cam2':{
#             'rgb':torch.tensor(rgb2,device=self.device),   # h,w,3, torch.uint8, 0~255
#             'depth':torch.tensor(depth2,device=self.device), # h,w, torch.float32
#             'intrinsics':intrinsics2,
#             'cam2base':self.cam2base2,
#             'positive_mask':positive_mask2,   # h,w, torch.bool
#             'passive_mask':passive_mask2,
#             }
#         }

#         return frame
    
#     def grasp(self,frame,use_mask = True,use_two_cam= True):
#         frame = self.get_observation(reset=True)
        
#         # 手動選擇想要抓取的點
#         grasppoint_cam1space = select_keypoint(frame['cam1'])   # cam1或者cam2,形狀是1，3
#         grasppoint_basespace = transform_pointcloud(grasppoint_cam1space,frame['cam1']['cam2base'])
        
#         if use_mask:
#             point_cloud1_camspace = depth_map_to_pointcloud_torch(frame['cam1']['depth'],frame['cam1']['positive_mask'],frame['cam1']['intrinsics']).cpu().numpy()
#             point_cloud1_basespace = transform_pointcloud(point_cloud1_camspace,frame['cam1']['cam2base'])
            
#             point_cloud2_camspace = depth_map_to_pointcloud_torch(frame['cam2']['depth'],frame['cam2']['positive_mask'],frame['cam2']['intrinsics']).cpu().numpy()
#             point_cloud2_basespace = transform_pointcloud(point_cloud2_camspace,frame['cam2']['cam2base'])
#         else:
#             point_cloud1_camspace = depth_map_to_pointcloud_torch(frame['cam1']['depth'],torch.ones_like(frame['cam1']['depth'],dtype=torch.bool),frame['cam1']['intrinsics']).cpu().numpy()
#             point_cloud1_basespace = transform_pointcloud(point_cloud1_camspace,frame['cam1']['cam2base'])
            
#             point_cloud2_camspace = depth_map_to_pointcloud_torch(frame['cam2']['depth'],torch.ones_like(frame['cam2']['depth'],dtype=torch.bool),frame['cam2']['intrinsics']).cpu().numpy()
#             point_cloud2_basespace = transform_pointcloud(point_cloud2_camspace,frame['cam2']['cam2base'])
            
#         if use_two_cam:
#             point_cloud_basespace = np.concatenate([point_cloud1_basespace,point_cloud2_basespace],axis=0)
#         else:
#             point_cloud_basespace = point_cloud1_basespace
            
#         grasp_pose_list = self.run_gsnet(point_cloud_basespace)
        
#         return grasp_pose_list
        
#     def run_gsnet(self,point_cloud_basespace):
#         url = "http://127.0.0.1:5001/gsnet_flask"  # 服务端地址
#         headers = {"Content-Type": "application/json"}
#         data = {
#             "pointcloud": point_cloud_basespace.tolist()
#         }

#         response = requests.post(url, json=data, headers=headers)

#         grasp_pose_list = np.array(response.json()['grasp_pose_list'])
        
#         return grasp_pose_list
    
 
        
#     #     select best grasp pose
#     #     從grasp_pose_list中選擇與want pose最接近的抓取姿勢
#     #     want_pose = np.array(???)
#     #     want_pose[:3,:3] = 手動給一個旋轉
#     #     want_pose[:3,3] = grasppoint_basespace[0]
#     #    計算與wantpose最接近的grasp_pose
#     #    grasp_pose = ???
        
        
#         # cv2.drawFrameAxes(frame['cam1']['rgb'].cpu().numpy,np.array([[frame['cam1']['intrinsics']['fx'],0,frame['cam1']['intrinsics']['cx']],
#         #                     [0,frame['cam1']['intrinsics']['fy'],frame['cam1']['intrinsics']['cy']],
#         #                     [0,0,1]]),np.array([0.,0.,0.,0.,0.]),cv2.Rodrigues(grasp_pose[:3,:3])[0],grasp_pose[:3,3],0.05)
#         # cv2.imshow('grasp pose in camspace',image_show)
#         # cv2.waitKey(5000)
#         # cv2.destroyAllWindows()
        
#         # self.robot.goto_pose(grasp_pose)


def grasp_gsnet(frame,target_point,use_mask = False,use_two_cam= False,use_camspace=True):
 
    if use_mask:
        point_cloud1_camspace = depth_map_to_pointcloud_torch(frame['cam1']['depth'],frame['cam1']['positive_mask'],frame['cam1']['intrinsics']).cpu().numpy()
        point_cloud1_basespace = transform_pointcloud(point_cloud1_camspace,frame['cam1']['cam2base'])
        
        point_cloud2_camspace = depth_map_to_pointcloud_torch(frame['cam2']['depth'],frame['cam2']['positive_mask'],frame['cam2']['intrinsics']).cpu().numpy()
        point_cloud2_basespace = transform_pointcloud(point_cloud2_camspace,frame['cam2']['cam2base'])
    else:
        point_cloud1_camspace = depth_map_to_pointcloud_torch(frame['cam1']['depth'],torch.ones_like(frame['cam1']['depth'],dtype=torch.bool),frame['cam1']['intrinsics']).cpu().numpy()
        point_cloud1_basespace = transform_pointcloud(point_cloud1_camspace,frame['cam1']['cam2base'])
        
        point_cloud2_camspace = depth_map_to_pointcloud_torch(frame['cam2']['depth'],torch.ones_like(frame['cam2']['depth'],dtype=torch.bool),frame['cam2']['intrinsics']).cpu().numpy()
        point_cloud2_basespace = transform_pointcloud(point_cloud2_camspace,frame['cam2']['cam2base'])
        
    if use_two_cam:
        point_cloud_basespace = np.concatenate([point_cloud1_basespace,point_cloud2_basespace],axis=0)
    else:
        point_cloud_basespace = point_cloud1_basespace
        
    mask = (point_cloud_basespace[:,2] > 0.1)*(point_cloud_basespace[:,2] < 0.9)*(point_cloud_basespace[:,0] > 0.1)*(point_cloud_basespace[:,0] < 1.5)*(point_cloud_basespace[:,1] > -0.7)*(point_cloud_basespace[:,1] < 0.7)
    point_cloud_basespace = point_cloud_basespace[mask]
    
    url = "http://127.0.0.1:5002/gsnet_flask"  # 服务端地址
    headers = {"Content-Type": "application/json"}
    
    if use_camspace:
        # 把两个相机的点云都转到cam1的坐标系下
        point_cloud_camspace = transform_pointcloud(point_cloud_basespace,np.linalg.inv(frame['cam1']['cam2base']))
        target_point_camspace = transform_pointcloud(target_point, np.linalg.inv(frame['cam1']['cam2base']))
        data = {
        "pointcloud": point_cloud_camspace.tolist(),   # n,3
        'target_point':target_point_camspace.tolist()    # 1,3
        }
        response = requests.post(url, json=data, headers=headers)
        grasp_pose_camspace_group = np.array(pickle.loads(response.content))  # n,4,4
        
    else:
        data = {
        "pointcloud": point_cloud_basespace.tolist(),   # n,3
        'target_point':target_point.tolist()    # 1,3
        }

        response = requests.post(url, json=data, headers=headers)
        grasp_pose_basespace = np.array(pickle.loads(response.content))  # n,4,4
    

    
    return grasp_pose_camspace_group


    
# if __name__ == '__main__':
    
#     observer = Observer(False)  

#     grasp_pose_list = observer.grasp()  # 调用 grasp 方法

#     print(f"Grasp Pose List: {grasp_pose_list}")



    







