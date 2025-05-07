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
from utils import filter_waypoints,quaternion_to_matrix_np,rotation_to_quaternion_np, calculate_2d_projections, smooth_trajectory_with_initial_position,smooth_trajectory_fitting_with_smoothness
from utils import interpolate_transform
from autolab_core import RigidTransform
from grasping.api import run_grasping
from online_point_track import OnlineCoTracker
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from franka_interface_msgs.msg import SensorDataGroup
import os
from test_grasp import grasp_gsnet
import pickle
import requests
import os
from datetime import datetime
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Bool
import tf.transformations

class Controller:
    def __init__(self,demo_scene_id,demo_start_id,demo_end_id,dynamic_keypoint=False,positive_obj=None,passive_obj=None,task=None):
        print('Controller init')
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.mkdir(f'./save_vis/{current_time}')
        self.save_dir = f'./save_vis/{current_time}'
        demo_length = demo_end_id - demo_start_id
        self.positive_obj = positive_obj
        self.passive_obj = passive_obj        
        self.task = task
        if self.passive_obj is None:
            self.articulate = True
        else:
            self.articulate = False

        for filename in os.listdir('./tmp_image'):
            os.remove(f'./tmp_image/{filename}')

        self.dynamic_keypoint = dynamic_keypoint
        print('dynamic_keypoint:',dynamic_keypoint)
        self.observer = Observer(self.dynamic_keypoint,positive_obj,passive_obj)
        time.sleep(2)
        # self.observer.robot.open_gripper()
        # self.observer.robot.reset_joints()
        # self.observer.robot.goto_joints([-0.10784433, -1.34797267,  0.12052593 ,-2.22357924 , 0.12155288 , 0.91885057,0.84606499],duration=5)
        # self.observer.robot.goto_joints([-0.07873095, -1.0950896 ,  0.01880419 ,-2.13605741 , 0.07799229 , 1.58570196 ,0.68514057])
        self.goto_pose(np.array([[ 0.9518 , -0.13755 , 0.27396,0.42127181],
                        [ -0.1465, -0.9891,  0.0125,-0.0563],
                        [ 0.26926 , -0.052 ,-0.961658,0.6889],
                        [0,0,0,1]]))
        self.matcher = Matcher(save_dir=self.save_dir)
        self.demo_loader = D415Data(estimate_depth=False,positive_obj=positive_obj,passive_obj=passive_obj)
        self.tracker = PointTracker('cotracker',self.matcher.feature_extracter,self.save_dir)
        
        
        self.observer_init = False
        self.positive_online_tracker = None
        self.observer_timer = rospy.Timer(rospy.Duration(0.5), self._observe_loop)
        
        # self.observer.robot.close_gripper()

        np.save(f'{self.save_dir}/c2b.npy',self.observer.cam2base)
        
        # 中间位置手动递茶壶
        #         self.goto_pose(np.array([[ 0.98101372, -0.06819174,  0.18150303,0.4941],
        #  [-0.09151538 ,-0.98811915,  0.1233958,0.1582 ],
        #  [ 0.17093204, -0.13766329, -0.97561782,0.4298],[0,0,0,1]]))

        
        self.demo_seq = self.demo_loader.get_sequence(scene_id=demo_scene_id, start_id=demo_start_id, length=demo_length)
        self.scene_id = demo_scene_id
        
        self.grasping = False
        self.draw_execute= False
        self.p2c_curr_init = None
        self.record_frames = []
        self.draw_execute_list = []
        self.save_execute_video_list = []
    
    def _observe_loop(self,e):
        # 只是为了让sam2的track不因跨度太大而失败
        if self.observer_init:
            self.current_frame = self.observer.get_observation(reset=False)
            if self.grasping:
                self.record_frames.append(self.current_frame)
                
            if (self.draw_execute) and (self.p2c_curr_init is not None):
                image = self.current_frame['cam1']['rgb'].cpu().numpy()
                self.draw_execute_list.append(self.draw_coord(image,self.current_frame['cam1']['intrinsics'],self.p2c_curr_init.copy(),show=False,save=False))
                self.save_execute_video_list.append(image)
                
            
        # # 同时做track，跟踪主动物体位姿
        # if self.positive_online_tracker is not None:
        #     self.positive_online_tracker.track_pose(self.current_frame)
        

    def close_loop_execute(self,):
        # step1 get trajectory from demo: a queue of 6d pose of objpositive2objpassive
        pass
        # step2 close-loop execution
        # while len(trajectory)>0:
            # step2-1 get observation of positive and passive 

            # step2-2 track current pose of positive and passive

            # step2-3 execute the first pose of queue

    def execute(self,):
        # step1 get initial observation of positive and passive

        # step2 get trajectory from demo: a queue of 6d pose of objpositive2objpassive

        # generate_keypoint(self.demo_seq,self.matcher.feature_extracter)
        positivenew2passive_list ,init_positive2cam_demo,init_passive2cam_demo= self.process_demo(self.demo_seq,self.task, self.positive_obj, self.passive_obj)
        torch.cuda.empty_cache()
        to_be_grasp_frame = self.observer.get_observation(reset=True,matcher=self.matcher,source_frame=self.demo_seq[0])
        # print('to_be_grasp_frame:',to_be_grasp_frame)
        cv2.imwrite(f'{self.save_dir}/to_be_grasp_frame.png',cv2.cvtColor(to_be_grasp_frame['cam1']['rgb'].cpu().numpy(),cv2.COLOR_RGB2BGR))
        self.observer_init = True
        # self.observer.get_all_keypoint_initframe(to_be_grasp_frame['cam1'],self.demo_seq,self.matcher)

        # 用尺度作用于轨迹
        scales = to_be_grasp_frame['cam1']['scales']    # 3
        print('to_be_grasp_frame')
        print(to_be_grasp_frame['cam1'])
        print(to_be_grasp_frame['cam1']['scales'])
        for i in range(len(positivenew2passive_list)):
            positivenew2passive = positivenew2passive_list[i]
            positivenew2passive[:3,3] = positivenew2passive[:3,3] * scales
            positivenew2passive_list[i] = positivenew2passive

        init_positive2cam_target = to_be_grasp_frame['cam1']['cs2ct_positive']@init_positive2cam_demo
        init_passive2cam_target = to_be_grasp_frame['cam1']['cs2ct_passive']@init_passive2cam_demo
        tmp = self.draw_coord(to_be_grasp_frame['cam1']['rgb'].cpu().numpy(),to_be_grasp_frame['cam1']['intrinsics'],init_positive2cam_target,show=False,save=False)
        tmp = self.draw_coord(tmp,to_be_grasp_frame['cam1']['intrinsics'],init_passive2cam_target,name='ungrasped_init2cam',show=False,save=True)

        # self.positive_online_tracker = OnlineCoTracker(init_positive2cam_target)
        # time.sleep(2)
        # self.positive_online_tracker.draw_pose_tracked()

        # step3 grasp positive obj
        # start_grasping_frame_id, first_grasped_positive2cam_target = self.grasp(self.demo_seq,to_be_grasp_frame,init_positive2cam_target)
        start_grasping_frame_id, first_grasped_positive2cam_target = self.grasp_with_other_camera(self.demo_seq,to_be_grasp_frame,init_positive2cam_target)
        torch.cuda.empty_cache()
        # 抓取之后，假设主动物体相对gripper是静止的，被动物体本身是静止的：
        # 记录第一帧的gripper2obj，即gripper2positiveinit;

        # if input('press q to quit this time execute') == 'q':
        #     return
        
        grasped_init_frame = self.observer.get_observation(reset=False,positive_target_keypoint=transform_pointcloud(self.observer.target_keypoint_list[0],first_grasped_positive2cam_target@np.linalg.inv(init_positive2cam_target))) # matcher=self.matcher,source_frame=self.demo_seq[start_grasping_frame_id])

        time.sleep(2)
        # first_grasped_positive2cam_target = self.positive_online_tracker.current_pose # @ init_positive2cam_target
        # init_passive2cam_target = grasped_init_frame['cs2ct_passive']@init_passive2cam_demo
        gripper2positiveinit = np.linalg.inv(first_grasped_positive2cam_target)@np.linalg.inv(grasped_init_frame['cam1']['cam2base'])@grasped_init_frame['cam1']['gripper2base']

        tmp = self.draw_coord(grasped_init_frame['cam1']['rgb'].cpu().numpy(),grasped_init_frame['cam1']['intrinsics'],first_grasped_positive2cam_target,show=False,save=False)
        tmp = self.draw_coord(tmp,grasped_init_frame['cam1']['intrinsics'],init_passive2cam_target,name='grasped_init2cam',show=False,save=True)


        # step4 semi-close-loop execution
        # 观测是实时更新的，但是假设了抓取后主动物体与gripper相对静止，假设了被动物体不动
        # 更新观测的目的是更新keypoint
        
        for positivenew2passive in positivenew2passive_list:
            tmp_pose = init_passive2cam_target@positivenew2passive
            tmp = self.draw_coord(tmp,grasped_init_frame['cam1']['intrinsics'],tmp_pose,show=False,save=False)
        cv2.imwrite(f'{self.save_dir}/trajectory.png',cv2.cvtColor(tmp,cv2.COLOR_RGB2BGR))
        

        execute_demo_frame_id = start_grasping_frame_id
        caminit2camfirstgrasp_positive =  first_grasped_positive2cam_target@ np.linalg.inv(init_positive2cam_target)        

        gripper2base_list = []
        while True:
            # 还原轨迹计算的是 positivenew2passive -> positivenew2cam -> positiveinit2cam -> gripper2cam -> gripper2base
            
            positivenew2passive = positivenew2passive_list[execute_demo_frame_id]
            # positivenew2passive -> positivenew2cam , 这里的前提是被动物体一直静止
            positivenew2cam = init_passive2cam_target @ positivenew2passive
            
            # positivenew2cam -> positiveinit2cam
            camnew2caminit_curr = np.eye(4)
            camnew2caminit_curr[:3,3] = -transform_pointcloud(self.observer.target_keypoint_list[0],(positivenew2cam)@np.linalg.inv(init_passive2cam_target@positivenew2passive_list[start_grasping_frame_id])) + transform_pointcloud(self.observer.target_keypoint_list[0],(positivenew2cam)@np.linalg.inv(init_passive2cam_target@positivenew2passive_list[start_grasping_frame_id]))
            
            positiveinit2cam = camnew2caminit_curr @ positivenew2cam
            
            # positiveinit2cam -> gripper2cam -> gripper2base
            gripper2cam = positiveinit2cam @ gripper2positiveinit
            gripper2base = grasped_init_frame['cam1']['cam2base'] @ gripper2cam
            gripper2base_list.append(gripper2base)

            execute_demo_frame_id = execute_demo_frame_id + 1
            
            if execute_demo_frame_id >= len(self.demo_seq):
                break

        # T_ee_world = self.observer.robot.get_pose()
        # gripper2base_curr = np.eye(4)
        # gripper2base_curr[:3,3] = T_ee_world.translation
        # gripper2base_curr[:3,:3] = quaternion_to_matrix_np(T_ee_world.quaternion)
        # 只执行
        # 伺服执行
        # gripper2base_list.insert(0,gripper2base_curr)
        
        
        # 对于非铰接，把初始1/3的轨迹删掉
        if not self.articulate:
            gripper2base_list = filter_waypoints(gripper2base_list)
            gripper2base_list = gripper2base_list[int(len(gripper2base_list)*0.5):]
        
        # waypoint_list_front_part = gripper2base_list[:len(gripper2base_list)//2]
        # waypoint_list_back_part = gripper2base_list[len(gripper2base_list)//2:]
        # waypoint_list_front_part = smooth_trajectory_fitting_with_smoothness(waypoint_list_front_part,num_points=10,t_smooth_factor=1.0,r_smooth_factor=1.0)
        # waypoint_list_back_part = smooth_trajectory_fitting_with_smoothness(waypoint_list_back_part,num_points=20,t_smooth_factor=0.0,r_smooth_factor=0.0)
        # waypoint_list = waypoint_list_front_part + waypoint_list_back_part
        # waypoint_list = smooth_trajectory_fitting_with_smoothness(waypoint_list,num_points=600,t_smooth_factor=0.05,r_smooth_factor=0.05)
        
        smooth_factor = 1.0 if self.articulate else 0.05
        waypoint_list = smooth_trajectory_fitting_with_smoothness(gripper2base_list,num_points=100,t_smooth_factor=smooth_factor,r_smooth_factor=smooth_factor)

        
        # np.save(f'{self.save_dir}/gripper2base_list',np.array(gripper2base_list))
        # np.save(f'{self.save_dir}/waypoint_list',np.array(waypoint_list))
        # np.save(f'{self.save_dir}/cam2base',grasped_init_frame['cam1']['cam2base'])
        # trajectory = [RigidTransform(translation=waypoint[:3,3],
        #                              rotation=waypoint[:3,:3],
        #                              from_frame='franka_tool',
        #                              to_frame='world') for waypoint in waypoint_list]
        self.goto_pose(waypoint_list[0])
        
        # self.observer.robot.goto_pose(trajectory[1], duration=10, dynamic=True, buffer_time=10,use_impedance=True if self.articulate else False)
        
        # init_time = rospy.Time.now().to_time()
        # pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        # rate = rospy.Rate(1 / 0.02)
        # self.draw_execute = True
        for i,waypoint in enumerate(waypoint_list):
            self.observer._publish_pose(waypoint)
            time.sleep(0.06)
            # timestamp = rospy.Time.now().to_time() - init_time
            # traj_gen_proto_msg = PosePositionSensorMessage(
            #     id=i, timestamp=timestamp, 
            #     position=waypoint.translation, quaternion=waypoint.quaternion
            # )
            # if self.articulate:
            #     fb_ctrlr_proto = CartesianImpedanceSensorMessage(
            #         id=i, timestamp=timestamp,
            #         translational_stiffnesses=[500.0,500.0,500.0],   # 默认是600
            #         rotational_stiffnesses=[40,40,40]   # 默认是50
            #     )
            # else:
            #     fb_ctrlr_proto = CartesianImpedanceSensorMessage(
            #         id=i, timestamp=timestamp,
            #         translational_stiffnesses=[800.0,800.0,800.0],   # 默认是600
            #         rotational_stiffnesses=[60,60,60]   # 默认是50
            #     )
            # ros_msg = make_sensor_group_msg(
            #     trajectory_generator_sensor_msg=sensor_proto2ros_msg(
            #         traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
            #     feedback_controller_sensor_msg=sensor_proto2ros_msg(
            #         fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE)
            #     )
            
            
            # g2b = np.eye(4)
            # g2b[:3,:3] = waypoint.rotation
            # g2b[:3,3] = waypoint.translation
            # self.p2c_curr_init = np.linalg.inv(grasped_init_frame['cam1']['cam2base'])@g2b@np.linalg.inv(gripper2positiveinit)

            
            # rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
            # pub.publish(ros_msg)
            # rate.sleep()
            
            
            # vis_frame = self.observer.get_observation(reset=False)
            # self.draw_coord(vis_frame['cam1'],positivenew2cam,name=f'new_{execute_demo_frame_id}',save=True)     # check
            # cv2.destroyAllWindows()
            # self.draw_coord(vis_frame['cam1'],positiveinit2cam,name=f'init_{execute_demo_frame_id}',save=True) # check
            # cv2.destroyAllWindows()
            # print('target position:',waypoint.translation)
            # print('current position:',self.observer.robot.get_pose().translation)
        # time.sleep(3)
        # self.draw_execute = False
        # self.save_video(self.draw_execute_list,f'{self.save_dir}/execute_video')
        # self.save_video(self.save_execute_video_list,f'{self.save_dir}/execute_video_rgb')
        
# 因为不需要动态更新了，所以提前计算好所有需要执行的轨迹，拟合一条平滑的轨迹并执行,下方应该是彻底废弃了
#         duration = 5
#         while True:
#             # time.sleep(5)
#             # 还原轨迹计算的是 positivenew2passive -> positivenew2cam -> positiveinit2cam -> gripper2cam -> gripper2base
            
#             # 当前gripper2base
#             T_ee_world = self.observer.robot.get_pose()
#             gripper2base_curr = np.eye(4)
#             gripper2base_curr[:3,3] = T_ee_world.translation
#             gripper2base_curr[:3,:3] = quaternion_to_matrix_np(T_ee_world.quaternion)

#             current_frame = self.observer.get_observation(reset=False,
# positive_target_keypoint=transform_pointcloud(self.observer.target_keypoint_list[execute_demo_frame_id],np.linalg.inv(grasped_init_frame['cam1']['cam2base'])@gripper2base_curr@np.linalg.inv(grasped_init_frame['cam1']['gripper2base'])@grasped_init_frame['cam1']['cam2base']@caminit2camfirstgrasp_positive))

#             positivenew2passive = positivenew2passive_list[execute_demo_frame_id]

#             # positivenew2passive -> positivenew2cam
#             positivenew2cam = init_passive2cam_target @ positivenew2passive

#             # positivenew2cam -> positiveinit2cam
#             camnew2caminit = np.eye(4)
#             camnew2caminit[:3,3] = - current_frame['cam1']['positive_keypoint'] + transform_pointcloud(grasped_init_frame['cam1']['positive_keypoint'],np.linalg.inv(current_frame['cam1']['cam2base'])@current_frame['cam1']['gripper2base']@np.linalg.inv(grasped_init_frame['cam1']['gripper2base'])@grasped_init_frame['cam1']['cam2base'])   # TODO 存疑，有待验证
#             positiveinit2cam = camnew2caminit @ positivenew2cam



#             # positiveinit2cam -> gripper2cam -> gripper2base
#             gripper2cam = positiveinit2cam @ gripper2positiveinit
#             gripper2base = current_frame['cam1']['cam2base'] @ gripper2cam

#             print('gripper2base',gripper2base)
#             self.goto_pose(gripper2base,duration)
#             duration = 1
            
#             vis_frame = self.observer.get_observation(reset=False,
# positive_target_keypoint=transform_pointcloud(self.observer.target_keypoint_list[execute_demo_frame_id],np.linalg.inv(grasped_init_frame['cam1']['cam2base'])@gripper2base_curr@np.linalg.inv(grasped_init_frame['cam1']['gripper2base'])@grasped_init_frame['cam1']['cam2base']@caminit2camfirstgrasp_positive))
            
#             self.draw_coord(vis_frame['cam1'],positivenew2cam,name=f'new_{execute_demo_frame_id}',save=True)     # check
#             cv2.destroyAllWindows()
            
#             self.draw_coord(vis_frame['cam1'],positiveinit2cam,name=f'init_{execute_demo_frame_id}',save=True) # check
#             cv2.destroyAllWindows()
            
#             execute_demo_frame_id = execute_demo_frame_id + 1
#             if execute_demo_frame_id >= len(self.demo_seq):
#                 print('executed all waypoints!!!!!!')
#                 break

    # def grasp(self,demo_seq,to_be_grasp_frame,init_positive2cam_target):
    #     # step1-1 detect the contact point in demo
    #     contact_point_demo, first_grasp_frame_id = find_contact_point(demo_seq) # 1,3
    #     # step1-2 map to observation
    #     match_inlier,_,_,_,_ = self.matcher.match_vfc_image(demo_seq[0],to_be_grasp_frame,which_obj='positive',vis=True)
    #     distance_source = np.linalg.norm(contact_point_demo - match_inlier[:,:3],axis=1)    # n
    #     target_contact_point = match_inlier[:,3:][np.argsort(distance_source)[:5]].mean(axis=0,keepdims=True)   # 1,3
        
    #     target_contact_point_2d = calculate_2d_projections(target_contact_point.transpose(),np.array([[to_be_grasp_frame['intrinsics']['fx'],0,to_be_grasp_frame['intrinsics']['cx']],
    #                         [0,to_be_grasp_frame['intrinsics']['fy'],to_be_grasp_frame['intrinsics']['cy']],
    #                         [0,0,1]]))[0]  # x,y
        
    #     image_show = cv2.cvtColor(to_be_grasp_frame['rgb'].cpu().numpy().copy(),cv2.COLOR_RGB2BGR)
    #     cv2.circle(image_show,(int(target_contact_point_2d[0]),int(target_contact_point_2d[1])),3,(0,0,255),-1)
    #     cv2.imshow('contact point',image_show)
    #     cv2.waitKey(2000)
    #     cv2.destroyAllWindows()

    #     # step1-3 transform to base space
    #     # target_contact_point_basespace = transform_pointcloud(target_contact_point,to_be_grasp_frame['cam2base'])

    #     # omnigrasp input data format
    #     # 'image': h,w,3, np,uint8
    #     # 'depth': h,w np,float32,m
    #     # 'cam_info''K':np.array((3,3)), 'W':w,'H':h,'scale':scale
    #     # 'mask': h,w np,bool
    #     # 'target_point': x,y

    #     point_mask = np.zeros_like(to_be_grasp_frame['positive_mask'].cpu().numpy())   
    #     point_mask[target_contact_point_2d[1],target_contact_point_2d[0]] = True    # y,x
    #     dist_to_point = distance_transform_edt(~point_mask)
    #     # dist_to_mask = distance_transform_edt(~to_be_grasp_frame['positive_mask'].cpu().numpy())
    #     target_mask = (dist_to_point <= 5) & to_be_grasp_frame['positive_mask'].cpu().numpy()
    #     print('target_mask.sum()',target_mask.sum())
        
    #     input_grasp_frame = {'image':to_be_grasp_frame['rgb'].cpu().numpy(),
    #                         'depth':to_be_grasp_frame['depth'].cpu().numpy(),
    #                         'cam_info':{'K':np.array([[to_be_grasp_frame['intrinsics']['fx'],0,to_be_grasp_frame['intrinsics']['cx']],
    #                         [0,to_be_grasp_frame['intrinsics']['fy'],to_be_grasp_frame['intrinsics']['cy']],
    #                         [0,0,1]]),'W':to_be_grasp_frame['depth'].cpu().numpy().shape[1],'H':to_be_grasp_frame['depth'].cpu().numpy().shape[0],'scale':1},
    #                         'mask':to_be_grasp_frame['positive_mask'].cpu().numpy(),
    #                         'target_point':target_contact_point_2d,
    #                         # 'target_mask':target_mask
    #                         # 'target_mask':to_be_grasp_frame['positive_mask'].cpu().numpy()
    #                         }

    #     grasp_pose_camspace = run_grasping(data=input_grasp_frame,T0=0.5)
    #     cv2.drawFrameAxes(image_show,np.array([[to_be_grasp_frame['intrinsics']['fx'],0,to_be_grasp_frame['intrinsics']['cx']],
    #                         [0,to_be_grasp_frame['intrinsics']['fy'],to_be_grasp_frame['intrinsics']['cy']],
    #                         [0,0,1]]),np.array([0.,0.,0.,0.,0.]),cv2.Rodrundistorted_rgbd
        
    #     pre_grasp2base = grasp2base @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-0.05],[0,0,0,1]])
    #     lift_grasp2base = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.1],[0,0,0,1]])@grasp2base
    #     self.goto_pose(pre_grasp2base,duration=5)
    #     self.grasping = True
    #     self.goto_pose(grasp2base,duration=2)
    #     self.observer.robot.close_gripper()
    #     self.goto_pose(lift_grasp2base)
    #     # self.observer.robot.reset_joints(duration=8)
    #     time.sleep(1.0)
    #     self.grasping = False
        
    #     graspedpositive2cam = self.tracker.track_pose(self.record_frames,init_positive2cam_target,which_obj='positive',grid_size=160,savename='target_positive',compute_last_only=True)[-1]     # 以第一帧的keypoint为坐标原点
    #     return first_grasp_frame_id,graspedpositive2cam

    def grasp_with_other_camera(self,demo_seq,to_be_grasp_frame,init_positive2cam_target):
        # step1-1 detect the contact point in demo
        # 改成在process demo的时候就做过了
        # contact_point_demo, first_grasp_frame_id = find_contact_point(demo_seq) # 1,3
        
        # step1-2 map to observation
        # match_inlier,_,_,_,_,rigid_target_point = self.matcher.match_vfc_image(demo_seq[0],to_be_grasp_frame['cam1'],which_obj='positive',vis=True)
        match_inlier, rigid_target_point = to_be_grasp_frame['cam1']['match_inlier'],to_be_grasp_frame['cam1']['rigid_target_point']
        
        distance_source = np.linalg.norm(self.contact_point_demo - match_inlier[:,:3],axis=1)    # n
        target_contact_point = match_inlier[:,3:][np.argsort(distance_source)[:1]].mean(axis=0,keepdims=True)   # 1,3
        
        
        # # gsnet
        # target_contact_point_basespace = transform_pointcloud(target_contact_point,to_be_grasp_frame['cam1']['cam2base'])
        # grasp_pose_camspace_group_all = grasp_gsnet(to_be_grasp_frame,target_contact_point_basespace,use_mask=False,use_two_cam=False,use_camspace=True)

        # grasp_dist = np.linalg.norm(grasp_pose_camspace_group_all[:,:3,3] - target_contact_point,axis=-1)   # n
        # grasp_mask = grasp_dist < 0.02
        # if grasp_mask.sum() == 0:
        #     grasp_mask = grasp_dist < 0.05
        # grasp_pose_camspace_group_all = grasp_pose_camspace_group_all[grasp_mask]
        # grasp_pose_basespace_group = np.array([to_be_grasp_frame['cam1']['cam2base']@grasp_pose_camspace for grasp_pose_camspace in grasp_pose_camspace_group_all])


        # omnigrasp
        target_contact_point = transform_pointcloud(target_contact_point, np.linalg.inv( to_be_grasp_frame['cam2']['cam2base'])@to_be_grasp_frame['cam1']['cam2base'])
        target_contact_point_2d = calculate_2d_projections(target_contact_point.transpose(),np.array([[to_be_grasp_frame['cam2']['intrinsics']['fx'],0,to_be_grasp_frame['cam2']['intrinsics']['cx']],
                            [0,to_be_grasp_frame['cam2']['intrinsics']['fy'],to_be_grasp_frame['cam2']['intrinsics']['cy']],
                            [0,0,1]]))[0]  # x,y
        
        point_mask = np.zeros_like(to_be_grasp_frame['cam2']['positive_mask'].cpu().numpy())   
        point_mask[target_contact_point_2d[1],target_contact_point_2d[0]] = True    # y,x
        dist_to_point = distance_transform_edt(~point_mask)
        # dist_to_mask = distance_transform_edt(~to_be_grasp_frame['positive_mask'].cpu().numpy())
        target_mask = (dist_to_point <= 15) & to_be_grasp_frame['cam2']['positive_mask'].cpu().numpy()
        print('target_mask.sum()',target_mask.sum())

        
        input_grasp_frame = {'image':to_be_grasp_frame['cam2']['rgb'].cpu().numpy(),
                    'depth':to_be_grasp_frame['cam2']['depth'].cpu().numpy(),
                    'cam_info':{'K':np.array([[to_be_grasp_frame['cam2']['intrinsics']['fx'],0,to_be_grasp_frame['cam2']['intrinsics']['cx']],
                    [0,to_be_grasp_frame['cam2']['intrinsics']['fy'],to_be_grasp_frame['cam2']['intrinsics']['cy']],
                    [0,0,1]]),'W':to_be_grasp_frame['cam2']['depth'].cpu().numpy().shape[1],'H':to_be_grasp_frame['cam2']['depth'].cpu().numpy().shape[0],'scale':1},
                    # 'mask':to_be_grasp_frame['positive_mask'].cpu().numpy(),
                    # 'target_point':target_contact_point_2d,
                    'target_mask':target_mask
                    # 'target_mask':to_be_grasp_frame['cam2']['positive_mask'].cpu().numpy()
                    }
        image_show = cv2.cvtColor(to_be_grasp_frame['cam2']['rgb'].cpu().numpy().copy(),cv2.COLOR_RGB2BGR)
        cv2.circle(image_show,(int(target_contact_point_2d[0]),int(target_contact_point_2d[1])),3,(0,0,255),-1)
        cv2.imwrite('./tmp_image/target_contact point.png',image_show)
        
        grasp_pose_camspace_group = run_grasping(data=input_grasp_frame,T0=0.5)
        grasp_pose_basespace_group = [to_be_grasp_frame['cam2']['cam2base']@grasp_pose_camspace for grasp_pose_camspace in grasp_pose_camspace_group]
        
        
        selected_id = self.select_best_grasp(grasp_pose_basespace_group)
        grasp_pose = grasp_pose_basespace_group[selected_id]
        
        grasp_pose_camspace = grasp_pose_camspace_group[selected_id]
        print('grasp_pose_camspace:',grasp_pose_camspace)
        cv2.drawFrameAxes(image_show,np.array([[to_be_grasp_frame['cam2']['intrinsics']['fx'],0,to_be_grasp_frame['cam2']['intrinsics']['cx']],
                            [0,to_be_grasp_frame['cam2']['intrinsics']['fy'],to_be_grasp_frame['cam2']['intrinsics']['cy']],
                            [0,0,1]]),np.array([0.,0.,0.,0.,0.]),cv2.Rodrigues(grasp_pose_camspace[:3,:3])[0],grasp_pose_camspace[:3,3],0.05)
        cv2.imwrite(f'{self.save_dir}/grasp_pose.png',image_show)


        #####
        grasp2base = grasp_pose@np.array([[-0.4480736, -0.8939967,  0.0000000,0],[0.8939967, -0.4480736,  0.0000000,0],[0.0000000,  0.0000000,  1.0000000 ,-0.0],[0,0,0,1]])
        grasp2base_rot180 = grasp2base@np.array([[ -1,  0,  0.0000000,0], [0, -1,  0.0000000,0],[0.0000000,  0.0000000,  1.0000000 ,0],[0,0,0,1]])

        reset_gripper2base = quaternion_to_matrix_np([ 2.51558578e-04 , 9.99996926e-01 ,-1.05503723e-04 ,-1.12252843e-03])
        rotdist1 = compute_rot_distance(grasp2base,reset_gripper2base)
        rotdist2 = compute_rot_distance(grasp2base_rot180,reset_gripper2base)
        if rotdist1 > rotdist2:
            grasp2base = grasp2base_rot180
        
        grasp2base = grasp2base @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-0.02],[0,0,0,1]])
        pre_grasp2base = grasp2base @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-0.05],[0,0,0,1]])
        pre_pre_grasp2base = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.2],[0,0,0,1]]) @ pre_grasp2base
        lift_grasp2base = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.1],[0,0,0,1]])@grasp2base
        
        self.goto_pose(pre_pre_grasp2base,duration=7)
        self.goto_pose(pre_grasp2base,duration=6)
        self.grasping = True
        self.goto_pose(grasp2base,duration=4)
        
        if self.articulate:
            # self.observer.robot.goto_gripper(force=50,width=0)
            self.observer._publish_open_gripper(False)
        else:
            #self.observer.robot.goto_gripper(force=50,width=0)
            self.observer._publish_open_gripper(False)
            time.sleep(2.0)
        
        if not self.articulate:
            self.goto_pose(lift_grasp2base,duration=4)

        time.sleep(2.0)
        self.grasping = False
    
        # graspedpositive2cam_list,_ = self.tracker.track_multiview_pose([self.record_frames[i] for i in range(len(self.record_frames))],init_positive2cam_target,which_obj='positive',savename='target_positive',rigid_point=rigid_target_point)     # 以第一帧的keypoint为坐标原点

        graspedpositive2cam_list,_ = self.tracker.track_pose([self.record_frames[i]['cam1'] for i in range(len(self.record_frames))],init_positive2cam_target,which_obj='positive',savename='target_positive',rigid_point=rigid_target_point)     # 以第一帧的keypoint为坐标原点
        graspedpositive2cam = graspedpositive2cam_list[-1]

        return self.first_grasp_frame_id,graspedpositive2cam
    
    
    def select_best_grasp(self,grasp_pose_basespace_list,):
        def calculate_angle_with_downward(matrix):
            # 提取方向向量
            direction_vector = matrix[:3, 2]
            # 竖直向下方向向量
            downward_vector = np.array([0, 0, -1])
            # 计算点积
            dot_product = np.dot(direction_vector, downward_vector)
            # 计算方向向量的模长
            norm_direction = np.linalg.norm(direction_vector)
            # 计算夹角
            cos_theta = dot_product / norm_direction
            theta = np.arccos(cos_theta)
            # 返回角度（弧度）
            return theta
        
        angle_list = []
        for grasp_pose_basespace in grasp_pose_basespace_list:
           angle_list.append(calculate_angle_with_downward(grasp_pose_basespace))
        selected_id = np.argmin(np.array(angle_list))  if (not self.articulate) else 0
        print('selected_id:',selected_id)
        return selected_id
    
    
    
    def goto_pose(self,pose,duration=7):
        # franka_pose = RigidTransform(
        #     rotation=rotation_to_quaternion_np(pose[:3,:3]),
        #     translation=pose[:3,3],
        #     from_frame='franka_tool', to_frame='world'
        # )

        # self.observer.robot.goto_pose(franka_pose,duration=duration,use_impedance=False)
        while True:
            try:
                steps = duration // 0.06
                target_pose_list = interpolate_transform(self.observer.g2b,pose,int(steps))
                for target_pose in target_pose_list:
                    self.observer._publish_pose(target_pose)
                    time.sleep(0.06)
                break
            except:
                print('waiting for gripper2base ros message')
                time.sleep(1.0)


    def draw_coord(self,image,intrinsics,pose,name=None,save=False,show=False):
        # pose: obj2cam
        tmp_image = image.copy()
        cv2.drawFrameAxes(tmp_image,np.array([[intrinsics['fx'],0,intrinsics['cx']],
                            [0,intrinsics['fy'],intrinsics['cy']],
                            [0,0,1]]),np.array([0.,0.,0.,0.,0.]),cv2.Rodrigues(pose[:3,:3])[0],pose[:3,3],0.05)

        if save:
            cv2.imwrite(f'{self.save_dir}/{name}.png',cv2.cvtColor(tmp_image,cv2.COLOR_RGB2BGR))
        if show:
            cv2.imshow(name,tmp_image)
            cv2.waitKey(3000)

        return tmp_image
        

    def process_demo(self,demo_seq,task,positive_obj,passvie_obj):    # 后续需要改成遥操的话，可以改写成一个类
        
        process_result = {}
        scene_idx = str(self.scene_id).rjust(6,'0')
        if os.path.exists(f'./test_data/{scene_idx}/processed_result.pkl'):
            with open(f'./test_data/{scene_idx}/processed_result.pkl','rb') as f:
                process_result = pickle.load(f)
       
        if 'contact_point_demo' not in process_result:
            contact_point_demo, first_grasp_frame_id = find_contact_point(demo_seq) # 1,3
            self.first_grasp_frame_id = first_grasp_frame_id
            self.contact_point_demo = contact_point_demo
            
            process_result['first_grasp_frame_id'] = first_grasp_frame_id
            process_result['contact_point_demo'] = contact_point_demo

            positive_keypoint,passive_keypoint = generate_keypoint(demo_seq,self.matcher.feature_extracter,self.passive_obj,task,positive_obj,passvie_obj,contact_point_demo,self.save_dir,process_result)
            process_result['positive_keypoint'] = positive_keypoint
            process_result['passive_keypoint'] = passive_keypoint
            
            # match_positive, init_rigid_transform_positive_camsource2camtarget,source_positive_keypoint, = self.matcher.match_vfc_image(demo_seq[0], target_frame,'positive')
            # match_passive, init_rigid_transform_passive_camsource2camtarget, source_passive_keypoint = self.matcher.match_vfc_image(demo_seq[0], target_frame,'passive')

            # if self.dynamic_keypoint:
            #     positive_keypoint_list = [demo_seq[i][f'positive_keypoint'] for i in range(len(demo_seq))]
            # else:
            positive_keypoint_list = [positive_keypoint]
            # passive_keypoint_list = [demo_seq[i][f'{'passive'}_keypoint'] for i in len(demo_seq)]
            passive_keypoint = demo_seq[0][f'passive_keypoint'] # 被动物体不动态

            init_positive2cam = np.eye(4)
            init_positive2cam[:3,3] = positive_keypoint_list[0]
            init_passive2cam = np.eye(4)
            init_passive2cam[:3,3] = passive_keypoint

            # init_positive2cam_target = init_rigid_transform_positive_camsource2camtarget@init_positive2cam_source
            # init_passive2cam_target = init_rigid_transform_passive_camsource2camtarget@init_passive2cam_source

            positiveinit2cam_list,rigid_point = self.tracker.track_pose(demo_seq,init_positive2cam,which_obj='positive',grid_size=160,savename='source_positive',articulate=self.articulate)     # 以第一帧的keypoint为坐标原点
            demo_seq[0]['rigid_point'] = rigid_point    # n,3
            process_result['rigid_point'] = rigid_point
            positivecurr2cam_list = []
            
            np.save(f'{self.save_dir}/positiveinit2cam_list',np.array(positiveinit2cam_list))
            np.save(f'{self.save_dir}/init_passive2cam',np.array(init_passive2cam))
            
            positive2new_vis_list = []
            positive2init_vis_list = []
            for i in range(len(demo_seq)):
                # 求positivecurr2cam_list
                # positivecurr2cam =  inv(camcurr2caminit) @ positive2caminit
                positiveinit2cam = positiveinit2cam_list[i]
                camcurr2caminit = np.eye(4)     # 注意，我们并不知道objcurr2objinit,只知道相机坐标系下的位移
                if self.dynamic_keypoint:
                    camcurr2caminit[:3,3] =  transform_pointcloud(positive_keypoint_list[0],(positiveinit2cam_list[i])@np.linalg.inv(positiveinit2cam_list[0])) - positive_keypoint_list[i] # 通过track的旋转平移把第一帧的keypoint转换到当前帧下，然后计算与当前帧新keypoint的位移
                else:
                    camcurr2caminit[:3,3] = [0,0,0]
                positivecurr2cam = np.linalg.inv(camcurr2caminit) @ positiveinit2cam
                positivecurr2cam_list.append(positivecurr2cam)

                positive2new_vis_list.append(self.draw_coord(demo_seq[i]['rgb'].cpu().numpy(),demo_seq[i]['intrinsics'],positivecurr2cam,save=False,show=False))
                positive2init_vis_list.append(self.draw_coord(demo_seq[i]['rgb'].cpu().numpy(),demo_seq[i]['intrinsics'],positiveinit2cam,save=False,show=False))

            self.save_video(positive2new_vis_list,f'{self.save_dir}/demo_positive2new')
            self.save_video(positive2init_vis_list,f'{self.save_dir}/demo_positive2init')

            # 假设被动物体是静止的，就不用track了
            # passiveinit2cam_list = self.tracker.track_pose(demo_seq,init_passive2cam,which_obj='passive',grid_size=160,savename='source_passive')     # 以第一帧的keypoint为坐标原点,如何考虑keypoint变换导致的translation？        被动物体不动态
            # 画一下被动的坐标系
            self.draw_coord(demo_seq[0]['rgb'].cpu().numpy(),demo_seq[0]['intrinsics'],init_passive2cam,name='passive_source',save=True,show=False)

            # positive2passive_list = [np.linalg.inv(passiveinit2cam_list[i]) @ positivecurr2cam_list[i] for i in range(len(passiveinit2cam_list))]   # 被动物体不动态
            positive2passive_list = [np.linalg.inv(init_passive2cam) @ positivecurr2cam_list[i] for i in range(len(positivecurr2cam_list))]   # 被动物体不动态
            process_result['positive2passive_list'] = positive2passive_list
            process_result['init_positive2cam'] = init_positive2cam
            process_result['init_passive2cam'] = init_passive2cam
            
            # save dict with pickle
            scene_idx = str(self.scene_id).rjust(6,'0')
            with open(f'./test_data/{scene_idx}/processed_result.pkl','wb') as f:
                pickle.dump(process_result,f)
            
            
        else:
            self.first_grasp_frame_id = process_result['first_grasp_frame_id']
            self.contact_point_demo = process_result['contact_point_demo']
            positive2passive_list = process_result['positive2passive_list']
            init_positive2cam = process_result['init_positive2cam']
            init_passive2cam = process_result['init_passive2cam']
            demo_seq[0]['rigid_point'] = process_result['rigid_point']
            demo_seq[0]['positive_keypoint'] = process_result['positive_keypoint']
            demo_seq[0]['passive_keypoint'] = process_result['passive_keypoint']

        return positive2passive_list,init_positive2cam,init_passive2cam    # 更新的坐标系的相对位姿

    def save_video(self,images,name):
        video_writer = cv2.VideoWriter(f'{name}.avi',cv2.VideoWriter_fourcc(*'mp4v'),10,(images[0].shape[1],images[0].shape[0]))
        for image in images:
            video_writer.write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        print(f'save video to {name}.avi')
    
    # def get_target_keypoint_with_demo(self,target_frame,demo,positivenew2passive_list,init_passive2cam_demo):
    #     target_keypoint_demo0space_list = []
    #     positive2cam_list_demo = [init_passive2cam_demo@positivenew2passive_list[i] for i in range(len(positivenew2passive_list))]
    #     for i in len(demo):
    #         _, _,_,target_keypoint_candidate,_,_ = self.matcher.match_vfc_image(demo[i],target_frame,'positive',)
    #         target_keypoint_demo0space = transform_pointcloud(target_keypoint_candidate,positive2cam_list_demo[0]@np.linalg.inv(positive2cam_list_demo[i])) # 1,3
    #         target_keypoint_demo0space_list.append(target_keypoint_demo0space)

        
        
class Observer:
    def __init__(self,dynamic_keypoint,positive_obj=None,passive_obj=None):
        print('Observer init')
        self.positive_obj=positive_obj
        self.passive_obj=passive_obj
        
        self.dynamic_keypoint = False
        self.rgb = None # np, h,w,3
        self.depth = None
        self.intrinsic = None
        self.cam2base = np.load('./c2b.npy')
        self.rgb2 = None # np, h,w,3
        self.depth2 = None
        self.intrinsic2 = None
        self.cam2gripper2 = np.load('./c2g2.npy')
        
        self._rgb1_sub = rospy.Subscriber('/obs/rgb1', Image, self._rgb_callback)
        self._rgb2_sub = rospy.Subscriber('/obs/rgb2', Image, self._rgb2_callback)
        self._depth1_sub = rospy.Subscriber('/obs/depth1', Image, self._depth_callback)
        self._depth2_sub = rospy.Subscriber('/obs/depth2', Image, self._depth2_callback)
        self._camera1_sub = rospy.Subscriber('obs/camera1_info', CameraInfo,self._camera_info_callback)
        self._camera2_sub = rospy.Subscriber('obs/camera2_info', CameraInfo,self._camera2_info_callback)
        self._g2b_sub = rospy.Subscriber('/obs/g2b', Pose, self._g2b_callback)
        
        self._pose_pub = rospy.Publisher("/target_pose", Pose, queue_size=10)
        self._gripper_pub = rospy.Publisher('/open_gripper', Bool, queue_size=1)
        self._publish_open_gripper(True)
        time.sleep(1.0)
        self.max_w=640
        self.passive_keypoint = None

        self.mask_predictor = MaskPredictor()
        self.mask_predictor2 = MaskPredictor()
        self.device='cuda'
        
        

    def _g2b_callback(self,msg):
        position = msg.position
        orientation = msg.orientation

        # 将四元数转换为旋转矩阵
        rotation_matrix = tf.transformations.quaternion_matrix([orientation.x, orientation.y, orientation.z, orientation.w])

        # 创建变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix[:3, :3]
        transform_matrix[:3, 3] = [position.x, position.y, position.z]

        self.g2b = transform_matrix
        
        
    def _publish_pose(self, target_pose):
        pose_msg = Pose()
        pose_msg.position.x = target_pose[0, 3]
        pose_msg.position.y = target_pose[1, 3]
        pose_msg.position.z = target_pose[2, 3]

        quaternion = R.from_matrix(target_pose[:3, :3]).as_quat()
        pose_msg.orientation.x = quaternion[0]
        pose_msg.orientation.y = quaternion[1]
        pose_msg.orientation.z = quaternion[2]
        pose_msg.orientation.w = quaternion[3]

        self._pose_pub.publish(pose_msg)
    
    def _publish_open_gripper(self,grasp):
        grasp_msg = Bool()
        grasp_msg.data = grasp
        self._gripper_pub.publish(grasp_msg)
        
    def _rgb_callback(self, msg):
        try:
            # 从消息中提取图像数据
            # 注意：msg.data 是图像数据的字节数组
            img_data = np.frombuffer(msg.data, dtype=np.uint8)
            
            # 获取图像的宽度、高度和通道数
            height = msg.height
            width = msg.width
            channels = 3  # 假设是 RGB 图像，3 个通道
            
            # 重塑数据为图像格式
            self.rgb = img_data.reshape((height, width, channels))
        except Exception as e:
            rospy.logerr("Error parsing raw image data: %s", str(e))

    def _depth_callback(self, msg):
        try:
            # 从消息中提取图像数据
            # 注意：msg.data 是图像数据的字节数组
            depth_data = np.frombuffer(msg.data, dtype=np.uint16)
            
            # 获取图像的宽度、高度和通道数
            height = msg.height
            width = msg.width
            
            # 重塑数据为图像格式
            depth = depth_data.reshape((height, width)).astype(np.float32)/1000.0

            depth[depth>10] = 0
            self.depth = depth

        except Exception as e:
            rospy.logerr("Error parsing depth data: %s", str(e))

    def _camera_info_callback(self,msg):
        if self.depth is None:
            return
        if self.intrinsic is None:
            self.intrinsic = {'fx':msg.K[0],
                        'fy':msg.K[4],
                        'cx':msg.K[2],
                        'cy':msg.K[5],
                        'height':self.depth.shape[0],
                        'weight':self.depth.shape[1]}
            print('cam1 intrinsic:',self.intrinsic)
            self.distortion_coefficients = np.array(msg.D)
            
            
    def _rgb2_callback(self, msg):
        try:
            # 从消息中提取图像数据
            # 注意：msg.data 是图像数据的字节数组
            img_data = np.frombuffer(msg.data, dtype=np.uint8)
            
            # 获取图像的宽度、高度和通道数
            height = msg.height
            width = msg.width
            channels = 3  # 假设是 RGB 图像，3 个通道
            
            # 重塑数据为图像格式
            self.rgb2 = img_data.reshape((height, width, channels))
        except Exception as e:
            rospy.logerr("Error parsing raw image data: %s", str(e))

    def _depth2_callback(self, msg):
        try:
            # 从消息中提取图像数据
            # 注意：msg.data 是图像数据的字节数组
            depth_data = np.frombuffer(msg.data, dtype=np.uint16)
            
            # 获取图像的宽度、高度和通道数
            height = msg.height
            width = msg.width
            
            # 重塑数据为图像格式
            depth = depth_data.reshape((height, width)).astype(np.float32)/1000.0

            depth[depth>10] = 0
            self.depth2 = depth

        except Exception as e:
            rospy.logerr("Error parsing depth data: %s", str(e))

    def _camera2_info_callback(self,msg):
        if self.depth2 is None:
            return
        if self.intrinsic2 is None:
            self.intrinsic2 = {'fx':msg.K[0],
                        'fy':msg.K[4],
                        'cx':msg.K[2],
                        'cy':msg.K[5],
                        'height':self.depth.shape[0],
                        'weight':self.depth.shape[1]}
            print('cam2 intrinsic:',self.intrinsic2)
            self.distortion_coefficients2 = np.array(msg.D)

        

            
    def get_observation(self,reset=False,matcher=None,source_frame=None,positive_target_keypoint=None):
        rgb = self.rgb.copy()
        w = min(self.max_w, rgb.shape[1])
        h = int(w*rgb.shape[0]/rgb.shape[1])
        rgb = cv2.resize(rgb,(w, h), interpolation=cv2.INTER_NEAREST)
        depth = self.depth.copy()
        depth = cv2.resize(depth,(w, h), interpolation=cv2.INTER_NEAREST)
        
        rgb2 = self.rgb2.copy()
        w2 = min(self.max_w, rgb2.shape[1])
        h2 = int(w2*rgb2.shape[0]/rgb2.shape[1])
        rgb2 = cv2.resize(rgb2,(w2, h2), interpolation=cv2.INTER_NEAREST)
        depth2 = self.depth2.copy()
        depth2 = cv2.resize(depth2,(w2, h2), interpolation=cv2.INTER_NEAREST)

        if reset:
            self.mask_predictor.sequence_predictor_initialize(rgb,self.positive_obj,self.passive_obj,gdino_name='gdino_annotated_cam1')
            self.mask_predictor2.sequence_predictor_initialize(rgb2,self.positive_obj,self.passive_obj,gdino_name='gdino_annotated_cam2') 
            
        positive_mask,passive_mask = self.mask_predictor.sequence_predictor_track(rgb)
        positive_mask = binary_erosion(positive_mask.cpu().numpy(),structure=np.ones((5,5),dtype=bool))
        passive_mask = binary_erosion(passive_mask.cpu().numpy(),structure=np.ones((5,5),dtype=bool))
        positive_mask = torch.tensor(positive_mask,device=self.device)
        passive_mask = torch.tensor(passive_mask,device=self.device)
        
        positive_mask2,passive_mask2 = self.mask_predictor2.sequence_predictor_track(rgb2)
        positive_mask2 = binary_erosion(positive_mask2.cpu().numpy(),structure=np.ones((5,5),dtype=bool))
        passive_mask2 = binary_erosion(passive_mask2.cpu().numpy(),structure=np.ones((5,5),dtype=bool))
        positive_mask2 = torch.tensor(positive_mask2,device=self.device)
        passive_mask2 = torch.tensor(passive_mask2,device=self.device)
        # positive_mask2,passive_mask2 = None,None

        if self.intrinsic['height'] == depth.shape[0]:
            intrinsics = self.intrinsic
        else:
            intrinsics = {}
            scale = depth.shape[0] / self.intrinsic['height']
            intrinsics['fx'] = scale * self.intrinsic['fx']
            intrinsics['fy'] = scale * self.intrinsic['fy']
            intrinsics['cx'] = scale * self.intrinsic['cx']
            intrinsics['cy'] = scale * self.intrinsic['cy']
            
            
        if self.intrinsic2['height'] == depth2.shape[0]:
            intrinsics2 = self.intrinsic2
        else:
            intrinsics2 = {}
            scale2 = depth2.shape[0] / self.intrinsic2['height']
            intrinsics2['fx'] = scale2 * self.intrinsic2['fx']
            intrinsics2['fy'] = scale2 * self.intrinsic2['fy']
            intrinsics2['cx'] = scale2 * self.intrinsic2['cx']
            intrinsics2['cy'] = scale2 * self.intrinsic2['cy']

        gripper2base = self.g2b

        frame = {'cam1':{
            'rgb':torch.tensor(rgb,device=self.device),   # h,w,3, torch.uint8, 0~255
            'depth':torch.tensor(depth,device=self.device), # h,w, torch.float32
            'positive_mask':positive_mask,   # h,w, torch.bool
            'passive_mask':passive_mask,
            'intrinsics':intrinsics,
            'cam2base':self.cam2base,
            'gripper2base':gripper2base,},

            'cam2':{
            'rgb':torch.tensor(rgb2,device=self.device),   # h,w,3, torch.uint8, 0~255
            'depth':torch.tensor(depth2,device=self.device), # h,w, torch.float32
            'intrinsics':intrinsics2,
            'cam2base':gripper2base@self.cam2gripper2,
            'positive_mask':positive_mask2,   # h,w, torch.bool
            'passive_mask':passive_mask2,
            }
        }

        if matcher is None:
            frame['cam1']['positive_keypoint'] = positive_target_keypoint
            frame['cam1']['passive_keypoint'] = self.passive_keypoint
            
            return frame
        else:
            # 但仅抓起来后的第一帧用，没写闭环 / 改成仅抓取前第一帧生成且使用
            # 追加一个从demo的keypoint计算当前帧keypoint的步骤
            #match_inlier,_,_,_,_,rigid_target_point
            match_inlier, transform,_,target_keypoint,scales,rigid_target_point  = matcher.match_vfc_image(source_frame,frame['cam1'],'positive',vis=True,)
            frame['cam1']['positive_keypoint'] = target_keypoint
            frame['cam1']['cs2ct_positive'] = transform
            
            frame['cam1']['match_inlier'] = match_inlier
            frame['cam1']['rigid_target_point'] = rigid_target_point
            
            self.target_keypoint_list = [target_keypoint]
            
            if self.passive_obj is not None:
                _, transform,_,target_keypoint,scales,_  = matcher.match_vfc_image(source_frame,frame['cam1'],'passive',vis=True)
                frame['cam1']['passive_keypoint'] = target_keypoint
                frame['cam1']['cs2ct_passive'] = transform
            else:
                frame['cam1']['passive_keypoint'] = target_keypoint
                frame['cam1']['cs2ct_passive'] = transform
                
            frame['cam1']['scales'] = scales    # 3
            if self.passive_keypoint is None:
                self.passive_keypoint = target_keypoint
                
            return frame

rospy.init_node("controller")