import rospy

from sensor_msgs.msg import Image,CameraInfo
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import shutil
import json
import time

class RGBDVideoRecorder:
    def __init__(self,save_dir, scene_id):
        rospy.init_node('rgbd_video_recorder',anonymous=True)
        self._rgb_sub = rospy.Subscriber('/camera/color/image_raw', Image, self._rgb_callback)
        self._depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self._depth_callback)
        self._timer = rospy.Timer(rospy.Duration(0.15), self._main_loop)
        self.camera_info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self._camera_info_callback)
        
        self.rgb = None # np, h,w,3
        self.depth = None
        self.intrinsic = None
        self.save_dir = save_dir
        self.scene_id = str(scene_id).rjust(6,'0')
        self.frame_id = 0

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
            rospy.loginfo(f"Received RGB image: width={width}, height={height}, channels={channels}")
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
            rospy.loginfo(f"Received Depth image: width={width}, height={height}")
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
            self.distortion_coefficients = np.array(msg.D)

            with open(f'{self.save_dir}/{self.scene_id}/camera_intrinsics.json','w') as f:
                json.dump(self.intrinsic,f)

    def _main_loop(self,e):

        if (self.depth is None) or (self.rgb is None):
            print('no image read')
            return
        
        rgb = self.rgb.copy()
        depth = self.depth.copy()

        print(self.frame_id)
        frame_idx = str(self.frame_id).rjust(6,'0')

        cv2.imwrite(f'{self.save_dir}/{scene_idx}/color/{frame_idx}_color.png',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))

        cv2.imwrite(f'{self.save_dir}/{scene_idx}/depth/{frame_idx}_depth.exr',depth)

        self.frame_id = self.frame_id + 1


save_dir = './test_data'
scene_id = 126
task_name = ' '
scene_idx = str(scene_id).rjust(6,'0')
if os.path.exists(f'{save_dir}/{scene_idx}'):
    delete_flag = input('same scene id exist,input y to delete,else exit')
    assert delete_flag == 'y'
    shutil.rmtree(f'{save_dir}/{scene_idx}',)


os.makedirs(f'{save_dir}/{scene_idx}/color')
os.makedirs(f'{save_dir}/{scene_idx}/depth')
with open(f'{save_dir}/{scene_idx}/task.txt', 'w', encoding='utf-8') as f:
    f.write(task_name)
    
recorder = RGBDVideoRecorder(save_dir,scene_id)
rospy.spin()
