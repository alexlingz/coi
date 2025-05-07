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
import numpy as np

robot = FrankaArm()
# robot.reset_joints()

# T_ee_world = robot.get_pose()
# gripper2base_curr = np.eye(4)
# gripper2base_curr[:3,3] = T_ee_world.translation
# gripper2base_curr[:3,:3] = quaternion_to_matrix_np(T_ee_world.quaternion)

# pose = np.array([[ 0.85321459 , 0.05470053 , 0.51867007,0.23165633],
#                         [ 0.04739576, -0.99849237,  0.02734251,0.00612714],
#                         [ 0.51938353 , 0.00125729 ,-0.85453272,0.68188164],
#                         [0,0,0,1]])

# pose_list = interpolate_transform(gripper2base_curr,pose,10)

# franka_pose = RigidTransform(
#     rotation=rotation_to_quaternion_np(pose[:3,:3]),
#     translation=pose[:3,3],
#     from_frame='franka_tool', to_frame='world'
# )

# robot.goto_pose(franka_pose)

# # robot.goto_joints([-0.07873095, -1.0950896 ,  0.01880419 ,-2.13605741 , 0.07799229 , 1.58570196 ,0.68514057])

print(robot.get_joints())
print(robot.get_pose())