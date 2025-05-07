from frankapy import FrankaArm, SensorDataMessageType
import time
from utils import filter_waypoints,quaternion_to_matrix_np,rotation_to_quaternion_np, calculate_2d_projections, smooth_trajectory_with_initial_position,smooth_trajectory_fitting_with_smoothness
from autolab_core import RigidTransform
from grasping.api import run_grasping
from online_point_track import OnlineCoTracker
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from franka_interface_msgs.msg import SensorDataGroup


robot = FrankaArm()

# robot.close_gripper()

robot.open_gripper()
