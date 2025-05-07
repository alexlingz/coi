import numpy as np
import torch
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R, Slerp

def farthest_point_sampling(points, n_samples):
    """
    Farthest point sampling using PyTorch tensors.

    Args:
        points (torch.Tensor): Tensor of shape (N, D) where N is the number of points, and D is the dimensionality.
        n_samples (int): The number of samples to select.

    Returns:
        torch.Tensor: Indices of the selected points of shape (n_samples,).
    """
    device = points.device
    N = points.shape[0]
    selected_pts = torch.zeros(n_samples, dtype=torch.long, device=device)
    dist_mat = torch.cdist(points, points, p=2)  # Pairwise Euclidean distance matrix
    
    # Start from the first point
    pt_idx = 0
    dist_to_set = dist_mat[:, pt_idx]  # Distance to the initial point
    for i in range(n_samples):
        selected_pts[i] = pt_idx
        dist_to_set = torch.minimum(dist_to_set, dist_mat[:, pt_idx])
        pt_idx = torch.argmax(dist_to_set)  # Select the farthest point

    return selected_pts

def calculate_2d_projections(coordinates_3d, intrinsics):   # 
    """
    Args:
        coordinates_3d: [3, N]
        intrinsics: [3, 3]

    Returns:
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / (projected_coordinates[2, :]+1e-7)
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.round(np.clip(projected_coordinates,0,1e4)).astype(np.int32)

    return projected_coordinates

def image_coords_to_camera_space(depth_map, coords_2d, intrinsics):
    # coords_2d: y, x
    # depth_map: h, w, np, float32
    # Unpack intrinsic matrix
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    
    #print(coords_2d.shape, coords_2d, np.max(coords_2d[:,0]), np.max(coords_2d[:,1]))
    depths = depth_map[coords_2d[:,0], coords_2d[:,1]]
    x = (coords_2d[:,1] - cx) * depths / fx
    y = (coords_2d[:,0] - cy) * depths / fy
    z = depths
    
    pointcloud = np.stack((x, y, z), axis=-1)
    return pointcloud

def depth_map_to_pointcloud_torch(depth_map, mask, intrinsics):
    # depth_map: h,w

    H, W = depth_map.shape

    if mask is not None:
        depth_map[mask == 0] = -1

    # Create grid of pixel coordinates
    u, v = torch.meshgrid(torch.arange(W,device=depth_map.device), torch.arange(H,device=depth_map.device),indexing='xy')
    # Convert pixel coordinates to camera coordinates
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

    return pointcloud

def transform_pointcloud(pointcloud, transformation_matrix):
    
    # Append a column of ones to make homogeneous coordinates
    homogeneous_points = np.hstack((pointcloud, np.ones((pointcloud.shape[0], 1))))
    
    # Perform transformation
    transformed_points = np.dot(transformation_matrix, homogeneous_points.T).T
    
    # Divide by the last coordinate (homogeneous division)
    transformed_points = transformed_points[:, :3] / transformed_points[:, 3][:, np.newaxis]
    
    return transformed_points

def transform_pointcloud_torch(pointcloud, transformation_matrix):
    # Convert the pointcloud to a torch tensor
    pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
    
    # Append a column of ones to make homogeneous coordinates
    ones = torch.ones((pointcloud.shape[0], 1), dtype=torch.float32,device=pointcloud.device)
    homogeneous_points = torch.cat((pointcloud, ones), dim=1)
    
    # Perform transformation (matrix multiplication)
    transformed_points = torch.matmul(homogeneous_points, transformation_matrix.T)
    
    # Divide by the last coordinate (homogeneous division)
    transformed_points = transformed_points[:, :3] / transformed_points[:, 3].unsqueeze(1)
    
    return transformed_points

def select_keypoint(frame):
    def select_point_for_image(image):
        def select_point_cv2(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append(np.array([x, y]))
                cv2.circle(img_visual, (x, y), 5, (0, 0, 255), -1)  # 绘制红色点
                cv2.imshow("Image", img_visual)
        points = []
        img_visual = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        cv2.imshow("Image", img_visual)
        cv2.setMouseCallback("Image", select_point_cv2)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # 必须按Esc退出，否则会卡住
                break
        cv2.destroyAllWindows()
        return points[0]
    keypoint_2d = select_point_for_image(frame['rgb'].cpu().numpy())    # 2
    keypoint = image_coords_to_camera_space(frame['depth'].cpu().numpy(),keypoint_2d[None,[1,0]],frame['intrinsics'])     # 1,3
    return keypoint

def quaternion_to_matrix_np(quaternion):
    """
    将四元数转换为旋转矩阵。
    
    参数:
        quaternion (numpy.ndarray): 四元数，形状为 (4, )，格式为 [w, x, y, z]，实部在前。
    
    返回:
        numpy.ndarray: 旋转矩阵，形状为 (3, 3)。
    """
    # 确保四元数是 numpy 数组
    quaternion = np.array(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError("四元数的形状必须为 (4,)")

    w, x, y, z = quaternion

    # 归一化四元数
    norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # 计算旋转矩阵
    rotation_matrix = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
    ])
    
    return rotation_matrix

def quaternion_to_matrix_torch(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def rotation_to_quaternion_np(rotation_matrix):
    """
    将旋转矩阵转换为四元数。
    
    参数:
        rotation_matrix (numpy.ndarray): 旋转矩阵，形状为 (3, 3)。
    
    返回:
        numpy.ndarray: 四元数，格式为 [w, x, y, z]，实部在前。
    """
    # 确保旋转矩阵是 numpy 数组
    rotation_matrix = np.array(rotation_matrix, dtype=np.float64)
    if rotation_matrix.shape != (3, 3):
        raise ValueError("旋转矩阵的形状必须为 (3, 3)")

    # 提取旋转矩阵的元素
    R = rotation_matrix
    trace = np.trace(R)

    if trace > 0:
        # 如果 trace > 0
        S = 2.0 * np.sqrt(1.0 + trace)
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        # 如果 R[0, 0] 是最大的对角元素
        S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        # 如果 R[1, 1] 是最大的对角元素
        S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        # 如果 R[2, 2] 是最大的对角元素
        S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    # 返回四元数
    quaternion = np.array([w, x, y, z])
    return quaternion


def compute_rot_distance(R1,R2):
    if R1.shape == (4,4):
        R1 = R1[:3,:3]
    if R2.shape == (4,4):
        R2 = R2[:3,:3]
    # 计算相对旋转矩阵
    R_relative = np.dot(np.linalg.inv(R1), R2)

    # 计算旋转角度
    angle = np.arccos(np.clip((np.trace(R_relative) - 1) / 2, -1.0, 1.0))

    # 将角度从弧度转换为度数
    angle_degrees = np.degrees(angle)
    
    return angle_degrees



import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import UnivariateSpline


def slerp_interpolation(rotations, t, t_new):
    """实现旋转的 SLERP 插值。
    rotations: Rotation 对象的列表（原始旋转）
    t: 原始时间点
    t_new: 新的时间点
    """
    quaternions = np.array([rot.as_quat() for rot in rotations])  # 提取四元数
    interpolated_quaternions = []

    for new_t in t_new:
        # 找到 new_t 在 t 中的范围
        i = np.searchsorted(t, new_t) - 1
        i = max(0, min(i, len(t) - 2))  # 确保索引合法
        t1, t2 = t[i], t[i + 1]
        q1, q2 = quaternions[i], quaternions[i + 1]

        # 计算插值比例
        alpha = (new_t - t1) / (t2 - t1)

        # 进行 SLERP 插值
        dot = np.dot(q1, q2)
        if dot < 0.0:  # 处理四元数符号不一致的问题
            q2 = -q2
            dot = -dot

        # 如果两点接近，直接线性插值
        if dot > 0.9995:
            interpolated_quat = q1 + alpha * (q2 - q1)
            interpolated_quat /= np.linalg.norm(interpolated_quat)
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)

            theta = alpha * theta_0
            sin_theta = np.sin(theta)

            s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
            s2 = sin_theta / sin_theta_0

            interpolated_quat = (s1 * q1) + (s2 * q2)

        interpolated_quaternions.append(interpolated_quat)

    return R.from_quat(interpolated_quaternions)

def smooth_quaternion_fitting(rotations, t, t_new, smooth_factor=0):
    """
    对四元数轨迹进行平滑拟合。
    rotations: Rotation 对象的列表
    t: 原始时间点
    t_new: 新的时间点
    smooth_factor: 平滑因子（越大越平滑）
    """
    quaternions = np.array([rot.as_quat() for rot in rotations])  # 提取四元数

    # 对每个分量单独拟合平滑样条
    splines = [UnivariateSpline(t, quaternions[:, i], s=smooth_factor) for i in range(4)]
    smooth_quaternions = np.stack([spl(t_new) for spl in splines], axis=-1)

    # 确保拟合结果归一化
    smooth_quaternions /= np.linalg.norm(smooth_quaternions, axis=1, keepdims=True)

    return R.from_quat(smooth_quaternions)

def smooth_trajectory_fitting_with_smoothness(waypoint_list, num_points=500, t_smooth_factor=0.02, r_smooth_factor=0.02):
    
    # s=0：强制通过所有点（类似 CubicSpline）。
    # s > 0：允许一定误差，平滑度越大。
    
    
    # 提取位置和旋转部分
    positions = np.array([wp[:3, 3] for wp in waypoint_list])  # 平移部分 (N x 3)
    rotations = [R.from_matrix(wp[:3, :3]) for wp in waypoint_list]  # 提取旋转部分

    # 对平移部分拟合平滑样条曲线
    t = np.linspace(0, 1, len(waypoint_list))  # 参数化时间
    t_new = np.linspace(0, 1, num_points)  # 新时间点

    spline_x = UnivariateSpline(t, positions[:, 0], s=t_smooth_factor)
    spline_y = UnivariateSpline(t, positions[:, 1], s=t_smooth_factor)
    spline_z = UnivariateSpline(t, positions[:, 2], s=t_smooth_factor)

    smooth_positions = np.stack([spline_x(t_new), spline_y(t_new), spline_z(t_new)], axis=1)

    # 对旋转部分进行 SLERP 插值
    smooth_rotations = smooth_quaternion_fitting(rotations, t, t_new,smooth_factor=r_smooth_factor).as_matrix()

    # 重组平滑轨迹
    smooth_trajectory = []
    for pos, rot in zip(smooth_positions, smooth_rotations):
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = pos
        smooth_trajectory.append(transform)

    return smooth_trajectory

def smooth_trajectory_with_initial_position(waypoint_list, initial_transform, num_points=500, alpha_func=None):
    """
    平滑地生成轨迹，同时全局考虑初始位置和旋转的影响。
    
    参数:
        waypoint_list: 目标轨迹的关键点列表，每个关键点是 4x4 的变换矩阵。
        initial_transform: 当前机械臂的初始变换矩阵 (4x4)。
        num_points: 生成的平滑轨迹点数量。
        alpha_func: 权重插值函数，接受参数 t (范围 [0, 1])，返回范围 [0, 1] 的权重值。
                    默认使用线性权重衰减。
    
    返回:
        smooth_trajectory: 平滑轨迹，包含一系列 4x4 的变换矩阵。
    """
    import numpy as np
    from scipy.interpolate import UnivariateSpline
    from scipy.spatial.transform import Rotation as R

    # 默认的线性权重衰减函数
    if alpha_func is None:
        alpha_func = lambda t: max((1-1.5*t),0)  # t=0 时权重为 1，t=1 时权重为 0

    # 提取初始位置和旋转
    initial_position = initial_transform[:3, 3]
    initial_rotation = R.from_matrix(initial_transform[:3, :3])

    # 提取轨迹的平移和旋转部分
    original_positions = np.array([wp[:3, 3] for wp in waypoint_list])
    original_rotations = [R.from_matrix(wp[:3, :3]) for wp in waypoint_list]

    # 平移整个轨迹，使其起点与初始位置一致
    adjusted_positions = original_positions + (initial_position - original_positions[0])
    adjusted_rotations = [
        initial_rotation * (rot * original_rotations[0].inv()) for rot in original_rotations
    ]  # 旋转调整基于初始旋转

    # 对平移部分进行样条插值
    t = np.linspace(0, 1, len(waypoint_list))  # 原始时间点
    t_new = np.linspace(0, 1, num_points)  # 新的时间点

    # 原始轨迹的样条拟合
    spline_x_original = UnivariateSpline(t, original_positions[:, 0], s=0)
    spline_y_original = UnivariateSpline(t, original_positions[:, 1], s=0)
    spline_z_original = UnivariateSpline(t, original_positions[:, 2], s=0)

    # 调整后的轨迹样条拟合
    spline_x_adjusted = UnivariateSpline(t, adjusted_positions[:, 0], s=0)
    spline_y_adjusted = UnivariateSpline(t, adjusted_positions[:, 1], s=0)
    spline_z_adjusted = UnivariateSpline(t, adjusted_positions[:, 2], s=0)

    smooth_positions = []
    for t_i in t_new:
        alpha = alpha_func(t_i)
        pos_original = np.array(
            [spline_x_original(t_i), spline_y_original(t_i), spline_z_original(t_i)]
        )
        pos_adjusted = np.array(
            [spline_x_adjusted(t_i), spline_y_adjusted(t_i), spline_z_adjusted(t_i)]
        )
        smooth_positions.append(alpha * pos_adjusted + (1 - alpha) * pos_original)

    smooth_positions = np.array(smooth_positions)

    # 对旋转部分进行插值
    smooth_rotations_original = smooth_quaternion_fitting(original_rotations, t, t_new)
    smooth_rotations_adjusted = smooth_quaternion_fitting(adjusted_rotations, t, t_new)

    smooth_rotations = []
    for i in range(len(t_new)):
        alpha = alpha_func(t_new[i])
        rot_original = smooth_rotations_original[i]
        rot_adjusted = smooth_rotations_adjusted[i]

        # 球面插值 (SLERP-like)
        smooth_rotation = R.from_quat(
            alpha * rot_adjusted.as_quat() + (1 - alpha) * rot_original.as_quat()
        )
        smooth_rotations.append(smooth_rotation)

    smooth_rotations = [rot.as_matrix() for rot in smooth_rotations]


    # 重组平滑轨迹
    smooth_trajectory = []
    for pos, rot in zip(smooth_positions, smooth_rotations):
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = pos
        smooth_trajectory.append(transform)
        
    smooth_trajectory = smooth_trajectory_fitting_with_smoothness(smooth_trajectory,len(smooth_trajectory),0.05,0.05)

    return smooth_trajectory

def is_significant_transform(mat1, mat2, translation_thresh, rotation_thresh,large_rotation_thresh=90):
    # 平移部分的欧几里得距离
    translation_diff = np.linalg.norm(mat1[:3, 3] - mat2[:3, 3])
    
    # 旋转部分的角度差（通过旋转矩阵）
    rotation_diff_matrix = np.dot(mat1[:3, :3].T, mat2[:3, :3])  # 相对旋转矩阵
    rotation_angle = np.arccos(np.clip((np.trace(rotation_diff_matrix) - 1) / 2, -1.0, 1.0))
    
    if rotation_angle > np.radians(large_rotation_thresh):
        return False
    
    return translation_diff > translation_thresh or rotation_angle > rotation_thresh

def filter_waypoints(waypoint_list, translation_thresh=0.002, rotation_thresh=0.002,large_rotation_thresh=90):
    filtered_waypoints = [waypoint_list.pop(0)]  # 保留第一个waypoint
    
    while len(waypoint_list) > 0:   
        if is_significant_transform(filtered_waypoints[-1], waypoint_list[0], translation_thresh, rotation_thresh,large_rotation_thresh):
            filtered_waypoints.append(waypoint_list.pop(0))
        else:
            waypoint_list.pop(0)

    return filtered_waypoints

def quaternion_apply(quaternion, point):
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]

def quaternion_invert(quaternion):
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    return quaternion * quaternion.new_tensor([1, -1, -1, -1])

def quaternion_raw_multiply(a, b):
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_invert(quaternion):
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    return quaternion * quaternion.new_tensor([1, -1, -1, -1])


def remove_outliers_with_open3d(pointcloud_np, nb_neighbors=16, std_ratio=2.0):
    """
    使用 Open3D 的统计滤波法移除点云中的离群点。
    
    参数:
    - pointcloud_np: numpy.ndarray, 点云数据，形状为 (N, 3)。
    - nb_neighbors: int, 每个点考虑的邻居数。
    - std_ratio: float, 离群点的标准差阈值。

    返回:
    - 过滤后的点云 (numpy.ndarray)。
    """
    # 将 numpy 点云转换为 Open3D 点云
    pointcloud_o3d = o3d.geometry.PointCloud()
    pointcloud_o3d.points = o3d.utility.Vector3dVector(pointcloud_np)

    # 使用统计滤波法
    cl, ind = pointcloud_o3d.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)


    return ind



def interpolate_transform(gripper2base_now, gripper2base_target, num_points):
    # 分解旋转和平移
    R_now = gripper2base_now[:3, :3]
    t_now = gripper2base_now[:3, 3]
    
    R_target = gripper2base_target[:3, :3]
    t_target = gripper2base_target[:3, 3]
    
    # 转换旋转矩阵到四元数
    rotations = R.from_matrix([R_now, R_target])

    # 创建 Slerp 对象
    slerp = Slerp([0, 1], rotations)

    # 插值参数 alpha
    alphas = np.linspace(0, 1, num_points)

    # 插值旋转
    interpolated_rotations = slerp(alphas).as_matrix()

    # 插值平移
    interpolated_translations = [
        (1 - alpha) * t_now + alpha * t_target for alpha in alphas
    ]

    # 合成变换矩阵
    trajectory = []
    for R_interp, t_interp in zip(interpolated_rotations, interpolated_translations):
        T_interp = np.eye(4)
        T_interp[:3, :3] = R_interp
        T_interp[:3, 3] = t_interp
        trajectory.append(T_interp)

    return trajectory