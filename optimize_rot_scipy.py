import numpy as np
from scipy.optimize import minimize, approx_fprime
from scipy.spatial.transform import Rotation as R
from utils import quaternion_to_matrix_np

# 生成示例点云
def generate_example_point_clouds():
    np.random.seed(42)
    source = np.random.rand(100, 3)  # 源点云
    true_rotation = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
    true_translation = np.array([1, -2, 3])
    true_deformation = 0.1 * np.random.randn(*source.shape)  # 轻微的变形

    target = source @ true_rotation.T + true_translation + true_deformation
    return source, target

source, target = generate_example_point_clouds()

def quaternion_constraint(params):
    quat = params[:4]
    return np.sum(quat**2) - 1  # 等式约束，模长为1

constraints = [{'type': 'eq', 'fun': quaternion_constraint}]

# Cauchy 损失
def cauchy_loss(residuals, c=1.0):
    # residuals: n
    loss = np.log(1 + (residuals / c) ** 2)
    return np.mean(loss)

# 构造优化目标函数
def loss_function(params,source, target,sigma ):

    # sigma = max(0.001, initial_sigma * (0.9 ** iteration_count))  # 基于迭代次数调整 sigma

    num_points = source.shape[0]
    scales = params[:3]
    # 提取旋转参数
    q_w, q_x, q_y, q_z = params[3:7]  # 旋转向量 (axis-angle 表示)
    affine_rotation_matrix = quaternion_to_rotation_matrix(np.array([q_w, q_x, q_y, q_z]))@ np.diag(scales)

    # 提取平移
    translation = params[7:10]
    # 提取变形场
    # deformation = params[10:].reshape(num_points, 3)     # n,3
    # 应用变换
    transformed_source = source @ affine_rotation_matrix.T + translation #+ deformation # n,3
    # 计算误差
    distance_residuals = cauchy_loss(np.linalg.norm(target-transformed_source,axis=-1))

    # 平滑变形场误差
    source_neighbor_id = np.argsort(np.linalg.norm(source[:,None,:]-source[None,:,:],axis=-1),axis=-1)[:,:8] # n,n -> n,8
    # smoothness_residuals = cauchy_loss(np.linalg.norm(deformation[:,None,:] - deformation[source_neighbor_id,:],axis=-1).reshape(-1))  # n,8,3 -> n,8 -> n*8

    # print('distance_residuals:',distance_residuals)
    # print('smoothness_residuals:',smoothness_residuals)

    return distance_residuals #+ sigma*smoothness_residuals

def run_optimize(source, target, init_scales, init_quaternion, init_translation):
    init_deformation = np.zeros_like(source)  # 初始变形场
    init_params = np.hstack([init_scales, init_quaternion, init_translation,])# init_deformation.ravel()])

    # 优化
    result = minimize(
        fun=loss_function,
        x0=init_params,
        args=(source, target,1),
        method='L-BFGS-B',
        jac='2-point',
        # constraints=constraints,
        options={'disp': True,
                }
    )

    # 提取优化结果
    optimized_params = result.x
    optimized_scale = optimized_params[:3]
    optimized_q = optimized_params[3:7]
    optimized_translation = optimized_params[7:10]
    # optimized_deformation = optimized_params[10:].reshape(source.shape)

    print('optimized_scale:',optimized_scale)
    print('optimized_q:',optimized_q)
    print('optimized_translation:',optimized_translation)

    return optimized_q


# def affine_alignment(x: np.ndarray, y: np.ndarray, with_scale='affine'):
#     """
#     Computes the weighted least squares solution parameters for scaling,
#     rotation, and translation between two point sets.

#     :param x: mxn matrix of source points (m = dimension, n = number of points)
#     :param y: mxn matrix of target points (same shape as x)
#     :param weights: array of weights for each point pair (n elements)
#     :param with_scale: set to True to align also the scale (default: False)
#     :return: scales, rotation matrix, translation vector, and transform matrix
#     """
#     if x.shape != y.shape:
#         raise ValueError("Source and target point clouds must have the same shape.")

#     # m = dimension, n = nr. of data points
#     m, n = x.shape
    
#     # Weighted means
#     mean_x = x.mean(axis=1)
#     mean_y = y.mean(axis=1)

#     # Centered points
#     x_centered = x - mean_x[:, np.newaxis]
#     y_centered = y - mean_y[:, np.newaxis]

#     # Weighted covariance matrix
#     cov_xy = (y_centered) @ x_centered.T

#     # SVD
#     u, d, v = np.linalg.svd(cov_xy)

#     # Ensure a right-handed coordinate system
#     s_matrix = np.eye(x.shape[0])
#     if np.linalg.det(u) * np.linalg.det(v) < 0.0:
#         s_matrix[-1, -1] = -1

#     # Rotation matrix
#     r = u @ s_matrix @ v

#     # Independent scale factors
#     if with_scale == 'affine':
#         scales = d / (((x_centered**2)).sum(axis=1))
#     elif with_scale == 'sim':
#         var_x = ((x_centered**2).sum(axis=0)).sum()
#         scale = np.trace(np.diag(d) @ s_matrix) / var_x
#         scales = np.array([scale,scale,scale])
#     elif with_scale == 'transform':
#         scales = np.ones(x.shape[0])

#     # Translation vector
#     t = mean_y - r @ (scales * mean_x)

#     # Transformation matrix
#     transform = np.eye(x.shape[0] + 1)
#     transform[:x.shape[0], :x.shape[0]] = r @ np.diag(scales)
#     transform[:x.shape[0], -1] = t

#     return scales, r, t, transform

def affine_alignment(x: np.ndarray, y: np.ndarray,with_scale='affine'):
    """
    Computes the weighted least squares solution parameters for scaling,
    rotation, and translation between two point sets.

    :param x: mxn matrix of source points (m = dimension, n = number of points)
    :param y: mxn matrix of target points (same shape as x)
    :param weights: array of weights for each point pair (n elements)
    :param with_scale: set to True to align also the scale (default: False)
    :return: scales, rotation matrix, translation vector, and transform matrix
    """

    weights = np.ones((x.shape[1]))

    if x.shape != y.shape:
        raise ValueError("Source and target point clouds must have the same shape.")
    if weights.shape[0] != x.shape[1]:
        raise ValueError("Number of weights must match the number of points.")

    # Normalize weights
    weights = weights / weights.sum()
    
    # Weighted means
    mean_x = (x * weights).sum(axis=1)
    mean_y = (y * weights).sum(axis=1)

    # Centered points
    x_centered = x - mean_x[:, np.newaxis]
    y_centered = y - mean_y[:, np.newaxis]

    # Weighted covariance matrix
    cov_xy = (y_centered * weights) @ x_centered.T

    # SVD
    u, d, v = np.linalg.svd(cov_xy)

    # Ensure a right-handed coordinate system
    s_matrix = np.eye(x.shape[0])
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        s_matrix[-1, -1] = -1

    # Rotation matrix
    r = u @ s_matrix @ v

    # Independent scale factors
    if with_scale == 'affine':
        scales = d / ((weights * (x_centered**2)).sum(axis=1))
    elif with_scale == 'sim':
        var_x = (weights * (x_centered**2).sum(axis=0)).sum()
        scale = np.trace(np.diag(d) @ s_matrix) / var_x
        scales = np.array([scale,scale,scale])
    elif with_scale == 'transform':
        scales = np.ones(x.shape[0])

    # Translation vector
    t = mean_y - r @ (scales * mean_x)

    # Transformation matrix
    transform = np.eye(x.shape[0] + 1)
    transform[:x.shape[0], :x.shape[0]] = r @ np.diag(scales)
    transform[:x.shape[0], -1] = t

    return scales, r, t, transform