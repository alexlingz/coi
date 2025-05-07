import numpy as np
import torch
import cv2
from utils import image_coords_to_camera_space, select_keypoint

def getRANSACInliers(source, target, MaxIterations=20000, PassThreshold=0.02, StopThreshold=0.001):
    # input n,3
    SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
    TargetHom = np.transpose(np.hstack([target, np.ones([source.shape[0], 1])]))

    BestResidual = 1e10
    BestInlierRatio = 0
    BestInlierIdx = np.arange(SourceHom.shape[1])
    BestScale = 0
    BestSimTransform = None
    for i in range(0, MaxIterations):
        # Pick 5 random (but corresponding) points from source and target
        RandIdx = np.random.randint(SourceHom.shape[1], size=5)

        scale, rotation, translation, SimTransform = umeyama_alignment(SourceHom[:3, RandIdx], TargetHom[:3, RandIdx])


        Residual, InlierRatio, InlierIdx = evaluateModel(SimTransform, SourceHom, TargetHom, PassThreshold, scale)
        # if Residual < BestResidual:     
        if InlierRatio > BestInlierRatio:   # 考虑inlier数量试试？
            # BestResidual = Residual
            BestInlierRatio = InlierRatio
            BestInlierIdx = InlierIdx
            BestScale = scale
            # BestSimTransform = {'s':scale, 'r':rotation, 't':translation, 'srt':SimTransform} 
            BestSimTransform = SimTransform
        # if BestResidual < StopThreshold:
        #     print(f'break in iter {i}')
        #     break

        # print('Iteration: ', i)
        # print('Residual: ', Residual)
        # print('Inlier ratio: ', InlierRatio)

    # return SourceHom[:, BestInlierIdx], TargetHom[:, BestInlierIdx], BestInlierRatio
    print('scale:', BestScale)
    print('BestInlierRatio:', BestInlierRatio)

    # s2t
    return source[BestInlierIdx], target[BestInlierIdx], BestSimTransform

def evaluateModel(OutTransform, SourceHom, TargetHom, PassThreshold, scale):
    Diff_s2t = TargetHom - np.matmul(OutTransform, SourceHom)   # s2t
    ResidualVec_s2t = np.linalg.norm(Diff_s2t[:3, :], axis=0)

    try:
        Diff_t2s = SourceHom - np.matmul(np.linalg.inv(OutTransform), TargetHom)
        ResidualVec_t2s = np.linalg.norm(Diff_t2s[:3, :], axis=0)
    except:
        ResidualVec_t2s = np.ones_like(ResidualVec_s2t)

    InlierIdx = np.where((ResidualVec_s2t < PassThreshold*scale) * (ResidualVec_t2s<PassThreshold/scale) )      # todo 这两个scale应该不一样

    Residual = np.linalg.norm(ResidualVec_s2t)  
    nInliers = np.count_nonzero(InlierIdx)
    InlierRatio = nInliers / SourceHom.shape[1]
    return Residual, InlierRatio, InlierIdx[0]

# def estimateSimilarityUmeyama(SourceHom, TargetHom):
#     # Copy of original paper is at: http://web.stanford.edu/class/cs273/refs/umeyama.pdf
#     SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
#     TargetCentroid = np.mean(TargetHom[:3, :], axis=1)
#     nPoints = SourceHom.shape[1]

#     CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
#     CenteredTarget = TargetHom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()

#     CovMatrix = np.matmul(CenteredTarget, np.transpose(CenteredSource)) / nPoints

#     if np.isnan(CovMatrix).any():
#         print('nPoints:', nPoints)
#         print(SourceHom.shape)
#         print(TargetHom.shape)
#         raise RuntimeError('There are NANs in the input.')

#     U, D, Vh = np.linalg.svd(CovMatrix, full_matrices=True)
#     d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
#     if d:
#         D[-1] = -D[-1]
#         U[:, -1] = -U[:, -1]

#     Rotation = np.matmul(U, Vh).T # Transpose is the one that works

#     varP = np.var(SourceHom[:3, :], axis=1).sum()
#     ScaleFact = 1/varP * np.sum(D) # scale factor
#     Scales = np.array([ScaleFact, ScaleFact, ScaleFact])
#     ScaleMatrix = np.diag(Scales)

#     Translation = TargetHom[:3, :].mean(axis=1) - SourceHom[:3, :].mean(axis=1).dot(ScaleFact*Rotation)

#     OutTransform = np.identity(4)
#     OutTransform[:3, :3] = ScaleMatrix @ Rotation
#     OutTransform[:3, 3] = Translation

#     # # Check
#     # Diff = TargetHom - np.matmul(OutTransform, SourceHom)
#     # Residual = np.linalg.norm(Diff[:3, :], axis=0)
#     return Scales, Rotation, Translation, OutTransform

def umeyama_alignment(x: np.ndarray, y: np.ndarray,
                      with_scale: bool = True):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        raise GeometryException("data matrices must have the same shape")

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)
    # if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:   # 通过ransac避免这个
    #     raise ValueError("Degenerate covariance rank, Umeyama alignment is not possible")


    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))


    OutTransform = np.eye(4)
    OutTransform[:3, :3] = c * r
    OutTransform[:3, 3] = t


    return c, r, t, OutTransform


class WeightedRansacAffine3D:
    def __init__(self,pass_threshold=0.01,max_iteration=10000,with_scale='affine'):
        # affine / sim / transfrom
        self.pass_threshold = pass_threshold
        self.max_iteration = max_iteration
        self.with_scale = with_scale


    def getRANSACInliers(self, source, target,weights):
        # input n,3
        SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
        TargetHom = np.transpose(np.hstack([target, np.ones([source.shape[0], 1])]))

        best_weighted_inlier_ratio = 0
        BestInlierIdx = np.arange(SourceHom.shape[1])
        best_affine_transform = None
        for i in range(0, self.max_iteration):
            # Pick 5 random (but corresponding) points from source and target
            RandIdx = np.random.randint(SourceHom.shape[1], size=5)
            
            try:
                scale, rotation, translation, affine_transform = self.weighted_affine_alignment(x=SourceHom[:3, RandIdx], 
                                                                                         y=TargetHom[:3, RandIdx],
                                                                                         weights=weights[RandIdx])
            except:
                print('estimate error,continue')
                continue


            weighted_inlier_ratio, InlierIdx = self.weighted_evaluateModel(affine_transform, SourceHom, TargetHom, scale,weights)
            # if Residual < BestResidual:     
            if weighted_inlier_ratio > best_weighted_inlier_ratio:   # 考虑weighted inlier

                best_weighted_inlier_ratio = weighted_inlier_ratio
                BestInlierIdx = InlierIdx
                best_affine_transform = affine_transform
                best_scale = scale

        _,_,_, best_rigid_transform = self.weighted_affine_alignment(x=SourceHom[:3, BestInlierIdx], 
                                                                                         y=TargetHom[:3, BestInlierIdx],
                                                                                         weights=weights[BestInlierIdx],final=True)

        # s2t
        print('best weighted inlier ratio:',best_weighted_inlier_ratio)
        print('inlier ratio:', len(BestInlierIdx)/len(weights))
        print('best_scale:',best_scale)
        return BestInlierIdx, best_affine_transform,best_rigid_transform


    def weighted_affine_alignment(self, x: np.ndarray, y: np.ndarray, weights=None, final=False):
        """
        Computes the weighted least squares solution parameters for scaling,
        rotation, and translation between two point sets.

        :param x: mxn matrix of source points (m = dimension, n = number of points)
        :param y: mxn matrix of target points (same shape as x)
        :param weights: array of weights for each point pair (n elements)
        :param with_scale: set to True to align also the scale (default: False)
        :return: scales, rotation matrix, translation vector, and transform matrix
        """
        if x.shape != y.shape:
            raise ValueError("Source and target point clouds must have the same shape.")
        # if weights.shape[0] != x.shape[1]:
        #     raise ValueError("Number of weights must match the number of points.")
        if weights is None:
            weights = np.ones(x.shape[1])

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
        if self.with_scale == 'affine':
            scales = d / ((weights * (x_centered**2)).sum(axis=1))
        elif self.with_scale == 'sim':
            var_x = (weights * (x_centered**2).sum(axis=0)).sum()
            scale = np.trace(np.diag(d) @ s_matrix) / var_x
            scales = np.array([scale,scale,scale])
        elif self.with_scale == 'transform':
            scales = np.ones(x.shape[0])

        if final:
            scales = np.ones(x.shape[0])

        # Translation vector
        t = mean_y - r @ (scales * mean_x)

        # Transformation matrix
        transform = np.eye(x.shape[0] + 1)
        transform[:x.shape[0], :x.shape[0]] = r @ np.diag(scales)
        transform[:x.shape[0], -1] = t

        return scales, r, t, transform

    def weighted_evaluateModel(self,affine_transform, SourceHom, TargetHom, scale, weight):

        # 还是以inlier数量作为最终选择的标准,不过统计inlier比例时加上权重

        Diff_s2t = TargetHom - np.matmul(affine_transform, SourceHom)   # s2t
        # ResidualVec_s2t = np.linalg.norm(Diff_s2t[:3, :], axis=0)  # 由于outlier非常多,所以要像umeyama一样限制不能通过极端的scale绕过阈值
        ResidualVec_s2t = np.abs(Diff_s2t[:3, :])   # 3,n

        try:
            Diff_t2s = SourceHom - np.matmul(np.linalg.inv(affine_transform), TargetHom)
            # ResidualVec_t2s = np.linalg.norm(Diff_t2s[:3, :], axis=0)
            ResidualVec_t2s = np.abs(Diff_t2s[:3, :])
        except:
            # print('cant compute inv of affine_transform')
            ResidualVec_t2s = np.ones_like(ResidualVec_s2t)

        # InlierIdx = np.where((ResidualVec_s2t < self.pass_threshold*scale) * (ResidualVec_t2s<self.pass_threshold/scale) )      # todo 这两个scale应该不一样
        InlierIdx = np.where((np.sum(np.abs(ResidualVec_s2t[:3, :])<(self.pass_threshold*scale)[:,None],axis=0) * np.sum(np.abs(ResidualVec_t2s[:3, :])<(self.pass_threshold/scale)[:,None],axis=0)) == 9)

        # Residual = np.linalg.norm(ResidualVec_s2t)  
        # nInliers = np.count_nonzero(InlierIdx)
        weighted_inlier_ratio =  weight[InlierIdx].sum()
        # InlierRatio = nInliers / SourceHom.shape[1]
        return weighted_inlier_ratio, InlierIdx[0]

