import numpy as np
from scipy.spatial.transform import Rotation as R
from utils import quaternion_to_matrix_torch,quaternion_to_matrix_np,transform_pointcloud_torch
import torch
from utils import rotation_to_quaternion_np, quaternion_apply,quaternion_to_matrix_np, smooth_trajectory_fitting_with_smoothness,quaternion_raw_multiply ,quaternion_invert
from torchmetrics.functional import pairwise_cosine_similarity

import plotly.graph_objects as go


class ResultRecorder:
    def __init__(self,init_source,init_target):
        self.source = init_source
        self.target = init_target
        self.quat_list = []
        self.transformed_source_list = []

    def record_result(self,quat,transformed_source):
        self.quat_list.append(quat)     # epoch,4
        self.transformed_source_list.append(transformed_source)     # epoch,n,3

    def vis_optimize_result(self,rot_only=True):

        if rot_only:
            roted_sources = np.stack([self.source@quaternion_to_matrix_np(self.quat_list[i]) for i in range(len(self.quat_list))])    # epoch,n,3
            points = np.concatenate([roted_sources-roted_sources.mean(axis=1,keepdims=True),np.repeat((self.target-self.target.mean(axis=0,keepdims=True))[None],repeats=len(self.transformed_source_list),axis=0)],axis=1)    # epoch,n+m,3
        else:
            points = np.concatenate([self.transformed_source_list,np.repeat(self.target[None],repeats=len(self.transformed_source_list),axis=0)],axis=1)    # epoch,n+m,3



        point_center = points.mean(axis=(0,1))
        cube_length = 1.0*np.std(points)
        print('cube length:',cube_length)

        color = np.ones_like(points[0])
        color[:self.source.shape[0]] = np.array([1,0,0])
        color[self.source.shape[0]:] = np.array([0,1,0])

        fig = go.Figure()
        # 添加初始数据点
        fig.add_trace(
            go.Scatter3d(
                x=points[0,:,0],
                y=points[0,:,1],
                z=points[0,:,2],
                mode='markers',
                marker=dict(size=4,
                            color=color,
                            opacity=0.4)
            )
        )
        # 更新数据并生成动画
        frames = []
        for i in range(len(points)):

            frame = go.Frame(
                data=[
                    go.Scatter3d(
                        x=points[i,:,0],
                        y=points[i,:,1],
                        z=points[i,:,2],
                        mode='markers',
                        marker=dict(size=4,
                                    color=color,
                                opacity=0.4)
                    )
                ],
                name=f'frame{i}'
            )
            frames.append(frame)
        fig.frames = frames
        # 设置布局
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[point_center[0]-cube_length/2, point_center[0]+cube_length/2]),
                yaxis=dict(range=[point_center[1]-cube_length/2, point_center[1]+cube_length/2]),
                zaxis=dict(range=[point_center[2]-cube_length/2, point_center[2]+cube_length/2]),
                aspectmode='cube'
            ),
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
            width = 800,
            height = 800
        )
        # 显示图形
        fig.show()

# Cauchy 损失
def cauchy_loss(residuals, c=0.1):
    # residuals: n
    loss = torch.log(1 + (residuals / c) ** 2)
    return torch.mean(loss)


# def run_optimize(landmark_source, landmark_target, init_scales, init_quaternion, init_translation,source_keypoint,max_iter=100):

#     landmark_source = torch.tensor(landmark_source,device='cuda',dtype=torch.float32)
#     landmark_target = torch.tensor(landmark_target,device='cuda',dtype=torch.float32)

#     result_recorder = ResultRecorder(landmark_source.cpu().numpy(),landmark_target.cpu().numpy())

#     init_scales = torch.tensor(init_scales,device='cuda',dtype=torch.float32).requires_grad_()
#     init_quaternion = torch.tensor(init_quaternion,device='cuda',dtype=torch.float32).requires_grad_()
#     init_translation = torch.tensor(init_translation,device='cuda',dtype=torch.float32).requires_grad_()
#     init_deformation = torch.zeros_like(landmark_source).requires_grad_()  # 初始变形场

    
#     optimizer = torch.optim.Adam([init_scales, init_quaternion,init_translation,init_deformation], lr=1e-3)
#     loss_last = 0
#     for i in range(max_iter):
#         optimizer.zero_grad()

#         affine_rotation_matrix = quaternion_to_matrix_torch(init_quaternion)@ torch.diag(init_scales)
#         # 应用变换
#         transformed_source = landmark_source @ affine_rotation_matrix.T + init_translation + init_deformation # n,3
#         # 计算误差
#         distance_residuals = cauchy_loss(torch.norm(landmark_target-transformed_source,dim=-1))

#         # 平滑变形场误差
#         source_neighbor_id = torch.argsort(torch.norm(source.unsqueeze(1)-source.unsqueeze(0),dim=-1),dim=-1)[:,:8] # n,n -> n,8
#         smoothness_residuals = cauchy_loss(torch.norm(init_deformation.unsqueeze(1) - init_deformation[source_neighbor_id,:],dim=-1).reshape(-1))  # n,8,3 -> n,8 -> n*8

#         quat_loss = (1 - torch.norm(init_quaternion)) ** 2

#         print('iteration:',i,'distance_residuals:',distance_residuals,'smoothness_residuals:',smoothness_residuals)

#         # loss = distance_residuals + 1e1*(0.98**i)*smoothness_residuals + 1e5*quat_loss
#         loss = distance_residuals + smoothness_residuals + 1e5*quat_loss

#         loss.backward()
#         optimizer.step()

#         result_recorder.record_result(init_quaternion.detach().cpu().numpy(),transformed_source.detach().cpu().numpy())

#         if loss-loss_last < 1e-4:
#             break

#     result_recorder.vis_optimize_result(rot_only=False)
#     result_recorder.vis_optimize_result(rot_only=True)
    
#     transformed_source_keypoint = torch.tensor(source_keypoint,device='cuda',dtype=torch.float32) @ affine_rotation_matrix.T + init_translation + init_deformation # n,3
#     nearest_target = target[torch.argmin(torch.norm(target.unsqueeze(1)-transformed_source_keypoint.unsqueeze(0),dim=-1),dim=0)]

#     return init_quaternion.detach().cpu().numpy(),nearest_target.detach().cpu().numpy()
import pickle
def run_optimize(source_point,source_feature,
                 target_point,target_feature,
                 rigid_mask,inlier,
                 scales,rotation,t):
    # rigid_mask:  num of match -> num of rigid  布尔
    # inlier: num of match -> num of inlier, 索引
    # rigid_mask[inlier] : num of inlier -> rigid in inlier 
    
    data_debug = {
        'source_point':source_point,
        'source_feature':source_feature,
        'target_point':target_point,
        'target_feature':target_feature,
        'rigid_mask':rigid_mask,
        'inlier':inlier,
        'scales':scales,
        'rotation':rotation,
        't':t
    }
    with open('optimize_debug.pkl','wb') as f:
        pickle.dump(data_debug,f)
    
    result_recorder = ResultRecorder(source_point[rigid_mask],target_point[rigid_mask])

    
    source_point,source_feature,target_point,target_feature,rigid_mask,inlier, = torch.tensor(source_point,device='cuda',dtype=torch.float32),torch.tensor(source_feature,device='cuda',dtype=torch.float32),torch.tensor(target_point,device='cuda',dtype=torch.float32),torch.tensor(target_feature,device='cuda',dtype=torch.float32),torch.tensor(rigid_mask,device='cuda',dtype=torch.bool),torch.tensor(inlier,device='cuda',dtype=torch.int32)    
    rigid_source_point = source_point[rigid_mask]
    rigid_target_point = target_point[rigid_mask]
    rigid_source_feature = source_feature[rigid_mask]
    rigid_target_feature = target_feature[rigid_mask]

    similarity_matrix = pairwise_cosine_similarity(rigid_source_feature,rigid_target_feature)# ni,ni1
    similarity_matrix = ((similarity_matrix+1)/2)**6
    def compute_feature_loss(cs2ct,rigid_source_point,rigid_target_point,deformation,similarity_matrix):
        # 与排序无关的特征损失
        transformed_rigid_source_point = transform_pointcloud_torch(rigid_source_point,cs2ct) + deformation
        
        distance_matrix = torch.norm(transformed_rigid_source_point.unsqueeze(1) - rigid_target_point.unsqueeze(0), dim=-1)    # ni,ni1
        dis_weight = (1/(distance_matrix+1e-3) )**3     # ni,ni1
        weight_similarity_matrix = dis_weight*similarity_matrix
        
        feature_loss = -weight_similarity_matrix.mean()
        return feature_loss
    
    def compute_landmark_loss(cs2ct,deformation_of_rigid_in_inlier,rigid_mask,inlier):
        # 与排序有关
        # deformation: num of rigid
        # landmark_loss 需要计算rigid in inlier的损失
        transformed_source_point = transform_pointcloud_torch(source_point,cs2ct) 
        landmark_loss = torch.norm(transformed_source_point[inlier][rigid_mask[inlier]] + deformation_of_rigid_in_inlier- target_point[inlier][rigid_mask[inlier]],dim=-1).mean()
        return landmark_loss
    
    def compute_smoothness_loss(rigid_source_point,deformation):
        
        source_neighbor_id = torch.argsort(torch.norm(rigid_source_point.unsqueeze(1)-rigid_source_point.unsqueeze(0),dim=-1),dim=-1)[:,:8] # n,n -> n,8
        smoothness_residuals = cauchy_loss(torch.norm(deformation.unsqueeze(1) - deformation[source_neighbor_id,:],dim=-1).reshape(-1))  # n,8,3 -> n,8 -> n*8
        return smoothness_residuals
    
    def compute_quat_loss(rotation):
        quat_loss = (1 - torch.norm(rotation)) ** 2
        return quat_loss

    def compute_icp_loss(cs2ct,rigid_source_point,rigid_target_point,deformation):
        transformed_rigid_source_point = transform_pointcloud_torch(rigid_source_point,cs2ct) + deformation
        distance_map = torch.norm(transformed_rigid_source_point.unsqueeze(1) - rigid_target_point.unsqueeze(0),dim=-1)
        distance_map = distance_map.min(dim=1).values
        distance_mask = distance_map < 0.01
        icp_loss = cauchy_loss(distance_map[distance_mask])
        return icp_loss
    
    
    scales,rotation,t = torch.tensor(scales,device='cuda',dtype=torch.float32).requires_grad_(),torch.tensor(rotation,device='cuda',dtype=torch.float32).requires_grad_(),torch.tensor(t,device='cuda',dtype=torch.float32).requires_grad_()
    deformation_of_match = torch.zeros_like(source_point,device='cuda',dtype=torch.float32).requires_grad_()
    
    optimizer = torch.optim.Adam([scales,rotation,t,deformation_of_match], lr=1e-3)
    for iteration in range(60):
        optimizer.zero_grad()
        print('scales:',scales)
        cs2ct = torch.eye(4,device='cuda',dtype=torch.float32)
        cs2ct[:3,:3] = rotation@ torch.diag(scales)
        cs2ct[:3,3] = t
        
        deformation_of_rigid = deformation_of_match[rigid_mask]
        deformation_of_rigid_in_inlier = deformation_of_match[inlier][rigid_mask[inlier]]
        
        feature_loss = 1e-6*compute_feature_loss(cs2ct,rigid_source_point,rigid_target_point,deformation_of_rigid,similarity_matrix)
        icp_loss = 1e2*compute_icp_loss(cs2ct,rigid_source_point,rigid_target_point,deformation_of_rigid)
        landmark_loss =  1e2*compute_landmark_loss(cs2ct,deformation_of_rigid_in_inlier,rigid_mask,inlier)
        smoothness_loss = 1e2*compute_smoothness_loss(rigid_source_point,deformation_of_rigid)
        quat_loss = 1e7*compute_quat_loss(rotation)
        deformation_loss = 1e3*deformation_of_match.norm(dim=-1).mean()
        
        loss = feature_loss + landmark_loss + smoothness_loss + quat_loss + icp_loss + deformation_loss
        loss.backward()
        optimizer.step()
        
        print('iteration:',iteration,'total loss:',loss,'\nicp_loss:',icp_loss,'\nlandmark_loss:',landmark_loss,'\nsmoothness_loss:',smoothness_loss,'\nquat_loss:',quat_loss,'\ndeformation_of_match:',deformation_of_match.norm(dim=-1).mean(),'\nfeature_loss:',feature_loss)
        
        transformed_rigid_source_point = transform_pointcloud_torch(rigid_source_point,cs2ct) + deformation_of_rigid
        result_recorder.record_result(rotation_to_quaternion_np(rotation.detach().cpu().numpy()),transformed_rigid_source_point.detach().cpu().numpy())

    result_recorder.vis_optimize_result(rot_only=False)
        
    return scales.detach().cpu().numpy(),rotation.detach().cpu().numpy(),t.detach().cpu().numpy(),deformation_of_match.detach().cpu().numpy()

if __name__ == '__main__':
    with open('optimize_debug_flower.pkl','rb') as f:
        data = pickle.load(f)
    run_optimize(data['source_point'],data['source_feature'],
                 data['target_point'],data['target_feature'],
                 data['rigid_mask'],data['inlier'],
                 data['scales'],data['rotation'],data['t'])