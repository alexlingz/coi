import numpy as np
from utils import select_keypoint,calculate_2d_projections, image_coords_to_camera_space
import torch
import pickle
import requests
import cv2

# input seq
# output keypoint 1,3 for each frame
# 实际执行时：

def generate_keypoint(demo,feature_extracter=None,passive_obj=None,task=None,positive_obj=None,passvie_obj=None,contact_point_demo=None,save_dir=None,process_result={}):
    print('start processing demo!')
    # # 从第一帧确定passive的keypoint
    # positive_keypoint,passive_keypoint = use_nearest_as_keypoint(demo[0],feature_extracter=feature_extracter)
    # demo[0]['positive_keypoint'] = positive_keypoint
    # demo[0]['passive_keypoint'] = passive_keypoint
    # # for i in range(1,len(demo)):
    # #     positive_keypoint,passive_keypoint = use_nearest_as_keypoint(demo[i],passive_keypoint=demo[0]['passive_keypoint'],feature_extracter=feature_extracter)
    # #     demo[i]['positive_keypoint'] = positive_keypoint
    # #     demo[i]['passive_keypoint'] = passive_keypoint
    
    # 手动选择keypoint
    # positive_keypoint = manual_select_keypoint(demo[0])
    # demo[0]['positive_keypoint'] = positive_keypoint
    # if passive_obj is not None:
    #     passive_keypoint = manual_select_keypoint(demo[0])
    #     demo[0]['passive_keypoint'] = passive_keypoint
    # else:
    #     # 如果是铰接，没有被动物体
    #     demo[0]['passive_keypoint'] = positive_keypoint
    
    # rekep出keypoint
    # rekep_keypoint(demo[0])
    


    # moka出keypoint
    if passive_obj is not None:
        positive_keypoint_3d,passive_keypoint_3d = moka_keypoint(demo[0],feature_extracter,task,positive_obj,passvie_obj,save_dir)
        demo[0]['positive_keypoint'] = positive_keypoint_3d
        demo[0]['passive_keypoint'] = passive_keypoint_3d
    else:
        demo[0]['positive_keypoint'] = contact_point_demo
        demo[0]['passive_keypoint'] = contact_point_demo

    
    np.save(f'{save_dir}/positive_keypoint.npy',demo[0]['positive_keypoint'])
    np.save(f'{save_dir}/passive_keypoint.npy',demo[0]['passive_keypoint'])
    return demo[0]['positive_keypoint'],demo[0]['passive_keypoint']
    

def use_nearest_as_keypoint(frame,passive_keypoint=None,feature_extracter=None):
    
    positive_point, _ = feature_extracter.extract(frame,'positive',sample_point_size=64)   # n,3
    positive_point = positive_point.cpu().numpy()
    if passive_keypoint is None:
        passive_keypoint, _ = feature_extracter.extract(frame,'passive',sample_point_size=64)
        passive_keypoint = passive_keypoint.cpu().numpy()   # m,3

    distance = np.linalg.norm(positive_point[:,None] - passive_keypoint[None],axis=-1)  # n,m
    
    # 找到最小值的扁平索引
    flat_index = np.argmin(distance)
    # 将扁平索引转换为二维索引
    row, col = np.unravel_index(flat_index, distance.shape)

    # 1,3  1,3
    return positive_point[row:row+1], passive_keypoint[col:col+1]

def manual_select_keypoint(frame):
    
    keypoint = select_keypoint(frame)
    # 1,3  1,3
    return keypoint

def rekep_keypoint(frame):
    
    depth_map = frame['depth'].clone()
    H,W = depth_map.shape
    intrinsics = frame['intrinsics']
    u, v = torch.meshgrid(torch.arange(W,device=depth_map.device), torch.arange(H,device=depth_map.device),indexing='xy')
    x = (u - intrinsics['cx']) * depth_map / intrinsics['fx']
    y = (v - intrinsics['cy']) * depth_map / intrinsics['fy']
    z = depth_map   # 
    pointcloud = torch.stack((x, y, z), dim=-1)
    
    
    masks = {
        0:frame['positive_mask'].cpu().numpy(),
        1:frame['passive_mask'].cpu().numpy()
    }

    data = {'rgb':frame['rgb'].cpu().numpy(),
            'points':pointcloud.cpu().numpy(),
            'masks':masks}
    serialized_data = pickle.dumps(data)
    
    response = requests.post('http://127.0.0.1:5004/keypoint', data=serialized_data)
    keypointresult = pickle.loads(response.content)
    # print(keypointresult)
    # cv2.imwrite('tmp_image/keypoint_img.png',cv2.cvtColor(keypointresult['projected_img'],cv2.COLOR_RGB2BGR))
    #  {'keypoints': keypoints, 'projected_img': projected_img}
    return keypointresult['keypoints']  # 12(pos6pas6),3 

def moka_keypoint(frame,feature_extracter,task,positive_obj,passvie_obj,save_dir):
    # 先最远点采样一些proposal
    # 最远点采样有问题，不应该这样做，
    positive_point_proposal_fps = feature_extracter.extract(frame,'positive',sample_point_size=6)[0].cpu().numpy()   # n,3
    passive_point_proposal_fps = feature_extracter.extract(frame,'passive',sample_point_size=6)[0].cpu().numpy()   # n,3
    
    # 基于feature聚类，用rekep提出proposal，看上去也没那么好
    keypointproposal = rekep_keypoint(frame)
    positive_point_proposal_fc = keypointproposal[:int(len(keypointproposal)/2)]  # n,3
    passive_point_proposal_fc = keypointproposal[int(len(keypointproposal)/2):]  # n,3
    
    positive_point_proposal = np.concatenate([positive_point_proposal_fps,positive_point_proposal_fc],axis=0)
    passive_point_proposal = np.concatenate([passive_point_proposal_fps,passive_point_proposal_fc],axis=0)
    
    positive_point_proposal = filter_close_points(positive_point_proposal, threshold=0.02)
    passive_point_proposal = filter_close_points(passive_point_proposal, threshold=0.02)
    
    np.save(f'{save_dir}/positive_keypoint_proposal.npy',positive_point_proposal)
    np.save(f'{save_dir}/passive_keypoint_proposal.npy',passive_point_proposal)

    positive_point_proposal_2d = calculate_2d_projections(positive_point_proposal.transpose(),np.array([[frame['intrinsics']['fx'],0,frame['intrinsics']['cx']],
                            [0,frame['intrinsics']['fy'],frame['intrinsics']['cy']],
                            [0,0,1]]))  # n,2
    positive_point_proposal_2d_mask = frame['positive_mask'][np.round(positive_point_proposal_2d[:,1]).astype(int),np.round(positive_point_proposal_2d[:,0].astype(int))]
    positive_point_proposal_2d = positive_point_proposal_2d[positive_point_proposal_2d_mask.cpu().numpy()]
    
    passive_point_proposal_2d = calculate_2d_projections(passive_point_proposal.transpose(),np.array([[frame['intrinsics']['fx'],0,frame['intrinsics']['cx']],
                            [0,frame['intrinsics']['fy'],frame['intrinsics']['cy']],
                            [0,0,1]]))  # n,2
    passive_point_proposal_2d_mask = frame['passive_mask'][np.round(passive_point_proposal_2d[:,1]).astype(int),np.round(passive_point_proposal_2d[:,0].astype(int))]
    passive_point_proposal_2d = passive_point_proposal_2d[passive_point_proposal_2d_mask.cpu().numpy()]
    
    positive_bbox = feature_extracter.get_2dbboxes(frame['positive_mask'])  # ymin,xmin,ymax,xmax, tensor
    passive_bbox = feature_extracter.get_2dbboxes(frame['passive_mask'])    # ymin,xmin,ymax,xmax, tensor
    merge_bbox = [torch.min(positive_bbox[0],passive_bbox[0]).tolist(),
                  torch.min(positive_bbox[1],passive_bbox[1]).tolist(),
                torch.max(positive_bbox[2],passive_bbox[2]).tolist(),
                torch.max(positive_bbox[3],passive_bbox[3]).tolist()]
    
    
    candidate_keypoints = {
        'active':positive_point_proposal_2d,
        'passive':passive_point_proposal_2d
    }
    data = {
        'task':{'instruction':task,
                'object_active':positive_obj,
                'object_passive':passvie_obj},
        'image':frame['rgb'].cpu().numpy(),
        'candidate_keypoints':candidate_keypoints,
        'bbox':merge_bbox
    }
    serialized_data = pickle.dumps(data)
    response = requests.post('http://127.0.0.1:5003/keypoint', data=serialized_data)
    result = pickle.loads(response.content)    
    positive_keypoint_2d,passive_keypoint_2d = result['active'],result['passive']   # n,3  n,3
    annotated_image = result['annotated_image']
    image_show = cv2.resize(np.array(annotated_image),(frame['rgb'].cpu().numpy().shape[1],frame['rgb'].cpu().numpy().shape[0]))

    for i in range(positive_keypoint_2d.shape[0]):
        cv2.circle(image_show, (positive_keypoint_2d[i][0],positive_keypoint_2d[i][1]), 5, (0, 255, 0), -1)  
    for i in range(passive_keypoint_2d.shape[0]):
        cv2.circle(image_show, (passive_keypoint_2d[i][0],passive_keypoint_2d[i][1]), 5, (0, 255, 0), -1)  
    cv2.imwrite(f"{save_dir}/selected_keypoint.png", cv2.cvtColor(image_show,cv2.COLOR_RGB2BGR))

    # 存在问题，可能会被depth噪声点干扰
    positive_keypoint_3d = image_coords_to_camera_space(frame['depth'].cpu().numpy(),positive_keypoint_2d[:,[1,0]],frame['intrinsics']).mean(axis=0,keepdims=True)
    passive_keypoint_3d = image_coords_to_camera_space(frame['depth'].cpu().numpy(),passive_keypoint_2d[:,[1,0]],frame['intrinsics']).mean(axis=0,keepdims=True)

    return positive_keypoint_3d,passive_keypoint_3d

def filter_close_points(points, threshold=0.02):
    """
    过滤掉相邻距离低于阈值的点，优先保留靠前的点。

    参数：
    - points: (n, 3) 的 NumPy 数组，每行表示一个点。
    - threshold: 两点之间的最小距离，小于此距离的点将被移除。

    返回：
    - 过滤后的点集，仍为 (n', 3) 的 NumPy 数组。
    """
    # 存储保留的点
    filtered_points = []

    # 按顺序检查点
    for point in points:
        if len(filtered_points) == 0:
            # 如果是第一个点，直接添加
            filtered_points.append(point)
        else:
            # 检查当前点与已保留点之间的最小距离
            distances = np.linalg.norm(np.array(filtered_points) - point, axis=1)
            if np.all(distances >= threshold):
                # 如果当前点与所有保留点之间距离大于阈值，则添加
                filtered_points.append(point)

    return np.array(filtered_points)