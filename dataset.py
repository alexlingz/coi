from mask_predictor import MaskPredictor
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import json
import numpy as np
import torch
import imageio
from IPython.display import HTML
from scipy.ndimage import binary_erosion
# import depth_pro
from utils import select_keypoint


# class RopeData:
#     def __init__(self): 

#         self.path = '/home/wts/data/omni6dpose/rope'
#         self.device = 'cuda'
#         self.mask_predictor = MaskPredictor()

#     def get_frame(self, scene_id, frame_id, mask_mode='img'):     
#         scene_id = str(scene_id).rjust(6,'0')
#         frame_id = str(frame_id).rjust(6,'0')
        
#         # mask_mode: img or seq
#         color = cv2.cvtColor(cv2.imread(f'{self.path}/{scene_id}/color/{frame_id}_color.png'),cv2.COLOR_BGR2RGB)
#         depth = cv2.imread(f'{self.path}/{scene_id}/depth/{frame_id}_depth.exr',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)



#         if mask_mode=='img':
#             mask = self.mask_predictor.predict_image(color)
#             mask = torch.tensor(mask,device=self.device)
#         elif mask_mode=='seq':
#             mask = self.mask_predictor.sequence_predictor_track(color)
#             mask = torch.tensor(mask,device=self.device)
#         else:
#             mask = None
        
#         with open(f'{self.path}/{scene_id}/meta/{frame_id}_meta.json') as f:
#             intrinsics_origin = json.load(f)['camera']['intrinsics']
#             if intrinsics_origin['height'] == depth.shape[0]:
#                 intrinsics = intrinsics_origin
#             else:
#                 intrinsics = {}
#                 scale = depth.shape[0] / intrinsics_origin['height']
#                 intrinsics['fx'] = scale * intrinsics_origin['fx']
#                 intrinsics['fy'] = scale * intrinsics_origin['fy']
#                 intrinsics['cx'] = scale * intrinsics_origin['cx']
#                 intrinsics['cy'] = scale * intrinsics_origin['cy']

#         frame = {
#             'rgb':torch.tensor(color,device=self.device),   # h,w,3, torch.uint8, 0~255
#             'depth':torch.tensor(depth,device=self.device), # h,w, torch.float32
#             'mask':mask,   # h,w, torch.bool
#             'intrinsics':intrinsics
#         }
#         return frame
    
#     def get_sequence(self,scene_id,start_id,length=30):
#         frames = []
#         for i in range(start_id,start_id+length):
            
#             if i==start_id:
#                 scene_idx = str(scene_id).rjust(6,'0')
#                 idx = str(i).rjust(6,'0')
#                 # 初始化mask tracker
#                 self.mask_predictor.sequence_predictor_initialize(
#                     cv2.cvtColor(cv2.imread(f'{self.path}/{scene_idx}/color/{idx}_color.png'),cv2.COLOR_BGR2RGB)
#                 )

#             frame = self.get_frame(scene_id,i,mask_mode='seq')

#             frames.append(frame)
#         return frames
    


class D415Data:
    def __init__(self, estimate_depth=False,positive_obj=None,passive_obj=None):
        self.max_w = 1280
        self.path = './test_data'
        self.device = 'cuda'
        self.mask_predictor = MaskPredictor()
        self.positive_obj = positive_obj
        self.passive_obj = passive_obj
        self.estimate_depth = estimate_depth
        # if self.estimate_depth:
            # self.depth_predictor, self.depth_predictor_transform = depth_pro.create_model_and_transforms(device=self.device)
            # self.depth_predictor.eval()

    def get_frame(self, scene_id, frame_id=0, mask_mode='img'):     
        scene_id = str(scene_id).rjust(6,'0')
        frame_id = str(frame_id).rjust(6,'0')
        
        # mask_mode: img or seq
        color = cv2.cvtColor(cv2.imread(f'{self.path}/{scene_id}/color/{frame_id}_color.png'),cv2.COLOR_BGR2RGB)
        w = min(self.max_w, color.shape[1])
        h = int(w*color.shape[0]/color.shape[1])
        color = cv2.resize(color,(w, h), interpolation=cv2.INTER_NEAREST)

        if self.estimate_depth:
            prediction = self.depth_predictor.infer(self.depth_predictor_transform(color))
            depth, focallength_px = prediction['depth'].cpu().numpy() , prediction["focallength_px"] 
        else:
            depth = cv2.imread(f'{self.path}/{scene_id}/depth/{frame_id}_depth.exr',cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = cv2.resize(depth,(w, h), interpolation=cv2.INTER_NEAREST)

        # 试图通过侵蚀一圈来避免depth的噪声
        if mask_mode=='img':
            positive_mask,passive_mask = self.mask_predictor.predict_image(color)
            positive_mask = binary_erosion(positive_mask,structure=np.ones((5,5),dtype=bool))
            passive_mask = binary_erosion(passive_mask,structure=np.ones((5,5),dtype=bool))
            positive_mask = torch.tensor(positive_mask,device=self.device)
            passive_mask = torch.tensor(passive_mask,device=self.device)
        elif mask_mode=='seq':
            positive_mask,passive_mask = self.mask_predictor.sequence_predictor_track(color)
            positive_mask = binary_erosion(positive_mask.cpu().numpy(),structure=np.ones((5,5),dtype=bool))
            passive_mask = binary_erosion(passive_mask.cpu().numpy(),structure=np.ones((5,5),dtype=bool))
            # np.save(f'{self.path}/{scene_id}/mask/{frame_id}_positive_mask.npy',positive_mask)
            # np.save(f'{self.path}/{scene_id}/mask/{frame_id}_positive_mask.npy',passive_mask)
            positive_mask = torch.tensor(positive_mask,device=self.device)
            passive_mask = torch.tensor(passive_mask,device=self.device)
        else:
            print('error type in mask predictor')

        if self.estimate_depth:
            focallength_px
            intrinsics = {}
            intrinsics['fx'] = focallength_px.item()
            intrinsics['fy'] = focallength_px.item()
            intrinsics['cx'] = w / 2
            intrinsics['cy'] = h / 2
        else:
            with open(f'{self.path}/{scene_id}/camera_intrinsics.json') as f:
                intrinsics_origin = json.load(f)
                if intrinsics_origin['height'] == depth.shape[0]:
                    intrinsics = intrinsics_origin
                else:
                    intrinsics = {}
                    scale = depth.shape[0] / intrinsics_origin['height']
                    intrinsics['fx'] = scale * intrinsics_origin['fx']
                    intrinsics['fy'] = scale * intrinsics_origin['fy']
                    intrinsics['cx'] = scale * intrinsics_origin['cx']
                    intrinsics['cy'] = scale * intrinsics_origin['cy']

        frame = {
            'rgb':torch.tensor(color,device=self.device),   # h,w,3, torch.uint8, 0~255
            'depth':torch.tensor(depth,device=self.device), # h,w, torch.float32
            'positive_mask':positive_mask,   # h,w, torch.bool
            'passive_mask':passive_mask,
            'intrinsics':intrinsics
        }
        return frame
    
    def get_sequence(self,scene_id,start_id=0,length=None):

        # initialize mask tracker
        scene_idx = str(scene_id).rjust(6,'0')
        start_idx = str(start_id).rjust(6,'0')
        if length is None:
            length = len(os.listdir(f'{self.path}/{scene_idx}/color'))
        # initialize mask tracker by reading the first frame
        color = cv2.cvtColor(cv2.imread(f'{self.path}/{scene_idx}/color/{start_idx}_color.png'),cv2.COLOR_BGR2RGB)
        if self.max_w < color.shape[1]:
            color = cv2.resize(color,(self.max_w, int(self.max_w*color.shape[0]/color.shape[1])), interpolation=cv2.INTER_NEAREST)

        self.mask_predictor.sequence_predictor_initialize(color,self.positive_obj,self.passive_obj,gdino_name='gdino_annotated_demo')

        # strat tracking
        frames = []
        for i in range(start_id,start_id+length):
            frame = self.get_frame(scene_id,i,mask_mode='seq')
            # frame['positive_keypoint'] = select_keypoint(frame)
            # if i == start_id:
            #     print('choose positive and passive keypoint for demo video')
            #     frame['positive_keypoint'] = select_keypoint(frame)
            #     frame['passive_keypoint'] = select_keypoint(frame)
            frames.append(frame)

        self.show_sequence_mask(frames,scene_id,'positive')
        self.show_sequence_mask(frames,scene_id,'passive')
            
        return frames
    
    def show_sequence_mask(self,frames,scene_id,which_obj):   # which_obj = positive or passive
        scene_idx = str(scene_id).rjust(6,'0')
        rgb_with_mask_list = []
        for frame in frames:
            rgb_with_mask = frame['rgb'].cpu().numpy()
            mask = frame[f'{which_obj}_mask'].cpu().numpy()
            color = np.array([30/255, 144/255, 255/255])
            mask_image =  mask.astype(np.uint8)[:,:,None] * color.reshape(1, 1, -1) * 255.0
            rgb_with_mask[mask] = (rgb_with_mask[mask]*0.5 + mask_image[mask]*0.5).astype(np.uint8)
            rgb_with_mask_list.append(rgb_with_mask)

        writer = imageio.get_writer(f'{self.path}/{scene_idx}/{which_obj}_mask.mp4', fps=10)
        for frame in rgb_with_mask_list:
            writer.append_data(frame)
        writer.close()
        print(f'mask video saved to', f'{self.path}/{scene_idx}/{which_obj}_mask.mp4')

        return HTML("""
            <video width="640" height="360" controls>
            <source src="{0}" type="video/mp4">
            Your browser does not support the video tag.
            </video>
            """.format(f'{self.path}/{scene_idx}/mask.mp4'))




