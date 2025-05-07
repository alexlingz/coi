import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
sys.path.append('/home/wts/code/coi/segmentanything2')

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pickle
import requests
device = 'cuda'

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=25):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


from sam2.build_sam import build_sam2,build_sam2_camera_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

class MaskPredictor:
    def __init__(self,vis_init=True):
        sam2_checkpoint = "/home/yan20/tianshuwu/coi/segmentanything2/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda')
        self.image_predictor = SAM2ImagePredictor(sam2_model)   # 两个模型大约各1g显存
        
        self.visiualized = vis_init


    def select_point_for_image(self,image):
        def select_point_cv2(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                positive.append((x, y))
                cv2.circle(img_visual, (x, y), 1, (0, 0, 255), -1)  # 绘制红色点
                cv2.imshow("Image", img_visual)
            if event == cv2.EVENT_RBUTTONDOWN:
                negative.append((x, y))
                cv2.circle(img_visual, (x, y), 1, (255, 0, 0), -1)  # 绘制蓝色点
                cv2.imshow("Image", img_visual)
        positive = []
        negative = []
        img_visual = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        cv2.imshow("Image", img_visual)
        cv2.setMouseCallback("Image", select_point_cv2)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # 必须按Esc退出，否则会卡住
                break
        cv2.destroyAllWindows()
        # import ipdb;ipdb.set_trace()
        points = positive + negative
        labels = [1]*len(positive) + [0]*len(negative)
        # exit()
        return points, labels


    def predict_image(self,image):
        # image h,w,3, np.ndarray,rgb
        self.image_predictor.set_image(image)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            print('select points for predict single image,positive object')
            positive_points, positive_labels = self.select_point_for_image(image)
            positive_masks, scores, logits = self.image_predictor.predict(
                point_coords=positive_points,
                point_labels=positive_labels,
                multimask_output=False
            )
            print('select points for predict single image,passive object')
            passive_points, passive_labels = self.select_point_for_image(image)
            passive_masks, scores, logits = self.image_predictor.predict(
                point_coords=passive_points,
                point_labels=passive_labels,
                multimask_output=False
            )   
        # mask: 1,h,w, np.array(float32)  
        # scores: 1
        # logits: 1,256,256,np.array(float32)

        show_masks(image,positive_masks,scores,point_coords=np.array(positive_points),input_labels=np.array(positive_labels))
        show_masks(image,passive_masks,scores,point_coords=np.array(passive_points),input_labels=np.array(passive_labels))

        return positive_masks[0].astype(bool),passive_masks[0].astype(bool)
    
    
    # input frame, output bbox
    def get_gdino_bbox(self,rgb,positive_obj=None,passive_obj=None,gdino_name='gdino_annotated_image'):
        print('rgb', rgb)
        
        def jaccard_similarity(s1, s2):
            set1 = set(s1)
            set2 = set(s2)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union
        
        if passive_obj is not None:
            data = {
                'image': rgb,
                'text': f'{positive_obj} . {passive_obj}',
                'gdino_name': gdino_name
            }
        else:
            data = {
                'image': rgb,
                'text': f'{positive_obj}',
                'gdino_name': gdino_name
            }
        
        # binary serialization
        serialized_data = pickle.dumps(data)
        

        response = requests.post('http://127.0.0.1:5001/gdino', data=serialized_data)
        # print('http://127.0.0.1:5001/gdino response', response)
        # responce content is encoded into binary string

        # raise Exception('stop here')
        
        print('response.content', response.content)
        
        gdino_result = pickle.loads(response.content)
        
        
        prompt_bbox = {}
        
        if passive_obj is not None:
            for i,objname in enumerate(gdino_result['phrases']):
                if jaccard_similarity(positive_obj,objname) > jaccard_similarity(passive_obj,objname):
                    prompt_bbox['positive'] = gdino_result['bboxes'][i]
                if jaccard_similarity(positive_obj,objname) < jaccard_similarity(passive_obj,objname):
                    prompt_bbox['passive'] = gdino_result['bboxes'][i]
        else:
            prompt_bbox['positive'] = gdino_result['bboxes'][0]

        return prompt_bbox  # {'positive': [x1,y1,x2,y2],'passive': [x1,y1,x2,y2]}  or {'positive': [x1,y1,x2,y2]}
        
    
    def sequence_predictor_initialize(self,init_image,positive_obj=None,passive_obj=None,gdino_name='gdino_annotated_image'):

        sam2_checkpoint = "/home/yan20/tianshuwu/coi/segmentanything2/checkpoints/sam2_hiera_large.pt"
        model_cfg = "configs/sam2/sam2_hiera_l.yaml"

        self.camera_predictor = build_sam2_camera_predictor(model_cfg,sam2_checkpoint,device='cuda')

        self.camera_predictor.load_first_frame(init_image)
        
        if positive_obj is not None:
            
            prompt_bbox = self.get_gdino_bbox(init_image,positive_obj,passive_obj,gdino_name)
            
            print('use gdino bbox as sam2 s initial prompt')
            _,_,_ = self.camera_predictor.add_new_prompt(
                frame_idx=0,
                obj_id=(0),
                # points=points,
                # labels=labels
                bbox=prompt_bbox['positive']
            )

            if passive_obj is not None:
                _,_,_ = self.camera_predictor.add_new_prompt(
                    frame_idx=0,
                    obj_id=(1),
                    # points=points,
                    # labels=labels
                    bbox=prompt_bbox['passive']
                )
        else:
            print('select prompt points for positive object')
            points, labels = self.select_point_for_image(init_image
            )
            _,_,_ = self.camera_predictor.add_new_prompt(
                frame_idx=0,
                obj_id=(0),
                points=points,
                labels=labels
            )
            print('select prompt points for passive object')
            points, labels = self.select_point_for_image(init_image
            )
            _,_,_ = self.camera_predictor.add_new_prompt(
                frame_idx=0,
                obj_id=(1),
                points=points,
                labels=labels
            )

        return 

    def sequence_predictor_track(self,image):
        obj_id, out_mask_logits = self.camera_predictor.track(image)
        # masks: 2,1,h,w logits
        positive_mask = out_mask_logits[0][0] > 0.0
        if len(out_mask_logits) > 1:
            passive_mask = out_mask_logits[1][0] > 0.0
        else:
            passive_mask = out_mask_logits[0][0] > 0.0
        
        if not self.visiualized:
            show_masks(image,[positive_mask.cpu().numpy()],[1.0],point_coords=None,input_labels=None)
            show_masks(image,[passive_mask.cpu().numpy()],[1.0],point_coords=None,input_labels=None)
            self.visiualized = True
        
        return positive_mask,passive_mask
        
