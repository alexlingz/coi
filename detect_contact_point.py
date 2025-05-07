
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torchvision.transforms as transforms
import torchvision.datasets as dset
# from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, vis_detections_filtered_objects_PIL, vis_detections_filtered_objects # (1) here add a function to viz
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
import logging
from utils import image_coords_to_camera_space


#----------------------------------------------------------------

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='./hand_object_detector/cfgs/res101.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models',
                      default="./hand_object_detector/models")

  parser.add_argument('--cuda', dest='cuda', 
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=8, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=132028, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      default=True)
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)
  parser.add_argument('--thresh_hand',
                      type=float, default=0.5,
                      required=False)
  parser.add_argument('--thresh_obj', default=0.5,
                      type=float,
                      required=False)
  args, unknown = parser.parse_known_args()
  # args = parser.parse_args()
  return args


def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

try:
  xrange          # Python 2
except NameError:
    xrange = range  # Python 3

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

args = parse_args()

# print('Called with args:')
# print(args)

if args.cfg_file is not None:
  cfg_from_file(args.cfg_file)
if args.set_cfgs is not None:
  cfg_from_list(args.set_cfgs)

cfg.USE_GPU_NMS = args.cuda
np.random.seed(cfg.RNG_SEED)

# load model
model_dir = args.load_dir + "/" + args.net + "_handobj_100K" + "/" + args.dataset
if not os.path.exists(model_dir):
  raise Exception('There is no input directory for loading network from ' + model_dir)
load_name = os.path.join(model_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

pascal_classes = np.asarray(['__background__', 'targetobject', 'hand']) 
args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32, 64]', 'ANCHOR_RATIOS', '[0.5, 1, 2]'] 

# initilize the network here.
if args.net == 'vgg16':
  fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
elif args.net == 'res101':
  fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
elif args.net == 'res50':
  fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
elif args.net == 'res152':
  fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
else:
  print("network is not defined")
  pdb.set_trace()

fasterRCNN.create_architecture()

print("load checkpoint %s" % (load_name))
if args.cuda > 0:
  checkpoint = torch.load(load_name)
else:
  checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
fasterRCNN.load_state_dict(checkpoint['model'])
if 'pooling_mode' in checkpoint.keys():
  cfg.POOLING_MODE = checkpoint['pooling_mode']

print('load model successfully!')

def detect_contact(im):



  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)
  box_info = torch.FloatTensor(1) 

  # ship to cuda
  if args.cuda > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()



  with torch.no_grad():
    if args.cuda > 0:
      cfg.CUDA = True

    if args.cuda > 0:
      fasterRCNN.cuda()

    fasterRCNN.eval()

    max_per_image = 100
    thresh_hand = args.thresh_hand 
    thresh_obj = args.thresh_obj

    blobs, im_scales = _get_image_blob(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    with torch.no_grad():
            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()
            box_info.resize_(1, 1, 5).zero_() 

    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info) 

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    # extact predicted params
    contact_vector = loss_list[0][0] # hand contact state info
    offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
    lr_vector = loss_list[2][0].detach() # hand side info (left/right)

    # get hand contact 
    _, contact_indices = torch.max(contact_vector, 2)
    contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

    # get hand side 
    lr = torch.sigmoid(lr_vector) > 0.5
    lr = lr.squeeze(0).float()

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
          if args.class_agnostic:
              if args.cuda > 0:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
              else:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

              box_deltas = box_deltas.view(1, -1, 4)
          else:
              if args.cuda > 0:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
              else:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
              box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= im_scales[0]

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()


    obj_dets, hand_dets = None, None
    for j in xrange(1, len(pascal_classes)):
        # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
        if pascal_classes[j] == 'hand':
          inds = torch.nonzero(scores[:,j]>thresh_hand).view(-1)
        elif pascal_classes[j] == 'targetobject':
          inds = torch.nonzero(scores[:,j]>thresh_obj).view(-1)

        # if there is det
        if inds.numel() > 0:
          cls_scores = scores[:,j][inds]
          _, order = torch.sort(cls_scores, 0, True)
          if args.class_agnostic:
            cls_boxes = pred_boxes[inds, :]
          else:
            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
          
          cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
          cls_dets = cls_dets[order]
          keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
          cls_dets = cls_dets[keep.view(-1).long()]
          if pascal_classes[j] == 'targetobject':
            obj_dets = cls_dets.cpu().numpy()
          if pascal_classes[j] == 'hand':
            hand_dets = cls_dets.cpu().numpy()
  im2show = np.copy(im)
  im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, thresh_hand, thresh_obj)


  is_contact = False
  if obj_dets is not None:
    for hand_det in hand_dets:
      if hand_det[5] > 0.0:
        is_contact = True
        break
  return is_contact,im2show
#----------------------------------------------------------------






import cv2
import mediapipe as mp

import numpy as np

def mark_grip_points(image,mask):
    # 屏蔽 MediaPipe 和 TensorFlow 日志
    logging.getLogger('mediapipe').setLevel(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # 使用 MediaPipe 手部模块
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.2,min_tracking_confidence=0.2)
    mp_drawing = mp.solutions.drawing_utils


    if image is None:
        print("Error: Image not found or unable to load.")
        return
    
    # 转换为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 进行手部检测
    results = hands.process(image_rgb)
    grasp_coord_list = []
    
    if results.multi_hand_landmarks:
        print("Hand landmarks detected.")
        for hand_landmarks in results.multi_hand_landmarks:
            # 遍历手部的关键点并进行可视化
            for landmark in hand_landmarks.landmark:
                # 关键点坐标
                h, w, _ = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                
                # 绘制每个关键点
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)  # 用绿色标记
            
            # 绘制手部骨架
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
            # 识别并标注握住杯柄的点
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # 获取拇指和食指的坐标
            h, w, _ = image.shape
            thumb_tip_coord = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_tip_coord = (int(index_tip.x * w), int(index_tip.y * h))

            # 标注握住杯柄的点
            cv2.circle(image, thumb_tip_coord, 3, (255, 0, 0), 5)  # 用蓝色标记拇指
            cv2.circle(image, index_tip_coord, 3, (255, 0, 0), 5)  # 用蓝色标记食指

            # 抓取点
            grasp_coord = ( int((thumb_tip_coord[0] + index_tip_coord[0]) / 2), int((thumb_tip_coord[1] + index_tip_coord[1]) / 2))
            grasp_coord_list.append(grasp_coord)
            #print(index_tip_coord)
            #print(grasp_coord)
            
        distance_map = distance_transform_edt(mask == 0)
        dis_list = []
        for grasp_coord in grasp_coord_list:
            dis = distance_map[grasp_coord[1],grasp_coord[0]]
            dis_list.append(dis)
        min_dis_idx = np.argmin(dis_list)
        grasp_coord = grasp_coord_list[min_dis_idx]
        cv2.circle(image, grasp_coord, 3, (0, 255, 0),5)  # 用蓝色标记抓取点



        # 输出坐标
        #print(f"Thumb tip coordinates: {thumb_tip_coord}")
        #print(f"Index tip coordinates: {index_tip_coord}")
        
    else:
        print("No hands detected.")
    
    # 显示图片
    # cv2.imshow("Hand Detection", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    return image, grasp_coord





def find_contact_point(demo):
    # step1 use hand_object_detector to detect the first frame that grasping the object
    # step2 use mediapipe to get the 2d coordinate of thumb and index finger
    # step3 取二点的中心，去第一帧（假设待操作物体静止）寻找位于mask内部的距离该点最近的点，并通过depth还原到3d点

    for i, frame in enumerate(demo):
        print(f"==== Processing image. Index: {i} ====")
        #print(seq)

        image = frame['rgb'].cpu().numpy() # RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # if i < 1:
        #    image[:,:,:]=0

        
        
        # cv2.waitKey(100)

        # Step1: detect the first frame
        is_contact,im2show = detect_contact(image)
        print("is_contact:", is_contact)


        # if the first frame
        if is_contact:
            # first_grasp_frame = demo[i]
            first_grasp_frame_id = i
            cv2.imwrite("tmp_image/demo_first_detect_grasping.png", cv2.cvtColor(np.array(im2show),cv2.COLOR_BGR2RGB))
            break

        


    # step2
    while True:
      try:
        image_show, grasp_coord = mark_grip_points(cv2.cvtColor(demo[first_grasp_frame_id]['rgb'].cpu().numpy(),cv2.COLOR_BGR2RGB),demo[first_grasp_frame_id]['positive_mask'].cpu().numpy()) # x,y
        break
      except:
        first_grasp_frame_id += 1
        print('No hand detected, please try again.')
        continue
    print('grasp_coord = ', grasp_coord)
    x,y = grasp_coord
    grasp_coord = (y,x)   # y,x


    # 假设物体在接触之前是一直不动的，所以用第一帧的mask
    first_frame = demo[0]

    positive_mask = first_frame['positive_mask']
    # positive_mask[:] = True


    # Get the indices of the points that are part of the positive mask (True values)
    positive_indices = torch.nonzero(positive_mask)     # y,x




    # plot mask
    #--------------------------------
    positive_mask_cpu = positive_mask.cpu().numpy()  # Now the mask is a numpy array on CPU

    # Create a blue overlay (BGR format, so Blue is [255, 0, 0])
    blue_overlay = np.zeros_like(first_frame['rgb'].cpu().numpy())  # Create an all-black image
    blue_overlay[positive_mask_cpu] = [255, 255, 255]  # Set mask region to blue

    # Combine the original image and the blue overlay
    overlayed_image = cv2.addWeighted(first_frame['rgb'].cpu().numpy(), 1, blue_overlay, 0.5, 0)

    # Display the result
    # cv2.imshow('Image with Blue Mask', overlayed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #--------------------------------


    # Convert grasp_coord to tensor for computation
    grasp_coord_tensor = torch.tensor(grasp_coord, device=positive_indices.device)  # Move grasp_coord to the same device as positive_indices

    # Calculate the Euclidean distance to each positive point
    distances = torch.norm(positive_indices.float() - grasp_coord_tensor.float(), dim=1)

    # Find the index of the minimum distance
    min_distance_index = torch.argmin(distances)

    # Get the coordinates of the closest point
    closest_point = positive_indices[min_distance_index]

    # Convert to tuple if necessary
    closest_point_tuple = tuple(closest_point.cpu().numpy())

    print("Closest point to grasp_coord:", closest_point_tuple)



    # Extract the coordinates from the closest point
    y, x = closest_point_tuple

    # cv2.circle(overlayed_image, (x,y), 5, (0, 0, 255), -1)  # 画一下最终抓取点
    # cv2.imshow('Image with Blue Mask', overlayed_image)
    # cv2.waitKey(0)


    cv2.circle(image_show, (x,y), 5, (0, 0, 255), -1)  # 画一下最终抓取点
    cv2.imwrite("tmp_image/demo_grasping_point.png", image_show)

    # 还原depth
    contact_point_3d = image_coords_to_camera_space(first_frame['depth'].cpu().numpy(),np.array([[y,x]]),first_frame['intrinsics'])

    return contact_point_3d, first_grasp_frame_id #contact_point3d


if __name__ == '__main__':
    from dataset import D415Data
    dataloader = D415Data(estimate_depth=False,positive_obj='teapot',passive_obj='cup')
    seq_demo_scene_id, seq_demo_start_id, seq_demo_length = 22, 23, 55
    seq_demo = dataloader.get_sequence(seq_demo_scene_id,seq_demo_start_id,seq_demo_length)
    contact_point_3d = find_contact_point(seq_demo)




