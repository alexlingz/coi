from coi_client import Controller
import torch

demo_scene_id, demo_start_id, demo_end_id,positive_obj,passive_obj,task = 125, 60, 77, 'teapot', 'red cup', 'Pour water into a cup with a teapot'    # 比较好看的demo
# demo_scene_id, demo_start_id, demo_end_id,positive_obj,passive_obj,task = 126, 37, 67, 'teapot', 'red cup', 'Pour water into a cup with a teapot'    # 比较好看的demo



# 注意！！！！！！！！第一帧最好不要遮挡

# while True:
controller= None
torch.cuda.empty_cache()
controller = Controller(demo_scene_id=demo_scene_id, demo_start_id=demo_start_id, demo_end_id=demo_end_id,dynamic_keypoint=False,positive_obj=positive_obj,passive_obj=passive_obj,task=task)    
controller.execute()
# input('Press Enter to continue...')