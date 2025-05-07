 启动franka；
cd ~/panmingjie/workspace/omniagent/omniagent/servers/pose   python server_tracking.py  启动相机，环境是pmj
cd ~/panmingjie/workspace/omniagent/ReKep 
unset ALL_PROXY
unset all_proxy
python main.py    启动rekep，环境是pmj,修改instruction为任务内容

修改robot配置 /home/yan20/panmingjie/workspace/omniagent/omniagent/hardware/configs/franka_pku_umi.json

修改rekep配置 /home/yan20/panmingjie/workspace/omniagent/ReKep/configs/config.yaml

修改标定结果：/home/yan20/panmingjie/workspace/omniagent/omniagent/hardware/configs/thirdeye_pku.json extrinsic

