cd ./grasping
/home/yan20/anaconda3/envs/omnigrasp/bin/python server.py &
cd ..
# cd ../graspness_unofficial
# /home/yan20/anaconda3/envs/gsnet/bin/python demo_flask.py
# cd ../coi


unset all_proxy
unset ALL_PROXY
cd keypoint
/home/yan20/anaconda3/envs/moka/bin/python keypoint_server_moka.py &
cd ..

cd /home/yan20/jiayuanzhang/GroundingDINO
/home/yan20/anaconda3/envs/gdino/bin/python server.py &

cd /home/yan20/tianshuwu/coi/keypoint
/home/yan20/anaconda3/envs/pmj/bin/python keypoint_server_rekep.py
