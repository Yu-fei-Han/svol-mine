
CUDA_VISIBLE_DEVICES=0 nohup python runner.py --config-name ours_cvpr testlist=scan140 >140.log   2>&1 
CUDA_VISIBLE_DEVICES=0 nohup python runner.py --config-name ours_cvpr testlist=scan141 >141.log   2>&1 &

