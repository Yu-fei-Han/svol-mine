
CUDA_VISIBLE_DEVICES=1 nohup python runner.py --config-name ours_cvpr testlist=scan142 >142.log   2>&1 
CUDA_VISIBLE_DEVICES=1 nohup python runner.py --config-name ours_cvpr testlist=scan143 >143.log   2>&1  &

