#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVL

# nohup bash eval_gpu2.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_IXC2.5-OL_gpu2_$(date +%Y%m%d%H%M%S).log 2>&1 &
