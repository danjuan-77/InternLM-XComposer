#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_6/AVSQA_v

# nohup bash eval_gpu7.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_IXC2.5-OL_gpu7_$(date +%Y%m%d%H%M%S).log 2>&1 &
