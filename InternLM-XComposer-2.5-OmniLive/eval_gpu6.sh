#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_6/AVSQA_av

# nohup bash eval_gpu6.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_IXC2.5-OL_gpu6_$(date +%Y%m%d%H%M%S).log 2>&1 &
