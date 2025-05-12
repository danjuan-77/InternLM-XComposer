#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python examples/eval_audio.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/VAH

# nohup bash eval_gpu0.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_IXC2.5-OL_gpu0_$(date +%Y%m%d%H%M%S).log 2>&1 &
