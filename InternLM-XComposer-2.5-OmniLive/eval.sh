#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

python examples/eval_audio.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LAQA

python examples/eval_audio.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_2/MAIC