#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVR

# python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/VAH

# python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/VAR

# # Level4
# python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVC

# python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVLG

# python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVQA

# Level5
python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVLG

# python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVQA

# nohup bash eval_1.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_IXC2.5-OL_gpu2_$(date +%Y%m%d%H%M%S).log 2>&1 &
