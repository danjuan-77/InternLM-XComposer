#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

# # Level1
# python examples/eval_audio.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LAQA

# python examples/eval_image.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LIQA

# python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LVQA

# # Level2
# python examples/eval_audio.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_2/MAIC

# python examples/eval_image.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_2/MVIC

# # Level3
# python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVH

python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVL

# python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVM
# python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVLG

# python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVQA
# nohup bash eval_0.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_IXC2.5-OL_gpu1_$(date +%Y%m%d%H%M%S).log 2>&1 &
