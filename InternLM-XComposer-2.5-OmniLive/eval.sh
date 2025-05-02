#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# Level1
python examples/eval_audio.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LAQA

python examples/eval_image.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LIQA

python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_1/LVQA

# Level2
python examples/eval_audio.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_2/MAIC

python examples/eval_image.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_2/MVIC

# Level3
python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVH

python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVL

python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVM

python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/AVR

python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/VAH

python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_3/VAR

# Level4
python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVC

python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVLG

python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_4/AVQA

# Level5
python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVLG

python examples/eval_video.py --task_path /share/nlp/tuwenming/projects/HAVIB/data/levels/level_5/AVQA

# nohup bash eval.sh > /share/nlp/tuwenming/projects/HAVIB/logs/eval_minicpm-o_gpu2_$(date +%Y%m%d%H%M%S).log 2>&1 &
