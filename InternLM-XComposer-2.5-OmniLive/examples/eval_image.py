import sys
sys.path.insert(0, '../')

import argparse

import torch
from transformers import AutoModel, AutoTokenizer

from example_code.utils import auto_configure_device_map

torch.set_grad_enabled(False)

import tempfile
import traceback
import json
import os
from tqdm import tqdm
from typing import List, Optional
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    concatenate_audioclips,
    ImageClip,
    concatenate_videoclips,
)
import argparse


tempfile.tempdir = "/share/nlp/tuwenming/projects/HAVIB/tmp"
pmp_avl_ans_format = "answer={'category1_id1': '[x_min, y_min, x_max, y_max]', 'category2_id2': '[x_min, y_min, x_max, y_max]'}"
avl_cls_list = ['dog', 'clarinet', 'banjo', 'cat', 'guzheng', 'tree', 'lion', 'tuba', 
        'ukulele', 'flute', 'piano', 'person', 'violin', 'airplane', 'bass', 'pipa', 
        'trumpet', 'accordion', 'saxophone', 'car', 'lawn-mower', 'cello', 'bassoon', 
        'horse', 'guitar', 'erhu', 'not sure', 'no available option']
prompt_avl = f"""
        ctaegories list: {avl_cls_list}
        (1) There may be multiple sounding instances, you can choose instance categories from the given categories list.
        (2) The naming format for instances is: category_id. Instance IDs start from 1, e.g., male_1, dog_2, dog_3, cat_4. 
        (3) The bbox format is: [x_min, y_min, x_max, y_max], where x_min, y_min represent the coordinates of the top-left corner. 
        (4) The bbox values should be normalized into the range of 0 and 1, e.g., [0.1, 0.12, 0.26, 0.14].
        Do not explain, you must strictly follow the format: {pmp_avl_ans_format}
    """

prompt_avlg = """
        Please output the answer in a format that strictly matches the following example, do not explain:
        answer={'frame_0': [x0_min, y0_min, x0_max, y0_max], 'frame_1': None, ..., 'frame_9': [x9, y9, w9, h9]}
        Note, 
        (1) x_min, y_min represent the coordinates of the top-left corner, while x_max, y_max for the bottom_right corner.
        (2) The bbox values should be normalized into the range of 0 and 1, e.g., [0.1, 0.12, 0.26, 0.14]. 
        (3) Frames should be ranged from frame_0 to frame_9.
    """

avqa_cls_list = ['ukulele', 'cello', 'clarinet', 'violin', 'bassoon', 'accordion', 'banjo', 'tuba', 'flute', 'electric_bass', 'bagpipe', 
        'drum', 'congas', 'suona', 'xylophone', 'saxophone', 'guzheng', 'trumpet', 'erhu', 'piano', 'acoustic_guitar', 'pipa', 'not sure', 'no available option']

havib_constants = {
    'L3_AVH': {
        'prompt_avh': "Please answer the question based on the given video.",
        'avh_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_VAH': {
        'prompt_vah': "Please answer the question based on the given audio.",
        'vah_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_AVL': {
        'prompt_avl': prompt_avl,
        'avl_cls_list': avl_cls_list,
    },


    'L4_AVC': {

    },

    'L4_AVLG': {
        'prompt_avlg': prompt_avlg,
    },

    'L4_AVQA': {
        'avqa_options_list_is': ['yes', 'no', 'not sure'],
    },

    'L5_AVLG': {
        'prompt_avlg': prompt_avlg,
    },

    'L5_AVQA': {
        'avqa_cls_list': avqa_cls_list,
        'avqa_options_list_is': ['yes', 'no', 'not sure'],
    },
}

def get_real_path(task_path: str, src_path: str) -> str:
    """传入taskpath和一些文件的path，构造文件的真实path

    Args:
        task_path (str): task path
        src_path (str): 每个文件的path

    Returns:
        str: 文件的真实path
    """
    temp_path = os.path.join(task_path, src_path)
    return os.path.normpath(temp_path)

def get_real_options_or_classes(d: dict) -> str:
    """Replace pseudo-options with real options text."""
    opts = d['input']['question'].get('options')
    if opts in havib_constants.get(d['task'], {}):
        opts = havib_constants[d['task']][opts]
    if opts:
        label = 'semantic categories' if 'cls' in opts else 'options'
        return f"Available {label} are: {opts}"
    return ''

def get_real_prompt(d: dict) -> str:
    """Replace pseudo-prompt with real prompt text."""
    prm = d['input']['question'].get('prompt')
    if prm in havib_constants.get(d['task'], {}):
        prm = havib_constants[d['task']][prm]
    return prm or ''

def get_real_input(d: dict) -> str:
    """Concatenate prompt, options, and question text into one input string."""
    prompt = get_real_prompt(d)
    options = get_real_options_or_classes(d)
    question = d['input']['question']['text'] or ''
    # 去掉多余的句点
    parts = [p for p in (prompt, options, question) if p]
    return " ".join(parts)

######################## Above are help function tools

def init_model():
    model_path = 'internlm-xcomposer2d5-ol-7b/base'
    # init model and tokenizer
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--dtype", default='fp16', type=str)
    parser.add_argument("--task_path", type=str, required=True, help="Path to the task folder containing data.json and media files")
    
    args = parser.parse_args()

    model, tokenizer = init_model()
    if args.dtype == 'fp16':
        model.half().cuda()
    elif args.dtype == 'fp32':
        model.cuda()
    if args.num_gpus > 1:
        from accelerate import dispatch_model
        device_map = auto_configure_device_map(args.num_gpus)
        model = dispatch_model(model, device_map=device_map)    
    model.tokenizer = tokenizer
    # model.generation_config.max_new_tokens = 256
    task_path = args.task_path
    task_name = f"L{task_path.rsplit('/', 1)[0][-1]}_{task_path.rsplit('/', 1)[-1]}"
    model_name = "IXC2.5-OL"
    save_prediction_json = f'/share/nlp/tuwenming/projects/HAVIB/eval/user_outputs/{model_name}/tasks/{task_name}.json'
    os.makedirs(os.path.dirname(save_prediction_json), exist_ok=True)
    print('>>> save res to:', save_prediction_json)

    data_json_path = os.path.join(task_path, "data.json")
    with open(data_json_path, "r", encoding='utf-8') as f:
        raw_data = json.load(f)
    print(">>>Finished load raw data...")
    parsed_data = []
    for item in raw_data:
        inp = item.get('input', {})
        entry = {
            'id': item.get('id'),
            'task': item.get('task'),
            'subtask': item.get('subtask', None),
            'text': get_real_input(item),
            'audio_list': inp.get('audio_list', None),
            'image_list': inp.get('image_list', None),
            'video': inp.get('video', None)
        }
        parsed_data.append(entry)

    print(">>>Finished parse raw data...")    
    
    predictions = []
    
    for data in tqdm(parsed_data):
        _id = data['id']
        _task = data['task']
        _subtask = data['subtask']
        text = data['text']
        audio_list = (
            [get_real_path(task_path, p) for p in data["audio_list"]]
            if data["audio_list"] else None
        )
        image_list = (
            [get_real_path(task_path, p) for p in data["image_list"]]
            if data["image_list"] else None
        )
        video = (
            get_real_path(task_path, data['video'])
            if data['video'] else None
        )
        print(f">>> text input=:{text}")



        # question = 'Analyze the given image in a detail manner'
        # image = ['../examples/dubai.png']
        try:
            question = text
            image = image_list

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                response, _ = model.chat(tokenizer, question, image, do_sample=False, num_beams=3, use_meta=True)
                
        except Exception as e:
            # 捕获任何异常，并把完整 traceback 当作 output
            tb = traceback.format_exc()
            response = f"Error during inference:\n{tb}"
        
        pred_record = {
            "task": _task,
            "subtask": _subtask,
            "id": _id,
            "predict": response,
        }
        predictions.append(pred_record)
        print('>>> ans=:', pred_record)
        
    with open(save_prediction_json, 'w', encoding='utf-8') as json_file:
        json.dump(predictions, json_file, ensure_ascii=False, indent=4)

# Expected output
"""
The image presents an infographic titled "Amazing Facts About Dubai," highlighting various aspects of the city. It begins with a depiction of Dubai's skyline, featuring iconic buildings like the Burj Al Arab and the Burj Khalifa. The infographic mentions that Palm Jumeirah is the largest artificial island in the world and is visible from space. It also states that in 1968, there were only 1.5 million cars in Dubai, whereas today there are more than 1.5 million cars. Dubai has the world's largest Gold Chain, with about 7 of the 10 tallest hotels in the world located there. The Gold Chain is 4.2 km long. The crime rate in Dubai is 0%, and the income tax rate is also 0%. Dubai Mall is the largest shopping mall in the world with 1200 stores. Dubai has no standard address system, and the Burj Khalifa is so tall that its residents on top floors need to wait longer to break fast during Ramadan. Dubai is building a climate-controlled City, and the Royal Suite at the Burj Al Arab costs $24,000 per night. The net worth of the four listed billionaires is roughly equivalent to the GDP of Honduras. The infographic concludes with a note that you need a license to drink alcohol even at home. The sources of the information are cited at the bottom, and the infographic was designed and compiled by www.fmextensions.ae.
"""


if __name__ == "__main__":
    main()