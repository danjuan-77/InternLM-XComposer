import sys
sys.path.append('internlm-xcomposer2d5-ol-7b')
sys.path.append('internlm-xcomposer2d5-ol-7b/memory')

import argparse

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from decord import VideoReader
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModelForCausalLM

from base.ixc_utils import Video_transform
from base.modeling_internlm_xcomposer2 import get_stopping_criteria
from memory.constants import DEFAULT_IMAGE_PATCH_TOKEN
from memory.grounding_qwen import GroundQwenForCausalLM
from memory.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from memory.mm_utils import tokenizer_image_token
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
tempfile.tempdir = "/share/nlp/tuwenming/projects/HAVIB/tmp"
maic_cls_list = ['bus', 'hair-dryer', 'pipa', 'man', 'ambulance', 'razor', 'harp', 'tabla', 'bass', 'handpan', 
        'girl', 'sitar', 'car', 'lion', 'guitar', 'vacuum-cleaner', 'cat', 'mower', 'helicopter', 'boy', 'drum', 
        'keyboard', 'tuba', 'saw', 'flute', 'cello', 'woman', 'gun', 'accordion', 'violin', 'clarinet', 'erhu', 
        'saxophone', 'guzheng', 'dog', 'baby', 'horse', 'male', 'wolf', 'bird', 'ukulele', 'piano', 'female', 
        'marimba', 'not sure', 'no available option']

mvic_cls_list = ['sushi', 'banana', 'cake', 'butterfly', 'bird', 'microphone', 'hamburger', 'pineapple', 
        'man', 'book', 'sunglasses', 'goat', 'tie', 'cabinetry', 'motorcycle', 'drawer', 'strawberry', 
        'sheep', 'pasta', 'parrot', 'bull', 'table', 'penguin', 'watch', 'pillow', 'shellfish', 'kangaroo', 
        'flower', 'paddle', 'rocket', 'helicopter', 'bus', 'mushroom', 'bee', 'tree', 'boat', 'saxophone', 
        'football', 'lizard', 'violin', 'dog', 'cucumber', 'cello', 'airplane', 'horse', 'drum', 'box', 
        'rabbit', 'car', 'door', 'orange', 'shelf', 'camera', 'poster', 'lemon', 'cat', 'fish', 'bread', 
        'piano', 'apple', 'glasses', 'bicycle', 'truck', 'deer', 'woman', 'wheelchair', 'cheese', 'chair', 
        'plate', 'tomato', 'bed', 'starfish', 'balloon', 'bottle', 'crab', 'beer', 'frog', 'shrimp', 'tower', 
        'guitar', 'pig', 'peach', 'train', 'pumpkin', 'elephant', 'jellyfish', 'parachute', 'monkey', 'flag',
        'not sure', 'no available option']

prompt_avl = """
        Based on the given image and audio, there may be multiple sounding instances.
        You can choose instance categories from the given categories list.
        The naming format for instances is: category_id. Instance IDs start from 1, e.g., male_1, dog_2, dog_3, cat_4. 
        The bbox format is: [x, y, w, h], where x and y represent the coordinates of the top-left corner, and w and h are the width and height. 
        The final answer must strictly adhere to the following format: 
        answer={"guzheng_1": "[269, 198, 83, 16]", "guzheng_2": "[147, 196, 75, 13]", "female_3": "[152, 108, 123, 36]"}
    """

avl_cls_list = ['dog', 'clarinet', 'banjo', 'cat', 'guzheng', 'tree', 'lion', 'tuba', 
        'ukulele', 'flute', 'piano', 'person', 'violin', 'airplane', 'bass', 'pipa', 
        'trumpet', 'accordion', 'saxophone', 'car', 'lawn-mower', 'cello', 'bassoon', 
        'horse', 'guitar', 'erhu', 'not sure', 'no available option']

prompt_avlg = """
        Please output the answer in a format that exactly matches the following example:
        answer={'frame_0': [x0, y0, w0, h0], 'frame_1': None, ..., 'frame_9': [x9, y9, w9, h9]}.
        Note, for [x, y, w, h], where x and y represent the top-left corner of the bounding box, 
        and w and h represent the width and height of the bounding box.
    """

avqa_cls_list = ['ukulele', 'cello', 'clarinet', 'violin', 'bassoon', 'accordion', 'banjo', 'tuba', 'flute', 'electric_bass', 'bagpipe', 
        'drum', 'congas', 'suona', 'xylophone', 'saxophone', 'guzheng', 'trumpet', 'erhu', 'piano', 'acoustic_guitar', 'pipa', 'not sure', 'no available option']

havib_constants = {
    'L1_LAQA': {
        'options_sound_clarity': ['first', 'last', 'same', 'not sure'],
        'options_sound_order': ['sound', 'noise', 'not sure'],
        'options_sound_volume': ['first', 'last', 'same', 'not sure'],
        'options_yes_no': ['yes', 'no', 'not sure'],
    },

    'L1_LIQA': {
        'get_from_background_binary': ['yes', 'no', 'not sure'],
        'get_from_image_binary': ['yes', 'no', 'not sure'],
        'get_from_foreground_binary': ['yes', 'no', 'not sure'],
        'get_from_image_triple': ['blurred', 'normal', 'clear', 'not sure'],
        'get_from_3d-task1': ['center', 'left', 'right', 'not sure'],
        'get_from_3d-task2': ['cone', 'cube', 'cylinder', 'cuboid', 'no available option', 'not sure'],
        # 'get_from_3d-task3': [0, 1, 2, 3, 4, 5, 6],
        'get_from_space_hard': ['center', 'top left', 'top center', 'top right', 'bottom left', 'bottom center', 'bottom right', 'no available option', 'not sure'],
        'get_from_color': ['blue', 'green', 'red', 'puprle', 'yellow', 'no available option', 'not sure'],
        'get_yes_no': ['yes', 'no', 'not sure'],
        # 'get_lines_count': [0, 1, 2, 3, 4],
        'get_lines_direction': ['horizontal', 'vertical', 'inclined', 'not sure'],
        'get_from_space_easy_area': ['the right one', 'the left one', 'the middle one', 'the bottom one', 'the top one'],
        'get_from_space_easy_bbrightness': ['the right one', 'the left one', 'the middle one', 'the bottom one', 'the top one'],
        'options_yes_no': ['yes', 'no', 'not sure'],
    },

    'L1_LVQA': {
        'which_object': ['square', 'circle', 'triangle', 'not sure', 'no available option', 'not sure'],
        'what_shape': ['Triangular pyramid', 'Cone', 'Cube', 'Sphere', 'None', 'not sure'],
        # 'how_many': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'what_movement_2d': ['horizontal', 'inclined', 'vertical', 'no movenment', 'None', 'not sure'],
        'what_movement_3d': ['Rotation', 'Shrinking', 'Translation', 'Enlarging', 'None', 'not sure'],
        'what_surface': ['Rough', 'Moderate', 'Smooth', 'None', 'not sure'],
        'spacial_change': ['Bottom-left to top-right', 'Bottom-right to top-left', 'Top-left to bottom-right', 'Top-right to bottom-left', 'None', 'not sure', 'No movement',],
        'options_yes_no': ['yes', 'no', 'not sure'],
    },

    'L2_MAIC': {
        'maic_cls_list': maic_cls_list,
        'prompt_maic': "There may be one or more sound-emitting objects in the provided audio. \nPlease strictly output the answer in the format answer={'cls_1': count_1, 'cls_2': count_2}, \nfor example, answer={'dog': 2, 'cat': 3, 'male': 1}. \n"
    },

    'L2_MVIC': {
        'mvic_cls_list': mvic_cls_list,
        'prompt_mvic': "There may be one or more visible objects in the provided image. \nPlease strictly output the answer in the format answer={'cls_1': count_1, 'cls_2': count_2}, \nfor example, answer={'dog': 2, 'cat': 3, 'male': 1}. \n Possible categoris are in the list: mvic_cls_list"
    },

    'L3_AVH': {
        'prompt_avh': "Please answer the question based on the given audio and video.",
        'avh_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_VAH': {
        'prompt_vah': "Please answer the question based on the given audio and video.",
        'vah_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_AVL': {
        'prompt_avl': prompt_avl,
        'avl_cls_list': avl_cls_list,
    },

    'L3_AVM': {
        'prompt_avm': 'Please answer the question based on the given audio and video.',
        'avm_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_AVR': {
        'prompt_avr': "Please output the indices of the images list, starting from 0. For example: [], or [0, 3], or [1, 4, 9]."
    },

    'L3_VAR': {
        'prompt_var': "Please output the indices of the wavs list, starting from 0. For example: [], or [0, 3], or [1, 4, 9]."
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

def concat_audio(audio_paths: List[str]) -> str:
    """
    Concatenate multiple audio files into one WAV file.
    Returns the path to the temp WAV file.
    """
    clips = [AudioFileClip(p) for p in audio_paths]
    final = concatenate_audioclips(clips)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = tmp.name
    final.write_audiofile(out_path, fps=16000, logger=None)
    return out_path

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

def images_to_video(image_paths: List[str], duration: float, fps: int = 1) -> str:
    """
    Turn a list of images into a silent video of total `duration` seconds.
    Each image is shown for `duration / len(image_paths)` seconds.
    Returns the path to the temp MP4 file.
    """
    single_dur = duration / len(image_paths)
    clips = [ImageClip(p).set_duration(single_dur) for p in image_paths]
    video = concatenate_videoclips(clips, method="compose")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    video.write_videofile(out_path, fps=fps, codec="libx264", audio=False, logger=None)
    return out_path

def images_and_audio_to_video(image_paths: List[str], audio_paths: List[str], fps: int = 1) -> str:
    """
    Concatenate audio_paths into one audio, then build a video from image_paths
    that matches the audio duration, and merge them.
    Returns the path to the temp MP4 file.
    """
    # 1) build the concatenated audio
    audio_path = concat_audio(audio_paths)
    audio_clip = AudioFileClip(audio_path)
    # 2) build video from images matching audio duration
    duration = audio_clip.duration
    vid_path = images_to_video(image_paths, duration, fps=fps)
    # 3) attach audio to video
    video_clip = AudioFileClip(audio_path)  # re-open to avoid MoviePy caching issues
    from moviepy.editor import VideoFileClip
    base_vid = VideoFileClip(vid_path)
    final = base_vid.set_audio(audio_clip)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    final.write_videofile(out_path, fps=fps, codec="libx264", logger=None)
    return out_path 
######################## Above are help function tools
def model_gen_withmem(model, text, images, glb, lol, need_bos=True, hd_num=36, max_new_token=2, beam=3, do_sample=False):
    temp_emb = []

    _, c = glb.shape
    glb = model.video_mem_proj(glb.view(1, -1, c))
    glb_text = model.encode_text('This is video overview memory:', add_special_tokens=False)
    temp_emb.append(glb_text)
    temp_emb.append(glb)

    if len(lol) > 0:
        _, c = lol.shape
        lol = model.video_mem_proj(lol.view(1, -1, c))
        lol_text = model.encode_text('This is question related video memory:', add_special_tokens=False)
        temp_emb.append(lol_text)
        temp_emb.append(lol)

    image = Video_transform(images, hd_num=hd_num)
    image_embeds = model.vis_processor(image).unsqueeze(0).cuda()
    image_embeds = model.encode_img(image_embeds)
    temp_emb.append(image_embeds)
    temp_emb = torch.cat(temp_emb, dim=1)
    images = temp_emb

    pt1 = 0
    embeds = []
    im_mask = []
    if images is None:
        images = []
        images_loc = []
    else:
        images = [images]
        images_loc = [len('[UNUSED_TOKEN_146]user\n')]
    for i, pts in enumerate(images_loc + [len(text)]):
        subtext = text[pt1:pts]
        if need_bos or len(subtext) > 0:
            text_embeds = model.encode_text(subtext, add_special_tokens=need_bos)
            embeds.append(text_embeds)
            im_mask.append(torch.zeros(text_embeds.shape[:2]).cuda())
            need_bos = False
        if i < len(images):
            image_embeds = images[i]
            embeds.append(image_embeds)
            im_mask.append(torch.ones(image_embeds.shape[:2]).cuda())
        pt1 = pts
    embeds = torch.cat(embeds, dim=1)
    im_mask = torch.cat(im_mask, dim=1)
    im_mask = im_mask.bool()

    stop_words_ids = [92542]
    stopping_criteria = get_stopping_criteria(stop_words_ids)

    outputs = model.generate(inputs_embeds=embeds, im_mask=im_mask,
                             temperature=1.0, max_new_tokens=max_new_token, num_beams=beam,
                             do_sample=False, repetition_penalty=1.00, stopping_criteria=stopping_criteria)

    output_token = outputs[0]
    if output_token[0] == 0 or output_token[0] == 1:
        output_token = output_token[1:]
    output_text = model.tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('[UNUSED_TOKEN_145]')[0].strip()
    output_text = output_text.split('<|im_end|>')[0].strip()
    return output_text


def img_process(imgs):
    new_imgs = []
    for img in imgs:
        w, h = img.size
        scale = w/h
        if w > h:
            new_w = 560 * 2
            new_h = int(560 * 2 / scale)
        else:
            new_w = int(560 * 2 * scale)
            new_h = 560 * 2
        img = transforms.functional.resize(img, [new_h, new_w],)
        new_imgs.append(img)
    imgs = new_imgs
    new_w = 0
    new_h = 0
    pad = 40
    if w > h:
        for im in imgs:
            w,h = im.size
            new_w = max(new_w, w)
            new_h += h + 10 + pad
        font = ImageFont.truetype("internlm-xcomposer2d5-ol-7b/base/SimHei.ttf", pad)
        new_img = Image.new('RGB', (new_w, new_h), 'white')
        draw = ImageDraw.Draw(new_img)
        curr_h = 0
        for idx, im in enumerate(imgs):
            w,h = im.size
            new_img.paste(im, (0, pad + curr_h))
            draw.text((0, curr_h ), f'<IMAGE {idx}>', font=font, fill='black')
            if idx + 1 < len(imgs):
                draw.line([(0, pad +curr_h + h +5), (new_w, pad +curr_h + h +5)], fill = 'black', width=2)
            curr_h += h + 10 + pad
        #print (new_w, new_h)
    else:
        for im in imgs:
            w,h = im.size
            new_w += w + 10
            new_h = max(new_h, h)
        new_h += pad
        font = ImageFont.truetype("internlm-xcomposer2d5-ol-7b/base/SimHei.ttf", pad)
        new_img = Image.new('RGB', (new_w, new_h), 'white')
        draw = ImageDraw.Draw(new_img)
        curr_w = 0
        for idx, im in enumerate(imgs):
            w,h = im.size
            new_img.paste(im, (curr_w, pad))
            draw.text((curr_w, 0), f'<IMAGE {idx}>', font=font, fill='black')
            if idx + 1 < len(imgs):
                draw.line([(curr_w + w + 5, 0), (curr_w + w + 5, new_h)], fill = 'black', width=2)
            curr_w += w + 10
    return new_img


def preprocess_time(time, num_clip, tokenizer):
    time = time.reshape(2, num_clip)
    seq = []

    block_size = 1
    for i in range(num_clip):
        start, end = time[:, i]
        start = int(np.round(start))
        end = int(np.round(end))
        if (i + 1) % block_size == 0:
            history_end = end
        sentence = 'This contains a clip sampled in %d to %d seconds' % (start, end) + DEFAULT_IMAGE_TOKEN
        sentence = tokenizer_image_token(sentence, tokenizer, return_tensors='pt')
        seq.append(sentence)
    return seq


def preprocess_question(questions, tokenizer):
    seq = []
    for q in questions:
        sentence = tokenizer_image_token(q, tokenizer, return_tensors='pt')
        seq.append(sentence)

    return seq


def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq


def get_seq_time(vr, frame_idx, num_clip):
    frm_per_clip = len(frame_idx) // num_clip
    key_frame = [[frame_idx[i * frm_per_clip], frame_idx[i * frm_per_clip + frm_per_clip - 1]] for i in range(num_clip)]
    time = vr.get_frame_timestamp(key_frame)
    return np.hstack([time[:, 0, 0], time[:, 1, 1]])


def load_video(vis_path, num_frm=16, max_clip=4):
    block_size = 1
    vr = VideoReader(vis_path)
    total_frame_num = len(vr)
    fps = vr.get_avg_fps()
    total_time = total_frame_num / fps

    num_clip = total_time / num_frm
    num_clip = int(block_size * np.round(num_clip / block_size)) if num_clip > block_size else int(np.round(num_clip))
    num_clip = max(num_clip, 5)
    num_clip = min(num_clip, max_clip)
    total_num_frm = num_frm * num_clip
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)

    time_idx = get_seq_time(vr, frame_idx, num_clip)
    img_array = vr.get_batch(frame_idx).asnumpy()  # (n_clips*num_frm, H, W, 3)

    img_array = img_array.reshape((1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

    clip_imgs = []
    for j in range(total_num_frm):
        clip_imgs.append(Image.fromarray(img_array[0, j]))

    return clip_imgs, time_idx, num_clip


def inference(args):
    ixc_tokenizer = AutoTokenizer.from_pretrained(f'{args.ixc_model_path}/merge_lora', trust_remote_code=True)
    ixc_model = AutoModelForCausalLM.from_pretrained(f'{args.ixc_model_path}/merge_lora', device_map="cuda", trust_remote_code=True).eval().cuda().to(torch.bfloat16)
    ixc_model.tokenizer = ixc_tokenizer

    
    
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

        if video and not audio_list and not image_list:
            # Case 1: 仅视频
            video_path = video
            
            
        elif video and audio_list:
            # Case 2: 视频+音频
            video_path = video
    
        elif image_list and audio_list and not video:
            # Case 5: 图像+音频 -> 合成视频, 使用视频的audio
            video_path = images_and_audio_to_video(image_list, audio_list, fps=1)
    
    
        kwargs = {"device_map": 'cuda'}
        kwargs['torch_dtype'] = torch.float16
        vs_tokenizer = AutoTokenizer.from_pretrained(f'{args.ixc_model_path}/memory', use_fast=False)
        vs_model = GroundQwenForCausalLM.from_pretrained(f'{args.ixc_model_path}/memory', low_cpu_mem_usage=True, **kwargs)
        mm_use_im_start_end = getattr(vs_model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(vs_model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            vs_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            vs_tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        vs_model.resize_token_embeddings(len(vs_tokenizer))

        vision_tower = vs_model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(0, dtype=torch.float16)
        image_processor = vision_tower.image_processor

        # video_path = 'examples/videos/needle_32.mp4'
        # question_ = 'What does the hand coming out of the computer do?'
        # candidates = ['Delivers a product', "Shakes the woman's hand", "Takes the woman's credit card", 'Points at something on the screen']
        question_ = text
        frames, time_idx, num_clips = load_video(video_path, num_frm=16, max_clip=32)
        video = image_processor.preprocess(frames, return_tensors='pt')['pixel_values']
        video = video.view(num_clips, 16, *video.shape[1:])
        seqs = preprocess_time(time_idx, num_clips, vs_tokenizer)
        seqs = torch.nn.utils.rnn.pad_sequence(
            seqs,
            batch_first=True,
            padding_value=vs_tokenizer.pad_token_id)
        compress_mask = seqs.ne(vs_tokenizer.pad_token_id)

        question = preprocess_question([question_], vs_tokenizer)
        question = torch.nn.utils.rnn.pad_sequence(
            question,
            batch_first=True,
            padding_value=vs_tokenizer.pad_token_id)
        qs_mask = question.ne(vs_tokenizer.pad_token_id)

        with torch.no_grad():
            with torch.inference_mode():
                with torch.cuda.amp.autocast():
                    similarity, glb, lol = vs_model.forward_grounding(
                        input_ids=seqs.to(device='cuda', non_blocking=True),
                        attention_mask=compress_mask.to(device='cuda', non_blocking=True),
                        images=video.to(dtype=torch.float16, device='cuda', non_blocking=True),
                        qs_ids=question.to(device='cuda', non_blocking=True),
                        qs_mask=qs_mask.to(device='cuda', non_blocking=True))

        lol = lol.view(-1, 64, 1536)
        sele_frames = []
        sele_lol = []
        for i in range(len(frames) // 16):
            if similarity[0][i] > args.vs_thresh:
                sele_frames.extend(frames[i * 16:(i + 1) * 16])
                sele_lol.append(lol[i])
        if len(sele_frames) == 0:
            print('grounding fail!!!')
            sele_frames = frames

        if len(sele_lol) > 0:
            sele_lol = torch.cat(sele_lol, dim=0)

        question = 'Here are some frames of a video. ' + question_
        # options = candidates
        # options_prompt = ''
        # for idx, item in enumerate(options):
        #     idx = chr(65 + idx)
        #     options_prompt += f'{idx}. {item}\n'

        if len(sele_frames) > args.max_frame:
            step = (len(sele_frames) - 1) / (args.max_frame - 1)
            sele_frames = [sele_frames[int(i * step)] for i in range(args.max_frame)]
        img = img_process(sele_frames)

        mid_prompt = 'Question: ' + question
        query = f'[UNUSED_TOKEN_146]user\n{mid_prompt}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\nThe answer is'

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                response = model_gen_withmem(ixc_model, query, img, glb, sele_lol, hd_num=36, do_sample=False, beam=1, max_new_token=1024)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ixc-model-path", type=str, default="internlm-xcomposer2d5-ol-7b")
    parser.add_argument("--max-frame", type=int, default=32)
    parser.add_argument("--vs-thresh", type=float, default=0.2)
    parser.add_argument("--task_path", type=str, required=True, help="Path to the task folder containing data.json and media files")
    args = parser.parse_args()
    # print(args)
    inference(args)