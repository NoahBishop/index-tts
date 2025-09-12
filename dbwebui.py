import json
import os
import sys
import threading
import time
import warnings
import torch
import numpy as np
import soundfile as sf
import io

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--use_deepspeed", action="store_true", default=False, help="Use Deepspeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use cuda kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="Auto")
MODE = 'local'
tts = IndexTTS2(model_dir=cmd_args.model_dir,
                cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
                use_fp16=cmd_args.fp16,
                use_deepspeed=cmd_args.use_deepspeed,
                use_cuda_kernel=cmd_args.cuda_kernel,
                )
# 支持的语言列表
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES = [i18n("与音色参考音频相同"),
                i18n("使用情感参考音频"),
                i18n("使用情感向量控制"),
                i18n("使用情感描述文本控制")]
os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)

# 创建数据库目录
DATABASE_DIR = os.path.join(current_dir, "database")
os.makedirs(DATABASE_DIR, exist_ok=True)
DATABASE_FILE = os.path.join(DATABASE_DIR, "voice_database.pt")

# 初始化音色数据库
if os.path.exists(DATABASE_FILE):
    voice_database = torch.load(DATABASE_FILE)
else:
    voice_database = {}
    torch.save(voice_database, DATABASE_FILE)

# 数据库操作锁
db_lock = threading.Lock()

def save_database():
    """保存数据库到文件"""
    with db_lock:
        torch.save(voice_database, DATABASE_FILE)

def add_to_database(name, audio_path):
    print(name)
    print(audio_path)
    """添加新音色到数据库"""
    if not name:
        raise gr.Error(i18n("名称不能为空"))
    
    if name in voice_database:
        raise gr.Error(i18n("名称已存在，请使用唯一名称"))
    
    # 读取音频文件并转换为二进制数据
    try:
        # 读取音频数据
        audio_data, sample_rate = sf.read(audio_path)
        
        # 将音频数据转换为二进制格式
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV')
        audio_bytes = buffer.getvalue()
        
        # 添加到数据库
        voice_database[name] = {
            'audio_bytes': audio_bytes,
            'sample_rate': sample_rate
        }
        save_database()
        return i18n("保存成功")
    except Exception as e:
        return f"{i18n('保存失败')}: {str(e)}"

def remove_from_database(name):
    """从数据库删除音色"""
    if name not in voice_database:
        raise gr.Error(i18n("名称不存在"))
    
    # 更新数据库
    del voice_database[name]
    save_database()
    return i18n("删除成功")

def load_from_folder(path):
    for root, dirs, files in os.walk(path):
        folder_name = os.path.basename(root)

        # 在当前文件夹中筛选.wav文件
        wav_files = [f for f in files if f.lower().endswith('.wav')]
        
        if wav_files:
            # 按文件名排序并获取第一个
            wav_files.sort()
            first_wav = wav_files[0]
            
            # 构建完整的文件路径
            full_path = os.path.join(root, first_wav)
            
            # 将完整路径传递给数据库函数
            add_to_database(folder_name, full_path)

def get_audio_path(name):
    """从数据库获取音频文件路径（临时文件）"""
    if name not in voice_database:
        return None
    
    # 从数据库获取音频数据
    audio_data = voice_database[name]
    audio_bytes = audio_data['audio_bytes']
    sample_rate = audio_data['sample_rate']
    
    # 创建临时文件
    temp_file = os.path.join(DATABASE_DIR, f"temp_{name}.wav")
    with open(temp_file, 'wb') as f:
        f.write(audio_bytes)
    
    return temp_file

def gen_single(emo_control_method, voice_name, text,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text, emo_random,
               max_text_tokens_per_segment=120,
                *args, progress=gr.Progress()):
    output_path = None
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    
    # 从数据库获取音色参考音频
    if voice_name not in voice_database:
        raise gr.Error(i18n("选择的音色不存在"))
    
    # 获取音频文件路径（临时文件）
    prompt = get_audio_path(voice_name)
    
    # set gradio progress
    tts.gr_progress = progress
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:
        emo_ref_path = None
        emo_weight = 1.0
    if emo_control_method == 1:
        emo_weight = emo_weight
    if emo_control_method == 2:
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec_sum = sum([vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8])
        if vec_sum > 1.5:
            gr.Warning(i18n("情感向量之和不能超过1.5，请调整后重试。"))
            return
    else:
        vec = None

    if emo_text == "":
        # erase empty emotion descriptions; `infer()` will then automatically use the main prompt
        emo_text = None

    print(f"Emo control mode:{emo_control_method},vec:{vec}")
    output = tts.infer(spk_audio_prompt=prompt, text=text,
                       output_path=output_path,
                       emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                       emo_vector=vec,
                       use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                       verbose=cmd_args.verbose,
                       max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                       **kwargs)
    os.remove(prompt)
    return gr.update(value=output,visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

def update_voice_dropdown():
    """更新音色下拉框选项"""
    return gr.Dropdown(choices=list(voice_database.keys()), value=list(voice_database.keys())[0] if voice_database else None)

with gr.Blocks(title="IndexTTS Demo") as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h2>
<p align="center">
<a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
</p>
    ''')
    
    with gr.Tab(i18n("音频生成")):
        with gr.Row():
            with gr.Column():
                # 音色选择下拉框
                voice_dropdown = gr.Dropdown(
                    label=i18n("选择音色"),
                    choices=list(voice_database.keys()),
                    value=list(voice_database.keys())[0] if voice_database else None
                )
                
                input_text_single = gr.TextArea(label=i18n("文本"),key="input_text_single", placeholder=i18n("请输入目标文本"), info=f"{i18n('当前模型版本')}{tts.model_version or '1.0'}")
                gen_button = gr.Button(i18n("生成语音"), key="gen_button",interactive=True)
            output_audio = gr.Audio(label=i18n("生成结果"), visible=True,key="output_audio")
        with gr.Accordion(i18n("功能设置")):
            # 情感控制选项部分
            with gr.Row():
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES,
                    type="index",
                    value=EMO_CHOICES[0],label=i18n("情感控制方式"))
        # 情感参考音频部分
        with gr.Group(visible=False) as emotion_reference_group:
            with gr.Row():
                emo_upload = gr.Audio(label=i18n("上传情感参考音频"), type="filepath")

            with gr.Row():
                emo_weight = gr.Slider(label=i18n("情感权重"), minimum=0.0, maximum=1.6, value=0.8, step=0.01)

        # 情感随机采样
        with gr.Row():
            emo_random = gr.Checkbox(label=i18n("情感随机采样"),value=False,visible=False)

        # 情感向量控制部分
        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label=i18n("喜"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec2 = gr.Slider(label=i18n("怒"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec3 = gr.Slider(label=i18n("哀"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec4 = gr.Slider(label=i18n("惧"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                with gr.Column():
                    vec5 = gr.Slider(label=i18n("厌恶"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec6 = gr.Slider(label=i18n("低落"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec7 = gr.Slider(label=i18n("惊喜"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec8 = gr.Slider(label=i18n("平静"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)

        with gr.Group(visible=False) as emo_text_group:
            with gr.Row():
                emo_text = gr.Textbox(label=i18n("情感描述文本"), placeholder=i18n("请输入情绪描述（或留空以自动使用目标文本作为情绪描述）"), value="", info=i18n("例如：高兴，愤怒，悲伤等"))

        with gr.Accordion(i18n("高级生成参数设置"), open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"**{i18n('GPT2 采样设置')}** _{i18n('参数会影响音频多样性和生成速度详见')} [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info=i18n("是否进行采样"))
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info=i18n("生成Token最大数量，过小导致音频被截断"), key="max_mel_tokens")
                with gr.Column(scale=2):
                    gr.Markdown(f'**{i18n("分句设置")}** _{i18n("参数会影响音频质量和生成速度")}_')
                    with gr.Row():
                        initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                        max_text_tokens_per_segment = gr.Slider(
                            label=i18n("分句最大Token数"), value=initial_value, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_segment",
                            info=i18n("建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高"),
                        )
                    with gr.Accordion(i18n("预览分句结果"), open=True) as segments_settings:
                        segments_preview = gr.Dataframe(
                            headers=[i18n("序号"), i18n("分句内容"), i18n("Token数")],
                            key="segments_preview",
                            wrap=True,
                        )
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
            ]

    with gr.Tab(i18n("音色数据库管理")):
        with gr.Row():
            with gr.Column():
                # 删除音色
                delete_dropdown = gr.Dropdown(
                    label=i18n("选择要删除的音色"),
                    choices=list(voice_database.keys()),
                    value=list(voice_database.keys())[0] if voice_database else None
                )
                delete_button = gr.Button(i18n("删除音色"))
                delete_result = gr.Textbox(label=i18n("操作结果"), interactive=False)
                
                # 删除操作
                def delete_voice(name):
                    try:
                        result = remove_from_database(name)
                        return result, update_voice_dropdown(), update_voice_dropdown()
                    except Exception as e:
                        return str(e), update_voice_dropdown(), update_voice_dropdown()
                
                delete_button.click(
                    delete_voice,
                    inputs=delete_dropdown,
                    outputs=[delete_result, voice_dropdown, delete_dropdown]
                )

            with gr.Column():
                # 上传新音色
                upload_audio = gr.Audio(label=i18n("上传音色参考音频"), sources=["upload", "microphone"], type="filepath")
                voice_name_input = gr.Textbox(label=i18n("音色名称"), placeholder=i18n("输入唯一名称"))
                save_button = gr.Button(i18n("保存到数据库"))
                save_result = gr.Textbox(label=i18n("操作结果"), interactive=False)
                
                # 保存操作
                def save_voice(audio_path, name):
                    try:
                        result = add_to_database(name, audio_path)
                        return result, update_voice_dropdown(), update_voice_dropdown()
                    except Exception as e:
                        return str(e), update_voice_dropdown(), update_voice_dropdown()
                
                save_button.click(
                    save_voice,
                    inputs=[upload_audio, voice_name_input],
                    outputs=[save_result, voice_dropdown, delete_dropdown]
                )


    def on_input_text_change(text, max_text_tokens_per_segment):
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)

            segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
            data = []
            for i, s in enumerate(segments):
                segment_str = ''.join(s)
                tokens_count = len(s)
                data.append([i, segment_str, tokens_count])
            return {
                segments_preview: gr.update(value=data, visible=True, type="array"),
            }
        else:
            df = pd.DataFrame([], columns=[i18n("序号"), i18n("分句内容"), i18n("Token数")])
            return {
                segments_preview: gr.update(value=df),
            }
    def on_method_select(emo_control_method):
        if emo_control_method == 1:
            return (gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )
        elif emo_control_method == 2:
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False)
                    )
        elif emo_control_method == 3:
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        else:
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )

    emo_control_method.select(on_method_select,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group,
                 emo_random,
                 emotion_vector_group,
                 emo_text_group]
    )

    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )
    max_text_tokens_per_segment.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    gen_button.click(gen_single,
                     inputs=[emo_control_method, voice_dropdown, input_text_single, emo_upload, emo_weight,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                             emo_text, emo_random,
                             max_text_tokens_per_segment,
                             *advanced_params,
                     ],
                     outputs=[output_audio])

if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port)