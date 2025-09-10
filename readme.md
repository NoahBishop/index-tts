## 👉🏻 IndexTTS2 👈🏻

<center><h3>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h3></center>

[![IndexTTS2](https://github.com/index-tts/index-tts/blob/main/assets/IndexTTS2_banner.png)](https://github.com/index-tts/index-tts/blob/main/assets/IndexTTS2_banner.png)

<div align="center">
  <a href='https://arxiv.org/abs/2506.21619'>
    <img src='https://img.shields.io/badge/ArXiv-2506.21619-red?logo=arxiv'/>
  </a>
  <br/>
  <a href='https://github.com/index-tts/index-tts'>
    <img src='https://img.shields.io/badge/GitHub-Code-orange?logo=github'/>
  </a>
  <a href='https://index-tts.github.io/index-tts2.github.io/'>
    <img src='https://img.shields.io/badge/GitHub-Demo-orange?logo=github'/>
  </a>
  <br/>
  <a href='https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo'>
    <img src='https://img.shields.io/badge/HuggingFace-Demo-blue?logo=huggingface'/>
  </a>
  <a href='https://huggingface.co/IndexTeam/IndexTTS-2'>
    <img src='https://img.shields.io/badge/HuggingFace-Model-blue?logo=huggingface' />
  </a>
  <br/>
  <a href='https://modelscope.cn/models/IndexTeam/IndexTTS-2'>
    <img src='https://img.shields.io/badge/ModelScope-Model-purple?logo=modelscope'/>
  </a>
</div>


### IndexTTS

A Windows-focused deployment of the IndexTTS project, using a Conda environment for dependency management instead of the UV. This setup is specifically designed for users running Windows with an NVIDIA GPU.

Note: This project is a modified setup of the original IndexTTS repository. All credit for the core model and research goes to the original authors.

#### 🚀 Key Changes from Official Setup

Conda Environment: Uses Anaconda for managing Python packages and dependencies, which is a familiar tool for many Windows users in the ML community.

Using Modelscope download reather than hf

create conda env:

```
conda create -n index-tts -y python=3.10
conda activate index-tts
conda install -c conda-forge ffmpeg
```

clone project and setup env
Note: because 'pip install torch' will install cpu version now
We use '-f https://mirrors.aliyun.com/pytorch-wheels/youcudaversion' find gpu version torch
use 
```bash
git clone https://github.com/NoahBishop/index-tts.git
cd index-tts
pip install torch -f https://mirrors.aliyun.com/pytorch-wheels/cu126/
pip install torchaudio -f https://mirrors.aliyun.com/pytorch-wheels/cu126/
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

download models to local:

```bash
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
modelscope download --model facebook/w2v-bert-2.0 --local_dir models/facebook/w2v-bert-2.0
modelscope download --model amphion/MaskGCT semantic_codec/model.safetensors --local_dir models/amphion/MaskGCT
modelscope download --model iic/speech_campplus_sv_zh-cn_16k-common campplus_cn_common.bin --local_dir models/iic/speech_campplus_sv_zh-cn_16k-common
modelscope download --model nv-community/bigvgan_v2_22khz_80band_256x bigvgan_generator.pt --local_dir models/nv-community/bigvgan_v2_22khz_80band_256x
modelscope download --model nv-community/bigvgan_v2_22khz_80band_256x config.json --local_dir models/nv-community/bigvgan_v2_22khz_80band_256x
```

run:

```bash
where python
"your env python path" webui.py
```

#### 🙏 Acknowledgments

- Original IndexTTS project: https://github.com/index-tts/index-tts/
