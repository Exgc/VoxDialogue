# VoxBench

## VoxDialogue(DataSet)

### Attribute Example
- ***Speaker Information***

  | **Attribute** | **Example** |
  |---------------|-------------|
  | Gender        |             |
  | Age           |             |
  | Language      |             |
  | Accent        |             |

- ***Paralinguistic Information***
    
  | **Attribute** | **Example** |
  |---------------|-------------|
  | Gender        |             |
  | Age           |             |
  | Language      |             |
  | Accent        |             |

### Dialogue Dataset Generation

#### Install

**Clone and install**

- Clone the repo
``` sh
git clone --recursive https://github.com/Exgc/VoxDialogue.git
```

- Create Conda env:

``` sh
conda create -n cosyvoice python=3.8
conda activate cosyvoice
# pynini is required by WeTextProcessing, use conda to install it as it can be executed on all platform.
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
pip install jiwer
pip install git+https://github.com/m-bain/whisperx.git

# If you encounter sox compatibility issues
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel
```

**Model download**

Note that, the version of Cosyvoice in current Voxdialogue is not the latest. We strongly recommend that you download CosyVoice from huggingface: `CosyVoice-300M` `CosyVoice-300M-SFT` `CosyVoice-300M-Instruct` model and `CosyVoice-ttsfrd` resource:


``` sh
# git模型下载，请确保已安装git lfs
pip install huggingface
mkdir -p pretrained_models
huggingface-cli download --resume-download model-scope/CosyVoice-300M --local-dir CosyVoice-300M
huggingface-cli download --resume-download model-scope/CosyVoice-300M-SFT --local-dir CosyVoice-300M-SFT
huggingface-cli download --resume-download model-scope/CosyVoice-300M-Instruct --local-dir CosyVoice-300M-Instruct
```

**Basic Usage**

For zero_shot/cross_lingual inference, please use `CosyVoice-300M` model.
For sft inference, please use `CosyVoice-300M-SFT` model.
For instruct inference, please use `CosyVoice-300M-Instruct` model.
First, add `third_party/Matcha-TTS` to your `PYTHONPATH`.

``` sh
export PYTHONPATH=third_party/Matcha-TTS
```


## Acknowledge

1. We borrowed a lot of code from [FunASR](https://github.com/modelscope/FunASR).
2. We borrowed a lot of code from [FunCodec](https://github.com/modelscope/FunCodec).
3. We borrowed a lot of code from [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS).
4. We borrowed a lot of code from [AcademiCodec](https://github.com/yangdongchao/AcademiCodec).
5. We borrowed a lot of code from [WeNet](https://github.com/wenet-e2e/wenet).

