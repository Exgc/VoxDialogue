# [VoxBench: Can Spoken Dialogue Systems Understand Information Beyond Words?]()
<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->


## 1. VoxBench
<hr>

### Dialogue Performance Evaluation



### Attribute Example
| **Category**                       | **Dialogue**                                                                                                              |
|------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| **I. Speaker Information**         |                                                                                                                           |
| Gender                             | A: Excuse me, I’m looking for a reliable smartphone that suits my needs. **(Man Voice)**                                  |
|                                    | B: Certainly, *Sir*! ... excellent performance and a long-lasting battery, perfect for *business tasks*.                  |
| Age                                | A: What dishes do you recommend? **(Young Voice)**                                                                        |
|                                    | B: Sure! You might enjoy our specialty *beef burger*, it’s fantastic with *fresh juice*.                                  |
| Accent                             | A: Could you help me with my schedule? **(British Accent)**                                                               |
|                                    | B: Of course, I’ll assist you in organizing your *timetable*.                                                             |
| Language                           | A: 今天天气怎么样? **(Speaking in Chinese)**                                                                                     |
|                                    | B: *今天天气晴朗，气温适中，适合外出活动。*                                                                                                  |
| **II. Paralinguistic Information** |                                                                                                                           |
| Emotion                            | A: Can you recommend a good movie? **(Sad Tone)**                                                                         |
|                                    | B: I get it. Maybe *Soul*—it’s emotional and might *bring you some comfort*.                                              |
| Volume                             | A: Can you tell me how to solve this problem? **(Loud Volume)**                                                           |
|                                    | B: Could you *speak more quietly*? People around are studying.                                                            |
| Speed                              | A: I’d like to get the burger, a soda, and also the pasta, please. **(Speaking Quickly)**                                 |
|                                    | B: Sorry, could you say that *more slowly*? I didn’t catch everything you said.                                           |
| Fidelity                           | A: Can you tell me the time of the meeting? **(Poor Audio Quality)**                                                      |
|                                    | B: I’m *having trouble hearing you*. Could you *improve the audio quality*?                                               |
| Stress                             | A: I really don't like making sushi. **(Emphasis on "making sushi")**                                                     |
|                                    | B: Oh, I see. If you don’t like making sushi, what *other types of cooking* do you enjoy?                                 |
| Non-verbal Expressions             | A: Could you help me move this box? **(With a sigh before speaking)**                                                     |
|                                    | B: Are you feeling okay? It seems like *you’re really tired*. I can take care of it for you.                              |
| **III. Environmental Information** |                                                                                                                           |
| Audio Events                       | A: What was that sound just now? **(Background sound: airplane engine sound, explosion sound)**                           |
|                                    | B: I’m sorry, it’s *hard to hear you with all that noise*. Could you repeat the question or maybe move to a quieter spot? |
| Audio Events                       | That was a loud explosion. It sounded like the plane exploded. Hope no one was hurt.                                      |
| Music                              | A: Hey, what instrument is this song played on? **(Music: Piano Song, Sad Song)**                                         |
|                                    | B: It should be the piano, it sounds so sad.                                                                              |

## Construct More Dialogue with Information Beyond Text.
<hr>

### Install


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
export PYTHONPATH=third_party/Matcha-TTS

# If you encounter sox compatibility issues
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel
```

### Prepare Checkpoint

Note that, the version of Cosyvoice in current Voxdialogue is not the latest. We strongly recommend that you download CosyVoice from huggingface: `CosyVoice-300M` `CosyVoice-300M-SFT` `CosyVoice-300M-Instruct` model and `CosyVoice-ttsfrd` resource:


``` sh
# git模型下载，请确保已安装git lfs
pip install huggingface
mkdir -p pretrained_models
huggingface-cli download --resume-download model-scope/CosyVoice-300M --local-dir CosyVoice-300M
huggingface-cli download --resume-download model-scope/CosyVoice-300M-SFT --local-dir CosyVoice-300M-SFT
huggingface-cli download --resume-download model-scope/CosyVoice-300M-Instruct --local-dir CosyVoice-300M-Instruct
```

### generate the Dialogue

``` sh
python get_

```


### More Process for Volume & Fidelity

Run the following scripts:

``` sh
python change_volume.py -j JSON_LOG_PATH -r ROOT_DIR
python reduce_fidelity.py -j JSON_LOG_PATH -r ROOT_DIR
```

`JSON_LOG_PATH`: The json process log file, i.e. corresponding `processed_dialog.json`.

`ROOT_DIR`: The root directory of generated audio files

## Acknowledge

1. We borrowed a lot of code from [CosyVoice](https://github.com/FunAudioLLM/CosyVoice).
2. We borrowed a lot of code from [WhisperX](https://github.com/m-bain/whisperX).
