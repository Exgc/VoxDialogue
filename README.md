Here’s a polished version of your README.md with improved clarity, grammar, and structure:

# [VoxBench: Can Spoken Dialogue Systems Understand Information Beyond Words?](#)

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

## 1. VoxBench

### Dialogue Performance Evaluation

TODO: Add standardized validation procedures. Specify what kind of input data users can provide to directly obtain results.

### Attribute Examples

| **Category**                     | **Dialogue**                                                                                                                                  |
|:----------------------------------:|-----------------------------------------------------------------------------------------------------------------------------------------------|
| **I. Speaker Information**        |                                                                                                                                               |
| Gender                            | A: Excuse me, I’m looking for a reliable smartphone that suits my needs. **(Man Voice)**                                                      |
|                                   | B: Certainly, *Sir*! ... excellent performance and a long-lasting battery, perfect for *business tasks*.                                       |
| Age                               | A: What dishes do you recommend? **(Young Voice)**                                                                                             |
|                                   | B: Sure! You might enjoy our specialty *beef burger*, it’s fantastic with *fresh juice*.                                                        |
| Accent                            | A: Could you help me with my schedule? **(British Accent)**                                                                                   |
|                                   | B: Of course, I’ll assist you in organizing your *timetable*.                                                                                  |
| Language                          | A: 今天天气怎么样? **(Speaking in Chinese)**                                                                                                  |
|                                   | B: *今天天气晴朗，气温适中，适合外出活动。*                                                                                                      |
| **II. Paralinguistic Information**|                                                                                                                                               |
| Emotion                           | A: Can you recommend a good movie? **(Sad Tone)**                                                                                             |
|                                   | B: I get it. Maybe *Soul*—it’s emotional and might *bring you some comfort*.                                                                  |
| Volume                            | A: Can you tell me how to solve this problem? **(Loud Volume)**                                                                                |
|                                   | B: Could you *speak more quietly*? People around are studying.                                                                                |
| Speed                             | A: I’d like to get the burger, a soda, and also the pasta, please. **(Speaking Quickly)**                                                      |
|                                   | B: Sorry, could you say that *more slowly*? I didn’t catch everything you said.                                                               |
| Fidelity                          | A: Can you tell me the time of the meeting? **(Poor Audio Quality)**                                                                          |
|                                   | B: I’m *having trouble hearing you*. Could you *improve the audio quality*?                                                                   |
| Stress                            | A: I really don't like making sushi. **(Emphasis on "making sushi")**                                                                          |
|                                   | B: Oh, I see. If you don’t like making sushi, what *other types of cooking* do you enjoy?                                                     |
| Non-verbal Expressions           | A: Could you help me move this box? **(With a sigh before speaking)**                                                                          |
|                                   | B: Are you feeling okay? It seems like *you’re really tired*. I can take care of it for you.                                                  |
| **III. Environmental Information**|                                                                                                                                               |
| Audio Events                      | A: What was that sound just now? **(Background sound: airplane engine sound, explosion sound)**                                                |
|                                   | B: That was a loud explosion. It sounded like the plane exploded. Hope no one was hurt.                                                       |
| Music                             | A: Hey, what instrument is this song played on? **(Music: Piano Song, Sad Song)**                                                              |
|                                   | B: It should be the piano, it sounds so sad.                                                                                                  |

## 2. Construct More Dialogue with Information Beyond Text

### Installation

1. **Clone the repository**:

    ```sh
    git clone --recursive https://github.com/Exgc/VoxDialogue.git
    ```

2. **Create a Conda environment**:

    ```sh
    conda create -n cosyvoice python=3.8
    conda activate cosyvoice
    # pynini is required by WeTextProcessing, use conda to install it for compatibility across platforms.
    conda install -y -c conda-forge pynini==2.1.5
    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
    pip install jiwer
    pip install git+https://github.com/m-bain/whisperx.git
    export PYTHONPATH=third_party/Matcha-TTS
    ```

3. **If you encounter compatibility issues with `sox`**:

   ```sh
   sudo apt-get install sox libsox-dev
   ```

### Prepare Checkpoints

Note: The current version of Cosyvoice in VoxDialogue is not the latest. We strongly recommend downloading the following CosyVoice models from Hugging Face:

- `CosyVoice-300M`
- `CosyVoice-300M-SFT`
- `CosyVoice-300M-Instruct`
- `CosyVoice-ttsfrd` resource

To download the models:

   ```sh
   pip install huggingface_hub
   mkdir -p pretrained_models
   huggingface-cli download --resume-download model-scope/CosyVoice-300M --local-dir CosyVoice-300M
   huggingface-cli download --resume-download model-scope/CosyVoice-300M-SFT --local-dir CosyVoice-300M-SFT
   huggingface-cli download --resume-download model-scope/CosyVoice-300M-Instruct --local-dir CosyVoice-300M-Instruct
   ```

### Generate Dialogue

To generate dialogue for different attributes, use the following commands:

- Paralinguistic Information
  - Emotion:
   ```sh
  python get_dialog_acoustic_emotion.py --json_dir PATH/TO/SCRIPTS_For_EMOTION --output_dir PATH/TO/EMOTION_OUTPUT --rank {rank_id} --nshard {shard_num} --generate --check --cosyvoice_checkpoints {cosyvoice_checkpoints_dir}
   ```
  - Fidelity:
     ```sh
     python get_dialog_acoustic_fidelity.py --json_dir PATH/TO/SCRIPTS_For_EMOTION --output_dir PATH/TO/EMOTION_OUTPUT --rank {rank_id} --nshard {shard_num} --generate --check --cosyvoice_checkpoints {cosyvoice_checkpoints_dir}
     ```
  - verbal Expressions:
     ```sh
     python get_dialog_acoustic_non.py --json_dir PATH/TO/SCRIPTS_For_EMOTION --output_dir PATH/TO/EMOTION_OUTPUT --rank {rank_id} --nshard {shard_num} --generate --check --cosyvoice_checkpoints {cosyvoice_checkpoints_dir}
     ```
  - Speed:
     ```sh
     python get_dialog_acoustic_emotion.py --json_dir PATH/TO/SCRIPTS_For_EMOTION --output_dir PATH/TO/EMOTION_OUTPUT --rank {rank_id} --nshard {shard_num} --generate --check --cosyvoice_checkpoints {cosyvoice_checkpoints_dir}
     ```
  - Stress:
     ```sh
    python get_dialog_acoustic_stress.py --json_dir PATH/TO/SCRIPTS_For_EMOTION --output_dir PATH/TO/EMOTION_OUTPUT --rank {rank_id} --nshard {shard_num} --generate --check --cosyvoice_checkpoints {cosyvoice_checkpoints_dir}
    ```
- **Speaker Information**
  - Age:
     ```shell
     python get_dialog_spk_emotion.py --json_dir PATH/TO/SCRIPTS_For_EMOTION --output_dir PATH/TO/EMOTION_OUTPUT --rank {rank_id} --nshard {shard_num} --generate --check --cosyvoice_checkpoints {cosyvoice_checkpoints_dir}
     ```
  - Gender:
     ```shell
     python get_dialog_spk_emotion.py --json_dir PATH/TO/SCRIPTS_For_EMOTION --output_dir PATH/TO/EMOTION_OUTPUT --rank {rank_id} --nshard {shard_num} --generate --check --cosyvoice_checkpoints {cosyvoice_checkpoints_dir}
     ```
  - Language:
    ```shell
    python get_dialog_spk_language.py --json_dir PATH/TO/SCRIPTS_For_EMOTION --output_dir PATH/TO/EMOTION_OUTPUT --rank {rank_id} --nshard {shard_num} --generate --check --cosyvoice_checkpoints {cosyvoice_checkpoints_dir}
    ```


### More Processing for Volume & Fidelity

To adjust volume and fidelity, run the following scripts:

   ``` shell
   python change_volume.py -j JSON_LOG_PATH -r ROOT_DIR
   python reduce_fidelity.py -j JSON_LOG_PATH -r ROOT_DIR
   ```

```JSON_LOG_PATH```: The path to the JSON processing log file, e.g., processed_dialog.json.

```ROOT_DIR```: The root directory of the generated audio files.

## Acknowledgments
	1.	A significant portion of the code is borrowed from CosyVoice.
	2.	We also use code from WhisperX.
