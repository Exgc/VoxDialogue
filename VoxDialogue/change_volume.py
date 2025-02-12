from pydub import AudioSegment
import math
import json
import os
from tqdm import tqdm
import argparse
import logging

os.makedirs('log', exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    filename='log/change_volume.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

# 按倍率调整音量
def change_volume_by_factor(audio, factor):
    # 计算对应的dB增益
    gain_dB = 20 * math.log10(factor)
    # 应用增益
    return audio.apply_gain(gain_dB)

def change_volume(audio_path, output_path, add_db=0, rate=1.0, change_type: str='add'):
    '''
    type: 'add' or 'rate'
    '''
    # 加载音频文件
    audio = AudioSegment.from_file(audio_path)

    if change_type == 'add':
        # 增加音量
        audio = audio.apply_gain(add_db)
        audio.export(output_path, format="wav")
        logger.info(f"Add {add_db} dB to file {audio_path} and save to {output_path}")
    elif change_type == 'rate':
        audio = change_volume_by_factor(audio, rate)
        audio.export(output_path, format="wav")
        logger.info(f"Change volume of file {audio_path} to {rate} and save to {output_path}")
    else:
        raise ValueError(f"Invalid type: {change_type}")
    
def run_change_volume(json_log, root_dir, change_type='rate', loud_rate=8.0, low_rate=0.5, add_db=0):
    process_log = json.load(open(json_log))
    dialog_names = list(process_log.keys())
    dialog_names = tqdm(dialog_names, leave=True)
    for dialog_name in dialog_names:
        dialog_dir = os.path.join(root_dir, dialog_name)
        logger.info(f"Processing dialog {dialog_dir}")
        print(f"Processing dialog \033[32m{dialog_dir}\033[0m")
        dialogs = process_log.get(dialog_name)
        dialog_ids = list(dialogs.keys())
        for dialog_id in dialog_ids:
            dialog = dialogs.get(dialog_id)
            for turn in dialog:
                # 识别volume
                volume_flag = '' # loud is True, low is False
                if 'loud' in turn['insturct_style'][-10:]:
                    volume_flag = 'loud'
                elif 'low' in turn['insturct_style'][-10:]:
                    volume_flag = 'low'
                if volume_flag == 'loud':
                    audio_file = turn['wav_path']
                    audio_file = os.path.join(root_dir, audio_file)
                    if os.path.exists(audio_file):
                        audio_name = os.path.basename(audio_file)
                        parent_dir = os.path.dirname(audio_file)
                        loud_audio_name = audio_name.replace('.wav', '_loud.wav')
                        loud_audio_path = os.path.join(parent_dir, loud_audio_name)
                        if not os.path.exists(loud_audio_path):
                            change_volume(audio_file, loud_audio_path, rate=loud_rate, add_db=add_db, change_type=change_type)
                        else:
                            logger.info(f"File {loud_audio_path} already exists, skip")
                    else:
                        logger.info(f"File {audio_file} not exists, skip")
                        continue
                elif volume_flag == 'low':
                    audio_file = turn['wav_path']
                    audio_file = os.path.join(root_dir, audio_file)
                    if os.path.exists(audio_file):
                        audio_name = os.path.basename(audio_file)
                        parent_dir = os.path.dirname(audio_file)
                        low_audio_name = audio_name.replace('.wav', '_low.wav')
                        low_audio_path = os.path.join(parent_dir, low_audio_name)
                        if not os.path.exists(low_audio_path):
                            change_volume(audio_file, low_audio_path, rate=low_rate, add_db=add_db, change_type=change_type)
                        else:
                            logger.info(f"File {low_audio_path} already exists, skip")
                    else:
                        logger.info(f"File {audio_file} not exists, skip")
                        continue
                else:
                    continue
            logger.info(f"Processed dialog {dialog_name} with id {dialog_id}")
        logger.info(f"Processed dialog {dialog_name}")
        print(f"Processed dialog \033[32m{dialog_name}\033[0m")
    print("\033[32mAll dialogs processed!!!\033[0m")

if __name__ == "__main__":
    add_db = 30
    loud_rate = 8
    low_rate = 0.5
    change_type = 'rate'
    parser = argparse.ArgumentParser()
    parser.add_argument('-j','--json_log_path', type=str, required=True, help='process json log path')
    parser.add_argument('-r','--root_dir', type=str, required=True, help='root directory of audio files')
    args = parser.parse_args()
    json_log_path = args.json_log_path
    root_dir = args.root_dir
    # json_log_path = '/mnt/disk1/chengxize/data/VoxDialog/acoustic_information/volume/processed_dialog.json'
    # root_dir = '/mnt/disk1/chengxize/data/VoxDialog/acoustic_information/volume/'
    run_change_volume(json_log_path, root_dir=root_dir, change_type=change_type, loud_rate=loud_rate, low_rate=low_rate, add_db=add_db)