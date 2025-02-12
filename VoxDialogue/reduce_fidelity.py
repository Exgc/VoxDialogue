from pydub import AudioSegment
import json
import os
from tqdm import tqdm
import argparse

def reduce_fidelity(file_path, output_path, sample_rate=4000):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_frame_rate(16000)
    audio.export(output_path, format="wav")
    print(f"Reduced fidelity of \033[32m{file_path}\033[0m to \033[32m{output_path}\033[0m with sample rate {sample_rate/1000:.1f}kHz")

def run_reduce_fidelity(json_log, root_dir, sample_rate=4000):
    process_log = json.load(open('/mnt/disk1/chengxize/data/VoxDialog/acoustic_information/fidelity/processed_dialog.json'))
    dialog_names = list(process_log.keys())
    
    dialog_names = tqdm(dialog_names)
    for dialog_name in dialog_names:
        dialog_dir = os.path.join(root_dir, dialog_name)
        print(f"Processing dialog \033[32m{dialog_dir}\033[0m")
        dialogs = process_log.get(dialog_name)
        dialog_ids = list(dialogs.keys())
        for dialog_id in dialog_ids:
            dialog = dialogs.get(dialog_id)
            for turn in dialog:
                # 识别fidelity
                fidelity_flag = True # good is True, bad is False
                if 'good' in turn['insturct_style'][-5:]:
                    fidelity_flag = True
                elif 'bad' in turn['insturct_style'][-5:]:
                    fidelity_flag = False
                if fidelity_flag == False:
                    # reduce fidelity
                    audio_file = turn['wav_path']
                    audio_file = os.path.join(root_dir, audio_file)
                    if os.path.exists(audio_file):
                        audio_name = os.path.basename(audio_file)
                        parent_dir = os.path.dirname(audio_file)
                        bad_audio_name = audio_name.replace('.wav', '_bad.wav')
                        bad_audio_path = os.path.join(parent_dir, bad_audio_name)
                        if not os.path.exists(bad_audio_path):
                            reduce_fidelity(audio_file, bad_audio_path, 4000)
                        else:
                            print(f"File \033[32m{bad_audio_path}\033[0m already exists, skip")
                    else:
                        print(f"File \033[31m{audio_file}\033[0m not exists, skip")
                        continue
            print(f"Processed dialog \033[32m{dialog_name}\033[0m with id \033[32m{dialog_id}\033[0m")
        print(f"Processed dialog \033[32m{dialog_name}\033[0m")
    print("\033[32mAll dialogs processed!!!\033[0m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j','--json_log_path', type=str, required=True, help='process json log path')
    parser.add_argument('-r','--root_dir', type=str, required=True, help='root directory of audio files')
    args = parser.parse_args()
    json_log_path = args.json_log_path
    root_dir = args.root_dir
    # json_log_path = '/mnt/disk1/chengxize/data/VoxDialog/acoustic_information/fidelity/processed_dialog.json'
    # root_dir = '/mnt/disk1/chengxize/data/VoxDialog/acoustic_information/fidelity'
    run_reduce_fidelity(json_log_path, root_dir, 4000)