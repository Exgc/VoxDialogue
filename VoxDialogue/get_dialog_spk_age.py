import csv
import random
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

import torchaudio
import os
from tqdm import tqdm
import json
from VoxDialogue.tools.asr import pass_or_not, asr
import torchaudio.transforms as T

resample_speech = T.Resample(orig_freq=22050, new_freq=16000)

from pydub import AudioSegment


def generate_wav(cosyvoice_model, insturct_style, spk_info, content_text, file_path, speaker_path, language='中文'):
    if os.path.exists(file_path):
        return
    if speaker_path is None:
        return

    # output = cosyvoice_model.inference_instruct(
    #     content_text,
    #     spk_info,
    #     insturct_style
    # )
    # print(speaker_path)
    speaker_path = os.path.join('/mnt/disk1/chengxize/data/voxceleb_age', speaker_path)
    audio = AudioSegment.from_file(speaker_path, format="m4a")
    speaker_path=speaker_path.replace('.m4a','.wav')
    audio.export(speaker_path, format="wav")
    prompt_speech_16k = load_wav(speaker_path, 16000)
    transcript = asr(speaker_path)[0]['text']
    output = cosyvoice_model.inference_zero_shot(
        content_text, transcript, prompt_speech_16k)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torchaudio.save(file_path, resample_speech(output['tts_speech']), 16000)


def get_age():
    gender_list = {}
    with open('voxceleb_age/voxceleb_gender.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过标题行
        for row in reader:
            if row[2].strip() == 'm':
                gender_list.setdefault(row[0].strip(), 'male')
            else:
                gender_list.setdefault(row[0].strip(), 'female')

    voxceleb_list = {}
    with open('voxceleb_age/voxceleb_age.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过标题行
        for row in reader:
            if row[3].strip() not in gender_list.keys():
                continue
            age = int(float(row[0]))
            gender = gender_list[row[3].strip()]
            voxceleb_list.setdefault(age, {})
            voxceleb_list[age].setdefault(gender, [])
            if os.path.exists(os.path.join('/mnt/disk1/chengxize/data/voxceleb_age', row[3].strip(), row[4])):
                voxceleb_list[age][gender].append(os.path.join(row[3].strip(), row[4]))
    return voxceleb_list


def process_style(style, language='中文'):
    # print(style)
    gender, emotion, age = [_.strip() for _ in style.strip('() ').split(',')]
    speaking_rate = 'normal'
    pitch = 'normal'
    if gender == 'female':
        spk_info = language + '女'
    else:
        spk_info = language + '男'
    insturct_style = f'A {gender} speaker with {pitch} pitch, {speaking_rate} speaking rate, and {emotion} emotion.'
    return spk_info, insturct_style, int(age), gender

def get_dialog_instruct(dialog_dir, json_dir, language='中文', rank=0, shard=10, generate=True, check=False, cosyvoice_dir=None):
    assert cosyvoice_dir is not None, 'cosyvoice_dir is required'
    cosyvoice = CosyVoice(f'{cosyvoice_dir}/CosyVoice-300M')

    voxceleb_list = get_age()
    print(voxceleb_list)
    processed_dialog = {}
    dialog_topic = {}
    con = 0
    dialogs = []
    for json_file in (os.listdir(json_dir)):
        with open(os.path.join(json_dir, json_file)) as f:
            dialogs.extend(json.load(f))

    pbar = tqdm(dialogs)
    for dia_idx, dialog in enumerate(pbar):
        pbar.set_description(f"finish {con} of {dia_idx * 3}")
        try:
            gen_flag = generate
            if dia_idx % shard != rank:
                gen_flag = False
            processed_dialog.setdefault(dialog['aspect_list'].replace(' ', '_'), {})
            dialog_topic.setdefault(dialog['aspect_list'].replace(' ', '_'), 0)
            topic = dialog['aspect_list'].replace(' ', '_')

            dialog_topic[topic] += 1
            try:
                history_turns = dialog['history_turns']
            except Exception as e:
                history_turns = dialog['history turns']
            for i in (range(3)):
                dia_name = 'dialog_%05d_%03d' % (dialog_topic[topic], i)
                for j in (range(2)):
                    speaker = {}
                    speaker.setdefault('A', None)
                    speaker.setdefault('B', None)
                    processed_dialog[topic].setdefault(dia_name, [])
                    for turn in history_turns:
                        # style, content = turn.split(':')
                        item = turn.split(': ')
                        style = item[0]
                        content = ': '.join(item[1:])
                        content = content.strip()
                        speaker_id = style[0]
                        spk_info, insturct_style, age, gender = process_style(style[1:], language=language)
                        if speaker[speaker_id] is None and age in voxceleb_list and gender in voxceleb_list[age]:
                                spk_id = random.choice(voxceleb_list[age][gender])
                                speaker[speaker_id] = os.path.join(spk_id, random.choice(
                                    os.listdir(os.path.join('/mnt/disk1/chengxize/data/voxceleb_age', spk_id))))
                        processed_dialog[topic][dia_name].append({
                            'turn_id': len(processed_dialog[topic][dia_name]),
                            'speaker_id': speaker_id,
                            'insturct_style': insturct_style,
                            'spk_info': spk_info,
                            'content': content,
                            'wav_path': os.path.join(topic.replace(' ', '_'), dia_name,
                                                     '%03d.wav' % (len(processed_dialog[topic][dia_name]) + 1)),
                            'reference_speaker': speaker[speaker_id]
                        })

                        if gen_flag:
                            generate_wav(cosyvoice_model=cosyvoice,
                                         insturct_style=insturct_style,
                                         spk_info=spk_info,
                                         content_text=content,
                                         file_path=os.path.join(dialog_dir, topic.replace(' ', '_'), dia_name,
                                                                '%03d.wav' % len(
                                                                    processed_dialog[topic][dia_name])),
                                         speaker_path=speaker[speaker_id],
                                         language=language)
                    # speaker_id, content = dialog['current_turn'].split(': ')
                    item = dialog['current_turn'].split(': ')
                    speaker_id = item[0]
                    content = ': '.join(item[1:])
                    spk_info, insturct_style, age, gender = process_style(dialog[f'current_turn_style_{i + 1}'],
                                                                          language=language)
                    processed_dialog[topic][dia_name].append({
                        'turn_id': len(processed_dialog[topic][dia_name]),
                        'speaker_id': speaker_id,
                        'insturct_style': insturct_style,
                        'spk_info': spk_info,
                        'content': content,
                        'wav_path': os.path.join(topic.replace(' ', '_'), dia_name,
                                                 '%03d.wav' % (len(processed_dialog[topic][dia_name]) + 1)),
                        'reference_speaker': speaker[speaker_id]
                    })
                    if gen_flag:
                        generate_wav(cosyvoice_model=cosyvoice,
                                     insturct_style=insturct_style,
                                     spk_info=spk_info,
                                     content_text=content,
                                     file_path=os.path.join(dialog_dir, topic.replace(' ', '_'), dia_name,
                                                            '%03d.wav' % (len(processed_dialog[topic][dia_name]))),
                                     speaker_path=speaker[speaker_id],
                                     language=language)

                    response = dialog[f'response_of_current_style_{i + 1}']
                    item = response.split(': ')
                    style = item[0]
                    content = ': '.join(item[1:])
                    content = content.strip()
                    speaker_id = style[0]
                    spk_info, insturct_style, age, gender = process_style(style[1:], language=language)
                    processed_dialog[topic][dia_name].append({
                        'turn_id': len(processed_dialog[topic][dia_name]),
                        'speaker_id': speaker_id,
                        'insturct_style': insturct_style,
                        'spk_info': spk_info,
                        'content': content,
                        'wav_path': os.path.join(topic.replace(' ', '_'), dia_name,
                                                 '%03d.wav' % (len(processed_dialog[topic][dia_name]) + 1)),
                        'reference_speaker': speaker[speaker_id]
                    })
                    if gen_flag:
                        generate_wav(cosyvoice_model=cosyvoice,
                                     insturct_style=insturct_style,
                                     spk_info=spk_info,
                                     content_text=content,
                                     file_path=os.path.join(dialog_dir, topic.replace(' ', '_'), dia_name,
                                                            '%03d.wav' % len(processed_dialog[topic][dia_name])),
                                     speaker_path=speaker[speaker_id],
                                     language=language)
                    try:
                        for inf in processed_dialog[topic][dia_name]:
                            turn_id = inf['turn_id']
                            insturct_style = inf['insturct_style']
                            style = insturct_style.split(' ')
                            assert inf['spk_info'] == processed_dialog[topic][dia_name][turn_id % 2][
                                'spk_info'], f'{turn_id} spk_info inconsistent' + '\t' + inf['spk_info'] + '\t' + \
                                             processed_dialog[topic][dia_name][turn_id % 2]['spk_info']
                            assert style[4] in ['normal', 'low',
                                                'high'], f'{turn_id} pitch {style[4]} is not allowed, {insturct_style}'
                            assert style[6] in ['normal', 'slow',
                                                'fast'], f'{turn_id} speaking rate {style[6]} is not allowed, {insturct_style}'
                            assert style[10] in ['neutral', 'happy', 'sad', 'angry', 'surprised', 'fearful',
                                                 'disgusted'], f'{turn_id} emotion {style[10]} is not allowed, {insturct_style}'
                    except Exception as e:
                        print(dialog)
                        # processed_dialog[topic][dia_name].pop()
                        print(e)
                    if gen_flag or check:
                        if speaker['A'] is not None and speaker['B'] is not None and pass_or_not(dialog_dir,
                                                                                                 processed_dialog[
                                                                                                     topic][dia_name]):
                            con += 1
                            with open(os.path.join(dialog_dir, topic.replace(' ', '_'), dia_name, 'dialog.json'), 'w',
                                      encoding='utf-8') as f:
                                json.dump(processed_dialog[topic][dia_name], f, indent=4)
                            break
                        else:
                            del processed_dialog[topic][dia_name]
        except Exception as e:
            # a=1
            print(e, dialog)

    with open(os.path.join(dialog_dir, 'processed_dialog_0.05.json'), 'w', encoding='utf-8') as f:
        json.dump(processed_dialog, f, indent=4)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Process spoken dialogue dataset.")

    # 添加命令行参数
    parser.add_argument("--json_dir", type=str, default="examples/speaker_info/age",
                        help="Path to the input JSON directory.")
    parser.add_argument("--output_dir", type=str, default="path/to/output/age",
                        help="Path to the output directory.")
    parser.add_argument("--language", type=str, default="英文",
                        help="Language for processing (e.g., English, 中文, etc.).")
    parser.add_argument("--rank", type=int, default=0,
                        help="Rank parameter for distributed processing (default: 0).")
    parser.add_argument("--shard", type=int, default=1,
                        help="Shard parameter for data partitioning (default: 1).")
    parser.add_argument("--check", action="store_true",
                        help="Whether to perform a validation check (default: False).")
    parser.add_argument("--generate", action="store_true",
                        help="Whether to generate new dialogue instructions (default: False).")
    parser.add_argument("--cosyvoice_checkpoints", action="store_true",
                        help="Path to the directory of cosyvoice checkpoints.")
    # 解析命令行参数
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 调用 get_dialog_instruct 函数
    get_dialog_instruct(
        dialog_dir=args.output_dir,
        json_dir=args.json_dir,
        language=args.language,
        generate=args.generate,
        rank=args.rank,
        shard=args.shard,
        check=args.check,
        cosyvoice_dir=args.cosyvoice_checkpoints
    )