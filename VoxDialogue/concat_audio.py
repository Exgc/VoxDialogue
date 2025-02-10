import os
from pydub import AudioSegment


def concat_audio(audio_file, dialog_file, save_path):
    # 加载音频文件
    # 在第一个dialog speech之前添加audio
    audio = AudioSegment.from_file(audio_file)  # 替换为音频A的路径
    dialog = AudioSegment.from_file(dialog_file)  # 替换为音频B的路径
    combined_audio = audio + dialog
    combined_audio.export(save_path, format="mp3")  # 替换为输出文件路径和格式


def repeat_audio(audio_segment, total_duration_ms):
    audio_length_ms = len(audio_segment)
    repeat_count = (total_duration_ms // audio_length_ms) + 1
    repeated_audio = audio_segment * repeat_count
    return repeated_audio[:total_duration_ms]


def concatenate_and_split(background_path, dialogues_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    """将背景音频重复拼接，然后添加每段对话，最后切分音频"""
    # 计算所有对话的总时长
    background = AudioSegment.from_file(background_path)  # 替换为音频A的路径
    dialogues = [AudioSegment.from_file(dialogue_path) for dialogue_path in dialogues_path]
    total_dialogue_duration = sum(len(dialogue) for dialogue in dialogues)

    # 将背景音频重复，直到其时长等于或超过所有对话的总时长
    repeated_background = repeat_audio(background, total_dialogue_duration)

    combined_audio = AudioSegment.silent(duration=0)  # 初始化为空音频
    current_position = 0

    # 拼接背景音频和每段对话
    for turn_id, dialogue in enumerate(dialogues):
        dialogue_duration = len(dialogue)

        # 截取与当前对话时长相匹配的背景音频段
        background_segment = repeated_background[current_position:current_position + dialogue_duration]

        # 将背景音频和对话顺序拼接
        combined_segment = background_segment.overlay(dialogue)

        # 更新当前位置
        current_position += dialogue_duration
        combined_segment.export(os.path.join(save_dir, f'{turn_id}.mp3'), format="mp3")


if __name__ == '__main__':
    print(1)
    # concat_audio('/nfs/chengxize.cxz/projects/ShareChatX/CosyVoice/zero_shot_prompt.wav',
    #              '/nfs/chengxize.cxz/projects/ShareChatX/CosyVoice/instruct.wav',
    #              '/nfs/chengxize.cxz/projects/ShareChatX/CosyVoice/test.wav')

    
    # concatenate_and_split('/nfs/chengxize.cxz/projects/ShareChatX/CosyVoice/instruct.wav',
    #                     ['/nfs/chengxize.cxz/projects/ShareChatX/CosyVoice/zero_shot_prompt.wav','/nfs/chengxize.cxz/projects/ShareChatX/CosyVoice/zero_shot_prompt.wav'],
    #                     '/nfs/chengxize.cxz/projects/ShareChatX/CosyVoice/save')
