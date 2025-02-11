import torch
from tqdm import tqdm
import json
import os
import whisperx
import gc
from pathlib import Path
import sys
import threading
import numpy as np
import typing as tp
import time
from multiprocessing import Process
from pydub import AudioSegment 
from jiwer import wer  
import string
from pypinyin import pinyin, Style


def to_pinyin(text):
    return ' '.join([i[0] for i in pinyin(text)])

import re
gpu_num = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

def remove_punctuation(s):  
    # # 创建一个转换表，将所有标点符号映射为空字符串  
    # translator = str.maketrans('', '', string.punctuation)  
    # # 使用translate方法应用转换表  
    # cleaned = s.translate(translator)  
    # cleaned=cleaned.lower().replace('\’','').replace('they’re','they were')
    s = re.sub(r'[^\u4e00-\u9fff]', '', s)

    s_en = to_pinyin(s)
    return s_en
    # return cleaned  

def get_text(segments):
    try:
        speaker_id = segments[0]['speaker']
        for segment in segments:
            if speaker_id != segment['speaker']:
                return None
        text = ''
        for segment in segments:
            text += segment["text"]+' '
        return text 
    except:
        return None
        


# model_dir = "/nfs/chengxize.cxz/hub/models--Systran--faster-whisper-large-v3"
model = (whisperx.load_model("large-v3",language = 'zh', device = device, compute_type=compute_type))
diarize_model = (whisperx.DiarizationPipeline(use_auth_token='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', device=device))
model_a, metadata = whisperx.load_align_model(language_code='zh', device=device)
def pass_or_not(audio_path:str, segments):
    torch.cuda.empty_cache() 
    audio1=[]
    audio2=[]
    for i,seg in enumerate(segments):
        if i % 2 ==0:
            audio1.append(os.path.join(audio_path,seg['wav_path']))
        else:
            audio2.append(os.path.join(audio_path,seg['wav_path']))
    audio_path = Path(os.path.dirname(audio1[0]))
    del_flag = True
    # if True:
    if not os.path.exists(os.path.join(audio_path,"A.wav")):
        c1 = AudioSegment.from_file(audio1[0])  
        for file in audio1[1:]:
            t = AudioSegment.from_file(file)
            c1 = c1 + t
        c1.export(os.path.join(os.path.dirname(audio1[0]),"A.wav"), format="wav")

        audio = whisperx.load_audio(os.path.join(audio_path,"A.wav"))
        result = model.transcribe(audio, batch_size=batch_size)
        language = result["language"]
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        textA = get_text(result["segments"])
        textAr=""
        for seg in segments:
            if seg['speaker_id'] == 'A':
                textAr += seg['content']+' '
        
        if textA is None:
            for file in audio1:
                os.remove(file)
            del_flag = False
            os.remove(os.path.join(audio_path,"A.wav"))
        else:
            textA = remove_punctuation(textA)
            textAr = remove_punctuation(textAr)
            werA = wer(textAr, textA)
            if werA > 0.05:
                for file in audio1:
                    os.remove(file)
                del_flag = False
                os.remove(os.path.join(audio_path,"A.wav"))
    
    # if True:
    if not os.path.exists(os.path.join(audio_path,"B.wav")):
        c2 = AudioSegment.from_file(audio2[0])  
        for file in audio2[1:]:
            t = AudioSegment.from_file(file)
            c2 = c2 + t
        c2.export(os.path.join(os.path.dirname(audio1[0]),"B.wav"), format="wav")

        audio = whisperx.load_audio(os.path.join(audio_path,"B.wav"))
        result = model.transcribe(audio, batch_size=batch_size)
        language = result["language"]
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        textB = get_text(result["segments"])
        textBr=""
        for seg in segments:
            if seg['speaker_id'] == 'B':
                textBr += seg['content']+' '
        if textB is None:
            werB = None
            for file in audio2:
                os.remove(file)
            del_flag = False
            os.remove(os.path.join(audio_path,"B.wav"))
        else:
            textB = remove_punctuation(textB)
            textBr = remove_punctuation(textBr)
            werB = wer(textBr, textB) 
            if werB > 0.05:
                for file in audio2:
                    os.remove(file)
                del_flag = False
                os.remove(os.path.join(audio_path,"B.wav"))
                
    return del_flag
