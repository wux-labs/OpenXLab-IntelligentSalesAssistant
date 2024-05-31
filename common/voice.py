import streamlit as st
import os

import requests
import whisper
from melo.api import TTS
from datetime import datetime

from utils import is_cuda_available, is_cuda_enough

def init_voice_config_form():
    model_type = st.selectbox("语音识别", key="config_voice_model_type", options=["本地", "远程"])
    model_size_option = ["base", "small", "medium"]
    if is_cuda_available() and is_cuda_enough(40950):
        model_size_option.append("large")
    model_size = st.selectbox("模型大小", key="config_voice_model_size", options=model_size_option)


def load_whisper_model(model_name:str="medium"):
    if f"whisper_model_{model_name}" not in st.session_state.keys():
        st.session_state[f"whisper_model_{model_name}"] = whisper.load_model(model_name, download_root="models/whisper")
    return st.session_state[f"whisper_model_{model_name}"]


def voice_to_text_remote(localdir, filename):
    request_url="https://api.deepinfra.com/v1/inference/openai/whisper-large"
    response = requests.post(url=request_url, files={'audio': (filename, open(f"{localdir}/{filename}", 'rb'))})
    return response.json().get("text")


def voice_to_text_local(localdir, filename):
    model = load_whisper_model(st.session_state.config_voice_model_size)
    result = model.transcribe(f"{localdir}/{filename}")
    return result["text"]


@st.cache_resource
def load_melo_model():
    model = TTS(language="ZH", device="auto")
    speaker_ids = model.hps.data.spk2id
    return model, speaker_ids


def text_to_voice(voice_text):
    speaker_model, speaker_ids = load_melo_model()
    localdir = f"users/{st.session_state.username}/voices"
    os.makedirs(localdir, exist_ok=True)
    output_path = f"{localdir}/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.mp3"
    speaker_model.tts_to_file(voice_text, speaker_ids['ZH'], output_path, speed=0.7)
    return output_path
