import streamlit as st
import base64
from datetime import datetime
from io import BytesIO
import os, sys
import json
import requests
import gc

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from common.chat import translate_to_english

def init_draw_config_form():
    model = st.selectbox("Model", key="config_image_model", options=["stabilityai/stable-diffusion-2-1", "runwayml/stable-diffusion-v1-5", "prompthero/openjourney", "stability-ai/sdxl"])

    col1, col2 = st.columns(2)
    with col1:
        width = st.slider("Width", key="config_image_width", min_value=1, max_value=2048, value=512, step=1)
        steps = st.slider("Steps", key="config_image_steps", min_value=1, max_value=100, value=20, step=1)
        sampler_name = st.selectbox("Sampler", key="config_image_sampler_name",
                                    options=["DDIM", "DPM++ 2M Karras", "DPM++ SDE Karras", "Heun"])
    with col2:
        height = st.slider("Height", key="config_image_height", min_value=1, max_value=2048, value=512, step=1)
        cfg_scale = st.slider("CFG_Scale", key="config_image_cfg_scale", min_value=1, max_value=32, value=7,
                            step=1)
        seed = st.number_input("Seed", key="config_image_seed", step=1, value=1)

    negative_prompt = st.text_input("Negative Prompt", key="config_image_negative_prompt")


def load_stable_diffusion(model):
    if "stable_diffusion_model" not in st.session_state.keys():
        st.session_state["stable_diffusion_model"] = StableDiffusionPipeline.from_pretrained(f"models/{model}", torch_dtype=torch.float16)
        st.session_state["stable_diffusion_model"].scheduler = DPMSolverMultistepScheduler.from_config(st.session_state["stable_diffusion_model"].scheduler.config)
        st.session_state["stable_diffusion_model"] = st.session_state["stable_diffusion_model"].to("cuda")
    return st.session_state["stable_diffusion_model"]


def save_draw_image(user_input_text):
    localdir = f"users/{st.session_state.username}/images"
    os.makedirs(localdir, exist_ok=True)
    localfile = f"{localdir}/{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.png"
    succ = False

    model = st.session_state.config_image_model
    prompt = translate_to_english(user_input_text)
    params = {
        "prompt": prompt,
        "width": st.session_state.config_image_width,
        "height": st.session_state.config_image_height,
        "steps": st.session_state.config_image_steps,
        "cfg_scale": st.session_state.config_image_cfg_scale,
        "seed": st.session_state.config_image_seed,
        "sampler_name": st.session_state.config_image_sampler_name,
        "negative_prompt": st.session_state.config_image_negative_prompt,  # 'nsfw',
    }

    if model == "stabilityai/stable-diffusion-2-1":
        pipe = load_stable_diffusion(model)
        output = pipe(prompt, 
                    #   width=st.session_state.config_image_width, 
                    #   height=st.session_state.config_image_height, 
                      num_inference_steps=st.session_state.config_image_steps, 
                      output_type="pil").images[0]
        output.save(localfile)
        succ = True
        gc.collect()
    elif model == "stability-ai/sdxl":
        request_url=f"{os.environ.get('DRAW_INFERENCE_BASE')}/stability-ai/sdxl"
        response = requests.post(url=request_url, data=json.dumps({"input" : params}))
        if response.status_code == 200:
            response_json = response.json()
            if response_json.get("output"):
                image_url = response_json.get("output")[0]
                from urllib.request import urlretrieve
                urlretrieve(image_url, localfile)
                succ = True
            else:
                raise Exception(response_json.get("error")).with_traceback(sys.exc_info()[2])
    else:
        request_url=f"{os.environ.get('DRAW_INFERENCE_BASE')}/{model}"
        response = requests.post(url=request_url, data=json.dumps(params))
        response_json = response.json()
        if response.status_code == 200:
            if response_json.get("images"):
                image_str = response_json.get("images")[0][22:]
                decoded_bytes = base64.b64decode(image_str)
                bytes_io = BytesIO(decoded_bytes)
                with open(localfile, 'wb') as f:
                    f.write(decoded_bytes)
                succ = True
            else:
                raise Exception(response_json.get("detail")).with_traceback(sys.exc_info()[2])
        else:
            raise Exception(response_json.get("detail")).with_traceback(sys.exc_info()[2])
    
    return localfile if succ else None
