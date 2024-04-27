import os
import streamlit as st


def download_sales_chat_model():
    with st.spinner("正在下载智能营销助手模型，请稍等..."):
        os.system(f'rm -rf models/ai-labs/sales-chat-1_8b')
        os.system(f'GIT_LFS_SKIP_SMUDGE=1 git clone https://code.openxlab.org.cn/AI-Labs/sales-chat-1_8b.git models/ai-labs/sales-chat-1_8b')
        os.system(f'cd models/ai-labs/sales-chat-1_8b && git lfs install && git lfs pull')
        pass


def download_internlm_xcomposer2_model():
    with st.spinner("正在下载书生浦语灵笔模型，请稍等..."):
        os.system(f'rm -rf models/internlm/internlm-xcomposer2-vl-7b-4bit')
        os.system(f'GIT_LFS_SKIP_SMUDGE=1 git clone https://code.openxlab.org.cn/InternLM-xcomposer/internlm-xcomposer2-vl-7b-4bit.git models/internlm/internlm-xcomposer2-vl-7b-4bit')
        os.system(f'cd models/internlm/internlm-xcomposer2-vl-7b-4bit && git lfs install && git lfs pull')
        pass


def download_internlm2_chat_7b_model():
    with st.spinner("正在下载internlm2-chat-7b，请稍等..."):
        os.system(f'rm -rf models/internlm/internlm2-chat-7b')
        os.system(f'GIT_LFS_SKIP_SMUDGE=1 git clone https://code.openxlab.org.cn/OpenLMLab/internlm2-chat-7b.git models/internlm/internlm2-chat-7b')
        os.system(f'cd models/internlm/internlm2-chat-7b && git lfs install && git lfs pull')
        pass


def download_chatglm3_6b_model():
    with st.spinner("正在下载chatglm3-6b，请稍等..."):
        os.system(f'rm -rf models/THUDM/chatglm3-6b')
        os.system(f'GIT_LFS_SKIP_SMUDGE=1 git clone https://code.openxlab.org.cn/THUDM/chatglm3-6b.git models/THUDM/chatglm3-6b')
        os.system(f'cd models/THUDM/chatglm3-6b && git lfs install && git lfs pull')
        pass


def download_stable_diffusion_base_model():
    with st.spinner("正在下载stable-diffusion-2-1-base，请稍等..."):
        os.system(f'rm -rf models/helenai/stabilityai-stable-diffusion-2-1-base-ov')
        # os.system(f'GIT_LFS_SKIP_SMUDGE=1 HF_ENDPOINT=https://hf-mirror.com git clone https://hf-mirror.com/helenai/stabilityai-stable-diffusion-2-1-base-ov.git models/helenai/stabilityai-stable-diffusion-2-1-base-ov')
        os.system(f'huggingface-cli download --resume-download --local-dir-use-symlinks False helenai/stabilityai-stable-diffusion-2-1-ov --local-dir models/helenai/stabilityai-stable-diffusion-2-1-ov')
        os.system(f'cd models/helenai/stabilityai-stable-diffusion-2-1-base-ov && git lfs install && git lfs pull')
        pass
