import os
import streamlit as st
from utils import is_cuda_enough
from utils import cuda_size_24gb, cuda_size_40gb


def download_sales_chat_model():
    with st.spinner("正在下载智能营销助手模型，请稍等..."):
        if is_cuda_enough(cuda_size_40gb):
            os.system(f'rm -rf models/internlm/internlm2-chat-20b')
            os.system(f'HF_ENDPOINT=https://hf-mirror.com huggingface-cli download --resume-download --local-dir-use-symlinks False internlm/internlm2-chat-20b --local-dir models/internlm/internlm2-chat-20b')
        elif is_cuda_enough(cuda_size_24gb):
            os.system(f'rm -rf models/internlm/internlm2-chat-7b')
            os.system(f'HF_ENDPOINT=https://hf-mirror.com huggingface-cli download --resume-download --local-dir-use-symlinks False internlm/internlm2-chat-7b --local-dir models/internlm/internlm2-chat-7b')
        else:
            os.system(f'rm -rf models/ai-labs/sales-chat-7b')
            os.system(f'GIT_LFS_SKIP_SMUDGE=1 git clone https://code.openxlab.org.cn/AI-Labs/sales-chat-7b.git models/ai-labs/sales-chat-7b')
            os.system(f'cd models/ai-labs/sales-chat-7b && git lfs install && git lfs pull')
        pass


def download_internlm_xcomposer2_model():
    with st.spinner("正在下载书生浦语灵笔模型，请稍等..."):
        if is_cuda_enough(cuda_size_40gb):
            os.system(f'rm -rf models/internlm/internlm-xcomposer2-vl-7b')
            os.system(f'HF_ENDPOINT=https://hf-mirror.com huggingface-cli download --resume-download --local-dir-use-symlinks False internlm/internlm-xcomposer2-vl-7b --local-dir models/internlm/internlm-xcomposer2-vl-7b')
        else:
            os.system(f'rm -rf models/internlm/internlm-xcomposer2-vl-7b-4bit')
            os.system(f'HF_ENDPOINT=https://hf-mirror.com huggingface-cli download --resume-download --local-dir-use-symlinks False internlm/internlm-xcomposer2-vl-7b-4bit --local-dir models/internlm/internlm-xcomposer2-vl-7b-4bit')
        pass


def download_stable_diffusion_model():
    with st.spinner("正在下载图片生成模型，请稍等..."):
        os.system(f'rm -rf models/stabilityai/stable-diffusion-2-1')
        os.system(f'HF_ENDPOINT=https://hf-mirror.com huggingface-cli download --resume-download --local-dir-use-symlinks False stabilityai/stable-diffusion-2-1 --local-dir models/stabilityai/stable-diffusion-2-1')
        pass


def download_other_model():
    with st.spinner("正在下载其他必要模型，请稍等..."):
        os.system(f'rm -rf models/GanymedeNil/text2vec-large-chinese')
        os.system(f'HF_ENDPOINT=https://hf-mirror.com huggingface-cli download --resume-download --local-dir-use-symlinks False GanymedeNil/text2vec-large-chinese --local-dir models/GanymedeNil/text2vec-large-chinese')
        pass


def download_anydoor_model():
    with st.spinner("正在下载衣服试穿模型，请稍等..."):
        if is_cuda_enough(cuda_size_40gb):
            os.system(f'rm -rf models/iic/AnyDoor')
            from modelscope.hub.snapshot_download import snapshot_download
            snapshot_download("iic/AnyDoor", cache_dir="models/")
        pass


if __name__ == '__main__':
    download_sales_chat_model()
    download_internlm_xcomposer2_model()
    download_stable_diffusion_model()
    download_other_model()
    download_anydoor_model()