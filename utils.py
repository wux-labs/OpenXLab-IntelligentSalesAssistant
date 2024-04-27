import streamlit as st
import os
import pandas as pd
import json
import base64
import platform
import numpy as np

from datetime import datetime
from typing import List, Optional, Union
from streamlit_extras.app_logo import add_logo

from database.database import engine
from sqlalchemy import text

import torch
from transformers import AutoTokenizer

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings


def init_page_header(title, icon):
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
    )
    # add_logo("statics/docs/logo.png", height=60)
    st.sidebar.header(f"{icon}{title}")
    st.subheader(f"{icon}{title}", divider='rainbow')
    

def init_session_state():
    if "userid" not in st.session_state.keys():
        st.session_state.userid = 1
    if "username" not in st.session_state.keys():
        st.session_state.username = "guest"
    if "fullname" not in st.session_state.keys():
        st.session_state.fullname = "游客"
    if "rolename" not in st.session_state.keys():
        st.session_state.rolename = "买家"
    if "aigc_temp_freq" not in st.session_state.keys():
        st.session_state.aigc_temp_freq = 3
    if "aigc_perm_freq" not in st.session_state.keys():
        st.session_state.aigc_perm_freq = 0
    if "showlimit" not in st.session_state.keys():
        st.session_state.showlimit = 10


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def get_avatar(model_id):
    if model_id == "ai-labs/sales-chat-1_8b":
        avatar = "statics/avatars/intern.png"
    elif model_id == "internlm/internlm2-chat-7b":
        avatar = "statics/avatars/intern.png"
    elif model_id == "THUDM/chatglm3-6b":
        avatar = "statics/avatars/chatglm.png"
    elif model_id == "ai-labs/stable-diffusion":
        avatar = "statics/avatars/stablediffusion.png"
    elif model_id == "myshell/melotts":
        avatar = "statics/avatars/melotts.png"
    else:
        avatar = None
    return avatar


def select_aigc_left_freq():
    count = None
    try:
        with engine.connect() as conn:
            sql = text(f'''
            select aigc_temp_freq, aigc_perm_freq from ai_labs_user where username = :username;
            ''')
            count = conn.execute(sql, [{'username': st.session_state.username}]).fetchone()
    except Exception as e:
        st.exception(e)
    return count


def update_aigc_perm_freq(count):
    try:
        with engine.connect() as conn:
            sql = text(f'''
            update ai_labs_user set aigc_perm_freq = aigc_perm_freq + {count} where username = :username and aigc_perm_freq > 0;
            ''')
            conn.execute(sql, [{'username': st.session_state.username}])
            conn.commit()
    except Exception as e:
        st.exception(e)


def update_aigc_temp_freq(count):
    try:
        with engine.connect() as conn:
            sql = text(f'''
            update ai_labs_user set aigc_temp_freq = aigc_temp_freq + {count} where username = :username;
            ''')
            conn.execute(sql, [{'username': st.session_state.username}])
            conn.commit()
    except Exception as e:
        st.exception(e)


def use_limited():
    st.warning("您今日已达到使用次数限制！玩游戏放松一下，还可以获得免费使用次数哦~", icon="😭")
    st.page_link("pages/71🎮休闲游戏.py", icon="🎮")


def is_cuda_available():
    # return False
    return torch.cuda.is_available()


@st.cache_resource
def load_model_by_id(model_id_or_path, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained("models/" + model_id_or_path,
                                              trust_remote_code=True)
    if is_cuda_available():
        from transformers import AutoModel
        model = AutoModel.from_pretrained("models/" + model_id_or_path,
                                                    trust_remote_code=True).half().eval().cuda()
    else:
        from bigdl.llm.transformers import AutoModel
        model = AutoModel.from_pretrained("models/" + model_id_or_path,
                                                    load_in_4bit=True,
                                                    trust_remote_code=True).eval()
    return tokenizer, model


@st.cache_resource
def load_huggingface_embedding():
    embedding = HuggingFaceEmbeddings(model_name="models/GanymedeNil/text2vec-large-chinese")
    return embedding


def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()

    # 将图像数据编码为Base64字符串
    base64_data = base64.b64encode(image_data).decode('utf-8')
    return base64_data
