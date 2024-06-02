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


global_system_prompt = "你的名字叫伍鲜，是AI-Labs团队的营销人员，也是一名经验丰富的服装营销人员，精通服装设计、服饰搭配、服装销售、服装信息咨询、售后服务等各类问题。你说话优雅、有艺术感、必要时可以引用典故，你总是称呼客户为朋友。"

check_use_limit = False
cuda_size_24gb = 22000 # 24566
cuda_size_40gb = 40000 # 40950

def init_page_header(title, icon):
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
        menu_items={
            "Get Help": "https://github.com/wux-labs/OpenXLab-IntelligentSalesAssistant",
            "Report a bug": "https://github.com/wux-labs/OpenXLab-IntelligentSalesAssistant/issues",
            "About": """
## 🏡智能营销助手

众所周知，获客、活客、留客是电商行业的三大难题，谁拥有跟客户最佳的沟通方式，谁就拥有客户。

随着用户消费逐渐转移至线上，电商行业面临以下一些问题：

* 用户交流体验差
* 商品推荐不精准
* 客户转化率低
* 退换货频率高
* 物流成本高

在这样的背景下，未来销售的引擎——大模型加持的智能营销助手就诞生了。

它能够与用户的对话，了解用户的需求，基于多模态的AIGC生成能力，持续输出更符合用户消费习惯的文本、图片和视频等营销内容，推荐符合用户的商品，将营销与经营结合。

""",
        },
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
    if model_id in ["ai-labs/sales-chat-7b", "internlm/internlm2-chat-7b", "internlm/internlm2-chat-20b"]:
        avatar = "statics/avatars/intern.png"
    elif model_id == "THUDM/chatglm3-6b":
        avatar = "statics/avatars/chatglm.png"
    elif model_id in ["ai-labs/stable-diffusion", "stabilityai/stable-diffusion-2-1"]:
        avatar = "statics/avatars/stablediffusion.png"
    elif model_id == "myshell/melotts":
        avatar = "statics/avatars/melotts.png"
    else:
        avatar = "statics/avatars/sales.png"
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


def is_cuda_enough(needs):
    if torch.cuda.device_count() > 1:
        return True
    else:
        properties = torch.cuda.get_device_properties(0)
        total_memory = int(f'{properties.total_memory / (1 << 20):.0f}')
        return total_memory >= needs


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clear_streamlit_cache(keeps):
    all_caches = ["chat_tokenizer", "chat_model", 
                  "stable_diffusion_model",
                  "xcomposer2_vl_tokenizer", "xcomposer2_vl_model",
                  "whisper_model_base", "whisper_model_small", "whisper_model_medium", "whisper_model_large",
                  "ask_product_history", "ask_product_llm",
                  "sales_agent_model"
                  ]

    for cache in all_caches:
        if cache not in keeps and cache in st.session_state.keys():
            del st.session_state[cache]

    clear_cuda_cache()


def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()

    # 将图像数据编码为Base64字符串
    base64_data = base64.b64encode(image_data).decode('utf-8')
    return base64_data
